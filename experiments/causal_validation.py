"""Causal validation + mechanism table with 1024-token sequences."""
import torch, numpy as np, time
from transformers import GPT2LMHeadModel

t0 = time.time()
model = GPT2LMHeadModel.from_pretrained("gpt2").eval().cuda()
tokens = np.load("/data/tokens/wikitext103_train_tokens.npy")[:500000]
SEQ_LEN = 1024
CONSENSUS = [2, 2361, 2460, 2928, 1831, 1245, 2600]
EXCEPTION = 2123
THRESH = 0.1

post_gelu = {}
def hook_fn(m, i, o): post_gelu['act'] = o.detach()

mlp_out = {}
def mlp_hook_fn(m, i, o): mlp_out['val'] = o.detach()

hook = model.transformer.h[11].mlp.act.register_forward_hook(hook_fn)
mlp_hook = model.transformer.h[11].mlp.register_forward_hook(mlp_hook_fn)

# First pass: get consensus levels for all tokens
all_cons_count = []
with torch.no_grad():
    for start in range(0, len(tokens) - SEQ_LEN + 1, SEQ_LEN):
        chunk = torch.tensor(tokens[start:start+SEQ_LEN], dtype=torch.long).unsqueeze(0).cuda()
        model(chunk)
        act = post_gelu['act'].squeeze(0)
        cc = (act[:, CONSENSUS] > THRESH).sum(dim=1).cpu()
        all_cons_count.append(cc)

cons_count = torch.cat(all_cons_count).numpy()
hook.remove()
mlp_hook.remove()
N = len(cons_count)
print(f"Pass 1 done: {N} tokens ({time.time()-t0:.0f}s)", flush=True)

# Second pass: for each consensus level, compute PPL with and without MLP
# We need to: (1) run full model, get loss per token, (2) run with MLP zeroed, get loss per token

def get_per_token_loss(model, tokens_arr, zero_mlp_layer=None):
    """Get per-token cross-entropy loss. Optionally zero a specific MLP layer's output."""
    hook_handle = None
    if zero_mlp_layer is not None:
        def zero_hook(m, i, o):
            return torch.zeros_like(o)
        hook_handle = model.transformer.h[zero_mlp_layer].mlp.register_forward_hook(zero_hook)
    
    all_losses = []
    with torch.no_grad():
        for start in range(0, len(tokens_arr) - SEQ_LEN + 1, SEQ_LEN):
            chunk = torch.tensor(tokens_arr[start:start+SEQ_LEN], dtype=torch.long).unsqueeze(0).cuda()
            outputs = model(chunk, labels=chunk)
            # Get per-token loss
            logits = outputs.logits[:, :-1, :]  # (1, seq-1, vocab)
            targets = chunk[:, 1:]  # (1, seq-1)
            loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
            losses = loss_fn(logits.squeeze(0), targets.squeeze(0))  # (seq-1,)
            all_losses.append(losses.cpu())
    
    if hook_handle:
        hook_handle.remove()
    return torch.cat(all_losses).numpy()

print("Computing full-model losses...", flush=True)
losses_full = get_per_token_loss(model, tokens)
print(f"Full losses: {len(losses_full)} ({time.time()-t0:.0f}s)", flush=True)

print("Computing no-MLP losses...", flush=True)
losses_nomlp = get_per_token_loss(model, tokens, zero_mlp_layer=11)
print(f"No-MLP losses: {len(losses_nomlp)} ({time.time()-t0:.0f}s)", flush=True)

# Align consensus counts with losses (losses are for predicting token t+1, grouped by consensus at t)
# losses[i] predicts token i+1, consensus[i] is for position i within each 1024-chunk
# But consensus is position 0..1023 within each chunk, losses are for positions 0..1022 (predicting 1..1023)
# So we drop the last position of each chunk from consensus
cons_aligned = []
for start in range(0, len(tokens) - SEQ_LEN + 1, SEQ_LEN):
    chunk_idx = start // SEQ_LEN
    cons_aligned.append(cons_count[chunk_idx*SEQ_LEN : chunk_idx*SEQ_LEN + SEQ_LEN - 1])
cons_aligned = np.concatenate(cons_aligned)

assert len(cons_aligned) == len(losses_full), f"Mismatch: {len(cons_aligned)} vs {len(losses_full)}"

# Causal validation table
print(f"\n{'Cons':>4} | {'N':>8} | {'Base PPL':>9} | {'No-MLP PPL':>11} | {'Delta':>7}")
print("-" * 55)
for c in range(8):
    mask = cons_aligned == c
    n = mask.sum()
    if n > 0:
        base_ppl = np.exp(losses_full[mask].mean())
        nomlp_ppl = np.exp(losses_nomlp[mask].mean())
        delta = (nomlp_ppl / base_ppl - 1) * 100
        print(f"{c}/7  | {n:>8,} | {base_ppl:>9.1f} | {nomlp_ppl:>11.1f} | {delta:>+6.1f}%")

# Overall
base_all = np.exp(losses_full.mean())
nomlp_all = np.exp(losses_nomlp.mean())
print(f"All  | {len(losses_full):>8,} | {base_all:>9.1f} | {nomlp_all:>11.1f} | {(nomlp_all/base_all-1)*100:>+6.1f}%")

# Mechanism table: KL, boost, rank change
print("\n\nMechanism table...", flush=True)

def get_logits(model, tokens_arr, zero_mlp_layer=None):
    hook_handle = None
    if zero_mlp_layer is not None:
        def zero_hook(m, i, o):
            return torch.zeros_like(o)
        hook_handle = model.transformer.h[zero_mlp_layer].mlp.register_forward_hook(zero_hook)
    
    all_logits = []
    with torch.no_grad():
        for start in range(0, len(tokens_arr) - SEQ_LEN + 1, SEQ_LEN):
            chunk = torch.tensor(tokens_arr[start:start+SEQ_LEN], dtype=torch.long).unsqueeze(0).cuda()
            outputs = model(chunk)
            all_logits.append(outputs.logits[:, :-1, :].squeeze(0).cpu())
    
    if hook_handle:
        hook_handle.remove()
    return torch.cat(all_logits, dim=0)  # (N-chunks, vocab)

print("Getting full logits...", flush=True)
logits_full = get_logits(model, tokens)
print(f"Full logits: {logits_full.shape} ({time.time()-t0:.0f}s)", flush=True)

print("Getting no-MLP logits...", flush=True)
logits_nomlp = get_logits(model, tokens, zero_mlp_layer=11)
print(f"No-MLP logits: {logits_nomlp.shape} ({time.time()-t0:.0f}s)", flush=True)

# Get target tokens for each position
targets_list = []
for start in range(0, len(tokens) - SEQ_LEN + 1, SEQ_LEN):
    targets_list.append(torch.tensor(tokens[start+1:start+SEQ_LEN], dtype=torch.long))
targets = torch.cat(targets_list)

print(f"\n{'Cons':>4} | {'N':>7} | {'KL div':>7} | {'Boost':>10} | {'DeltaRank':>10}")
print("-" * 55)
for c in range(8):
    mask = cons_aligned == c
    idx = np.where(mask)[0]
    if len(idx) < 10:
        continue
    
    # Subsample if too many (for memory)
    if len(idx) > 50000:
        rng = np.random.default_rng(42)
        idx = rng.choice(idx, 50000, replace=False)
    
    lf = logits_full[idx]   # (n, vocab)
    ln = logits_nomlp[idx]  # (n, vocab)
    tgt = targets[idx]      # (n,)
    
    # KL divergence (full || no-mlp)
    p = torch.softmax(lf, dim=-1)
    q = torch.softmax(ln, dim=-1)
    kl = (p * (p.log() - q.log())).sum(dim=-1).mean().item()
    
    # Boost: P(correct|full) / P(correct|no-mlp)
    p_correct_full = p[range(len(tgt)), tgt].mean().item()
    p_correct_nomlp = q[range(len(tgt)), tgt].mean().item()
    boost = p_correct_full / p_correct_nomlp if p_correct_nomlp > 0 else float('inf')
    
    # Rank change
    ranks_full = (lf > lf[range(len(tgt)), tgt].unsqueeze(1)).sum(dim=-1).float()
    ranks_nomlp = (ln > ln[range(len(tgt)), tgt].unsqueeze(1)).sum(dim=-1).float()
    delta_rank = (ranks_full - ranks_nomlp).mean().item()
    
    print(f"{c}/7  | {mask.sum():>7,} | {kl:>7.3f} | {boost:>8.2f}x | {delta_rank:>+9.1f}")

print(f"\nTotal time: {time.time()-t0:.0f}s")
