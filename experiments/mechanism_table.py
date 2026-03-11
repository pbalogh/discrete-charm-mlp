"""Mechanism table (KL, boost, rank) with 1024-token sequences. Chunked to avoid OOM."""
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

# Accumulate per-consensus-level stats
from collections import defaultdict
stats = defaultdict(lambda: {'kl_sum': 0, 'boost_num': 0, 'boost_den': 0, 
                              'rank_full_sum': 0, 'rank_nomlp_sum': 0, 'n': 0})

n_chunks = 0
for start in range(0, len(tokens) - SEQ_LEN + 1, SEQ_LEN):
    chunk = torch.tensor(tokens[start:start+SEQ_LEN], dtype=torch.long).unsqueeze(0).cuda()
    
    # Get consensus levels
    hook = model.transformer.h[11].mlp.act.register_forward_hook(hook_fn)
    with torch.no_grad():
        model(chunk)
    act = post_gelu['act'].squeeze(0)
    cons = (act[:, CONSENSUS] > THRESH).sum(dim=1).cpu().numpy()  # (1024,)
    hook.remove()
    
    # Full model logits
    with torch.no_grad():
        out_full = model(chunk)
    logits_full = out_full.logits.squeeze(0)[:-1]  # (1023, vocab)
    
    # No-MLP logits
    def zero_hook(m, i, o): return torch.zeros_like(o)
    zh = model.transformer.h[11].mlp.register_forward_hook(zero_hook)
    with torch.no_grad():
        out_nomlp = model(chunk)
    logits_nomlp = out_nomlp.logits.squeeze(0)[:-1]  # (1023, vocab)
    zh.remove()
    
    # Targets (predicting next token)
    targets = chunk.squeeze(0)[1:]  # (1023,)
    cons_aligned = cons[:-1]  # consensus at position t for predicting t+1
    
    # Compute per-level stats
    p_full = torch.softmax(logits_full, dim=-1)
    p_nomlp = torch.softmax(logits_nomlp, dim=-1)
    
    for c in range(8):
        mask = cons_aligned == c
        if mask.sum() == 0:
            continue
        idx = np.where(mask)[0]
        
        pf = p_full[idx]
        pn = p_nomlp[idx]
        tgt = targets[idx]
        
        # KL(full || nomlp)
        kl = (pf * (pf.log() - pn.log())).sum(dim=-1)
        stats[c]['kl_sum'] += kl.sum().item()
        
        # Boost
        p_correct_full = pf[range(len(tgt)), tgt]
        p_correct_nomlp = pn[range(len(tgt)), tgt]
        stats[c]['boost_num'] += p_correct_full.sum().item()
        stats[c]['boost_den'] += p_correct_nomlp.sum().item()
        
        # Rank
        lf = logits_full[idx]
        ln = logits_nomlp[idx]
        ranks_full = (lf > lf[range(len(tgt)), tgt].unsqueeze(1)).sum(dim=-1).float()
        ranks_nomlp = (ln > ln[range(len(tgt)), tgt].unsqueeze(1)).sum(dim=-1).float()
        stats[c]['rank_full_sum'] += ranks_full.sum().item()
        stats[c]['rank_nomlp_sum'] += ranks_nomlp.sum().item()
        
        stats[c]['n'] += len(idx)
    
    n_chunks += 1
    if n_chunks % 50 == 0:
        print(f"  {n_chunks} chunks ({time.time()-t0:.0f}s)", flush=True)

print(f"\n{'Cons':>4} | {'N':>8} | {'KL':>7} | {'Boost':>10} | {'DeltaRank':>10}")
print("-" * 55)
for c in range(8):
    s = stats[c]
    if s['n'] == 0:
        continue
    kl = s['kl_sum'] / s['n']
    boost = s['boost_num'] / s['boost_den'] if s['boost_den'] > 0 else float('inf')
    delta_rank = (s['rank_full_sum'] - s['rank_nomlp_sum']) / s['n']
    print(f"{c}/7  | {s['n']:>8,} | {kl:>7.3f} | {boost:>8.2f}x | {delta_rank:>+9.1f}")

print(f"\nTotal time: {time.time()-t0:.0f}s")
