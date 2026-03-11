"""Recompute full consensus gradient table with 1024-token sequences and forward hooks.
This is the canonical methodology — proper positional embeddings."""
import torch
import numpy as np
from transformers import GPT2LMHeadModel
import time

t0 = time.time()
model = GPT2LMHeadModel.from_pretrained("gpt2").eval().cuda()
tokens = np.load("/data/tokens/wikitext103_train_tokens.npy")[:500000]

CONSENSUS = [2, 2361, 2460, 2928, 1831, 1245, 2600]
EXCEPTION = 2123
SEQ_LEN = 1024
THRESH = 0.1

post_gelu = {}
def hook_fn(module, input, output):
    post_gelu['act'] = output.detach()

norm_data = {}
def norm_hook_fn(module, input, output):
    norm_data['out'] = output.detach()

hook = model.transformer.h[11].mlp.act.register_forward_hook(hook_fn)
norm_hook = model.transformer.h[11].mlp.register_forward_hook(norm_hook_fn)

all_cons_count = []
all_exc_firing = []
all_norms = []

with torch.no_grad():
    for start in range(0, len(tokens) - SEQ_LEN + 1, SEQ_LEN):
        chunk = torch.tensor(tokens[start:start+SEQ_LEN], dtype=torch.long).unsqueeze(0).cuda()
        model(chunk)
        act = post_gelu['act'].squeeze(0)  # (1024, 3072)
        nrm = norm_data['out'].squeeze(0).norm(dim=-1)  # (1024,)
        
        cons = (act[:, CONSENSUS] > THRESH).cpu()  # (1024, 7)
        exc = (act[:, EXCEPTION] > THRESH).cpu()    # (1024,)
        
        all_cons_count.append(cons.sum(dim=1))  # (1024,)
        all_exc_firing.append(exc)
        all_norms.append(nrm.cpu())

hook.remove()
norm_hook.remove()

cons_count = torch.cat(all_cons_count).numpy()
exc_firing = torch.cat(all_exc_firing).numpy()
mlp_norms = torch.cat(all_norms).numpy()
N = len(cons_count)
print(f"Total tokens: {N} ({time.time()-t0:.0f}s)\n")

# Full table
print(f"{'Cons':>4} | {'Count':>8} | {'%':>6} | {'N2123 FR':>9} | {'Avg Norm':>9}")
print("-" * 50)
for c in range(8):
    mask = cons_count == c
    n = mask.sum()
    pct = n / N * 100
    fr = exc_firing[mask].mean() * 100 if n > 0 else 0
    norm = mlp_norms[mask].mean() if n > 0 else 0
    print(f"{c}/7  | {n:>8,} | {pct:>5.1f}% | {fr:>8.1f}% | {norm:>9.1f}")

total = sum(1 for c in range(8) if (cons_count == c).sum() > 0)
print(f"\nGradient range: {exc_firing[cons_count==0].mean()*100 - exc_firing[cons_count==7].mean()*100:.1f}pp")
print(f"Norm ratio: {mlp_norms[cons_count==0].mean() / mlp_norms[cons_count==7].mean():.2f}x")

# Also compute exclusivity
print(f"\nExclusivity (pairwise, exc vs each consensus neuron):")
cons_firing = np.column_stack([(cons_count > 0)] * 7)  # dummy, need actual
# Recompute from raw data
all_cons_firing = []
all_exc_raw = []
hook2 = model.transformer.h[11].mlp.act.register_forward_hook(hook_fn)
with torch.no_grad():
    for start in range(0, len(tokens) - SEQ_LEN + 1, SEQ_LEN):
        chunk = torch.tensor(tokens[start:start+SEQ_LEN], dtype=torch.long).unsqueeze(0).cuda()
        model(chunk)
        act = post_gelu['act'].squeeze(0)
        all_cons_firing.append((act[:, CONSENSUS] > THRESH).cpu())
        all_exc_raw.append((act[:, EXCEPTION] > THRESH).cpu())
hook2.remove()

cf = torch.cat(all_cons_firing).numpy()  # (N, 7)
ef = torch.cat(all_exc_raw).numpy()      # (N,)

excl_vals = []
for i, cn in enumerate(CONSENSUS):
    both = (ef & cf[:, i]).sum()
    either = (ef | cf[:, i]).sum()
    excl = (1 - both/either) * 100 if either > 0 else 100
    excl_vals.append(excl)
    print(f"  N{cn}: {excl:.1f}%")
print(f"  Average: {np.mean(excl_vals):.1f}%")

# Overall fire rates
print(f"\nOverall fire rates:")
print(f"  N2123: {ef.mean()*100:.1f}%")
for i, cn in enumerate(CONSENSUS):
    print(f"  N{cn}: {cf[:,i].mean()*100:.1f}%")

# Co-fire stats for N2123 and N2
n2123_fires = ef.sum()
n2_fires = cf[:,0].sum()  # N2 is first in CONSENSUS
cofire = (ef & cf[:,0]).sum()
expected = ef.mean() * cf[:,0].mean() * N
print(f"\nN2123-N2 co-fire: {cofire:,} (expected under independence: {expected:,.0f}, reduction: {(1-cofire/expected)*100:.1f}%)")

print(f"\nTotal time: {time.time()-t0:.0f}s")
