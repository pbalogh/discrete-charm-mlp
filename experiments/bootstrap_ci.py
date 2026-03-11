"""Bootstrap CIs for key paper claims, 500K tokens, forward hooks."""
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

post_gelu = {}
def hook_fn(module, input, output):
    post_gelu['act'] = output.detach()

norm_data = {}
def norm_hook_fn(module, input, output):
    norm_data['out'] = output.detach()

hook = model.transformer.h[11].mlp.act.register_forward_hook(hook_fn)
norm_hook = model.transformer.h[11].mlp.register_forward_hook(norm_hook_fn)

all_acts = []
all_norms = []
with torch.no_grad():
    for start in range(0, len(tokens) - SEQ_LEN + 1, SEQ_LEN):
        chunk = torch.tensor(tokens[start:start+SEQ_LEN], dtype=torch.long).unsqueeze(0).cuda()
        model(chunk)
        all_acts.append(post_gelu['act'].squeeze(0).cpu())
        all_norms.append(norm_data['out'].squeeze(0).norm(dim=-1).cpu())
        if len(all_acts) % 100 == 0:
            print(f"  {len(all_acts)} chunks ({time.time()-t0:.0f}s)", flush=True)

hook.remove()
norm_hook.remove()

activations = torch.cat(all_acts, dim=0).numpy()
mlp_norms = torch.cat(all_norms, dim=0).numpy()
print(f"Data: {activations.shape}, {time.time()-t0:.0f}s", flush=True)

# Precompute
THRESH = 0.1
cons_firing = activations[:, CONSENSUS] > THRESH
exc_firing = activations[:, EXCEPTION] > THRESH
cons_count = cons_firing.sum(axis=1)
N = len(activations)

# Free large arrays
del activations, all_acts, all_norms
import gc; gc.collect()

def compute_metrics(idx):
    cc = cons_count[idx]
    ef = exc_firing[idx]
    cf = cons_firing[idx]
    norms = mlp_norms[idx]
    
    mask0 = cc == 0
    mask7 = cc == 7
    n0, n7 = mask0.sum(), mask7.sum()
    if n0 == 0 or n7 == 0:
        return np.nan, np.nan, np.nan, False
    
    rate0 = ef[mask0].mean() * 100
    rate7 = ef[mask7].mean() * 100
    gradient_range = rate0 - rate7
    
    norm0 = norms[mask0].mean()
    norm7 = norms[mask7].mean()
    norm_ratio = norm0 / norm7 if norm7 > 0 else np.nan
    
    excl_vals = []
    for i in range(7):
        both = (ef & cf[:, i]).sum()
        either = (ef | cf[:, i]).sum()
        if either > 0:
            excl_vals.append((1 - both/either) * 100)
    avg_excl = np.mean(excl_vals)
    
    rates = []
    for c in range(8):
        mask = cc == c
        if mask.sum() > 0:
            rates.append(ef[mask].mean())
        else:
            rates.append(np.nan)
    valid = [r for r in rates if not np.isnan(r)]
    monotonic = all(a >= b for a, b in zip(valid, valid[1:]))
    
    return gradient_range, norm_ratio, avg_excl, monotonic

# Point estimates
g, n, e, m = compute_metrics(np.arange(N))
print(f"\nPoint estimates:", flush=True)
print(f"  Gradient range: {g:.1f}pp", flush=True)
print(f"  Norm ratio: {n:.2f}x", flush=True)
print(f"  Avg exclusivity: {e:.1f}%", flush=True)
print(f"  Monotonic: {m}", flush=True)

# Bootstrap
N_BOOT = 10000
rng = np.random.default_rng(42)
boot_grad = np.zeros(N_BOOT)
boot_norm = np.zeros(N_BOOT)
boot_excl = np.zeros(N_BOOT)
boot_mono = np.zeros(N_BOOT)

print(f"\nRunning {N_BOOT} bootstrap samples...", flush=True)
for b in range(N_BOOT):
    if b % 1000 == 0:
        print(f"  {b}/{N_BOOT} ({time.time()-t0:.0f}s)", flush=True)
    idx = rng.choice(N, size=N, replace=True)
    g2, n2, e2, m2 = compute_metrics(idx)
    boot_grad[b] = g2
    boot_norm[b] = n2
    boot_excl[b] = e2
    boot_mono[b] = float(m2)

print(f"\n95% CIs (percentile):", flush=True)
for name, arr, point in [("Gradient range (pp)", boot_grad, g),
                          ("Norm ratio (x)", boot_norm, n),
                          ("Avg exclusivity (%)", boot_excl, e)]:
    lo, hi = np.nanpercentile(arr, [2.5, 97.5])
    print(f"  {name}: {point:.2f} [{lo:.2f}, {hi:.2f}]", flush=True)

mono_pct = boot_mono.mean() * 100
print(f"  Monotonicity: {mono_pct:.1f}% of bootstrap samples", flush=True)
print(f"\nTotal: {time.time()-t0:.0f}s", flush=True)
