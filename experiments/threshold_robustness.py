"""Threshold robustness with NEW consensus neurons, 500K tokens, forward hooks."""
import torch
import numpy as np
from transformers import GPT2LMHeadModel
import time

t0 = time.time()
model = GPT2LMHeadModel.from_pretrained("gpt2").eval().cuda()
tokens = np.load("/data/tokens/wikitext103_train_tokens.npy")[:500000]

CONSENSUS = [2, 2361, 2460, 2928, 1831, 1245, 2600]
EXCEPTION = 2123
THRESHOLDS = [0.01, 0.05, 0.10, 0.50, 1.00]
SEQ_LEN = 1024

post_gelu = {}
def hook_fn(module, input, output):
    post_gelu['act'] = output.detach()

hook = model.transformer.h[11].mlp.act.register_forward_hook(hook_fn)

# Process in 1024-token sequences
all_acts = []
with torch.no_grad():
    for start in range(0, len(tokens) - SEQ_LEN + 1, SEQ_LEN):
        chunk = torch.tensor(tokens[start:start+SEQ_LEN], dtype=torch.long).unsqueeze(0).cuda()
        model(chunk)
        all_acts.append(post_gelu['act'].squeeze(0).cpu())
        if len(all_acts) % 50 == 0:
            print(f"  {len(all_acts)} chunks done ({time.time()-t0:.0f}s)", flush=True)

hook.remove()
activations = torch.cat(all_acts, dim=0).numpy()  # (~488K, 3072)
print(f"Activations: {activations.shape}, took {time.time()-t0:.0f}s", flush=True)

for thresh in THRESHOLDS:
    cons_firing = activations[:, CONSENSUS] > thresh
    exc_firing = activations[:, EXCEPTION] > thresh
    cons_count = cons_firing.sum(axis=1)

    exc_rates = []
    for c in range(8):
        mask = cons_count == c
        n = mask.sum()
        if n > 0:
            exc_rates.append(exc_firing[mask].mean() * 100)
        else:
            exc_rates.append(None)

    valid_rates = [r for r in exc_rates if r is not None]
    monotonic = all(a >= b for a, b in zip(valid_rates, valid_rates[1:]))
    rate_range = valid_rates[0] - valid_rates[-1] if len(valid_rates) >= 2 else 0

    exc_overall = exc_firing.mean() * 100
    # Avg per-consensus-neuron fire rate
    cons_avg_fr = cons_firing.mean() * 100  # overall across all 7

    # Pairwise exclusivity
    excl_vals = []
    for i in range(7):
        both = (exc_firing & cons_firing[:, i]).sum()
        either = (exc_firing | cons_firing[:, i]).sum()
        if either > 0:
            excl_vals.append((1 - both/either) * 100)
    avg_excl = np.mean(excl_vals)

    mono_str = "✓" if monotonic else "✗"
    print(f"\nThreshold {thresh}:", flush=True)
    print(f"  N2123 FR: {exc_overall:.1f}%", flush=True)
    print(f"  Def-ON avg FR: {cons_avg_fr:.1f}%", flush=True)
    print(f"  Exclusivity: {avg_excl:.1f}%", flush=True)
    print(f"  Monotonic: {mono_str}", flush=True)
    print(f"  Range: {rate_range:.1f}pp", flush=True)
    print(f"  Rates: {[f'{r:.1f}' if r is not None else '-' for r in exc_rates]}", flush=True)

print(f"\nTotal: {time.time()-t0:.0f}s", flush=True)
