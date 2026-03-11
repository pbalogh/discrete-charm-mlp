"""Test: does consensus structure exist in UNTRAINED GPT-2 Small?
If yes, it's architectural. If no, it's learned."""
import torch, numpy as np, time
from transformers import GPT2LMHeadModel, GPT2Config

t0 = time.time()

# Random-weight model (same architecture, random init)
config = GPT2Config()
model_rand = GPT2LMHeadModel(config).eval().cuda()
print(f"Random-weight GPT-2 Small loaded ({time.time()-t0:.0f}s)")

# Also load trained for comparison
model_trained = GPT2LMHeadModel.from_pretrained("gpt2").eval().cuda()
print(f"Trained GPT-2 Small loaded ({time.time()-t0:.0f}s)")

tokens = np.load("/data/tokens/wikitext103_train_tokens.npy")[:100000]  # 100K enough
SEQ_LEN = 1024
CONSENSUS = [2, 2361, 2460, 2928, 1831, 1245, 2600]
EXCEPTION = 2123
THRESH = 0.1

def analyze_model(model, name):
    post_gelu = {}
    def hook_fn(m, i, o): post_gelu['act'] = o.detach()
    hook = model.transformer.h[11].mlp.act.register_forward_hook(hook_fn)
    
    all_cons = []
    all_exc = []
    all_norms = []
    
    norm_data = {}
    def norm_hook_fn(m, i, o): norm_data['out'] = o.detach()
    nhook = model.transformer.h[11].mlp.register_forward_hook(norm_hook_fn)
    
    with torch.no_grad():
        for start in range(0, len(tokens) - SEQ_LEN + 1, SEQ_LEN):
            chunk = torch.tensor(tokens[start:start+SEQ_LEN], dtype=torch.long).unsqueeze(0).cuda()
            model(chunk)
            act = post_gelu['act'].squeeze(0)
            all_cons.append((act[:, CONSENSUS] > THRESH).cpu())
            all_exc.append((act[:, EXCEPTION] > THRESH).cpu())
            all_norms.append(norm_data['out'].squeeze(0).norm(dim=-1).cpu())
    
    hook.remove()
    nhook.remove()
    
    cf = torch.cat(all_cons).numpy()
    ef = torch.cat(all_exc).numpy()
    norms = torch.cat(all_norms).numpy()
    cc = cf.sum(axis=1)
    N = len(cc)
    
    print(f"\n=== {name} ===")
    print(f"N2123 overall fire rate: {ef.mean()*100:.1f}%")
    print(f"Consensus neuron avg fire rate: {cf.mean()*100:.1f}%")
    
    print(f"\n{'Cons':>4} | {'Count':>7} | {'%':>6} | {'N2123 FR':>9} | {'Norm':>8}")
    for c in range(8):
        mask = cc == c
        n = mask.sum()
        if n > 10:
            fr = ef[mask].mean() * 100
            nm = norms[mask].mean()
            print(f"{c}/7  | {n:>7,} | {n/N*100:>5.1f}% | {fr:>8.1f}% | {nm:>8.1f}")
    
    # Gradient
    mask0, mask7 = cc == 0, cc == 7
    if mask0.sum() > 0 and mask7.sum() > 0:
        grad = ef[mask0].mean()*100 - ef[mask7].mean()*100
        nratio = norms[mask0].mean() / norms[mask7].mean() if norms[mask7].mean() > 0 else float('nan')
        print(f"\nGradient range: {grad:.1f}pp")
        print(f"Norm ratio: {nratio:.2f}x")
    
    # Exclusivity
    excl_vals = []
    for i in range(7):
        both = (ef & cf[:, i]).sum()
        either = (ef | cf[:, i]).sum()
        if either > 0:
            excl_vals.append((1 - both/either) * 100)
    if excl_vals:
        print(f"Avg exclusivity: {np.mean(excl_vals):.1f}%")
    
    # Monotonicity
    rates = []
    for c in range(8):
        mask = cc == c
        if mask.sum() > 10:
            rates.append(ef[mask].mean())
    monotonic = all(a >= b for a, b in zip(rates, rates[1:]))
    print(f"Monotonic: {monotonic}")

analyze_model(model_rand, "RANDOM WEIGHTS")
analyze_model(model_trained, "TRAINED (control)")

print(f"\nTotal time: {time.time()-t0:.0f}s")
