"""Exact exclusivity counts with 1024-token sequences."""
import torch, numpy as np, time
from transformers import GPT2LMHeadModel

t0 = time.time()
model = GPT2LMHeadModel.from_pretrained("gpt2").eval().cuda()
tokens = np.load("/data/tokens/wikitext103_train_tokens.npy")[:500000]
CONSENSUS = [2, 2361, 2460, 2928, 1831, 1245, 2600]
EXCEPTION = 2123
SEQ_LEN = 1024
THRESH = 0.1

post_gelu = {}
def hook_fn(m, i, o): post_gelu['act'] = o.detach()
hook = model.transformer.h[11].mlp.act.register_forward_hook(hook_fn)

both_counts = np.zeros(7, dtype=np.int64)
union_counts = np.zeros(7, dtype=np.int64)

with torch.no_grad():
    for start in range(0, len(tokens) - SEQ_LEN + 1, SEQ_LEN):
        chunk = torch.tensor(tokens[start:start+SEQ_LEN], dtype=torch.long).unsqueeze(0).cuda()
        model(chunk)
        act = post_gelu['act'].squeeze(0)
        exc = (act[:, EXCEPTION] > THRESH).cpu().numpy()
        for i, cn in enumerate(CONSENSUS):
            cf = (act[:, cn] > THRESH).cpu().numpy()
            both_counts[i] += (exc & cf).sum()
            union_counts[i] += (exc | cf).sum()

hook.remove()
print("Neuron Pair | Both Fire | Union Fire | Exclusivity")
for i, cn in enumerate(CONSENSUS):
    excl = (1 - both_counts[i]/union_counts[i]) * 100
    print(f"2123 vs N{cn:>4} | {both_counts[i]:>9,} | {union_counts[i]:>10,} | {excl:.1f}%")
print(f"\nDone in {time.time()-t0:.0f}s")
