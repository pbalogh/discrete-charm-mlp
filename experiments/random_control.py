#!/usr/bin/env python3
"""Random neuron control v2 — memory-efficient."""
import numpy as np
import torch
import time

def main():
    from transformers import GPT2LMHeadModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    tokens_all = np.load("/data/tokens/wikitext103_train_tokens.npy", mmap_mode='r')[:500_000]
    
    seq_len = 128
    n_seqs = len(tokens_all) // seq_len
    tokens_t = torch.tensor(tokens_all[:n_seqs * seq_len], dtype=torch.long).reshape(n_seqs, seq_len)
    total = n_seqs * seq_len
    
    threshold = 0.1
    
    # Only store fire rates + the specific neurons we need
    print("Computing fire rates for all 3072 neurons...")
    t0 = time.time()
    captured = {}
    def hook_c_fc(module, input, output):
        captured['c_fc_out'] = output.detach()
    handle = model.transformer.h[11].mlp.c_fc.register_forward_hook(hook_c_fc)
    captured2 = {}
    def hook_mlp(module, input, output):
        captured2['mlp_out'] = output.detach()
    handle2 = model.transformer.h[11].mlp.register_forward_hook(hook_mlp)
    mlp = model.transformer.h[11].mlp
    
    fire_counts = np.zeros(3072, dtype=np.int64)
    
    # Store firing for consensus + exception + norms
    real_consensus = [2, 2361, 2460, 2928, 1831, 1245, 2600]
    real_exception = 2123
    
    # For random trials, we need to store all neuron firings — use packed bits
    # Actually just store firing for a random subset of candidate neurons
    # First pass: get fire rates
    with torch.no_grad():
        for i in range(0, n_seqs, 32):
            batch = tokens_t[i:i+32].to(device)
            _ = model(batch)
            post = mlp.act(captured['c_fc_out']).reshape(-1, 3072)
            fire_counts += (post > threshold).sum(dim=0).cpu().numpy()
            captured.clear(); captured2.clear()
    handle.remove(); handle2.remove()
    
    fire_rates = fire_counts / total
    high_fire = np.where(fire_rates > 0.5)[0]
    low_fire = np.where((fire_rates > 0.01) & (fire_rates < 0.10))[0]
    print(f"  {len(high_fire)} high-firing, {len(low_fire)} low-firing ({time.time()-t0:.0f}s)")
    
    # Second pass: store only the neurons we need (consensus + exception + candidates)
    # Keep all high_fire and low_fire neurons
    keep_neurons = sorted(set(list(high_fire) + list(low_fire) + real_consensus + [real_exception]))
    neuron_idx = {n: i for i, n in enumerate(keep_neurons)}
    n_keep = len(keep_neurons)
    print(f"  Keeping {n_keep} neurons for random trials")
    
    print("Second pass: collecting firing patterns + norms...")
    handle = model.transformer.h[11].mlp.c_fc.register_forward_hook(hook_c_fc)
    handle2 = model.transformer.h[11].mlp.register_forward_hook(hook_mlp)
    
    # Use uint8 to save memory
    all_fires = np.zeros((total, n_keep), dtype=np.uint8)
    all_norms = np.zeros(total, dtype=np.float32)
    idx = 0
    
    with torch.no_grad():
        for i in range(0, n_seqs, 32):
            batch = tokens_t[i:i+32].to(device)
            _ = model(batch)
            post = mlp.act(captured['c_fc_out']).reshape(-1, 3072)
            fires = (post[:, keep_neurons] > threshold).cpu().numpy().astype(np.uint8)
            norms = torch.norm(captured2['mlp_out'].reshape(-1, 768), dim=1).cpu().numpy()
            n = len(fires)
            all_fires[idx:idx+n] = fires
            all_norms[idx:idx+n] = norms
            idx += n
            captured.clear(); captured2.clear()
    handle.remove(); handle2.remove()
    
    print(f"  Done ({time.time()-t0:.0f}s)")
    
    # Real metrics
    rc_idx = [neuron_idx[n] for n in real_consensus]
    re_idx = neuron_idx[real_exception]
    
    real_cons_count = all_fires[:, rc_idx].sum(axis=1)
    real_exc = all_fires[:, re_idx].astype(bool)
    
    rate_0 = real_exc[real_cons_count == 0].mean() * 100
    rate_7 = real_exc[real_cons_count == 7].mean() * 100
    real_range = rate_0 - rate_7
    
    norm_0 = all_norms[real_cons_count == 0].mean()
    norm_7 = all_norms[real_cons_count == 7].mean()
    real_norm_ratio = norm_0 / norm_7
    
    real_excls = []
    for n in real_consensus:
        ni = neuron_idx[n]
        both = (all_fires[:, re_idx] & all_fires[:, ni]).sum()
        either = (all_fires[:, re_idx] | all_fires[:, ni]).sum()
        real_excls.append(1.0 - both / either if either > 0 else 0)
    real_avg_excl = np.mean(real_excls) * 100
    
    print(f"\nReal neurons:")
    print(f"  Gradient range: {real_range:.1f}pp")
    print(f"  Norm ratio: {real_norm_ratio:.2f}x")
    print(f"  Avg exclusivity: {real_avg_excl:.1f}%")
    
    # Random trials
    print(f"\nRunning 1000 random trials...")
    rng = np.random.RandomState(42)
    
    hf_idx = [neuron_idx[n] for n in high_fire]
    lf_idx = [neuron_idx[n] for n in low_fire]
    
    n_beat_range = 0; n_beat_ratio = 0; n_beat_excl = 0
    best_range = 0
    
    for trial in range(1000):
        cons_idx = rng.choice(hf_idx, 7, replace=False)
        exc_idx = rng.choice(lf_idx, 1)[0]
        
        cons_count = all_fires[:, cons_idx].sum(axis=1)
        exc = all_fires[:, exc_idx].astype(bool)
        
        mask_0 = cons_count == 0
        mask_7 = cons_count == 7
        
        if mask_0.sum() > 100 and mask_7.sum() > 100:
            r0 = exc[mask_0].mean() * 100
            r7 = exc[mask_7].mean() * 100
            trial_range = r0 - r7
            n0 = all_norms[mask_0].mean()
            n7 = all_norms[mask_7].mean()
            trial_ratio = n0 / n7 if n7 > 0 else 0
        else:
            trial_range = 0
            trial_ratio = 0
        
        if trial_range > best_range:
            best_range = trial_range
        if trial_range >= real_range:
            n_beat_range += 1
        if trial_ratio >= real_norm_ratio:
            n_beat_ratio += 1
        
        excls = []
        for ci in cons_idx:
            both = (all_fires[:, exc_idx] & all_fires[:, ci]).sum()
            either = (all_fires[:, exc_idx] | all_fires[:, ci]).sum()
            excls.append(1.0 - both / either if either > 0 else 0)
        if np.mean(excls) * 100 >= real_avg_excl:
            n_beat_excl += 1
    
    print(f"\nResults (1000 trials):")
    print(f"  Gradient range: {n_beat_range}/1000 beat {real_range:.1f}pp (best random: {best_range:.1f}pp)")
    print(f"  Norm ratio: {n_beat_ratio}/1000 beat {real_norm_ratio:.2f}x")
    print(f"  Avg exclusivity: {n_beat_excl}/1000 beat {real_avg_excl:.1f}%")
    print(f"\nDone in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
