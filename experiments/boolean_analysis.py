#!/usr/bin/env python3
"""
Rigorous boolean extraction at 500K tokens, memory-efficient.
Uses nonlinearity delta (least-squares residual) as the metric.
"""
import numpy as np
import torch
import gc
import time
from pathlib import Path
from collections import Counter

def main():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    for p in [
        Path("/data/tokens/wikitext103_train_tokens.npy"),
        Path("/home/ubuntu/clawd/projects/sense-stack/code/wikitext103_train_tokens.npy"),
    ]:
        if p.exists():
            all_tokens = np.array(np.load(str(p), mmap_mode='r')[:1_100_000])
            print(f"Loaded from {p}")
            break
    
    layer = 11
    mlp = model.transformer.h[layer].mlp
    seq_len = 128
    n_seqs = 500_000 // seq_len  # 3906
    N = n_seqs * seq_len  # 499,968
    
    # Collect post_gelu, mlp_input, mlp_output in float32 chunks
    pg_chunks, mi_chunks, mo_chunks = [], [], []
    bufs = {"pg": [], "mi": [], "mo": []}
    
    def ih(m, a): bufs["mi"].append(a[0].detach().cpu().to(torch.float32).reshape(-1, a[0].shape[-1]))
    def ph(m, a, o): bufs["pg"].append(o.detach().cpu().to(torch.float32).reshape(-1, o.shape[-1]))
    def oh(m, a, o):
        y = o if not isinstance(o, tuple) else o[0]
        bufs["mo"].append(y.detach().cpu().to(torch.float32).reshape(-1, y.shape[-1]))
    
    h1 = mlp.register_forward_pre_hook(ih)
    h2 = mlp.act.register_forward_hook(ph)
    h3 = mlp.c_proj.register_forward_hook(oh)
    
    tokens_t = torch.tensor(all_tokens[:N], dtype=torch.long).reshape(n_seqs, seq_len)
    
    t0 = time.time()
    print(f"Collecting {N:,} tokens...")
    with torch.no_grad():
        for i in range(0, n_seqs, 64):
            if i % 640 == 0: print(f"  {i}/{n_seqs} ({time.time()-t0:.0f}s)")
            model(tokens_t[i:i+64].to(device))
            pg_chunks.append(bufs["pg"][-1].numpy())
            mi_chunks.append(bufs["mi"][-1].numpy())
            mo_chunks.append(bufs["mo"][-1].numpy())
            bufs["pg"].clear(); bufs["mi"].clear(); bufs["mo"].clear()
    
    h1.remove(); h2.remove(); h3.remove()
    
    # Free GPU
    del model; gc.collect()
    if device == "cuda": torch.cuda.empty_cache()
    
    post_gelu = np.concatenate(pg_chunks)[:N]; del pg_chunks
    mlp_input = np.concatenate(mi_chunks)[:N]; del mi_chunks
    mlp_output = np.concatenate(mo_chunks)[:N]; del mo_chunks
    token_ids = all_tokens[:N]
    gc.collect()
    
    print(f"Collected {N:,} tokens in {time.time()-t0:.0f}s")
    
    # ====== Compute nonlinearity delta ======
    print("Computing least-squares fit (subsampled 50K)...")
    rng = np.random.RandomState(42)
    fit_idx = rng.choice(N, 50000, replace=False)
    X_fit = np.hstack([mlp_input[fit_idx], np.ones((50000, 1), dtype=np.float32)])
    W_aug = np.linalg.lstsq(X_fit.astype(np.float64), mlp_output[fit_idx].astype(np.float64), rcond=None)[0].astype(np.float32)
    del X_fit
    
    # Compute delta norms in chunks
    delta_norms = np.zeros(N, dtype=np.float32)
    for i in range(0, N, 50000):
        s = slice(i, min(i+50000, N))
        pred = mlp_input[s] @ W_aug[:-1] + W_aug[-1]
        delta_norms[s] = np.linalg.norm(mlp_output[s] - pred, axis=1)
    
    del mlp_input, mlp_output, W_aug; gc.collect()
    print(f"  Delta distribution: mean={delta_norms.mean():.3f}, median={np.median(delta_norms):.3f}, p95={np.percentile(delta_norms, 95):.3f}")
    
    # ====== Define groups ======
    linear = delta_norms < np.percentile(delta_norms, 25)
    barely = (delta_norms >= np.percentile(delta_norms, 50)) & (delta_norms < np.percentile(delta_norms, 70))
    highly = delta_norms > np.percentile(delta_norms, 95)
    
    print(f"\nGroups: linear={linear.sum():,}, barely={barely.sum():,}, highly={highly.sum():,}")
    
    # ====== Neuron analysis ======
    threshold = 0.1
    fire = (post_gelu > threshold)
    
    fire_linear = fire[linear].mean(axis=0)
    fire_barely = fire[barely].mean(axis=0)
    fire_highly = fire[highly].mean(axis=0)
    
    fire_diff_barely = fire_barely - fire_linear
    fire_diff_highly = fire_highly - fire_linear
    
    top_barely = np.argsort(np.abs(fire_diff_barely))[::-1][:20]
    top_highly = np.argsort(np.abs(fire_diff_highly))[::-1][:20]
    
    print(f"\n  Top 20 neurons distinguishing BARELY-NONLINEAR from linear:")
    print(f"  {'Rank':>4} {'Neuron':>8} {'Lin%':>7} {'Barely%':>8} {'Diff':>8} {'Direction':>10}")
    for rank, idx in enumerate(top_barely):
        direction = "MORE" if fire_diff_barely[idx] > 0 else "LESS"
        print(f"  {rank+1:>4} N{idx:>6} {fire_linear[idx]:>7.1%} {fire_barely[idx]:>8.1%} "
              f"{fire_diff_barely[idx]:>+8.1%} {direction:>10}")
    
    print(f"\n  Top 20 neurons distinguishing HIGHLY-NONLINEAR from linear:")
    print(f"  {'Rank':>4} {'Neuron':>8} {'Lin%':>7} {'High%':>8} {'Diff':>8} {'Direction':>10}")
    for rank, idx in enumerate(top_highly):
        direction = "MORE" if fire_diff_highly[idx] > 0 else "LESS"
        print(f"  {rank+1:>4} N{idx:>6} {fire_linear[idx]:>7.1%} {fire_highly[idx]:>8.1%} "
              f"{fire_diff_highly[idx]:>+8.1%} {direction:>10}")
    
    # ====== Boolean patterns with top 8 barely neurons ======
    TOP_K = 8
    top_neurons = top_barely[:TOP_K]
    print(f"\n{'='*70}")
    print(f"BOOLEAN PATTERNS — top {TOP_K} neurons: {list(top_neurons)}")
    print(f"{'='*70}")
    
    binary_barely = fire[barely][:, top_neurons].astype(np.int8)
    binary_linear = fire[linear][:, top_neurons].astype(np.int8)
    
    # Convert to pattern strings efficiently
    powers = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.int16)[:TOP_K]
    codes_barely = (binary_barely * powers).sum(axis=1)
    codes_linear = (binary_linear * powers).sum(axis=1)
    
    count_barely = Counter(codes_barely.tolist())
    count_linear = Counter(codes_linear.tolist())
    n_barely = barely.sum()
    n_linear = linear.sum()
    
    print(f"\n  Top enriched patterns (min 100 tokens in barely-NL):")
    print(f"  {'Pattern':>10} {'Count':>6} {'Barely%':>8} {'Linear%':>8} {'Ratio':>7} {'Top tokens'}")
    
    enriched = []
    for code in count_barely:
        bc = count_barely[code]
        if bc < 100: continue
        lc = count_linear.get(code, 0)
        b_frac = bc / n_barely
        l_frac = lc / n_linear
        ratio = b_frac / max(l_frac, 1e-8)
        
        pat_str = format(code, f'0{TOP_K}b')
        
        # Get tokens for this pattern
        mask = codes_barely == code
        barely_idx = np.where(barely)[0]
        toks = token_ids[barely_idx[mask]]
        top_toks = Counter(toks.tolist()).most_common(6)
        tok_str = ", ".join(f"'{tokenizer.decode([t]).strip()}'" for t, _ in top_toks)
        
        enriched.append((pat_str, bc, ratio, b_frac, l_frac, tok_str))
    
    enriched.sort(key=lambda x: x[2], reverse=True)
    for pat, bc, ratio, bf, lf, tok_str in enriched[:20]:
        print(f"  {pat:>10} {bc:>6} {bf:>8.2%} {lf:>8.2%} {ratio:>7.1f}x {tok_str}")
    
    # ====== Neuron descriptions ======
    # Reload model just for weights (lightweight)
    from transformers import GPT2LMHeadModel
    model2 = GPT2LMHeadModel.from_pretrained("gpt2").cpu().eval()
    W1 = model2.transformer.h[layer].mlp.c_fc.weight.data.numpy().T  # (3072, 768)
    b1 = model2.transformer.h[layer].mlp.c_fc.bias.data.numpy()
    W2 = model2.transformer.h[layer].mlp.c_proj.weight.data.numpy().T  # (768, 3072)
    del model2; gc.collect()
    
    print(f"\n{'='*70}")
    print(f"NEURON DESCRIPTIONS (top {TOP_K})")
    print(f"{'='*70}")
    
    pre_gelu = post_gelu  # We don't have pre_gelu, use post for token analysis
    
    for n_idx in top_neurons:
        activations = post_gelu[:, n_idx]
        top_act_idx = np.argsort(activations)[-30:][::-1]
        top_toks = Counter([token_ids[i] for i in top_act_idx]).most_common(5)
        tok_str = ", ".join(f"'{tokenizer.decode([t]).strip()}'" for t, _ in top_toks)
        
        bot_act_idx = np.argsort(activations)[:30]
        bot_toks = Counter([token_ids[i] for i in bot_act_idx]).most_common(5)
        bot_str = ", ".join(f"'{tokenizer.decode([t]).strip()}'" for t, _ in bot_toks)
        
        output_norm = np.linalg.norm(W2[:, n_idx])
        fires_more = fire_diff_barely[n_idx] > 0
        direction = "fires MORE" if fires_more else "fires LESS"
        
        print(f"\n  N{n_idx} ({direction} for barely-NL, rank {list(top_neurons).index(n_idx)+1}):")
        print(f"    Fires strongest for: {tok_str}")
        print(f"    Suppressed for: {bot_str}")
        print(f"    Bias: {b1[n_idx]:.3f}")
        print(f"    Output magnitude: {output_norm:.3f}")
        print(f"    Fire rate: linear={fire_linear[n_idx]:.1%}, barely={fire_barely[n_idx]:.1%}, highly={fire_highly[n_idx]:.1%}")
    
    # ====== Generate pseudocode ======
    print(f"\n{'='*70}")
    print(f"GENERATED PSEUDOCODE")
    print(f"{'='*70}")
    print(f"# Layer 11 MLP — extracted Boolean logic")
    print(f"# Top {TOP_K} discriminative neurons (by nonlinearity delta, 500K tokens)")
    print(f"# Neurons: {', '.join(f'N{n}' for n in top_neurons)}")
    print(f"#")
    
    for pat, bc, ratio, bf, lf, tok_str in enriched[:8]:
        if ratio < 2.0: continue
        conditions = []
        for i, bit in enumerate(pat):
            n_idx = top_neurons[i]
            if bit == "1":
                conditions.append(f"N{n_idx}")
            else:
                not_neurons = [top_neurons[j] for j in range(TOP_K) if j != i and pat[j] == "0"]
                # just use NOT for this neuron
                conditions.append(f"NOT N{n_idx}")
        
        # Simplified: show AND of firing neurons, NOT of silent ones
        firing = [f"N{top_neurons[i]}" for i in range(TOP_K) if pat[i] == "1"]
        silent = [f"{top_neurons[i]}" for i in range(TOP_K) if pat[i] == "0"]
        
        fire_str = " AND ".join(firing) if firing else "NONE fire"
        silent_str = ",".join(silent) if silent else "none"
        
        print(f"# Pattern {pat} ({ratio:.1f}x enriched): {tok_str[:60]}")
        print(f"IF {fire_str} AND NOT({silent_str}):")
        print(f"    # {bc} tokens ({bf:.1%} of barely-NL)")
        print(f"    apply_correction()")
        print()
    
    print(f"ELSE:")
    print(f"    use_linear_default()")
    
    # ====== Pairwise interactions ======
    print(f"\n{'='*70}")
    print(f"PAIRWISE NEURON INTERACTIONS (top highly-NL neurons)")
    print(f"{'='*70}")
    
    top_high = top_highly[:10]
    fire_int = fire.astype(np.int8)
    
    print(f"\n  Anti-correlated pairs (mutual exclusivity > 90%):")
    for i in range(min(10, len(top_high))):
        for j in range(i+1, min(10, len(top_high))):
            ni, nj = top_high[i], top_high[j]
            both = (fire_int[:, ni] == 1) & (fire_int[:, nj] == 1)
            either = (fire_int[:, ni] == 1) | (fire_int[:, nj] == 1)
            if either.sum() < 50: continue
            excl = 1 - both.sum() / max(either.sum(), 1)
            if excl > 0.90:
                print(f"  (N{ni:>4}, N{nj:>4}): {excl:.1%} exclusive")
    
    # ====== Consensus gradient ======
    print(f"\n{'='*70}")
    print(f"CONSENSUS GRADIENT")
    print(f"{'='*70}")
    consensus_neurons = [2, 762, 2361, 2460, 2928, 1831, 2727]
    cc = fire_int[:, consensus_neurons].sum(axis=1)
    ef = fire_int[:, 2123]
    mlp_norms = np.linalg.norm(np.concatenate([np.zeros((1, 768), dtype=np.float32)] * 0), axis=1) if False else delta_norms  # use delta_norms
    
    print(f"{'Count':>6} {'%tok':>7} {'N2123%':>8} {'AvgDelta':>10}")
    for c in range(8):
        m = cc == c
        if m.sum() == 0: continue
        print(f"{c:>6} {m.mean():>7.1%} {ef[m].mean():>8.1%} {delta_norms[m].mean():>10.1f}")
    
    print(f"\n  N2123 exclusivity with consensus neurons:")
    for n in consensus_neurons:
        both = (fire_int[:, 2123] == 1) & (fire_int[:, n] == 1)
        either = (fire_int[:, 2123] == 1) | (fire_int[:, n] == 1)
        excl = 1 - both.sum() / max(either.sum(), 1)
        print(f"    N{n:>4}: {excl:.1%} exclusive")
    
    print(f"\n{'='*70}")
    print(f"DONE — use generated pseudocode to update paper")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
