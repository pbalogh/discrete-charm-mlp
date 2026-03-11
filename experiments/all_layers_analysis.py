#!/usr/bin/env python3
"""
Full 12-layer MLP analysis for GPT-2 Small.
For each layer: consensus architecture, top neurons, boolean patterns, gateway detection.
Memory-efficient, GPU-accelerated.
"""
import numpy as np
import torch
import gc
import time
import json
from pathlib import Path
from collections import Counter

def analyze_layer(model, tokenizer, all_tokens, layer, device, N=499968):
    """Full analysis of one MLP layer. Returns structured results dict."""
    seq_len = 128
    n_seqs = N // seq_len
    N = n_seqs * seq_len
    
    mlp = model.transformer.h[layer].mlp
    
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
    with torch.no_grad():
        for i in range(0, n_seqs, 64):
            model(tokens_t[i:i+64].to(device))
            pg_chunks.append(bufs["pg"][-1].numpy())
            mi_chunks.append(bufs["mi"][-1].numpy())
            mo_chunks.append(bufs["mo"][-1].numpy())
            bufs["pg"].clear(); bufs["mi"].clear(); bufs["mo"].clear()
    
    h1.remove(); h2.remove(); h3.remove()
    
    post_gelu = np.concatenate(pg_chunks)[:N]; del pg_chunks
    mlp_input = np.concatenate(mi_chunks)[:N]; del mi_chunks
    mlp_output = np.concatenate(mo_chunks)[:N]; del mo_chunks
    token_ids = all_tokens[:N]
    
    collect_time = time.time() - t0
    
    # Compute nonlinearity delta
    rng = np.random.RandomState(42)
    fit_idx = rng.choice(N, 50000, replace=False)
    X_fit = np.hstack([mlp_input[fit_idx], np.ones((50000, 1), dtype=np.float32)])
    W_aug = np.linalg.lstsq(X_fit.astype(np.float64), mlp_output[fit_idx].astype(np.float64), rcond=None)[0].astype(np.float32)
    del X_fit
    
    delta_norms = np.zeros(N, dtype=np.float32)
    for i in range(0, N, 50000):
        s = slice(i, min(i+50000, N))
        pred = mlp_input[s] @ W_aug[:-1] + W_aug[-1]
        delta_norms[s] = np.linalg.norm(mlp_output[s] - pred, axis=1)
    
    output_norms = np.linalg.norm(mlp_output, axis=1).astype(np.float32)
    del mlp_input, mlp_output, W_aug; gc.collect()
    
    # Groups
    threshold = 0.1
    fire = (post_gelu > threshold)
    del post_gelu; gc.collect()
    
    linear = delta_norms < np.percentile(delta_norms, 25)
    barely = (delta_norms >= np.percentile(delta_norms, 50)) & (delta_norms < np.percentile(delta_norms, 70))
    highly = delta_norms > np.percentile(delta_norms, 95)
    
    fire_linear = fire[linear].mean(axis=0)
    fire_barely = fire[barely].mean(axis=0)
    fire_highly = fire[highly].mean(axis=0)
    fire_all = fire.mean(axis=0)
    
    fire_diff_barely = fire_barely - fire_linear
    fire_diff_highly = fire_highly - fire_linear
    
    # ====== Find consensus neurons (high fire rate, anti-correlated with exception) ======
    # Exception handler candidate: highest positive diff for highly-NL
    top_highly = np.argsort(np.abs(fire_diff_highly))[::-1][:30]
    
    # Exception handler: fires MORE for highly-NL tokens (positive diff, low base rate)
    exception_candidates = [(i, fire_diff_highly[i], fire_all[i]) 
                           for i in top_highly 
                           if fire_diff_highly[i] > 0.3 and fire_all[i] < 0.3]
    
    # Consensus neurons: fire LESS for highly-NL (negative diff, high base rate >60%)
    consensus_candidates = [(i, fire_diff_highly[i], fire_all[i])
                           for i in top_highly
                           if fire_diff_highly[i] < -0.4 and fire_all[i] > 0.6]
    
    # Build consensus gradient if we found the architecture
    fire_int = fire.astype(np.int8)
    consensus_gradient = None
    exception_neuron = None
    consensus_neurons = []
    exclusivities = {}
    
    if exception_candidates and consensus_candidates:
        exception_neuron = int(exception_candidates[0][0])
        # Take top 7 consensus neurons by |diff|
        consensus_candidates.sort(key=lambda x: x[1])  # most negative first
        consensus_neurons = [int(c[0]) for c in consensus_candidates[:7]]
        
        # Compute gradient
        cc = fire_int[:, consensus_neurons].sum(axis=1)
        ef = fire_int[:, exception_neuron]
        
        consensus_gradient = {}
        for c in range(len(consensus_neurons) + 1):
            m = cc == c
            if m.sum() == 0: continue
            consensus_gradient[c] = {
                "pct_tokens": float(m.mean()),
                "exception_fire_rate": float(ef[m].mean()),
                "avg_delta": float(delta_norms[m].mean()),
                "avg_output_norm": float(output_norms[m].mean()),
                "n_tokens": int(m.sum()),
            }
        
        # Exclusivity
        for n in consensus_neurons:
            both = (fire_int[:, exception_neuron] == 1) & (fire_int[:, n] == 1)
            either = (fire_int[:, exception_neuron] == 1) | (fire_int[:, n] == 1)
            excl = 1 - both.sum() / max(either.sum(), 1)
            exclusivities[int(n)] = float(excl)
    
    # ====== Barely-NL discriminators ======
    top_barely = np.argsort(np.abs(fire_diff_barely))[::-1][:20]
    barely_discriminators = []
    for idx in top_barely:
        barely_discriminators.append({
            "neuron": int(idx),
            "fire_rate_linear": float(fire_linear[idx]),
            "fire_rate_barely": float(fire_barely[idx]),
            "fire_rate_highly": float(fire_highly[idx]),
            "diff": float(fire_diff_barely[idx]),
            "direction": "MORE" if fire_diff_barely[idx] > 0 else "LESS",
        })
    
    # ====== Boolean patterns with top 8 barely neurons ======
    top8 = top_barely[:8]
    binary_barely = fire[barely][:, top8].astype(np.int8)
    binary_linear = fire[linear][:, top8].astype(np.int8)
    
    powers = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.int16)
    codes_barely = (binary_barely * powers).sum(axis=1)
    codes_linear = (binary_linear * powers).sum(axis=1)
    
    count_barely = Counter(codes_barely.tolist())
    count_linear = Counter(codes_linear.tolist())
    n_barely = barely.sum()
    n_linear = linear.sum()
    
    patterns = []
    for code in count_barely:
        bc = count_barely[code]
        if bc < 50: continue
        lc = count_linear.get(code, 0)
        b_frac = bc / n_barely
        l_frac = lc / n_linear
        ratio = b_frac / max(l_frac, 1e-8)
        pat_str = format(code, '08b')
        
        # Get tokens
        mask = codes_barely == code
        barely_idx = np.where(barely)[0]
        toks = token_ids[barely_idx[mask]]
        top_toks = Counter(toks.tolist()).most_common(6)
        tok_str = [tokenizer.decode([t]).strip() for t, _ in top_toks]
        
        # Which neurons fire in this pattern?
        firing_neurons = [int(top8[i]) for i in range(8) if pat_str[i] == "1"]
        silent_neurons = [int(top8[i]) for i in range(8) if pat_str[i] == "0"]
        
        patterns.append({
            "pattern": pat_str,
            "count": int(bc),
            "barely_pct": float(b_frac),
            "linear_pct": float(l_frac),
            "enrichment": float(ratio),
            "top_tokens": tok_str,
            "firing_neurons": firing_neurons,
            "silent_neurons": silent_neurons,
        })
    
    patterns.sort(key=lambda x: x["enrichment"], reverse=True)
    
    # ====== Gateway detection ======
    # Does any single neuron appear in most enriched patterns?
    gateway = None
    if patterns:
        top20 = patterns[:20]
        for i in range(8):
            n_idx = int(top8[i])
            appears_in = sum(1 for p in top20 if n_idx in p["firing_neurons"])
            if appears_in >= 15:  # appears in 75%+ of top patterns
                gateway = {
                    "neuron": n_idx,
                    "appears_in": appears_in,
                    "of_total": len(top20),
                    "fire_rate_linear": float(fire_linear[n_idx]),
                    "fire_rate_barely": float(fire_barely[n_idx]),
                    "direction": "MORE" if fire_diff_barely[n_idx] > 0 else "LESS",
                }
                break
    
    del fire, fire_int; gc.collect()
    
    result = {
        "layer": layer,
        "n_tokens": N,
        "collect_time_s": collect_time,
        "delta_stats": {
            "mean": float(delta_norms.mean()),
            "median": float(np.median(delta_norms)),
            "p95": float(np.percentile(delta_norms, 95)),
        },
        "output_norm_stats": {
            "mean": float(output_norms.mean()),
            "median": float(np.median(output_norms)),
            "p95": float(np.percentile(output_norms, 95)),
        },
        "consensus_architecture": {
            "found": exception_neuron is not None,
            "exception_neuron": exception_neuron,
            "exception_fire_rate": float(fire_all[exception_neuron]) if exception_neuron is not None else None,
            "consensus_neurons": consensus_neurons,
            "consensus_fire_rates": {int(n): float(fire_all[n]) for n in consensus_neurons},
            "gradient": consensus_gradient,
            "exclusivities": exclusivities,
        },
        "barely_nl_discriminators": barely_discriminators[:20],
        "top8_neurons": [int(n) for n in top8],
        "gateway": gateway,
        "patterns": patterns[:30],  # top 30 patterns
    }
    
    return result


def print_layer_summary(r):
    """Pretty-print one layer's results."""
    layer = r["layer"]
    ca = r["consensus_architecture"]
    
    print(f"\n{'='*70}")
    print(f"LAYER {layer}")
    print(f"{'='*70}")
    print(f"  Delta: mean={r['delta_stats']['mean']:.1f}, median={r['delta_stats']['median']:.1f}, p95={r['delta_stats']['p95']:.1f}")
    print(f"  Output norm: mean={r['output_norm_stats']['mean']:.1f}, p95={r['output_norm_stats']['p95']:.1f}")
    
    if ca["found"]:
        print(f"\n  CONSENSUS ARCHITECTURE: YES")
        print(f"    Exception handler: N{ca['exception_neuron']} (fires {ca['exception_fire_rate']:.1%} overall)")
        print(f"    Consensus neurons: {ca['consensus_neurons']}")
        print(f"    Exclusivities: {', '.join(f'N{n}:{v:.1%}' for n,v in ca['exclusivities'].items())}")
        
        if ca["gradient"]:
            print(f"    Gradient:")
            print(f"    {'Count':>6} {'%tok':>7} {'Exc%':>7} {'AvgΔ':>8}")
            for c in sorted(ca["gradient"].keys()):
                g = ca["gradient"][c]
                print(f"    {c:>6} {g['pct_tokens']:>7.1%} {g['exception_fire_rate']:>7.1%} {g['avg_delta']:>8.1f}")
    else:
        print(f"\n  CONSENSUS ARCHITECTURE: NOT FOUND")
        print(f"    (no neuron met criteria: >30% highly-NL diff AND <30% base rate)")
    
    # Top barely-NL discriminators
    print(f"\n  Top 8 barely-NL discriminators:")
    for d in r["barely_nl_discriminators"][:8]:
        print(f"    N{d['neuron']:>4} {d['direction']:>4} lin={d['fire_rate_linear']:.1%} barely={d['fire_rate_barely']:.1%} diff={d['diff']:+.1%}")
    
    # Gateway
    if r["gateway"]:
        g = r["gateway"]
        print(f"\n  GATEWAY NEURON: N{g['neuron']} (in {g['appears_in']}/{g['of_total']} top patterns, fires {g['direction']})")
    else:
        print(f"\n  GATEWAY NEURON: none detected")
    
    # Top patterns
    if r["patterns"]:
        print(f"\n  Top 5 enriched patterns:")
        for p in r["patterns"][:5]:
            toks = ", ".join(f"'{t}'" for t in p["top_tokens"][:4])
            print(f"    {p['pattern']} {p['enrichment']:>6.1f}x ({p['count']:>5} tokens) {toks}")


def main():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"{'='*70}")
    print(f"GPT-2 Small: Full 12-Layer MLP Analysis (500K tokens)")
    print(f"{'='*70}")
    
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    for p in [
        Path("/data/tokens/wikitext103_train_tokens.npy"),
        Path("/home/ubuntu/clawd/projects/sense-stack/code/wikitext103_train_tokens.npy"),
    ]:
        if p.exists():
            all_tokens = np.array(np.load(str(p), mmap_mode='r')[:1_100_000])
            print(f"Loaded tokens from {p}")
            break
    
    all_results = []
    t_start = time.time()
    
    for layer in range(12):
        print(f"\n>>> Analyzing layer {layer}/11...")
        t0 = time.time()
        
        result = analyze_layer(model, tokenizer, all_tokens, layer, device)
        all_results.append(result)
        
        print_layer_summary(result)
        print(f"  (layer took {time.time()-t0:.0f}s)")
        
        # Save incrementally
        out_path = Path("/data/all_layers_analysis.json")
        if not out_path.parent.exists():
            out_path = Path("/home/ubuntu/all_layers_analysis.json")
        with open(str(out_path), "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Saved to {out_path}")
    
    total_time = time.time() - t_start
    
    # ====== Cross-layer summary ======
    print(f"\n\n{'='*70}")
    print(f"CROSS-LAYER SUMMARY ({total_time:.0f}s total)")
    print(f"{'='*70}")
    
    print(f"\n{'Layer':>5} {'Consensus?':>10} {'Exception':>10} {'ExcRate':>8} {'Gateway':>10} {'TopEnrich':>10} {'MeanΔ':>7}")
    for r in all_results:
        ca = r["consensus_architecture"]
        exc = f"N{ca['exception_neuron']}" if ca['found'] else "—"
        exc_rate = f"{ca['exception_fire_rate']:.1%}" if ca['found'] else "—"
        gw = f"N{r['gateway']['neuron']}" if r['gateway'] else "—"
        top_e = f"{r['patterns'][0]['enrichment']:.1f}x" if r['patterns'] else "—"
        print(f"{r['layer']:>5} {'YES' if ca['found'] else 'NO':>10} {exc:>10} {exc_rate:>8} {gw:>10} {top_e:>10} {r['delta_stats']['mean']:>7.1f}")
    
    print(f"\nDone! Results saved to {out_path}")


if __name__ == "__main__":
    main()
