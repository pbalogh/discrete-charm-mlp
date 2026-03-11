# The Discrete Charm of the MLP

**Binary Routing of Continuous Signals in Transformer Feed-Forward Layers**

Peter Balogh, 2026

## Abstract

We show that MLP layers in transformer language models perform *binary routing of continuous signals*: the decision of whether a token needs nonlinear processing is well-captured by binary neuron activations, even though the signals being routed are continuous.

In GPT-2 Small (124M parameters), we find that specific neurons implement a **consensus architecture** — seven "default-ON" neurons and one exception handler (N2123) that are 93–98% mutually exclusive — creating a binary routing switch. A cross-layer analysis reveals a **three-phase developmental arc**: early layers use gateway routing, middle layers show diffuse processing, and late layers crystallize full consensus/exception architectures. Causal validation confirms the routing is functional: removing the MLP at consensus breakdown costs 43.3% perplexity, while at full consensus it costs only 10.1%.

A **random-weight control** confirms the consensus structure is learned, not architectural: untrained GPT-2 Small shows zero consensus gradient.

## Repository Structure

```
paper/          LaTeX source, figures, and style file
experiments/    All experiment scripts with README
```

## Quick Start

```bash
pip install torch transformers numpy
cd experiments
# Edit token path in scripts, then:
python consensus_table.py        # Core result: Two Regimes table
python random_weight_control.py  # Learned vs architectural control
python bootstrap_ci.py           # Confidence intervals
```

See `experiments/README.md` for full details on each script.

## Key Findings

1. **Binary routing**: MLP neurons operate in near-binary regimes; their co-activation patterns implement interpretable routing logic
2. **Consensus architecture**: 7 default-ON neurons + 1 exception handler, 93–98% mutually exclusive
3. **Three-phase developmental arc**: scaffold (L0–3) → diffuse (L4–6) → decision (L7–11)
4. **Distributional reshaping**: The MLP's contribution at consensus breakdown is primarily distributional (KL = 0.414) rather than concentrated on the correct token (boost = 1.15×)
5. **Learned, not architectural**: Random-weight GPT-2 shows no consensus structure

## Citation

```bibtex
@article{balogh2026discrete,
  title={The Discrete Charm of the {MLP}: Binary Routing of Continuous Signals in Transformer Feed-Forward Layers},
  author={Balogh, Peter},
  year={2026}
}
```

## License

MIT
