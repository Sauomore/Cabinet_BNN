# Design & Research Direction

[← Back to README](../README.md) ｜ [中文](DESIGN.zh.md)

The idea underneath the implementation, the measured facts that constrain it,
and the open problem being worked on.

## Research direction

Measured fact that motivates the current work:

> An **independently randomised** `param` segment **destroys** the weight geometry:
> `corr(hash Hamming distance, weight cosine similarity) = +0.0019` (i.e. none).
> Using only the semantic `mask` gives `−0.9885` (near-perfect monotonicity).

Interpretation: a single `k`-dimensional sign vector **cannot** simultaneously carry the
*geometry* and provide *independent degrees of freedom*.

The direction being explored instead is **code-as-index**:

```
code (hash)     →  index        →  geometry is preserved by construction (locality of addressing)
information code →  content      →  free to hold anything; needs no geometric constraint
```

together with **MoE-style sparse activation**, so that editing one token's information
code touches only the `k'` basis matrices it actually activates — i.e. sparse writes, not
just sparse reads.

**Open problem** (this is where the author believes the elegant answer lies):

> Can a set of shared basis matrices be learned such that editing any single code has a
> **bounded** effect on all other tokens' outputs?
>
> If yes, this would be a parameterisation-level formulation of the
> stability–plasticity trade-off — and locality is exactly the mechanism by which
> biological synapses (STDP) achieve continual learning without global gradients.

---

[← Back to README](../README.md) ｜ [中文](../README.zh.md)
