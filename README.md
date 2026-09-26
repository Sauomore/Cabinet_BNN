# Cabinet-BNN

**English** ｜ [中文](README.zh.md)

> A binary-activation language model whose per-token weights are generated from a
> **structured 64-bit semantic hash code**, instead of being looked up from a global
> weight tensor.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![HSH-64](https://img.shields.io/badge/based%20on-HSH--64-informational)](https://github.com/Sauomore/Cabinet_hsh64)

---

### What this is

Cabinet-BNN explores a single question:

> **Can a token's weight matrix be *generated from* a compact discrete code, instead of
> being looked up from a global weight tensor?**

Concretely, each token `t` gets a **128-bit weight code**:

```
hash segment (64 bit)  =  feat(4) + sim(52) + abs(8)     ← the HSH-64 retrieval code
param segment (64 bit) =  per-token information code      ← new in this project
```

and its weight matrix is built from shared basis matrices:

```
s_j(t) = mask_j(hash_t) · p_t[j]                 (+1 / −1)
W_t    = Σ_j s_j(t) · (U_j V_jᵀ)                 shared low-rank basis
y      = sign(W_t x) · scale                     binary activation + STE
```

The `hash` segment comes from [HSH-64](https://github.com/Sauomore/Cabinet_hsh64) (the
author's previous work), which maps embeddings to a 64-bit code where **Hamming distance
≈ semantic distance**, queryable in one `popcnt`.


### At a glance

| component | state |
|---|---|
| HSH-64 codec + greedy bit-flip post-processing | ✅ verified against the paper (Recall@10 0.7426) |
| Causal attention + RoPE + KV cache | ✅ regression-tested |
| Code-generated FFN, low-rank and memory-efficient | ✅ regression-tested |
| Bidirectional attention relation gate | ✅ implemented — ❌ **measured to hurt** |
| MoE-style sparse FFN, code-driven routing | ✅ implemented and tested — ⬜ fair comparison pending |
| Code-as-index geometry transfer | ✅ verified with no training needed |
| Language model (41.6 M) | ✅ val_loss 3.0111 / ppl 20.3 |
| BNN mechanism wired into the trained LM | ⬜ **in progress** |

**Documents**

| | |
|---|---|
| **[Results & Findings](docs/RESULTS.md)** | every measured number, including negative results |
| **[Design & Research Direction](docs/DESIGN.md)** | the idea, what constrains it, the open problem |
| **[Architecture & Reproduction](docs/ARCHITECTURE.md)** | layout, install, full pipeline |
| **[Testing, Status & Known Issues](docs/TESTING.md)** | correctness suites, what works, what does not |

> ⚠️ **Two things this project does *not* claim**, both measured:
> the bidirectional relation gate makes the language model **worse**, and the
> zero forgetting from per-domain code tables comes from **physical isolation**
> (capacity cost), not from a new mechanism. See
> [RESULTS.md](docs/RESULTS.md) for the controlled comparisons.

### Why

Two motivations, and they should be kept separate:

| Goal | Status |
|---|---|
| **Lower cost** — binary activations, low-rank generated weights | Measured — see [RESULTS.md](docs/RESULTS.md) |
| **Editable parameters** — flip one bit to change weights; each domain keeps its own code table | Demonstrated on a small scale |

> ⚠️ **Binary ≠ brain-like.** Binarisation buys *cost*, not *intelligence*. The property
> that actually matters for continual learning is **locality** of updates — see
> [Design & Research Direction](docs/DESIGN.md).

### Quick start

```bash
git clone https://github.com/Sauomore/Cabinet_BNN.git
cd Cabinet_BNN
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

python -m cabinet_bnn.paths                      # check path resolution
python scripts/selftest_attention.py             # 4 regression suites, all should pass
python scripts/selftest_moe_ffn.py
```

Full pipeline: [ARCHITECTURE.md](docs/ARCHITECTURE.md).

### Acknowledgements

Built on **HSH-64** by the same author —
[github.com/Sauomore/Cabinet_hsh64](https://github.com/Sauomore/Cabinet_hsh64).
The structured bit layout (`feat4 + sim52 + abs8`), the Deep Hash projection head, the
recall-oriented greedy bit-flip algorithm and the codec semantics are from that work; this
repository extends the code from a *read-only retrieval key* to a *writable per-token
parameter carrier*.

The greedy bit-flipping optimisation is closely related to
[Greedy Hash (NeurIPS 2018)](https://proceedings.neurips.cc/paper_files/paper/2018/file/13f3cf8c531952d72e5847c4183e6910-Paper.pdf).

### License

MIT — see [LICENSE](LICENSE).
