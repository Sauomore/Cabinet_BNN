# Testing, Status & Known Issues

[← Back to README](../README.md) ｜ [中文](TESTING.zh.md)

Correctness suites, current implementation status, and the issues that are
known and not yet fixed.

## Correctness tests

Four regression suites, all passing. They exist because each one caught a bug that
**silently produced wrong conclusions** rather than crashing:

```bash
python scripts/selftest_post_optimize.py   # greedy bit-flip: Δ formula vs measured ΔO
python scripts/selftest_attention.py       # causal mask leakage, KV-cache equivalence
python scripts/selftest_code_ffn.py        # einsum equivalence, batch indexing, geometry
python tools/verify_paper_math.py          # HSH-64 paper propositions, numerically
```


## Current status

| Component | State |
|---|---|
| HSH-64 codec + post-processing | ✅ verified against the paper |
| Bucket-structure validation | ✅ done (max bucket 3 / 256) |
| Causal attention (RoPE, KV cache) | ✅ implemented, regression-tested |
| Code-generated FFN (low-rank, memory-efficient) | ✅ implemented, regression-tested |
| Continual-learning evaluation harness | ✅ done |
| Bidirectional attention relation gate | ✅ implemented, regression-tested — ❌ **but it does not help** (see below) |
| Code-as-index geometry transfer | ✅ verified (no training needed) |
| **BNN mechanism wired into the trained LM** | ⬜ **in progress** |
| MoE-style sparse FFN, code-driven routing | ✅ implemented and tested — ⬜ fair comparison pending |
| New-token hot-add | ⬜ planned |
| SFT to conversational ability | ⬜ planned |

#### Negative result: the relation gate hurts

A per-token relation vector `u_t` gates the attention logits:
`att_ij = softmax_j( (q_i·k_j)/√D · (1 + γ·⟨u_i,u_j⟩/√r) )`.
The implementation is correct (γ=0 is a bit-exact identity, symmetric, KV-cache
equivalent to 4e-7, and γ does escape zero under training). **It nonetheless makes the
language model worse.**

Controlled run from the same checkpoint — identical seed, LR schedule, data order and
step count; the only difference is whether the gate is active:

| config | val_loss | ppl |
|---|---|---|
| base (start) | 3.0111 | 20.3 |
| gate **frozen** at γ=0 | **2.9668** | **19.4** |
| gate **trainable** | 2.9863 | 19.8 |

The gate-on run is worse by 0.0194, and the gap is stable across all five evaluation
points (+0.016 … +0.020), so it is not noise.

Three measured causes:

1. **The gate signal is far too weak.** Deviation scale `|γ|·std(s) ≈ 0.006` against an
   attention-logit standard deviation of 0.33 — a 1.8 % perturbation. It adds 4.1 M
   trainable parameters while barely doing anything.
2. **The relation vectors collapse.** 32 k tokens in `r = 16` dimensions: all pairwise
   cosines crowd into the 0.88–0.92 band, and the top-scoring "relations" are meaningless
   (`grape ↔ measure`).
3. **No co-occurrence structure is learned.** `corr(PMI, relation score) = −0.0021` over
   1.56 M token pairs. The model chose not to use the mechanism.

> **A methodological note, recorded because it matters.** The first observation was
> `3.0111 → 2.9863`, which looks like an improvement. It is not: without a control run
> the effect of the gate cannot be separated from the effect of 6000 extra training
> steps, and the entire apparent gain came from those extra steps. The control was
> essential, and the initial reading was wrong.

What survives is **not** the gate but the idea underneath it: `scripts/17` shows that
deriving `u_t` **from the code** transfers code geometry into relation geometry with **no
training at all** (corr −0.6969, 98.4 % cluster agreement vs a 6.2 % random baseline).
What failed is *this parameterisation* — a narrow scalar gate — not code-as-index.

## Known issues

- The `41.6 M` model was trained on a corpus with **77 % instruction data**, which is too
  high for pretraining and causes format overfitting (empty markdown tables, repeated
  lists). Correct ratio: general text 70–80 %, instruction data 10–20 % — reserve
  instruction data for SFT.
- **No README-level usage example for the BNN path yet**, because the mechanism is not
  wired into the LM.
- The bidirectional relation gate **is implemented but should not be enabled** — see the
  negative result above. `relation_gate=False` remains the recommended default.

---

[← Back to README](../README.md) ｜ [中文](../README.zh.md)
