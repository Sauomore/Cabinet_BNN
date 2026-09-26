# Results & Findings

[← Back to README](../README.md) ｜ [中文](RESULTS.zh.md)

All numbers below were produced by the scripts in this repository and are
reproducible. **Negative and inconclusive results are reported alongside the
positive ones** — they are the reason several design decisions were changed.


#### HSH-64 code reproduction (3,109 words, pure Hamming)

| | Recall@10 | pos-pair dist | separation |
|---|---|---|---|
| Before post-processing | 0.4988 | 19.28 bit | 11.45 bit |
| **After post-processing** | **0.7426** | **9.17 bit** | **16.91 bit** |
| Paper (h512) | 0.7404 | ≈8 | ≈18 |

Bucket structure: **3,075 semantic buckets for 3,109 words, largest bucket = 3**
(`abs` slot limit is 256) → the perfect hash holds with large margin.

> ⚠️ Configuration caveat: `bge-small` weights were unavailable locally, so the student
> input is a PCA-512 truncation of `bge-large` — the same model used as teacher. This
> **favours the numbers**, so they are not directly comparable to the paper.

#### BNN ablations — **negative result, reported as such**

Two questions, both answered with measured numbers. Full report:
[`docs/EVALUATION_REPORT.md`](EVALUATION_REPORT.md).

**(a) Does the code-generated weight beat a plain global weight at equal parameter budget?**

| mode | params | val_loss | tok_acc |
|---|---|---|---|
| `shared` (global weights) | 524,629 | **3.973** | **0.3094** |
| `code` (code-generated) | 452,928 | 4.162 | 0.2587 |
| `token_embed` (classic per-token) | 510,357 | 4.783 | 0.2252 |

→ `code` **loses** to `shared`, but **beats** `token_embed`.
**Conclusion: per-token parameterisation gives no gain when parameter count is not the
bottleneck; its value lies in *editability*, not *expressiveness*.**

**(b) Does per-domain code swapping avoid forgetting?**

| method | own-domain acc | forgetting | rollback |
|---|---|---|---|
| `shared_ft` (sequential fine-tune) | 0.2875 | 0.0301 | — |
| `code_param` (code bits only) | 0.2862 | 0.0543 | inexact |
| **`code_swap` (per-domain code table)** | **0.3278** | **0.0000** | **exact** |
| `code_full` | 0.2591 | 0.0678 | inexact |
| `shared_joint` (upper bound) | 0.3264 | 0 (non-incremental) | — |

→ Zero forgetting **by construction** (physically isolated per-domain state), bit-exact
rollback, and own-domain accuracy slightly above the joint-training upper bound.

> ⚠️ **Scope of claim.** Zero forgetting here comes from *physical isolation*, the same
> mechanism as PackNet / HAT / PNN — it is **paid for with capacity** (250 KB per domain),
> not a new mechanism. The contribution is doing it at lower cost with bit-level
> reversibility, **not** "solving catastrophic forgetting".

#### Language model

| model | params | tokens | val_loss | ppl |
|---|---|---|---|---|
| char-level (baseline) | 4.5 M | 111 M | 5.06 | 157.5 |
| char-level | 11.3 M | 111 M | 4.48 | 88.1 |
| **`base` (d=512, 8 layers, 32k BPE)** | **41.6 M** | **817 M** | **3.011** | **20.3** |

Cross-model comparison on identical Chinese text, using **bits-per-character** (the only
fair metric across different tokenizers):

| model | params | vocab | **bpc ↓** |
|---|---|---|---|
| this repo (41.6 M) | 41.6 M | 32k | **4.720** |
| Qwen2.5-0.5B | 494 M | 152k | **3.418** |

→ 38 % worse on bpc with **12× fewer parameters** and ~20,000× less training data.

> Note: the `41.6 M` model is a **standard Transformer** (`ffn_mode=global`) — the BNN
> weight-code mechanism is **not yet enabled** in it. It exists to serve as a trustworthy
> baseline. See *Current status*.


#### MoE-style sparse FFN — implemented and tested, comparison pending

Instead of summing over all `k` basis matrices, each token activates only `k'`:

```
dense:  W_t = Σ_{j=1..k}     s_j(t)·B_j
MoE:    W_t = Σ_{j ∈ TopK'}  s_j(t)·B_j
```

Routing is derived from the hash bits (`expert_j = (h >> off_j) % k`), so there is
**no router network and no load-balancing loss** — balance is structural.

Measured (base preset, `k=16`, `k'=4`, `d_ff=256`, `rank=16`):

| property | value |
|---|---|
| parameters per layer | **327,680** = 15.6 % of a standard FFN (2,097,152) |
| load balance | normalised entropy **1.0000**, all 16 experts used, max share 0.064 |
| sparse **write** | editing one token changes only its own `k'=4` experts; **0** other tokens affected |
| KV-cache equivalence | 1.2e-06 against the growing-prefix path |

The sparse-write property is the one that matters for the project's goal: the
blast radius of editing a token's code is `k'/k` rather than 1.

> ⚠️ **The fine-tuning comparison was not a fair test and is not reported as a
> result.** The MoE run replaced a *trained* FFN with a *random* one, so at
> step 1000 it was still relearning an FFN (val_loss 4.5492 against the control's
> ~2.97) and could not isolate the MoE mechanism. A fair comparison must train
> both sides from scratch, or reset the FFN on both sides. The run was stopped
> rather than reported.


---

[← Back to README](../README.md) ｜ [中文](../README.zh.md)
