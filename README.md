# Cabinet-BNN

**English** ｜ [中文](README.zh.md)

> A binary-activation language model whose per-token weights are generated from a
> **structured 64-bit semantic hash code**, instead of being looked up from a global
> weight tensor.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![HSH-64](https://img.shields.io/badge/based%20on-HSH--64-informational)](https://github.com/Sauomore/Cabinet_hsh64)

---

## English

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

### Why

Two motivations, and they should be kept separate:

| Goal | Status |
|---|---|
| **Lower cost** — binary activations, low-rank generated weights | Measured, see below |
| **Editable parameters** — flip one bit to change weights; each domain keeps its own code table | Demonstrated on a small scale |

> ⚠️ **Binary ≠ brain-like.** Binarisation buys *cost*, not *intelligence*. The property
> that actually matters for continual learning is **locality** of updates — see
> [Research direction](#research-direction) below.

### Repository layout

```
cabinet_bnn/
├── hsh/            HSH-64 codec — bit layout, popcount, greedy bit-flip post-processing
│   ├── codec.py            pack / unpack / Hamming / perfect-hash seed search
│   ├── deep_hash.py        MLP projection head with STE (binary-compatible with HSH-64)
│   ├── post_optimize.py    recall-oriented greedy bit flipping (paper Algorithm 1)
│   ├── metrics.py          Recall@K, distance distributions, codebook health
│   └── pos_map.py          POS → feat (4-bit) mapping
├── bnn/            Binary-activation network
│   ├── model.py            WeightCodeTable, CodeWeightLinear, binary activation + STE
│   ├── code_ffn.py         memory-efficient low-rank code-generated FFN
│   ├── attention.py        causal self-attention (RoPE, KV cache) + BNN Transformer
│   ├── corpus.py           char-level corpus construction
│   └── domains.py          domain splitting for continual-learning evaluation
├── data/           embedding cache I/O
└── paths.py        single point of path resolution (no hard-coded absolute paths)

scripts/            full pipeline, numbered by stage
tools/              paper-math verification, weight conversion
docs/               technical roadmap, evaluation report, figures
```

### Install

```bash
# 1. Install PyTorch matching your CUDA first (see requirements.txt for why)
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121

# 2. Then the rest
pip install -r requirements.txt
```

Check that paths resolve correctly:

```bash
python -m cabinet_bnn.paths
```

Resource locations can be overridden by environment variables — see
[`cabinet_bnn/paths.py`](cabinet_bnn/paths.py) for the full list:

```
CABINET_BGE_MODEL      bge model directory
CABINET_QWEN_MODEL     Qwen model directory (for comparison)
CABINET_VOCAB          3109-word vocabulary path
CABINET_DATA_DIR       data directory
CABINET_RESULTS_DIR    results directory
```

> `huggingface.co` is unreachable in some regions. Use
> `export HF_ENDPOINT=https://hf-mirror.com`.

### Reproduce

```bash
# --- Stage 1: HSH-64 code reproduction + bucket-structure validation ---
python scripts/01_gen_embeddings.py          # bge-large embeddings
python scripts/02_mvp_hsh64.py --epochs 500 --hidden-dim 512 --post-iters 10

# --- Stage 2: BNN ablations ---
python scripts/03_bnn_ablation.py --preset shared token_embed code
python scripts/04_continual_eval.py --n-domains 5 --n-steps 5

# --- Stage 3: language-model pretraining ---
python scripts/07_prepare_data.py --vocab-size 32000
python scripts/08_train_lm.py --preset base --max-steps 49900

# --- Evaluation & figures ---
python scripts/10_eval_lm.py  --ckpt results/lm_base_1ep/final.pt
python scripts/12_generate.py --ckpt results/lm_base_1ep/final.pt
python scripts/13_compare_ppl.py
python scripts/05_make_figures.py
```

### Correctness tests

Four regression suites, all passing. They exist because each one caught a bug that
**silently produced wrong conclusions** rather than crashing:

```bash
python scripts/selftest_post_optimize.py   # greedy bit-flip: Δ formula vs measured ΔO
python scripts/selftest_attention.py       # causal mask leakage, KV-cache equivalence
python scripts/selftest_code_ffn.py        # einsum equivalence, batch indexing, geometry
python tools/verify_paper_math.py          # HSH-64 paper propositions, numerically
```

### Results

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
[`docs/EVALUATION_REPORT.md`](docs/EVALUATION_REPORT.md).

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

### Research direction

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

### Current status

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
| MoE-style sparse activation | ⬜ planned |
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

### Known issues

- The `41.6 M` model was trained on a corpus with **77 % instruction data**, which is too
  high for pretraining and causes format overfitting (empty markdown tables, repeated
  lists). Correct ratio: general text 70–80 %, instruction data 10–20 % — reserve
  instruction data for SFT.
- **No README-level usage example for the BNN path yet**, because the mechanism is not
  wired into the LM.
- The bidirectional relation gate **is implemented but should not be enabled** — see the
  negative result above. `relation_gate=False` remains the recommended default.

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
