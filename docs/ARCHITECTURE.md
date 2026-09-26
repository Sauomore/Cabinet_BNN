# Architecture & Reproduction

[← Back to README](../README.md)

Repository layout, installation, and the full pipeline for reproducing every
number in [RESULTS.md](RESULTS.md).

## Repository layout

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

## Install

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
[`cabinet_bnn/paths.py`](../cabinet_bnn/paths.py) for the full list:

```
CABINET_BGE_MODEL      bge model directory
CABINET_QWEN_MODEL     Qwen model directory (for comparison)
CABINET_VOCAB          3109-word vocabulary path
CABINET_DATA_DIR       data directory
CABINET_RESULTS_DIR    results directory
```

> `huggingface.co` is unreachable in some regions. Use
> `export HF_ENDPOINT=https://hf-mirror.com`.

## Reproduce

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
