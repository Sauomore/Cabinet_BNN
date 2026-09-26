# 架构与复现

[← 返回 README](../README.zh.md)

仓库布局、安装步骤，以及复现 [RESULTS.zh.md](RESULTS.zh.md) 中每个数字的完整流程。

## 安装

```bash
# 1. 先按你的 CUDA 版本装 PyTorch（原因见 requirements.txt 注释）
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121

# 2. 再装其余依赖
pip install -r requirements.txt
```

检查路径解析是否正确：

```bash
python -m cabinet_bnn.paths
```

外部资源位置可用环境变量覆盖（完整列表见
[`cabinet_bnn/paths.py`](../cabinet_bnn/paths.py)）：

```
CABINET_BGE_MODEL      bge 模型目录
CABINET_QWEN_MODEL     Qwen 模型目录（对比用）
CABINET_VOCAB          3109 词表路径
CABINET_DATA_DIR       数据目录
CABINET_RESULTS_DIR    结果目录
```

> 国内访问 `huggingface.co` 不通时，用 `set HF_ENDPOINT=https://hf-mirror.com`。

## 复现

```bash
# --- 第一阶段：HSH-64 码复现 + 桶结构验证 ---
python scripts/01_gen_embeddings.py
python scripts/02_mvp_hsh64.py --epochs 500 --hidden-dim 512 --post-iters 10

# --- 第二阶段：BNN 消融实验 ---
python scripts/03_bnn_ablation.py --preset shared token_embed code
python scripts/04_continual_eval.py --n-domains 5 --n-steps 5

# --- 第三阶段：语言模型预训练 ---
python scripts/07_prepare_data.py --vocab-size 32000
python scripts/08_train_lm.py --preset base --max-steps 49900

# --- 评测与绘图 ---
python scripts/10_eval_lm.py  --ckpt results/lm_base_1ep/final.pt
python scripts/12_generate.py --ckpt results/lm_base_1ep/final.pt
python scripts/13_compare_ppl.py
python scripts/05_make_figures.py
```
