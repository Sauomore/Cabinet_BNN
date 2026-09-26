# Cabinet-BNN

[English](README.md) ｜ **中文**

> 一个二值激活语言模型，其 per-token 权重由 **64 位结构化语义哈希码**生成，
> 而不是从全局权重张量里查出来。

### 这是什么

本仓库探索一个问题：

> **一个 token 的权重矩阵，能不能由一段紧凑的离散码「生成」出来，而不是从全局权重张量里查出来？**

具体地，每个 token `t` 有一个 **128 位权重码**：

```
hash 段 (64 bit)  =  feat(4) + sim(52) + abs(8)      ← HSH-64 的检索码
param 段 (64 bit) =  per-token 信息码                 ← 本项目引入
```

权重由共享基矩阵组合而成：

```
s_j(t) = mask_j(hash_t) · p_t[j]                 (+1 / −1)
W_t    = Σ_j s_j(t) · (U_j V_jᵀ)                 共享低秩基
y      = sign(W_t x) · scale                     二值激活 + STE
```

`hash` 段来自 [HSH-64](https://github.com/Sauomore/Cabinet_hsh64)（作者前作）：把嵌入映射为
64 位码，**Hamming 距离 ≈ 语义距离**，一次 `popcnt` 即可查询。

### 为什么做

两个动机，**必须分开看**：

| 目标 | 状态 |
|---|---|
| **降成本** —— 二值激活、低秩生成权重 | 已有实测数据，见下 |
| **参数可编辑** —— 翻一位即改权重；每个领域各存一份码表 | 小规模已验证 |

> ⚠️ **二值化 ≠ 像大脑。** 二值化买的是**成本**，不是**智能**。
> 对持续学习真正重要的是更新的**局部性** —— 见下方「研究方向」。

### 安装

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
[`cabinet_bnn/paths.py`](cabinet_bnn/paths.py)）：

```
CABINET_BGE_MODEL      bge 模型目录
CABINET_QWEN_MODEL     Qwen 模型目录（对比用）
CABINET_VOCAB          3109 词表路径
CABINET_DATA_DIR       数据目录
CABINET_RESULTS_DIR    结果目录
```

> 国内访问 `huggingface.co` 不通时，用 `set HF_ENDPOINT=https://hf-mirror.com`。

### 复现

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

### 正确性测试

四套回归测试，全部通过。它们的存在是因为**每一个都抓到过「不崩溃但结论错」的 bug**：

```bash
python scripts/selftest_post_optimize.py   # 贪心翻转：Δ 公式 vs 实测 ΔO
python scripts/selftest_attention.py       # 因果掩码泄漏、KV cache 等价性
python scripts/selftest_code_ffn.py        # einsum 等价性、批量索引、几何
python tools/verify_paper_math.py          # HSH-64 论文命题数值核验
```

### 实验结果

#### HSH-64 码复现（3,109 词，纯 Hamming）

| | Recall@10 | 正样本对距离 | 分离度 |
|---|---|---|---|
| 后处理前 | 0.4988 | 19.28 bit | 11.45 bit |
| **后处理后** | **0.7426** | **9.17 bit** | **16.91 bit** |
| 论文 (h512) | 0.7404 | ≈8 | ≈18 |

桶结构：**3,109 个词落在 3,075 个语义桶，最大桶 3**（`abs` 槽位上限 256）
→ 准完美哈希有大量余量。

> ⚠️ 配置偏差：本机缺 `bge-small` 权重，学生输入是 `bge-large` 的 PCA-512 截断
> —— 而教师也是 `bge-large`。这**对指标有利**，故不能与论文数字直接并列。

#### BNN 消融 —— **负面结果，如实报告**

完整报告见 [`docs/EVALUATION_REPORT.md`](docs/EVALUATION_REPORT.md)。

**（a）等参数预算下，码生成权重能否胜过普通全局权重？**

| 模式 | 参数量 | val_loss | tok_acc |
|---|---|---|---|
| `shared`（全局权重） | 524,629 | **3.973** | **0.3094** |
| `code`（码生成） | 452,928 | 4.162 | 0.2587 |
| `token_embed`（经典 per-token） | 510,357 | 4.783 | 0.2252 |

→ `code` **输给** `shared`，但**胜过** `token_embed`。
**结论：参数量不是瓶颈时，per-token 参数化没有收益；它的价值在「可编辑性」而非「表达力」。**

**（b）每域独立码表能否避免遗忘？**

| 方法 | 本域精度 | 遗忘量 | 回滚 |
|---|---|---|---|
| `shared_ft`（顺序微调） | 0.2875 | 0.0301 | — |
| `code_param`（只改码位） | 0.2862 | 0.0543 | 不精确 |
| **`code_swap`（每域独立码表）** | **0.3278** | **0.0000** | **精确** |
| `code_full` | 0.2591 | 0.0678 | 不精确 |
| `shared_joint`（联合训练上界） | 0.3264 | 0（非增量） | — |

→ 零遗忘是**结构性保证**（每域状态物理隔离），回滚位级精确，本域精度略超联合训练上界。

> ⚠️ **声明边界。** 这里的零遗忘来自**物理隔离**，与 PackNet / HAT / PNN 同一机制，
> 是**用容量换来的**（每域 250 KB），**不是新机制**。贡献在于以更低代价实现，
> 并附带位级可逆性 —— **而不是「解决了灾难性遗忘」**。

#### 语言模型

| 模型 | 参数量 | 训练 tokens | val_loss | ppl |
|---|---|---|---|---|
| 字符级（基线） | 4.5 M | 111 M | 5.06 | 157.5 |
| 字符级 | 11.3 M | 111 M | 4.48 | 88.1 |
| **`base` (d=512, 8 层, 32k BPE)** | **41.6 M** | **817 M** | **3.011** | **20.3** |

跨模型对比（同一段中文，用 **bpc/每字符比特** —— 不同 tokenizer 之间唯一公平的口径）：

| 模型 | 参数量 | 词表 | **bpc ↓** |
|---|---|---|---|
| 本仓库 (41.6 M) | 41.6 M | 32k | **4.720** |
| Qwen2.5-0.5B | 494 M | 152k | **3.418** |

→ 参数少 **12 倍**、数据少约 **20000 倍**，bpc 只差 38%。

> 注意：这个 `41.6 M` 模型是**标准 Transformer**（`ffn_mode=global`），
> **尚未启用** BNN 权重码机制。它存在的意义是提供一个可信的对照基座。

### 研究方向

促使当前工作的实测事实：

> **独立随机化的** `param` 段会**破坏**权重几何：
> `corr(hash 的 Hamming 距离, 权重余弦相似度) = +0.0019`（即完全无关）。
> 只用语义 `mask` 时为 `−0.9885`（近乎完美单调）。

解释：同一个 `k` 维 sign 向量**无法**同时承载**几何**与**独立自由度**。

正在探索的方向是 **码作为索引**：

```
码（hash）    →  索引   →  几何由寻址天然保证（局部性）
信息码        →  内容   →  可以承载任何东西，无需遵守几何约束
```

以及 **MoE 式稀疏激活**：让「编辑某个 token 的信息码」只影响它真正激活的那 `k'` 个基矩阵
—— 即**稀疏写**，而不只是稀疏读。

**开放问题**（作者认为「优雅解」就在这里）：

> 能否学出一组共享基矩阵，使得「编辑任意一个码」对其它所有 token 输出的影响**有可界定的上界**？
>
> 若能，这将是稳定性–可塑性困境在**参数化层面**的表述 ——
> 而局部性正是生物突触（STDP）无需全局梯度即可持续学习的机制。

### 当前状态

| 组件 | 状态 |
|---|---|
| HSH-64 编解码 + 后处理 | ✅ 已对论文验证 |
| 桶结构验证 | ✅ 完成（最大桶 3 / 上限 256） |
| 因果注意力（RoPE, KV cache） | ✅ 已实现，有回归测试 |
| 码生成 FFN（低秩、省显存） | ✅ 已实现，有回归测试 |
| 持续学习评测框架 | ✅ 完成 |
| **BNN 机制接入已训练的语言模型** | ⬜ **进行中** |
| MoE 式稀疏激活 | ⬜ 计划中 |
| 双向注意力关系门 | ✅ 已实现、有回归测试 —— ❌ **但实测有害**（见下） |
| 码作为索引的几何传导 | ✅ 已验证（无需训练） |
| 新 token 热添加 | ⬜ 计划中 |
| SFT 出对话能力 | ⬜ 计划中 |

#### 负面结果：关系门有害

per-token 关系向量 `u_t` 用于对注意力 logits 加门：
`att_ij = softmax_j( (q_i·k_j)/√D · (1 + γ·⟨u_i,u_j⟩/√r) )`。
**实现是正确的**（γ=0 逐位恒等、对称、KV cache 等价到 4e-7、γ 能在训练中离开 0），
**但它让语言模型变差了。**

从同一 checkpoint 出发的受控实验 —— seed、lr 调度、数据顺序、步数全部相同，
唯一差异是门是否生效：

| 配置 | val_loss | ppl |
|---|---|---|
| base（起点） | 3.0111 | 20.3 |
| 门**冻结**为 γ=0 | **2.9668** | **19.4** |
| 门**可训练** | 2.9863 | 19.8 |

门开启时差 0.0194，且该差值在全部 5 个评测点上稳定（+0.016 … +0.020），不是噪声。

三个实测原因：

1. **门的信号太弱。** 偏差量级 `|γ|·std(s) ≈ 0.006`，而注意力 logits 的 std 是 0.33
   —— 仅 1.8% 的扰动。它增加了 4.1 M 个可训练参数，却几乎不起作用。
2. **关系向量塌缩。** 32000 个 token 挤在 `r = 16` 维里，所有余弦相似度落在
   0.88–0.92 窄带，最强「关系对」毫无意义（`葡萄 ↔ 措施`）。
3. **没有学到共现结构。** `corr(PMI, 关系分数) = −0.0021`（156 万 token 对）。
   模型选择了不使用这个机制。

> **一条必须记录的方法论教训。** 最初观察到的是 `3.0111 → 2.9863`，看起来是改善。
> 其实不是：没有对照组，就无法把门的效应和「多训了 6000 步」的效应分开，
> 而全部表观收益都来自后者。**对照是必需的，最初的读数是错的。**

存活下来的是**门背后的想法**而不是门本身：`scripts/17` 显示，
把 `u_t` **由码导出**能在**完全不训练**的情况下把码几何传导到关系几何
（corr −0.6969，簇一致率 98.4%，随机基线 6.2%）。
失败的是**这个参数化方式**（窄标量门），不是「码作为索引」。

### 已知问题

- 该 `41.6 M` 模型的语料中**指令数据占 77%**，对预训练而言过高，导致格式过拟合
  （空 Markdown 表格、重复列表）。正确配比应为：通用文本 70–80%、指令数据 10–20%，
  指令数据留给 SFT 阶段。
- **BNN 路径尚无 README 级用法示例**，因为机制还没接进语言模型。
- 双向关系门**已实现但不建议启用** —— 见上方负面结果。默认仍为 `relation_gate=False`。

### 致谢

基于同一作者的 **HSH-64** ——
[github.com/Sauomore/Cabinet_hsh64](https://github.com/Sauomore/Cabinet_hsh64)。
结构化位布局（`feat4 + sim52 + abs8`）、Deep Hash 投影头、召回导向贪心比特翻转算法
与编解码语义均来自该工作；本仓库把码从「只读检索键」扩展为「可写的 per-token 参数载体」。

贪心比特翻转优化与
[Greedy Hash (NeurIPS 2018)](https://proceedings.neurips.cc/paper_files/paper/2018/file/13f3cf8c531952d72e5847c4183e6910-Paper.pdf)
密切相关。

### 许可

MIT —— 见 [LICENSE](LICENSE)。
