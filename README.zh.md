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


### 一览

| 组件 | 状态 |
|---|---|
| HSH-64 编解码 + 贪心比特翻转后处理 | ✅ 已对论文验证（Recall@10 0.7426） |
| 因果注意力 + RoPE + KV cache | ✅ 有回归测试 |
| 码生成 FFN（低秩、省显存） | ✅ 有回归测试 |
| 双向注意力关系门 | ✅ 已实现 —— ❌ **实测有害** |
| MoE 式稀疏 FFN（码驱动路由） | ✅ 已实现并通过测试 —— ⬜ 公平对比待做 |
| 码作为索引的几何传导 | ✅ 已验证（无需训练） |
| 语言模型（41.6 M） | ✅ val_loss 3.0111 / ppl 20.3 |
| BNN 机制接入已训练的语言模型 | ⬜ **进行中** |

**文档**

| | |
|---|---|
| **[实验结果与发现](docs/RESULTS.zh.md)** | 全部实测数字，含负面结果 |
| **[设计原理与研究方向](docs/DESIGN.zh.md)** | 想法、约束、开放问题 |
| **[架构与复现](docs/ARCHITECTURE.zh.md)** | 布局、安装、完整流程 |
| **[测试、状态与已知问题](docs/TESTING.zh.md)** | 测试套件、做到了什么、哪里没做到 |

> ⚠️ **本项目【不】声称的两件事**，均有实测支持：
> 双向关系门让语言模型**变差**；每域码表的零遗忘来自**物理隔离**（用容量换的），
> 不是新机制。受控对比见 [RESULTS.zh.md](docs/RESULTS.zh.md)。

### 为什么做

两个动机，**必须分开看**：

| 目标 | 状态 |
|---|---|
| **降成本** —— 二值激活、低秩生成权重 | 已有实测数据，见 [RESULTS.zh.md](docs/RESULTS.zh.md) |
| **参数可编辑** —— 翻一位即改权重；每个领域各存一份码表 | 小规模已验证 |

> ⚠️ **二值化 ≠ 像大脑。** 二值化买的是**成本**，不是**智能**。
> 对持续学习真正重要的是更新的**局部性** —— 见 [设计原理与研究方向](docs/DESIGN.zh.md)。

### 快速开始

```bash
git clone https://github.com/Sauomore/Cabinet_BNN.git
cd Cabinet_BNN
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

python -m cabinet_bnn.paths                      # 检查路径解析
python scripts/selftest_attention.py             # 4 套回归测试，应全部通过
python scripts/selftest_moe_ffn.py
```

完整流程见 [架构与复现](docs/ARCHITECTURE.zh.md)。

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
