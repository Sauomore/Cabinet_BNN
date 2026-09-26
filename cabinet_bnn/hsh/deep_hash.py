# -*- coding: utf-8 -*-
"""
Deep Hash v3（HSH-64 版本）：MLP + STE + 多目标损失。

与参考实现 scripts/train_deep_hash_v3_64.py 在以下方面保持严格一致：
  · 网络结构： x -> Linear -> BatchNorm1d -> ReLU -> Linear -> u
  · STE：      ste_sign(x) = x + (sign(x) - x).detach()
  · 损失：     L = L_info + λ_q·L_quant + λ_b·L_balance + λ_p·L_pair
  · 目标相似度：由【教师】向量中心化后归一化计算（支持蒸馏）
  · 量化判定：  bit = 1 ⟺ u >= 0（含零）
  · 导出格式：  魔数 0xCAB1_DE3D，全 big-endian，与 Rust DeepHashProjection 对齐

改进（不改变语义，仅提速）：
  · 导出用 numpy 批量打包，替代逐元素 struct.pack（原版对 h512 会写 100 万次 struct.pack）
  · 支持 device 参数（参考实现硬编码 CPU）
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAGIC = 0xCAB1_DE3D
VERSION_V1 = 1
VERSION_V2 = 2


def ste_sign(x: torch.Tensor) -> torch.Tensor:
    """Straight-Through Estimator for sign。前向 sign，反向恒等。"""
    return x + (torch.sign(x) - x).detach()


class DeepHashMLP(nn.Module):
    """dim -> hidden -> n_bits 的单隐层 MLP（v1 格式）。"""

    def __init__(self, dim: int, hidden_dim: int, n_bits: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_bits, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.fc1(x)))
        return self.fc2(h)


# ---------------------------------------------------------------- 邻域掩码

def build_neighborhood_masks(x: np.ndarray, pos_k: int = 10, neg_k: int = 50):
    """基于余弦相似度构建正/负样本掩码（教师空间）。"""
    xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    sim = xn @ xn.T
    np.fill_diagonal(sim, -2.0)
    n = x.shape[0]

    pos_mask = np.zeros((n, n), dtype=bool)
    neg_mask = np.zeros((n, n), dtype=bool)
    pos_idx = np.argsort(-sim, axis=1)[:, :pos_k]
    neg_idx = np.argsort(sim, axis=1)[:, :neg_k]
    rows = np.arange(n)[:, None]
    pos_mask[rows, pos_idx] = True
    neg_mask[rows, neg_idx] = True
    return pos_mask, neg_mask


def build_hard_negatives(z: torch.Tensor, pos_mask: torch.Tensor, neg_k: int) -> torch.Tensor:
    """在【当前连续投影空间】里挖最难的负样本（排除正样本与自身）。"""
    z_norm = F.normalize(z, dim=1)
    sim = z_norm @ z_norm.T
    n = sim.shape[0]
    sim = sim.masked_fill(torch.eye(n, device=sim.device, dtype=torch.bool), -1e9)
    neg_sim = sim.masked_fill(pos_mask, -1e9)
    k = min(neg_k, n - 1)
    _, neg_idx = torch.topk(neg_sim, k=k, dim=1)
    neg_mask = torch.zeros_like(pos_mask)
    neg_mask.scatter_(1, neg_idx, True)
    return neg_mask


# ---------------------------------------------------------------- 损失

def info_nce_loss(z, pos_mask, neg_mask, temperature: float = 0.07) -> torch.Tensor:
    """InfoNCE：在 pos ∪ neg 候选集上做 softmax，正样本为目标。"""
    z_norm = F.normalize(z, dim=1)
    sim = z_norm @ z_norm.T / temperature
    n = sim.shape[0]
    sim = sim.masked_fill(torch.eye(n, device=sim.device, dtype=torch.bool), -1e9)

    pos_logsum = torch.logsumexp(sim.masked_fill(~pos_mask, -1e9), dim=1)
    neg_logsum = torch.logsumexp(sim.masked_fill(~neg_mask, -1e9), dim=1)
    denom = torch.logsumexp(torch.stack([pos_logsum, neg_logsum], dim=1), dim=1)
    loss = -pos_logsum + denom

    valid = pos_mask.sum(dim=1) > 0
    return loss[valid].mean()


def pairwise_mse_loss(b: torch.Tensor, target_sim: torch.Tensor) -> torch.Tensor:
    """让二进制码归一化内积逼近教师余弦相似度。"""
    pred = b @ b.T / b.shape[1]
    return F.mse_loss(pred, target_sim)


# ---------------------------------------------------------------- 训练

@dataclass
class TrainHistory:
    total: list[float] = field(default_factory=list)
    info: list[float] = field(default_factory=list)
    quant: list[float] = field(default_factory=list)
    balance: list[float] = field(default_factory=list)
    pair: list[float] = field(default_factory=list)


def train_deep_hash(
    x: np.ndarray,
    n_bits: int = 52,
    hidden_dim: int = 512,
    epochs: int = 500,
    lr: float = 1e-3,
    pos_k: int = 10,
    neg_k: int = 50,
    temperature: float = 0.07,
    lambda_quant: float = 1.0,
    lambda_balance: float = 0.5,
    lambda_pair: float = 0.0,
    seed: int = 42,
    teacher_x: np.ndarray | None = None,
    device: str = "cpu",
    log_every: int = 100,
    verbose: bool = True,
) -> tuple[np.ndarray, DeepHashMLP, TrainHistory]:
    """训练 Deep Hash MLP。

    学生输入用 x（中心化后），量化目标相似度与邻域掩码用 teacher_x（若给出）。
    返回 (mean, model, history)；mean 是学生输入的中心化均值，导出时必须一起写。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    n, dim = x.shape
    dev = torch.device(device)

    mean = x.mean(axis=0)
    xc = (x - mean).astype(np.float32)
    x_tensor = torch.from_numpy(xc).to(dev)

    # 教师空间决定：邻域掩码 + 目标相似度
    src = teacher_x if teacher_x is not None else x
    src_mean = src.mean(axis=0)
    src_c = src - src_mean
    src_n = src_c / (np.linalg.norm(src_c, axis=1, keepdims=True) + 1e-8)
    target_sim = torch.from_numpy((src_n @ src_n.T).astype(np.float32)).to(dev)

    pos_np, _ = build_neighborhood_masks(src, pos_k, neg_k)
    pos_mask = torch.from_numpy(pos_np).to(dev)

    model = DeepHashMLP(dim, hidden_dim, n_bits).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = TrainHistory()
    model.train()

    for epoch in range(epochs):
        opt.zero_grad()
        u = model(x_tensor)
        b = ste_sign(u)

        neg_mask = build_hard_negatives(u.detach(), pos_mask, neg_k)

        l_info = info_nce_loss(u, pos_mask, neg_mask, temperature)
        l_quant = (u - b).abs().mean()
        bit_mean = (b + 1.0).mean(dim=0) / 2.0
        l_balance = ((bit_mean - 0.5) ** 2).mean()
        l_pair = pairwise_mse_loss(b, target_sim)

        # λ_q 退火：前 50% epoch 从 1.0 线性到 lambda_quant（默认目标也是 1.0，故通常恒定）
        cur_lq = 1.0 + (lambda_quant - 1.0) * min(epoch / max(epochs * 0.5, 1.0), 1.0)

        loss = l_info + cur_lq * l_quant + lambda_balance * l_balance + lambda_pair * l_pair
        loss.backward()
        opt.step()

        hist.total.append(float(loss.item()))
        hist.info.append(float(l_info.item()))
        hist.quant.append(float(l_quant.item()))
        hist.balance.append(float(l_balance.item()))
        hist.pair.append(float(l_pair.item()))

        if verbose and ((epoch + 1) % log_every == 0 or epoch == 0):
            print(
                f"  [epoch {epoch + 1:04d}] total={loss.item():.4f} "
                f"info={l_info.item():.4f} quant={l_quant.item():.4f} "
                f"balance={l_balance.item():.4f}",
                flush=True,
            )

    return mean, model, hist


# ---------------------------------------------------------------- 导出 / 加载

def export_deep_hash(path: str | Path, mean: np.ndarray, model: DeepHashMLP) -> Path:
    """导出为 Rust DeepHashProjection 可读的 v1 二进制（全 big-endian）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        sd = model.state_dict()
        w1 = sd["fc1.weight"].cpu().numpy().astype(np.float32)      # (hidden, dim)
        b1 = sd["fc1.bias"].cpu().numpy().astype(np.float32)
        gamma = sd["bn1.weight"].cpu().numpy().astype(np.float32)
        beta = sd["bn1.bias"].cpu().numpy().astype(np.float32)
        rmean = sd["bn1.running_mean"].cpu().numpy().astype(np.float32)
        rvar = sd["bn1.running_var"].cpu().numpy().astype(np.float32)
        w2 = sd["fc2.weight"].cpu().numpy().astype(np.float32)      # (n_bits, hidden)
        b2 = sd["fc2.bias"].cpu().numpy().astype(np.float32)

    dim = int(w1.shape[1])
    hidden = int(w1.shape[0])
    n_bits = int(w2.shape[0])

    # 一次性拼装并转 big-endian（比逐元素 struct.pack 快两个数量级，保证字节序完全一致）
    blob = np.concatenate([
        mean.astype(np.float32).ravel(),
        w1.ravel(), b1.ravel(), gamma.ravel(), beta.ravel(), rmean.ravel(), rvar.ravel(),
        w2.ravel(), b2.ravel(),
    ]).astype(">f4")

    header = struct.pack(">IIIII", MAGIC, VERSION_V1, dim, hidden, n_bits)
    with path.open("wb") as f:
        f.write(header)
        f.write(blob.tobytes())
    return path


@dataclass
class LoadedDeepHash:
    mean: np.ndarray
    model: DeepHashMLP
    dim: int
    hidden_dim: int
    n_bits: int
    version: int


def load_deep_hash(path: str | Path, device: str = "cpu") -> LoadedDeepHash:
    """加载 DeepHash 二进制（支持 v1/v2），校验字段长度。"""
    path = Path(path)
    data = path.read_bytes()
    if len(data) < 20:
        raise ValueError(f"文件过短: {len(data)} 字节")

    magic = struct.unpack_from(">I", data, 0)[0]
    version = struct.unpack_from(">I", data, 4)[0]
    if magic != MAGIC:
        raise ValueError(f"magic 不匹配：0x{magic:08X}")
    if version not in (VERSION_V1, VERSION_V2):
        raise ValueError(f"version 不支持：{version}")

    dim = struct.unpack_from(">I", data, 8)[0]
    hidden = struct.unpack_from(">I", data, 12)[0]
    if version == VERSION_V1:
        depth, n_bits, header = 1, struct.unpack_from(">I", data, 16)[0], 20
    else:
        depth = struct.unpack_from(">I", data, 16)[0]
        n_bits = struct.unpack_from(">I", data, 20)[0]
        header = 24
    if depth != 1:
        raise ValueError(f"本实现仅支持 depth=1（v1 结构），实际 depth={depth}")

    expect = header + dim * 4
    expect += hidden * dim * 4 + hidden * 4 * 5
    expect += n_bits * hidden * 4 + n_bits * 4
    if len(data) < expect:
        raise ValueError(f"文件截断：期望至少 {expect} 字节，实际 {len(data)}")

    arr = np.frombuffer(data, dtype=">f4", count=(len(data) - header) // 4, offset=header)
    arr = arr.astype(np.float32)
    p = 0

    def take(k: int) -> np.ndarray:
        nonlocal p
        out = arr[p : p + k]
        p += k
        return out

    mean = take(dim).copy()
    w1 = take(hidden * dim).reshape(hidden, dim).copy()
    b1 = take(hidden).copy()
    gamma = take(hidden).copy()
    beta = take(hidden).copy()
    rmean = take(hidden).copy()
    rvar = take(hidden).copy()
    w2 = take(n_bits * hidden).reshape(n_bits, hidden).copy()
    b2 = take(n_bits).copy()

    model = DeepHashMLP(dim, hidden, n_bits)
    model.load_state_dict({
        "fc1.weight": torch.from_numpy(w1),
        "fc1.bias": torch.from_numpy(b1),
        "bn1.weight": torch.from_numpy(gamma),
        "bn1.bias": torch.from_numpy(beta),
        "bn1.running_mean": torch.from_numpy(rmean),
        "bn1.running_var": torch.from_numpy(rvar),
        "fc2.weight": torch.from_numpy(w2),
        "fc2.bias": torch.from_numpy(b2),
    })
    model.to(device).eval()
    return LoadedDeepHash(mean, model, dim, hidden, n_bits, version)


# ---------------------------------------------------------------- 推理

@torch.no_grad()
def project(model: DeepHashMLP, mean: np.ndarray, x: np.ndarray, device: str = "cpu",
            batch: int = 4096) -> np.ndarray:
    """连续投影 u = model(x - mean)。"""
    model.eval()
    xc = (x - mean).astype(np.float32)
    outs = []
    for s in range(0, xc.shape[0], batch):
        t = torch.from_numpy(xc[s : s + batch]).to(device)
        outs.append(model(t).cpu().numpy())
    return np.vstack(outs) if outs else np.zeros((0, model.fc2.out_features), dtype=np.float32)


def quantize_bits(u: np.ndarray) -> np.ndarray:
    """连续 logits -> 位（1 ⟺ u >= 0，与 Rust 一致）。"""
    return (u >= 0.0).astype(np.uint8)


def bits_to_u64(bits: np.ndarray) -> np.ndarray:
    """(N, n_bits) 的 0/1 位矩阵打包成 u64 数组（第 i 位对应 bit i）。"""
    bits = np.asarray(bits, dtype=np.uint64)
    n, m = bits.shape
    if m > 64:
        raise ValueError(f"位宽 {m} > 64，无法装入单个 u64")
    weights = (np.uint64(1) << np.arange(m, dtype=np.uint64))
    return (bits * weights[None, :]).sum(axis=1, dtype=np.uint64)
