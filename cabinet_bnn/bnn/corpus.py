# -*- coding: utf-8 -*-
"""
由 vocab_3109 构造字符级语言建模语料。

为什么用字符级（而不是 subword）：
  · 与 HSH-64 的语义单元完全对齐 —— 每个"词"就是一个训练样本
  · 无 tokenizer 依赖，可复现
  · 训练极快，适合机制验证

任务：给定前 L 个字符，预测下一个字符（next-char prediction）。
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PAD, UNK, BOS, EOS = 0, 1, 2, 3


@dataclass
class CharCorpus:
    chars: list[str]                 # 索引 → 字符（含 4 个特殊符号）
    stoi: dict[str, int]             # 字符 → 索引
    words: list[str]                 # 原始词表
    sequences: np.ndarray            # (N, L+1) int32，[BOS] + word[:L]
    seq_lens: np.ndarray             # (N,) 每条真实长度（含 BOS）
    vocab_size: int

    def stats(self) -> str:
        n, L = self.sequences.shape
        return (f"  词数 {n}，序列长 {L}（含 BOS）\n"
                f"  字符表大小 {self.vocab_size}（含 PAD/UNK/BOS/EOS）\n"
                f"  平均词长 {self.seq_lens.mean():.2f}")


def build_char_corpus(
    vocab_path: str | Path,
    min_count: int = 3,
    max_chars: int = 512,
    max_len: int = 8,
) -> CharCorpus:
    """从词表构造字符级语料。

    Args:
        vocab_path: 每行一个词的词表
        min_count:  最低字符频次（低于此归 UNK）
        max_chars:  字符表上限（取最高频的若干个）
        max_len:    单条序列最大长度（超出截断）
    """
    path = Path(vocab_path)
    words = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not words:
        raise ValueError(f"词表为空: {path}")

    # 字符频率统计
    counter = collections.Counter("".join(words))
    kept = [c for c, k in counter.most_common(max_chars) if k >= min_count]
    if not kept:
        raise ValueError("没有字符满足 min_count")

    chars = ["<pad>", "<unk>", "<bos>", "<eos>"] + kept
    stoi = {c: i for i, c in enumerate(chars)}
    v = len(chars)

    # 构造序列：BOS + word 的字符（截断到 max_len-1），右侧补 PAD
    L = min(max_len, max(len(w) for w in words) + 1)
    seqs = np.full((len(words), L), PAD, dtype=np.int32)
    lens = np.zeros(len(words), dtype=np.int32)
    for i, w in enumerate(words):
        ids = [BOS] + [stoi.get(c, UNK) for c in w[: L - 1]]
        seqs[i, : len(ids)] = ids
        lens[i] = len(ids)

    return CharCorpus(
        chars=chars, stoi=stoi, words=words,
        sequences=seqs, seq_lens=lens, vocab_size=v,
    )


def train_val_split(corpus: CharCorpus, val_ratio: float = 0.1, seed: int = 42):
    """按词切分训练/验证集（不是按字符，避免同词泄漏）。"""
    rng = np.random.default_rng(seed)
    n = len(corpus.words)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_ratio))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return train_idx, val_idx


def make_batches(seqs: np.ndarray, idx: np.ndarray, batch_size: int, shuffle: bool = True,
                 seed: int = 0):
    """把序列切成 (输入, 目标) 批次。

    输入 = seq[:-1]，目标 = seq[1:]（next-char prediction），PAD 位置不计损失。
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(idx)) if shuffle else np.arange(len(idx))
    idx = idx[order]
    for s in range(0, len(idx), batch_size):
        chunk = idx[s : s + batch_size]
        full = seqs[chunk]                      # (B, L)
        x = full[:, :-1]                        # (B, L-1)
        y = full[:, 1:]                         # (B, L-1)
        # PAD 目标置 -100（CrossEntropyLoss 的 ignore_index）
        y = np.where(y == PAD, -100, y)
        yield x.astype(np.int64), y.astype(np.int64)
