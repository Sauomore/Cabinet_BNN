# -*- coding: utf-8 -*-
"""
HSH-64 embedding 缓存读写。

与参考实现 generators 逐字节兼容：
    [u32 BE: magic = 0xCAB1_EBED]
    [u32 BE: version = 1]
    [u32 BE: dim]
    [u32 BE: vocab_size]
    for each word:
        [u16 BE: len(word_bytes)]
        [bytes:  UTF-8 word]
        [f32 BE × dim: vector]

设计说明：所有标量一律 big-endian，与 Rust 端 u32::from_be_bytes / f32::from_be_bytes 对齐。
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

EMBEDDING_MAGIC = 0xCAB1_EBED
EMBEDDING_VERSION = 1


def write_cache(path: str | Path, dim: int, items: Iterable[tuple[str, np.ndarray]]) -> Path:
    """写入 embedding 缓存。

    Args:
        path: 目标路径
        dim: 向量维度
        items: (word, vector) 可迭代对象，vector 长度须为 dim
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    items = list(items)

    with path.open("wb") as f:
        f.write(struct.pack(">I", EMBEDDING_MAGIC))
        f.write(struct.pack(">I", EMBEDDING_VERSION))
        f.write(struct.pack(">I", dim))
        f.write(struct.pack(">I", len(items)))
        for word, vec in items:
            wb = word.encode("utf-8")
            if len(wb) > 0xFFFF:
                raise ValueError(f"单词过长（{len(wb)} 字节）: {word[:32]}...")
            if len(vec) != dim:
                raise ValueError(f"向量维度不符：期望 {dim}，实际 {len(vec)}（词: {word}）")
            f.write(struct.pack(">H", len(wb)))
            f.write(wb)
            # 一次性打包整行，比逐元素 pack 快得多
            f.write(np.asarray(vec, dtype=">f4").tobytes())
    return path


def read_cache(path: str | Path) -> tuple[int, list[str], np.ndarray]:
    """读取 embedding 缓存，返回 (dim, words, matrix)。

    matrix 形状 (N, dim)，dtype float32，native 字节序。
    """
    path = Path(path)
    data = path.read_bytes()
    if len(data) < 16:
        raise ValueError(f"缓存文件过短（{len(data)} 字节）: {path}")

    magic = struct.unpack_from(">I", data, 0)[0]
    version = struct.unpack_from(">I", data, 4)[0]
    dim = struct.unpack_from(">I", data, 8)[0]
    count = struct.unpack_from(">I", data, 12)[0]

    if magic != EMBEDDING_MAGIC:
        raise ValueError(f"magic 不匹配：期望 0x{EMBEDDING_MAGIC:08X}，实际 0x{magic:08X}")
    if version != EMBEDDING_VERSION:
        raise ValueError(f"version 不支持：{version}")

    offset = 16
    words: list[str] = []
    vecs = np.empty((count, dim), dtype=np.float32)

    for i in range(count):
        if offset + 2 > len(data):
            raise ValueError(f"数据截断（读第 {i} 个词的长度时）")
        wlen = struct.unpack_from(">H", data, offset)[0]
        offset += 2
        if offset + wlen > len(data):
            raise ValueError(f"数据截断（读第 {i} 个词的字节时）")
        words.append(data[offset : offset + wlen].decode("utf-8"))
        offset += wlen
        nbytes = dim * 4
        if offset + nbytes > len(data):
            raise ValueError(f"数据截断（读第 {i} 个词的向量时）")
        vecs[i] = np.frombuffer(data, dtype=">f4", count=dim, offset=offset)
        offset += nbytes

    return dim, words, vecs


def normalize_rows(x: np.ndarray) -> np.ndarray:
    """按行 L2 归一化（避免除零）。"""
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-8)


def cosine_matrix(x: np.ndarray) -> np.ndarray:
    """计算余弦相似度矩阵。"""
    xn = normalize_rows(x)
    return xn @ xn.T
