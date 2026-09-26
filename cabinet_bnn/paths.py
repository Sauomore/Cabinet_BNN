# -*- coding: utf-8 -*-
"""
Cabinet-BNN 路径配置（唯一解析点）。

设计原则：
    · 所有路径默认相对于【仓库根目录】解析，不写死任何绝对路径
    · 外部资源（模型、语料）允许用环境变量覆盖，便于不同机器部署
    · 路径不存在时给出可操作的提示，而不是抛裸异常

环境变量（全部可选）：
    CABINET_BGE_MODEL    bge 模型目录（默认依次尝试 models/、../hsh64_models/）
    CABINET_VOCAB        3109 词表路径
    CABINET_CORPUS_DIR   语料目录
    CABINET_DATA_DIR     数据输出目录
    CABINET_RESULTS_DIR  结果输出目录
    CABINET_MODELS_DIR   模型目录

用法：
    from cabinet_bnn.paths import repo_root, results_dir, bge_model_dir
"""

from __future__ import annotations

import os
from pathlib import Path

#: 仓库根目录（本文件位于 <root>/cabinet_bnn/paths.py，故上溯两级）
REPO_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str) -> Path | None:
    v = os.environ.get(name)
    return Path(v).expanduser() if v else None


# ---------------------------------------------------------------- 目录

def repo_root() -> Path:
    return REPO_ROOT


def data_dir() -> Path:
    """数据目录（语料、token 编码等），默认 <root>/data。"""
    p = _env("CABINET_DATA_DIR") or REPO_ROOT / "data"
    p.mkdir(parents=True, exist_ok=True)
    return p


def corpus_dir() -> Path:
    """语料目录，默认 <root>/data/corpus。"""
    p = _env("CABINET_CORPUS_DIR") or data_dir() / "corpus"
    p.mkdir(parents=True, exist_ok=True)
    return p


def corpus_all_dir() -> Path:
    """合并语料目录，默认 <root>/data/corpus_all。"""
    p = _env("CABINET_CORPUS_ALL_DIR") or data_dir() / "corpus_all"
    p.mkdir(parents=True, exist_ok=True)
    return p


def results_dir() -> Path:
    """结果目录，默认 <root>/results。"""
    p = _env("CABINET_RESULTS_DIR") or REPO_ROOT / "results"
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir() -> Path:
    """模型目录，默认 <root>/models。"""
    p = _env("CABINET_MODELS_DIR") or REPO_ROOT / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def logs_dir() -> Path:
    p = _env("CABINET_LOGS_DIR") or REPO_ROOT / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def reference_dir() -> Path:
    """第三方参考实现目录（HSH-64 原仓库），默认 <root>/reference。"""
    return _env("CABINET_REFERENCE_DIR") or REPO_ROOT / "reference"


# ---------------------------------------------------------------- 具体资源

def vocab_3109() -> Path:
    """3109 词表。

    查找顺序：
        1. $CABINET_VOCAB
        2. <root>/reference/Cabinet_hsh64-main/tests/data/vocab_3109.txt
        3. <root>/data/vocab_3109.txt
    """
    cands = [
        _env("CABINET_VOCAB"),
        reference_dir() / "Cabinet_hsh64-main/tests/data/vocab_3109.txt",
        data_dir() / "vocab_3109.txt",
    ]
    for c in cands:
        if c and c.exists():
            return c
    raise FileNotFoundError(
        "找不到 vocab_3109.txt。请任选一种方式提供：\n"
        "  · 设置环境变量 CABINET_VOCAB=/path/to/vocab_3109.txt\n"
        "  · 放到 <repo>/data/vocab_3109.txt\n"
        "  · 克隆 HSH-64 到 <repo>/reference/Cabinet_hsh64-main/"
    )


def bge_model_dir(name: str = "bge-large-zh-v1.5") -> Path:
    """bge 模型目录。

    查找顺序：
        1. $CABINET_BGE_MODEL
        2. <root>/models/<name>
        3. <root>/../hsh64_models/<name>   （HSH-64 项目常见布局）
        4. ~/.cache/huggingface/hub/models--BAAI--<name>/snapshots/*/  （取最新）
    """
    env = _env("CABINET_BGE_MODEL")
    cands = [
        env,
        models_dir() / name,
        REPO_ROOT.parent / "hsh64_models" / name,
    ]
    for c in cands:
        if c and (c / "config.json").exists():
            return c

    # HF 缓存里的 snapshot
    hub = Path.home() / ".cache/huggingface/hub" / f"models--BAAI--{name}"
    snaps = hub / "snapshots"
    if snaps.exists():
        for s in sorted(snaps.iterdir(), reverse=True):
            if (s / "config.json").exists():
                return s

    raise FileNotFoundError(
        f"找不到 {name}。请任选一种方式提供：\n"
        f"  · 设置环境变量 CABINET_BGE_MODEL=/path/to/{name}\n"
        f"  · 放到 <repo>/models/{name}\n"
        f"  · 用 tools/convert_bge_to_safetensors.py 准备好权重后重试\n"
        f"提示：可用 HF_ENDPOINT=https://hf-mirror.com 从镜像下载。"
    )


def qwen_model_dir(name: str = "Qwen2.5-0.5B") -> Path:
    """Qwen 模型目录（用于对比评测）。"""
    env = _env("CABINET_QWEN_MODEL")
    cands = [env, models_dir() / name]
    for c in cands:
        if c and (c / "config.json").exists():
            return c
    raise FileNotFoundError(
        f"找不到 {name}。请设置 CABINET_QWEN_MODEL，或放到 <repo>/models/{name}"
    )


# ---------------------------------------------------------------- 便捷

def describe() -> str:
    """打印当前路径解析结果，便于排查部署问题。"""
    lines = [f"repo_root   = {REPO_ROOT}"]
    for label, fn in [("data", data_dir), ("corpus", corpus_dir),
                      ("corpus_all", corpus_all_dir), ("results", results_dir),
                      ("models", models_dir), ("logs", logs_dir),
                      ("reference", reference_dir)]:
        try:
            lines.append(f"{label:<12}= {fn()}")
        except Exception as e:
            lines.append(f"{label:<12}= <错误: {e}>")
    for label, fn in [("vocab_3109", vocab_3109),
                      ("bge", bge_model_dir), ("qwen", qwen_model_dir)]:
        try:
            lines.append(f"{label:<12}= {fn()}")
        except FileNotFoundError:
            lines.append(f"{label:<12}= <未找到>")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
