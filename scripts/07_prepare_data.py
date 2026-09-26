# -*- coding: utf-8 -*-
"""
语料准备：清洗中文文本 → 训练 BPE tokenizer → 编码为训练用二进制。

为什么要自己训 tokenizer 而不是直接用 Qwen 的：
    · 自训 tokenizer 让整个流程可复现、无外部模型依赖
    · 32k 词表在 500 MB 语料上足够，且比 152k 词表省 5 倍嵌入参数
    · 后续若改用 Qwen tokenizer 做蒸馏，只需重跑本脚本并指定 --tokenizer

输出：
    data/corpus/text.txt          清洗后的纯文本（每行一段）
    data/corpus/tokenizer.json    BPE tokenizer
    data/corpus/train.bin         uint16 token 流（训练集）
    data/corpus/val.bin           uint16 token 流（验证集）

用法：
    python scripts/07_prepare_data.py --vocab-size 32000
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- 清洗

_WS = re.compile(r"[ \t\u3000]+")
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def clean(text: str, min_len: int = 50, max_len: int = 4000) -> str | None:
    """清洗单条文本。返回 None 表示丢弃。"""
    if not text:
        return None
    t = _CTRL.sub("", text)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = _WS.sub(" ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    if len(t) < min_len:
        return None
    if len(t) > max_len:
        t = t[:max_len]
    # 中文字符占比过滤（剔除纯外文/纯符号段落）
    cjk = sum(1 for c in t if "\u4e00" <= c <= "\u9fff")
    if cjk / max(len(t), 1) < 0.30:
        return None
    return t


def load_wikipedia(path: Path, out_txt: Path, max_docs: int | None = None) -> int:
    """解析维基 JSON（list of {completion, source}）并写出纯文本。"""
    print(f"[读取] {path.name} ({path.stat().st_size/1024**2:.1f} MB)", flush=True)
    with io.open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"  原始条数 {len(data):,}", flush=True)

    n_kept = 0
    n_bytes = 0
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with io.open(out_txt, "w", encoding="utf-8") as w:
        for i, item in enumerate(data):
            if max_docs is not None and n_kept >= max_docs:
                break
            t = item.get("completion") if isinstance(item, dict) else None
            c = clean(t) if t else None
            if c is None:
                continue
            w.write(c.replace("\n", " ") + "\n")
            n_kept += 1
            n_bytes += len(c)
            if n_kept % 50000 == 0:
                print(f"  已处理 {n_kept:,} 条 / {n_bytes/1024**2:.0f} MB", flush=True)

    print(f"[清洗] 保留 {n_kept:,} / {len(data):,} 条，共 {n_bytes/1024**2:.1f} MB 文本",
          flush=True)
    return n_kept


# ---------------------------------------------------------------- tokenizer

def train_tokenizer(txt_path: Path, out_json: Path, vocab_size: int) -> None:
    """训练 byte-level BPE tokenizer。"""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    print(f"[tokenizer] 训练 BPE，词表 {vocab_size:,}", flush=True)
    t0 = time.time()
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train([str(txt_path)], trainer)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_json))
    print(f"[tokenizer] 完成，用时 {time.time()-t0:.0f}s -> {out_json.name}", flush=True)


# ---------------------------------------------------------------- 编码

def encode_corpus(txt_path: Path, tok_json: Path, out_dir: Path,
                  val_ratio: float = 0.005, batch_bytes: int = 64 * 1024 * 1024) -> dict:
    """把文本编码成 uint16 token 流，切出训练/验证集。"""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tok_json))
    vocab_size = tok.get_vocab_size()
    if vocab_size > 65536:
        raise ValueError(f"词表 {vocab_size} 超过 uint16 上限，需改用 uint32")
    print(f"[编码] 词表 {vocab_size:,}，使用 uint16", flush=True)

    total_bytes = txt_path.stat().st_size
    n_val = int(total_bytes * val_ratio)
    split_at = total_bytes - n_val
    print(f"  文本 {total_bytes/1024**2:.0f} MB，验证集约 {n_val/1024**2:.0f} MB", flush=True)

    f_train = io.open(out_dir / "train.bin", "wb")
    f_val = io.open(out_dir / "val.bin", "wb")
    n_tok_train = n_tok_val = 0
    t0 = time.time()
    buf_lines: list[str] = []
    buf_bytes = 0
    consumed = 0

    def flush(lines: list[str]) -> None:
        nonlocal n_tok_train, n_tok_val
        if not lines:
            return
        encs = tok.encode_batch(lines)
        ids = np.fromiter((i for e in encs for i in e.ids), dtype=np.uint16)
        # 按已消费字节数判断落入哪个集合
        if consumed < split_at:
            f_train.write(ids.tobytes())
            n_tok_train += ids.size
        else:
            f_val.write(ids.tobytes())
            n_tok_val += ids.size

    with io.open(txt_path, encoding="utf-8") as f:
        for line in f:
            buf_lines.append(line.rstrip("\n"))
            buf_bytes += len(line.encode("utf-8"))
            if buf_bytes >= batch_bytes:
                flush(buf_lines)
                consumed += buf_bytes
                buf_lines, buf_bytes = [], 0
                print(f"  已编码 {consumed/1024**2:.0f} MB，"
                      f"{n_tok_train/1e6:.1f} M train tokens  ({time.time()-t0:.0f}s)",
                      flush=True)
    flush(buf_lines)
    f_train.close()
    f_val.close()

    stats = {
        "vocab_size": vocab_size,
        "train_tokens": int(n_tok_train),
        "val_tokens": int(n_tok_val),
        "text_bytes": int(total_bytes),
        "encode_seconds": round(time.time() - t0, 1),
    }
    print(f"[编码] train {n_tok_train/1e6:.1f} M tokens，val {n_tok_val/1e6:.2f} M tokens，"
          f"用时 {stats['encode_seconds']:.0f}s", flush=True)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="语料准备：清洗 + tokenizer + 编码")
    ap.add_argument("--wiki", type=Path,
                    default=ROOT / "data/corpus/wikipedia-cn-20230720-filtered.json")
    ap.add_argument("--extra", type=Path, nargs="*", default=[],
                    help="额外的 jsonl 语料（SkyPile 等）")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data/corpus")
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--skip-clean", action="store_true", help="复用已有 text.txt")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    txt = args.out_dir / "text.txt"
    tok_json = args.out_dir / "tokenizer.json"

    if not args.skip_clean or not txt.exists():
        if not args.wiki.exists():
            print(f"[错误] 找不到语料 {args.wiki}", file=sys.stderr)
            return 1
        n = load_wikipedia(args.wiki, txt, args.max_docs)
        if n == 0:
            print("[错误] 清洗后无有效文本", file=sys.stderr)
            return 1

    n_lines = sum(1 for _ in io.open(txt, encoding="utf-8"))
    print(f"[文本] {txt.name}: {n_lines:,} 行，{txt.stat().st_size/1024**2:.1f} MB", flush=True)

    if not tok_json.exists():
        train_tokenizer(txt, tok_json, args.vocab_size)
    else:
        print(f"[tokenizer] 已存在，跳过: {tok_json.name}", flush=True)

    stats = encode_corpus(txt, tok_json, args.out_dir)
    stats["n_lines"] = n_lines
    (args.out_dir / "data_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[完成] 统计写入 {args.out_dir/'data_stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
