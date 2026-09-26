# -*- coding: utf-8 -*-
"""
从已合并的语料文本编码为 token 流。

与 07_prepare_data.py 的区别：
    07 走「wiki JSON -> text.txt -> 训 tokenizer -> 编码」的完整流程；
    本脚本走「已合并的 text_mixed.txt -> 复用 tokenizer -> 编码」。

为什么复用 tokenizer（不重训）：
    这样【语料配比】是唯一的变量，模型结构与词表都不变，与旧 corpus_all
    的对比才干净。重训 tokenizer 会让两个变量同时变，无法归因。

⚠️ 踩过的坑（务必保留校验）：
    之前一次编码静默失败 —— 抽样文件 0 字节导致 `if not sample.exists()`
    判断成立后跳过了重新抽样，tokenizer 在空文本上训练出只有 260 个 token
    的字节级词表，白写了 4 GB 垃圾 token。
    本脚本因此强制校验：① 词表大小 ② 往返无损 ③ 输出 token 数与文本量相称。

用法：
    python scripts/23_encode_corpus.py --src data/corpus_mixed/text_mixed.txt \
        --tok data/corpus_all/tokenizer.json --out-dir data/corpus_mixed
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent


def verify_tokenizer(tok, txt_path: Path) -> None:
    """强制校验 tokenizer 可用 —— 防止静默产生垃圾 token。"""
    vocab = tok.get_vocab_size()
    if vocab < 10000:
        raise SystemExit(
            f"词表只有 {vocab} 个 token，明显异常（预期 ~32000）。"
            f"拒绝用它编码 —— 之前正是这个错误静默白写了 4 GB 垃圾数据。")

    # 往返无损：解回来必须与原文一致
    with io.open(txt_path, encoding="utf-8", errors="ignore") as f:
        sample_lines = [f.readline().rstrip("\n") for _ in range(200)]
    sample = "\n".join(l for l in sample_lines if l)
    if len(sample) < 5000:
        raise SystemExit(f"校验样本过短（{len(sample)} 字符），无法确认 tokenizer 可用")
    enc = tok.encode(sample)
    dec = tok.decode(enc.ids)
    if dec.strip() != sample.strip():
        raise SystemExit("tokenizer 往返校验失败（编码后解码与原文不一致）")
    print(f"  [校验] 词表 {vocab:,}  token 数 {len(enc.ids):,}  "
          f"压缩率 {len(sample)/len(enc.ids):.2f} 字/token  往返无损 ✅")


def main() -> int:
    ap = argparse.ArgumentParser(description="编码语料为 token 流")
    ap.add_argument("--src", type=Path,
                    default=ROOT / "data/corpus_mixed/text_mixed.txt")
    ap.add_argument("--tok", type=Path,
                    default=ROOT / "data/corpus_all/tokenizer.json")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data/corpus_mixed")
    ap.add_argument("--val-frac", type=float, default=0.0008,
                    help="验证集比例（按字节）")
    ap.add_argument("--batch-mb", type=int, default=32)
    args = ap.parse_args()

    if not args.src.exists():
        raise SystemExit(f"找不到源文本 {args.src}")
    if not args.tok.exists():
        raise SystemExit(f"找不到 tokenizer {args.tok}")

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(args.tok))

    print("=" * 76)
    print("编码语料为 token 流")
    print("=" * 76)
    total_bytes = args.src.stat().st_size
    print(f"  源文本 {args.src.name}  {total_bytes/1024**3:.2f} GB")
    print(f"  tokenizer {args.tok}")
    verify_tokenizer(tok, args.src)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    f_tr = io.open(args.out_dir / "train.bin", "wb")
    f_va = io.open(args.out_dir / "val.bin", "wb")
    # 按字节计数切换 train/val，保证 val 均匀分布在整个文件上
    val_budget = int(total_bytes * args.val_frac)
    val_used = 0
    tr_tokens = va_tokens = 0
    n_lines = 0
    t0 = time.time()
    batch_mb = args.batch_mb
    buf: list[str] = []
    buf_bytes = 0

    def flush() -> None:
        nonlocal tr_tokens, va_tokens, val_used, buf, buf_bytes
        if not buf:
            return
        text = "\n".join(buf)
        enc = tok.encode(text).ids
        arr = np.asarray(enc, dtype=np.uint16)
        # 按 token 数比例切分到 val（保持与配额一致）
        n_val = 0
        if val_used < val_budget and len(arr) > 0:
            frac = min(1.0, (val_budget - val_used) / max(len(arr), 1))
            n_val = int(len(arr) * frac)
        if n_val > 0:
            f_va.write(arr[:n_val].tobytes())
            va_tokens += n_val
            val_used += buf_bytes * (n_val / max(len(arr), 1))
        f_tr.write(arr[n_val:].tobytes())
        tr_tokens += len(arr) - n_val
        buf = []
        buf_bytes = 0

    with io.open(args.src, encoding="utf-8", errors="ignore") as f:
        for line in f:
            buf.append(line.rstrip("\n"))
            buf_bytes += len(line)
            n_lines += 1
            if buf_bytes >= batch_mb * 1024 * 1024:
                flush()
                done = (tr_tokens + va_tokens) / max(total_bytes, 1)
                print(f"    {tr_tokens+va_tokens:>12,} tokens  "
                      f"({time.time()-t0:.0f}s)  {n_lines:,} 行", flush=True)
        flush()

    f_tr.close()
    f_va.close()

    total_tok = tr_tokens + va_tokens
    print("\n" + "=" * 76)
    print("完成")
    print("=" * 76)
    print(f"  train.bin  {tr_tokens:>13,} tokens")
    print(f"  val.bin    {va_tokens:>13,} tokens")
    print(f"  合计       {total_tok:>13,} tokens")
    print(f"  压缩率     {total_bytes/total_tok:.2f} 字节/token")
    print(f"  行数       {n_lines:,}")
    print(f"  用时       {(time.time()-t0)/60:.1f} 分钟")

    # 校验：token 数与文本量应相称
    expect = total_bytes / 5.0                    # 中文约 4-6 字节/token
    if total_tok < expect * 0.3:
        print(f"\n  ⚠️ [警告] token 数 {total_tok:,} 远低于预期量级"
              f"（约 {expect:,.0f}）—— 可能编码异常，请检查")

    stats = {
        "vocab_size": tok.get_vocab_size(),
        "train_tokens": tr_tokens,
        "val_tokens": va_tokens,
        "text_bytes": total_bytes,
        "lines": n_lines,
        "minutes": (time.time() - t0) / 60,
        "tokenizer": str(args.tok),
        "source": str(args.src),
    }
    (args.out_dir / "data_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  统计 -> {args.out_dir/'data_stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
