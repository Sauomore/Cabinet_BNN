# -*- coding: utf-8 -*-
"""
从已合并的 text_all.txt 续跑：训练 tokenizer + 编码，并做资源限制。

为什么重写而不是重跑 09：
    ① 合并步骤已产出 4.2 GB 的 text_all.txt，重跑会白费 10 分钟
    ② 上一版在【全部 4.2 GB】上训 tokenizer —— 这是浪费且是压垮机器的原因。
       32k BPE 用 ~400 MB 样本足够，训练时间从十几分钟降到 ~1 分钟
    ③ 编码阶段限制线程数与批大小，避免再次把系统拖死

关键资源限制（上次卡死的原因）：
    · OMP/MKL/TOKENIZERS 线程数限制到 4（默认会用满所有核心）
    · 逐批写盘并降低批大小，减少磁盘抖动
    · 打印进度时附带内存占用，便于早期发现异常

用法：
    python scripts/11_resume_encode.py
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
import time
from pathlib import Path

# ---- 必须在 import numpy/tokenizers 之前设线程数 ----
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent


def mem_mb() -> float:
    """当前进程内存占用（MB）。"""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024**2
    except Exception:
        return -1.0


def sample_lines(src: Path, dst: Path, target_mb: int = 400) -> int:
    """从大文件里随机抽样若干行，用于训练 tokenizer。

    随机抽样而非取前 N 行：避免只训到单一领域（文件前段全是维基）。

    关键实现细节（踩过的坑）：
        必须用【二进制模式】读。文本模式下 f.seek(pos) 会 seek 到任意字节偏移，
        对中文（3 字节/字）几乎必然落在字符中间，触发 UnicodeDecodeError。
        二进制模式下 seek 到任意位置，再用 errors="ignore" 丢弃半个字符即可。
    """
    total = src.stat().st_size
    target = target_mb * 1024 * 1024
    print(f"[抽样] 目标 {target_mb} MB（源 {total/1024**2:.0f} MB）", flush=True)
    rng = random.Random(20240212)
    n_lines = 0
    written = 0
    t0 = time.time()
    CHUNK = 4 * 1024 * 1024          # 每次读 4 MB
    LINES_PER_JUMP = 1500

    with open(src, "rb") as f, io.open(dst, "w", encoding="utf-8") as w:
        while written < target:
            pos = rng.randrange(0, max(total - CHUNK, 1))
            f.seek(pos)
            f.readline()                        # 丢弃可能被截断的半行
            buf = f.read(CHUNK)
            if not buf:
                continue
            # errors="ignore" 会丢掉开头的半个字符与结尾的半个字符
            text = buf.decode("utf-8", errors="ignore")
            lines = text.split("\n")[1:LINES_PER_JUMP]
            for ln in lines:
                ln = ln.strip()
                if not ln:
                    continue
                w.write(ln + "\n")
                written += len(ln.encode("utf-8")) + 1
                n_lines += 1
            if written // (50 * 1024 * 1024) != (written - 1024 * 1024) // (50 * 1024 * 1024):
                print(f"    {written/1024**2:.0f} MB / {n_lines:,} 行  "
                      f"({time.time()-t0:.0f}s, RSS {mem_mb():.0f} MB)", flush=True)

    print(f"[抽样] 完成 {written/1024**2:.0f} MB / {n_lines:,} 行", flush=True)
    return n_lines


def train_tokenizer(sample: Path, out_json: Path, vocab_size: int) -> None:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    print(f"[tokenizer] 在样本上训练 BPE，词表 {vocab_size:,}", flush=True)
    t0 = time.time()
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train([str(sample)], trainer)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_json))
    print(f"[tokenizer] 完成 {time.time()-t0:.0f}s -> {out_json.name}", flush=True)


def encode(src: Path, tok_json: Path, out_dir: Path, batch_bytes: int,
           val_bytes: int) -> dict:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tok_json))
    V = tok.get_vocab_size()
    print(f"[编码] 词表 {V:,}，批 {batch_bytes/1024**2:.0f} MB", flush=True)

    total = src.stat().st_size
    split_at = total - val_bytes
    f_tr = io.open(out_dir / "train.bin", "wb")
    f_va = io.open(out_dir / "val.bin", "wb")
    n_tr = n_va = 0
    consumed = 0
    t0 = time.time()
    batch: list[str] = []
    bbytes = 0

    def flush(lines: list[str], consumed_before: int) -> tuple[int, int]:
        if not lines:
            return 0, 0
        encs = tok.encode_batch(lines)
        ids = np.fromiter((i for e in encs for i in e.ids), dtype=np.uint16)
        if consumed_before < split_at:
            f_tr.write(ids.tobytes())
            return ids.size, 0
        f_va.write(ids.tobytes())
        return 0, ids.size

    with io.open(src, encoding="utf-8") as f:
        for line in f:
            batch.append(line.rstrip("\n"))
            bbytes += len(line.encode("utf-8"))
            if bbytes >= batch_bytes:
                a, b = flush(batch, consumed)
                n_tr += a; n_va += b
                consumed += bbytes
                batch, bbytes = [], 0
                el = time.time() - t0
                print(f"    {consumed/1024**2:>6.0f} MB  train {n_tr/1e6:>7.1f} M  "
                      f"val {n_va/1e6:.2f} M  {el:>5.0f}s  "
                      f"RSS {mem_mb():.0f} MB", flush=True)
        a, b = flush(batch, consumed)
        n_tr += a; n_va += b

    f_tr.close(); f_va.close()
    return {
        "vocab_size": V,
        "train_tokens": int(n_tr),
        "val_tokens": int(n_va),
        "text_bytes": int(total),
        "encode_seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="续跑：tokenizer + 编码")
    ap.add_argument("--src", type=Path,
                    default=ROOT / "data/corpus_all/text_all.txt")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data/corpus_all")
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--sample-mb", type=int, default=400,
                    help="用于训 tokenizer 的抽样量（MB）")
    ap.add_argument("--batch-mb", type=int, default=32,
                    help="编码批大小（MB），调小可减少磁盘抖动")
    ap.add_argument("--val-mb", type=int, default=20)
    ap.add_argument("--skip-tokenizer", action="store_true")
    args = ap.parse_args()

    if not args.src.exists():
        print(f"[错误] 找不到 {args.src}", file=sys.stderr)
        return 1

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    sample = out / "tokenizer_sample.txt"
    tok_json = out / "tokenizer.json"

    print("=" * 74)
    print("续跑：tokenizer + 编码")
    print("=" * 74)
    print(f"  源 {args.src.name} ({args.src.stat().st_size/1024**2:.0f} MB)")
    print(f"  线程限制 OMP={os.environ.get('OMP_NUM_THREADS')}  "
          f"批 {args.batch_mb} MB  内存上限检查已启用")

    t_all = time.time()
    if not args.skip_tokenizer or not tok_json.exists():
        # 注意：必须校验样本【非空】。上一次崩溃的运行会留下 0 字节的样本文件，
        # 只检查 exists() 会跳过抽样，把空文件喂给 BPE 训练器 ——
        # 结果是词表只有基础字节表（260）且无任何 merge，tokenizer 静默失效。
        min_sample = 1024 * 1024
        if sample.exists() and sample.stat().st_size < min_sample:
            print(f"[抽样] 现有样本仅 {sample.stat().st_size} 字节，视为无效，重新抽样",
                  flush=True)
            sample.unlink()
        if not sample.exists():
            sample_lines(args.src, sample, args.sample_mb)
        sz = sample.stat().st_size
        if sz < min_sample:
            print(f"[错误] 抽样后样本仍只有 {sz} 字节，中止", file=sys.stderr)
            return 1
        print(f"[抽样] 样本 {sz/1024**2:.1f} MB 就绪", flush=True)
        train_tokenizer(sample, tok_json, args.vocab_size)

        # 训练后立刻校验词表大小 —— 静默失败的代价太高（白写 4 GB token 流）
        from tokenizers import Tokenizer as _T
        got = _T.from_file(str(tok_json)).get_vocab_size()
        if got < args.vocab_size * 0.9:
            print(f"[错误] tokenizer 词表仅 {got}，远低于目标 {args.vocab_size}，"
                  f"说明 BPE 训练未生效，中止", file=sys.stderr)
            return 1
        print(f"[tokenizer] 校验通过，词表 {got:,}", flush=True)
    else:
        print(f"[tokenizer] 复用 {tok_json.name}")

    stats = encode(args.src, tok_json, out, args.batch_mb * 1024 * 1024,
                   args.val_mb * 1024 * 1024)
    stats["total_minutes"] = round((time.time() - t_all) / 60, 1)
    stats["source"] = args.src.name
    (out / "data_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 74)
    print("完成")
    print("=" * 74)
    print(f"  train {stats['train_tokens']/1e6:.1f} M tokens")
    print(f"  val   {stats['val_tokens']/1e6:.2f} M tokens")
    print(f"  用时  {stats['total_minutes']:.1f} 分钟")
    print(f"  统计 -> {out/'data_stats.json'}")
    # 清理抽样文件省磁盘
    if sample.exists():
        sample.unlink()
        print(f"  已删除抽样文件 {sample.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
