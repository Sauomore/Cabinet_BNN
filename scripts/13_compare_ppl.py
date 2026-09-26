# -*- coding: utf-8 -*-
"""
跨模型困惑度对比：我们的模型 vs Qwen2.5-0.5B（在同一段中文文本上）。

为什么必须用 per-char（bits per character）而不是 per-token ppl：
    两者用了不同的 tokenizer（32k vs 152k），per-token ppl 不可直接比较。
    正确做法是换算成「每字符多少比特」：
        bpc = (Σ -log2 p(token)) / n_chars
    这个量与 tokenizer 无关，是唯一公平的口径。

用法：
    python scripts/13_compare_ppl.py
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cabinet_bnn.bnn.attention import build_transformer


def load_text(path: Path, n_lines: int) -> str:
    lines = []
    with io.open(path, encoding="utf-8") as f:
        for i, l in enumerate(f):
            if i >= n_lines:
                break
            lines.append(l.strip())
    return "\n".join(lines)


# ---------------------------------------------------------------- 我们的模型

@torch.no_grad()
def ours_bpc(ckpt: Path, data_dir: Path, text: str, device,
             n_windows: int = 60, batch: int = 8) -> dict:
    from tokenizers import Tokenizer
    stats = json.loads((data_dir / "data_stats.json").read_text(encoding="utf-8"))
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    a = ck.get("args", {})
    model = build_transformer(
        vocab_size=stats["vocab_size"], n_words=stats["vocab_size"],
        preset=a.get("preset", "base"), max_len=a.get("ctx", 256),
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    tok = Tokenizer.from_file(str(data_dir / "tokenizer.json"))

    ids = tok.encode(text).ids
    ctx = model.cfg.max_len
    n_chars = len(text)

    # 采样若干窗口（整段太长，跑完没意义且慢）
    rng = np.random.default_rng(0)
    starts = rng.integers(0, max(len(ids) - ctx - 1, 1), size=n_windows)
    total_nll = 0.0
    total_tok = 0
    for s in range(0, len(starts), batch):
        chunk_starts = starts[s:s + batch]
        x = np.stack([np.asarray(ids[i:i + ctx], dtype=np.int64) for i in chunk_starts])
        y = np.stack([np.asarray(ids[i + 1:i + ctx + 1], dtype=np.int64) for i in chunk_starts])
        with torch.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            logits, _ = model(torch.from_numpy(x).to(device))
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   torch.from_numpy(y).to(device).reshape(-1),
                                   reduction="sum")
        total_nll += float(loss)
        total_tok += y.size

    # 用实际字符数换算：这些 token 对应的字符数
    chars_covered = sum(len(tok.decode(ids[i:i + ctx + 1])) for i in starts)
    nll_per_tok = total_nll / total_tok
    bpc = nll_per_tok / math.log(2) * (total_tok / chars_covered)
    return {
        "name": f"ours ({ckpt.parent.name})",
        "params": sum(p.numel() for p in model.parameters()),
        "vocab": stats["vocab_size"],
        "ppl_token": math.exp(min(nll_per_tok, 20)),
        "bpc": bpc,
        "chars_per_token": chars_covered / total_tok,
        "n_tokens_evaluated": total_tok,
    }


# ---------------------------------------------------------------- Qwen

@torch.no_grad()
def qwen_bpc(model_dir: Path, text: str, device, ctx: int = 1024,
             max_tokens: int = 40000) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.float32)
    model.to(device).eval()

    ids = tok(text, return_tensors="pt").input_ids[0]
    if len(ids) > max_tokens:
        ids = ids[:max_tokens]
    total_nll = 0.0
    total_tok = 0
    for s in range(0, len(ids) - 1, ctx):
        chunk = ids[s:s + ctx + 1]
        if chunk.numel() < 2:
            break
        inp = chunk[:-1].unsqueeze(0).to(device)
        tgt = chunk[1:].unsqueeze(0).to(device)
        out = model(inp)
        loss = F.cross_entropy(out.logits.reshape(-1, out.logits.shape[-1]),
                               tgt.reshape(-1), reduction="sum")
        total_nll += float(loss)
        total_tok += tgt.numel()

    # 这 total_tok 个 token 覆盖的字符数
    covered = tok.decode(ids[:total_tok + 1])
    n_chars = len(covered)
    nll_per_tok = total_nll / total_tok
    bpc = nll_per_tok / math.log(2) * (total_tok / n_chars)
    return {
        "name": "Qwen2.5-0.5B",
        "params": sum(p.numel() for p in model.parameters()),
        "vocab": tok.vocab_size,
        "ppl_token": math.exp(min(nll_per_tok, 20)),
        "bpc": bpc,
        "chars_per_token": n_chars / total_tok,
        "n_tokens_evaluated": total_tok,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="跨模型困惑度对比（per-char 口径）")
    ap.add_argument("--ckpt", type=Path, default=ROOT / "results/lm_base_1ep/best.pt")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data/corpus_all")
    ap.add_argument("--qwen-dir", type=Path, default=ROOT / "models/Qwen2.5-0.5B")
    ap.add_argument("--n-lines", type=int, default=1200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=ROOT / "results/ppl_compare.json")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    text = load_text(args.data_dir / "text_all.txt", args.n_lines)
    print("=" * 78)
    print("跨模型困惑度对比")
    print("=" * 78)
    print(f"  评测文本 {len(text):,} 字符（取自训练语料前 {args.n_lines} 行）")
    print(f"  设备 {device}")
    print(f"  ⚠️ 口径说明：per-token ppl 因 tokenizer 不同【不可比】，")
    print(f"     下表以 bpc（每字符比特数）为准 —— 越低越好。")

    results = []
    try:
        r = ours_bpc(args.ckpt, args.data_dir, text, device)
        results.append(r)
        print(f"\n  [我们的] {r['name']}  参数 {r['params']:,}")
    except Exception as e:
        print(f"\n  [我们的] 失败: {type(e).__name__}: {e}")

    try:
        r = qwen_bpc(args.qwen_dir, text, device)
        results.append(r)
        print(f"  [Qwen]   Qwen2.5-0.5B  参数 {r['params']:,}")
    except Exception as e:
        print(f"  [Qwen] 失败: {type(e).__name__}: {str(e)[:200]}")

    if results:
        print("\n" + "-" * 78)
        print(f"  {'模型':<26}{'参数量':>14}{'词表':>8}{'字/token':>10}"
              f"{'per-token ppl':>15}{'bpc ↓':>9}")
        print("  " + "-" * 74)
        for r in results:
            print(f"  {r['name']:<26}{r['params']:>14,}{r['vocab']:>8,}"
                  f"{r['chars_per_token']:>10.2f}{r['ppl_token']:>15.2f}{r['bpc']:>9.3f}")
        if len(results) == 2:
            a, b = results[0], results[1]
            print(f"\n  bpc 比值 (ours/Qwen) = {a['bpc']/b['bpc']:.3f}")
            print(f"  参数量比值          = {a['params']/b['params']:.3f}")
            print(f"\n  解读：bpc 比值 {a['bpc']/b['bpc']:.2f} 表示我们的模型"
                  f"在同等文本上每字符多/少用 {(a['bpc']/b['bpc']-1)*100:+.1f}% 的比特。")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  结果 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
