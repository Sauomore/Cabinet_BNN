# -*- coding: utf-8 -*-
"""
新旧基座的生成质量对比（同提示词、同采样参数、同 seed）。

为什么必须这样比：
    两个基座训练语料不同，ppl 不可直接比较（已用交叉评测确认）。
    生成质量也一样 —— 必须用相同的提示词与采样参数，并且固定随机种子，
    否则「差异」可能只是采样噪声。

用法：
    python scripts/30_compare_bases.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from cabinet_bnn.bnn.attention import build_transformer

PROMPTS = [
    "",
    "中国的首都是",
    "请解释什么是机器学习。",
    "今天天气怎么样？",
    "问题：如何学习编程？\n回答：",
    "人工智能的发展",
    "北京是",
    "他说道：",
    "一、",
    "1. 首先",
]


def load(ckpt: Path, data_dir: Path, device):
    from tokenizers import Tokenizer
    stats = json.loads((data_dir / "data_stats.json").read_text(encoding="utf-8"))
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    a = ck.get("args", {})
    m = build_transformer(vocab_size=stats["vocab_size"],
                          n_words=stats["vocab_size"],
                          preset=a.get("preset", "base"),
                          max_len=a.get("ctx", 256),
                          ffn_mode="global").to(device)
    m.load_state_dict(ck["model"], strict=False)
    m.eval()
    tok = Tokenizer.from_file(str(data_dir / "tokenizer.json"))
    return m, tok, ck


@torch.no_grad()
def gen(model, tok, prompt, n, temp, top_k, seed):
    torch.manual_seed(seed)                 # ← 固定种子，两次生成可比
    ids = tok.encode(prompt).ids if prompt else [tok.token_to_id("<bos>")]
    idx = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
    out = model.generate(idx, max_new_tokens=n, temperature=temp, top_k=top_k)
    return tok.decode(out[0].tolist()[len(ids):])


def main() -> int:
    ap = argparse.ArgumentParser(description="新旧基座生成对比")
    ap.add_argument("--old-ckpt", type=Path,
                    default=ROOT / "results/lm_base_1ep/final.pt")
    ap.add_argument("--old-data", type=Path, default=ROOT / "data/corpus_all")
    ap.add_argument("--new-ckpt", type=Path,
                    default=ROOT / "results/lm_base_mixed/final.pt")
    ap.add_argument("--new-data", type=Path, default=ROOT / "data/corpus_mixed")
    ap.add_argument("--n", type=int, default=70)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 78)
    print("新旧基座生成对比（同提示词 · 同采样参数 · 同种子）")
    print("=" * 78)

    old_m, old_tok, old_ck = load(args.old_ckpt, args.old_data, device)
    new_m, new_tok, new_ck = load(args.new_ckpt, args.new_data, device)

    print(f"  旧基座 {args.old_ckpt.parent.name}/final.pt  "
          f"step={old_ck.get('step')}  val_loss={old_ck.get('val_loss', float('nan')):.4f}")
    print(f"        语料 corpus_all（指令 ~89%）")
    print(f"  新基座 {args.new_ckpt.parent.name}/final.pt  "
          f"step={new_ck.get('step')}  val_loss={new_ck.get('val_loss', float('nan')):.4f}")
    print(f"        语料 corpus_mixed（指令 25%）")
    print(f"  采样 temperature={args.temp} top_k={args.top_k} n={args.n} seed={args.seed}")

    results = []
    for p in PROMPTS:
        label = p.replace("\n", "\\n") if p else "(空提示)"
        print("\n" + "─" * 78)
        print(f"  【{label}】")
        print("─" * 78)
        o = gen(old_m, old_tok, p, args.n, args.temp, args.top_k, args.seed)
        nw = gen(new_m, new_tok, p, args.n, args.temp, args.top_k, args.seed)
        print("  [旧基座]")
        for i in range(0, len(o), 66):
            print(f"    {o[i:i+66]}")
        print("  [新基座]")
        for i in range(0, len(nw), 66):
            print(f"    {nw[i:i+66]}")
        results.append({"prompt": p, "old": o, "new": nw})

    out = ROOT / "results/base_generation_compare.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n  结果 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
