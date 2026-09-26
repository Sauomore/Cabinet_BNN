# -*- coding: utf-8 -*-
"""
交互式生成：加载 checkpoint，对一组提示词生成文本。

为什么单独写而不是用训练脚本里那段：
    训练脚本只在 eval 时打一句样本，看不到模型在不同提示下的表现差异。
    这个脚本用同一模型跑一批提示 × 多组采样参数，便于直观判断能力边界。

用法：
    python scripts/12_generate.py --ckpt results/lm_base_1ep/best.pt
    python scripts/12_generate.py --ckpt ... --sweep        # 扫描温度/top_k
    python scripts/12_generate.py --ckpt ... --prompt "中国" --interactive
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cabinet_bnn.bnn.attention import build_transformer

# 覆盖不同类型：陈述 / 问答 / 对话 / 列表 / 续写
DEFAULT_PROMPTS = [
    "",
    "中国的首都是",
    "请解释什么是机器学习。",
    "今天天气怎么样？",
    "问题：如何学习编程？\n回答：",
    "一、",
    "人工智能的发展",
    "他说：",
    "1. 首先",
    "北京是",
]


@torch.no_grad()
def generate(model, tok, prompt: str, n: int, temp: float, top_k: int, device,
             rep_penalty: float = 1.0) -> str:
    model.eval()
    ids = tok.encode(prompt).ids if prompt else [tok.token_to_id("<bos>")]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=n, temperature=temp,
                         top_k=top_k or None)
    gen = out[0].tolist()[len(ids):]
    return tok.decode(gen)


def load(ckpt: Path, data_dir: Path, device):
    from tokenizers import Tokenizer
    stats = json.loads((data_dir / "data_stats.json").read_text(encoding="utf-8"))
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    a = ck.get("args", {})
    model = build_transformer(
        vocab_size=stats["vocab_size"], n_words=stats["vocab_size"],
        preset=a.get("preset", "base"), max_len=a.get("ctx", 256),
        ffn_mode=a.get("ffn_mode", "global"),
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    tok = Tokenizer.from_file(str(data_dir / "tokenizer.json"))
    return model, tok, ck


def main() -> int:
    ap = argparse.ArgumentParser(description="生成样本")
    ap.add_argument("--ckpt", type=Path,
                    default=ROOT / "results/lm_base_1ep/best.pt")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data/corpus_all")
    ap.add_argument("--n-tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, tok, ck = load(args.ckpt, args.data_dir, device)
    ps = model.n_params()

    print("=" * 78)
    print("Cabinet-BNN 生成样本")
    print("=" * 78)
    print(f"  模型 {args.ckpt.parent.name}/{args.ckpt.name}")
    print(f"  参数量 {ps['total']:,}   step {ck.get('step','?')}   "
          f"val_loss {ck.get('val_loss', float('nan')):.4f}")
    print(f"  配置 preset={model.cfg.preset if hasattr(model.cfg,'preset') else '?'} "
          f"d={model.cfg.d_model} layers={model.cfg.n_layers} "
          f"ctx={model.cfg.max_len} 词表={model.cfg.vocab_size:,}")

    if args.interactive:
        print("\n进入交互模式（空行退出）。输入提示词后回车：")
        while True:
            try:
                p = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not p:
                break
            print(generate(model, tok, p, args.n_tokens, args.temperature,
                           args.top_k, device))
        return 0

    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS
    settings = ([(0.7, 40), (0.8, 40), (0.9, 60), (1.0, 100)]
                if args.sweep else [(args.temperature, args.top_k)])

    for temp, tk in settings:
        print(f"\n{'─'*78}")
        print(f"  temperature={temp}  top_k={tk}  长度={args.n_tokens}")
        print("─" * 78)
        for p in prompts:
            txt = generate(model, tok, p, args.n_tokens, temp, tk, device)
            label = p if p else "(空提示)"
            print(f"\n  【{label}】")
            # 按显示宽度换行，便于阅读
            for i in range(0, len(txt), 62):
                print(f"    {txt[i:i+62]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
