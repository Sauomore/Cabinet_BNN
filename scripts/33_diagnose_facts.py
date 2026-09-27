# -*- coding: utf-8 -*-
"""
诊断：新基座是不是被我改坏了？

动机：
    我把语料从「指令 89%」改成「指令 25%，skypile 65%」，
    理由是原配置导致格式过拟合。但现在看：
      · 问答能力几乎归零
      · 重复明显
      · 事实错误严重
    需要判断这是【语料改动造成的】还是【41.6M 规模本来就这样】。

做法：不做生成（随机性大），直接看【下一 token 概率分布】——
      这是确定性的，能精确回答「模型到底认为首都该填什么」。

对照：
    旧基座 (lm_base_1ep, 语料指令 89%)
    新基座 (lm_base_mixed, 语料指令 25%)
    两者在同一批探针句上，看 top-1 预测与概率。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from cabinet_bnn.bnn.attention import build_transformer

#: (探针前缀, 期望的合理后续词列表)
PROBES = [
    ("中国的首都是", ["北京"]),
    ("中华人民共和国首都是", ["北京"]),
    ("北京是中国的", ["首都"]),
    ("水的化学式是", ["H", "h", "Ｈ"]),
    ("一加一等于", ["2", "二", "两"]),
    ("地球围绕", ["太阳"]),
    ("《红楼梦》的作者是", ["曹雪芹"]),
    ("太阳从", ["东"]),
    ("法国的首都是", ["巴黎"]),
    ("请问", ["你", "您"]),
    ("答案是", ["：", ":"]),
    ("因为", ["它", "他", "这"]),
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
    return m, Tokenizer.from_file(str(data_dir / "tokenizer.json")), ck


@torch.no_grad()
def top_preds(model, tok, prefix, k=5):
    ids = tok.encode(prefix).ids
    idx = torch.tensor([ids], dtype=torch.long,
                       device=next(model.parameters()).device)
    logits, _ = model(idx)
    logits = logits[0, -1, :].float()
    probs = F.softmax(logits, dim=-1)
    top = torch.topk(probs, k)
    return [(tok.decode([int(i)]), float(p))
            for p, i in zip(top.values, top.indices)]


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    old, old_tok, old_ck = load(ROOT / "results/lm_base_1ep/final.pt",
                                ROOT / "data/corpus_all", device)
    new, new_tok, new_ck = load(ROOT / "results/lm_base_mixed/final.pt",
                                ROOT / "data/corpus_mixed", device)

    print("=" * 84)
    print("下一 token 预测对比（确定性，非采样）")
    print("=" * 84)
    print(f"  旧基座 step={old_ck.get('step')}  val_loss={old_ck.get('val_loss'):.4f}"
          f"  语料: 指令 ~89%")
    print(f"  新基座 step={new_ck.get('step')}  val_loss={new_ck.get('val_loss'):.4f}"
          f"  语料: 指令 25% / skypile 65%")
    print()
    print(f"  {'前缀':<24}{'旧基座 top-1':<20}{'新基座 top-1':<20}{'旧p':>7}{'新p':>7}")
    print("  " + "-" * 76)

    hit_old = hit_new = n = 0
    for prefix, expect in PROBES:
        o = top_preds(old, old_tok, prefix)
        w = top_preds(new, new_tok, prefix)
        o1, op = o[0]
        w1, wp = w[0]
        o_ok = any(e in o1 for e in expect)
        w_ok = any(e in w1 for e in expect)
        hit_old += o_ok
        hit_new += w_ok
        n += 1
        mark = lambda ok: "✓" if ok else "✗"
        print(f"  {prefix:<24}{o1[:16]:<18}{mark(o_ok)}  "
              f"{w1[:16]:<18}{mark(w_ok)}  {op:>6.3f} {wp:>6.3f}")

    print()
    print(f"  命中期望词: 旧 {hit_old}/{n}   新 {hit_new}/{n}")

    # 详细看两个关键探针的前 5 个候选
    for prefix in ("中国的首都是", "北京是中国的"):
        print(f"\n  【{prefix}】前 5 候选:")
        print(f"    旧: " + "  ".join(f"{t!r}:{p:.3f}"
                                     for t, p in top_preds(old, old_tok, prefix)))
        print(f"    新: " + "  ".join(f"{t!r}:{p:.3f}"
                                     for t, p in top_preds(new, new_tok, prefix)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
