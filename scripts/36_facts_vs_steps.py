# -*- coding: utf-8 -*-
"""
事实知识 vs 训练步数：用旧基座的中间 checkpoint 画曲线。

为什么这是决定性验证（且零成本）：
    旧基座保留了 ckpt_10000/20000/30000/40000 + final(49900)。
    对每个 checkpoint 跑同一套事实探针，就能看出：
      · 事实知识随步数如何增长
      · 达到 7/12 需要多少步
      · 新基座 2/12 是否落在「30000 步应有的水平」上

    如果 30000 步时的旧基座也只有 2-3/12，就证明新基座的问题是训练量，
    不是语料 —— 且能直接算出重训需要多少步。

用法：
    python scripts/36_facts_vs_steps.py
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


def load(ckpt: Path, device):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    a = ck.get("args", {})
    vocab = ck["model"]["embed.weight"].shape[0]
    m = build_transformer(vocab_size=vocab, n_words=vocab,
                          preset=a.get("preset", "base"),
                          max_len=a.get("ctx", 256),
                          ffn_mode="global").to(device)
    m.load_state_dict(ck["model"], strict=False)
    m.eval()
    return m


@torch.no_grad()
def probe(model, tok, prefix, k=3):
    ids = tok.encode(prefix).ids
    idx = torch.tensor([ids], dtype=torch.long,
                       device=next(model.parameters()).device)
    logits, _ = model(idx)
    p = F.softmax(logits[0, -1, :].float(), dim=-1)
    top = torch.topk(p, k)
    return [(tok.decode([int(i)]), float(v)) for v, i in zip(top.values, top.indices)]


@torch.no_grad()
def evaluate_ckpt(model, tok):
    hits, probs = 0, []
    detail = []
    for prefix, expect in PROBES:
        cands = probe(model, tok, prefix)
        t1, p1 = cands[0]
        ok = any(e in t1 for e in expect)
        # 期望词的累计概率（更细的度量）
        pe = sum(p for t, p in cands if any(e in t for e in expect))
        hits += ok
        probs.append(p1)
        detail.append((prefix, t1, ok, pe))
    return hits, probs, detail


def main() -> int:
    from tokenizers import Tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.from_file(str(ROOT / "data/corpus_all/tokenizer.json"))

    ckpts = [
        ("10000", ROOT / "results/lm_base_1ep/ckpt_10000.pt"),
        ("20000", ROOT / "results/lm_base_1ep/ckpt_20000.pt"),
        ("30000", ROOT / "results/lm_base_1ep/ckpt_30000.pt"),
        ("40000", ROOT / "results/lm_base_1ep/ckpt_40000.pt"),
        ("49900", ROOT / "results/lm_base_1ep/final.pt"),
    ]
    new_ck = ROOT / "results/lm_base_mixed/final.pt"

    print("=" * 78)
    print("事实知识 vs 训练步数（旧基座 checkpoint 扫描）")
    print("=" * 78)
    print(f"  语料: corpus_all（指令 89%）    探针: {len(PROBES)} 条\n")
    print(f"  {'step':>7}{'命中':>8}{'平均最大概率':>14}{'期望词累计概率':>16}")
    print("  " + "-" * 48)

    curve = []
    for label, path in ckpts:
        if not path.exists():
            print(f"  {label:>7}   文件缺失，跳过")
            continue
        m = load(path, device)
        hits, probs, detail = evaluate_ckpt(m, tok)
        pe = [d[3] for d in detail]
        avg_p = sum(probs) / len(probs)
        avg_pe = sum(pe) / len(pe)
        print(f"  {label:>7}{hits:>5}/{len(PROBES):<3}{avg_p:>13.4f}{avg_pe:>15.4f}")
        curve.append({"step": int(label), "hits": hits,
                      "avg_maxp": avg_p, "avg_expect_p": avg_pe})
        del m
        torch.cuda.empty_cache()

    # 新基座对照
    print()
    print(f"  {'新基座':>7}{'':>3}", end="")
    mn = load(new_ck, device)
    hits_n, probs_n, detail_n = evaluate_ckpt(mn, tok)
    pe_n = [d[3] for d in detail_n]
    avg_pn = sum(probs_n) / len(probs_n)
    avg_pen = sum(pe_n) / len(pe_n)
    print(f"{hits_n:>5}/{len(PROBES):<3}{avg_pn:>13.4f}{avg_pen:>15.4f}")
    print("      (语料 corpus_mixed，指令 25%)")

    print("\n" + "=" * 78)
    print("判别：新基座 2/12 是否落在「30000 步应有的水平」")
    print("=" * 78)
    c30 = next((c for c in curve if c["step"] == 30000), None)
    if c30:
        print(f"  旧基座 @30000 步:  命中 {c30['hits']}/12   "
              f"平均最大概率 {c30['avg_maxp']:.4f}")
        print(f"  新基座 @30000 步:  命中 {hits_n}/12   "
              f"平均最大概率 {avg_pn:.4f}")
        print()
        d_hit = hits_n - c30["hits"]
        d_p = avg_pn - c30["avg_maxp"]
        print(f"  差异:  命中 {d_hit:+d}   最大概率 {d_p:+.4f}")
        if abs(d_hit) <= 1 and abs(d_p) < 0.06:
            print("\n  >> 两者【基本相当】-> 证明新基座的问题就是【训练量】")
            print("     （30000 步本来就只有这个水平，与语料无关）")
        elif d_hit < -1:
            print("\n  >> 新基座在同步数下【明显更差】-> 语料确实有负面影响")
        else:
            print("\n  >> 差异不显著，需要更多 checkpoint 才能判定")

    out = ROOT / "results/facts_vs_steps.json"
    out.write_text(json.dumps({"curve": curve,
                               "new": {"step": 30000, "hits": hits_n,
                                       "avg_maxp": avg_pn,
                                       "avg_expect_p": avg_pen}},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  结果 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
