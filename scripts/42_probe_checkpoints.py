# -*- coding: utf-8 -*-
"""
在训练过程中抽检 checkpoint 的事实探针，尽早判断好坏。

动机：
    lm_base_clean 的训练样本在 step 17500 后出现退化迹象
    （数字堆砌、词重复、句子破碎），和上次失败时的形态一样。
    但 val_loss 曲线是正常的 —— 上次也是。

    ppl 曲线不能作为判据，事实探针可以。而 checkpoint 已经存了，
    所以不必等训练跑完再看，现在就能查。

对比组：
    A. corpus_clean（本次，过滤 3.2%）
    B. corpus_all  （旧语料，未过滤）—— 已有 ckpt_10000/20000
    C. corpus_mixed（上次失败，指令 25% + skypile 65%）

用法：
    python scripts/42_probe_checkpoints.py
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
    return m, ck.get("step"), ck.get("val_loss", float("nan"))


@torch.no_grad()
def probe(model, tok, prefix, k=5):
    ids = tok.encode(prefix).ids
    idx = torch.tensor([ids], dtype=torch.long,
                       device=next(model.parameters()).device)
    logits, _ = model(idx)
    p = F.softmax(logits[0, -1, :].float(), dim=-1)
    top = torch.topk(p, k)
    return [(tok.decode([int(i)]), float(v)) for v, i in zip(top.values, top.indices)]


@torch.no_grad()
def evaluate(model, tok):
    hits, probs, expect_p, detail = 0, [], [], []
    for prefix, exp in PROBES:
        cands = probe(model, tok, prefix)
        t1, p1 = cands[0]
        ok = any(e in t1 for e in exp)
        pe = sum(p for t, p in cands if any(e in t for e in exp))
        hits += ok
        probs.append(p1)
        expect_p.append(pe)
        detail.append((prefix, t1, ok))
    n = len(PROBES)
    return hits, sum(probs) / n, sum(expect_p) / n, detail


def main() -> int:
    from tokenizers import Tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.from_file(str(ROOT / "data/corpus_all/tokenizer.json"))

    candidates = [
        ("clean@10000",  ROOT / "results/lm_base_clean/ckpt_10000.pt"),
        ("clean@20000",  ROOT / "results/lm_base_clean/ckpt_20000.pt"),
        ("all@10000",    ROOT / "results/lm_base_1ep/ckpt_10000.pt"),
        ("all@20000",    ROOT / "results/lm_base_1ep/ckpt_20000.pt"),
        ("mixed@30000",  ROOT / "results/lm_base_mixed/final.pt"),
    ]

    print("=" * 82)
    print("checkpoint 事实探针抽检（训练中途即可判断）")
    print("=" * 82)
    print(f"  {'checkpoint':<16}{'step':>7}{'val_loss':>11}{'命中':>9}"
          f"{'平均最大概率':>14}{'期望词累计概率':>16}")
    print("  " + "-" * 74)

    rows = []
    for label, path in candidates:
        if not path.exists():
            print(f"  {label:<16}   文件缺失，跳过")
            continue
        m, step, vl = load(path, device)
        hits, avg_p, avg_pe, detail = evaluate(m, tok)
        print(f"  {label:<16}{str(step):>7}{vl:>11.4f}{hits:>6}/{len(PROBES)}"
              f"{avg_p:>14.4f}{avg_pe:>16.4f}")
        rows.append({"label": label, "step": step, "val_loss": vl,
                     "hits": hits, "avg_maxp": avg_p, "avg_expect_p": avg_pe,
                     "detail": detail})
        del m
        torch.cuda.empty_cache()

    print("\n" + "=" * 82)
    print("判据：命中 >= 6/12  且  期望词概率 >= 0.19")
    print("=" * 82)
    for r in rows:
        ok = (r["hits"] >= 6) and (r["avg_expect_p"] >= 0.19)
        print(f"  {r['label']:<16} 命中 {r['hits']}/12  "
              f"期望词 {r['avg_expect_p']:.4f}   -> "
              f"{'✅ 通过' if ok else '❌ 未通过'}")

    # 逐条对比 clean@20000 与 all@20000（同步数，唯一变量是语料）
    a = next((r for r in rows if r["label"] == "clean@20000"), None)
    b = next((r for r in rows if r["label"] == "all@20000"), None)
    if a and b:
        print("\n" + "=" * 82)
        print("同 step 对照（20000 步）：过滤 vs 未过滤")
        print("=" * 82)
        print(f"  {'前缀':<22}{'clean':<16}{'all':<16}")
        print("  " + "-" * 54)
        for (p1, t1, ok1), (p2, t2, ok2) in zip(a["detail"], b["detail"]):
            m1 = "✓" if ok1 else "✗"
            m2 = "✓" if ok2 else "✗"
            print(f"  {p1:<22}{t1[:12]:<14}{m1} {t2[:12]:<14}{m2}")

    out = ROOT / "results/probe_checkpoints.json"
    out.write_text(json.dumps(
        [{k: v for k, v in r.items() if k != "detail"} for r in rows],
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  结果 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
