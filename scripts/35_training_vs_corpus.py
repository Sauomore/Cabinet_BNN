# -*- coding: utf-8 -*-
"""
关键验证：新基座变差，是「训练量不足」还是「语料变坏」？

为什么必须做这个：
    旧基座 指令89% / 49900步 / 见过 0.82B tokens  ->  事实探针 7/12
    新基座 指令25% / 30000步 / 见过 0.49B tokens  ->  事实探针 2/12
    【两个变量同时变了】，我昨天却只归因给「语料」，这是错的。

判别方法：
    用【同一批文本】喂两个模型，逐位置看下一 token 分布。

    · 若新基座【全面】置信度下降（各种文本上 max_prob 都低、entropy 都高）
      -> 训练量不足（欠训练/欠拟合）

    · 若新基座只在【事实类】文本上差，通用文本上相当
      -> 语料问题（知识被稀释）

用法：
    python scripts/35_training_vs_corpus.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from cabinet_bnn.bnn.attention import build_transformer


def load(ckpt: Path, device):
    """加载模型，词表从权重推断（两个模型词表都是 32000，tokenizer 同一个）。"""
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    a = ck.get("args", {})
    vocab = ck["model"]["embed.weight"].shape[0]
    m = build_transformer(vocab_size=vocab, n_words=vocab,
                          preset=a.get("preset", "base"),
                          max_len=a.get("ctx", 256),
                          ffn_mode="global").to(device)
    m.load_state_dict(ck["model"], strict=False)
    m.eval()
    return m, ck


@torch.no_grad()
def per_position_stats(model, tok, text, ctx=256):
    """返回每个位置的 (max_prob, entropy, top1_token)，用于逐位置对比。"""
    ids = tok.encode(text).ids
    if len(ids) < 3:
        return None
    ids = ids[:ctx]
    idx = torch.tensor([ids], dtype=torch.long,
                       device=next(model.parameters()).device)
    logits, _ = model(idx)
    logits = logits[0, :-1, :].float()          # 预测位置 1..L-1
    tgt = idx[0, 1:]
    logp = F.log_softmax(logits, dim=-1)
    maxp = logp.exp().max(dim=-1).values
    ent = -(logp.exp() * logp).sum(dim=-1)
    nll = -logp.gather(1, tgt.unsqueeze(1)).squeeze(1)
    top1 = logits.argmax(dim=-1)
    return {"maxp": maxp.cpu(), "entropy": ent.cpu(),
            "nll": nll.cpu(), "top1": top1.cpu(), "tgt": tgt.cpu(),
            "n": len(tgt)}


def main() -> int:
    from tokenizers import Tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.from_file(str(ROOT / "data/corpus_all/tokenizer.json"))

    old, old_ck = load(ROOT / "results/lm_base_1ep/final.pt", device)
    new, new_ck = load(ROOT / "results/lm_base_mixed/final.pt", device)

    print("=" * 78)
    print("训练量 vs 语料：谁该为事实退化负责？")
    print("=" * 78)
    print(f"  旧基座 step={old_ck.get('step')}  "
          f"见过 {(old_ck.get('step') or 0)*16384/1e9:.2f} B tokens")
    print(f"  新基座 step={new_ck.get('step')}  "
          f"见过 {(new_ck.get('step') or 0)*16384/1e9:.2f} B tokens")
    print(f"  训练量比 {new_ck.get('step',1)/max(old_ck.get('step',1),1):.2f}x")
    print()

    # ---- 两类文本 ----
    # A. 事实类（知识密集）
    fact_texts = [
        "中国的首都是北京。北京是中国的政治中心，也是历史文化名城。",
        "水的化学式是H2O。地球围绕太阳公转，一圈大约需要365天。",
        "《红楼梦》的作者是曹雪芹。法国的首都是巴黎。",
        "太阳从东方升起，从西方落下。一加一等于二。",
        "中华人民共和国成立于1949年10月1日，首都是北京。",
    ]
    # B. 通用类（无具体事实，纯语言结构）
    gen_texts = [
        "因此，我们需要进一步分析这个问题，并考虑各方面的因素。",
        "在这个过程中，我们可以看到一些明显的趋势和变化。",
        "综上所述，这种方法具有一定的优势和局限性，需要权衡。",
        "首先，我们要明确目标；其次，制定可行的方案；最后，执行并评估。",
        "这是一个复杂的问题，涉及到多个层面的因素和相互关系。",
    ]

    def compare(label, texts):
        rows = []
        for t in texts:
            so = per_position_stats(old, tok, t)
            sn = per_position_stats(new, tok, t)
            if so is None or sn is None:
                continue
            n = min(so["n"], sn["n"])
            rows.append({
                "text": t[:30],
                "old_maxp": float(so["maxp"][:n].mean()),
                "new_maxp": float(sn["maxp"][:n].mean()),
                "old_ent": float(so["entropy"][:n].mean()),
                "new_ent": float(sn["entropy"][:n].mean()),
                "old_nll": float(so["nll"][:n].mean()),
                "new_nll": float(sn["nll"][:n].mean()),
            })
        if not rows:
            return None
        m = {k: float(np.mean([r[k] for r in rows]))
             for k in rows[0] if k != "text"}
        print(f"\n  【{label}】{len(rows)} 段")
        print(f"    {'指标':<14}{'旧基座':>10}{'新基座':>10}{'变化':>10}")
        print("    " + "-" * 44)
        for k, name in [("maxp", "平均最大概率"), ("ent", "平均熵"),
                        ("nll", "平均 NLL")]:
            o, nn = m[f"old_{k}"], m[f"new_{k}"]
            print(f"    {name:<14}{o:>10.4f}{nn:>10.4f}{nn-o:>+10.4f}")
        return m

    mf = compare("事实类文本", fact_texts)
    mg = compare("通用类文本", gen_texts)

    print("\n" + "=" * 78)
    print("判别")
    print("=" * 78)
    if mf and mg:
        d_fact = mf["new_nll"] - mf["old_nll"]
        d_gen = mg["new_nll"] - mg["old_nll"]
        d_ent_f = mf["new_ent"] - mf["old_ent"]
        d_ent_g = mg["new_ent"] - mg["old_ent"]

        print(f"  NLL 变化:   事实类 {d_fact:+.4f}   通用类 {d_gen:+.4f}")
        print(f"  熵 变化:    事实类 {d_ent_f:+.4f}   通用类 {d_ent_g:+.4f}")
        print()
        ratio = d_fact / d_gen if abs(d_gen) > 1e-6 else float("inf")
        print(f"  事实类恶化 / 通用类恶化 = {ratio:.2f}")
        print()
        if abs(d_fact - d_gen) < 0.15 * max(abs(d_fact), abs(d_gen), 1e-6):
            print("  >> 两类【同等程度】退化 -> 指向【训练量不足】（欠拟合）")
        elif ratio > 1.5:
            print("  >> 事实类退化【明显更重】 -> 指向【语料问题】（知识被稀释）")
        else:
            print("  >> 两者都有影响，但事实类更重")

        # 置信度是否整体塌陷
        print()
        print(f"  最大概率: 事实类 {mf['old_maxp']:.4f} -> {mf['new_maxp']:.4f}"
              f"  ({'下降' if mf['new_maxp']<mf['old_maxp'] else '上升'})")
        print(f"           通用类 {mg['old_maxp']:.4f} -> {mg['new_maxp']:.4f}"
              f"  ({'下降' if mg['new_maxp']<mg['old_maxp'] else '上升'})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
