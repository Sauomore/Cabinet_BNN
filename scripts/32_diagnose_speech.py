# -*- coding: utf-8 -*-
"""
诊断：模型说胡话，是哪种原因？

三种可能，必须分开：
  A. 用法问题 —— 它是【续写】模型，我按【问答】用它，它当然不会「回答」
  B. 采样问题 —— 温度/top-k/重复惩罚设得不好，导致重复或跑题
  C. 模型问题 —— 语料配比改动后，模型确实变差了

做法：同一批提示词，分三组对照跑，看差异落在哪一组：
  组1  原始续写（用训练时的目标：给一段开头，让它接）
  组2  问答格式（问：…\n答：，靠近语料里的结构）
  组3  不同采样参数（温度、top-k、重复惩罚）

用法：
    python scripts/32_diagnose_speech.py
"""

from __future__ import annotations

import argparse
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
def gen(model, tok, prompt, n, temp, top_k, rep=1.0, seed=7):
    torch.manual_seed(seed)
    ids = tok.encode(prompt).ids if prompt else [tok.token_to_id("<bos>")]
    device = next(model.parameters()).device
    ctx = model.cfg.max_len
    if len(ids) >= ctx:
        ids = ids[-(ctx - 1):]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    caches, produced = None, []
    for _ in range(n):
        cur = idx if caches is None else idx[:, -1:]
        logits, caches = model(cur, None, caches, return_caches=True, token_ids=idx)
        logits = logits[:, -1, :].float()
        if rep != 1.0 and produced:
            uniq = torch.tensor(sorted(set(produced)), device=device)
            logits[0, uniq] /= rep
        logits = logits / max(temp, 1e-6)
        if top_k and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
            logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
        probs = F.softmax(logits, dim=-1)
        nxt = int(torch.multinomial(probs, 1).item())
        produced.append(nxt)
        idx = torch.cat([idx, torch.tensor([[nxt]], device=device)], dim=1)
    return tok.decode(produced)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=ROOT / "results/lm_base_mixed/final.pt")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data/corpus_mixed")
    ap.add_argument("--n", type=int, default=45)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, ck = load(args.ckpt, args.data_dir, device)
    print("=" * 78)
    print(f"胡话诊断  step={ck.get('step')}  val_loss={ck.get('val_loss'):.4f}")
    print("=" * 78)

    # ---------- 组1：原始续写（训练时的真实目标） ----------
    print("\n" + "=" * 78)
    print("组1  原始续写 —— 给它一段【完整的句子开头】，看它接得像不像话")
    print("     （这才是它被训练来做的事）")
    print("=" * 78)
    cont = [
        "昭通机场位于中国云南昭通，始建于1935年，",
        "人工智能技术在医疗领域的应用正在增加。例如，",
        "北京是中国首都，也是历史文化名城。故宫位于",
        "改革开放以来，中国的经济结构发生了显著变化。",
        "这个故事的结尾是，",
    ]
    for p in cont:
        o = gen(model, tok, p, args.n, 0.8, 40, 1.0)
        print(f"\n  【{p}】")
        for i in range(0, len(o), 64):
            print(f"    {o[i:i+64]}")

    # ---------- 组2：问答格式 ----------
    print("\n" + "=" * 78)
    print("组2  问答格式 —— 用「问：…\\n答：」，靠近语料里的结构")
    print("=" * 78)
    qa = [
        "问：什么是机器学习？\n答：",
        "问：中国的首都是哪里？\n答：",
        "问：如何学习编程？\n答：",
        "问题：今天天气怎么样？\n回答：",
    ]
    for p in qa:
        o = gen(model, tok, p, args.n, 0.8, 40, 1.0)
        print(f"\n  【{p.replace(chr(10), ' / ')}】")
        for i in range(0, len(o), 64):
            print(f"    {o[i:i+64]}")

    # ---------- 组3：采样参数 ----------
    print("\n" + "=" * 78)
    print("组3  同一提示词，不同采样参数 —— 看重复是否是采样问题")
    print("=" * 78)
    p3 = "人工智能技术在医疗领域的应用正在增加。例如，"
    for temp, tk, rep in [(0.5, 20, 1.0), (0.8, 40, 1.0), (0.8, 40, 1.15),
                          (0.6, 10, 1.2), (1.0, 80, 1.1)]:
        o = gen(model, tok, p3, args.n, temp, tk, rep)
        # 量化重复度
        toks = tok.encode(o).ids
        uniq = len(set(toks)) / max(len(toks), 1)
        print(f"\n  T={temp} k={tk} rep={rep}   唯一token率 {uniq:.2f}")
        print(f"    {o[:130]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
