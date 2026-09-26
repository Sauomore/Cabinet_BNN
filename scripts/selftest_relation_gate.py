# -*- coding: utf-8 -*-
"""
双向关系门自检。

必须验证的四件事（都会静默出错）：
  ① γ=0 时【严格等于】原始注意力 —— 这是从已训练权重微调的前提。
     若不等，等于给基座注入了一个随机扰动，会污染所有对比实验。
  ② 关系矩阵对称 —— 双向的定义。
  ③ KV cache 与全量前向等价 —— 生成正确性的前提。
  ④ 梯度能流到 rel_embed 与 gamma。

及一个【不需要训练】的科学验证：
  ⑤ 关系分数与码的 Hamming 距离是否相关（HSH 核心前提的检验）。

运行：
    python scripts/selftest_relation_gate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from cabinet_bnn.bnn.attention import build_transformer

PASS, FAIL = "[PASS]", "[FAIL]"
_failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {PASS if cond else FAIL} {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        _failures.append(name)


def make_pair(preset="nano", vocab=256, seed=0, **kw):
    """构造两个结构相同、权重相同的模型，一个开关系门一个不开。"""
    torch.manual_seed(seed)
    a = build_transformer(vocab_size=vocab, n_words=vocab, preset=preset, **kw)
    torch.manual_seed(seed)
    b = build_transformer(vocab_size=vocab, n_words=vocab, preset=preset, **kw)
    return a, b


def test_identity_at_gamma_zero() -> None:
    print("\n[1] γ=0 时严格等于原始注意力（微调安全性的前提）")
    vocab, L, B = 256, 32, 2
    torch.manual_seed(0)
    plain = build_transformer(vocab_size=vocab, n_words=vocab, preset="nano")
    torch.manual_seed(0)
    gated = build_transformer(vocab_size=vocab, n_words=vocab, preset="nano",
                              relation_gate=True, relation_dim=16)
    # 把 gated 的主干权重拷成与 plain 一致（关系门自身参数保持初始）
    sd = plain.state_dict()
    missing, unexpected = gated.load_state_dict(sd, strict=False)
    # 期望 only 关系门相关键是 missing
    rel_keys = [k for k in missing if "rel_" in k]
    other_missing = [k for k in missing if "rel_" not in k]
    check("主干权重全部载入", len(other_missing) == 0,
          f"未匹配: {other_missing[:3]}" if other_missing else "全部匹配")
    check("仅关系门参数为新增", len(rel_keys) > 0,
          f"新增 {len(rel_keys)} 个关系门参数")

    x = torch.randint(0, vocab, (B, L))
    with torch.no_grad():
        la, _ = plain(x)
        lb, _ = gated(x)
    err = (la - lb).abs().max().item()
    check("logits 完全一致", err < 1e-6, f"最大误差 {err:.2e}")

    # γ 非零后应当确实改变输出
    with torch.no_grad():
        for blk in gated.blocks:
            blk.attn.rel_gamma.fill_(0.5)
        lc, _ = gated(x)
    diff = (la - lc).abs().max().item()
    check("γ≠0 后输出确实改变", diff > 1e-4, f"最大差异 {diff:.2e}")


def test_symmetry() -> None:
    print("\n[2] 关系矩阵对称（双向的定义）")
    vocab = 128
    m = build_transformer(vocab_size=vocab, n_words=vocab, preset="nano",
                          relation_gate=True, relation_dim=16)
    ids = torch.randint(0, vocab, (2, 24))
    r = m.blocks[0].attn.relation_matrix(ids)
    asym = (r - r.transpose(-1, -2)).abs().max().item()
    check("R == Rᵀ", asym < 1e-6, f"最大不对称 {asym:.2e}")

    # 对角元 = <u,u>/sqrt(r) = |u|²/sqrt(r) >= 0
    diag = torch.diagonal(r, dim1=-2, dim2=-1)
    check("自关系非负（对角 >= 0）", bool((diag >= -1e-6).all()),
          f"最小对角值 {diag.min().item():.4f}")


def test_kv_cache_equivalence() -> None:
    print("\n[3] KV cache 与【逐步扩前缀】等价（关系门开启）")
    print("    基准必须选对：位置 t 的注意力依赖前缀关系结构，")
    print("    所以「前缀 [0..t]」与「全长 [0..L]」本就不该相同。")
    vocab = 128
    for L in (8, 16, 32):
        torch.manual_seed(0)
        m = build_transformer(vocab_size=vocab, n_words=vocab, preset="nano",
                              relation_gate=True, relation_dim=16)
        with torch.no_grad():
            for blk in m.blocks:
                blk.attn.rel_gamma.fill_(0.3)
        m.eval()
        x = torch.randint(0, vocab, (1, L))
        with torch.no_grad():
            ref = torch.cat([m(x[:, :t + 1], token_ids=x[:, :t + 1])[0][:, -1:]
                             for t in range(L)], dim=1)
            caches, outs = None, []
            for t in range(L):
                cur = x[:, :t + 1] if caches is None else x[:, t:t + 1]
                lg, caches = m(cur, None, caches, return_caches=True,
                               token_ids=x[:, :t + 1])
                outs.append(lg[:, -1:])
            inc = torch.cat(outs, dim=1)
        err = (ref - inc).abs().max().item()
        check(f"L={L} 缓存路径 == 扩前缀路径", err < 1e-5, f"最大误差 {err:.2e}")


def test_gate_effectiveness() -> None:
    """门的量级验证 —— 这一步抓到过一个真实缺陷。

    踩过的坑：rel_embed 用 std=0.02 初始化时，关系分数 s = <u_i,u_j>/sqrt(r)
    的 std 只有 4e-4，门只变化 ±0.0007，而注意力 logits 的 std 约 0.33。
    门的扰动小三四个数量级 -> 等于没施加，但代码「看起来」完全正确。

    因此必须直接测门的数值量级，而不是只看代码逻辑。
    """
    print("\n[1b] 门的数值量级（必须与注意力 logits 同量级才有效）")
    import math
    from cabinet_bnn.bnn.attention import CausalSelfAttention

    torch.manual_seed(0)
    B, L, d, H, vocab = 4, 32, 64, 4, 256
    x = torch.randn(B, L, d)
    ids = torch.randint(0, vocab, (B, L))
    att = CausalSelfAttention(d, H, relation_gate=True, n_relation=vocab,
                              relation_dim=16)
    with torch.no_grad():
        q = att.wq(x).view(B, L, H, att.head_dim).transpose(1, 2)
        k = att.wk(x).view(B, L, H, att.head_dim).transpose(1, 2)
        logit_std = float(((q @ k.transpose(-2, -1)) / math.sqrt(att.head_dim)).std())
        att.rel_gamma.fill_(1.0)
        g = att._relation_gate(ids)
        dev = float((g - 1).abs().max())
    ratio = dev / max(logit_std, 1e-9)
    print(f"    注意力 logits std = {logit_std:.4f}")
    print(f"    γ=1 时门的最大偏差 = {dev:.4f}   （比值 {ratio:.3f}）")
    check("门的量级与 logits 相当（比值 > 0.1）", ratio > 0.1,
          f"比值 {ratio:.3f}；若太小说明 rel_embed 初始化过小，门形同虚设")


def test_gradients() -> None:
    """用【交叉熵】测梯度。

    踩过的坑：最初用 out.pow(2).mean() 作损失，它对注意力的均匀缩放不变
    （softmax 温度缩放不改变加权平均的期望），导致 γ 和 u 的梯度恒为 0
    —— 那是【损失函数选错】，不是实现缺陷。必须用真实训练损失。
    """
    print("\n[4] 梯度流动（交叉熵损失）")
    vocab, d = 256, 64
    torch.manual_seed(0)
    m = build_transformer(vocab_size=vocab, n_words=vocab, preset="nano",
                          relation_gate=True, relation_dim=16)
    # nano 的 d_model=128；把 lm_head 换成小维度以便构造交叉熵目标
    ids = torch.randint(0, vocab, (2, 20))
    tgt = torch.randint(0, vocab, (2, 20))
    logits, _ = m(ids)
    loss = F.cross_entropy(logits.reshape(-1, vocab), tgt.reshape(-1))
    loss.backward()
    a = m.blocks[0].attn
    gg = a.rel_gamma.grad
    check("rel_gamma 有梯度", gg is not None and gg.abs().max().item() > 1e-8,
          f"|∇γ| = {0 if gg is None else gg.abs().max().item():.3e}")
    # 注意：此模型 γ 仍为初始值 0，故 rel_embed 梯度【必然为 0】
    # （g = 1 + 0·s 与 u 无关）。u 的梯度在下面的 γ≠0 用例中检查。
    check("主干 embed 有梯度", m.embed.weight.grad is not None
          and m.embed.weight.grad.abs().max().item() > 0)

    # γ=0 时 rel_embed 梯度必然为 0：g = 1 + 0·s 与 u 无关。
    # 但 γ 自身的梯度在 γ=0 时【不为 0】（dg/dγ = s ≠ 0），所以能自己走出来。
    # 因此要分两种状态测 u 的梯度：γ=0（应为 0）与 γ≠0（应非 0）。
    torch.manual_seed(0)
    m2 = build_transformer(vocab_size=vocab, n_words=vocab, preset="nano",
                           relation_gate=True, relation_dim=16)
    lg2, _ = m2(ids)
    F.cross_entropy(lg2.reshape(-1, vocab), tgt.reshape(-1)).backward()
    g2 = m2.blocks[0].attn.rel_embed.weight.grad
    gg2 = m2.blocks[0].attn.rel_gamma.grad
    check("γ=0 时 rel_embed 梯度为 0（数学必然，非缺陷）",
          g2 is None or g2.abs().max().item() < 1e-12,
          "因为 g = 1 + 0·s 与 u 无关")
    check("γ=0 时 rel_gamma 梯度【非零】（能自行启动）",
          gg2 is not None and gg2.abs().max().item() > 1e-8,
          f"|∇γ| = {0 if gg2 is None else gg2.abs().max().item():.3e}"
          "  -> 关系门能自己从 0 走出来，无需手工 warmup")

    # γ≠0 后 u 应能收到梯度
    torch.manual_seed(0)
    m3 = build_transformer(vocab_size=vocab, n_words=vocab, preset="nano",
                           relation_gate=True, relation_dim=16)
    with torch.no_grad():
        for blk in m3.blocks:
            blk.attn.rel_gamma.fill_(0.3)
    lg3, _ = m3(ids)
    F.cross_entropy(lg3.reshape(-1, vocab), tgt.reshape(-1)).backward()
    g3 = m3.blocks[0].attn.rel_embed.weight.grad
    check("γ≠0 后 rel_embed 有梯度（关系向量可学）",
          g3 is not None and g3.abs().max().item() > 1e-8,
          f"|∇u| = {0 if g3 is None else g3.abs().max().item():.3e}")


def test_hamming_correlation() -> None:
    print("\n[5] 关系分数 vs 码的 Hamming 距离（HSH 核心前提的检验）")
    print("    预期：未训练时无关（u 随机）；训练后若 HSH 前提成立，应负相关")
    vocab = 512
    m = build_transformer(vocab_size=vocab, n_words=vocab, preset="nano",
                          relation_gate=True, relation_dim=16)
    ids = torch.arange(vocab).unsqueeze(0)
    r = m.blocks[0].attn.relation_matrix(ids)[0]          # (V, V)
    iu = torch.triu_indices(vocab, vocab, offset=1)
    rv = r[iu[0], iu[1]]

    # 随机码下的 Hamming 距离（模拟未训练状态）
    torch.manual_seed(0)
    bits = torch.randint(0, 2, (vocab, 64))
    hd = (bits[:, None, :] != bits[None, :, :]).sum(-1).float()[iu[0], iu[1]]

    corr = float(torch.corrcoef(torch.stack([hd, rv]))[0, 1])
    print(f"    corr(Hamming, 关系分数) = {corr:+.4f}")
    check("未训练时接近 0（符合预期，非缺陷）", abs(corr) < 0.3,
          f"实测 {corr:+.4f}")


def main() -> int:
    print("=" * 74)
    print("双向关系门自检")
    print("=" * 74)
    test_identity_at_gamma_zero()
    test_gate_effectiveness()
    test_symmetry()
    test_kv_cache_equivalence()
    test_gradients()
    test_hamming_correlation()

    print("\n" + "=" * 74)
    if _failures:
        print(f"失败 {len(_failures)} 项：")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("全部通过 [OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
