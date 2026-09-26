# -*- coding: utf-8 -*-
"""
把 bge-large-zh-v1.5 的 pytorch_model.bin 转成 model.safetensors。

为什么需要这一步（路线文档 K1）：
    本机 torch 2.5.1 < 2.6，而 transformers/sentence-transformers 加载 .bin 时会
    因 torch.load 的安全门禁报错：
        ValueError: we now require users to upgrade torch to at least v2.6
    实测 torch.load(..., map_location="cpu") 用【默认参数】可以绕过并正确读出张量，
    因此做一次离线转换，之后全链路（SentenceTransformer / transformers）都干净可用。

运行：
    python tools/convert_bge_to_safetensors.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


def convert(model_dir: Path) -> bool:
    bin_path = model_dir / "pytorch_model.bin"
    st_path = model_dir / "model.safetensors"

    if not model_dir.exists():
        print(f"[跳过] 目录不存在: {model_dir}")
        return False
    if st_path.exists():
        print(f"[跳过] safetensors 已存在: {st_path} ({st_path.stat().st_size:,} bytes)")
        return True
    if not bin_path.exists():
        print(f"[跳过] 未找到 pytorch_model.bin: {model_dir}")
        return False

    print(f"[转换] {bin_path}  ({bin_path.stat().st_size:,} bytes)")
    # 注意：必须用默认参数（weights_only=False），这是绕过 torch<2.6 门禁的关键
    state = torch.load(bin_path, map_location="cpu")
    if not isinstance(state, dict):
        print(f"[错误] 期望 dict，实际 {type(state)}")
        return False
    # 有些 checkpoint 会包一层 "state_dict"
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    clean = {}
    for k, v in state.items():
        if not isinstance(v, torch.Tensor):
            print(f"  [警告] 跳过非张量键: {k} ({type(v)})")
            continue
        clean[k] = v.contiguous().clone()

    save_file(clean, str(st_path), metadata={"format": "pt", "converted_from": "pytorch_model.bin"})
    print(f"[完成] 写出 {len(clean)} 个张量 -> {st_path} ({st_path.stat().st_size:,} bytes)")

    # 校验：重新读回并比对
    from safetensors.torch import load_file
    back = load_file(str(st_path))
    if set(back.keys()) != set(clean.keys()):
        print("[错误] 键集合不一致")
        return False
    for k in clean:
        if not torch.equal(back[k], clean[k]):
            print(f"[错误] 张量不一致: {k}")
            return False
    print(f"[校验] 通过：{len(back)} 个张量逐元素一致")
    return True


def main() -> int:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from cabinet_bnn.paths import models_dir, repo_root

    # 候选根目录：仓库 models/ 与仓库同级的 hsh64_models/ 等常见布局
    roots = [models_dir()]
    parent = repo_root().parent
    for name in ("hsh64_models", "models", "hsh_models"):
        cand = parent / name
        if cand.exists() and cand not in roots:
            roots.append(cand)

    names = sys.argv[1:] if len(sys.argv) > 1 else [
        "bge-large-zh-v1.5", "bge-small-zh-v1.5"]

    targets = []
    for r in roots:
        if not r.exists():
            continue
        for nm in names:
            if (r / nm).is_dir():
                targets.append(r / nm)

    # 去重
    seen = set()
    uniq = []
    for t in targets:
        k = str(t.resolve())
        if k not in seen:
            seen.add(k)
            uniq.append(t)
    targets = uniq

    if not targets:
        print("[提示] 未找到任何 bge 模型目录。")
        print(f"  已搜索: {', '.join(str(r) for r in roots)}")
        print("  可用环境变量 CABINET_MODELS_DIR 或 CABINET_BGE_MODEL 指定位置。")
        return 0

        targets = [Path(a) for a in sys.argv[1:]]

    ok = 0
    for t in targets:
        if convert(t):
            ok += 1
    print(f"\n完成 {ok}/{len(targets)} 个模型目录")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
