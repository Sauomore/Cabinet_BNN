# -*- coding: utf-8 -*-
"""下载中文语料。先取体积可控的整份 JSON，再抽样 SkyPile 补量。"""
import os, time, sys, json
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import hf_hub_download

import sys as _sys
_sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from cabinet_bnn.paths import corpus_dir

OUT = str(corpus_dir())
os.makedirs(OUT, exist_ok=True)

jobs = [
    ("pleisto/wikipedia-cn-20230720-filtered", "wikipedia-cn-20230720-filtered.json"),
    ("BelleGroup/train_3.5M_CN", "train_3.5M_CN.json"),
]
for repo, fn in jobs:
    t0 = time.time()
    try:
        p = hf_hub_download(repo, fn, repo_type="dataset", local_dir=OUT)
        sz = os.path.getsize(p)
        print(f"  OK {fn}  {sz/1024**2:.1f} MB  ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"  FAIL {fn}: {type(e).__name__}: {str(e)[:200]}", flush=True)

# SkyPile 抽样若干分片
for i in range(3):
    fn = f"data/2020-40_zh_head_{i:04d}.jsonl"
    t0 = time.time()
    try:
        p = hf_hub_download("skywork/SkyPile-150B", fn, repo_type="dataset", local_dir=OUT)
        sz = os.path.getsize(p)
        print(f"  OK {fn}  {sz/1024**2:.1f} MB  ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"  FAIL {fn}: {type(e).__name__}: {str(e)[:200]}", flush=True)
print("DONE", flush=True)
