# -*- coding: utf-8 -*-
"""用 bge-large-zh-v1.5 为 vocab_3109 生成嵌入缓存。"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cabinet_bnn.paths import bge_model_dir, results_dir, vocab_3109
from cabinet_bnn.data.embedding_cache import write_cache
from sentence_transformers import SentenceTransformer
import numpy as np

vocab = vocab_3109()
words = [l.strip() for l in vocab.read_text(encoding="utf-8").splitlines() if l.strip()]
print(f"词数: {len(words)}", flush=True)
t0 = time.time()
m = SentenceTransformer(str(bge_model_dir()), device="cuda")
print(f"模型加载 {time.time()-t0:.1f}s dim={m.get_sentence_embedding_dimension()}", flush=True)
t0 = time.time()
V = m.encode(words, batch_size=128, show_progress_bar=False, normalize_embeddings=True)
print(f"编码 {time.time()-t0:.1f}s shape={V.shape}", flush=True)
out = results_dir() / "bge_large_3109.cache"
write_cache(out, V.shape[1], list(zip(words, V.astype(np.float32))))
print(f"写出 {out} ({out.stat().st_size:,} bytes)", flush=True)
