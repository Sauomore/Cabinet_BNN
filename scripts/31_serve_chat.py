# -*- coding: utf-8 -*-
"""
Cabinet-BNN 对话服务：把训练好的模型接到浏览器。

设计取舍：
    · 零额外依赖 —— 只用标准库 http.server。装了 flask/gradio 也行，
      但别人 clone 下来最怕「还差一个包」，所以坚持用标准库。
    · SSE 流式输出 —— 逐 token 推送，不用等整段生成完。
    · 单模型常驻显存 —— 启动时加载一次，之后每次请求复用。
    · 带 KV cache 的增量生成 —— 见 BNNTransformerLM.generate。

⚠️ 关于「对话」的重要说明：
    这是【基座模型】，不是指令微调模型。它学到的是「续写」而不是「回答」。
    所以界面按【续写】设计：你给一段开头，它往下接。
    写「问：…\\n答：」这种格式会更像问答，因为语料里有这种结构。

用法：
    python scripts/31_serve_chat.py --ckpt results/lm_base_mixed/final.pt
    python scripts/31_serve_chat.py --data-dir data/corpus_mixed --port 8000
然后浏览器打开 http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from cabinet_bnn.bnn.attention import build_transformer

# 全局模型状态（单例）
STATE: dict = {"model": None, "tok": None, "lock": threading.Lock(),
               "info": {}, "device": None}
HTML_PATH = ROOT / "web" / "chat.html"


# ---------------------------------------------------------------- 模型

def load_model(ckpt: Path, data_dir: Path, device: torch.device):
    """加载模型与 tokenizer。返回 (model, tokenizer, info)。"""
    from tokenizers import Tokenizer
    stats = json.loads((data_dir / "data_stats.json").read_text(encoding="utf-8"))
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    a = ck.get("args", {})
    model = build_transformer(
        vocab_size=stats["vocab_size"], n_words=stats["vocab_size"],
        preset=a.get("preset", "base"), max_len=a.get("ctx", 256),
        ffn_mode=a.get("ffn_mode", "global"),
    ).to(device)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    model.eval()
    tok = Tokenizer.from_file(str(data_dir / "tokenizer.json"))
    info = {
        "ckpt": str(ckpt),
        "params": sum(p.numel() for p in model.parameters()),
        "step": ck.get("step"),
        "val_loss": ck.get("val_loss"),
        "vocab": stats["vocab_size"],
        "ctx": model.cfg.max_len,
        "preset": a.get("preset", "base"),
        "data_dir": str(data_dir),
        "missing_keys": len(missing),
        "device": str(device),
    }
    return model, tok, info


# ---------------------------------------------------------------- 生成

@torch.no_grad()
def stream_generate(prompt: str, max_new: int, temperature: float,
                    top_k: int, repetition_penalty: float, seed: int | None):
    """逐 token 生成，yield (token_id, text, finished)。"""
    model, tok = STATE["model"], STATE["tok"]
    device = STATE["device"]
    if seed is not None:
        torch.manual_seed(seed)

    ids = tok.encode(prompt).ids if prompt else [tok.token_to_id("<bos>")]
    ctx = model.cfg.max_len
    # 上下文超长时保留尾部
    if len(ids) >= ctx:
        ids = ids[-(ctx - 1):]
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    caches = None
    out_ids = list(ids)
    produced: list[int] = []

    for _ in range(max_new):
        cur = idx if caches is None else idx[:, -1:]
        logits, caches = model(cur, None, caches, return_caches=True,
                               token_ids=idx)
        logits = logits[:, -1, :].float()

        # 重复惩罚：降低已出现 token 的分数
        if repetition_penalty and repetition_penalty != 1.0 and produced:
            uniq = torch.tensor(sorted(set(produced)), device=device)
            logits[0, uniq] /= repetition_penalty

        logits = logits / max(temperature, 1e-6)
        if top_k and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
            logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
        probs = F.softmax(logits, dim=-1)
        nxt = int(torch.multinomial(probs, num_samples=1).item())

        produced.append(nxt)
        out_ids.append(nxt)
        idx = torch.cat([idx, torch.tensor([[nxt]], device=device)], dim=1)
        if len(out_ids) > ctx:
            out_ids = out_ids[-ctx:]

        text = tok.decode(produced)
        yield nxt, text, False

    yield None, tok.decode(produced), True


# ---------------------------------------------------------------- HTTP

class Handler(BaseHTTPRequestHandler):
    server_version = "CabinetBNN/1.0"

    def log_message(self, fmt, *args):        # 静音默认日志
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            if not HTML_PATH.exists():
                self._send(500, b"web/chat.html not found",
                           "text/plain; charset=utf-8")
                return
            self._send(200, HTML_PATH.read_bytes(), "text/html; charset=utf-8")
        elif u.path == "/info":
            self._send(200, json.dumps(STATE["info"], ensure_ascii=False,
                                       indent=2).encode("utf-8"),
                       "application/json; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/generate":
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception as e:
            self._send(400, str(e).encode("utf-8"), "text/plain; charset=utf-8")
            return

        prompt = req.get("prompt", "")
        max_new = int(req.get("max_new", 80))
        temp = float(req.get("temperature", 0.8))
        top_k = int(req.get("top_k", 40))
        rep = float(req.get("repetition_penalty", 1.0))
        seed = req.get("seed")

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()

        t0 = time.time()
        ntok = 0
        try:
            with STATE["lock"]:                     # 串行化，避免并发抢显存
                for _tid, text, done in stream_generate(
                        prompt, max_new, temp, top_k, rep, seed):
                    ntok += 1
                    payload = json.dumps({"text": text, "done": done},
                                         ensure_ascii=False)
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    if done:
                        break
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            dt = time.time() - t0
            meta = json.dumps({"info": True, "tokens": ntok,
                               "seconds": round(dt, 2),
                               "tok_per_s": round(ntok / max(dt, 1e-6), 1)},
                              ensure_ascii=False)
            try:
                self.wfile.write(f"data: {meta}\n\n".encode("utf-8"))
                self.wfile.flush()
            except Exception:
                pass


def main() -> int:
    ap = argparse.ArgumentParser(description="Cabinet-BNN 对话服务")
    ap.add_argument("--ckpt", type=Path,
                    default=ROOT / "results/lm_base_mixed/final.pt")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data/corpus_mixed")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    if not args.ckpt.exists():
        print(f"找不到模型 {args.ckpt}")
        return 1
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("Cabinet-BNN 对话服务")
    print("=" * 72)
    print(f"  加载 {args.ckpt} …")
    t0 = time.time()
    model, tok, info = load_model(args.ckpt, args.data_dir, device)
    STATE.update({"model": model, "tok": tok, "device": device, "info": info})

    print(f"  参数量 {info['params']:,}   step {info['step']}   "
          f"val_loss {info['val_loss']:.4f}   词表 {info['vocab']:,}   "
          f"ctx {info['ctx']}")
    print(f"  设备 {device}   加载用时 {time.time()-t0:.1f}s")
    print()
    print(f"  打开 http://{args.host}:{args.port}")
    print("  ⚠️ 这是【基座模型】，擅长续写而不是问答。")
    print("     想要更像问答，可以写「问：…\n答：」这样的格式。")
    print("  Ctrl+C 停止")
    print("=" * 72)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(
            f"http://{args.host}:{args.port}")).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
