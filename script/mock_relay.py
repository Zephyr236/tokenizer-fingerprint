#!/usr/bin/env python3
"""假的 OpenAI 兼容端点，用本地 tokenizer 算 prompt_tokens。

用来在没 key 的情况下端到端验证 detect.py。

    python mock_relay.py --model qwen3-8b --port 8118 --overhead 8
    python detect.py --base-url http://127.0.0.1:8118/v1 --api-key sk-test \
                     --model Qwen/Qwen3-8B --expect qwen3-8b

    --corrupt   假装这家中转用的是估算器（按字节数/4 估），验证检测器会不会报警
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_fingerprint as B  # noqa: E402


class Handler(BaseHTTPRequestHandler):
    model_label = ""
    overhead = 0
    corrupt = False
    tok = None
    calls = 0

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._send(400, {"error": {"message": "bad json"}})

        want = body.get("model", "")
        if want and want != self.server.accept_model:
            return self._send(404, {"error": {"message": f"model {want} not found, try {self.server.accept_model}"}})

        text = ""
        for m in body.get("messages", []):
            c = m.get("content")
            if isinstance(c, str):
                text += c
            elif isinstance(c, list):
                text += "".join(p.get("text", "") for p in c if isinstance(p, dict))
        if "input" in body:
            text = body["input"] if isinstance(body["input"], str) else text

        if Handler.corrupt:
            cnt = max(1, math.ceil(len(text.encode("utf-8")) / 4))
        else:
            cnt = int(Handler.tok.counts([text])[0])
        Handler.calls += 1
        payload = {
            "id": f"mock-{Handler.calls}",
            "object": "chat.completion",
            "created": 0,
            "model": want or Handler.model_label,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "OK"}}],
            "usage": {"prompt_tokens": Handler.overhead + cnt,
                      "completion_tokens": 1,
                      "total_tokens": Handler.overhead + cnt + 1},
        }
        self._send(200, payload)

    def _send(self, code, payload):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def load_one(label):
    """只加载被模拟的那一个 tokenizer，启动快（不用把 55 个全装进来）。"""
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "tokenizers")
    for pattern, factory in ((f"{label}.tokenizer.json", B.HFTok),
                             (f"{label}.tokenizer.model", B.SPTok),
                             (f"{label}.tiktoken.model", B.TicTok)):
        p = os.path.join(base, pattern)
        if os.path.exists(p):
            return factory(label, p)
    raise SystemExit(f"找不到 {label} 的 tokenizer 文件，检查 data/tokenizers/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="模拟哪个本地 tokenizer")
    ap.add_argument("--accept-model", default="", help="只接受这个 model 名，默认等于 --model")
    ap.add_argument("--port", type=int, default=8118)
    ap.add_argument("--overhead", type=int, default=3)
    ap.add_argument("--corrupt", action="store_true")
    args = ap.parse_args()

    Handler.model_label = args.model
    Handler.overhead = args.overhead
    Handler.corrupt = args.corrupt
    Handler.tok = load_one(args.model)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    srv.accept_model = args.accept_model or args.model
    mode = "估算器(损坏模式)" if args.corrupt else f"真实 tokenizer {args.model}"
    print(f"mock relay on http://127.0.0.1:{args.port}/v1  [{mode}]  "
          f"overhead={args.overhead}  accept_model={srv.accept_model}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
