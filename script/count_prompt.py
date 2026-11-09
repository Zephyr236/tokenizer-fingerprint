#!/usr/bin/env python3
"""How many tokens does one prompt become, in each tokenizer?

    python count_prompt.py "你的提示词"
    python count_prompt.py "你的提示词" --csv        # 追加到 token_counts.csv
    python count_prompt.py --file prompts.txt       # 一行一段，批量

Output: one line per tokenizer, sorted by token count, plus the spread.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_fingerprint as B  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = os.path.join(HERE, "token_counts.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", default="", help="要统计的提示词")
    ap.add_argument("--file", default="", help="每行一段提示词的文件")
    ap.add_argument("--csv", action="store_true", help="追加到 token_counts.csv")
    ap.add_argument("--pieces", action="store_true", help="同时显示每个 tokenizer 把它切成了哪几块")
    ap.add_argument("--csv-path", default=TABLE)
    args = ap.parse_args()

    if args.file:
        prompts = [ln.rstrip("\n") for ln in open(args.file, encoding="utf-8") if ln.strip()]
    elif args.prompt:
        prompts = [args.prompt]
    else:
        raise SystemExit('用法: python count_prompt.py "提示词"')

    models = B.load_all()
    names = sorted(models)

    for prompt in prompts:
        counts = {nm: int(models[nm].counts([prompt])[0]) for nm in names}
        spread = max(counts.values()) - min(counts.values())
        distinct = len(set(counts.values()))
        print(f"\n{prompt!r}")
        print(f"  长度 {len(prompt)} 字，spread {spread}，取值 {distinct} 种\n")
        for nm, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            extra = ""
            if args.pieces:
                pieces = split_pieces(models[nm], prompt)
                extra = "   " + " | ".join(pieces)
            print(f"  {c:>4}  {nm}{extra}")
        if args.csv:
            write_header = not os.path.exists(args.csv_path)
            with open(args.csv_path, "a", encoding="utf-8-sig", newline="") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(["prompt", "len", "spread", "distinct"] + names)
                w.writerow([prompt, len(prompt), spread, distinct] + [counts[nm] for nm in names])
            print(f"\n  已追加到 {args.csv_path}")


def unbpe(tok: str):
    try:
        return bytes(B.BYTE_DECODER[c] for c in tok).decode("utf-8")
    except Exception:
        return None


def split_pieces(model, prompt):
    """这个 tokenizer 到底把 prompt 切成了哪几块。

    用 tokenizer 自己的 decoder 还原每块文本；还原出替换字符说明这块是半个 UTF-8
    字符（字节级 BPE 常见），标成 <半个字>。
    """
    if isinstance(model, B.HFTok):
        out = []
        for i in model.t.encode(prompt, add_special_tokens=False).ids:
            s = model.t.decode([i])
            out.append("<半个字>" if "\ufffd" in s else s)
        return out
    if isinstance(model, B.SPTok):
        return [p.replace("\u2581", " ") for p in model.t.encode(prompt, out_type=str)]
    out = []
    for i in model.t.encode_ordinary(prompt):
        try:
            out.append(model.t.decode_single_token_bytes(i).decode("utf-8"))
        except UnicodeDecodeError:
            out.append("<半个字>")
    return out


if __name__ == "__main__":
    main()
