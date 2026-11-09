#!/usr/bin/env python3
"""Look inside a tokenizer.json vocabulary without going blind.

tokenizer.json 里的 vocab 是"byte-level 字母表"，不是文本：
    空格 -> Ġ      汉字的一个字节 -> ä 之类
所以用记事本打开看到的全是乱码。这个脚本把它还原成人能读的形式。

用法:
    python inspect_vocab.py --model deepseek-v3 --id 33626
    python inspect_vocab.py --model deepseek-v3 --id 33606-33626
    python inspect_vocab.py --model deepseek-v3 --search 中国特色
    python inspect_vocab.py --model deepseek-v3 --dump vocab.csv --limit 20000
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_fingerprint as B  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def entries(model):
    """yield (id, raw_token, decoded_text, n_bytes)"""
    if isinstance(model, B.HFTok):
        items = []
        for tok, i in model.t.get_vocab().items():
            text = model.t.decode([i])
            ok = "\ufffd" not in text
            items.append((i, tok, text if ok else None, len(tok)))
        return sorted(items)
    if isinstance(model, B.SPTok):
        out = []
        for i in range(model.vocab_size):
            p = model.t.id_to_piece(i)
            out.append((i, p, p.replace("\u2581", " "), len(p)))
        return out
    out = []
    for b, i in sorted(model.t._mergeable_ranks.items(), key=lambda kv: kv[1]):
        try:
            out.append((i, b.decode("utf-8", "replace"), b.decode("utf-8"), len(b)))
        except Exception:
            out.append((i, repr(b), None, len(b)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--id", default="", help="单个 id，或 100-120 这样的区间")
    ap.add_argument("--search", default="", help="按解码后的文本查找")
    ap.add_argument("--dump", default="", help="导出 csv")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    models = B.load_all()
    if args.model not in models:
        raise SystemExit(f"没有 {args.model}，可选: {sorted(models)}")
    model = models[args.model]
    rows = entries(model)
    print(f"{args.model}: {len(rows)} 个 token（含特殊 token）\n")

    if args.search:
        hits = [(i, raw, txt, nb) for i, raw, txt, nb in rows if txt and args.search in txt]
        print(f"包含 {args.search!r} 的 token：{len(hits)} 个")
        for i, raw, txt, nb in hits[:30]:
            print(f"  #{i:<7} {nb:>2}B  {txt!r}")
        if len(hits) > 30:
            print(f"  ... 共 {len(hits)} 个")
        return

    if args.id:
        if "-" in args.id:
            lo, hi = (int(x) for x in args.id.split("-"))
            sel = [(i, raw, txt, nb) for i, raw, txt, nb in rows if lo <= i <= hi]
        else:
            sel = [(i, raw, txt, nb) for i, raw, txt, nb in rows if i == int(args.id)]
        for i, raw, txt, nb in sel:
            flag = "" if txt is not None else "   ← 半个 UTF-8 字符（字节级 BPE 的中间片段）"
            print(f"  #{i:<7} {nb:>2}B  raw={raw!r:<16} text={txt!r}{flag}")
        return

    if args.dump:
        n = args.limit or len(rows)
        with open(args.dump, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "n_bytes", "text", "raw_bytelevel"])
            for i, raw, txt, nb in rows[:n]:
                w.writerow([i, nb, txt or "", raw])
        print(f"导出 {n} 行 -> {args.dump}")
        return

    print("用法见 --help：--id / --search / --dump 三选一")


if __name__ == "__main__":
    main()
