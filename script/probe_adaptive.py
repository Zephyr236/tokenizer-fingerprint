#!/usr/bin/env python3
"""Adaptive tokenizer identification: walk the decision tree, one probe per level.

Unlike probe_online.py (fixed 48-probe codebook + nearest neighbour), this walks
`tree.json`: probe the root fragment, read the token count, jump to the matching
branch, repeat until the leaf names the tokenizer group. Average 2-3 requests.

A wrong count sends you down the wrong branch with no recovery, so:
  * every branch value is an exact token count, not a range
  * `--beam` keeps alternative branches alive and reports all surviving groups

Usage
-----
    python probe_adaptive.py --dry-run --model deepseek-v3
    set RELAY_API_KEY=sk-xxx
    python probe_adaptive.py --base-url https://relay.example.com/v1 --model gpt-4o
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.join(HERE, "tree.json")
CONTROL = "a"  # 1 token in every model


def http_count(base_url, api_key, model, probe, timeout=60):
    body = json.dumps({"model": model,
                       "messages": [{"role": "user", "content": probe}],
                       "max_tokens": 4, "temperature": 0, "stream": False}).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {api_key}"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read().decode("utf-8", "replace"))
    usage = payload.get("usage") or {}
    for key in ("prompt_tokens", "input_tokens"):
        if isinstance(usage.get(key), int):
            return usage[key]
    raise RuntimeError("no prompt_tokens in response")


def local_counter(model):
    import build_fingerprint as B
    models = B.load_all()
    if model not in models:
        raise SystemExit(f"unknown model {model!r}; try: {sorted(models)}")
    tok = models[model]
    return lambda s: int(tok.counts([s])[0])


def walk(node, count_fn, overhead, beam, trace, depth=0):
    """returns list of candidate leaves (with path), possibly several under --beam."""
    if "leaf" in node:
        return [(node["leaf"], node.get("ambiguous", False), list(trace))]
    frag = node["fragment"]
    c = count_fn(frag) - overhead
    keys = sorted(int(k) for k in node["branches"])
    trace = trace + [(frag, c, " ".join(map(str, keys)), c in keys)]

    picked = [c] if c in keys else sorted(keys, key=lambda k: abs(k - c))[:1]
    if beam and len(picked) == 1:
        near = sorted(keys, key=lambda k: abs(k - c))
        picked = [picked[0]] + [k for k in near[1:] if abs(k - c) <= 1][:2]

    out = []
    for k in picked:
        out += walk(node["branches"][str(k)], count_fn, overhead, False, trace, depth + 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("RELAY_BASE_URL", "https://api.openai.com/v1"))
    ap.add_argument("--api-key", default=os.environ.get("RELAY_API_KEY") or os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--model", required=True)
    ap.add_argument("--tree", default=TREE)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--beam", action="store_true", help="保留相邻分支，输出所有存活分组")
    ap.add_argument("--repeat", type=int, default=1, help="每条探针采样次数，取中位数")
    args = ap.parse_args()

    with open(args.tree, encoding="utf-8") as f:
        tree = json.load(f)
    print(f"tree: {len(tree['groups'])} groups, depth {tree['depth']}, "
          f"avg {tree['avg_probes']:.2f} probes")

    if args.dry_run:
        counter = local_counter(args.model)
        overhead = 0
        print(f"dry run, local tokenizer = {args.model}")
    else:
        if not args.api_key:
            raise SystemExit("set RELAY_API_KEY, or pass --api-key, or use --dry-run")

        def counter(p):
            return statistics.median([http_count(args.base_url, args.api_key, args.model, p)
                                      for _ in range(args.repeat)])
        overhead = statistics.median([counter(CONTROL) for _ in range(args.repeat)]) - 1
        print(f"endpoint={args.base_url} model={args.model} template overhead ≈ {overhead}")

    leaves = walk(tree["tree"], counter, overhead, args.beam, [])

    # 同一批 trace 只打一次
    seen = set()
    for leaf, amb, trace in leaves:
        for frag, c, keys, ok in trace:
            if frag in seen:
                continue
            seen.add(frag)
            flag = "" if ok else "  ← 不在任何分支上"
            print(f"  probe `{frag}` -> {c} token (branches: {keys}){flag}")

    print()
    best = leaves[0]
    if len(best[0]) == 1:
        print(f"命中 **{best[0][0]}**（{len(best[2])} 次探针）")
    else:
        print(f"落在分组 **{' = '.join(best[0])}**（{len(best[2])} 次探针）——同分词器，无法再细分")
    if best[1]:
        print("注意：该节点分支过宽，可能已被判错，建议用 --beam 复测。")
    if len(leaves) > 1:
        print("其它存活分支：")
        for leaf, _, tr in leaves[1:]:
            print(f"  - {' = '.join(leaf)}（{len(tr)} 次探针）")
    return 0 if len(best[0]) == 1 else 2


if __name__ == "__main__":
    sys.exit(main())
