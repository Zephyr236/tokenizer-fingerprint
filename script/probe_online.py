#!/usr/bin/env python3
"""Probe a (possibly relayed) OpenAI-compatible endpoint and name the tokenizer behind it.

Idea
----
`usage.prompt_tokens` is an exact token count produced by the serving stack.
Sending a bare probe as the only user message gives

    prompt_tokens(probe) = template_overhead + token_count(probe)

so the *differences* between probes are pure tokenizer output. Compare that
count vector against the codebook in fingerprint.json and the nearest neighbour
is the tokenizer family behind the endpoint.

Usage
-----
    # 1) dry run: simulate a relay locally, no network, no key
    python probe_online.py --dry-run --model deepseek-v3

    # 2) real relay
    set RELAY_API_KEY=sk-xxx
    python probe_online.py --base-url https://relay.example.com/v1 --model gpt-4o

    # 3) which probes / how many
    python probe_online.py --dry-run --model qwen3-8b --set min --repeat 1

Exit code 0 when the best candidate is unique and inside the noise margin.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
FINGERPRINT = os.path.join(HERE, "fingerprint.json")
CONTROL = "a"  # 1 token in every model in the codebook


def load_fingerprint(path):
    with open(path, encoding="utf-8") as f:
        fp = json.load(f)
    return fp


def post_counts(base_url, api_key, model, probe, timeout=60, max_tokens=4):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": probe}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read().decode("utf-8", "replace"))
    usage = payload.get("usage") or {}
    for key in ("prompt_tokens", "input_tokens"):
        if isinstance(usage.get(key), int):
            return usage[key]
    raise RuntimeError(f"no prompt_tokens in response: {json.dumps(payload)[:300]}")


def local_counts(model):
    """dry run: use the local tokenizers from build_fingerprint.py"""
    import build_fingerprint as B
    models = B.load_all()
    if model not in models:
        raise SystemExit(f"unknown local model {model!r}; available: {sorted(models)}")
    tok = models[model]

    def count(s):
        return int(tok.counts([s])[0])
    return count


def measure(probes, count_fn, repeat, verbose=True):
    values = {}
    for i, p in enumerate(probes, 1):
        reps = [count_fn(p) for _ in range(repeat)]
        values[p] = statistics.median(reps)
        if verbose:
            print(f"\r  {i}/{len(probes)} {p!r:<14}", end="", flush=True)
    if verbose:
        print()
    return values


def match(obs, codes):
    """offset-invariant L1: median-centre the residual, then sum |·|."""
    out = {}
    for name, code in codes.items():
        r = [o - c for o, c in zip(obs, code)]
        med = statistics.median(r)
        out[name] = sum(abs(x - med) for x in r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("RELAY_BASE_URL", "https://api.openai.com/v1"))
    ap.add_argument("--api-key", default=os.environ.get("RELAY_API_KEY") or os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--model", required=True, help="model name to send to the endpoint")
    ap.add_argument("--set", default="eng", choices=["eng", "min"], help="eng=48 probes (robust), min=12 probes (cheap)")
    ap.add_argument("--repeat", type=int, default=3, help="samples per probe, median is used")
    ap.add_argument("--dry-run", action="store_true", help="simulate a relay with the local tokenizers")
    ap.add_argument("--fingerprint", default=FINGERPRINT)
    ap.add_argument("--save", default="", help="write the observed vector to this json path")
    args = ap.parse_args()

    fp = load_fingerprint(args.fingerprint)
    key = "engineering" if args.set == "eng" else "minimal"
    probes = fp[key]["probes"]
    codes = fp[key]["codes"]
    margin = fp[key]["min_margin"]
    print(f"fingerprint: {len(codes)} groups, {len(probes)} probes, min L1 margin = {margin}")

    if args.dry_run:
        count_fn = local_counts(args.model)
        overhead = 0
        print(f"dry run, local tokenizer = {args.model}")
    else:
        if not args.api_key:
            raise SystemExit("set RELAY_API_KEY (or OPENAI_API_KEY), or pass --api-key")

        def count_fn(p):
            return post_counts(args.base_url, args.api_key, args.model, p)
        ctl = statistics.median([post_counts(args.base_url, args.api_key, args.model, CONTROL) for _ in range(args.repeat)])
        overhead = ctl - 1  # CONTROL is exactly 1 token
        print(f"endpoint={args.base_url} model={args.model} template overhead ≈ {overhead} tokens")

    all_probes = [CONTROL] + list(probes)
    got = measure(all_probes, count_fn, args.repeat)
    obs = [got[p] - overhead for p in probes]

    scores = match(obs, codes)
    ranked = sorted(scores.items(), key=lambda kv: kv[1])
    best, best_d = ranked[0]
    second, second_d = ranked[1]
    ratio = best_d / second_d if second_d else 0.0

    print("\nobserved token counts :", obs)
    print("nearest candidates:")
    for name, d in ranked[:5]:
        print(f"  {d:8.1f}  {name}")

    if best_d == 0:
        verdict = f"唯一命中 **{best}**（码字完全一致）"
    elif best_d <= max(1.0, margin / 3):
        verdict = f"命中 **{best}**（残差 {best_d:.0f}，在噪声容限内）"
    else:
        verdict = f"最接近 **{best}**（残差 {best_d:.0f}，与次优 {second} 差 {second_d - best_d:.0f}），置信度低"

    print(f"\n{verdict}")
    if best_d > 0 and (second_d - best_d) < 0.25 * max(margin, 1):
        print(f"注意：与 `{second}` 难以区分，建议增加 --repeat 或换 --set eng 复测。")
    if best_d > margin:
        print("注意：残差超过码距，可能是：该模型不在样本内 / 中转层计数不是模型自身的 tokenizer。")

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump({"model": args.model, "set": args.set, "overhead": overhead,
                       "probes": probes, "observed": obs, "ranked": ranked[:10]}, f,
                      ensure_ascii=False, indent=1)
        print(f"saved -> {args.save}")

    return 0 if best_d <= margin else 1


if __name__ == "__main__":
    sys.exit(main())
