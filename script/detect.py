#!/usr/bin/env python3
"""判断一个 OpenAI 兼容端点背后的模型，是不是你声明的那一个。

    python detect.py --list
    python detect.py --base-url https://api.xxx.com/v1 --api-key sk-xxx --model gpt-4o
    python detect.py ... --model Qwen/Qwen3-8B --expect qwen3-8b
    python detect.py --self-test                 # 本地 mock，验证流水线

原理：tokenizer 决定一段文本是几个 token。只发纯片段、什么都不问，读 usage.prompt_tokens，
减掉模板基线，就得到该文本在这个分词器里的 token 数。几条探针下来就能把 tokenizer 家族锁定，
再和"你声明的模型"所属家族对比。

只发 3-4 次请求（决策树），不会真的让模型干活。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.join(HERE, "tree.json")
MANIFEST = os.path.join(HERE, "data", "models.json")
CONTROL = "a"  # 所有样本里都是 1 个 token，用来扣模板基线


# ---------------------------------------------------------------- endpoint
class Endpoint:
    def __init__(self, base_url, api_key, model, timeout=60, repeat=1, retries=5, min_interval=1.5):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.repeat = repeat
        self.retries = retries
        self.min_interval = min_interval
        self.calls = 0
        self._last = 0.0

    def _pace(self):
        gap = time.time() - self._last
        if gap < self.min_interval:
            time.sleep(self.min_interval - gap)
        self._last = time.time()

    def _post(self, text):
        self._pace()
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": 1,
            "temperature": 0,
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=body, method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        self.calls += 1
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))

    def prompt_tokens(self, text):
        """429 / 5xx / 网络抖动一律退避重试。中转站常见按 TPM 限流，退避要给足。"""
        last = None
        for attempt in range(self.retries):
            try:
                payload = self._post(text)
                break
            except urllib.error.HTTPError as e:
                last = e
                if e.code in (400, 401, 403, 404):
                    raise
                time.sleep(min(5 * 2 ** attempt, 60))
            except Exception as e:
                last = e
                time.sleep(min(3 * 2 ** attempt, 30))
        else:
            raise last if last else RuntimeError("请求失败")
        usage = payload.get("usage") or {}
        for key in ("prompt_tokens", "input_tokens"):
            if isinstance(usage.get(key), int):
                return usage[key]
        raise RuntimeError(f"响应里没有 prompt_tokens：{json.dumps(payload, ensure_ascii=False)[:300]}")

    def count(self, text):
        vals = [self.prompt_tokens(text) for _ in range(self.repeat)]
        return statistics.median(vals)

    def baseline(self):
        """模板开销：控制串固定 1 token，多出来的就是模板"""
        ctl = self.count(CONTROL)
        return ctl - 1


# ---------------------------------------------------------------- tree walk
def walk(node, endpoint, overhead, beam=False, trace=None, depth=0):
    trace = trace or []
    if "leaf" in node:
        return [(node["leaf"], node.get("ambiguous", False), list(trace))]
    frag = node["fragment"]
    obs = int(round(endpoint.count(frag) - overhead))
    keys = sorted(int(k) for k in node["branches"])
    hit = obs in keys
    trace = trace + [(frag, obs, keys, hit)]

    picks = [obs] if hit else sorted(keys, key=lambda k: abs(k - obs))[:1]
    if beam:
        near = sorted(keys, key=lambda k: abs(k - obs))
        picks = picks + [k for k in near[1:] if abs(k - obs) <= 1][:2]

    out = []
    for k in picks:
        out += walk(node["branches"][str(k)], endpoint, overhead, False, trace, depth + 1)
    return out


class LocalEndpoint:
    """自测用：直接用本地 tokenizer 当"服务器"，不需要网络和 key"""

    def __init__(self, label, models, overhead=0):
        self.label = label
        self.tok = models[label]
        self.overhead = overhead
        self.calls = 0

    def count(self, text):
        self.calls += 1
        return int(self.tok.counts([text])[0]) + self.overhead

    def baseline(self):
        return self.overhead


# ---------------------------------------------------------------- helpers
def load_manifest():
    if not os.path.exists(MANIFEST):
        return {}
    with open(MANIFEST, encoding="utf-8") as f:
        return {m["label"]: m for m in json.load(f) if m.get("label")}


def load_verify():
    path = os.path.join(HERE, "verify.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def group_of(label, groups):
    for g in groups:
        if label in [x.strip() for x in g.split("=")]:
            return g
    return None


def show_list(tree, manifest):
    groups = tree["groups"]
    print(f"可检测：{len(groups)} 个 tokenizer 家族"
          f"（{sum(len(g.split('=')) for g in groups)} 个模型共享，同族内部无法区分）\n")
    for i, g in enumerate(sorted(groups), 1):
        labels = [x.strip() for x in g.split("=")]
        fams = {(manifest.get(l) or {}).get("family") or ("openai" if l.startswith("openai-") else "?")
                for l in labels}
        print(f"{i:>2}. [{','.join(sorted(fams))}]  {' = '.join(labels)}")
    print("\n提示：同一行里的模型 tokenizer 完全相同，只能判定到这一行为止。")


def verdict(leaves, expect, tree):
    """leaves: [([组名, ...], ambiguous, trace), ...]；组名形如 "a = b = c"。"""
    obs_list = leaves[0][0]
    obs_names = [x.strip() for name in obs_list for x in name.split("=")]
    obs_disp = " | ".join(obs_list)
    if expect:
        if group_of(expect, tree["groups"]) is None:
            return ("unknown", f"列表里没有 {expect}，先用 --list 看支持的模型")
        if expect in obs_names:
            return ("match", f"符合：{expect} 与观测到的分词器属于同一组（{obs_disp}）")
        return ("mismatch", f"不符：声明的是 {expect}，但观测到的分词器是 {obs_disp}")
    return ("observed", f"观测到的 tokenizer：{obs_disp}")


def main():
    ap = argparse.ArgumentParser(description="检测中转站背后的真实模型")
    ap.add_argument("--base-url", default=os.environ.get("RELAY_BASE_URL", ""))
    ap.add_argument("--api-key", default=os.environ.get("RELAY_API_KEY") or os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--model", default="", help="发给端点的模型名（中转站上的名字）")
    ap.add_argument("--expect", default="", help="你声明/怀疑的模型 label，例如 qwen3-8b")
    ap.add_argument("--list", action="store_true", help="列出支持检测的模型")
    ap.add_argument("--beam", action="store_true", help="计数异常时保留相邻分支")
    ap.add_argument("--repeat", type=int, default=1, help="每条探针采样次数，取中位数")
    ap.add_argument("--verify", type=int, default=8, help="收窄后用几条验证探针确认（0=不验证）")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--json", action="store_true", help="输出 json")
    ap.add_argument("--self-test", action="store_true", help="用本地 tokenizer 模拟端点，验证流水线")
    ap.add_argument("--tree", default=TREE)
    args = ap.parse_args()

    with open(args.tree, encoding="utf-8") as f:
        tree = json.load(f)
    manifest = load_manifest()

    if args.list:
        show_list(tree, manifest)
        return 0

    if args.self_test:
        import build_fingerprint as B
        models = B.load_all()
        labels = [x.strip() for g in tree["groups"] for x in g.split("=") if x.strip() in models]
        print(f"自测：{len(labels)} 个模型，模拟模板开销 3 tokens\n")
        ok = bad = 0
        for label in sorted(labels):
            ep = LocalEndpoint(label, models, overhead=3)
            leaves = walk(tree["tree"], ep, overhead=ep.baseline(), beam=False)
            obs_list = leaves[0][0]
            names = [x.strip() for name in obs_list for x in name.split("=")]
            hit = label in names
            ok += hit
            bad += not hit
            mark = "✓" if hit else "✗"
            print(f"{mark} {label:<22} -> {' | '.join(obs_list)}")
        print(f"\n{ok}/{ok+bad} 命中")
        return 0 if bad == 0 else 1

    if not (args.base_url and args.model):
        ap.print_help()
        return 2
    if not args.api_key:
        print("缺少 api key：--api-key 或环境变量 RELAY_API_KEY", file=sys.stderr)
        return 2

    ep = Endpoint(args.base_url, args.api_key, args.model, timeout=args.timeout, repeat=args.repeat)
    try:
        overhead = ep.baseline()
        print(f"端点 {args.base_url}  模型 {args.model}  模板基线 {overhead} tokens")
        leaves = walk(tree["tree"], ep, overhead, beam=args.beam)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:300]
        print(f"HTTP {e.code}: {body}", file=sys.stderr)
        return 3
    except Exception as e:
        print(f"请求失败: {type(e).__name__}: {e}", file=sys.stderr)
        return 3

    leaf, ambiguous, trace = leaves[0]
    print("\n收窄过程：")
    for frag, obs, keys, hit in trace:
        flag = "" if hit else "   ← 不在任何分支上"
        print(f"  {frag!r:<14} -> {obs} token (分支 {keys}){flag}")

    # ---- 验证：树只管收窄，叶子未必是真的 ----
    # 优先用"这一组专属"的验证探针（对它 vs 其他所有组区分力最强），
    # 公共那套对各组区分力不够，实测会让外来 tokenizer 蒙对。
    ver = load_verify()
    matches, verify_rows = {}, []
    if ver:
        cand = leaf[0] if len(leaf) == 1 else None
        gset = (ver.get("per_group", {}) or {}).get(cand) if cand else None
        use = gset or ver
        vprobes = use["probes"][:args.verify]
        obs_vec = []
        for p in vprobes:
            c = int(round(ep.count(p) - overhead))
            obs_vec.append(c)
            verify_rows.append((p, c))
        for gname, code in use["codes"].items():
            matches[gname] = sum(1 for a, b in zip(obs_vec, code[:len(vprobes)]) if a == b)
        ranked = sorted(matches.items(), key=lambda kv: -kv[1])
        kind_set = "分组专属" if gset else "公共"
        print("\n验证 %d 条（%s集，命中数 / %d）：" % (len(vprobes), kind_set, len(vprobes)))
        for gname, m in ranked[:3]:
            tag = "  ← 本轮收窄的候选" if gname in leaf else ""
            print(f"  {m}/{len(vprobes)}  {gname}{tag}")
    else:
        ranked = [(leaf[0], 0)]

    best_group, best_hit = ranked[0]
    second_hit = ranked[1][1] if len(ranked) > 1 else 0
    gap = best_hit - second_hit
    total = len(ver["probes"][:args.verify]) if ver else 0
    if total:
        # 全中才算确认；只差一条时必须明显领先第二名，否则可能只是小整数碰撞
        if best_hit == total and gap >= 1:
            conf = "confirmed"
        elif best_hit >= total - 1 and gap >= 2:
            conf = "likely"
        else:
            conf = "unknown"
    else:
        conf = "unverified"

    # ---- 判定 ----
    kind, msg = "unknown", ""
    observed_group = best_group if conf in ("confirmed", "likely") else " = ".join(leaf)
    if conf == "unknown":
        msg = (f"无法确认：最接近的是 {best_group}，但只命中 {best_hit}/{total}"
               f"（第二名 {ranked[1][0] if len(ranked) > 1 else '-'} {second_hit}/{total}）。"
               f"这个 tokenizer 不在样本里，或者端点不是用模型自身的 tokenizer 计数。")
        kind = "unknown"
    elif args.expect:
        if group_of(args.expect, tree["groups"]) is None:
            kind, msg = "unknown", f"列表里没有 {args.expect}，先用 --list 看支持的模型"
        elif args.expect in [x.strip() for name in best_group.split(" | ") for x in name.split("=")]:
            kind, msg = "match", f"符合：{args.expect} 与观测到的分词器属于同一组（{best_group}）"
        else:
            kind = "mismatch"
            msg = f"不符：声明的是 {args.expect}，但观测到的分词器是 {best_group}"
    else:
        kind, msg = "observed", f"观测到的 tokenizer：{best_group}"

    if conf == "likely":
        msg += "（有 2 条计数对不上，可能是端点计数抖动，建议 --repeat 3 复测）"

    print(f"\n判定：{msg}")
    n_in = sum(len(x.split("=")) for x in best_group.split(" | ")) if conf != "unknown" else 0
    if n_in > 1:
        print(f"注意：这一组内 {n_in} 个模型的 tokenizer 完全相同，无法再细分。")
    if ambiguous:
        print("注意：该节点分支较宽，可能已经走错，建议加 --beam 复测。")
    if len(leaves) > 1:
        print("其它存活分支：" + "；".join(" = ".join(l) for l, _, _ in leaves[1:]))
    if not all(t[3] for t in trace):
        print("注意：有计数没有落在任何分支上，说明端点计数口径异常"
              "（可能不是模型自身 tokenizer，而是估算器）。")
    print(f"\n共 {ep.calls} 次请求")

    if args.json:
        print(json.dumps({"model": args.model, "expect": args.expect, "verdict": kind,
                          "confidence": conf, "observed": observed_group,
                          "message": msg, "verify_hits": ranked[:3],
                          "verify": [{"probe": p, "count": c} for p, c in verify_rows],
                          "trace": [{"probe": f, "count": o, "branches": k, "hit": h}
                                    for f, o, k, h in trace],
                          "requests": ep.calls}, ensure_ascii=False, indent=1))
    return 0 if kind in ("match", "observed") else 1


if __name__ == "__main__":
    sys.exit(main())
