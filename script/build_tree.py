#!/usr/bin/env python3
"""Build a difference catalog + decision tree over tokenizers.

Two outputs, both from the same count matrix:

1. `diff_catalog.csv` — 每一个候选片段的分词差异统计：
   哪些分词器把它当成 1 个 token、哪些当成 2 个……按"判别力"排序。
   这就是「先统计有哪些词语的分词差异比较大」的答案。

2. `tree.json` — ID3 决策树：节点是片段，分支是观测到的 token 数，
   叶子是锁定到的分词器分组。线上按树一层层收窄，而不是一次打一堆码字。

Usage: python build_tree.py [--depth 6] [--min-readable-depth 2]
Deps:  tokenizers sentencepiece tiktoken numpy
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_fingerprint as B  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def make_groups(mat, names):
    groups = []
    for i in range(len(names)):
        for g in groups:
            if np.array_equal(mat[i], mat[g[0]]):
                g.append(i)
                break
        else:
            groups.append([i])
    return groups


def partition_entropy(row, groups_idx):
    """entropy of the count partition restricted to groups_idx."""
    v = row[groups_idx]
    _, cnt = np.unique(v, return_counts=True)
    n = len(groups_idx)
    return float(-sum((c / n) * math.log2(c / n) for c in cnt))


def build_tree(GV, cand, groups_idx, depth, cfg, used, readable_flags, pool, path=()):
    """GV: (n_cand, n_groups) counts; cand: list of positions into GV.

    cfg: dict(max_depth, max_arity, readable_depth)
    分支宽度上限 = 抗噪：一条探针切出 18 个桶看起来爽，但计数错 1 就进错桶且无法回头；
    切成 2-4 桶、桶间距离大，才能靠后续探针纠错。
    """
    if len(groups_idx) == 1 or depth >= cfg["max_depth"]:
        return {"leaf": [int(g) for g in groups_idx], "depth": depth}

    parent_h = math.log2(len(groups_idx)) if len(groups_idx) > 1 else 0.0
    best = None
    for j in cand:
        if j in used:
            continue
        if depth < cfg["readable_depth"] and not readable_flags[j]:
            continue
        row = GV[j]
        if np.all(row[groups_idx] == row[groups_idx[0]]):
            continue  # this fragment does not split the node at all
        buckets = {}
        for g in groups_idx:
            buckets.setdefault(int(row[g]), []).append(g)
        if not (2 <= len(buckets) <= cfg["max_arity"]):
            continue
        n = len(groups_idx)
        after = sum((len(b) / n) * math.log2(len(b)) for b in buckets.values() if len(b) > 1)
        gain = parent_h - after
        if gain <= 1e-9:
            continue
        # 同 gain 时优先：可读 > 桶大小更均衡（最小桶更大）> 更短
        balance = min(len(b) for b in buckets.values())
        key = (round(gain, 9), readable_flags[j], balance, -len(pool[j]))
        if best is None or key > best[0]:
            best = (key, gain, readable_flags[j], j, buckets)
    if best is None:
        return {"leaf": [int(g) for g in groups_idx], "depth": depth, "ambiguous": True}

    key, gain, rflag, j, buckets = best
    used = used | {j}
    node = {"fragment": pool[j], "gain": round(gain, 4), "readable": bool(rflag), "branches": {}}
    for val, gs in sorted(buckets.items()):
        node["branches"][str(val)] = build_tree(GV, cand, gs, depth + 1, cfg, used,
                                                readable_flags, pool, path + (j,))
    return node


def tree_stats(node, groups_of_leaf, depth=0):
    """returns list of (leaf_key, path_len) and tree depth."""
    if "leaf" in node:
        return [(tuple(node["leaf"]), depth)], depth
    out, mx = [], depth
    for val, child in node["branches"].items():
        sub, d = tree_stats(child, groups_of_leaf, depth + 1)
        out += sub
        mx = max(mx, d)
    return out, mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--arity", type=int, default=4, help="每个节点最多切几个桶（抗噪）")
    ap.add_argument("--readable-depth", type=int, default=2, help="前几层只允许可读片段")
    ap.add_argument("--top", type=int, default=400, help="catalog rows to export")
    ap.add_argument("--mining-cap", type=int, default=6000, help="每个词表最多挖多少条中文候选")
    ap.add_argument("--verify", type=int, default=8, help="收窄后的验证探针条数")
    args = ap.parse_args()
    cfg = {"max_depth": args.depth, "max_arity": args.arity, "readable_depth": args.readable_depth}

    print("[1/4] loading tokenizers")
    models = B.load_all()
    names = sorted(models)

    print("[2/4] building pool + counting")
    cp = B.corpus_probes()
    vp = B.vocab_probes(models, cap_cjk=args.mining_cap)
    seen, pool = set(), []
    for pl in (cp, vp):
        for s in pl:
            if s not in seen:
                seen.add(s)
                pool.append(s)
    B.POOL = pool
    M = np.zeros((len(names), len(pool)), dtype=np.int16)
    for i, nm in enumerate(names):
        M[i] = models[nm].counts(pool)

    # 分组只在"可作探针的候选"上做：特殊 token 之类不算
    cand = [j for j in range(len(pool))
            if pool[j] and not (pool[j].startswith("<") and pool[j].endswith(">"))
            and "\ufffd" not in pool[j] and pool[j].strip() == pool[j]]
    usable = M[:, cand]
    groups = make_groups(usable, names)
    gname = [" = ".join(names[i] for i in g) for g in groups]
    G = len(groups)
    print(f"      {len(names)} tokenizers -> {G} groups on {len(cand)} usable candidates")

    GV = np.array([[int(M[g[0], j]) for g in groups] for j in cand], dtype=np.int16)
    readable_flags = [1 if B.readable(pool[j]) else 0 for j in cand]
    cand_pool = [pool[j] for j in cand]

    print("[3/4] building decision tree")
    tree = build_tree(GV, list(range(len(cand))), list(range(G)), 0, cfg, frozenset(),
                      readable_flags, cand_pool)
    paths, mx = tree_stats(tree, None)
    depths = [d for _, d in paths]
    sizes = [len(g) for g, _ in paths]
    singletons = sum(1 for s in sizes if s == 1)
    avg_probes = sum(depths) / len(depths)
    worst = max(depths)
    print(f"      depth={mx} leaves={len(paths)} identified={singletons}/{G} "
          f"avg_probes={avg_probes:.2f} worst={worst}")

    # ---- 验证集：树只管收窄，收窄后要再验一遍，否则不在样本里的 tokenizer 会被硬套到某个叶子 ----
    # 优先只用可读文本；某组跟别的组在可读文本上完全一样时（例如 GLM-4.5 与 GLM-5 只差稀有字符），
    # 再放开到全部候选（含 IPA / 稀有字母），否则永远分不开。
    ridx = [k for k in range(len(cand)) if readable_flags[k]]
    pen_r = np.array([-len(cand_pool[k]) * 1e-3 for k in ridx])
    pen_a = np.array([-len(cand_pool[k]) * 1e-3 for k in range(len(cand))])

    def pick(pairs_a, pairs_b, min_margin=3):
        pos, marg = B.select(GV[ridx], pairs_a, pairs_b, args.verify, pen_r)
        idx = [ridx[p] for p in pos]
        if marg.min() < min_margin:                       # 可读集不够分，放开候选范围
            pos, marg = B.select(GV, pairs_a, pairs_b, args.verify, pen_a)
            idx = list(pos)
        return idx, int(np.min(marg))

    al = [(a, b) for a in range(G) for b in range(a + 1, G)]
    pa = np.array([p[0] for p in al])
    pb = np.array([p[1] for p in al])
    vidx, vmargin = pick(pa, pb)
    codes = {gname[g]: [int(GV[j][g]) for j in vidx] for g in range(G)}
    out = {"probes": [cand_pool[j] for j in vidx], "codes": codes,
           "min_margin": vmargin, "per_group": {}}

    # 每个分组再单独挑一套"专打它 vs 其他所有组"的探针：公共验证集对某些分组区分力不够，
    # 会出现外来 tokenizer 8/8 蒙对的情况（实测 Qwen3.5-122B 被误判成 deepseek 组）。
    for g in range(G):
        gp = [p for p in al if g in p]
        gidx, gmarg = pick(np.array([p[0] for p in gp]), np.array([p[1] for p in gp]))
        out["per_group"][gname[g]] = {
            "probes": [cand_pool[j] for j in gidx],
            "codes": {gname[h]: [int(GV[j][h]) for j in gidx] for h in range(G)},
            "min_margin": gmarg,
        }
    with open(os.path.join(HERE, "verify.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    margins = [v["min_margin"] for v in out["per_group"].values()]
    print(f"      验证集 {len(vidx)} 条（码距 {vmargin}）+ "
          f"{G} 套分组专属验证（码距中位数 {sorted(margins)[G // 2]}）-> verify.json")

    # ---- difference catalog ----
    print("[4/4] writing diff_catalog.csv + tree.json")
    rows = []
    for j in range(len(cand)):
        v = GV[j]
        _, cnt = np.unique(v, return_counts=True)
        h = float(-sum((c / G) * math.log2(c / G) for c in cnt))
        if len(cnt) < 2:
            continue
        buckets = {}
        for gi in range(G):
            buckets.setdefault(int(v[gi]), []).append(gname[gi])
        summary = " | ".join(f"{k}token:{len(gs)}" for k, gs in sorted(buckets.items()))
        rows.append((h, len(cnt), len(pool[cand[j]]), pool[cand[j]], readable_flags[j],
                     summary, {str(k): gs for k, gs in sorted(buckets.items())}))
    rows.sort(key=lambda r: (-r[0], -r[4], r[1], r[2]))

    with open(os.path.join(HERE, "diff_catalog.csv"), "w", encoding="utf-8-sig") as f:
        f.write("fragment,len,entropy,distinct_counts,readable,summary,detail_json\n")
        for h, nd, ln, frag, rf, summary, detail in rows[:args.top]:
            js = json.dumps(detail, ensure_ascii=False).replace('"', '""')
            f.write(f'"{frag}",{ln},{h:.3f},{nd},{rf},"{summary}","{js}"\n')

    # tree.json（分组名 + 分支）
    def dump(node):
        if "leaf" in node:
            return {"leaf": [gname[g] for g in node["leaf"]], "ambiguous": node.get("ambiguous", False)}
        return {"fragment": node["fragment"], "readable": node["readable"], "gain": node["gain"],
                "branches": {k: dump(v) for k, v in node["branches"].items()}}
    with open(os.path.join(HERE, "tree.json"), "w", encoding="utf-8") as f:
        json.dump({"groups": gname, "depth": mx, "leaves": len(paths),
                   "identified": singletons, "avg_probes": avg_probes,
                   "counts_of_groups": {gname[g]: [int(GV[j][g]) for j in range(min(12, len(cand)))]
                                        for g in range(G)},
                   "tree": dump(tree)}, f, ensure_ascii=False, indent=1)

    # ---- report ----
    L = ["# 分词差异知识库 + 决策树\n",
         f"- 候选片段 **{len(cand):,}** 条（可用作探针：不含特殊 token、无首尾空白）",
         f"- 分词器 **{len(names)}** 个 → 去重 **{G}** 个分组",
         f"- 决策树深度 **{mx}**，平均 **{avg_probes:.2f}** 次探针定位，能唯一识别 **{singletons}/{G}** 个分组\n",
         "## 1. 树长什么样\n"]

    def render(node, indent=0):
        pad = "  " * indent
        if "leaf" in node:
            tag = "❓ " if node.get("ambiguous") else ""
            return [f"{pad}- → {tag}**{' ; '.join(gname[g] for g in node['leaf'])}**"]
        out = [f"{pad}- 探 `{node['fragment']}`（gain {node['gain']}）"]
        for val, child in node["branches"].items():
            out.append(f"{pad}  - = **{val}** token")
            out += render(child, indent + 2)
        return out
    L += render(tree)

    L.append("\n## 2. 差异最大的片段（按判别力排序，前 40）\n")
    L.append("| 片段 | 熵 | 取值数 | 分布 |")
    L.append("|---|---|---|---|")
    for h, nd, ln, frag, rf, summary, _ in rows[:40]:
        L.append(f"| `{frag}` | {h:.2f} | {nd} | {summary} |")

    L.append("\n## 3. 「1 token vs 多个 token」的典型片段（按 1-token 桶的成员分组）\n")
    two = [r for r in rows if r[1] == 2 and "1" in r[6]]
    by_bucket = {}
    for r in two:
        by_bucket.setdefault(tuple(sorted(r[6]["1"])), []).append(r)
    L.append("| 1 token 的分词器 | 多 token 的分词器 | 例（片段 → 多 token 侧计数） | 碎片数 |")
    L.append("|---|---|---|---|")
    for ones, rs in sorted(by_bucket.items(), key=lambda kv: (-len(kv[1]), -len(kv[0])))[:18]:
        others = sorted({m for r in rs for k, gs in r[6].items() if k != "1" for m in gs})
        ex = "；".join(f"`{r[3]}`→{max(int(k) for k in r[6])}" for r in rs[:3])
        L.append(f"| {', '.join(ones)} | {', '.join(others)} | {ex} | {len(rs)} |")

    L.append("\n完整目录见 `diff_catalog.csv`（可用 Excel 打开，detail_json 列给出每个桶的完整成员）。")
    with open(os.path.join(HERE, "tree_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"      wrote diff_catalog.csv ({min(args.top, len(rows))} rows), tree.json, tree_report.md")


if __name__ == "__main__":
    main()
