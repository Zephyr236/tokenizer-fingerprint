#!/usr/bin/env python3
"""Build the tokenizer fingerprint used to identify the model behind a relay (zhongzhuan).

Pipeline
--------
1. load every tokenizer under data/tokenizers (+ OpenAI tiktoken as reference)
2. build a probe pool: readable corpus n-grams + surfaces mined from every vocab
3. count tokens for every (model, probe) pair -> matrix
4. group models whose count rows are identical -> tokenizer-level zero signal
5. two-phase greedy: cover every distinguishable pair, then maximise the minimum
   L1 code distance (redundancy against noisy counting)
6. jitter-test the codebook under realistic serving-stack noise
7. emit fingerprint.json, report.md, ranked_probes.csv

Usage:  python build_fingerprint.py [--k 32]
Deps:   pip install tokenizers sentencepiece tiktoken numpy
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
TOKDIR = os.path.join(HERE, "data", "tokenizers")

CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
ALLOW = re.compile(
    r"^[\u4e00-\u9fffA-Za-z0-9 ,.:;!?()\[\]{}\"'/\\_+\-=<>@#$%&*|~`"
    r"\u3001\u3002\uff0c\uff1a\uff1b\uff01\uff1f\u201c\u201d\u2018\u2019\u2014\u2026\u00b7]+$"
)

# 语料：中文常用/正式/成语/繁体 + 英文 + 代码 + SQL + URL + 数字 + 全角 + 多语种
CORPUS = (
    "为了降低推理成本，很多团队会从中转站购买模型接口，然后把它当成官方服务来用。问题在于，你无法确认对面跑的是不是它宣称的那个模型。\n"
    "不同的模型使用不同的分词器，而分词器决定了同一句话会被切成多少个 token。这种差异可以通过接口返回的用量字段间接观测。\n"
    "我们设计了一组探针，覆盖中文常用词、成语、数字分组、标点符号、代码缩进、英文大小写以及全角半角字符。\n"
    "经济政策的变化会影响到企业投资、居民消费和出口竞争力，因此需要综合评估。人工智能技术正在快速演进。\n"
    "中国特色社会主义市场经济体制下，宏观调控与市场机制相互配合。传统文化中的仁义礼智信至今仍然影响着人们的日常交往方式。\n"
    "服务器部署在机房中，通过负载均衡把请求分发到多个节点，再用缓存减少数据库压力。\n"
    "分词器的差异主要体现在词表覆盖率和合并规则上。词表越大，单个 token 携带的信息越多，但嵌入矩阵也越大。\n"
    "一二三四五六七八九十，百千万亿兆；甲乙丙丁戊己庚辛壬癸；春夏秋冬，东南西北，上下左右。\n"
    "The quick brown fox jumps over the lazy dog. Tokenization is not the same across model families.\n"
    "def compute_attention(query, key, value, mask=None): return softmax(query @ key.T / sqrt(d)) @ value\n"
    "SELECT user_id, COUNT(*) AS n FROM events WHERE created_at > '2026-01-01' GROUP BY user_id ORDER BY n DESC LIMIT 10;\n"
    "https://api.example.com/v1/chat/completions?model=gpt-4o&stream=true\n"
    "2026年11月7日 14:38:29 UTC+8 版本 v3.1.2-beta 共 1,234,567 条记录，增长了 12.5%。\n"
    "繁體中文與簡體中文在字形上有差異，詞表往往只覆蓋其中一種，另一種需要拆成多個字元。\n"
    "こんにちは、世界。안녕하세요 세계. Привет, мир. مرحبا بالعالم.\n"
    "ｈｅｌｌｏ　ｗｏｒｌｄ　１２３ ＦＵＬＬＷＩＤＴＨ\n"
)

KIMI_PAT = "|".join([
    r"""[\p{Han}]+""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""\p{N}{1,3}""",
    r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
    r"""\s*[\r\n]+""",
    r"""\s+(?!\S)""",
    r"""\s+""",
])


class HFTok:
    kind = "hf"

    def __init__(self, name, path):
        from tokenizers import Tokenizer
        self.name = name
        self.t = Tokenizer.from_file(path)
        self.vocab_size = self.t.get_vocab_size()

    def counts(self, texts):
        return np.array([len(e.ids) for e in self.t.encode_batch(texts, add_special_tokens=False)], dtype=np.int16)

    def surfaces(self):
        out = set()
        for tok in self.t.get_vocab():
            try:
                out.add(bytes(BYTE_DECODER[c] for c in tok).decode("utf-8"))
            except Exception:
                pass
        return out


class SPTok:
    kind = "sp"

    def __init__(self, name, path):
        import sentencepiece as spm
        self.name = name
        # sentencepiece 的 C++ 文件层在 Windows 上打不开含非 ASCII 字符的路径，
        # 先切到该目录，再用纯 ASCII 文件名打开。
        cwd = os.getcwd()
        try:
            os.chdir(os.path.dirname(os.path.abspath(path)) or ".")
            self.t = spm.SentencePieceProcessor(model_file=os.path.basename(path))
        finally:
            os.chdir(cwd)
        self.vocab_size = self.t.get_piece_size()

    def counts(self, texts):
        return np.array([len(x) for x in self.t.encode(texts, out_type=str)], dtype=np.int16)

    def surfaces(self):
        out = set()
        for i in range(self.vocab_size):
            p = self.t.id_to_piece(i)
            if not (p.startswith("<") and p.endswith(">")):
                out.add(p.replace("\u2581", " "))
        return out


class TicTok:
    kind = "tiktoken"

    def __init__(self, name, path=None, enc=None):
        import tiktoken
        self.name = name
        if enc is not None:
            self.t = tiktoken.get_encoding(enc)
        else:
            from tiktoken.load import load_tiktoken_bpe
            self.t = tiktoken.Encoding(name=name, pat_str=KIMI_PAT,
                                       mergeable_ranks=load_tiktoken_bpe(path),
                                       special_tokens={"<|begin_of_text|>": 163584, "<|end_of_text|>": 163585})
        self.vocab_size = self.t.n_vocab

    def counts(self, texts):
        return np.array([len(x) for x in self.t.encode_ordinary_batch(texts)], dtype=np.int16)

    def surfaces(self):
        out = set()
        for b in self.t._mergeable_ranks:
            try:
                out.add(b.decode("utf-8"))
            except Exception:
                pass
        return out


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


BYTE_DECODER = {v: k for k, v in bytes_to_unicode().items()}


def load_all():
    models = {}
    for path in sorted(glob.glob(os.path.join(TOKDIR, "*.tokenizer.json"))):
        name = os.path.basename(path).split(".tokenizer.json")[0]
        models[name] = HFTok(name, path)
    for path in sorted(glob.glob(os.path.join(TOKDIR, "*.tokenizer.model"))):
        name = os.path.basename(path).split(".tokenizer.model")[0]
        models[name] = SPTok(name, path)
    for path in sorted(glob.glob(os.path.join(TOKDIR, "*.tiktoken.model"))):
        name = os.path.basename(path).split(".tiktoken.model")[0]
        models[name] = TicTok(name, path)
    try:
        models["openai-cl100k"] = TicTok("openai-cl100k", enc="cl100k_base")
        models["openai-o200k"] = TicTok("openai-o200k", enc="o200k_base")
    except Exception as e:  # offline: skip reference encodings
        print("tiktoken reference unavailable:", e, file=sys.stderr)
    return models


def corpus_probes():
    probes = set()
    for line in CORPUS.splitlines():
        line = line.strip()
        if not line:
            continue
        for n in (2, 3, 4, 5):
            for i in range(len(line) - n + 1):
                probes.add(line[i:i + n])
    for m in re.finditer(r"[A-Za-z][A-Za-z0-9_./:@\-]{3,}", CORPUS):
        s = m.group(0)
        probes.add(s)
        for i in range(len(s) - 3):
            probes.add(s[i:i + 4])
    for m in re.finditer(r"[\d.,:%+\-]{3,}", CORPUS):
        probes.add(m.group(0))
    probes.update(["    ", "\t", "\u2026\u2026", "\u2014\u2014", "\uff01\uff01\uff01", "\u00b7", "\u203b"])
    return sorted(probes)


def readable(s: str) -> bool:
    # 两端不能有空白：很多 chat 模板会 strip，探针必须保持原样
    if not (2 <= len(s) <= 12) or s != s.strip() or not ALLOW.match(s):
        return False
    return bool(CJK.search(s)) or len(s) >= 4


def vocab_probes(models, cap_cjk=15000, cap_other=2000):
    pool = set()
    for m in models.values():
        try:
            surf = {s for s in m.surfaces()
                    if 2 <= len(s) <= 10 and not any(c in s for c in "\x00\ufffd\n\r")}
        except Exception:
            continue
        cjk = sorted((s for s in surf if CJK.search(s)), key=lambda x: (len(x), x))
        oth = sorted((s for s in surf if not CJK.search(s)), key=lambda x: (len(x), x))
        pool.update(cjk[:cap_cjk])
        pool.update(oth[:cap_other])
    return sorted(pool)


def select(V, pa, pb, k, lenpen):
    """two-phase greedy: (1) cover pairs still at distance 0, (2) maximise min L1."""
    dif = np.abs(V[:, pa] - V[:, pb])
    cur = np.zeros(dif.shape[1], dtype=np.float32)
    avail = np.ones(len(V), dtype=bool)
    chosen = []
    for _ in range(k):
        zero = cur == 0
        if zero.any():
            score = np.where(avail, (dif[:, zero] > 0).sum(axis=1) * 1000.0, -1)
        else:
            score = np.where(avail, (cur[None, :] + dif).min(axis=1), -1) * 1000.0
        j = int(np.argmax(score + lenpen))
        if not avail[j]:
            break
        chosen.append(j)
        cur = cur + dif[j]
        avail[j] = False
    return chosen, cur


def match(obs, codes):
    """offset-invariant L1: median-centre the residual before summing."""
    out = []
    for c in codes:
        r = obs - c
        out.append(np.abs(r - np.median(r)).sum())
    return np.array(out)


def jitter_test(codes, mode, trials=4000, seed=11):
    rnd = random.Random(seed)
    C = np.array(codes, dtype=float)
    G, K = C.shape
    hit = 0
    per = [[0, 0] for _ in range(G)]
    for _ in range(trials):
        g = rnd.randrange(G)
        obs = C[g].copy()
        if mode == "const":
            obs += rnd.choice([-3, -2, -1, 1, 2, 3])
        elif mode == "sporadic":
            for _ in range(rnd.randint(1, 2)):
                i = rnd.randrange(K)
                obs[i] = max(1, obs[i] + rnd.choice([-1, 1]))
        elif mode == "dense":
            obs = np.maximum(1, obs + np.array([rnd.choice([-1, 0, 1]) for _ in range(K)]))
        d = match(obs, C)
        pred = int(np.argmin(d))
        ok = pred == g and int((d == d[pred]).sum()) == 1
        hit += ok
        per[g][0] += ok
        per[g][1] += 1
    return hit / trials, per


def md_table(headers, rows):
    out = ["| " + " | ".join(map(str, headers)) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=48, help="engineering probe count")
    ap.add_argument("--mining-cap", type=int, default=15000, help="CJK probes mined per vocab")
    args = ap.parse_args()

    print("[1/5] loading tokenizers")
    models = load_all()
    names = sorted(models)
    print(f"      {len(names)} tokenizers")

    print("[2/5] building probe pool")
    cp = corpus_probes()
    vp = vocab_probes(models, cap_cjk=args.mining_cap)
    seen, pool = set(), []
    for pl in (cp, vp):
        for s in pl:
            if s not in seen:
                seen.add(s)
                pool.append(s)
    print(f"      corpus={len(cp)} mined={len(vp)} union={len(pool)}")

    print("[3/5] counting")
    M = np.zeros((len(names), len(pool)), dtype=np.int16)
    for i, nm in enumerate(names):
        M[i] = models[nm].counts(pool)

    def make_groups(mat):
        gs = []
        for i in range(len(names)):
            for g in gs:
                if np.array_equal(mat[i], mat[g[0]]):
                    g.append(i)
                    break
            else:
                gs.append([i])
        return gs

    def gv(mat, gs):
        return np.array([[int(mat[g[0], j]) for g in gs] for j in range(mat.shape[1])], dtype=np.float32)

    ridx = [j for j in range(len(pool)) if readable(pool[j])]
    Mr = M[:, ridx]

    # 两套分组：全池分组（含特殊 token 差异）与「正文可读」分组（不含）
    groups_full = make_groups(M)
    groups = make_groups(Mr)
    gname_full = [" = ".join(names[i] for i in g) for g in groups_full]
    gname = [" = ".join(names[i] for i in g) for g in groups]
    G_full, G = len(groups_full), len(groups)
    print(f"      {G_full} distinct on full pool / {G} distinct on readable text")
    hidden = [gname_full[k] for k, g in enumerate(groups_full)
              if not any(set(g) <= set(h) for h in groups)]

    GVr = gv(Mr, groups)
    GVf = gv(M, groups_full)

    print("[4/5] selecting probes")
    pr = [(a, b) for a in range(G) for b in range(a + 1, G)]
    pa = np.array([p[0] for p in pr])
    pb = np.array([p[1] for p in pr])
    eng_pos, eng_marg = select(GVr, pa, pb, args.k, np.array([-len(pool[j]) * 1e-3 for j in ridx]))
    min_pos, min_marg = select(GVr, pa, pb, 12, np.array([-len(pool[j]) * 1e-3 for j in ridx]))
    eng = [ridx[p] for p in eng_pos]
    minset = [ridx[p] for p in min_pos]
    codes_eng = [[int(M[groups[g][0], j]) for j in eng] for g in range(G)]
    codes_min = [[int(M[groups[g][0], j]) for j in minset] for g in range(G)]

    print("[5/5] noise test + report")
    modes = ["none", "const", "sporadic", "dense"]
    acc = {(t, m): jitter_test(cs, m) for t, cs in (("eng", codes_eng), ("min", codes_min)) for m in modes}

    dd = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            dv = np.abs(M[i].astype(float) - M[j].astype(float))
            dn = np.maximum(M[i], M[j]).astype(float)
            dd.append((float(np.mean(dv / dn)), names[i], names[j]))
    dd.sort()

    stat = []
    for k, j in enumerate(ridx):
        v = GVr[k]
        _, cnt = np.unique(v, return_counts=True)
        h = float(-sum((c / G) * math.log2(c / G) for c in cnt))
        stat.append((h, int(cnt.max()), len(pool[j]), pool[j], v))
    stat.sort(key=lambda x: (-x[0], x[1], x[2]))

    conf = sorted((int(np.abs(GVr[:, a] - GVr[:, b]).sum()), gname[a], gname[b])
                  for a in range(G) for b in range(a + 1, G))

    L = ["# 从 tokenizer 视角识别中转站模型：实测报告\n",
         f"- 样本 **{len(names)}** 个 tokenizer → 去重后 **{G}** 个独立分词器",
         f"- 探针池 **{len(pool)}** 条（语料 n-gram {len(cp)} + 词表挖掘 {len(vp)}）",
         f"- 工程探针集 **{len(eng)}** 条，最小 L1 码距 **{int(eng_marg.min())}**\n",
         "## 1. 样本\n",
         md_table(["模型", "vocab", "实现", "分组"],
                  [[nm, models[nm].vocab_size, models[nm].kind,
                    next(gname[gi] for gi, g in enumerate(groups) if i in g)] for i, nm in enumerate(names)]),
         "\n## 2. 分词完全一致的分组（tokenizer 零信号）\n"]
    for g in groups:
        if len(g) > 1:
            L.append(f"- **{' = '.join(names[i] for i in g)}**")
    if hidden:
        L.append("\n以下几组在正文分词上完全一致，差异只存在于特殊 token（可读探针无法区分）：")
        for h in hidden:
            L.append(f"- **{h}**")
    gvec = {j: GVr[k] for k, j in enumerate(ridx)}
    L += ["\n## 3. 工程探针集（可读文本）\n",
          md_table(["#", "探针", "len", "各分组 token 数"],
                   [[i + 1, f"`{pool[j]}`", len(pool[j]), " / ".join(str(int(x)) for x in gvec[j])]
                    for i, j in enumerate(eng)]),
          "\n**codebook**\n",
          md_table(["分组"] + [f"p{i+1}" for i in range(len(eng))],
                   [[gname[g]] + [str(v) for v in codes_eng[g]] for g in range(G)]),
          "\n## 4. 抗噪表现\n",
          md_table(["探针集", "无噪声", "常数偏移 ±3", "个别探针 ±1", "全坐标 ±1"],
                   [["工程集 %d 条（码距 %d）" % (len(eng), int(eng_marg.min()))]
                    + [f"{acc[('eng', m)][0]*100:.1f}%" for m in modes],
                    ["最小集 %d 条（码距 %d）" % (len(minset), int(min_marg.min()))]
                    + [f"{acc[('min', m)][0]*100:.1f}%" for m in modes]]),
          "\n说明：无噪声＝中转层直接用模型自身 tokenizer 计数（最常见）；常数偏移＝chat 模板口径差异，"
          "匹配时用中位数去偏移即可消除；个别探针 ±1＝估算器舍入；全坐标 ±1＝中转层用近似估算器，"
          "这时 tokenizer 指纹本身失效，需要换成基于 logprobs / 拒绝分布的行为探针。",
          "\n## 5. 单探针判别力（可读候选，按熵）\n",
          md_table(["探针", "熵(bit)", "最大同值组数", "len", "各分组 token 数"],
                   [[f"`{s[3]}`", f"{s[0]:.2f}", s[1], s[2], " / ".join(str(int(x)) for x in s[4])]
                    for s in stat[:25]]),
          "\n## 6. 最易混淆的分组对\n",
          md_table(["可读池累计 L1", "分组 A", "分组 B"], [[c, a, b] for c, a, b in conf[:10]]),
          "\n## 7. 两两差异（相对 count 差）\n",
          f"- 最大：`{dd[-1][1]}` vs `{dd[-1][2]}` = **{dd[-1][0]:.3f}**"]
    nz = next(k for k, v in enumerate(dd) if v[0] > 0)
    L += [f"- 最小非零：`{dd[nz][1]}` vs `{dd[nz][2]}` = **{dd[nz][0]:.5f}**（仅特殊 token 不同，正文完全一致）",
          f"- 完全相同：{sum(1 for x in dd if x[0] == 0)} 对（{len(names) - G} 个模型与同族兄弟位级相同）\n",
          "## 8. 用法\n",
          "见 `probe_online.py`：对任意 OpenAI 兼容端点发送探针，按 `prompt_tokens` 反推码字，",
          "用去偏移 L1 最近邻比对 `fingerprint.json` 的 codebook。"]

    with open(os.path.join(HERE, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    with open(os.path.join(HERE, "fingerprint.json"), "w", encoding="utf-8") as f:
        json.dump({"models": names, "groups": gname,
                   "engineering": {"probes": [pool[j] for j in eng],
                                   "codes": {gname[g]: codes_eng[g] for g in range(G)},
                                   "min_margin": int(eng_marg.min())},
                   "minimal": {"probes": [pool[j] for j in minset],
                               "codes": {gname[g]: codes_min[g] for g in range(G)},
                               "min_margin": int(min_marg.min())},
                   "accuracy": {f"{t}_{m}": acc[(t, m)][0] for t in ("eng", "min") for m in modes}},
                  f, ensure_ascii=False, indent=1)
    with open(os.path.join(HERE, "ranked_probes.csv"), "w", encoding="utf-8-sig") as f:
        f.write("fragment,len,entropy,ncat,max_share," + ",".join(gname) + "\n")
        for s in stat:
            f.write('"' + s[3].replace('"', '""') + f'",{s[2]},{s[0]:.3f},{len(set(int(x) for x in s[4]))},{s[1]},'
                    + ",".join(str(int(x)) for x in s[4]) + "\n")
    print(f"      done: {G} groups, eng={len(eng)} margin={int(eng_marg.min())} "
          f"acc_none={acc[('eng','none')][0]:.3f} acc_dense={acc[('eng','dense')][0]:.3f}")


if __name__ == "__main__":
    main()
