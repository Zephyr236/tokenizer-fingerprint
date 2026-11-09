#!/usr/bin/env python3
"""Export a plain token-count table: one row per prompt, one column per tokenizer.

    prompt | len | spread | distinct | <each tokenizer's token count>

Sorted by spread (max-min) descending, so the fragments whose tokenization differs
the most float to the top. No derived scores, no bucket summaries -- just counts.

Usage: python make_token_table.py [--max-rows 0]
Deps:  tokenizers sentencepiece tiktoken numpy
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_fingerprint as B  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# 词表里挖出来的高频碎片大量是 SEO/垃圾站残片，区分度高但不适合当提示词。
# 这里按关键词挡掉博彩/色情类，另外挡掉"整体是短串重复"和"同一字符连出 3 次"的碎片。
JUNK_WORDS = (
    "彩票", "快三", "时时", "博彩", "彩神", "大发", "申博", "赛车", "波胆", "投注", "开奖",
    "提现", "六合", "合彩", "百家乐", "老虎机", "棋牌", "娱乐城", "娱乐场", "开户",
    "无码", "毛片", "激情", "情色", "三级", "中文字幕", "影音先锋", "先锋影音",
    "久久", "少妇", "色情", "裸", "偷拍", "自拍", "大片", "一级", "特黄", "主人",
    "视频", "在线", "播放", "观看", "免费", "综合",
    # 新闻/站点模板残片
    "新浪", "搜狐", "版权", "免责", "转载", "责任编辑", "图片来源于", "原文",
)


def is_repetitive(s: str) -> bool:
    """亿亿亿亿亿亿亿亿 / 基督教基督教基督教 这类整体是短串重复的碎片。"""
    n = len(s)
    for period in range(1, min(4, n - 1) + 1):   # 注意 period 必须小于 n，否则恒真
        if n % period == 0 and s == s[:period] * (n // period):
            return True
    if any(s[i] == s[i + 1] == s[i + 2] for i in range(n - 2)):
        return True
    return False


def is_clean(s: str) -> bool:
    if any(w in s for w in JUNK_WORDS):
        return False
    if is_repetitive(s):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=0, help="0 = 全部导出")
    ap.add_argument("--keep-junk", action="store_true", help="保留博彩/SEO 残片和重复字符片段")
    ap.add_argument("--out", default=os.path.join(HERE, "token_counts.csv"))
    args = ap.parse_args()

    print("[1/4] loading tokenizers")
    models = B.load_all()
    names = sorted(models)

    print("[2/4] collecting prompts")
    corpus = {s for s in B.corpus_probes() if B.readable(s)}
    mined = {s for s in B.vocab_probes(models) if B.readable(s)}
    total = len(corpus | mined)
    if not args.keep_junk:
        corpus = {s for s in corpus if is_clean(s)}
        mined = {s for s in mined if is_clean(s)}
        print(f"      {total} readable -> {len(corpus | mined)} after dropping junk "
              f"({total - len(corpus | mined)} removed)")
    prompts = sorted(corpus | mined, key=lambda s: (len(s), s))
    src = ["两者" if (s in corpus and s in mined) else ("语料" if s in corpus else "词表挖掘")
           for s in prompts]
    print(f"      {len(prompts)} prompts "
          f"(语料 {sum(1 for x in src if x == '语料')}, 词表挖掘 {sum(1 for x in src if x == '词表挖掘')})")

    print("[3/4] counting")
    M = np.zeros((len(names), len(prompts)), dtype=np.int16)
    for i, nm in enumerate(names):
        M[i] = models[nm].counts(prompts)

    print("[4/4] writing csv")
    spread = M.max(axis=0).astype(int) - M.min(axis=0).astype(int)
    distinct = np.array([len(set(M[:, j].tolist())) for j in range(len(prompts))])
    # 主序：spread 降序（差异最大的排最前），次序：取值数降序，末序：片段长度升序
    order = np.lexsort((np.array([len(p) for p in prompts]), -distinct, -spread))

    rows = args.max_rows if args.max_rows > 0 else len(order)
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prompt", "len", "spread", "distinct", "source"] + names)
        for k in order[:rows]:
            p = prompts[k]
            w.writerow([p, len(p), int(spread[k]), int(distinct[k]), src[k]]
                       + [int(x) for x in M[:, k]])
    print(f"      {rows} rows -> {args.out}")
    print(f"      spread>0: {int((spread > 0).sum())}, spread>=2: {int((spread >= 2).sum())}, "
          f"distinct>=5: {int((distinct >= 5).sum())}")


if __name__ == "__main__":
    main()
