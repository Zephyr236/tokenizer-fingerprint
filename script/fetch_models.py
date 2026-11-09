#!/usr/bin/env python3
"""Download tokenizer files for the mainstream open-weight model roster.

Maintains data/models.json (the manifest) and data/tokenizers/ (the files).
Every entry lists fallback repos because gated repos 403 on HF and mirrors differ.

    python fetch_models.py            # 只下缺的
    python fetch_models.py --check    # 只探测可用性，不下载
    python fetch_models.py --force    # 重下
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
TOKDIR = os.path.join(DATA, "tokenizers")
MIRRORS = ["https://hf-mirror.com", "https://huggingface.co"]
MANIFEST = os.path.join(DATA, "models.json")

# (label, family, [repo 候选...], 文件名候选)
ROSTER = [
    # ---- DeepSeek ----
    ("deepseek-v3",      "deepseek", ["deepseek-ai/DeepSeek-V3"], ["tokenizer.json"]),
    ("deepseek-v3.1",    "deepseek", ["deepseek-ai/DeepSeek-V3.1"], ["tokenizer.json"]),
    ("deepseek-r1",      "deepseek", ["deepseek-ai/DeepSeek-R1"], ["tokenizer.json"]),
    ("deepseek-v2-lite", "deepseek", ["deepseek-ai/DeepSeek-V2-Lite"], ["tokenizer.json"]),
    ("deepseek-v2.5",    "deepseek", ["deepseek-ai/DeepSeek-V2.5"], ["tokenizer.json"]),
    ("deepseek-coder-v2","deepseek", ["deepseek-ai/DeepSeek-Coder-V2-Lite-Base"], ["tokenizer.json"]),
    ("deepseek-vl2",     "deepseek", ["deepseek-ai/deepseek-vl2"], ["tokenizer.json"]),
    ("deepseek-v4-flash","deepseek", ["deepseek-ai/DeepSeek-V4-Flash"], ["tokenizer.json"]),
    ("deepseek-v4-pro",  "deepseek", ["deepseek-ai/DeepSeek-V4-Pro"], ["tokenizer.json"]),
    ("deepseek-v4.1-flash", "deepseek", ["deepseek-ai/DeepSeek-V4.1-Flash"], ["tokenizer.json"]),
    ("deepseek-coder-6.7b", "deepseek", ["deepseek-ai/deepseek-coder-6.7b-base"], ["tokenizer.json"]),
    # ---- Qwen ----
    ("qwen3-8b",         "qwen", ["Qwen/Qwen3-8B"], ["tokenizer.json"]),
    ("qwen3-30b-a3b",    "qwen", ["Qwen/Qwen3-30B-A3B"], ["tokenizer.json"]),
    ("qwen3-coder-30b",  "qwen", ["Qwen/Qwen3-Coder-30B-A3B-Instruct"], ["tokenizer.json"]),
    ("qwen3-235b-a22b",  "qwen", ["Qwen/Qwen3-235B-A22B"], ["tokenizer.json"]),
    ("qwen3-0.6b",       "qwen", ["Qwen/Qwen3-0.6B"], ["tokenizer.json"]),
    ("qwen3-1.7b",       "qwen", ["Qwen/Qwen3-1.7B"], ["tokenizer.json"]),
    ("qwen3-4b",         "qwen", ["Qwen/Qwen3-4B"], ["tokenizer.json"]),
    ("qwen3-14b",        "qwen", ["Qwen/Qwen3-14B"], ["tokenizer.json"]),
    ("qwen3-32b",        "qwen", ["Qwen/Qwen3-32B"], ["tokenizer.json"]),
    ("qwen3-30b-2507",   "qwen", ["Qwen/Qwen3-30B-A3B-Instruct-2507"], ["tokenizer.json"]),
    ("qwen3-235b-2507",  "qwen", ["Qwen/Qwen3-235B-A22B-Instruct-2507"], ["tokenizer.json"]),
    ("qwen3-coder-480b", "qwen", ["Qwen/Qwen3-Coder-480B-A35B-Instruct"], ["tokenizer.json"]),
    ("qwen3-next-80b",   "qwen", ["Qwen/Qwen3-Next-80B-A3B-Instruct"], ["tokenizer.json"]),
    ("qwen2.5-7b",       "qwen", ["Qwen/Qwen2.5-7B-Instruct"], ["tokenizer.json"]),
    ("qwen2.5-coder-7b", "qwen", ["Qwen/Qwen2.5-Coder-7B-Instruct"], ["tokenizer.json"]),
    ("qwen2.5-72b",      "qwen", ["Qwen/Qwen2.5-72B-Instruct"], ["tokenizer.json"]),
    ("qwen2.5-0.5b",     "qwen", ["Qwen/Qwen2.5-0.5B-Instruct"], ["tokenizer.json"]),
    ("qwen2.5-1.5b",     "qwen", ["Qwen/Qwen2.5-1.5B-Instruct"], ["tokenizer.json"]),
    ("qwen2.5-3b",       "qwen", ["Qwen/Qwen2.5-3B-Instruct"], ["tokenizer.json"]),
    ("qwen2.5-14b",      "qwen", ["Qwen/Qwen2.5-14B-Instruct"], ["tokenizer.json"]),
    ("qwen2.5-32b",      "qwen", ["Qwen/Qwen2.5-32B-Instruct"], ["tokenizer.json"]),
    ("qwen2.5-coder-32b","qwen", ["Qwen/Qwen2.5-Coder-32B-Instruct"], ["tokenizer.json"]),
    ("qwen2.5-vl-72b",   "qwen", ["Qwen/Qwen2.5-VL-72B-Instruct"], ["tokenizer.json"]),
    ("qwen2-vl-7b",      "qwen", ["Qwen/Qwen2-VL-7B-Instruct"], ["tokenizer.json"]),
    ("qwen3-vl-8b",      "qwen", ["Qwen/Qwen3-VL-8B-Instruct"], ["tokenizer.json"]),
    ("qwen3-vl-32b",     "qwen", ["Qwen/Qwen3-VL-32B-Instruct"], ["tokenizer.json"]),
    ("qwen3.5-9b",       "qwen", ["Qwen/Qwen3.5-9B"], ["tokenizer.json"]),
    ("qwen3.5-27b",      "qwen", ["Qwen/Qwen3.5-27B"], ["tokenizer.json"]),
    ("qwen3.6-27b",      "qwen", ["Qwen/Qwen3.6-27B"], ["tokenizer.json"]),
    ("qwen2-7b",         "qwen", ["Qwen/Qwen2-7B-Instruct"], ["tokenizer.json"]),
    ("qwen1.5-7b",       "qwen", ["Qwen/Qwen1.5-7B-Chat"], ["tokenizer.json"]),
    ("qwen-vl",          "qwen", ["Qwen/Qwen2.5-VL-7B-Instruct"], ["tokenizer.json"]),
    # ---- GLM / Zhipu ----
    ("glm-4.5",          "glm", ["zai-org/GLM-4.5"], ["tokenizer.json"]),
    ("glm-4.5-air",      "glm", ["zai-org/GLM-4.5-Air"], ["tokenizer.json"]),
    ("glm-4.5v",         "glm", ["zai-org/GLM-4.5V"], ["tokenizer.json"]),
    ("glm-4.6",          "glm", ["zai-org/GLM-4.6"], ["tokenizer.json"]),
    ("glm-4.1v-9b",      "glm", ["zai-org/GLM-4.1V-9B-Thinking"], ["tokenizer.json"]),
    ("glm-4-9b-0414",    "glm", ["zai-org/GLM-4-9B-0414"], ["tokenizer.json"]),
    ("glm-4-32b-0414",   "glm", ["zai-org/GLM-4-32B-0414"], ["tokenizer.json"]),
    ("glm-z1-9b",        "glm", ["zai-org/GLM-Z1-9B-0414", "THUDM/GLM-Z1-9B-0414"], ["tokenizer.json"]),
    ("glm-z1-rumination-32b", "glm", ["zai-org/GLM-Z1-Rumination-32B-0414"], ["tokenizer.json"]),
    ("glm-5",            "glm", ["zai-org/GLM-5"], ["tokenizer.json"]),
    ("glm-5.1",          "glm", ["zai-org/GLM-5.1"], ["tokenizer.json"]),
    ("glm-5.2",          "glm", ["zai-org/GLM-5.2"], ["tokenizer.json"]),
    ("glm-5.3",          "glm", ["zai-org/GLM-5.3"], ["tokenizer.json"]),
    # ---- Kimi / MiniMax ----
    ("kimi-k2",          "kimi", ["moonshotai/Kimi-K2-Instruct", "moonshotai/Kimi-K2-Base"], ["tokenizer.json", "tiktoken.model"]),
    ("kimi-k2-base",     "kimi", ["moonshotai/Kimi-K2-Base"], ["tokenizer.json", "tiktoken.model"]),
    ("kimi-k2-0905",     "kimi", ["moonshotai/Kimi-K2-Instruct-0905"], ["tokenizer.json", "tiktoken.model"]),
    ("kimi-k2-thinking", "kimi", ["moonshotai/Kimi-K2-Thinking"], ["tokenizer.json", "tiktoken.model"]),
    ("kimi-k2.5",        "kimi", ["moonshotai/Kimi-K2.5"], ["tokenizer.json", "tiktoken.model"]),
    ("kimi-k2.6",        "kimi", ["moonshotai/Kimi-K2.6"], ["tokenizer.json", "tiktoken.model"]),
    ("kimi-k2.7-code",   "kimi", ["moonshotai/Kimi-K2.7-Code"], ["tokenizer.json", "tiktoken.model"]),
    ("kimi-vl-thinking", "kimi", ["moonshotai/Kimi-VL-A3B-Thinking"], ["tokenizer.json", "tiktoken.model"]),
    ("kimi-vl-2506",     "kimi", ["moonshotai/Kimi-VL-A3B-Thinking-2506"], ["tokenizer.json", "tiktoken.model"]),
    ("kimi-linear-48b",  "kimi", ["moonshotai/Kimi-Linear-48B-A3B-Instruct"], ["tokenizer.json", "tiktoken.model"]),
    ("kimi-dev-72b",     "kimi", ["moonshotai/Kimi-Dev-72B"], ["tokenizer.json"]),
    ("kimi-audio-7b",    "kimi", ["moonshotai/Kimi-Audio-7B-Instruct"], ["tokenizer.json", "tiktoken.model"]),
    ("moonlight-16b",    "kimi", ["moonshotai/Moonlight-16B-A3B-Instruct"], ["tokenizer.json", "tiktoken.model"]),
    ("minimax-m1",       "minimax", ["MiniMaxAI/MiniMax-M1-40k"], ["tokenizer.json"]),
    ("minimax-m2",       "minimax", ["MiniMaxAI/MiniMax-M2"], ["tokenizer.json"]),
    # ---- Baidu / Tencent / others CN ----
    ("ernie-4.5",        "ernie", ["baidu/ERNIE-4.5-0.3B-PT"], ["tokenizer.json"]),
    ("hunyuan-7b",       "hunyuan", ["tencent/Hunyuan-7B-Instruct", "tencent/Hunyuan-7B"], ["tokenizer.json", "tokenizer.model"]),
    ("yi-1.5-9b",        "yi", ["01-ai/Yi-1.5-9B-Chat"], ["tokenizer.json"]),
    ("yi-coder-9b",      "yi", ["01-ai/Yi-Coder-9B-Chat"], ["tokenizer.json", "tokenizer.model"]),
    ("baichuan2-7b",     "baichuan", ["baichuan-inc/Baichuan2-7B-Chat"], ["tokenizer.model"]),
    ("internlm2.5-7b",   "internlm", ["internlm/internlm2_5-7b-chat"], ["tokenizer.model", "tokenizer.json"]),
    ("internlm3-8b",     "internlm", ["internlm/internlm3-8b-instruct"], ["tokenizer.json", "tokenizer.model"]),
    ("minicpm4-8b",      "minicpm", ["openbmb/MiniCPM4-8B"], ["tokenizer.json"]),
    ("telechat2",        "telechat", ["tele-ai/TeleChat2-7B"], ["tokenizer.json"]),
    ("yuan2-2b",         "yuan", ["IEITYuan/Yuan2-2B-Moon-32k"], ["tokenizer.json"]),
    ("xverse-7b",        "xverse", ["xverse/XVERSE-7B-Chat"], ["tokenizer.json", "tokenizer.model"]),
    # ---- Meta Llama ----
    ("llama-2-7b",       "llama", ["NousResearch/Llama-2-7b-hf"], ["tokenizer.model"]),
    ("llama-3-8b",       "llama", ["NousResearch/Meta-Llama-3-8B-Instruct"], ["tokenizer.json"]),
    ("llama-3.1-8b",     "llama", ["NousResearch/Meta-Llama-3.1-8B-Instruct", "unsloth/Meta-Llama-3.1-8B"], ["tokenizer.json"]),
    ("llama-3.2-3b",     "llama", ["unsloth/Llama-3.2-3B-Instruct"], ["tokenizer.json"]),
    ("llama-3.3-70b",    "llama", ["unsloth/Llama-3.3-70B-Instruct"], ["tokenizer.json"]),
    ("codellama-7b",     "llama", ["codellama/CodeLlama-7b-hf"], ["tokenizer.json"]),
    ("openchat-3.5",     "llama", ["openchat/openchat-3.5-0106"], ["tokenizer.json"]),
    ("nemotron-70b",     "llama", ["nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"], ["tokenizer.json"]),
    # ---- Mistral ----
    ("mistral-7b-v0.1",  "mistral", ["mistralai/Mistral-7B-v0.1"], ["tokenizer.json", "tokenizer.model"]),
    ("mistral-nemo",     "mistral", ["mistralai/Mistral-Nemo-Base-2407"], ["tokenizer.json"]),
    ("mistral-small-24b","mistral", ["mistralai/Mistral-Small-24B-Instruct-2501"], ["tokenizer.json"]),
    ("mistral-large-2411","mistral", ["mistralai/Mistral-Large-Instruct-2411"], ["tokenizer.json"]),
    ("codestral-22b",    "mistral", ["mistralai/Codestral-22B-v0.1"], ["tokenizer.json"]),
    ("ministral-8b",     "mistral", ["mistralai/Ministral-8B-Instruct-2410"], ["tokenizer.json"]),
    ("mixtral-8x22b",    "mistral", ["mistralai/Mixtral-8x22B-v0.1"], ["tokenizer.json"]),
    ("mixtral-8x7b",     "mistral", ["mistralai/Mixtral-8x7B-v0.1"], ["tokenizer.json"]),
    ("zephyr-7b",        "mistral", ["HuggingFaceH4/zephyr-7b-beta"], ["tokenizer.json"]),
    # ---- Google / Microsoft / others ----
    ("gemma-2-9b",       "gemma", ["unsloth/gemma-2-9b-it", "google/gemma-2-9b-it"], ["tokenizer.json"]),
    ("gemma-3-4b",       "gemma", ["unsloth/gemma-3-4b-it", "google/gemma-3-4b-it"], ["tokenizer.json"]),
    ("phi-3-mini",       "phi", ["microsoft/Phi-3-mini-4k-instruct"], ["tokenizer.json"]),
    ("phi-3.5-mini",     "phi", ["microsoft/Phi-3.5-mini-instruct"], ["tokenizer.json"]),
    ("phi-4",            "phi", ["microsoft/phi-4"], ["tokenizer.json"]),
    ("phi-4-mini",       "phi", ["microsoft/Phi-4-mini-instruct"], ["tokenizer.json"]),
    ("gpt2",             "gpt2", ["openai-community/gpt2"], ["tokenizer.json"]),
    ("falcon-7b",        "falcon", ["tiiuae/falcon-7b"], ["tokenizer.json"]),
    ("olmo-2-7b",        "olmo", ["allenai/OLMo-2-1124-7B"], ["tokenizer.json"]),
    ("starcoder2-3b",    "starcoder", ["bigcode/starcoder2-3b"], ["tokenizer.json"]),
    ("starcoder2-7b",    "starcoder", ["bigcode/starcoder2-7b"], ["tokenizer.json"]),
    ("nemotron-mini",    "nemotron", ["nvidia/Nemotron-Mini-4B-Instruct"], ["tokenizer.json"]),
    ("granite-3.1-8b",   "granite", ["ibm-granite/granite-3.1-8b-instruct"], ["tokenizer.json"]),
    ("jamba-v0.1",       "jamba", ["ai21labs/Jamba-v0.1"], ["tokenizer.json"]),
    ("yi-34b",           "yi", ["01-ai/Yi-34B-Chat"], ["tokenizer.json"]),
    ("hunyuan-a13b",     "hunyuan", ["tencent/Hunyuan-A13B-Instruct"], ["tokenizer.json"]),
    ("ernie-4.5-21b",    "ernie", ["baidu/ERNIE-4.5-21B-A3B-PT"], ["tokenizer.json"]),
    ("kimi-vl",          "kimi", ["moonshotai/Kimi-VL-A3B-Instruct"], ["tokenizer.json", "tiktoken.model"]),
]

EXT = {".json": ".tokenizer.json", ".model": ".tokenizer.model"}


def candidates(label, repo_file):
    name = repo_file
    if name == "tiktoken.model":
        return f"{label}.tiktoken.model"
    return f"{label}{EXT.get(os.path.splitext(name)[1], '.tokenizer.json')}"


def fetch(url, dst, timeout=240):
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    if len(data) < 1000:
        raise ValueError(f"too small ({len(data)}B)")
    with open(dst, "wb") as f:
        f.write(data)
    return len(data)


def handle(item, check_only, force):
    label, family, repos, files = item
    res = {"label": label, "family": family, "file": None, "repo": None, "error": None}
    # 缓存检查要看全部候选文件名，不能只看 files[0]——
    # 有些模型第一候选是 tokenizer.json 但实际只有 tiktoken.model
    if not force:
        for name in files:
            dst = os.path.join(TOKDIR, candidates(label, name))
            if os.path.exists(dst) and os.path.getsize(dst) > 1000:
                res.update(file=os.path.basename(dst), repo="(cached)", size=os.path.getsize(dst))
                return res
    for repo in repos:
        for name in files:
            dst = os.path.join(TOKDIR, candidates(label, name))
            for mirror in MIRRORS:
                url = f"{mirror}/{repo}/resolve/main/{name}"
                try:
                    if check_only:
                        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "curl/8"})
                        with urllib.request.urlopen(req, timeout=60) as r:
                            size = int(r.headers.get("Content-Length") or 0)
                        res.update(file=os.path.basename(dst), repo=repo, size=size)
                        return res
                    size = fetch(url, dst)
                    res.update(file=os.path.basename(dst), repo=repo, size=size)
                    return res
                except Exception as e:
                    res["error"] = f"{repo}/{name}@{mirror.split('//')[1]}: {type(e).__name__}"
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(TOKDIR, exist_ok=True)
    results = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(lambda it: handle(it, args.check, args.force), ROSTER):
            results.append(r)
            ok = "OK  " if r["file"] else "MISS"
            size = f"{r.get('size', 0)/1e6:.1f}MB" if r.get("size") else "-"
            print(f"{ok} {r['label']:<20} {r['family']:<10} {size:<8} {r['repo'] or r['error']}", flush=True)

    ok = [r for r in results if r["file"]]
    print(f"\n{len(ok)}/{len(results)} available, "
          f"{sum(r.get('size', 0) for r in ok)/1e6:.0f} MB")
    if not args.check:
        with open(MANIFEST, "w", encoding="utf-8") as f:
            json.dump(ok, f, ensure_ascii=False, indent=1)
        print(f"manifest -> {MANIFEST}")
    missing = [r["label"] for r in results if not r["file"]]
    if missing:
        print("missing:", ", ".join(missing))


if __name__ == "__main__":
    main()
