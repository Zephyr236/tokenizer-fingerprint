# tokenizer-fingerprint

用 tokenizer 指纹判断中转站（第三方 API 转发）背后到底是什么模型。

思路见 [ARTICLE.md](ARTICLE.md)：不同模型的分词器不同，而 API 会返回 `usage.prompt_tokens`。
发一段纯文本、扣掉模板基线，就得到这段文本在该分词器下的真实 token 数。
几条探针的组合足以把分词器锁定到某一个家族。

支持 **117 个模型 / 34 个独立分词器**，清单见 [script/SUPPORTED.md](script/SUPPORTED.md)。

## 能做什么

```bash
cd script

# 看支持哪些模型
python detect.py --list

# 检测一个中转站：它到底跑的是不是它宣称的那个模型
export RELAY_API_KEY=sk-xxx
python detect.py --base-url https://relay.example.com/v1 \
                 --model deepseek-v4.1-flash --expect deepseek-v4.1-flash
```

输出（真实 SiliconFlow 结果）：

```
端点 https://api.siliconflow.cn/v1  模型 Qwen/Qwen3-8B  模板基线 8 tokens

收窄过程：
  '勘查'  -> 1 token (分支 [1, 2, 3, 5])
  '労働'  -> 3 token (分支 [1, 2, 3, 4])

验证 8 条（分组专属集）：
  8/8  qwen-vl = qwen1.5-7b = ... = qwen3-8b = qwen3-coder-30b  ← 候选
  2/8  deepseek-coder-v2 = deepseek-v2-lite = deepseek-v2.5
  1/8  ernie-4.5

判定：符合：qwen3-8b 与观测到的分词器属于同一组
注意：这一组内 10 个模型的 tokenizer 完全相同，无法再细分。
```

一次检测 ≈ 12 次请求，每次只发一个短语、`max_tokens=1`，不会真的让模型干活。

三种判定：

| 判定 | 含义 |
|---|---|
| `符合` | 观测到的 tokenizer 与声明模型同组 |
| `不符` | 观测到的是另一组，会告诉你是哪一组 |
| `无法确认` | 计数与样本里任何 tokenizer 都对不上：模型太新，或中转层用的是估算器 |

## 安装

```bash
cd script
pip install -r requirements.txt        # 或者 uv sync
```

首次使用要拉词表（约 950 MB，落在 `script/data/tokenizers/`，不进 git）：

```bash
python fetch_models.py                 # 从 hf-mirror.com 下载 115 个 tokenizer
python build_tree.py                   # 生成 tree.json / verify.json / diff_catalog.csv
```

HuggingFace 直连不通时脚本会自动走镜像；国内网络下 `hf-mirror.com` 可用。

## 目录

```
.
├── ARTICLE.md                    # 思路长文：从 api 反推 tokenizer，到找出差异最大的分词
├── article.pdf                   # 同一篇的 PDF
├── assets/                       # 文章配图
└── script/
    ├── detect.py                 # ★ 检测中转站：收窄 → 验证 → 判定
    ├── fetch_models.py           # 维护模型清单，下载 tokenizer
    ├── build_tree.py             # 统计差异 + 建决策树 + 生成验证集
    ├── build_fingerprint.py      # 另一条路线：选固定探针码本
    ├── probe_adaptive.py         # 按决策树收窄（早期版本，无验证层）
    ├── probe_online.py           # 固定码本 + 去偏移最近邻
    ├── mock_relay.py             # 假端点，离线验证检测器
    ├── count_prompt.py           # 一段提示词在各 tokenizer 下各算几个 token
    ├── make_token_table.py       # 导出 token_counts.csv（提示词 × 模型的计数矩阵）
    ├── inspect_vocab.py          # 看 tokenizer.json 的词表（解 byte-level 编码）
    ├── main.py                   # 手工最小验证：usage.prompt_tokens 的差值
    ├── SUPPORTED.md              # 支持清单
    ├── README.md                 # 工程细节、方法论、实测记录
    └── data/models.json          # 模型清单（label / family / 上游 repo）
```

## 已实测的中转站

| 端点上的模型 | 判定 | 证据 |
|---|---|---|
| SiliconFlow · Qwen3-8B / Qwen2.5-72B / Qwen3-32B | 符合 qwen 主组 | 8/8，次优 2/8 |
| SiliconFlow · DeepSeek-V3 / R1 / V4-Pro / V4.1-Flash | 符合 deepseek-v3 组 | 8/8，次优 2/8 |
| SiliconFlow · GLM-5.3 | 观测到 glm-5 新组 | 8/8，glm-4 组 6/8 |
| SiliconFlow · Kimi-K2.7-Code | 符合 kimi 组（12 个成员） | 8/8，其余 0/8 |
| SiliconFlow · Qwen3.5-9B | 观测到 qwen3.5 新组 | 8/8，其余 1/8 |
| SiliconFlow · Hunyuan-A13B | 观测到 hunyuan 组 | 8/8 |
| openlearn.store · deepseek-v4.1-flash | 符合 deepseek-v3 组 | 8/8，次优 2/8 |
| SiliconFlow · 把 Qwen2.5-72B 声明成 deepseek-v3 | **不符** | 报出实际是 qwen 组 |
| SiliconFlow · Seed-OSS-36B / Qwen3.5-122B | 无法确认 | 计分并列 / 4/8，正确拒绝 |

本地自测：`python detect.py --self-test`，用 117 个模型各自的 tokenizer 模拟端点，117/117 命中。

## 能力边界

1. **只能证伪家族，不能验证 checkpoint。** 同一套词表被多个模型共用（例如 qwen 主组 30 个、kimi 主组 12 个、deepseek 组 7 个）。`8B` 冒充 `70B`、旧版冒新版、量化冒满血，**tokenizer 完全测不出来**。
2. **决策树会硬给答案，所以加了验证层。** 收窄到候选组后再跑 8 条该组的专属验证探针，全中才认；只差一条且明显领先第二名才算"很可能"，否则判 `无法确认`。
3. **中转层可能不是在用模型自己的 tokenizer 计数。** 若它用 tiktoken 或自建估算器计费，整条路失效——表现为计数不落在任何分支上。

## 已知问题

- `telechat2`、`yuan2-2b` 在 HF 镜像上 404，未收录。
- 小模型（免费额度）TPM 限流严重，每次探针还会生成几十上百个 reasoning token；建议先测大模型。`detect.py` 内置节流与 429 退避。

## License

MIT
