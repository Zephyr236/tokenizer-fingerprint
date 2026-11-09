# 用 tokenizer 指纹判断中转站模型

把 [readme.md](../readme.md) 里的思路（增量法反推 tokenizer → 找分词差异最大的片段）落地成可复现的流水线，并且真的能对着中转站跑。

## 快速开始：检测一个中转站

```bash
cd script
pip install tokenizers sentencepiece tiktoken numpy

python detect.py --list                       # 看支持检测的模型（26 个 tokenizer 家族 / 55 个模型）
export RELAY_API_KEY=sk-xxx
python detect.py --base-url https://api.xxx.com/v1 --model Qwen/Qwen3-8B --expect qwen3-8b
```

输出长这样（下面是真跑 SiliconFlow 的结果）：

```
端点 https://api.siliconflow.cn/v1  模型 Qwen/Qwen3-8B  模板基线 8 tokens

收窄过程：
  '勘查'           -> 1 token (分支 [1, 2, 3, 5])
  '労働'           -> 3 token (分支 [1, 2, 3, 4])

验证 5 条（命中数 / 5）：
  5/5  qwen-vl = qwen1.5-7b = ... = qwen3-8b = qwen3-coder-30b  ← 本轮收窄的候选
  4/5  deepseek-coder-v2 = deepseek-v2-lite = deepseek-v2.5
  4/5  ernie-4.5

判定：符合：qwen3-8b 与观测到的分词器属于同一组（qwen-vl = ... = qwen3-coder-30b）
注意：这一组内 9 个模型的 tokenizer 完全相同，无法再细分。
```

一次检测 = 1 次基线 + 2~3 次收窄 + 8 次验证 ≈ 12 次请求，每次只发一个短语、`max_tokens=1`，不会真的让模型干活。

三种判定：

| 判定 | 含义 |
|---|---|
| `符合` | 观测到的 tokenizer 与声明模型同组 |
| `不符` | 观测到的是另一组，会告诉你是哪一组 |
| `无法确认` | 计数与样本里任何 tokenizer 都对不上：模型太新，或中转层用的是估算器 |

### 支持范围

`python detect.py --list` 会打印完整清单：**117 个模型 → 34 个独立分词器**。同组内部 tokenizer 完全相同，只能判到组这一级：

| 家族 | 组内成员 |
|---|---|
| **qwen（主组，30 个）** | qwen1.5-7b = qwen2-7b/2-vl-7b = qwen2.5-0.5b/1.5b/3b/7b/14b/32b/72b/coder-7b/coder-32b/vl-72b = qwen3-0.6b/1.7b/4b/8b/14b/32b/30b-a3b/30b-2507/235b-a22b/235b-2507/coder-30b/coder-480b/next-80b/vl-8b/vl-32b = qwen-vl = kimi-dev-72b |
| **qwen3.5/3.6（新词表，3 个）** | qwen3.5-9b = qwen3.5-27b = qwen3.6-27b |
| **glm-4 系（9 个）** | glm-4-9b-0414 = glm-4-32b-0414 = glm-4.1v-9b = glm-4.5 = glm-4.5-air = glm-4.5v = glm-4.6 = glm-z1-9b = glm-z1-rumination-32b |
| **glm-5 系（新词表，4 个）** | glm-5 = glm-5.1 = glm-5.2 = glm-5.3 |
| **kimi（主组，12 个）** | kimi-k2 = k2-base = k2-0905 = k2-thinking = k2.5 = k2.6 = k2.7-code = kimi-vl = vl-thinking = vl-2506 = kimi-linear-48b = moonlight-16b |
| kimi 异系 | kimi-audio-7b 单词表；kimi-dev-72b 落在 qwen 主组（Qwen2.5-Coder 微调） |
| deepseek-v3 系 | v3 = v3.1 = r1 = vl2 = v4-flash = v4-pro = v4.1-flash |
| deepseek-v2 系 | v2-lite = v2.5 = coder-v2；coder-6.7b 单列 |
| llama-3 | llama-3-8b = 3.1-8b = 3.2-3b = 3.3-70b = nemotron-70b |
| mistral-7b 系 | mistral-7b-v0.1 = mixtral-8x7b = mixtral-8x22b = codestral-22b = openchat-3.5 = zephyr-7b |
| mistral-tekken | nemo = small-24b = ministral-8b；mistral-large-2411 单列 |
| llama-2 / phi-3 | llama-2-7b = phi-3-mini = phi-3.5-mini |
| hunyuan / ernie / yi | hunyuan-7b = hunyuan-a13b；ernie-4.5 = ernie-4.5-21b；yi-1.5-9b = yi-34b = yi-coder-9b |
| 其他单体 | gemma-2 / gemma-3、phi-4 / phi-4-mini、granite-3.1、jamba、codellama、starcoder2、falcon、olmo-2、minicpm4、minimax-m1/m2、baichuan2、internlm2.5/3、xverse、gpt2、cl100k、o200k |

### 在 SiliconFlow 上的实测（真实 API）

| 端点上的模型 | 判定 | 命中 |
|---|---|---|
| **Qwen/Qwen3.5-9B** | **观测到 qwen3.5 新组** | **8/8**（其余组 1/8） |
| **zai-org/GLM-5.3** | **观测到 glm-5 新组** | **8/8**（glm-4 组 6/8） |
| THUDM/GLM-4-9B-0414 `--expect glm-4-9b-0414` | 符合 glm-4 组 | 8/8（glm-5 组 5/8） |
| **deepseek-ai/DeepSeek-V4.1-Flash** | **符合 deepseek-v3 组** | **8/8** |
| **moonshotai/Kimi-K2.7-Code** | **符合 kimi 组（12 个成员）** | **8/8**（其余组 0/8） |
| deepseek-ai/DeepSeek-V4-Pro | 观测到 deepseek 组 | 8/8 |
| deepseek-ai/DeepSeek-V3 / Pro-V3 | 观测到 deepseek 组 | 8/8 |
| Qwen/Qwen2.5-72B-Instruct、Qwen3-32B | 观测到 qwen 主组 | 8/8 |
| Qwen/Qwen3-8B `--expect qwen3-8b` | 符合 | 8/8 |
| tencent/Hunyuan-A13B-Instruct | 观测到 hunyuan 组 | 8/8 |
| Qwen2.5-72B 声明成 `deepseek-v3` | 不符 | 报出实际是 qwen 组 |
| ByteDance-Seed/Seed-OSS-36B-Instruct | 无法确认 | 7/8 但第二名并列 → 拒绝 |
| Qwen/Qwen3.5-122B-A10B | 无法确认 | 4/8 → 拒绝 |

本地 `--self-test` 用 117 个模型各自的 tokenizer 模拟端点，**117/117 命中**；`mock_relay.py --corrupt` 用按字节估算的假计数器，检测器正确拒绝。

## 结论先说

- 样本：**22 个 tokenizer**（20 个从 HuggingFace 下载 + 2 个 OpenAI tiktoken 参考系），去重后 **18 个独立分词器**
- 探针池：**52,906** 条候选（语料 n-gram 3,518 + 从各词表真实 token 挖掘 49,851）
- 工程探针集：**48 条可读文本**，最小 L1 码距 **14**，无噪声与常数偏移下唯一识别率 **100%**
- 最小探针集：**12 条**，码距 6，仍能 100% 唯一识别
- 差异表 + 决策树路线：**平均 2.22 次、最多 3 次**请求即可唯一锁定（18/18 分组）

### 差异最大的分词片段（单条熵最高）

顺序 = `baichuan2 / deepseek-v2-lite / deepseek-v3 / ernie-4.5 / gemma-2 / glm-4.9B=4.5 / gpt2 / internlm2.5 / kimi-k2 / llama-2=phi-3 / llama-3.1 / minimax / mistral-7B / mistral-nemo / cl100k / o200k / qwen2=2.5=3 / yi-1.5`

| 片段 | 单条熵 (bit) | token 数向量 |
|---|---|---|
| `扫黑除恶专项斗争` | 3.31 / 4.00 | 1 / 1 / 3 / 6 / 7 / 3 / **20** / 6 / 1 / **19** / 8 / 6 / 11 / 9 / 12 / 7 / 2 / 2 |
| `涡轮增压发动机` | 3.20 | 1 / 1 / 3 / 3 / 5 / 2 / **17** / 3 / 1 / **16** / 8 / 2 / 10 / 8 / 10 / 7 / 5 / 4 |
| `阿弥陀佛` | 3.06 | 1 / 1 / 2 / 4 / 4 / 1 / **10** / 4 / 1 / **11** / 5 / 3 / 9 / 6 / 8 / 6 / 4 / 3 |
| `疫情防控工作` | 2.86 | 1 / 1 / 1 / 2 / 3 / 1 / **14** / 2 / 1 / **11** / 5 / 2 / 9 / 6 / 10 / 4 / 2 / 1 |
| `日内与新浪网联系` | 2.86 | 4 / 1 / 5 / 5 / 6 / 4 / **19** / 5 / 1 / **11** / 7 / 5 / 9 / 9 / 8 / 6 / 5 / 5 |
| `為什麼` | 2.79 | 3 / 5 / 1 / 2 / 1 / 3 / **8** / 2 / 4 / **10** / 3 / 2 / 6 / 4 / 8 / 3 / 1 / 1 |
| `猕猴桃` | 2.97 | 1 / 1 / 2 / 3 / 3 / 2 / **9** / 1 / 1 / **10** / 5 / 2 / 8 / 6 / 8 / 4 / 3 / 5 |

读法：同一句话，DeepSeek/Qwen/GLM 只花 1-3 个 token，gpt2 / llama-2 / phi-3 要花 10-20 个。**分词效率本身就是语种覆盖率的指纹**。

### 分词完全一致的分组（tokenizer 视角零信号）

| 分组 | 原因 |
|---|---|
| `qwen2-7b = qwen2.5-72b = qwen3-8b` | vocab(151643)、merges(151387) 内容完全相同，仅 merges 序列化格式不同（`"Ġ Ġ"` vs `["Ġ","Ġ"]`） |
| `glm-4-9b-0414 = glm-4.5` | vocab(151329)、merges(318088) 相同，仅 `added_tokens` 不同（14 vs 36） |
| `llama-2-7b = phi-3-mini` | 正文分词**逐条相同**（40,291 条可读探针上完全相同），唯一差异是 `<s>`：llama-2 切成 `<`+`s`+`>`，phi-3 是 1 个特殊 token |

## 两条路线

| 路线 | 请求数 | 抗单点错误 | 可解释性 | 脚本 |
|---|---|---|---|---|
| 固定码本 + 最近邻 | 48（或 12） | 强，最小码距 14 | 弱，只给距离 | `build_fingerprint.py` → `probe_online.py` |
| 差异表 + 决策树自适应收窄 | **2.2 平均 / 3 最多** | 弱，错一步全错 | 强，一层层看得懂 | `build_tree.py` → `probe_adaptive.py` |

决策树的形状（深度 3，18 个分组全部唯一锁定）：

```
探 `切磋`  → 1 token → 探 `亜鉛` → 1/2/3/4 → minimax / baichuan2 / deepseek-v3 / kimi
           → 2 token → 探 `上昇` → 1/2/3/4 → gemma / (再问 `万亿`) / glm / (再问 `一種`)
           → 3 token → 探 `矩阵` → 1/2/3/4 → deepseek-v2-lite / o200k / llama-3.1 / nemo
           → 5 token → 探 `企业` → 2/3/4/5 → cl100k / mistral-7B / gpt2 / llama-2=phi-3
```

建树时的两个约束（否则树会退化成"一条探针切 18 桶"，错 1 个 token 就进错枝）：分支宽度上限 4 桶、前两层只允许可读片段。
线上用 `probe_adaptive.py --beam` 可以在计数异常时保留相邻分支、输出所有存活分组。

## 文件

```
script/
├── make_token_table.py    # 直接产出 token_counts.csv：一行一个片段，后面是各模型的 token 数
├── count_prompt.py        # 任意一段提示词，看它在 22 个 tokenizer 里各算几个 token
├── build_fingerprint.py   # tokenizer → 计数矩阵 → 选码本探针 → fingerprint.json + report.md
├── probe_online.py        # 码本路线：打固定探针集，去偏移最近邻命中
├── build_tree.py          # 差异路线：统计差异目录 + 建 ID3 决策树
├── probe_adaptive.py      # 差异路线：按树逐层收窄，2-3 次请求定位
├── token_counts.csv       # 主表：prompt / len / spread / distinct / source + 22 个模型的 token 数
├── fingerprint.json       # 码本：探针列表 + 每组 token 数向量 + 抗噪指标
├── tree.json              # 决策树：节点=片段，分支=观测到的 token 数
├── diff_catalog.csv       # 差异目录：候选片段 × 判别力 × 各计数桶的成员
├── report.md              # 完整实测报告（codebook 全表、混淆对、抗噪表）
├── tree_report.md         # 决策树全文 + 差异片段排行
├── ranked_probes.csv      # 可读候选按单条判别力排序
└── data/tokenizers/       # 20 个 tokenizer 文件（约 143 MB）
```

`data/tokenizers/` 命名规则 `<label>.tokenizer.json` / `.tokenizer.model` / `.tiktoken.model`，来源：

| label | 上游 repo | 类型 |
|---|---|---|
| deepseek-v3 / deepseek-v2-lite | `deepseek-ai/DeepSeek-V3`、`-V2-Lite` | HF tokenizer.json |
| qwen3-8b / qwen2.5-72b / qwen2-7b | `Qwen/Qwen3-8B`、`Qwen2.5-72B-Instruct`、`Qwen2-7B-Instruct` | HF |
| llama-3.1-8b | `NousResearch/Meta-Llama-3.1-8B-Instruct` | HF |
| llama-2-7b | `NousResearch/Llama-2-7b-hf` | SentencePiece |
| baichuan2-7b / internlm2.5-7b | `baichuan-inc/Baichuan2-7B-Chat`、`internlm/internlm2_5-7b-chat` | SentencePiece |
| glm-4.5 / glm-4-9b-0414 | `zai-org/GLM-4.5`、`zai-org/GLM-4-9B-0414` | HF |
| kimi-k2 | `moonshotai/Kimi-K2-Base` | tiktoken 词表 |
| minimax-m1 | `MiniMaxAI/MiniMax-M1-40k` | HF |
| ernie-4.5 | `baidu/ERNIE-4.5-0.3B-PT` | HF |
| gemma-2-9b | `unsloth/gemma-2-9b-it` | HF |
| mistral-7b-v0.1 / mistral-nemo | `mistralai/...` | HF |
| phi-3-mini / yi-1.5-9b / gpt2 | `microsoft/...`、`01-ai/...`、`openai-community/gpt2` | HF |
| cl100k / o200k | tiktoken 内置 | tiktoken |

## 方法

1. **取 token 数**：`usage.prompt_tokens` 是服务栈给出的精确计数。只发一条 user 消息时
   `prompt_tokens(probe) = 模板开销 + count(probe)`，模板开销对同一端点恒定，做差即得纯 tokenizer 结果。
   单条探针只给一个整数，所以用**多条探针的组合**当指纹（codeword）。
2. **造探针池**：语料 n-gram（中英、代码、SQL、URL、日期、全角、繁简、多语种）+ 从每个词表挖真实 token 表面形式。
   挖出来的碎片区分度最高，但大量是 SEO/垃圾站残片（`彩神争霸`、`毛片免费视频` 之类），所以工程集限定"可读文本"。
3. **分组去重**：在计数矩阵上逐位比较，把分词行为完全相同的模型合并——这一步直接暴露"哪些模型没法用 tokenizer 分开"。
4. **选探针（两阶段贪心）**：先保证每对分组至少被一条探针分开，再最大化「码字之间的最小 L1 距离」。
   只最大化最小距离会在第一轮全平局时退化成按字典序瞎选，是实现里最容易踩的坑。
5. **匹配**：观测码字与码本做**去中位数偏移的 L1** 最近邻，chat 模板口径差异（常数偏移）因此不影响判定。
6. **抗噪测试**：给真实码字加抖动再反查，量化"计数错几个 token 还能认对"。

## 复现

```bash
cd script
pip install tokenizers sentencepiece tiktoken numpy    # 或 uv sync
python build_fingerprint.py          # 码本路线，约 1-2 分钟，重写 fingerprint.json / report.md
python build_tree.py                 # 差异路线，重写 tree.json / diff_catalog.csv / tree_report.md
```

`build_tree.py` 的参数就是"抗噪 vs 请求数"的旋钮：

```bash
python build_tree.py --arity 2                 # 最多 2 桶：平均 4.2 次探针，最抗噪
python build_tree.py --arity 4 --readable-depth 2   # 默认：平均 2.2 次，最多 3 次
python build_tree.py --arity 6                 # 平均 2.0 次，桶宽、更脆
```

`data/tokenizers/` 里文件已下好。重下时 HuggingFace 直连不通可用镜像：

```bash
python -c "import urllib.request as u; u.urlretrieve('https://hf-mirror.com/deepseek-ai/DeepSeek-V3/resolve/main/tokenizer.json','data/tokenizers/deepseek-v3.tokenizer.json')"
```

## 打中转站

```bash
# 只是一段提示词想知道各模型算几个 token（纯本地，不打 API）
python count_prompt.py "请帮我把下面这段中文翻译成英文，注意保持专业术语的准确性。"
python count_prompt.py --file prompts.txt        # 批量
python count_prompt.py "..." --csv               # 顺手追加到 token_counts.csv

# 路线一：差异表 + 决策树，2-3 次请求（推荐先跑这个）
python probe_adaptive.py --dry-run --model deepseek-v3
export RELAY_API_KEY=sk-xxx
python probe_adaptive.py --base-url https://relay.example.com/v1 --model gpt-4o
python probe_adaptive.py --base-url ... --model ... --beam      # 计数异常时保留相邻分支

# 路线二：固定码本，48 条（或 12 条）一次打完
python probe_online.py --dry-run --model deepseek-v3
python probe_online.py --base-url ... --model ... --repeat 3
python probe_online.py --base-url ... --model ... --set min    # 只发 12 条
```

`probe_adaptive.py` 输出逐层收窄的 trace 和最终命中；`probe_online.py` 输出观测码字 + 最近 5 个候选 + 残差，`--save out.json` 存档对比。

## 局限（重要）

1. **只能证伪家族，不能验证 checkpoint。** 同族不同尺寸/量化/微调共用同一个 tokenizer（8B 冒充 70B、旧版冒新版、量化冒满血），tokenizer 完全测不出来。22 个样本里 4 个与同族兄弟位级相同。
2. **中转层计数口径。** `prompt_tokens` 由服务栈给，不保证是模型自己的 tokenizer。若中转层用 tiktoken 或自建估算器计费，整套方法失效——表现为残差远超码距、或决策树里的 token 数不在任何分支上（`probe_adaptive.py` 会给 `← 不在任何分支上` 标记）。先用官方直连端点校准一次。
3. **决策树是"错一步全错"结构。** 想要抗噪就压分支宽度（`--arity 2`）、开 `--beam`，或者回到码本路线复核。真正稳妥的组合是"树收窄到 1-2 个候选 + 码本少量探针复核"。
4. **探针要保持原样。** 两端不能有空白；emoji / 全角 / 韩日文区分度高但易被中转层过滤或改写，只做辅助确认。主探针用中文常用词 + 稀有字 + 数字串 + 代码片段。
5. **词表挖掘的副产品**：最"脏"的片段区分度最高（博彩/色情/新闻版权残片），因为它直接反映各家爬取语料；这也说明这类碎片 token 边界不稳定，不适合当探针。
