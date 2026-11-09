# 最小的手工验证：读一次 usage.prompt_tokens，看空输入和加词之后的差值。
# 这条脚本对应文章里"构造差异"那一段的截图，不依赖本项目的其它模块。
#
#   export RELAY_API_KEY=sk-xxx
#   export RELAY_BASE_URL=https://api.siliconflow.cn/v1
#   python main.py [model]
import os
import sys

from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("RELAY_API_KEY", ""),
    base_url=os.environ.get("RELAY_BASE_URL", "https://api.siliconflow.cn/v1"),
)
model = sys.argv[1] if len(sys.argv) > 1 else "deepseek-ai/DeepSeek-V3"


def count(content):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=1,
        temperature=0,
    )
    return resp.usage.prompt_tokens


base = count("a")
probe = "中国特色社会主义"
print(f"model      = {model}")
print(f"baseline('a') = {base} tokens")
print(f"+ {probe} = {count(probe)} tokens  → 该词占 {count(probe) - (base - 1)} 个 token")
