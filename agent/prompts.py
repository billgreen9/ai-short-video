"""智能体提示词与意图定义。"""
from __future__ import annotations

from typing import Any

# 意图定义：标识符 -> 中文描述
INTENTS = {
    "audio": "音频相关处理（TTS 语音合成、ASR 音频转写、音频剪辑、音色克隆、降噪、背景音处理等）",
    "video": "视频相关处理（视频生成、剪辑、特效、转场、视频翻译、画面合成等）",
    "subtitle": "字幕相关（字幕识别、字幕翻译、字幕样式调整、SRT/VTT/ASS 处理、字幕烧录等）",
    "short": "短视频或内容缩写（短视频脚本、内容摘要、精简、速览、口播稿压缩等）",
    "synthetical": "综合任务（跨多个意图的复合任务、整体工作流、无法归入单一意图的请求）",
}

# instruction.list / instruction.domain 说明，写入规划提示词供模型参考
INSTRUCTION_DOMAIN_HINTS = {
    "outline": "大纲：介绍整个 agent 系统能够支持的操作、禁止的操作；只是指导性说明，不参与任何数据修改",
    "subtitle": "字幕操作相关",
    "short": "内容缩写",
    "video": "视频处理",
    "shot": "分镜处理",
    "audio": "音频处理",
    "synthetical": "综合操作，一般是以上操作的组合",
}


def build_intents_block() -> str:
    """将意图字典格式化为提示词中的列表文本。"""
    return "\n".join(f"- {name}: {desc}" for name, desc in INTENTS.items())


def build_instruction_domain_hints() -> str:
    """格式化 instruction.list 支持的 domain 说明。"""
    return "\n".join(
        f"- {name}: {desc}" for name, desc in INSTRUCTION_DOMAIN_HINTS.items()
    )


def build_instruction_blocks(rows: list[dict[str, Any]]) -> str:
    """将 instruction 表记录拼成操作说明段落。"""
    blocks: list[str] = []
    for row in rows:
        title = row.get("title") or ""
        domain = row.get("domain") or ""
        text = row.get("text") or ""
        type_ = row.get("type") or ""
        en_name = row.get("en_name") or ""
        blocks.append(
            "\n".join(
                [
                    f"--------{title}操作说明start------",
                    f"domain:{domain}",
                    f"type:{type_}",
                    f"en_name:{en_name}",
                    f"正文:{text}",
                    f"--------{title}操作说明end-------",
                ]
            )
        )
    return "\n\n".join(blocks)


# 路由提示词模板（仅用于 agent 节点当次 LLM 调用，不写入 messages）
# {num} 在运行时从 config.json 注入
ROUTE_PROMPT_TEMPLATE = """
当前阶段：任务路由。
任务：路由选择器，根据用户输入选择最合适的意图。

意图列表：
{intents_block}

返回结果：
以 JSON 格式返回置信度最高的 {num} 个意图，严格遵循如下格式（不要附加任何文字、不要使用 markdown 代码块）：
{{"results": [{{"intent": "audio", "degree": 1.0}}, {{"intent": "synthetical", "degree": 0.8}}]}}

要求：
1. degree 为置信度，取值 0-1，越大匹配度越高；
2. 结果按 degree 从大到小排序，最多返回 {num} 个；
3. intent 字段必须是上述列表中的英文标识符：audio / video / subtitle / short / synthetical；
4. 仅输出 JSON 本身，不要包含解释或多余文本。
""".strip()


# 规划节点提示词：仅声明下一步动作的 JSON 协议。
# 注意：用户请求与命中的操作说明（instruction blocks）已在 messages 中：
#   - 用户请求：首条 HumanMessage
#   - 操作说明：instruction 节点追加的 HumanMessage(input_type=agent)
# 本提示词由 plan 节点首次进入时注入一次，不重复携带说明正文。
PLAN_PROMPT_TEMPLATE = """请根据对话中已给出的用户请求与操作说明，进行下一步规划。
注意：返回必须以定义的 json 格式返回，不要附加任何文字、不要使用 markdown 代码块。

1. 当前处理用户请求尚且不清晰，需要用户补充信息，并给出指导话术：
{{"action":"user_input","answner":"指导话术"}}

2. 当前用户请求清晰，但是执行缺少参数，比如视频 id、分镜 id 等：
{{"action":"param","msg":"缺少参数"}}

3. 对如何完成用户请求尚且不明白，还缺少必要说明：
{{"action":"instruction","list":["subtitle","short"],"help":false}}

4. 输出给后端详细的 plan 计划，按执行先后返回；order 越小越先执行，order 相同可并发：
{{"action":"plan","plans":[{{"en_name":"xx","order":1}}]}}

5. 已经将规划上的所有方法变更为原子方法，可以执行：
{{"action":"can_execute","msg":"所有方法已经是原子方法了，可以执行了","plans":[{{"en_name":"xx","order":1}}]}}

instruction.list 当前支持的值：
{domain_hints}

要求：
1. action 只能是 user_input / param / instruction / plan / can_execute；
2. 仅输出 JSON 本身。
""".strip()

# plan 自检回流时追加的确认话术
PLAN_ATOMIC_CONFIRM_PROMPT = (
    "请确认是否每个节点都是function节点，是否存在环，"
    "如果不是请拆分为原子节点，然后确认无环时返回。"
)
