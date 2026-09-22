"""智能体提示词与技能定义。

注意：本模块独立于 intent/selector.py 中的 SKILLS 定义，
两处可能存在重复；这是"接口完全独立"的刻意设计。
如后续需统一，可抽取到公共常量模块。
"""
from __future__ import annotations

# 技能定义：标识符 -> 中文描述
SKILLS = {
    "audio": "音频相关处理（TTS 语音合成、ASR 音频转写、音频剪辑、音色克隆、降噪、背景音处理等）",
    "video": "视频相关处理（视频生成、剪辑、特效、转场、视频翻译、画面合成等）",
    "subtitle": "字幕相关（字幕识别、字幕翻译、字幕样式调整、SRT/VTT/ASS 处理、字幕烧录等）",
    "short": "短视频或内容缩写（短视频脚本、内容摘要、精简、速览、口播稿压缩等）",
    "synthetical": "综合任务（跨多个技能的复合任务、整体工作流、无法归入单一技能的请求）",
}


def build_skills_block() -> str:
    """将技能字典格式化为提示词中的列表文本。"""
    return "\n".join(f"- {name}: {desc}" for name, desc in SKILLS.items())


# 系统提示词模板
# {num} 在运行时从 config.json 注入；{user_input} 不在此处——用户输入作为
# HumanMessage 放入 messages 字段，由 agent_node 拼到 LLM 请求中。
SYSTEM_PROMPT_TEMPLATE = """你是一个路由选择器，根据用户输入选择最合适的技能。

技能列表：
{skills_block}

返回结果：
以 JSON 格式返回置信度最高的 {num} 个技能，严格遵循如下格式（不要附加任何文字、不要使用 markdown 代码块）：
{{"results": [{{"skill": "audio", "degree": 1.0}}, {{"skill": "synthetical", "degree": 0.8}}]}}

要求：
1. degree 为置信度，取值 0-1，越大匹配度越高；
2. 结果按 degree 从大到小排序，最多返回 {num} 个；
3. skill 字段必须是上述列表中的英文标识符：audio / video / subtitle / short / synthetical；
4. 仅输出 JSON 本身，不要包含解释或多余文本。
""".strip()
