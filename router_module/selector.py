"""路由选择器核心：调用 LLM 对用户输入做技能路由，返回置信度最高的 2 个技能。"""
from __future__ import annotations

import json
import re
from typing import List, Optional

from openai import OpenAI

from .config import settings
from .schemas import SkillScore

# 技能定义：标识符 -> 中文描述。描述用于辅助 LLM 准确路由。
SKILLS = {
    "audio": "音频相关处理（TTS 语音合成、ASR 音频转写、音频剪辑、音色克隆、降噪、背景音处理等）",
    "video": "视频相关处理（视频生成、剪辑、特效、转场、视频翻译、画面合成等）",
    "subtitle": "字幕相关（字幕识别、字幕翻译、字幕样式调整、SRT/VTT/ASS 处理、字幕烧录等）",
    "short": "短视频或内容缩写（短视频脚本、内容摘要、精简、速览、口播稿压缩等）",
    "synthetical": "综合任务（跨多个技能的复合任务、整体工作流、无法归入单一技能的请求）",
}

# 系统提示词：构建 LLM 的角色与输出格式约束。
SYSTEM_PROMPT = """你是一个路由选择器，根据用户输入选择最合适的技能。

技能列表：
{skills_block}

用户输入：{user_input}

返回结果：
以 JSON 格式返回置信度最高的 2 个技能，严格遵循如下格式（不要附加任何文字、不要使用 markdown 代码块）：
{{"results": [{{"skill": "audio", "degree": 1.0}}, {{"skill": "synthetical", "degree": 0.8}}]}}

要求：
1. degree 为置信度，取值 0-1，越大匹配度越高；
2. 结果按 degree 从大到小排序，最多返回 2 个；
3. skill 字段必须是上述列表中的英文标识符：audio / video / subtitle / short / synthetical；
4. 仅输出 JSON 本身，不要包含解释或多余文本。
""".strip()


def _build_skills_block() -> str:
    """将技能字典格式化为提示词中的列表文本。"""
    return "\n".join(f"- {name}: {desc}" for name, desc in SKILLS.items())


def _parse_skills(raw: str) -> List[SkillScore]:
    """从 LLM 输出中解析出技能列表。

    优先按 {"results": [...]} 解析；失败时尝试直接解析为列表；
    再失败时用正则提取首个 JSON 片段。最终做范围截断与去重。
    """
    text = raw.strip()
    # 去掉可能被模型误加的 markdown 代码块
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()

    parsed: Optional[List[dict]] = None

    # 方式 1：标准对象包装
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "results" in obj:
            parsed = obj["results"]
        elif isinstance(obj, list):
            parsed = obj
    except json.JSONDecodeError:
        pass

    # 方式 2：正则兜底，提取第一个 JSON 数组
    if parsed is None:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = None

    if not isinstance(parsed, list):
        raise ValueError(f"无法从 LLM 输出解析技能列表: {raw!r}")

    seen = set()
    scores: List[SkillScore] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        skill = item.get("skill")
        degree = item.get("degree")
        if not isinstance(skill, str) or not isinstance(degree, (int, float)):
            continue
        if skill not in SKILLS or skill in seen:
            continue
        seen.add(skill)
        scores.append(SkillScore(skill=skill, degree=float(degree)))

    # 按 degree 降序，最多保留 2 个
    scores.sort(key=lambda s: s.degree, reverse=True)
    return scores[:2]


class SkillRouter:
    """路由选择器。封装 LLM 调用与结果解析。"""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        disable_thinking: Optional[bool] = None,
    ) -> None:
        self.base_url = base_url or settings.OPENAI_BASE_URL
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model = model or settings.OPENAI_MODEL
        self.temperature = temperature
        # None 时回退到 settings 中的默认值；显式传入则覆盖
        self.disable_thinking = (
            settings.OPENAI_DISABLE_THINKING
            if disable_thinking is None
            else disable_thinking
        )
        # 延迟构造客户端，便于测试注入参数；key 缺失时仍允许构造，但调用会抛错。
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            if not self.api_key:
                raise RuntimeError(
                    "未配置 OPENAI_API_KEY，请在 .env 或环境变量中设置 LLM 服务的 API Key。"
                )
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    def select(self, user_input: str) -> List[SkillScore]:
        """根据用户输入返回置信度最高的 2 个技能。"""
        if not user_input or not user_input.strip():
            return []

        prompt = SYSTEM_PROMPT.format(
            skills_block=_build_skills_block(),
            user_input=user_input,
        )

        # 构造请求参数；关闭推理时通过 extra_body 传递厂商私有参数
        # (Ark doubao-seed 等推理模型支持 thinking.type=disabled 以跳过思考阶段)
        kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": "你是一个严谨的技能路由选择器。"},
                {"role": "user", "content": prompt},
            ],
        }
        if self.disable_thinking:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

        completion = self.client.chat.completions.create(**kwargs)
        raw = completion.choices[0].message.content or ""
        return _parse_skills(raw)
