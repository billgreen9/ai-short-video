"""智能体 HTTP 接口：通过 FastAPI 暴露 LangGraph agent。

与 api/routes.py (/router/route) 完全独立：
- 不共享 SkillRouter、RouteResponse
- num 从 config.json 读取，不接受请求参数
- agent 的回复作为 message 返回，同时解析出 skills 列表
"""
from __future__ import annotations

import json
import re
from typing import List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from agent.config import agent_config
from agent.graph import compiled_graph
from agent.prompts import SYSTEM_PROMPT_TEMPLATE, build_skills_block
from langchain_core.messages import HumanMessage

agent_router = APIRouter(prefix="/agent", tags=["agent"])


class AgentSkillItem(BaseModel):
    """智能体解析出的单个技能。"""

    skill: str
    degree: float


class AgentRouteResponse(BaseModel):
    """智能体路由响应。"""

    user_input: str = Field(..., description="原始用户输入")
    message: str = Field(..., description="agent 的原始回复内容（JSON 字符串）")
    skills: List[AgentSkillItem] = Field(
        default_factory=list,
        description="从 message 中解析出的技能列表，按 degree 降序",
    )


def _parse_skills_from_message(content: str) -> List[AgentSkillItem]:
    """从 agent 回复的消息中解析技能列表。

    支持 {"results": [...]} 对象、直接数组、正则兜底三种方式。
    """
    text = content.strip()
    # 去掉可能被模型误加的 markdown 代码块
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()

    parsed = None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "results" in obj:
            parsed = obj["results"]
        elif isinstance(obj, list):
            parsed = obj
    except json.JSONDecodeError:
        pass

    if parsed is None:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = None

    if not isinstance(parsed, list):
        return []

    result: List[AgentSkillItem] = []
    for item in parsed:
        if isinstance(item, dict) and "skill" in item and "degree" in item:
            try:
                result.append(
                    AgentSkillItem(skill=item["skill"], degree=float(item["degree"]))
                )
            except (TypeError, ValueError):
                continue
    # 按配置的 num 截断
    return result[: agent_config.num]


@agent_router.get(
    "/route",
    response_model=AgentRouteResponse,
    summary="智能体技能路由（LangGraph）",
)
def agent_route(
    user_input: str = Query(..., description="用户原始输入文本", min_length=1),
) -> AgentRouteResponse:
    """通过 LangGraph 智能体调用技能路由。

    num 从全局 config.json 读取，不作为请求参数。
    修改 num：编辑 config.json 后重启服务。

    请求示例：
    ```
    GET /agent/route?user_input=帮我把视频字幕翻译成英文并生成中文配音
    ```
    """
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        skills_block=build_skills_block(),
        num=agent_config.num,
    )
    try:
        result = compiled_graph.invoke(
            {
                "messages": [HumanMessage(content=user_input)],
                "system_prompt": system_prompt,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"智能体调用失败: {e}")

    messages = result.get("messages", [])
    if not messages:
        raise HTTPException(status_code=500, detail="智能体未返回消息")

    last = messages[-1]
    content = last.content if hasattr(last, "content") else str(last)
    skills = _parse_skills_from_message(content)

    return AgentRouteResponse(
        user_input=user_input,
        message=content,
        skills=skills,
    )
