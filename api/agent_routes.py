"""智能体 HTTP 接口：通过 FastAPI 暴露 LangGraph agent。

num / degree_threshold 从 config.json 读取，不接受请求参数。
plan 从 messages 最后一条解析。
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from agent.graph import build_initial_messages, compiled_graph
from agent.parse import message_text, parse_plan

agent_router = APIRouter(prefix="/agent", tags=["agent"])


class AgentRouteResponse(BaseModel):
    """智能体规划响应。"""

    plan: Optional[dict[str, Any]] = Field(
        default=None,
        description='规划结果，如 {"action":"user_input","answer":"..."} '
        '或 {"action":"instruction","list":["subtitle"]}',
    )


@agent_router.get(
    "/start",
    response_model=AgentRouteResponse,
    summary="智能体意图 start",
)
def start(
    user_input: str = Query(..., description="用户原始输入文本", min_length=1),
    id: str = Query("", description="框架元信息 id，可为空"),
    video_id: str = Query("", description="框架元信息视频 id，可为空"),
) -> AgentRouteResponse:
    """启动时写入 outline 系统提示 + 用户输入（含可选 id / video_id），再跑 agent → plan。"""
    try:
        initial_messages = build_initial_messages(
            user_input,
            id=id,
            video_id=video_id,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        result = compiled_graph.invoke({"messages": initial_messages})
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"智能体调用失败: {e}")

    messages = result.get("messages") or []
    ai_messages = [m for m in messages if isinstance(m, AIMessage)]
    if not ai_messages:
        raise HTTPException(status_code=500, detail="智能体未返回消息")

    try:
        plan = parse_plan(message_text(ai_messages[-1]))
    except ValueError:
        plan = None

    return AgentRouteResponse(plan=plan)
