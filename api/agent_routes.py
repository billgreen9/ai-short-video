"""智能体 HTTP 接口：通过 FastAPI 暴露 LangGraph agent。

num / degree_threshold 从 config.json 读取，不接受请求参数。
plan 优先取自 GraphState.plan。
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from agent.graph import build_initial_state, compiled_graph

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
) -> AgentRouteResponse:
    """启动时写入 outline 系统提示 + 用户输入，再跑 intent → instruction → plan。"""
    try:
        initial_state = build_initial_state(
            user_input,
            id=id,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        result = compiled_graph.invoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"智能体调用失败: {e}")

    plan = result.get("plan") or None
    return AgentRouteResponse(plan=plan if plan else None)
