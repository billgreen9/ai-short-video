"""智能体 HTTP 接口：通过 FastAPI 暴露 LangGraph agent。

- GET /agent/start  发起一轮；若 plan 返回 user_input/param，图在 human_input 节点
  通过 interrupt() 挂起，响应 status=paused 并携带指导话术/缺参载荷
- GET /agent/resume 提交用户补充（Command(resume=...)），图从挂起点继续，
  可能再次暂停或走到 can_execute/end 完成

会话按 thread_id 持久化（PostgresSaver）：thread_id 优先用 id，缺省生成 UUID。
num / degree_threshold / 循环次数阈值均从 config.json 读取。
"""
from __future__ import annotations

import uuid
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from langgraph.types import Command
from pydantic import BaseModel, Field

from agent.graph import build_initial_state, compiled_graph

agent_router = APIRouter(prefix="/agent", tags=["agent"])


class AgentResponse(BaseModel):
    """智能体响应：完成时带 action；挂起时带 interrupt 载荷。"""

    thread_id: str = Field(..., description="会话 ID，resume 时原样带回")
    status: str = Field(..., description="paused=等待用户输入；done=本轮图执行结束")
    interrupt: Optional[dict[str, Any]] = Field(
        default=None,
        description='挂起载荷：{"type":"user_input","answer":"指导话术"} '
        '或 {"type":"param","msg":"缺少参数"}',
    )
    action: Optional[dict[str, Any]] = Field(
        default=None,
        description="done 时的规划结果（user_input/param/instruction/plan/can_execute/end）",
    )


def _run(inputs: Any, thread_id: str) -> AgentResponse:
    """以指定 thread_id 驱动图，统一识别 interrupt 挂起与正常结束。"""
    config = {"configurable": {"thread_id": thread_id}}
    try:
        result = compiled_graph.invoke(inputs, config=config)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"智能体调用失败: {e}")

    if not isinstance(result, dict):
        raise HTTPException(status_code=500, detail=f"图返回结构异常: {type(result)!r}")

    interrupts = result.get("__interrupt__")
    if interrupts:
        # tuple[Interrupt, ...]；当前流程同一时刻至多一个挂起点
        intr = interrupts[0]
        payload = getattr(intr, "value", None)
        if not isinstance(payload, dict):
            payload = {"type": "user_input", "answer": str(payload)}
        return AgentResponse(thread_id=thread_id, status="paused", interrupt=payload)

    action = result.get("action") or None
    return AgentResponse(
        thread_id=thread_id,
        status="done",
        action=action if action else None,
    )


@agent_router.get("/start", response_model=AgentResponse, summary="发起一轮智能体规划")
def start(
    user_input: str = Query(..., description="用户原始输入文本", min_length=1),
    id: str = Query("", description="会话/框架 id；为空时自动生成 UUID"),
) -> AgentResponse:
    """写入 outline 系统提示 + 用户输入，跑 intent → instruction → plan。

    若 plan 判定需要用户补充信息/参数，图在 human_input 节点挂起，
    返回 status=paused；随后用 GET /agent/resume 继续同一会话。
    """
    thread_id = id or str(uuid.uuid4())
    try:
        initial_state = build_initial_state(user_input, id=thread_id)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return _run(initial_state, thread_id)


@agent_router.get("/resume", response_model=AgentResponse, summary="提交用户补充并恢复挂起的图")
def resume(
    thread_id: str = Query(..., description="start 响应中返回的会话 ID", min_length=1),
    user_input: str = Query(..., description="用户补充的信息/参数", min_length=1),
) -> AgentResponse:
    """对处于 paused 的会话提交补充内容，图从 human_input 节点继续回到 plan。"""
    return _run(Command(resume=user_input), thread_id)
