"""LangGraph 智能体：基于消息的技能路由 agent。

图结构：
    START --> agent --> END

- agent 节点读取 system_prompt + messages，调用 LLM，将回复追加到 messages
- 所有交互消息都存放在 messages 字段（add_messages reducer 自动累加）
- num 等业务参数在构建 system_prompt 时从 config.json 注入，不进入图状态
"""
from __future__ import annotations

from typing import Annotated, Optional

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from config import settings


# 模块级 LLM 单例：复用客户端连接
_llm: Optional[ChatOpenAI] = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        kwargs = {
            "model": settings.OPENAI_MODEL,
            "base_url": settings.OPENAI_BASE_URL,
            "api_key": settings.OPENAI_API_KEY,
            "temperature": 0.0,
        }
        if settings.OPENAI_DISABLE_THINKING:
            # 关闭推理模型的思考阶段，降低路由分类场景的延迟
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        _llm = ChatOpenAI(**kwargs)
    return _llm


class GraphState(TypedDict):
    """智能体状态。

    messages:      所有对话消息，通过 add_messages reducer 累加
    system_prompt: 系统提示词（含 num 等业务参数，在调用前格式化注入）
    """

    messages: Annotated[list[BaseMessage], add_messages]
    system_prompt: str


def agent_node(state: GraphState) -> dict:
    """Agent 节点：用 system_prompt 作为系统消息，调用 LLM 回复用户。

    返回的 AIMessage 由 add_messages reducer 自动追加到 messages 字段。
    """
    system_prompt = state["system_prompt"]
    messages = state["messages"]
    # 将系统消息拼到对话最前面
    full = [SystemMessage(content=system_prompt)] + list(messages)
    response = _get_llm().invoke(full)
    return {"messages": [response]}


def build_graph():
    """构造并编译智能体图。"""
    graph = StateGraph(GraphState)
    graph.add_node("agent", agent_node)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    return graph.compile()


# 模块级编译图单例：HTTP 层直接复用
compiled_graph = build_graph()
