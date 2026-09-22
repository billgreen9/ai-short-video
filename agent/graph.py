"""LangGraph 智能体：路由 + 规划。

图结构：
    START --> intent --> plan --> END

    单独设计intent是为了以后的预留召回设计，防止路由膨胀，必须要借助于“关键词+语义”检索召回

状态只保留 messages：
- 启动时写入 [SystemMessage(outline), HumanMessage(user_input)]
- intent：路由提示词仅用于本次 LLM 调用，不写入 messages；回复追加为 AIMessage
- plan：读取 messages 最后一条（路由结果）做规划，回复追加为 AIMessage
"""
from __future__ import annotations

import json
from typing import Annotated, Any, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from agent.config import agent_config
from agent.instruction_store import fetch_instructions, load_outline_prompt
from agent.parse import message_text, parse_plan, parse_route_intents
from agent.prompts import (
    PLAN_PROMPT_TEMPLATE,
    ROUTE_PROMPT_TEMPLATE,
    build_instruction_blocks,
    build_instruction_domain_hints,
    build_intents_block,
)
from config import settings


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
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        _llm = ChatOpenAI(**kwargs)
    return _llm


class GraphState(TypedDict):
    """智能体状态：仅维护 messages。"""

    messages: Annotated[list[BaseMessage], add_messages]


def build_initial_messages(
    user_input: str,
    *,
    id: str = "",
    video_id: str = "",
) -> list[BaseMessage]:
    """启动 graph 时的初始消息：outline 系统提示 + 带框架元信息的用户输入。

    元信息只写在用户消息里，id / video_id 均可为空。
    """
    outline = load_outline_prompt()
    human_payload = {
        "id": id or "0",
        "video_id": video_id or "",
        "text": user_input,
    }
    return [
        SystemMessage(content=outline),
        HumanMessage(content=json.dumps(human_payload, ensure_ascii=False)),
    ]


def intent_node(state: GraphState) -> dict:
    """路由节点：路由提示词只参与本次调用，不写入 messages。"""
    stage_text = ROUTE_PROMPT_TEMPLATE.format(
        intents_block=build_intents_block(),
        num=agent_config.num,
    )
    # 路由提示词仅用于本次 LLM 请求；messages 里保留 outline 系统提示与用户消息
    base_text = state["messages"][0].content  # 全局基线
    combined_sys = SystemMessage(f"""
    {base_text}
    {stage_text}
    """)

    full = [combined_sys,
        *state["messages"][1:]
    ]
    response = _get_llm().invoke(full)
    return {"messages": [response]}


def _user_input_from_messages(messages: list[BaseMessage]) -> str:
    """从对话中取第一条用户消息的文本内容。"""
    for msg in messages:
        if not isinstance(msg, HumanMessage):
            continue
        raw = message_text(msg).strip()
        if not raw:
            return ""
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        if isinstance(obj, dict) and "text" in obj:
            return str(obj.get("text") or "").strip()
        return raw
    return ""


def _dump_plan(plan: dict[str, Any]) -> str:
    return json.dumps(plan, ensure_ascii=False)


def plan_node(state: GraphState) -> dict:
    """规划节点：解析 messages 最后一条路由结果，查 instruction 后做下一步规划。"""
    messages = list(state.get("messages") or [])
    if not messages:
        plan = {
            "action": "user_input",
            "answer": "用户输入信息不全，无法进行规划，请补充",
        }
        return {"messages": [AIMessage(content=_dump_plan(plan))]}

    last_content = message_text(messages[-1])
    intents = parse_route_intents(last_content)
    threshold = agent_config.degree_threshold
    matched = [item for item in intents if item["degree"] > threshold]
    domains = [item["intent"] for item in matched]

    user_input = _user_input_from_messages(messages)
    if not user_input:
        plan = {
            "action": "user_input",
            "answer": "用户输入信息不全，无法进行规划，请补充",
        }
        return {"messages": [AIMessage(content=_dump_plan(plan))]}

    if not domains:
        plan = {
            "action": "user_input",
            "answer": "未匹配到足够置信度的操作，请补充更具体的需求",
        }
        return {"messages": [AIMessage(content=_dump_plan(plan))]}

    rows = fetch_instructions(domains)
    if not rows:
        plan = {"action": "instruction", "list": domains}
        return {"messages": [AIMessage(content=_dump_plan(plan))]}

    prompt = PLAN_PROMPT_TEMPLATE.format(
        user_input=user_input,
        instruction_blocks=build_instruction_blocks(rows),
        domain_hints=build_instruction_domain_hints(),
    )
    # 规划提示词同样只用于本次调用，不写入 messages
    response = _get_llm().invoke(
        [SystemMessage(content=message_text(messages[0])), HumanMessage(content=prompt)]
        if messages and isinstance(messages[0], SystemMessage)
        else [HumanMessage(content=prompt)]
    )
    raw = message_text(response)
    try:
        parse_plan(raw)
    except ValueError:
        plan = {
            "action": "user_input",
            "answer": "规划结果无法解析，请补充更具体的需求",
        }
        return {"messages": [AIMessage(content=_dump_plan(plan))]}

    return {"messages": [AIMessage(content=raw)]}


def build_graph():
    """构造并编译智能体图。"""
    graph = StateGraph(GraphState)
    graph.add_node("agent", intent_node)
    graph.add_node("plan", plan_node)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", "plan")
    graph.add_edge("plan", END)
    return graph.compile()


compiled_graph = build_graph()
