"""LangGraph 智能体：路由 + 加载说明 + 规划。

图结构：
    START --> intent --> instruction --> plan
                              ^                  |
                              |                  |
                              +-- action=instruction（且仍有可加载说明）
                              |
                              +--> END（user_input / 无说明可加载等）

    单独设计 intent 是为了以后的预留召回设计，防止路由膨胀，必须要借助于“关键词+语义”检索召回

- instruction：按 domains 查表，把规划说明以 HumanMessage(input_type=agent) 写入 messages
- plan：读取该 HumanMessage 请求 LLM，回复追加为 AIMessage
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
    """智能体状态。"""

    messages: Annotated[list[BaseMessage], add_messages]

    # intent 节点
    intents: list[dict[str, Any]]  # [{"intent": "...", "degree": 0.9}, ...]

    # intent 筛选结果，或 plan(action=instruction).list，供 instruction 节点使用
    domains: list[str]

    # instruction 节点
    instructions: list[dict[str, Any]]  # [{title, domain, text, type, en_name}, ...]

    # plan 节点
    plan: dict[str, Any]  # {"action":"user_input","answer":"..."} 或 {"action":"instruction","list":[...]}
    prev_node: str


def build_initial_messages(
    user_input: str,
    *,
    id: str = "",
) -> list[BaseMessage]:
    """启动 graph 时的初始消息：outline 系统提示 + 用户输入。

    用户消息 content 为正文；additional_kwargs 仅保留 id。
    """
    outline = load_outline_prompt()
    return [
        SystemMessage(content=outline),
        _build_user_input_message(user_input, id=id),
    ]


def build_initial_state(
    user_input: str,
    *,
    id: str = "",
) -> GraphState:
    """构造 graph 初始状态。"""
    return {
        "messages": build_initial_messages(user_input, id=id),
        "intents": [],
        "domains": [],
        "instructions": [],
        "plan": {},
    }


def intent_node(state: GraphState) -> dict:
    """路由节点：路由提示词只参与本次调用；命中 domain 写入状态后进入 instruction。"""
    stage_text = ROUTE_PROMPT_TEMPLATE.format(
        intents_block=build_intents_block(),
        num=agent_config.num,
    )
    base_text = state["messages"][0].content
    combined_sys = SystemMessage(
        f"""
    {base_text}
    {stage_text}
    """
    )

    full = [combined_sys, *state["messages"][1:]]
    response = _get_llm().invoke(full)
    intents = parse_route_intents(message_text(response))
    threshold = agent_config.degree_threshold
    domains = [
        item["intent"] for item in intents if item["degree"] > threshold
    ]
    return {
        "messages": [response],
        "intents": intents,
        "domains": domains,
        "pre_node":"intent"
    }


def _user_input_from_messages(messages: list[BaseMessage]) -> str:
    """从对话中取第一条用户消息的正文。"""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            return message_text(msg).strip()
    return ""


def _message_fields(msg: BaseMessage) -> dict[str, str]:
    """读取消息上仅允许的框架字段：id、input_type。"""
    kwargs = getattr(msg, "additional_kwargs", None) or {}
    if not isinstance(kwargs, dict):
        kwargs = {}
    return {
        "id": str(kwargs.get("id") or "0"),
        "input_type": str(kwargs.get("input_type") or ""),
    }


def _framework_id_from_messages(messages: list[BaseMessage]) -> str:
    """从首条用户消息读取框架 id。"""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            return _message_fields(msg)["id"]
    return "0"


def _build_user_input_message(
    text: str,
    *,
    id: str = "0",
    input_type: str | None = None,
) -> HumanMessage:
    """构造用户输入消息：content 为正文；additional_kwargs 可含 id、input_type。"""
    kwargs: dict[str, str] = {"id": id or "0"}
    if input_type is not None:
        kwargs["input_type"] = input_type
    return HumanMessage(content=text, additional_kwargs=kwargs)


def _dump_plan(plan: dict[str, Any]) -> str:
    return json.dumps(plan, ensure_ascii=False)


def instruction_node(state: GraphState) -> dict:
    """按 state.domains 查 instruction，写入状态，并以 HumanMessage(input_type=agent) 追加到 messages。"""
    messages = list(state.get("messages") or [])
    user_input = _user_input_from_messages(messages)
    if not user_input:
        return {"instructions": []}

    domains = [d for d in (state.get("domains") or []) if d]
    if not domains:
        return {"instructions": []}

    rows = fetch_instructions(domains)
    if not rows:
        return {
            "instructions": [],
            "domains": domains,
        }

    prompt = PLAN_PROMPT_TEMPLATE.format(
        user_input=user_input,
        instruction_blocks=build_instruction_blocks(rows),
        domain_hints=build_instruction_domain_hints(),
    )
    return {
        "instructions": rows,
        "domains": domains,
        "messages": [
            _build_user_input_message(
                prompt,
                id=_framework_id_from_messages(messages),
                input_type="agent",
            )
        ],
    }


def _latest_agent_human_message(
    messages: list[BaseMessage],
) -> HumanMessage | None:
    """取最近一条 input_type=agent 的用户消息。"""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and _message_fields(msg)["input_type"] == "agent":
            return msg
    return None


def plan_node(state: GraphState) -> dict:
    """规划节点：用 instruction 写入的 HumanMessage(input_type=agent) 请求 LLM。"""
    messages = list(state.get("messages") or [])
    domains = list(state.get("domains") or [])
    rows = list(state.get("instructions") or [])
    user_input = _user_input_from_messages(messages)

    if not user_input or not messages:
        plan = {
            "action": "user_input",
            "answer": "用户输入信息不全，无法进行规划，请补充",
        }
        return {"messages": [AIMessage(content=_dump_plan(plan))], "plan": plan}

    if not domains:
        plan = {
            "action": "user_input",
            "answer": "未匹配到足够置信度的操作，请补充更具体的需求",
        }
        return {"messages": [AIMessage(content=_dump_plan(plan))], "plan": plan}

    if not rows:
        plan = {"action": "instruction", "list": domains}
        # instructions 为空时 plan_route 会结束，避免空查死循环
        return {"messages": [AIMessage(content=_dump_plan(plan))], "plan": plan}

    agent_human = _latest_agent_human_message(messages)
    if agent_human is None:
        plan = {
            "action": "user_input",
            "answer": "规划前置说明缺失，请补充更具体的需求",
        }
        return {"messages": [AIMessage(content=_dump_plan(plan))], "plan": plan}

    llm_messages: list[BaseMessage] = []
    if isinstance(messages[0], SystemMessage):
        llm_messages.append(SystemMessage(content=message_text(messages[0])))
    llm_messages.append(agent_human)

    response = _get_llm().invoke(llm_messages)
    # todo 后续这种结果返回尽量使用 with_structured_output 方式。
    raw = message_text(response)
    try:
        plan = parse_plan(raw)
    except ValueError:
        plan = {
            "action": "user_input",
            "answer": "规划结果无法解析，请补充更具体的需求",
        }
        return {
            "messages": [
                AIMessage(content=raw),
                _build_user_input_message(
                    plan["answer"],
                    id=_framework_id_from_messages(messages),
                    input_type="agent",
                ),
            ],
            "plan": plan,
        }

    updates: dict[str, Any] = {
        "messages": [AIMessage(content=raw)],
        "plan": plan,
    }
    # 需要继续加载说明时，把 list 写回 domains，供 instruction 节点使用
    if plan.get("action") == "instruction":
        updates["domains"] = [str(x) for x in (plan.get("list") or []) if str(x)]
    return updates


def plan_route(state: GraphState) -> str:
    """plan 返回 action=instruction 且仍有可加载内容时，回流 instruction。"""
    plan = state.get("plan") or {}
    if plan.get("action") != "instruction":
        return END
    # 上次未查到说明：直接结束，把 instruction list 交给调用方
    if not (state.get("instructions") or []):
        return END
    if not (state.get("domains") or []):
        return END
    return "instruction"


def build_graph():
    """构造并编译智能体图。"""
    graph = StateGraph(GraphState)
    graph.add_node("intent", intent_node)
    graph.add_node("instruction", instruction_node)
    graph.add_node("plan", plan_node)
    graph.add_edge(START, "intent")
    graph.add_edge("intent", "instruction")
    graph.add_edge("instruction", "plan")
    graph.add_conditional_edges(
        "plan",
        plan_route,
        {
            "instruction": "instruction",
            END: END,
        },
    )
    return graph.compile()


compiled_graph = build_graph()
