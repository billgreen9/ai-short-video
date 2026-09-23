"""LangGraph 智能体：意图路由 + 说明加载 + 规划。

主路径：
    START --> intent --> instruction --(+ instruction_route)--> plan
                              ^                                  |
                              |                                  |
                              +---- plan_route (action=instruction)
                              |
                              +---- instruction_route (非 intent 进入时回 prev_node)

instruction_route：
    - prev_node == "intent" -> plan
    - 其他 -> prev_node（通常为 plan）

plan_route：
    - action.action == "instruction" 且 instructions/domains 非空 -> instruction
    - 其余 -> END

各节点职责：
    - intent：写入 intents / domains，prev_node=intent
    - instruction：只按 domains 加载说明到 HumanMessage(input_type=agent)；不改 prev_node
    - plan：首次用 PLAN_PROMPT_TEMPLATE 请求；之后用「请继续尝试规划」；
      写入 action；action=instruction 时更新 domains 并回流 instruction

启动 messages：
    [SystemMessage(outline), HumanMessage(用户正文, additional_kwargs={id})]
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
_CONTINUE_PLAN_PROMPT = "请继续尝试规划"


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

    # domains 仅由上游写入：intent（命中意图）或 plan（action=instruction 的 list）
    domains: list[str]

    # instruction 节点查表结果
    instructions: list[dict[str, Any]]  # [{title, domain, text, type, en_name}, ...]

    # plan 节点输出（动作结果）
    action: dict[str, Any]  # {"action":"user_input","answer":"..."} 或 {"action":"instruction","list":[...]}
    prev_node: str
    # 是否已用 PLAN_PROMPT_TEMPLATE 请求过规划
    plan_prompt_used: bool


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
        "action": {},
        "prev_node": "",
        "plan_prompt_used": False,
    }


def intent_node(state: GraphState) -> dict:
    """路由节点：解析意图，把命中的 domains 写入状态，交给 instruction 加载。"""
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
        "prev_node": "intent",
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
    """只根据上游 domains 加载说明消息；不修改 prev_node（供 instruction_route 判断来源）。"""
    domains = [d for d in (state.get("domains") or []) if d]
    if not domains:
        return {"instructions": []}

    rows = fetch_instructions(domains)
    if not rows:
        return {"instructions": []}

    messages = list(state.get("messages") or [])
    return {
        "instructions": rows,
        "messages": [
            _build_user_input_message(
                build_instruction_blocks(rows),
                id=_framework_id_from_messages(messages),
                input_type="agent",
            )
        ],
    }


def instruction_route(state: GraphState) -> str:
    """intent 进入 -> plan；其它来源回 prev_node。"""
    prev = state.get("prev_node") or ""
    if prev == "intent":
        return "plan"
    if prev and prev != "instruction":
        return prev
    return "plan"


def _latest_agent_human_message(
    messages: list[BaseMessage],
) -> HumanMessage | None:
    """取最近一条 input_type=agent 的用户消息。"""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and _message_fields(msg)["input_type"] == "agent":
            return msg
    return None


def plan_node(state: GraphState) -> dict:
    """规划节点：首次用 PLAN_PROMPT_TEMPLATE，之后用「请继续尝试规划」。"""
    messages = list(state.get("messages") or [])
    domains = list(state.get("domains") or [])
    rows = list(state.get("instructions") or [])
    user_input = _user_input_from_messages(messages)
    plan_prompt_used = bool(state.get("plan_prompt_used"))

    if not user_input or not messages:
        plan = {
            "action": "user_input",
            "answer": "用户输入信息不全，无法进行规划，请补充",
        }
        return {
            "messages": [AIMessage(content=_dump_plan(plan))],
            "action": plan,
            "prev_node": "plan",
        }

    if not domains:
        plan = {
            "action": "user_input",
            "answer": "未匹配到足够置信度的操作，请补充更具体的需求",
        }
        return {
            "messages": [AIMessage(content=_dump_plan(plan))],
            "action": plan,
            "prev_node": "plan",
        }

    if not rows:
        plan = {"action": "instruction", "list": domains}
        return {
            "messages": [AIMessage(content=_dump_plan(plan))],
            "action": plan,
            "prev_node": "plan",
        }

    if not plan_prompt_used:
        request_text = PLAN_PROMPT_TEMPLATE.format(
            user_input=user_input,
            instruction_blocks=build_instruction_blocks(rows),
            domain_hints=build_instruction_domain_hints(),
        )
    else:
        request_text = _CONTINUE_PLAN_PROMPT

    request_msg = _build_user_input_message(
        request_text,
        id=_framework_id_from_messages(messages),
        input_type="agent",
    )

    llm_messages: list[BaseMessage] = []
    if isinstance(messages[0], SystemMessage):
        llm_messages.append(SystemMessage(content=message_text(messages[0])))
    # 继续规划时带上最近一次 instruction 加载的说明消息
    if plan_prompt_used:
        loaded = _latest_agent_human_message(messages)
        if loaded is not None:
            llm_messages.append(loaded)
    llm_messages.append(request_msg)

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
                request_msg,
                AIMessage(content=raw),
                _build_user_input_message(
                    plan["answer"],
                    id=_framework_id_from_messages(messages),
                    input_type="agent",
                ),
            ],
            "action": plan,
            "prev_node": "plan",
            "plan_prompt_used": True,
        }

    updates: dict[str, Any] = {
        "messages": [request_msg, AIMessage(content=raw)],
        "action": plan,
        "prev_node": "plan",
        "plan_prompt_used": True,
    }
    if plan.get("action") == "instruction":
        updates["domains"] = [str(x) for x in (plan.get("list") or []) if str(x)]
    return updates


def plan_route(state: GraphState) -> str:
    """state.action.action == instruction 且仍有可加载内容时，回流 instruction。"""
    action = state.get("action") or {}
    if action.get("action") != "instruction":
        return END
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
    graph.add_conditional_edges(
        "instruction",
        instruction_route,
        {
            "plan": "plan",
        },
    )
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
