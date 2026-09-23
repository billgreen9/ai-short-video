"""LangGraph 智能体：意图路由 + 说明加载 + 规划 + 执行前/执行。

主路径：
    START --> intent --> instruction --(+ instruction_route)--> plan
                              ^                                  |
                              |                                  +--(+ plan 自环)
                              |                                  |
                              +---- plan_route (instruction)     |
                                                                 v
                                              user_input/param → 暂停(END)
                                              can_execute → before_execute → execute → END

plan 节点：
    - 首次进入（plan_prompt_used=False）：向全量 messages 末尾注入一次
      PLAN_PROMPT_TEMPLATE（5 种 action 协议），再请求 LLM；操作说明已在 messages 中，不重复携带
    - 回流（plan 自环 / instruction / param 等）：直接基于当前全量 messages 请求 LLM
    - 按返回 action 分支（见 plan_route）

循环次数限制（阈值在 config.json）：
    - plan_enter_count：plan 节点每次执行都 +1；instruction/param 等回流也计入；
      超过 plan_enter_max(15) 强制结束（action=end）
    - plan_review_count：仅当因 action=plan 自环（原子化自检）回流时 +1；
      因 instruction/param/user_input 造成的回流不计；
      超过 plan_review_max(10) 暂停（action=user_input 指导话术）

instruction_route：
    - prev_node == "intent" -> plan
    - 其他 -> prev_node（通常为 plan）
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
    PLAN_ATOMIC_CONFIRM_PROMPT,
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

    # domains 仅由上游写入：intent（命中意图）或 plan（action=instruction 的 list）
    domains: list[str]

    # instruction 节点查表结果
    instructions: list[dict[str, Any]]  # [{title, domain, text, type, en_name}, ...]

    # plan 节点输出（动作结果）
    action: dict[str, Any]  # {"action":"user_input","answer":"..."} 或 {"action":"instruction","list":[...]}
    prev_node: str
    # 是否已用 PLAN_PROMPT_TEMPLATE 请求过规划
    plan_prompt_used: bool

    # plan 节点执行计数（循环保护）
    plan_enter_count: int    # plan 节点总执行次数（含 instruction/param 等回流）
    plan_review_count: int   # 仅 action=plan 原子化自检自环的次数


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
        "plan_enter_count": 0,
        "plan_review_count": 0,
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


def _plan_node_impl(state: GraphState) -> dict:
    """规划节点主体（不含循环计数，计数由 plan_node 包装器统一写入）。

    - 首次进入（plan_prompt_used=False）：向全量 messages 末尾注入一次
      PLAN_PROMPT_TEMPLATE（action 协议），再请求 LLM；操作说明已在 messages 中，不重复携带
    - 回流：直接基于当前全量 messages 请求 LLM
    - 按 LLM 返回的 action 写状态，由 plan_route 跳转或暂停
    """
    messages = list(state.get("messages") or [])
    user_input = _user_input_from_messages(messages)
    fw_id = _framework_id_from_messages(messages)

    if not user_input or not messages:
        plan = {
            "action": "user_input",
            "answner": "用户输入信息不全，无法进行规划，请补充",
        }
        return {
            "messages": [AIMessage(content=_dump_plan(plan))],
            "action": plan,
            "prev_node": "plan",
        }

    llm_messages: list[BaseMessage]
    extra_messages: list[BaseMessage] = []

    if state.get("plan_prompt_used"):
        # 回流（plan 自环 / instruction / param 等）：直接用当前全量对话请求 LLM
        llm_messages = list(messages)
    else:
        # 首次进入：注入一次 action 协议提示词。
        # 用户请求与操作说明已在 messages 中，这里不再重复携带，避免同一份说明发两遍。
        request_msg = _build_user_input_message(
            PLAN_PROMPT_TEMPLATE.format(
                domain_hints=build_instruction_domain_hints(),
            ),
            id=fw_id,
            input_type="agent",
        )
        extra_messages = [request_msg]
        llm_messages = list(messages) + [request_msg]

    response = _get_llm().invoke(llm_messages)
    # todo 后续这种结果返回尽量使用 with_structured_output 方式。
    raw = message_text(response)
    try:
        plan = parse_plan(raw)
    except ValueError:
        plan = {
            "action": "user_input",
            "answner": "规划结果无法解析，请补充更具体的需求",
        }
        return {
            "messages": extra_messages
            + [
                AIMessage(content=raw),
                _build_user_input_message(
                    plan["answner"],
                    id=fw_id,
                    input_type="agent",
                ),
            ],
            "action": plan,
            "prev_node": "plan",
            "plan_prompt_used": True,
        }

    ai_msg = AIMessage(content=raw)
    kind = plan.get("action")

    # 1/2 暂停：写入 action，由 plan_route 结束
    if kind in ("user_input", "param"):
        return {
            "messages": extra_messages + [ai_msg],
            "action": plan,
            "prev_node": "plan",
            "plan_prompt_used": True,
        }

    # 3 加载说明：消息入列，list → domains，进 instruction
    if kind == "instruction":
        domains_next = [str(x) for x in (plan.get("list") or []) if str(x)]
        return {
            "messages": extra_messages + [ai_msg],
            "action": plan,
            "domains": domains_next,
            "prev_node": "plan",
            "plan_prompt_used": True,
        }

    # 4 详细 plan：消息入列，再追加原子/无环确认话术，自环回 plan
    if kind == "plan":
        confirm_msg = _build_user_input_message(
            PLAN_ATOMIC_CONFIRM_PROMPT,
            id=fw_id,
            input_type="agent",
        )
        return {
            "messages": extra_messages + [ai_msg, confirm_msg],
            "action": plan,
            "prev_node": "plan",
            "plan_prompt_used": True,
        }

    # 5 can_execute：进 before_execute
    return {
        "messages": extra_messages + [ai_msg],
        "action": plan,
        "prev_node": "plan",
        "plan_prompt_used": True,
    }


def plan_node(state: GraphState) -> dict:
    """规划节点包装器：统一做循环计数与超限保护，再执行主体逻辑。

    - plan_enter_count：每次进入 plan 都 +1（instruction/param 等回流也计入），
      超过 plan_enter_max 强制结束（action=end）
    - plan_review_count：仅因上一轮 action=plan 自环回流时 +1，
      超过 plan_review_max 暂停（action=user_input 指导话术）
    """
    prev_action = (state.get("action") or {}).get("action")
    enter_total = int(state.get("plan_enter_count") or 0) + 1
    review_total = int(state.get("plan_review_count") or 0) + (
        1 if prev_action == "plan" else 0
    )

    counters = {
        "plan_enter_count": enter_total,
        "plan_review_count": review_total,
    }

    # 总执行次数超限：强制结束（优先于暂停判定）
    if enter_total > agent_config.plan_enter_max:
        plan = {
            "action": "end",
            "msg": f"plan 节点执行次数已超过上限 {agent_config.plan_enter_max} 次，强制结束",
        }
        return {
            "messages": [AIMessage(content=_dump_plan(plan))],
            "action": plan,
            "prev_node": "plan",
            "plan_prompt_used": True,
            **counters,
        }

    # 原子化自检自环超限：暂停，给出指导话术（由 plan_route 结束，等待用户补充后重新发起）
    if review_total > agent_config.plan_review_max:
        plan = {
            "action": "user_input",
            "answner": (
                f"规划自检已达 {agent_config.plan_review_max} 轮，"
                "请确认计划是否已全部拆分为无环的原子方法，或补充必要信息后重新发起"
            ),
        }
        return {
            "messages": [AIMessage(content=_dump_plan(plan))],
            "action": plan,
            "prev_node": "plan",
            "plan_prompt_used": True,
            **counters,
        }

    result = _plan_node_impl(state)
    result.update(counters)
    return result


def plan_route(state: GraphState) -> str:
    """按 action 跳转：暂停 / instruction / plan 自环 / before_execute / 强制结束。"""
    action = state.get("action") or {}
    kind = action.get("action")

    if kind in ("user_input", "param"):
        return END

    # 循环保护强制结束（仅由 plan_node 计数守卫产生，LLM 不会返回该 action）
    if kind == "end":
        return END

    if kind == "instruction":
        if state.get("domains"):
            return "instruction"
        return END

    if kind == "plan":
        return "plan"

    if kind == "can_execute":
        return "before_execute"

    return END


def before_execute_node(state: GraphState) -> dict:
    """执行前处理：校验 / 收集参数（占位）。"""
    return {"prev_node": "before_execute"}


def execute_node(state: GraphState) -> dict:
    """执行节点（占位，后续可挂工具调用 / 工作流执行等）。"""
    return {"prev_node": "execute"}


def build_graph():
    """构造并编译智能体图。"""
    graph = StateGraph(GraphState)
    graph.add_node("intent", intent_node)
    graph.add_node("instruction", instruction_node)
    graph.add_node("plan", plan_node)
    graph.add_node("before_execute", before_execute_node)
    graph.add_node("execute", execute_node)
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
            "plan": "plan",
            "before_execute": "before_execute",
            END: END,
        },
    )
    graph.add_edge("before_execute", "execute")
    graph.add_edge("execute", END)
    return graph.compile()


compiled_graph = build_graph()
