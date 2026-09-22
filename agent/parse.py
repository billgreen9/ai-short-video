"""从模型回复中解析路由意图和规划动作。"""
from __future__ import annotations

import json
import re
from typing import Any

_PLAN_ACTIONS = {"user_input", "instruction"}

# instruction.list 当前支持的 domain
INSTRUCTION_DOMAINS = {
    "outline",
    "subtitle",
    "short",
    "video",
    "shot",
    "audio",
    "synthetical",
}


def message_text(message: Any) -> str:
    """取出消息里的纯文本。"""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for block in content:
            if isinstance(block, str):
                chunks.append(block)
            elif isinstance(block, dict):
                chunks.append(str(block.get("text") or ""))
        return "".join(chunks)
    return str(content)


def _strip_fence(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    return cleaned


def _load_json(text: str) -> Any:
    cleaned = _strip_fence(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for pattern in (r"\{.*\}", r"\[.*\]"):
        match = re.search(pattern, cleaned, re.DOTALL)
        if not match:
            continue
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
    return None


def parse_route_intents(content: str) -> list[dict[str, Any]]:
    """解析路由节点返回的意图列表。解析失败时返回空列表。"""
    parsed = _load_json(content)
    if isinstance(parsed, dict) and "results" in parsed:
        parsed = parsed["results"]
    if not isinstance(parsed, list):
        return []

    intents: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict) or "intent" not in item or "degree" not in item:
            continue
        try:
            intents.append(
                {"intent": str(item["intent"]), "degree": float(item["degree"])}
            )
        except (TypeError, ValueError):
            continue
    return intents


def parse_plan(content: str) -> dict[str, Any]:
    """解析规划节点的 action JSON。"""
    parsed = _load_json(content)
    if not isinstance(parsed, dict):
        raise ValueError(f"无法解析规划结果: {content!r}")

    action = parsed.get("action")
    if action not in _PLAN_ACTIONS:
        raise ValueError(f"未知的规划动作: {action!r}")

    if action == "user_input":
        answer = parsed.get("answer", "")
        return {
            "action": "user_input",
            "answer": "" if answer is None else str(answer),
        }

    raw_list = parsed.get("list") or []
    if not isinstance(raw_list, list):
        raw_list = []
    domains = [str(item).strip() for item in raw_list if str(item).strip()]
    return {"action": "instruction", "list": domains}
