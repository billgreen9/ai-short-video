"""按 domain 读取 instruction 表中的有效操作说明。"""
from __future__ import annotations

from typing import Any

import psycopg

from config import settings


def fetch_instructions(domains: list[str]) -> list[dict[str, Any]]:
    """查询 status=1 且 domain 落在给定列表中的操作说明，按 id 升序。"""
    if not domains:
        return []
    if not settings.POSTGRES_DSN:
        raise RuntimeError("未配置 POSTGRES_DSN，无法查询 instruction 表。")

    sql = """
        SELECT title, domain, "text", type, en_name
        FROM instruction
        WHERE status = 1
          AND domain = ANY(%s)
        ORDER BY id
    """
    with psycopg.connect(settings.POSTGRES_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (domains,))
            rows = cur.fetchall()
    return [
        {
            "title": title,
            "domain": domain,
            "text": text,
            "type": type_,
            "en_name": en_name,
        }
        for title, domain, text, type_, en_name in rows
    ]


def load_outline_prompt() -> str:
    """加载 domain=outline 的系统提示词正文。多条时按 id 拼接。"""
    rows = fetch_instructions(["outline"])
    texts = [str(row.get("text") or "").strip() for row in rows]
    texts = [t for t in texts if t]
    if not texts:
        raise RuntimeError(
            "instruction 表缺少 status=1 且 domain=outline 的有效记录，无法启动 graph。"
        )
    return "\n\n".join(texts)
