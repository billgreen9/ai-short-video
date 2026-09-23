"""LangGraph 检查点存储：PostgresSaver。

作用：图在 action=user_input/param 时通过 interrupt() 真正挂起，
会话状态（messages、计数器、action 等）按 thread_id 持久化到 PostgreSQL，
服务重启后仍可用 /agent/resume 恢复。

checkpoint 表由 saver.setup() 自动创建（CREATE TABLE IF NOT EXISTS），
建在 POSTGRES_DSN 所指向的库中（与 instruction 表同库）。
"""
from __future__ import annotations

import atexit
from typing import Optional

from langgraph.checkpoint.postgres import PostgresSaver

from config import settings

_saver: Optional[PostgresSaver] = None
# 保存连接池上下文管理器，便于进程退出时释放
_cm = None


def get_checkpointer() -> PostgresSaver:
    """获取进程级 PostgresSaver 单例；首次调用时建连接池并建表。"""
    global _saver, _cm
    if _saver is None:
        if not settings.POSTGRES_DSN:
            raise RuntimeError("未配置 POSTGRES_DSN，无法初始化 LangGraph 检查点存储。")
        _cm = PostgresSaver.from_conn_string(settings.POSTGRES_DSN)
        _saver = _cm.__enter__()
        # 幂等建表：checkpoints / checkpoint_writes / checkpoint_blobs
        _saver.setup()
        atexit.register(_close)
    return _saver


def _close() -> None:
    global _saver, _cm
    if _cm is not None:
        try:
            _cm.__exit__(None, None, None)
        except Exception:
            pass
    _saver = None
    _cm = None
