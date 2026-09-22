"""LangGraph 智能体模块：意图路由 + 规划。

业务参数从 config.json 读取。
"""
from .graph import compiled_graph

__all__ = ["compiled_graph"]
