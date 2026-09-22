"""LangGraph 智能体模块：基于消息的技能路由 agent。

与 intent/ (SkillRouter) 和 api/routes.py (/router/route) 完全独立，
不共享状态、不共享响应模型。num 等业务参数从 config.json 读取。
"""
from .graph import compiled_graph

__all__ = ["compiled_graph"]
