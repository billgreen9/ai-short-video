"""请求/响应数据模型。"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class RouteRequest(BaseModel):
    """路由选择请求。"""

    user_input: str = Field(..., description="用户原始输入文本")


class SkillScore(BaseModel):
    """单个技能的匹配结果。"""

    skill: str = Field(..., description="技能标识符")
    degree: float = Field(..., ge=0.0, le=1.0, description="置信度，0-1，越大越匹配")


class RouteResponse(BaseModel):
    """路由选择响应。返回置信度最高的 2 个技能。"""

    user_input: str = Field(..., description="原始用户输入，便于回溯")
    skills: List[SkillScore] = Field(
        ..., description="置信度最高的 2 个技能，按 degree 降序排列"
    )
