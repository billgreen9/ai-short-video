"""HTTP 响应数据模型。"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from intent.schemas import SkillScore


class RouteResponse(BaseModel):
    """路由选择响应。返回置信度最高的 2 个技能。"""

    user_input: str = Field(..., description="原始用户输入，便于回溯")
    skills: List[SkillScore] = Field(
        ..., description="置信度最高的 2 个技能，按 degree 降序排列"
    )
