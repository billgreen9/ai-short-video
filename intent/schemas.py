"""意图识别相关数据模型。"""
from __future__ import annotations

from pydantic import BaseModel, Field


class SkillScore(BaseModel):
    """单个技能的匹配结果。"""

    skill: str = Field(..., description="技能标识符")
    degree: float = Field(..., ge=0.0, le=1.0, description="置信度，0-1，越大越匹配")
