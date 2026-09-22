"""智能体全局配置：从 config.json 加载业务参数。

config.json 存放业务配置（num 等），与 .env 中的 LLM 服务配置（API Key 等）分离：
- .env：密钥、服务地址、模型名（敏感/环境相关）
- config.json：业务参数（返回个数、阈值等，可提交到仓库）
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

# config.json 位于项目根目录（agent/ 的上一级）
CONFIG_PATH = Path(__file__).parent.parent / "config.json"


class AgentConfig(BaseModel):
    """智能体业务配置。"""

    # 路由返回的意图个数上限
    num: int = 2
    # 进入规划节点前，意图置信度必须严格大于该阈值
    degree_threshold: float = Field(default=0.72, ge=0.0, le=1.0)


def load_config() -> AgentConfig:
    """加载 config.json；文件不存在时返回默认配置。"""
    if not CONFIG_PATH.exists():
        return AgentConfig()
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return AgentConfig(**data)


# 模块级单例：启动时加载一次
agent_config = load_config()
