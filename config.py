"""配置加载：通过环境变量读取 LLM 服务参数。"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """运行时配置。优先级：环境变量 > .env 文件 > 默认值。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # LLM 服务 (OpenAI 兼容协议)
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    # 是否关闭推理模型的思考过程 (Ark doubao-seed 等推理模型在路由分类场景下建议关闭以大幅降低延迟)
    OPENAI_DISABLE_THINKING: bool = True

    # PostgreSQL：instruction 表（系统提示词与各 domain 操作说明）
    POSTGRES_DSN: str = ""


settings = Settings()
