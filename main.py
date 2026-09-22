"""AI 短视频辅助服务的 FastAPI 入口。

启动方式：
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
或直接运行：
    python main.py
"""
from __future__ import annotations

import uvicorn
from fastapi import FastAPI

from api.agent_routes import agent_router

app = FastAPI(
    title="AI 短视频辅助服务",
    description="根据用户输入路由到合适的意图（audio/video/subtitle/short/synthetical）",
    version="0.1.0",
)

app.include_router(agent_router)


@app.get("/health", summary="健康检查")
def health() -> dict:
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
