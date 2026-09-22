"""FastAPI 路由：暴露 /route 接口供外部调用。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .schemas import RouteRequest, RouteResponse
from .selector import SkillRouter

router = APIRouter(prefix="/router", tags=["router"])

# 模块级单例：复用 LLM 客户端连接
_router_instance: SkillRouter | None = None


def get_router() -> SkillRouter:
    global _router_instance
    if _router_instance is None:
        _router_instance = SkillRouter()
    return _router_instance


@router.post("/route", response_model=RouteResponse, summary="技能路由选择")
def route(req: RouteRequest) -> RouteResponse:
    """根据用户输入选择置信度最高的 2 个技能。

    请求体示例：
    ```json
    {"user_input": "帮我把这段视频的字幕翻译成英文并生成中文配音"}
    ```

    响应示例：
    ```json
    {
      "user_input": "帮我把这段视频的字幕翻译成英文并生成中文配音",
      "skills": [
        {"skill": "subtitle", "degree": 1.0},
        {"skill": "audio", "degree": 0.9}
      ]
    }
    ```
    """
    try:
        scores = get_router().select(req.user_input)
    except RuntimeError as e:
        # 配置错误（如缺少 API Key）
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        # LLM 调用或解析失败
        raise HTTPException(
            status_code=502, detail=f"路由选择失败: {e}"
        )

    return RouteResponse(user_input=req.user_input, skills=scores)
