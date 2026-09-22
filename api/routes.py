"""FastAPI 路由：暴露 /route 接口供外部调用。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from intent.selector import SkillRouter

from .schemas import RouteResponse

router = APIRouter(prefix="/router", tags=["router"])

# 模块级单例：复用 LLM 客户端连接
_router_instance: SkillRouter | None = None


def get_router() -> SkillRouter:
    global _router_instance
    if _router_instance is None:
        _router_instance = SkillRouter()
    return _router_instance


@router.get("/route", response_model=RouteResponse, summary="技能路由选择")
def route(
    user_input: str = Query(
        ..., description="用户原始输入文本", min_length=1
    ),
    num: int = Query(
        2, description="返回的技能个数上限", ge=1, le=10
    ),
) -> RouteResponse:
    """根据用户输入选择置信度最高的 num 个技能（默认 2 个）。

    请求示例（参数走 query string）：
    ```
    GET /router/route?user_input=帮我把这段视频的字幕翻译成英文并生成中文配音&num=2
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
        scores = get_router().select(user_input, num=num)
    except RuntimeError as e:
        # 配置错误（如缺少 API Key）
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        # LLM 调用或解析失败
        raise HTTPException(
            status_code=502, detail=f"路由选择失败: {e}"
        )

    return RouteResponse(user_input=user_input, skills=scores)
