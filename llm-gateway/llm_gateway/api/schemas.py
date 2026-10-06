"""API 契约门面：统一请求/响应模型。

领域模型定义在 core.models（三层共享），本模块作为 API 出口的聚合门面；
FastAPI 在入口校验请求、在 response_model 校验统一响应出口。
"""
from llm_gateway.core.models import (  # noqa: F401
    CallTrace,
    LLMRequest,
    LLMResponse,
    Message,
    PromptSelection,
    Usage,
)
