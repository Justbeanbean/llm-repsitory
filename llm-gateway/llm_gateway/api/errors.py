"""统一错误出口：OpenAI 规范错误体 {error: {message, type, code, param}}。

永不包含堆栈、URL、密钥环境变量名或上游原始错误。
"""
from __future__ import annotations

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from llm_gateway.core.exceptions import GatewayError
from llm_gateway.core.logging_setup import get_logger

logger = get_logger("api.error")


def _error_type(status_code: int) -> str:
    """错误类型映射（OpenAI 规范），供 OpenAI SDK 客户端结构化解析。"""
    if status_code == 401:
        return "authentication_error"
    if status_code == 429:
        return "rate_limit_error"
    if 400 <= status_code < 500:
        return "invalid_request_error"
    return "api_error"


def _error_body(message: str, status_code: int, code: str) -> dict:
    return {"error": {"message": message, "type": _error_type(status_code), "code": code, "param": None}}


async def gateway_error_handler(_: Request, exc: GatewayError) -> JSONResponse:
    logger.warning(
        "request_failed",
        extra={
            "error_code": exc.code,
            "status": exc.status_code,
            "failure_class": exc.failure_class.value if exc.failure_class else None,
        },
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_body(exc.message, exc.status_code, exc.code),
        headers=exc.headers,
    )


async def http_exception_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    """HTTPException（如流式入口预校验 400）也统一为 OpenAI 错误体。"""
    detail = exc.detail if isinstance(exc.detail, dict) else {}
    message = str(detail.get("message") or exc.detail or "请求无效")
    code = str(detail.get("code") or "invalid_request")
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_body(message, exc.status_code, code),
        headers=getattr(exc, "headers", None),
    )


async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    """请求体校验失败（未支持字段/非法组合）→ 422 OpenAI 错误体，模型调用前明确失败。"""
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "message": "Invalid request",
                "type": "invalid_request_error",
                "code": "validation_error",
                "param": None,
                "details": exc.errors(),
            }
        },
    )


async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    """未预期异常（bug）→ 500 OpenAI 错误体。

    堆栈已由中间件以 request_failed_unhandled 记录（logger.exception）；
    响应体只含通用文案，绝不外泄堆栈与内部信息。
    """
    return JSONResponse(
        status_code=500,
        content=_error_body("网关内部错误，请稍后重试", 500, "internal_error"),
    )
