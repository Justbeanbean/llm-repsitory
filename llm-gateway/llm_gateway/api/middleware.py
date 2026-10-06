"""请求上下文中间件（纯 ASGI 实现，流式安全）。

- 生成/透传 x-request-id，写入 contextvars，全链路日志自动携带
  （纯 ASGI 包裹整个调用周期，SSE 流式响应期间 request_id 不丢）
- 入参侧脱敏：调用方传入的 x-request-id 先 sanitize 再使用
- 请求出入口日志：method/path/status/latency，不记录消息体
"""
from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

from llm_gateway.core.logging_setup import (
    get_logger,
    request_id_var,
    run_id_var,
    step_id_var,
)
from llm_gateway.core.masking import sanitize

logger = get_logger("api.http")


class RequestContextMiddleware:
    def __init__(self, app: Callable[..., Awaitable[None]]) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", [])
        }
        request_id = sanitize(headers.get("x-request-id", ""))[:64] or uuid4().hex
        token = request_id_var.set(request_id)
        # Run / Step / Call 关联：调用方经请求头传入，贯穿日志与账本（脱敏后截断）
        run_token = run_id_var.set(sanitize(headers.get("x-run-id", ""))[:64] or "-")
        step_token = step_id_var.set(sanitize(headers.get("x-step-id", ""))[:64] or "-")
        started = time.perf_counter()
        status = 0

        async def send_wrapper(message: dict[str, Any]) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message["headers"] = [
                    *message.get("headers", []),
                    (b"x-request-id", request_id.encode("latin-1")),
                ]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
            logger.info(
                "request_completed",
                extra={
                    "method": scope.get("method"),
                    "path": scope.get("path"),
                    "status": status,
                    "latency_ms": round((time.perf_counter() - started) * 1000),
                },
            )
        except Exception:
            logger.exception(
                "request_failed_unhandled",
                extra={"method": scope.get("method"), "path": scope.get("path")},
            )
            raise
        finally:
            request_id_var.reset(token)
            run_id_var.reset(run_token)
            step_id_var.reset(step_token)
