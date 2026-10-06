"""鉴权与限流依赖：Bearer API Key + 令牌桶 + 每调用方并发限制（api 层入口执行）。

开发模式（yaml api_keys 留空）：不校验调用方，匿名放行（限流仍按全局配置生效）。
"""
from __future__ import annotations

from typing import AsyncIterator

from fastapi import Request

from llm_gateway.core.exceptions import GatewayError
from llm_gateway.core.logging_setup import caller_fingerprint_var
from llm_gateway.core.security import Caller, authenticate


async def require_caller(request: Request) -> AsyncIterator[Caller]:
    """业务端点统一依赖：鉴权 → 令牌桶 → 调用方并发 → 写入调用方指纹上下文。

    yield 依赖：请求结束时释放并发槽位（client 断开同样走 finally）。
    """
    settings = request.app.state.settings

    if settings.auth_required:
        caller = authenticate(
            request.headers.get("authorization", ""), settings.api_keys
        )
        if caller is None:
            raise GatewayError("invalid_api_key", "缺少或无效的 API Key", 401)
    else:
        # 开发模式：api_keys 留空 → 匿名放行
        caller = Caller(
            name="anonymous",
            fingerprint="anonymous",
            requests_per_second=settings.rate_limit.requests_per_second,
            burst=settings.rate_limit.burst,
            max_concurrency=settings.max_caller_concurrency,
        )

    if settings.rate_limit.enabled and not request.app.state.rate_limiter.allow(caller):
        raise GatewayError(
            "rate_limited", "请求过于频繁，请稍后重试", 429,
            headers={"Retry-After": "1"},
        )

    guard = request.app.state.concurrency_guard
    slot = f"caller:{caller.name}"
    if not guard.try_acquire(slot, caller.max_concurrency):
        raise GatewayError(
            "concurrency_limit", "超出调用方并发上限", 429,
            headers={"Retry-After": "1"},
        )

    # contextvar 在请求 task 内生效，账本与日志自动携带
    caller_fingerprint_var.set(caller.fingerprint)
    try:
        yield caller
    finally:
        guard.release(slot)
