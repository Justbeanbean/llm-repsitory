"""统一重试执行器：网络类失败（L1/L2）的唯一重试实现。

- 重试条件唯一依据 ProviderError.retryable（分类点在 core Adapter，单一事实源）
- 指数退避 + 随机抖动；429 优先采用上游 Retry-After（受 max_delay 封顶）
- 修复重试（L4 反喂）不在此处，由 service 层编排——网络/修复两套预算分离
- Provider SDK 内置重试保持关闭（max_retries=0），重试口径全局唯一
"""
from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from llm_gateway.core.exceptions import DeadlineExceededError, ProviderError
from llm_gateway.core.logging_setup import get_logger

logger = get_logger("core.retry")

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3              # 单个 route（同模型）内的尝试次数
    base_delay: float = 0.2            # 首次退避（秒）
    max_delay: float = 5.0             # 退避上限
    backoff_factor: float = 2.0        # 指数退避系数
    jitter: bool = True                # 抖动，防止雪崩


def compute_delay(policy: RetryPolicy, failed_attempt: int, retry_after: float | None) -> float:
    if retry_after is not None:
        # 429 优先上游 Retry-After，仍受 max_delay 封顶
        return min(retry_after, policy.max_delay)
    delay = min(
        policy.base_delay * (policy.backoff_factor ** (failed_attempt - 1)),
        policy.max_delay,
    )
    if policy.jitter:
        delay *= random.uniform(0.5, 1.0)
    return delay


async def retry_async(
    factory: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy,
    context: dict[str, Any],
    deadline: float | None = None,
) -> tuple[T, int]:
    """执行带重试的调用，返回 (结果, 实际尝试次数)。

    - 不可重试或耗尽时原样上抛 ProviderError，并回填 exc.attempts
    - deadline（Run 预算）耗尽时抛 DeadlineExceededError：重试/fallback/修复
      共用同一预算，由 service 层统一转为 504
    """
    attempt = 0
    while True:
        attempt += 1
        if deadline is not None and time.perf_counter() >= deadline:
            raise DeadlineExceededError()
        try:
            return await factory(), attempt
        except ProviderError as exc:
            exc.attempts = attempt
            if not exc.retryable or attempt >= policy.max_attempts:
                logger.error(
                    "provider_call_failed",
                    extra={
                        **context,
                        "attempt": attempt,
                        "failure_class": exc.failure_class.value,
                        "error_code": exc.error_code,
                        "exhausted": not exc.retryable,
                    },
                )
                raise
            delay = compute_delay(policy, attempt, exc.retry_after)
            logger.warning(
                "retry_scheduled",
                extra={
                    **context,
                    "attempt": attempt,
                    "failure_class": exc.failure_class.value,
                    "error_code": exc.error_code,
                    "delay_s": round(delay, 3),
                    "retry_after": exc.retry_after,
                },
            )
            await asyncio.sleep(delay)
