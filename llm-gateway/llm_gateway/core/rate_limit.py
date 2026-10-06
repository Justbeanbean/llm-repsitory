"""进程内令牌桶限流（按调用方）。

刻意保留的边界：单进程内存实现，多副本部署需迁移到 Redis 等共享存储。
"""
from __future__ import annotations

import time

from llm_gateway.core.security import Caller


class TokenBucketLimiter:
    """每个调用方一个桶：容量 burst，匀速 requests_per_second 回填。"""

    def __init__(self) -> None:
        # name → (当前令牌数, 上次回填时间)
        self._buckets: dict[str, tuple[float, float]] = {}

    def allow(self, caller: Caller) -> bool:
        now = time.monotonic()
        tokens, last = self._buckets.get(caller.name, (float(caller.burst), now))
        tokens = min(float(caller.burst), tokens + (now - last) * caller.requests_per_second)
        allowed = tokens >= 1.0
        self._buckets[caller.name] = (tokens - 1.0 if allowed else tokens, now)
        return allowed
