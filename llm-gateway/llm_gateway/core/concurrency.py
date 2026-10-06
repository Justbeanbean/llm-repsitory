"""进程内并发守卫：每调用方并发（非阻塞拒绝）+ 每供应商并发（排队等待）。

刻意保留的边界：单进程内存实现，多副本部署需迁移共享存储。
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager


class ConcurrencyGuard:
    """key → 当前占用计数；事件循环单线程内 dict 操作原子。"""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def try_acquire(self, key: str, limit: int) -> bool:
        """非阻塞：超出 limit 返回 False（调用方侧 → 429）。"""
        if limit <= 0:
            return True
        if self._counts.get(key, 0) >= limit:
            return False
        self._counts[key] = self._counts.get(key, 0) + 1
        return True

    def release(self, key: str) -> None:
        self._counts[key] = max(0, self._counts.get(key, 0) - 1)

    @asynccontextmanager
    async def slot(self, key: str, limit: int):
        """阻塞等待型：供应商侧限并发（超限排队而非拒绝）。"""
        if limit <= 0:
            yield
            return
        while not self.try_acquire(key, limit):
            await asyncio.sleep(0.01)
        try:
            yield
        finally:
            self.release(key)
