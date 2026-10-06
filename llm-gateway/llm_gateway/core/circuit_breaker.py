"""进程内熔断器（按 provider:provider_model 维度）。

- 仅 L1（网络）/ L2（5xx）类失败计入熔断（由 service 层判定类别）
- closed → 连续失败达阈值 → open（直接跳过该候选）
- open → 冷却期到 → half_open（放行一次探测）→ 成功回 closed / 失败回 open

刻意保留的边界：单进程内存实现，多副本部署需迁移共享存储。
"""
from __future__ import annotations

import time


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 30.0) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        # key → {"state", "failures", "opened_at"}
        self._states: dict[str, dict] = {}

    def allow(self, key: str) -> bool:
        state = self._states.get(key)
        if state is None or state["state"] == "closed":
            return True
        if state["state"] == "open":
            if time.monotonic() - state["opened_at"] >= self._recovery_timeout:
                state["state"] = "half_open"   # 放行一次探测
                return True
            return False
        return True  # half_open：探测调用放行

    def record_success(self, key: str) -> None:
        self._states.pop(key, None)   # 重置为 closed

    def record_failure(self, key: str) -> None:
        state = self._states.setdefault(key, {"state": "closed", "failures": 0, "opened_at": 0.0})
        if state["state"] == "half_open":
            # 探测失败：立即重新打开
            state["state"] = "open"
            state["opened_at"] = time.monotonic()
            return
        state["failures"] += 1
        if state["failures"] >= self._failure_threshold:
            state["state"] = "open"
            state["opened_at"] = time.monotonic()

    def state_of(self, key: str) -> str:
        return self._states.get(key, {"state": "closed"})["state"]

    def status(self) -> dict[str, str]:
        """全部熔断键状态（/admin/routes 观测用）。"""
        return {key: state["state"] for key, state in self._states.items()}
