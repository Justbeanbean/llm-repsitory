"""Provider 接口契约：业务流程不依赖具体 SDK。

新供应商接入 = 实现本 Protocol + 在 ProviderRegistry 注册一行。
- complete 统一返回 ProviderResult（内容 + 用量 + 供应商请求 ID）
- stream 返回统一事件流：
    {"type": "delta", "delta": str}            增量文本
    {"type": "usage", "usage": Usage}          流末用量（stream_options.include_usage）
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol

from llm_gateway.core.models import CandidateConfig, Message, ProviderResult

ProviderStream = AsyncIterator[dict[str, Any]]


class Provider(Protocol):
    """供应商 Adapter 的统一接口。"""

    provider_id: str

    async def complete(
        self,
        config: CandidateConfig,
        messages: list[Message],
        timeout_seconds: float,
        response_schema: dict[str, Any] | None,
    ) -> ProviderResult: ...

    def stream(
        self,
        config: CandidateConfig,
        messages: list[Message],
        timeout_seconds: float,
    ) -> ProviderStream: ...
