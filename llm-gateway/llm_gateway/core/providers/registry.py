"""供应商注册表：provider_id → Adapter 实例。

新增供应商 = 一个 Adapter 文件 + 一行 register()，其余零改动；
动态切换供应商 = 改 providers.yaml 后重启/热加载，调用方零感知。
"""
from __future__ import annotations

from llm_gateway.core.exceptions import GatewayError
from llm_gateway.core.providers.base import Provider


class ProviderRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, Provider] = {}

    def register(self, provider: Provider) -> None:
        self._adapters[provider.provider_id] = provider

    def get(self, provider_id: str) -> Provider:
        adapter = self._adapters.get(provider_id)
        if adapter is None:
            raise GatewayError(
                "gateway_misconfigured",
                f"供应商 {provider_id} 未注册",
                503,
            )
        return adapter
