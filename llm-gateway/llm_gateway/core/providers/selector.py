"""模型路由：公开别名解析 → 策略排序（priority / weighted）→ 能力校验。

- priority：按 yaml 列表顺序，首个为主、其余为 fallback 链
- weighted：平滑加权轮询选主，剩余候选按权重降序作 fallback 链
- 结构化能力过滤：不支持 Structured Output 的候选被跳过（阻止不等价 fallback），
  全部不支持 → 400
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm_gateway.core.exceptions import GatewayError
from llm_gateway.core.models import CandidateConfig, ModelRoute
from llm_gateway.core.providers.base import Provider
from llm_gateway.core.providers.registry import ProviderRegistry


@dataclass(frozen=True)
class Route:
    """一条可执行的候选路由：公开别名 + 候选配置 + 已解析的 Adapter + 路由策略。"""

    alias: str
    candidate: CandidateConfig
    provider: Provider
    strategy: str = "priority"


class ModelSelector:
    def __init__(self, routes: dict[str, ModelRoute], registry: ProviderRegistry) -> None:
        self._routes = routes
        self._registry = registry
        # alias → {candidate_alias: 当前平滑权重}，进程内轮询状态
        self._wrr_state: dict[str, dict[str, int]] = {}

    def resolve(
        self,
        alias: str,
        *,
        response_schema: dict[str, Any] | None = None,
        require_responses: bool = False,
    ) -> list[Route]:
        """返回排序后的候选路由链。别名未知 → 400。

        - response_schema：过滤不支持 Structured Output 的候选（阻止不等价 fallback）
        - require_responses：/v1/responses 协议要求候选 api=both
        """
        model = self._routes.get(alias)
        if model is None:
            raise GatewayError("unknown_model", "模型不在 Gateway 允许列表中", 400)

        candidates = list(model.candidates)
        if require_responses:
            candidates = [c for c in candidates if c.api == "both"]
            if not candidates:
                raise GatewayError(
                    "responses_api_unsupported", "该模型的候选均不支持 Responses API", 400
                )
        if response_schema is not None:
            candidates = [c for c in candidates if c.supports_structured_output]
            if not candidates:
                raise GatewayError(
                    "structured_output_unsupported",
                    "模型不支持 Structured Output",
                    400,
                )

        ordered = self._order(model, candidates)
        return [
            Route(
                alias=alias,
                candidate=c,
                provider=self._registry.get(c.adapter),
                strategy=model.strategy,
            )
            for c in ordered
        ]

    def _order(
        self,
        model: ModelRoute,
        candidates: list[CandidateConfig],
    ) -> list[CandidateConfig]:
        if model.strategy == "weighted" and len(candidates) > 1:
            primary = self._smooth_weighted_pick(model.alias, candidates)
            rest = sorted(
                (c for c in candidates if c is not primary),
                key=lambda c: -c.weight,
            )
            return [primary, *rest]
        return candidates

    def _smooth_weighted_pick(
        self,
        alias: str,
        candidates: list[CandidateConfig],
    ) -> CandidateConfig:
        state = self._wrr_state.setdefault(alias, {c.alias: 0 for c in candidates})
        total = sum(c.weight for c in candidates)
        best = max(candidates, key=lambda c: state.get(c.alias, 0) + c.weight)
        for candidate in candidates:
            state[candidate.alias] = (
                state.get(candidate.alias, 0) + candidate.weight - (total if candidate is best else 0)
            )
        return best
