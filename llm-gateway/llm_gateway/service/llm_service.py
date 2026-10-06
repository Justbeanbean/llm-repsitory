"""LLM 调用编排（service 层核心）：L1-L4 判定链 + 双重试预算 + 路由/fallback + 熔断 + Run 预算。

一次请求的完整链路：
  鉴权/限流/并发（api 层依赖）→ Prompt Bundle 注入 → 候选路由选择与重试
  → 上游 HTTP 请求（core providers，供应商并发受限）→ 用量记账（ledger_service）

判定链（「模型是否成功」的最终裁决在本层）：
  L1/L2 网络/状态码 → core.retry_async 统一重试（指数退避 + Retry-After）
  L3    协议 choice → core Adapter 显式判定，取不到文本 → 422
  L4    Schema/业务规则 → validation_service；取得到文本 → 反喂重修一次（同 route）

策略铁律：
  - 内容问题（L3/L4）一律不换 provider，也不计入熔断
  - 取得到文本就反喂重修一次（同 route）；取不到就 422
  - 纯文本请求完全跳过 L4
  - 重试、Fallback 和修复调用共用同一 Run 预算（deadline），耗尽 → 504
  - 客户端取消：下游生成器关闭 + 唯一 cancelled 终态
  - TTFT 口径 = 首个业务可见 delta（非连接建立、非首事件）
"""
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from contextlib import aclosing
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from llm_gateway.core.circuit_breaker import CircuitBreaker
from llm_gateway.core.concurrency import ConcurrencyGuard
from llm_gateway.core.exceptions import (
    FAILURE_LAYER,
    DeadlineExceededError,
    FailureClass,
    GatewayError,
    ProviderError,
)
from llm_gateway.core.logging_setup import (
    caller_fingerprint_var,
    get_logger,
    request_id_var,
    run_id_var,
    step_id_var,
)
from llm_gateway.core.models import (
    CallTrace,
    LLMRequest,
    LLMResponse,
    Message,
    ProviderResult,
    Usage,
    schema_fingerprint,
)
from llm_gateway.core.providers.selector import ModelSelector, Route
from llm_gateway.core.retry import RetryPolicy, retry_async
from llm_gateway.service.ledger_service import LedgerService, calculate_cost
from llm_gateway.service.prompt_service import PromptContext, PromptService
from llm_gateway.service.validation_service import ValidationService

logger = get_logger("service.llm")

_REPAIR_INSTRUCTION = (
    "你上面的输出不符合要求：{errors}。"
    "请重新回答，只返回一个合法 JSON 对象，严格符合此前的 JSON Schema 与业务规则，"
    "不要返回 Markdown 或任何额外文字。"
)

# 只有 L1（网络）与 L2（5xx）失败计入熔断；429 / 内容问题不熔断
_BREAKER_TRIP_CLASSES = frozenset({FailureClass.NETWORK, FailureClass.PROVIDER_SERVER})

_DEFAULT_TIMEOUT_SECONDS = 30.0


class LLMService:
    def __init__(
        self,
        selector: ModelSelector,
        prompt_service: PromptService,
        validation_service: ValidationService,
        ledger: LedgerService,
        retry_policy: RetryPolicy,
        breaker: CircuitBreaker,
        *,
        run_budget_seconds: float = 120.0,
        concurrency_guard: ConcurrencyGuard | None = None,
        max_concurrency_per_candidate: int = 8,
        max_repair_attempts: int = 1,
        checkpoint_config: dict[str, Any] | None = None,
    ) -> None:
        self._selector = selector
        self._prompts = prompt_service
        self._validation = validation_service
        self._ledger = ledger
        self._retry_policy = retry_policy
        self._breaker = breaker
        self._run_budget_seconds = run_budget_seconds
        self._guard = concurrency_guard or ConcurrencyGuard()
        self._max_concurrency_per_candidate = max_concurrency_per_candidate
        self._max_repair_attempts = max(0, max_repair_attempts)  # yaml: structured_output_retries
        self._checkpoint_config = checkpoint_config or {}        # yaml: stream_checkpoint

    # ------------------------------------------------------------------
    # 公共小工具
    # ------------------------------------------------------------------

    @staticmethod
    def _breaker_key(route: Route) -> str:
        return f"{route.candidate.provider}:{route.candidate.provider_model}"

    @staticmethod
    def _provider_slot_key(route: Route) -> str:
        return f"provider:{route.candidate.provider}:{route.candidate.provider_model}"

    # ------------------------------------------------------------------
    # 非流式
    # ------------------------------------------------------------------

    async def complete(self, request: LLMRequest, *, responses_api: bool = False) -> LLMResponse:
        """responses_api 为内部路由提示（api 层传入，调用方不可伪造）：
        True = /v1/responses 协议，候选必须支持 api=both。"""
        request_id = request_id_var.get() or uuid4().hex
        caller = caller_fingerprint_var.get("-")
        started = time.perf_counter()
        deadline = started + self._run_budget_seconds

        ctx = self._prompts.build_context(request.messages, request.prompt)
        effective_schema = request.response_schema or (
            ctx.template.output_schema if ctx.template else None
        )
        effective_timeout = (
            request.timeout_seconds
            or (ctx.template.default_timeout_seconds if ctx.template else None)
            or _DEFAULT_TIMEOUT_SECONDS
        )
        business_model = ctx.template.business_model if ctx.template else None
        routes = self._selector.resolve(
            request.model,
            response_schema=effective_schema,
            require_responses=responses_api,
        )
        logger.info(
            "model_selected",
            extra={
                "requested_model": request.model,
                "route_chain": [route.candidate.alias for route in routes],
                "prompt_name": request.prompt.name if request.prompt else None,
                "stream": False,
            },
        )

        stats: dict[str, Any] = {
            "network_attempts": 0,
            "repair_attempts": 0,
            "fallback_count": 0,
            "final_candidate": None,
            "final_provider": None,
            "upstream_request_id": None,
            "route_decisions": [],
        }
        try:
            for index, route in enumerate(routes):
                self._check_deadline(deadline)
                breaker_key = self._breaker_key(route)
                if not self._breaker.allow(breaker_key):
                    self._decide(stats, route, "skipped", reason="circuit_open")
                    logger.warning(
                        "circuit_open_skip",
                        extra={"model": route.alias, "candidate": route.candidate.alias},
                    )
                    continue
                try:
                    content, usage = await self._attempt_route(
                        route, ctx, request, stats,
                        effective_schema, effective_timeout, business_model, deadline,
                    )
                except ProviderError as exc:
                    if exc.failure_class in _BREAKER_TRIP_CLASSES:
                        self._breaker.record_failure(breaker_key)
                    if exc.fallback_worthy and index < len(routes) - 1:
                        stats["fallback_count"] += 1
                        self._decide(
                            stats, route, "failed",
                            reason=f"fallback_after_{exc.error_code}",
                            error_code=exc.error_code,
                        )
                        logger.warning(
                            "fallback_triggered",
                            extra={
                                "from_candidate": route.candidate.alias,
                                "to_candidate": routes[index + 1].candidate.alias,
                                "failure_class": exc.failure_class.value,
                                "error_code": exc.error_code,
                            },
                        )
                        continue
                    self._decide(
                        stats, route, "failed",
                        reason="no_more_candidates" if exc.fallback_worthy else "not_fallback_worthy",
                        error_code=exc.error_code,
                    )
                    raise self._to_gateway_error(exc) from exc

                self._breaker.record_success(breaker_key)
                stats["final_candidate"] = route.candidate.alias
                stats["final_provider"] = route.candidate.provider
                self._decide(
                    stats, route, "selected",
                    reason=(
                        "priority_first" if route.strategy == "priority" else "weighted_pick"
                    )
                    if index == 0
                    else "fallback_candidate",
                )
                latency_ms = int((time.perf_counter() - started) * 1000)
                response = LLMResponse(
                    request_id=request_id,
                    model=route.alias,
                    content=content,
                    parsed=json.loads(content) if effective_schema is not None else None,
                    usage=usage,
                    latency_ms=latency_ms,
                    attempts=stats["network_attempts"] + stats["repair_attempts"],
                )
                await self._record_success(
                    caller, request_id, request, route, usage, None, latency_ms, stats,
                    effective_schema=effective_schema, template=ctx.template,
                )
                return response

            # 走到这里 = 全部候选被熔断跳过
            raise GatewayError("model_unavailable", "所有候选模型均不可用或熔断打开", 503)
        except DeadlineExceededError as exc:
            # Run 预算耗尽：重试/fallback/修复共用同一预算，立即停止一切尝试
            error = GatewayError(
                "deadline_exceeded", "调用预算耗尽", 504,
                failure_class=FailureClass.UNKNOWN,
            )
            await self._record_failure(
                caller, request_id, request, error,
                int((time.perf_counter() - started) * 1000), stats,
                effective_schema=effective_schema, template=ctx.template,
            )
            raise error from exc
        except GatewayError as exc:
            await self._record_failure(
                caller, request_id, request, exc,
                int((time.perf_counter() - started) * 1000), stats,
                effective_schema=effective_schema, template=ctx.template,
            )
            raise
        except asyncio.CancelledError:
            # 客户端断开：唯一 cancelled 终态
            await self._record_cancelled(
                caller, request_id, request,
                int((time.perf_counter() - started) * 1000), stats,
                effective_schema=effective_schema, template=ctx.template,
            )
            raise

    async def _attempt_route(
        self,
        route: Route,
        ctx: PromptContext,
        request: LLMRequest,
        stats: dict[str, Any],
        effective_schema: dict[str, Any] | None,
        effective_timeout: float,
        business_model: str | None,
        deadline: float,
    ) -> tuple[str, Usage]:
        """单 route 判定链：L1/L2 重试 → L3 协议 → L4 校验 + 反喂重修一次。

        - ProviderError 且 fallback_worthy → 上抛由外层换 route
        - L3/L4 内容问题 → 直接抛 GatewayError(422)，外层不换 route、不熔断
        """
        candidate = route.candidate
        provider = route.provider
        context: dict[str, Any] = {
            "model": route.alias,
            "candidate": candidate.alias,
            "provider": candidate.provider,
            "provider_model": candidate.provider_model,
        }
        slot_key = self._provider_slot_key(route)

        async def call(msgs: list[Message]) -> ProviderResult:
            remaining = max(0.001, deadline - time.perf_counter())
            # 请求显式值 / Bundle 默认 / 候选（provider 级）默认，取最小者
            timeout = min(effective_timeout, candidate.timeout_seconds, remaining)
            async with self._guard.slot(slot_key, self._max_concurrency_per_candidate):
                return await provider.complete(
                    candidate, msgs, timeout, effective_schema
                )

        result, _ = await self._call_with_retry(
            call, ctx.messages, context, stats, deadline
        )
        content, usage, upstream_id = result.content, result.usage, result.upstream_id
        stats["upstream_request_id"] = upstream_id
        logger.info(
            "provider_call_completed",
            extra={
                **context,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "usage_missing": usage.usage_missing,
                "upstream_id": upstream_id,
            },
        )

        # 纯文本请求：完全跳过 L4
        if effective_schema is None:
            return content, usage

        # L4：schema + 业务规则校验
        verdict = self._validation.check(
            content, effective_schema, business_model=business_model
        )
        if verdict.ok:
            return content, usage

        if not content.strip():
            # 空文本视同取不到 → 422，不重试不换
            raise GatewayError(
                "empty_content", "上游未返回文本内容", 422,
                failure_class=FailureClass.PROTOCOL,
            )

        # 取得到文本 → 反喂重修（次数 = structured_output_retries，默认 1；同 route，绝不换 provider）
        current = content
        current_usage = usage
        for repair_index in range(self._max_repair_attempts):
            logger.warning(
                "repair_retry_scheduled",
                extra={
                    **context,
                    "error_code": verdict.error_code,
                    "repair_index": repair_index + 1,
                },
            )
            repair_messages = [
                *ctx.messages,
                Message(role="assistant", content=current[:2000]),
                Message(role="user", content=_REPAIR_INSTRUCTION.format(errors=verdict.errors)),
            ]
            stats["repair_attempts"] += 1
            repair_result, _ = await self._call_with_retry(
                call, repair_messages, {**context, "phase": "repair"}, stats, deadline
            )
            stats["upstream_request_id"] = repair_result.upstream_id or upstream_id
            current_usage = current_usage + repair_result.usage
            verdict = self._validation.check(
                repair_result.content, effective_schema, business_model=business_model
            )
            if verdict.ok:
                return repair_result.content, current_usage
            current = repair_result.content
        raise GatewayError(
            verdict.error_code,
            "模型输出不符合校验且修复重试失败",
            422,
            failure_class=FailureClass.SCHEMA,
        )

    async def _call_with_retry(
        self,
        call: Any,
        msgs: list[Message],
        context: dict[str, Any],
        stats: dict[str, Any],
        deadline: float,
    ) -> tuple[Any, int]:
        """包一层 retry_async，失败时也把尝试次数累计进 stats（供失败 trace 归因）。"""
        try:
            result = await retry_async(
                lambda: call(msgs),
                policy=self._retry_policy,
                context=context,
                deadline=deadline,
            )
        except ProviderError as exc:
            stats["network_attempts"] += exc.attempts
            raise
        stats["network_attempts"] += result[1]
        return result

    # ------------------------------------------------------------------
    # 流式
    # ------------------------------------------------------------------

    def validate_stream_request(self, request: LLMRequest, *, responses_api: bool = False) -> None:
        """流开始前拦截 400 类错误（模型未知 / 模板缺失 / 变量缺失）。"""
        self._prompts.build_context(request.messages, request.prompt)
        self._selector.resolve(request.model, require_responses=responses_api)

    async def stream(
        self, request: LLMRequest, *, responses_api: bool = False
    ) -> AsyncIterator[dict[str, Any]]:
        """SSE 事件流（dict 形式，由 api 层编码为各协议格式）。

        首个内容块前可重试/fallback；发出后仅发流内错误，绝不换模型续写。
        客户端断开（生成器被 aclose）→ 下游生成器关闭 + 唯一 cancelled 终态。
        """
        request_id = request_id_var.get() or uuid4().hex
        caller = caller_fingerprint_var.get("-")
        started = time.perf_counter()
        deadline = started + self._run_budget_seconds

        ctx = self._prompts.build_context(request.messages, request.prompt)
        effective_timeout = (
            request.timeout_seconds
            or (ctx.template.default_timeout_seconds if ctx.template else None)
            or _DEFAULT_TIMEOUT_SECONDS
        )
        routes = self._selector.resolve(request.model, require_responses=responses_api)
        logger.info(
            "model_selected",
            extra={
                "requested_model": request.model,
                "route_chain": [route.candidate.alias for route in routes],
                "prompt_name": request.prompt.name if request.prompt else None,
                "stream": True,
            },
        )

        attempts = 0
        fallback_count = 0
        emitted = False
        ttft_ms: int | None = None
        final_route: Route | None = None
        decisions: list[dict[str, Any]] = []
        last_error: Exception | None = None
        # 流式 usage：上游 stream_options.include_usage 回传则账本可得（否则三态=未知）
        stream_usage = Usage(usage_missing=True)
        stream_upstream_id: str | None = None

        # 流式检查点（yaml stream_checkpoint；默认关闭，避免未评估隐私策略前保存正文）
        checkpoint_on = bool(self._checkpoint_config.get("enabled", False))
        checkpoint_max_chars = int(self._checkpoint_config.get("max_chars", 200_000))
        checkpoint_interval = float(self._checkpoint_config.get("flush_interval_seconds", 1.0))
        checkpoint_buffer: list[str] = []
        checkpoint_chars = 0
        checkpoint_last_flush = started

        async def flush_checkpoint(status: str) -> None:
            if checkpoint_on:
                await self._ledger.save_checkpoint(
                    request_id=request_id,
                    model=request.model,
                    status=status,
                    content="".join(checkpoint_buffer)[:checkpoint_max_chars],
                )

        try:
            for index, route in enumerate(routes):
                self._check_deadline(deadline)
                breaker_key = self._breaker_key(route)
                if not self._breaker.allow(breaker_key):
                    decisions.append(self._decision(route, "skipped", reason="circuit_open"))
                    logger.warning(
                        "circuit_open_skip",
                        extra={"model": route.alias, "candidate": route.candidate.alias, "stream": True},
                    )
                    continue
                attempts += 1
                try:
                    # aclosing：客户端断开时显式关闭下游生成器（取消传播到上游流）
                    async with (
                        self._guard.slot(
                            self._provider_slot_key(route), self._max_concurrency_per_candidate
                        ),
                        aclosing(
                            route.provider.stream(
                                route.candidate, ctx.messages, effective_timeout
                            )
                        ) as stream_iter,
                    ):
                        async for event in stream_iter:
                            if event.get("type") == "usage":
                                stream_usage = event["usage"]
                                continue
                            if event.get("type") == "meta":
                                stream_upstream_id = event.get("upstream_id")
                                continue
                            delta = event.get("delta")
                            if not delta:
                                continue
                            if ttft_ms is None:
                                # TTFT 口径：首个业务可见 delta（不是连接建立、不是首事件）
                                ttft_ms = int((time.perf_counter() - started) * 1000)
                            emitted = True
                            yield {"type": "content.delta", "delta": delta}
                            if checkpoint_on and checkpoint_chars < checkpoint_max_chars:
                                checkpoint_buffer.append(delta)
                                checkpoint_chars += len(delta)
                                if time.perf_counter() - checkpoint_last_flush >= checkpoint_interval:
                                    await flush_checkpoint("streaming")
                                    checkpoint_last_flush = time.perf_counter()
                except (ProviderError, GatewayError) as exc:
                    last_error = exc
                    if isinstance(exc, ProviderError):
                        if exc.failure_class in _BREAKER_TRIP_CLASSES:
                            self._breaker.record_failure(breaker_key)
                        if exc.fallback_worthy and not emitted:
                            fallback_count += 1
                            decisions.append(
                                self._decision(
                                    route, "failed",
                                    reason=f"fallback_after_{exc.error_code}",
                                    error_code=exc.error_code,
                                )
                            )
                            logger.warning(
                                "fallback_triggered",
                                extra={
                                    "from_candidate": route.candidate.alias,
                                    "failure_class": exc.failure_class.value,
                                    "error_code": exc.error_code,
                                    "stream": True,
                                },
                            )
                            continue
                    decisions.append(
                        self._decision(route, "failed", reason="stream_failed")
                    )
                    break
                self._breaker.record_success(breaker_key)
                final_route = route
                decisions.append(
                    self._decision(
                        route, "selected",
                        reason=(
                            "priority_first" if route.strategy == "priority" else "weighted_pick"
                        )
                        if index == 0
                        else "fallback_candidate",
                    )
                )
                break

            latency_ms = int((time.perf_counter() - started) * 1000)
            failure_class = (
                last_error.failure_class if isinstance(last_error, ProviderError) else None
            )

            if final_route is not None:
                await self._ledger.record(
                    self._build_trace(
                        caller=caller,
                        request_id=request_id,
                        request=request,
                        route=final_route,
                        usage=stream_usage,   # 上游回传则真实计数，否则三态=未知
                        ttft_ms=ttft_ms,
                        latency_ms=latency_ms,
                        attempts=attempts,
                        fallback_count=fallback_count,
                        network_attempts=attempts,
                        status="success",
                        template=ctx.template,
                        route_decisions=decisions,
                        stats={"upstream_request_id": stream_upstream_id},
                    )
                )
                await flush_checkpoint("completed")
                yield {"type": "response.completed", "model": final_route.alias}
                return

            logger.error(
                "upstream_stream_failed",
                extra={
                    "requested_model": request.model,
                    "attempts": attempts,
                    "failure_class": failure_class.value if failure_class else None,
                },
            )
            await self._ledger.record(
                self._build_trace(
                    caller=caller,
                    request_id=request_id,
                    request=request,
                    usage=Usage(usage_missing=True),
                    ttft_ms=ttft_ms,
                    latency_ms=latency_ms,
                    attempts=attempts,
                    fallback_count=fallback_count,
                    network_attempts=attempts,
                    status="failed",
                    error_code="upstream_stream_failed",
                    failure_class=failure_class,
                    template=ctx.template,
                    route_decisions=decisions,
                )
            )
            await flush_checkpoint("failed")
            yield {"type": "response.failed", "error": "upstream_stream_failed"}
        except DeadlineExceededError as exc:
            error = GatewayError(
                "deadline_exceeded", "调用预算耗尽", 504,
                failure_class=FailureClass.UNKNOWN,
            )
            await self._ledger.record(
                self._build_trace(
                    caller=caller,
                    request_id=request_id,
                    request=request,
                    usage=Usage(usage_missing=True),
                    ttft_ms=ttft_ms,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    attempts=attempts,
                    fallback_count=fallback_count,
                    network_attempts=attempts,
                    status="failed",
                    error_code=error.code,
                    template=ctx.template,
                    route_decisions=decisions,
                )
            )
            raise error from exc
        except (asyncio.CancelledError, GeneratorExit):
            # 客户端断开（生成器被 aclose → GeneratorExit / 任务取消 → CancelledError）：
            # 下游生成器已随 aclosing 显式关闭；这里只产生一个 cancelled 终态
            await self._ledger.record(
                self._build_trace(
                    caller=caller,
                    request_id=request_id,
                    request=request,
                    usage=Usage(usage_missing=True),
                    ttft_ms=ttft_ms,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    attempts=attempts,
                    fallback_count=fallback_count,
                    network_attempts=attempts,
                    status="cancelled",
                    error_code="client_disconnected",
                    template=ctx.template,
                    route_decisions=decisions,
                )
            )
            await flush_checkpoint("cancelled")
            raise

    # ------------------------------------------------------------------
    # 路由决策与错误映射
    # ------------------------------------------------------------------

    @staticmethod
    def _check_deadline(deadline: float) -> None:
        if time.perf_counter() >= deadline:
            raise DeadlineExceededError()

    @staticmethod
    def _decision(
        route: Route,
        action: str,
        *,
        reason: str,
        error_code: str | None = None,
    ) -> dict[str, Any]:
        """可解释 RouteDecision：候选 → 决策 → 理由（进 Trace）。"""
        return {
            "candidate": route.candidate.alias,
            "provider": route.candidate.provider,
            "provider_model": route.candidate.provider_model,
            "action": action,
            "reason": reason,
            **({"error_code": error_code} if error_code else {}),
        }

    @staticmethod
    def _decide(
        stats: dict[str, Any], route: Route, action: str, *, reason: str,
        error_code: str | None = None,
    ) -> None:
        stats["route_decisions"].append(
            LLMService._decision(route, action, reason=reason, error_code=error_code)
        )

    @staticmethod
    def _to_gateway_error(exc: ProviderError) -> GatewayError:
        if exc.failure_class == FailureClass.RATE_LIMIT:
            # 429 不掩盖为 502：调用方应感知限流并自行退避
            retry_after = str(int(exc.retry_after or 5))
            return GatewayError(
                "rate_limited", "上游持续限流，请稍后重试", 429,
                failure_class=exc.failure_class,
                headers={"Retry-After": retry_after},
            )
        if exc.failure_class == FailureClass.PROTOCOL:
            # 内容问题 → 422（请求合法但无法产出合格内容）
            return GatewayError(
                exc.error_code, exc.message, 422,
                failure_class=exc.failure_class,
            )
        if exc.failure_class == FailureClass.PROVIDER_CLIENT:
            # 上游 4xx 语义透传：上下文超长/非法参数等问题是调用方可修复的，
            # 折叠成 502 会让调用方误判为网关故障而盲目重试
            status = exc.status_code if exc.status_code and 400 <= exc.status_code < 500 else 502
            return GatewayError(
                exc.error_code, "上游拒绝请求", status,
                failure_class=exc.failure_class,
            )
        return GatewayError(
            "model_unavailable", "所有候选模型均不可用", 502,
            failure_class=exc.failure_class,
        )

    # ------------------------------------------------------------------
    # 账本
    # ------------------------------------------------------------------

    def _build_trace(
        self,
        *,
        caller: str,
        request_id: str,
        request: LLMRequest,
        route: Route | None = None,
        usage: Usage | None = None,
        ttft_ms: int | None,
        latency_ms: int,
        attempts: int,
        fallback_count: int,
        network_attempts: int,
        status: str,
        error_code: str | None = None,
        failure_class: FailureClass | None = None,
        repair_attempts: int = 0,
        template: Any = None,
        effective_schema: dict[str, Any] | None = None,
        stats: dict[str, Any] | None = None,
        route_decisions: list[dict[str, Any]] | None = None,
    ) -> CallTrace:
        usage = usage or Usage()
        return CallTrace(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc),
            caller_fingerprint=caller if caller != "-" else None,
            run_id=run_id_var.get("-") if run_id_var.get() != "-" else None,
            step_id=step_id_var.get("-") if step_id_var.get() != "-" else None,
            requested_model=request.model,
            final_candidate=(
                route.candidate.alias if route else (stats or {}).get("final_candidate")
            ),
            provider=route.candidate.provider if route else (stats or {}).get("final_provider"),
            upstream_request_id=(stats or {}).get("upstream_request_id"),
            prompt_name=(
                template.name if template is not None
                else (request.prompt.name if request.prompt else None)
            ),
            # 记录实际解析到的版本（"latest" 会被展开为真实版本号）
            prompt_version=(
                template.version if template is not None
                else (request.prompt.version if request.prompt else None)
            ),
            prompt_hash=template.content_hash if template is not None else None,
            schema_hash=schema_fingerprint(effective_schema) if effective_schema else None,
            price_version=route.candidate.price_version if route else None,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_tokens=usage.cached_tokens,
            usage_missing=usage.usage_missing,
            cost_usd=calculate_cost(route.candidate, usage) if route is not None else None,
            ttft_ms=ttft_ms,
            latency_ms=latency_ms,
            network_attempts=network_attempts,
            repair_attempts=repair_attempts,
            fallback_count=fallback_count,
            attempts=attempts,
            status=status,  # type: ignore[arg-type]
            error_code=error_code,
            failure_class=failure_class,
            failure_layer=FAILURE_LAYER.get(failure_class) if failure_class else None,
            route_decisions=route_decisions or (stats or {}).get("route_decisions"),
        )

    async def _record_success(
        self,
        caller: str,
        request_id: str,
        request: LLMRequest,
        route: Route,
        usage: Usage,
        ttft_ms: int | None,
        latency_ms: int,
        stats: dict[str, Any],
        *,
        effective_schema: dict[str, Any] | None,
        template: Any,
    ) -> None:
        await self._ledger.record(
            self._build_trace(
                caller=caller,
                request_id=request_id,
                request=request,
                route=route,
                usage=usage,
                ttft_ms=ttft_ms,
                latency_ms=latency_ms,
                attempts=stats["network_attempts"] + stats["repair_attempts"],
                fallback_count=stats["fallback_count"],
                network_attempts=stats["network_attempts"],
                repair_attempts=stats["repair_attempts"],
                status="success",
                template=template,
                effective_schema=effective_schema,
                stats=stats,
            )
        )

    async def _record_failure(
        self,
        caller: str,
        request_id: str,
        request: LLMRequest,
        exc: GatewayError,
        latency_ms: int,
        stats: dict[str, Any],
        *,
        effective_schema: dict[str, Any] | None,
        template: Any,
    ) -> None:
        await self._ledger.record(
            self._build_trace(
                caller=caller,
                request_id=request_id,
                request=request,
                usage=Usage(),
                ttft_ms=None,
                latency_ms=latency_ms,
                attempts=stats["network_attempts"] + stats["repair_attempts"],
                fallback_count=stats["fallback_count"],
                network_attempts=stats["network_attempts"],
                repair_attempts=stats["repair_attempts"],
                status="failed",
                error_code=exc.code,
                failure_class=exc.failure_class,
                template=template,
                effective_schema=effective_schema,
                stats=stats,
            )
        )

    async def _record_cancelled(
        self,
        caller: str,
        request_id: str,
        request: LLMRequest,
        latency_ms: int,
        stats: dict[str, Any],
        *,
        effective_schema: dict[str, Any] | None,
        template: Any,
    ) -> None:
        await self._ledger.record(
            self._build_trace(
                caller=caller,
                request_id=request_id,
                request=request,
                usage=Usage(),
                ttft_ms=None,
                latency_ms=latency_ms,
                attempts=stats["network_attempts"],
                fallback_count=stats["fallback_count"],
                network_attempts=stats["network_attempts"],
                status="cancelled",
                error_code="client_disconnected",
                template=template,
                effective_schema=effective_schema,
                stats=stats,
            )
        )
