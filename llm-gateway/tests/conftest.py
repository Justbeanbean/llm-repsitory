"""测试夹具：Fake Adapter（稳定复现成功/限流/超时/流中断/非法输出/延迟）+ 应用工厂。"""
from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from llm_gateway.core.circuit_breaker import CircuitBreaker
from llm_gateway.core.exceptions import FailureClass, ProviderError
from llm_gateway.core.models import (
    CandidateConfig,
    ModelRoute,
    Price,
    PromptTemplate,
    ProviderResult,
    Usage,
)
from llm_gateway.core.providers.registry import ProviderRegistry
from llm_gateway.core.rate_limit import TokenBucketLimiter
from llm_gateway.core.retry import RetryPolicy
from llm_gateway.main import create_app
from llm_gateway.settings import ApiKeySetting, BreakerConfig, Settings


def net_error(retryable: bool = True) -> ProviderError:
    return ProviderError(
        FailureClass.NETWORK, "network_error", "测试网络错误", retryable=retryable
    )


def rate_limit_error(retry_after: float | None = None) -> ProviderError:
    return ProviderError(
        FailureClass.RATE_LIMIT, "rate_limited", "测试限流",
        retryable=True, retry_after=retry_after,
    )


def empty_content_error() -> ProviderError:
    return ProviderError(
        FailureClass.PROTOCOL, "empty_content", "上游未返回文本内容", retryable=False
    )


class FakeProvider:
    """脚本化 Fake Adapter：outcomes 按序消费，耗尽后走 default_error 或默认成功。

    outcome 形态：
      {"error": ProviderError}          抛出该错误
      {"content": str, "usage": Usage}  完成调用成功
      {"delta": str}                    流式产出一段文本
      {"delay": float}                  complete 延迟（模拟慢上游，供并发/deadline 测试）
    """

    def __init__(self, outcomes=None, default_error=None, provider_id="fake", delay=0.0):
        self.provider_id = provider_id
        self.outcomes = list(outcomes or [])
        self.default_error = default_error
        self.delay = delay
        self.complete_calls = 0
        self.stream_calls = 0
        self.stream_closed = False
        self.received: list[list] = []

    async def complete(self, config, messages, timeout_seconds, response_schema):
        self.complete_calls += 1
        self.received.append(list(messages))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if outcome.get("error"):
                raise outcome["error"]
            if outcome.get("delay"):
                await asyncio.sleep(outcome["delay"])
            return ProviderResult(
                content=outcome.get("content", "ok"),
                usage=outcome.get("usage", Usage(input_tokens=1, output_tokens=1)),
                upstream_id=f"fake-req-{self.complete_calls}",
            )
        if self.default_error is not None:
            raise self.default_error
        return ProviderResult(
            content="ok",
            usage=Usage(input_tokens=1, output_tokens=1),
            upstream_id=f"fake-req-{self.complete_calls}",
        )

    def stream(self, config, messages, timeout_seconds):
        return self._stream(config, messages, timeout_seconds)

    async def _stream(self, config, messages, timeout_seconds):
        self.stream_calls += 1
        try:
            while self.outcomes:
                item = self.outcomes.pop(0)
                if item.get("error"):
                    raise item["error"]
                if item.get("delta"):
                    yield {"type": "delta", "delta": item["delta"]}
                if item.get("usage") is not None:
                    yield {"type": "usage", "usage": item["usage"]}
                if item.get("delay"):
                    await asyncio.sleep(item["delay"])
            if self.default_error is not None:
                raise self.default_error
            yield {"type": "delta", "delta": "ok"}
        finally:
            self.stream_closed = True   # 取消传播：生成器关闭可观测


def make_candidate(alias: str, provider: str = "fake", **kw) -> CandidateConfig:
    defaults = dict(
        provider=provider,
        adapter=provider,   # 测试中 Adapter 注册名与供应商名一致（FakeProvider）
        provider_model="fake-model",
        base_url="http://fake.local",
        api_key="fake-key-for-tests",
        timeout_seconds=120,
        price=Price(input_per_million=1.0, output_per_million=1.0),
    )
    defaults.update(kw)
    return CandidateConfig(alias=alias, **defaults)


TASK_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "goal": {"type": "string"},
        "steps": {"type": "array", "items": {"type": "string"}},
        "completion_criteria": {"type": "array", "items": {"type": "string"}},
        "requires_tools": {"type": "boolean"},
    },
    "required": ["goal", "steps", "completion_criteria", "requires_tools"],
    "additionalProperties": False,
}

TASK_INPUT_SCHEMA = {
    "type": "object",
    "properties": {"product_name": {"type": "string", "minLength": 1}},
    "required": ["product_name"],
    "additionalProperties": False,
}

VALID_TASK_PLAN = (
    '{"goal": "g", "steps": ["s1"], '
    '"completion_criteria": ["done"], "requires_tools": false}'
)
INVALID_BUSINESS_PLAN = (
    '{"goal": "g", "steps": [], '
    '"completion_criteria": ["done"], "requires_tools": false}'
)


def _default_templates() -> dict:
    return {
        ("greet", "v1"): PromptTemplate(
            name="greet", version="v1", system_template="你是${x}助手。"
        ),
        ("task", "v1"): PromptTemplate(
            name="task",
            version="v1",
            system_template="你是${product_name}的任务规划器。",
            input_schema=TASK_INPUT_SCHEMA,
            output_schema=TASK_OUTPUT_SCHEMA,
            business_model="task_plan",
            default_model="test-model",
            default_timeout_seconds=30,
        ),
        ("task", "v2"): PromptTemplate(   # 候选版本：内容 hash 必须与 v1 不同
            name="task",
            version="v2",
            system_template="你是${product_name}的任务规划器。拆解为不超过 5 步。",
            input_schema=TASK_INPUT_SCHEMA,
            output_schema=TASK_OUTPUT_SCHEMA,
            business_model="task_plan",
            default_model="test-model",
            default_timeout_seconds=30,
        ),
    }


@pytest.fixture
def app_factory(tmp_path):
    def _make(
        provider=None,
        providers=None,        # [(provider_id, FakeProvider)]，用于区分候选的加权轮询测试
        models=None,
        api_keys=None,
        templates=None,
        breaker=None,
        limiter=None,
        retry=None,
        run_budget=120.0,
        checkpoint=None,
    ):
        _make.counter = getattr(_make, "counter", 0) + 1
        registry = ProviderRegistry()
        if providers:
            for provider_id, fake in providers:
                registry.register(fake)
        else:
            registry.register(provider or FakeProvider())

        routes = models or {
            "test-model": ModelRoute(
                alias="test-model",
                strategy="priority",
                # provider_model 不同 → 熔断键不同（熔断按 provider:provider_model 维度）
                candidates=(
                    make_candidate("c1"),
                    make_candidate("c2", provider_model="fake-model-b"),
                ),
            ),
        }
        settings = Settings(
            models=routes,
            templates=templates or _default_templates(),
            retry_policy=retry
            or RetryPolicy(max_attempts=2, base_delay=0.0, max_delay=0.0),
            api_keys=api_keys
            or (
                ApiKeySetting(
                    name="tester", key="test-key",
                    requests_per_second=100, burst=100, max_concurrency=4,
                ),
            ),
            database_path=tmp_path / f"usage-{_make.counter}.db",
            breaker_config=BreakerConfig(failure_threshold=2, recovery_timeout_seconds=3600.0),
            run_budget_seconds=run_budget,
            stream_checkpoint=checkpoint or {"enabled": False},
        )
        return create_app(
            settings,
            registry=registry,
            breaker=breaker or CircuitBreaker(failure_threshold=2, recovery_timeout=3600.0),
            limiter=limiter or TokenBucketLimiter(),
        )

    return _make


@pytest.fixture
def client_factory(app_factory):
    def _make(**kwargs):
        return TestClient(app_factory(**kwargs))

    return _make


@pytest.fixture
def auth():
    return {"Authorization": "Bearer test-key"}


def chat_body(model: str = "test-model", **extra) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": "hi"}]}
    body.update(extra)
    return body
