"""并发治理：每调用方并发上限（超限 429）+ 每供应商候选并发（排队）。"""
import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from conftest import FakeProvider
from llm_gateway.settings import ApiKeySetting


@pytest.mark.asyncio
async def test_caller_concurrency_limit(app_factory):
    # max_concurrency=1 + 慢上游：第二个并发请求被拒（429），而非排队
    provider = FakeProvider(delay=0.2)
    app = app_factory(
        provider=provider,
        api_keys=(
            ApiKeySetting(
                name="tight",
                key="tight-key",
                requests_per_second=100,
                burst=100,
                max_concurrency=1,
            ),
        ),
    )
    headers = {"Authorization": "Bearer tight-key"}
    body = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        first, second = await asyncio.gather(
            client.post("/v1/llm", json=body, headers=headers),
            client.post("/v1/llm", json=body, headers=headers),
        )
    assert {first.status_code, second.status_code} == {200, 429}
    rejected = second if second.status_code == 429 else first
    assert rejected.json()["error"]["code"] == "concurrency_limit"
    assert provider.complete_calls == 1   # 被拒请求没有打到上游


@pytest.mark.asyncio
async def test_provider_slot_queues_instead_of_rejecting(tmp_path):
    # 供应商侧并发：超限排队等待（两个并发调用都成功，而不是一个被拒）
    from llm_gateway.core.circuit_breaker import CircuitBreaker
    from llm_gateway.core.concurrency import ConcurrencyGuard
    from llm_gateway.core.models import CandidateConfig, ModelRoute, Price
    from llm_gateway.core.providers.registry import ProviderRegistry
    from llm_gateway.core.rate_limit import TokenBucketLimiter
    from llm_gateway.core.retry import RetryPolicy
    from llm_gateway.main import create_app
    from llm_gateway.settings import ApiKeySetting, BreakerConfig, Settings

    provider = FakeProvider(delay=0.1)
    registry = ProviderRegistry()
    registry.register(provider)
    settings = Settings(
        models={
            "test-model": ModelRoute(
                alias="test-model",
                strategy="priority",
                candidates=(
                    CandidateConfig(
                        alias="c1",
                        provider="fake",
                        adapter="fake",
                        provider_model="fake-model",
                        base_url="http://fake.local",
                        api_key="fake-key",
                        supports_structured_output=True,
                        price=Price(1.0, 1.0),
                    ),
                ),
            )
        },
        templates={},
        retry_policy=RetryPolicy(max_attempts=2, base_delay=0.0, max_delay=0.0),
        api_keys=(
            ApiKeySetting(
                name="t", key="k", requests_per_second=100, burst=100, max_concurrency=4
            ),
        ),
        database_path=tmp_path / "usage.db",
        breaker_config=BreakerConfig(failure_threshold=2, recovery_timeout_seconds=3600.0),
        max_concurrency_per_candidate=1,
    )
    app = create_app(
        settings,
        registry=registry,
        breaker=CircuitBreaker(2, 3600.0),
        limiter=TokenBucketLimiter(),
        concurrency_guard=ConcurrencyGuard(),
    )
    headers = {"Authorization": "Bearer k"}
    body = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        first, second = await asyncio.gather(
            client.post("/v1/llm", json=body, headers=headers),
            client.post("/v1/llm", json=body, headers=headers),
        )
    # 排队语义：两个都成功（第二个等待第一个释放槽位）
    assert first.status_code == 200
    assert second.status_code == 200
    assert provider.complete_calls == 2
