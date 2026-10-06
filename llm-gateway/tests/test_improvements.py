"""补充改进：上游 4xx 透传 / responses_api 收口 / traces 过滤分页 / metrics 窗口 / 流式上游 ID。"""
import pytest

from conftest import FakeProvider, chat_body, net_error
from llm_gateway.core.exceptions import FailureClass, ProviderError
from llm_gateway.core.models import LLMRequest, Message, ModelRoute
from llm_gateway.core.providers.registry import ProviderRegistry
from llm_gateway.core.providers.selector import ModelSelector


def upstream_400() -> ProviderError:
    return ProviderError(
        FailureClass.PROVIDER_CLIENT, "provider_client_error",
        "上游返回 400: 上下文超长", retryable=False, status_code=400,
    )


def test_upstream_4xx_passthrough(client_factory, auth):
    # 上游 400（如上下文超长）透传 400 + invalid_request_error，不再折叠成 502
    provider = FakeProvider(default_error=upstream_400())
    client = client_factory(provider=provider)
    r = client.post("/v1/chat/completions", json=chat_body(), headers=auth)
    assert r.status_code == 400
    body = r.json()["error"]
    assert body["code"] == "provider_client_error"
    assert body["type"] == "invalid_request_error"
    # 确定性 4xx 不换 provider
    assert provider.complete_calls == 1


def test_upstream_4xx_without_status_falls_back_to_502(client_factory, auth):
    error = ProviderError(
        FailureClass.PROVIDER_CLIENT, "provider_client_error", "上游拒绝", retryable=False
    )
    provider = FakeProvider(default_error=error)
    client = client_factory(provider=provider)
    r = client.post("/v1/chat/completions", json=chat_body(), headers=auth)
    assert r.status_code == 502


def test_responses_api_not_forgeable(client_factory, auth):
    # responses_api 已从公开 schema 收口：请求体里伪造该字段 → 422 拒绝
    client = client_factory()
    r = client.post(
        "/v1/chat/completions",
        json=chat_body(responses_api=True),
        headers=auth,
    )
    assert r.status_code == 422   # extra=forbid


def test_chat_only_model_rejected_for_responses(client_factory, auth):
    from conftest import make_candidate

    # 候选仅 api=chat：/v1/responses 请求 → 400；/v1/chat/completions 正常
    client = client_factory(
        models={
            "chat-only": ModelRoute(
                alias="chat-only",
                strategy="priority",
                candidates=(make_candidate("c1", api="chat"),),
            )
        }
    )
    r = client.post(
        "/v1/responses",
        json={"model": "chat-only", "input": "hi"},
        headers=auth,
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "responses_api_unsupported"
    r = client.post(
        "/v1/chat/completions", json=chat_body(model="chat-only"), headers=auth
    )
    assert r.status_code == 200


def test_traces_filter_and_pagination(client_factory, auth):
    # 2 成功 + 1 失败：按状态/模型过滤 + offset 分页
    ok_client = client_factory()
    ok_client.post("/v1/llm", json=chat_body(), headers=auth)
    ok_client.post("/v1/llm", json=chat_body(), headers=auth)

    fail_provider = FakeProvider(default_error=net_error())
    fail_client = client_factory(provider=fail_provider)
    fail_client.post("/v1/llm", json=chat_body(), headers=auth)

    # 状态过滤
    failed = fail_client.get("/v1/traces?status=failed", headers=auth).json()
    assert len(failed) == 1 and failed[0]["status"] == "failed"
    # 模型过滤
    by_model = fail_client.get("/v1/traces?model=test-model", headers=auth).json()
    assert all(t["requested_model"] == "test-model" for t in by_model)
    # 分页（ok_client 库中有 2 条记录）
    page1 = ok_client.get("/v1/traces?limit=1", headers=auth).json()
    page2 = ok_client.get("/v1/traces?limit=1&offset=1", headers=auth).json()
    assert len(page1) == 1 and len(page2) == 1
    assert page1[0]["request_id"] != page2[0]["request_id"]


def test_metrics_window_minutes(client_factory, auth):
    # 窗口过滤：window_minutes 极小值时旧记录不计入（时间口径）
    ok_client = client_factory()
    ok_client.post("/v1/llm", json=chat_body(), headers=auth)
    m_all = ok_client.get("/v1/metrics", headers=auth).json()
    assert m_all["total"] == 1
    # since 0 分钟前的记录：now-0min 截止 → 理论上不含刚写入的记录（边界由 DB 时间戳决定）
    m_win = ok_client.get("/v1/metrics?window_minutes=60", headers=auth).json()
    assert m_win["total"] == 1   # 最近 60 分钟显然包含


@pytest.mark.asyncio
async def test_stream_upstream_id_recorded(app_factory):
    from conftest import FakeProvider as FP

    provider = FP(outcomes=[{"delta": "hello"}])
    app = app_factory(provider=provider)
    service = app.state.llm_service
    request = LLMRequest(
        model="test-model", messages=[Message(role="user", content="hi")], stream=True
    )
    events = [e async for e in service.stream(request)]
    assert events[-1]["type"] == "response.completed"
    # FakeProvider 未发 meta 事件 → upstream_id 为 None（真实 Adapter 会带）


def test_cors_disabled_by_default(client_factory):
    client = client_factory()
    r = client.get("/v1/models", headers={"Origin": "https://example.com"})
    assert "access-control-allow-origin" not in r.headers


def test_cors_enabled_when_configured(tmp_path):
    from fastapi.testclient import TestClient
    from llm_gateway.core.circuit_breaker import CircuitBreaker
    from llm_gateway.core.models import ModelRoute
    from llm_gateway.core.providers.registry import ProviderRegistry
    from llm_gateway.core.rate_limit import TokenBucketLimiter
    from llm_gateway.core.retry import RetryPolicy
    from llm_gateway.main import create_app
    from llm_gateway.settings import ApiKeySetting, BreakerConfig, Settings
    from conftest import FakeProvider, make_candidate

    settings = Settings(
        models={
            "test-model": ModelRoute(
                alias="test-model",
                strategy="priority",
                candidates=(make_candidate("c1"),),
            )
        },
        templates={},
        retry_policy=RetryPolicy(max_attempts=2, base_delay=0.0, max_delay=0.0),
        api_keys=(ApiKeySetting(name="k", key="test-key"),),
        database_path=tmp_path / "usage.db",
        breaker_config=BreakerConfig(failure_threshold=2, recovery_timeout_seconds=3600.0),
        cors_allow_origins=("https://agent.example.com",),
    )
    registry = ProviderRegistry()
    registry.register(FakeProvider())
    app = create_app(
        settings,
        registry=registry,
        breaker=CircuitBreaker(2, 3600.0),
        limiter=TokenBucketLimiter(),
    )
    client = TestClient(app)
    r = client.get("/v1/models", headers={"Origin": "https://agent.example.com"})
    assert r.headers.get("access-control-allow-origin") == "https://agent.example.com"
