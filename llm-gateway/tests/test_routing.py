"""路由：优先级 fallback / 加权轮询 / 未知模型 / 结构化能力过滤。"""
from conftest import FakeProvider, chat_body, make_candidate, net_error
from llm_gateway.core.models import ModelRoute


def _traces(client, auth):
    return client.get("/v1/traces", headers=auth).json()


def test_unknown_model_400(client_factory, auth):
    client = client_factory()
    r = client.post("/v1/chat/completions", json=chat_body(model="no-such"), headers=auth)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unknown_model"


def test_structured_unsupported_400(client_factory, auth):
    client = client_factory(
        models={
            "plain-model": ModelRoute(
                alias="plain-model",
                strategy="priority",
                candidates=(make_candidate("c1", supports_structured_output=False),),
            )
        }
    )
    r = client.post(
        "/v1/chat/completions",
        json=chat_body(
            model="plain-model",
            response_format={
                "type": "json_schema",
                "json_schema": {"schema": {"type": "object"}},
            },
        ),
        headers=auth,
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "structured_output_unsupported"


def test_priority_fallback_to_backup(client_factory, auth):
    # c1 两次网络失败（重试耗尽）→ fallback c2 成功
    provider = FakeProvider(outcomes=[{"error": net_error()}, {"error": net_error()}, {}])
    client = client_factory(provider=provider)
    r = client.post("/v1/llm", json=chat_body(), headers=auth)
    assert r.status_code == 200
    assert r.json()["attempts"] == 3
    trace = _traces(client, auth)[0]
    assert trace["final_candidate"] == "c2"
    assert trace["fallback_count"] == 1
    assert trace["network_attempts"] == 3
    assert trace["status"] == "success"


def test_weighted_round_robin(client_factory, auth):
    # 等权重 1:1 → 严格交替；4 次请求各命中 2 次
    pa, pb = FakeProvider(provider_id="fake-a"), FakeProvider(provider_id="fake-b")
    client = client_factory(
        providers=[("fake-a", pa), ("fake-b", pb)],
        models={
            "balanced": ModelRoute(
                alias="balanced",
                strategy="weighted",
                candidates=(
                    make_candidate("wa", provider="fake-a", weight=1),
                    make_candidate("wb", provider="fake-b", weight=1),
                ),
            )
        },
    )
    for _ in range(4):
        r = client.post("/v1/chat/completions", json=chat_body(model="balanced"), headers=auth)
        assert r.status_code == 200
    assert pa.complete_calls == 2
    assert pb.complete_calls == 2
