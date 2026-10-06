"""重试与 fallback：L1/L2 重试、429 透传、L3 内容 422、L4 反喂修复一次。"""
from conftest import FakeProvider, chat_body, empty_content_error, net_error, rate_limit_error


def _traces(client, auth):
    return client.get("/v1/traces", headers=auth).json()


SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "integer"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def test_retry_same_route_then_success(client_factory, auth):
    # 第 1 次网络失败，重试第 2 次成功：同 route 内消化，不触发 fallback
    provider = FakeProvider(outcomes=[{"error": net_error()}, {}])
    client = client_factory(provider=provider)
    r = client.post("/v1/llm", json=chat_body(), headers=auth)
    assert r.status_code == 200
    assert r.json()["attempts"] == 2
    trace = _traces(client, auth)[0]
    assert trace["fallback_count"] == 0
    assert trace["network_attempts"] == 2


def test_all_candidates_fail_502(client_factory, auth):
    provider = FakeProvider(default_error=net_error())
    client = client_factory(provider=provider)
    r = client.post("/v1/llm", json=chat_body(), headers=auth)
    assert r.status_code == 502
    assert r.json()["error"]["code"] == "model_unavailable"
    trace = _traces(client, auth)[0]
    assert trace["status"] == "failed"
    assert trace["network_attempts"] == 4        # c1×2 + c2×2
    assert trace["fallback_count"] == 1
    assert trace["failure_layer"] == "l1_network"


def test_upstream_rate_limit_passthrough_429(client_factory, auth):
    provider = FakeProvider(default_error=rate_limit_error(retry_after=7))
    client = client_factory(provider=provider)
    r = client.post("/v1/llm", json=chat_body(), headers=auth)
    assert r.status_code == 429
    assert r.json()["error"]["code"] == "rate_limited"
    assert r.headers["retry-after"] == "7"


def test_empty_content_422_no_fallback(client_factory, auth):
    # L3 取不到文本：立即 422，不重试、不换 provider
    provider = FakeProvider(outcomes=[{"error": empty_content_error()}])
    client = client_factory(provider=provider)
    r = client.post("/v1/llm", json=chat_body(), headers=auth)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "empty_content"
    assert provider.complete_calls == 1


def test_schema_repair_once_success(client_factory, auth):
    # L4：第一次非法 JSON → 反喂重修一次成功（同 route）
    provider = FakeProvider(outcomes=[{"content": "这不是 JSON"}, {"content": '{"answer": 1}'}])
    client = client_factory(provider=provider)
    r = client.post("/v1/llm", json=chat_body(response_schema=SCHEMA), headers=auth)
    assert r.status_code == 200
    assert r.json()["parsed"] == {"answer": 1}
    trace = _traces(client, auth)[0]
    assert trace["repair_attempts"] == 1
    assert trace["network_attempts"] == 2
    assert trace["attempts"] == 3


def test_schema_repair_once_then_422(client_factory, auth):
    # 修复重试仍失败 → 422，绝不换 provider
    provider = FakeProvider(
        outcomes=[{"content": "bad"}, {"content": "还是不行"}]
    )
    client = client_factory(provider=provider)
    r = client.post("/v1/llm", json=chat_body(response_schema=SCHEMA), headers=auth)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_json"
    trace = _traces(client, auth)[0]
    assert trace["repair_attempts"] == 1
    assert trace["fallback_count"] == 0
    assert trace["failure_layer"] == "l4_schema"


def test_plain_text_skips_l4(client_factory, auth):
    # 纯文本请求：不校验 schema，非法 JSON 原样返回
    provider = FakeProvider(outcomes=[{"content": "随便什么文本"}])
    client = client_factory(provider=provider)
    r = client.post("/v1/llm", json=chat_body(), headers=auth)
    assert r.status_code == 200
    assert r.json()["content"] == "随便什么文本"
    assert provider.complete_calls == 1
