"""Run 预算：重试、Fallback、修复共用同一 deadline，耗尽 → 504。"""
from conftest import FakeProvider, chat_body, net_error
from llm_gateway.core.retry import RetryPolicy


def test_deadline_stops_retry_and_fallback(client_factory, auth):
    # 预算 0.03s，首次失败后重试需等 0.05s → 第二次尝试前 deadline 耗尽
    provider = FakeProvider(default_error=net_error())
    client = client_factory(
        provider=provider,
        retry=RetryPolicy(max_attempts=3, base_delay=0.05, max_delay=0.05, jitter=False),
        run_budget=0.03,
    )
    r = client.post("/v1/llm", json=chat_body(), headers=auth)
    assert r.status_code == 504
    assert r.json()["error"]["code"] == "deadline_exceeded"
    # 预算耗尽后不再有第二次尝试，更不会 fallback 到 c2
    assert provider.complete_calls == 1
    trace = client.get("/v1/traces", headers=auth).json()[0]
    assert trace["status"] == "failed"
    assert trace["error_code"] == "deadline_exceeded"


def test_deadline_covers_repair_calls(client_factory, auth):
    # 修复调用同样受预算约束：首次调用耗时 0.05s 已超预算（0.02s）→ 修复前拦截
    provider = FakeProvider(outcomes=[{"content": "不是 JSON", "delay": 0.05}])
    client = client_factory(
        provider=provider,
        retry=RetryPolicy(max_attempts=1, base_delay=0.0, max_delay=0.0),
        run_budget=0.02,
    )
    r = client.post(
        "/v1/llm",
        json=chat_body(response_schema={"type": "object"}),
        headers=auth,
    )
    assert r.status_code == 504
    assert r.json()["error"]["code"] == "deadline_exceeded"
    assert provider.complete_calls == 1   # 修复调用被预算拦下，未打到上游
