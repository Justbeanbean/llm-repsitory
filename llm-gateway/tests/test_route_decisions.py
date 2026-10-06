"""可解释路由：从 Trace 看到最终端点和路由理由。"""
from conftest import FakeProvider, chat_body, net_error


def test_fallback_decision_visible_in_trace(client_factory, auth):
    # c1 网络失败 → fallback → c2 selected：完整决策链进账本
    provider = FakeProvider(outcomes=[{"error": net_error()}, {"error": net_error()}, {}])
    client = client_factory(provider=provider)
    client.post("/v1/llm", json=chat_body(), headers=auth)
    trace = client.get("/v1/traces", headers=auth).json()[0]
    decisions = trace["route_decisions"]
    assert decisions[0]["candidate"] == "c1"
    assert decisions[0]["action"] == "failed"
    assert decisions[0]["reason"] == "fallback_after_network_error"
    assert decisions[1]["candidate"] == "c2"
    assert decisions[1]["action"] == "selected"
    assert decisions[1]["reason"] == "fallback_candidate"
    # 最终端点可见：候选名 + 供应商真实模型 + 上游请求 ID + 价格版本
    assert trace["final_candidate"] == "c2"
    assert decisions[1]["provider_model"] == "fake-model-b"
    assert trace["upstream_request_id"].startswith("fake-req-")
    assert trace["price_version"] == "v1"


def test_circuit_open_skip_visible_in_trace(client_factory, auth):
    # 熔断打开的候选：决策 = skipped, reason = circuit_open
    provider = FakeProvider(default_error=net_error())
    client = client_factory(provider=provider)
    # 两轮全失败 → 两个候选都打开熔断
    client.post("/v1/llm", json=chat_body(), headers=auth)
    client.post("/v1/llm", json=chat_body(), headers=auth)
    r = client.post("/v1/llm", json=chat_body(), headers=auth)
    assert r.status_code == 503
    trace = client.get("/v1/traces", headers=auth).json()[0]
    decisions = trace["route_decisions"]
    assert len(decisions) == 2
    assert all(d["action"] == "skipped" for d in decisions)
    assert all(d["reason"] == "circuit_open" for d in decisions)
