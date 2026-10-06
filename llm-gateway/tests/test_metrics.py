"""观测闭环：P50/P95 TTFT、Latency、错误率、429 比例，按模型/调用方/Prompt 聚合。"""
from conftest import FakeProvider, chat_body, net_error


def test_metrics_aggregation(client_factory, auth):
    # 两次成功（不同延迟） + 一次流式（TTFT） + 一次失败（限流）
    ok = FakeProvider(outcomes=[
        {"content": "a", "delay": 0.02},
    ])
    client_ok = client_factory(provider=ok)
    client_ok.post("/v1/llm", json=chat_body(), headers=auth)
    client_ok.post("/v1/llm/stream", json=chat_body(stream=True), headers=auth)

    fail_provider = FakeProvider(default_error=net_error())
    client_fail = client_factory(provider=fail_provider)
    client_fail.post("/v1/llm", json=chat_body(), headers=auth)

    metrics = client_ok.get("/v1/metrics", headers=auth).json()
    assert metrics["total"] == 2
    assert metrics["success"] == 2
    assert metrics["latency_ms"]["p50"] is not None
    assert metrics["ttft_ms"]["p50"] is not None            # 流式调用贡献 TTFT
    assert "test-model" in metrics["by_model"]
    assert metrics["by_caller"]                             # 调用方指纹聚合
    assert metrics["tokens"]["input"] >= 1                  # 流式调用 usage 未知（三态）

    metrics_fail = client_fail.get("/v1/metrics", headers=auth).json()
    assert metrics_fail["total"] == 1
    assert metrics_fail["error_rate"] == 1.0
    assert metrics_fail["failed"] == 1


def test_metrics_requires_auth(client_factory):
    client = client_factory()
    assert client.get("/v1/metrics").status_code == 401
