"""流式 usage 采集：上游回传 usage → 账本真实计数与计价。"""
import pytest

from conftest import FakeProvider, chat_body
from llm_gateway.core.models import LLMRequest, Message, Usage


def test_stream_usage_recorded_when_upstream_reports(client_factory, auth):
    # FakeProvider 流末回传 usage → trace 不再是 usage_missing，成本可计价
    provider = FakeProvider(
        outcomes=[
            {"delta": "hello"},
            {"usage": Usage(input_tokens=5, output_tokens=7)},
        ]
    )
    client = client_factory(provider=provider)
    client.post("/v1/llm/stream", json=chat_body(stream=True), headers=auth)
    trace = client.get("/v1/traces", headers=auth).json()[0]
    assert trace["status"] == "success"
    assert trace["usage_missing"] is False
    assert trace["input_tokens"] == 5
    assert trace["output_tokens"] == 7
    assert trace["cost_usd"] == pytest.approx(12 / 1_000_000)   # 价格 1/1 每百万


def test_stream_usage_missing_when_upstream_silent(client_factory, auth):
    # 上游不回传 usage → 三态=未知（占位 0 + usage_missing，不计价）
    provider = FakeProvider()
    client = client_factory(provider=provider)
    client.post("/v1/llm/stream", json=chat_body(stream=True), headers=auth)
    trace = client.get("/v1/traces", headers=auth).json()[0]
    assert trace["usage_missing"] is True
    assert trace["cost_usd"] is None


def test_usage_event_does_not_break_sse_output(client_factory, auth):
    # usage 事件只进账本，绝不混入对调用方的 SSE 事件流
    provider = FakeProvider(
        outcomes=[
            {"delta": "a"},
            {"usage": Usage(input_tokens=1, output_tokens=1)},
            {"delta": "b"},
        ]
    )
    client = client_factory(provider=provider)
    r = client.post("/v1/llm/stream", json=chat_body(stream=True), headers=auth)
    assert r.status_code == 200
    assert "content.delta" in r.text
    assert "usage" not in r.text        # usage 事件不外泄到 SSE
    assert "response.completed" in r.text
