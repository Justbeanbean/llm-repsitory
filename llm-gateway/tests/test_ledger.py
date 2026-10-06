"""SQLite 用量账本：Token / Cost 三态 / TTFT / 重试 / fallback / 错误归因 / 调用方指纹。"""
import pytest

from conftest import FakeProvider, chat_body, net_error
from llm_gateway.core.models import Usage


def test_success_record_with_cost_and_caller(client_factory, auth):
    provider = FakeProvider(
        outcomes=[{"content": "ok", "usage": Usage(input_tokens=5, output_tokens=7)}]
    )
    client = client_factory(provider=provider)
    client.post("/v1/llm", json=chat_body(), headers=auth)

    trace = client.get("/v1/traces", headers=auth).json()[0]
    assert trace["status"] == "success"
    assert trace["input_tokens"] == 5
    assert trace["output_tokens"] == 7
    assert trace["cost_usd"] == pytest.approx(12 / 1_000_000)   # 价格 1/1 每百万
    assert trace["caller_fingerprint"]                            # 只保留短指纹
    assert trace["final_candidate"] == "c1"
    assert trace["network_attempts"] == 1


def test_usage_missing_cost_unknown(client_factory, auth):
    # 上游没回 usage：tokens 占位 0 + usage_missing，成本未知（None ≠ 0）
    provider = FakeProvider(
        outcomes=[{"content": "ok", "usage": Usage(usage_missing=True)}]
    )
    client = client_factory(provider=provider)
    client.post("/v1/llm", json=chat_body(), headers=auth)

    trace = client.get("/v1/traces", headers=auth).json()[0]
    assert trace["usage_missing"] is True
    assert trace["cost_usd"] is None
    assert trace["input_tokens"] == 0


def test_failure_record_with_attribution(client_factory, auth):
    provider = FakeProvider(default_error=net_error())
    client = client_factory(provider=provider)
    assert client.post("/v1/llm", json=chat_body(), headers=auth).status_code == 502

    trace = client.get("/v1/traces", headers=auth).json()[0]
    assert trace["status"] == "failed"
    assert trace["error_code"] == "model_unavailable"
    assert trace["failure_class"] == "network"
    assert trace["failure_layer"] == "l1_network"
    assert trace["cost_usd"] is None          # 失败调用成本未知
    assert trace["network_attempts"] == 4
    assert trace["fallback_count"] == 1


def test_stream_records_ttft(client_factory, auth):
    client = client_factory()
    client.post("/v1/llm/stream", json=chat_body(stream=True), headers=auth)
    trace = client.get("/v1/traces", headers=auth).json()[0]
    assert trace["status"] == "success"
    assert trace["ttft_ms"] is not None
    assert trace["usage_missing"] is True     # 流式协议不回 usage → 账本未知
    assert trace["cost_usd"] is None
