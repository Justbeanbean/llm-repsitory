"""流式：OpenAI chunk 格式 / [DONE] / 首块前 fallback / 首块后不换模型 / TTFT。"""
import json

from conftest import FakeProvider, chat_body, net_error


def _sse_events(text: str) -> list[dict]:
    events = []
    for line in text.split("\n\n"):
        if line.startswith("data: "):
            payload = line[len("data: "):]
            if payload != "[DONE]":
                events.append(json.loads(payload))
    return events


def test_chat_completion_stream_format(client_factory, auth):
    client = client_factory()
    r = client.post(
        "/v1/chat/completions", json=chat_body(stream=True), headers=auth
    )
    assert r.status_code == 200
    assert r.text.endswith("data: [DONE]\n\n")
    events = _sse_events(r.text)
    assert all(e["object"] == "chat.completion.chunk" for e in events)
    assert events[0]["choices"][0]["delta"]["role"] == "assistant"
    assert any(e["choices"][0]["delta"].get("content") == "ok" for e in events)
    assert events[-1]["choices"][0]["finish_reason"] == "stop"
    # 防代理缓冲头（真实逐块到达客户端，保证 TTFT）
    assert r.headers.get("cache-control") == "no-cache, no-transform"
    assert r.headers.get("x-accel-buffering") == "no"
    assert r.headers.get("x-request-id")


def test_responses_api_stream(client_factory, auth):
    client = client_factory()
    r = client.post(
        "/v1/responses", json={"model": "test-model", "input": "hi", "stream": True},
        headers=auth,
    )
    assert r.status_code == 200
    events = _sse_events(r.text)
    types = [e["type"] for e in events]
    assert types[0] == "response.created"
    assert "response.output_text.delta" in types
    assert types[-1] == "response.completed"
    assert events[-1]["response"]["output"][0]["content"][0]["text"] == "ok"


def test_fallback_before_first_chunk(client_factory, auth):
    # 首块前网络失败 → 换候选；TTFT 记账为首个业务 delta
    provider = FakeProvider(outcomes=[{"error": net_error()}])
    client = client_factory(provider=provider)
    r = client.post(
        "/v1/llm/stream", json=chat_body(stream=True), headers=auth
    )
    assert r.status_code == 200
    assert "content.delta" in r.text
    assert "response.completed" in r.text
    trace = client.get("/v1/traces", headers=auth).json()[0]
    assert trace["status"] == "success"
    assert trace["fallback_count"] == 1
    assert trace["ttft_ms"] is not None


def test_no_fallback_after_first_delta(client_factory, auth):
    # 首块后失败：绝不换模型拼接，只发流内错误
    provider = FakeProvider(outcomes=[{"delta": "hello"}, {"error": net_error()}])
    client = client_factory(provider=provider)
    r = client.post(
        "/v1/llm/stream", json=chat_body(stream=True), headers=auth
    )
    assert r.status_code == 200
    assert "hello" in r.text
    assert "response.failed" in r.text
    assert provider.stream_calls == 1   # 没有第二次流式调用
    trace = client.get("/v1/traces", headers=auth).json()[0]
    assert trace["status"] == "failed"
    assert trace["error_code"] == "upstream_stream_failed"


def test_stream_prevalidation_400(client_factory, auth):
    client = client_factory()
    r = client.post(
        "/v1/llm/stream", json=chat_body(model="no-such", stream=True), headers=auth
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unknown_model"
