"""OpenAI 兼容端点：chat/completions 契约、responses text.format、不支持字段明确失败。"""
from conftest import FakeProvider, chat_body


SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "integer"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def test_chat_completion_shape(client_factory, auth):
    client = client_factory()
    r = client.post("/v1/chat/completions", json=chat_body(), headers=auth)
    assert r.status_code == 200
    body = r.json()
    assert body["id"].startswith("chatcmpl-")
    assert body["object"] == "chat.completion"
    assert body["model"] == "test-model"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["total_tokens"] == 2


def test_unsupported_field_fails_before_model(client_factory, auth):
    # 未支持的字段在模型调用前明确失败（不静默忽略）
    from conftest import FakeProvider

    fake = FakeProvider()
    client = client_factory(provider=fake)
    r = client.post(
        "/v1/chat/completions",
        json=chat_body(temperature=0.7),
        headers=auth,
    )
    assert r.status_code == 422
    assert fake.complete_calls == 0


def test_response_format_json_schema_local_validation(client_factory, auth):
    # 本地二次校验 + 一次修复：首答非法 → 反喂重修成功
    provider = FakeProvider(outcomes=[{"content": "不是 JSON"}, {"content": '{"answer": 42}'}])
    client = client_factory(provider=provider)
    r = client.post(
        "/v1/chat/completions",
        json=chat_body(
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": SCHEMA},
            }
        ),
        headers=auth,
    )
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == '{"answer": 42}'
    assert provider.complete_calls == 2


def test_responses_endpoint_shape(client_factory, auth):
    client = client_factory()
    r = client.post(
        "/v1/responses",
        json={"model": "test-model", "input": "你好"},
        headers=auth,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "response"
    assert body["status"] == "completed"
    assert body["output"][0]["content"][0]["type"] == "output_text"
    assert body["output"][0]["content"][0]["text"] == "ok"
    assert body["usage"]["total_tokens"] == 2


def test_responses_text_format_structured(client_factory, auth):
    provider = FakeProvider(outcomes=[{"content": "垃圾"}, {"content": '{"answer": 7}'}])
    client = client_factory(provider=provider)
    r = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "回答一个整数",
            "text": {"format": {"type": "json_schema", "name": "answer", "schema": SCHEMA}},
        },
        headers=auth,
    )
    assert r.status_code == 200
    assert r.json()["output"][0]["content"][0]["text"] == '{"answer": 7}'
    assert provider.complete_calls == 2


def test_json_object_passes_through_without_l4(client_factory, auth):
    # json_object 无 schema：不做本地校验，直通返回
    provider = FakeProvider(outcomes=[{"content": "随便什么"}])
    client = client_factory(provider=provider)
    r = client.post(
        "/v1/chat/completions",
        json=chat_body(response_format={"type": "json_object"}),
        headers=auth,
    )
    assert r.status_code == 200
    assert provider.complete_calls == 1
