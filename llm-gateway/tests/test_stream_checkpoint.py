"""流式检查点（yaml stream_checkpoint）：默认关闭零开销；开启后终态可查询。"""
from conftest import FakeProvider, chat_body


def test_checkpoint_disabled_by_default(client_factory, auth):
    # 默认关闭（隐私边界）：流结束后无检查点记录
    client = client_factory()
    client.post("/v1/llm/stream", json=chat_body(stream=True), headers=auth)
    request_id = client.get("/v1/traces", headers=auth).json()[0]["request_id"]
    r = client.get(f"/v1/stream-checkpoints/{request_id}", headers=auth)
    assert r.status_code == 404


def test_checkpoint_enabled_records_terminal_state(client_factory, auth):
    provider = FakeProvider(outcomes=[{"delta": "hello "}, {"delta": "world"}])
    client = client_factory(
        provider=provider,
        checkpoint={"enabled": True, "flush_interval_seconds": 1, "max_chars": 200},
    )
    client.post("/v1/llm/stream", json=chat_body(stream=True), headers=auth)

    request_id = client.get("/v1/traces", headers=auth).json()[0]["request_id"]
    r = client.get(f"/v1/stream-checkpoints/{request_id}", headers=auth)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed"
    # FakeProvider 在脚本化 delta 之后还有默认尾块 "ok"
    assert body["content"] == "hello worldok"
    assert body["model"] == "test-model"


def test_checkpoint_respects_max_chars(client_factory, auth):
    provider = FakeProvider(outcomes=[{"delta": "x" * 50}, {"delta": "y" * 50}])
    client = client_factory(
        provider=provider,
        checkpoint={"enabled": True, "flush_interval_seconds": 1, "max_chars": 60},
    )
    client.post("/v1/llm/stream", json=chat_body(stream=True), headers=auth)
    request_id = client.get("/v1/traces", headers=auth).json()[0]["request_id"]
    body = client.get(f"/v1/stream-checkpoints/{request_id}", headers=auth).json()
    assert len(body["content"]) == 60   # 截断到 max_chars


def test_checkpoint_requires_auth(client_factory):
    client = client_factory(checkpoint={"enabled": True})
    assert client.get("/v1/stream-checkpoints/whatever").status_code == 401
