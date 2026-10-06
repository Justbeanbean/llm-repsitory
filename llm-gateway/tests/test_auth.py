"""鉴权与公开端点：Bearer API Key / 401 / 公开的 models 与 healthz。"""
from conftest import chat_body


def test_missing_key_401(client_factory):
    client = client_factory()
    r = client.post("/v1/chat/completions", json=chat_body())
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "invalid_api_key"


def test_wrong_key_401(client_factory):
    client = client_factory()
    r = client.post(
        "/v1/chat/completions",
        json=chat_body(),
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert r.status_code == 401


def test_valid_key_200(client_factory, auth):
    client = client_factory()
    r = client.post("/v1/chat/completions", json=chat_body(), headers=auth)
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "ok"


def test_llm_endpoint_also_requires_auth(client_factory):
    client = client_factory()
    assert (
        client.post("/v1/llm", json=chat_body()).status_code == 401
    )


def test_models_requires_auth_and_lists_aliases(client_factory, auth):
    # /v1/models 需鉴权，带 created 时间戳
    client = client_factory()
    assert client.get("/v1/models").status_code == 401
    r = client.get("/v1/models", headers=auth)
    assert r.status_code == 200
    model = r.json()["data"][0]
    assert model["id"] == "test-model"
    assert model["created"] > 0


def test_healthz_ok(client_factory):
    client = client_factory()
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
