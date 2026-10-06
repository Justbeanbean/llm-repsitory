"""OpenAI 规范错误体：{error: {message, type, code, param}}。"""
from conftest import chat_body


def test_gateway_error_openai_shape(client_factory, auth):
    client = client_factory()
    r = client.post("/v1/chat/completions", json=chat_body(model="no-such"), headers=auth)
    assert r.status_code == 400
    body = r.json()["error"]
    assert body["code"] == "unknown_model"
    assert body["type"] == "invalid_request_error"
    assert body["message"]
    assert body["param"] is None


def test_auth_error_type(client_factory):
    client = client_factory()
    r = client.post("/v1/chat/completions", json=chat_body())
    assert r.status_code == 401
    assert r.json()["error"]["type"] == "authentication_error"


def test_validation_error_openai_shape(client_factory, auth):
    # 未支持字段（temperature）→ 422 OpenAI 错误体，模型调用前明确失败
    client = client_factory()
    r = client.post(
        "/v1/chat/completions", json=chat_body(temperature=0.7), headers=auth
    )
    assert r.status_code == 422
    body = r.json()["error"]
    assert body["type"] == "invalid_request_error"
    assert body["code"] == "validation_error"
    assert body["details"]            # 校验明细（哪个字段不支持）


def test_rate_limit_error_type(client_factory):
    from llm_gateway.settings import ApiKeySetting

    client = client_factory(
        api_keys=(ApiKeySetting(name="t", key="k", requests_per_second=0.0, burst=1),)
    )
    headers = {"Authorization": "Bearer k"}
    client.post("/v1/chat/completions", json=chat_body(), headers=headers)
    r = client.post("/v1/chat/completions", json=chat_body(), headers=headers)
    assert r.status_code == 429
    assert r.json()["error"]["type"] == "rate_limit_error"


def test_unhandled_exception_returns_openai_500(app_factory, auth):
    # 未预期 bug（RuntimeError）→ 500 OpenAI 错误体，不外泄堆栈
    # raise_server_exceptions=False：让 TestClient 返回 500 响应而非重抛异常
    from fastapi.testclient import TestClient
    from conftest import FakeProvider

    app = app_factory(provider=FakeProvider(default_error=RuntimeError("内部 bug")))
    client = TestClient(app, raise_server_exceptions=False)
    r = client.post("/v1/chat/completions", json=chat_body(), headers=auth)
    assert r.status_code == 500
    body = r.json()["error"]
    assert body["code"] == "internal_error"
    assert body["type"] == "api_error"
    assert "内部 bug" not in body["message"]      # 不外泄异常细节
    assert "Traceback" not in str(body)
