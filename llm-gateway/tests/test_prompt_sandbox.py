"""Prompt 模板：版本选择、沙箱渲染（缺变量/多变量前置失败）、请求注入。"""
from conftest import chat_body


def _prompt(variables):
    return {"name": "greet", "version": "v1", "variables": variables}


def test_unknown_template_400(client_factory, auth):
    client = client_factory()
    r = client.post(
        "/v1/llm",
        json=chat_body(prompt={"name": "nope", "version": "v9", "variables": {}}),
        headers=auth,
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unknown_prompt_template"


def test_missing_variable_fails_before_model(client_factory, auth):
    from conftest import FakeProvider

    fake = FakeProvider()
    client = client_factory(provider=fake)
    r = client.post("/v1/llm", json=chat_body(prompt=_prompt({})), headers=auth)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "missing_prompt_variable"
    assert fake.complete_calls == 0       # 调用模型前失败


def test_extra_variable_rejected(client_factory, auth):
    client = client_factory()
    r = client.post(
        "/v1/llm", json=chat_body(prompt=_prompt({"x": "测试", "evil": "注入"})), headers=auth
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unexpected_prompt_variable"


def test_template_injected_as_system_message(client_factory, auth):
    from conftest import FakeProvider

    fake = FakeProvider()
    client = client_factory(provider=fake)
    r = client.post("/v1/llm", json=chat_body(prompt=_prompt({"x": "测试"})), headers=auth)
    assert r.status_code == 200
    # 渲染后的系统消息注入到消息列表首位
    messages = fake.received[0]
    assert messages[0].role == "system"
    assert messages[0].content == "你是测试助手。"
    # 账本记录实际使用的 Prompt 版本
    trace = client.get("/v1/traces", headers=auth).json()[0]
    assert trace["prompt_name"] == "greet"
    assert trace["prompt_version"] == "v1"
