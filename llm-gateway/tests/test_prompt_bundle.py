"""Prompt Bundle：输入 Schema、输出 Schema 注入、内容 hash、双版本、参数默认值。"""
from conftest import VALID_TASK_PLAN, FakeProvider, chat_body


def _task_prompt(version: str = "v1", product_name: str = "测试"):
    return {"name": "task", "version": version, "variables": {"product_name": product_name}}


def test_input_schema_rejects_bad_variables_before_model(client_factory, auth):
    fake = FakeProvider()
    client = client_factory(provider=fake)
    # input_schema 要求 minLength 1：空产品名在调用模型前失败
    r = client.post(
        "/v1/llm", json=chat_body(prompt=_task_prompt(product_name="")), headers=auth
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_prompt_variables"
    assert fake.complete_calls == 0


def test_bundle_output_schema_injected(client_factory, auth):
    # 请求未带 response_schema：Bundle 的 output_schema 自动生效（L4 照常执行）
    provider = FakeProvider(
        outcomes=[{"content": "不是 JSON"}, {"content": VALID_TASK_PLAN}]
    )
    client = client_factory(provider=provider)
    r = client.post("/v1/llm", json=chat_body(prompt=_task_prompt()), headers=auth)
    assert r.status_code == 200
    assert r.json()["parsed"] is not None      # L4 跑了：非法首答被修复
    trace = client.get("/v1/traces", headers=auth).json()[0]
    assert trace["schema_hash"] is not None    # 使用的 Schema 可定位
    assert trace["prompt_hash"] is not None    # 模板内容 hash 进账本
    assert trace["prompt_version"] == "v1"


def test_bundle_two_versions_distinct_hash(client_factory, auth):
    provider = FakeProvider()
    client = client_factory(provider=provider)
    client.post("/v1/llm", json=chat_body(prompt=_task_prompt("v1")), headers=auth)
    client.post("/v1/llm", json=chat_body(prompt=_task_prompt("v2")), headers=auth)
    traces = client.get("/v1/traces", headers=auth).json()
    hashes = {t["prompt_hash"] for t in traces}
    versions = {t["prompt_version"] for t in traces}
    assert versions == {"v1", "v2"}            # 旧版 + 候选版并存
    assert len(hashes) == 2                    # 内容 hash 各不相同


def test_prompt_rendered_into_system_message(client_factory, auth):
    fake = FakeProvider()
    client = client_factory(provider=fake)
    client.post("/v1/llm", json=chat_body(prompt=_task_prompt()), headers=auth)
    assert fake.received[0][0].content == "你是测试的任务规划器。"
