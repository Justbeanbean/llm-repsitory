"""Prompt 管理 API：创建版本 / 列表 / 获取 / 渲染预览 / 创建后立即可调用。"""
from conftest import FakeProvider, chat_body


def test_create_prompt_auto_version(client_factory, auth):
    client = client_factory()
    # yaml 种子里 task 已有 v1/v2 → 新版本自动递增为 v3
    r = client.post(
        "/v1/prompts",
        json={"id": "task", "template": "你是${product_name}的 v3 规划器。"},
        headers=auth,
    )
    assert r.status_code == 201
    record = r.json()
    assert record["id"] == "task"
    assert record["version"] == "v3"
    assert record["role"] == "system"
    assert record["content_hash"]

    # 全新 id → v1
    r = client.post(
        "/v1/prompts",
        json={"id": "brand-new", "template": "回复必须是中文。"},
        headers=auth,
    )
    assert r.status_code == 201
    assert r.json()["version"] == "v1"


def test_list_and_get_prompts(client_factory, auth):
    client = client_factory()
    records = client.get("/v1/prompts", headers=auth).json()
    ids = {(r["id"], r["version"]) for r in records}
    assert ("task", "v1") in ids and ("task", "v2") in ids   # yaml 种子已入库

    # 默认取最新版本
    latest = client.get("/v1/prompts/task", headers=auth).json()
    assert latest["version"] == "v2"
    # 指定版本
    old = client.get("/v1/prompts/task?version=v1", headers=auth).json()
    assert old["version"] == "v1"
    # 不存在 → 404
    r = client.get("/v1/prompts/nope", headers=auth)
    assert r.status_code == 404


def test_render_prompt_preview(client_factory, auth):
    client = client_factory()
    r = client.post(
        "/v1/prompts/task/render",
        json={"variables": {"product_name": "智答助手"}},
        headers=auth,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "task"
    assert body["version"] == "v2"          # 默认最新
    assert body["role"] == "system"
    assert body["content"].startswith("你是智答助手的任务规划器")

    # 缺变量：与正式调用相同的沙箱校验 → 400
    r = client.post("/v1/prompts/task/render", json={"variables": {}}, headers=auth)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "missing_prompt_variable"


def test_created_version_immediately_callable(client_factory, auth):
    # 通过 API 创建的新版本立即可用于调用（无需重启）
    provider = FakeProvider()
    client = client_factory(provider=provider)
    client.post(
        "/v1/prompts",
        json={"id": "qa", "template": "你是${topic}的问答官。"},
        headers=auth,
    )
    r = client.post(
        "/v1/llm",
        json=chat_body(prompt={"name": "qa", "version": "latest", "variables": {"topic": "天文学"}}),
        headers=auth,
    )
    assert r.status_code == 200
    assert provider.received[0][0].content == "你是天文学的问答官。"
    trace = client.get("/v1/traces", headers=auth).json()[0]
    assert trace["prompt_version"] == "v1"


def test_prompt_admin_requires_auth(client_factory):
    client = client_factory()
    assert client.get("/v1/prompts").status_code == 401
    assert client.post("/v1/prompts", json={"id": "x", "template": "t"}).status_code == 401
