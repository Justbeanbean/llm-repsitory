"""本地业务规则：JSON 合法但业务不合法 → 不进入 Agent Loop（422）+ 一次修复。"""
from conftest import (
    INVALID_BUSINESS_PLAN,
    VALID_TASK_PLAN,
    FakeProvider,
    chat_body,
)

TASK_PROMPT = {"name": "task", "version": "v1", "variables": {"product_name": "测试"}}


def test_business_rule_violation_repaired_once(client_factory, auth):
    # steps=[] 结构合法（jsonschema 过）但业务不合法（TaskPlan min_length=1）→ 反喂修复成功
    provider = FakeProvider(
        outcomes=[{"content": INVALID_BUSINESS_PLAN}, {"content": VALID_TASK_PLAN}]
    )
    client = client_factory(provider=provider)
    r = client.post("/v1/llm", json=chat_body(prompt=TASK_PROMPT), headers=auth)
    assert r.status_code == 200
    assert r.json()["parsed"]["steps"] == ["s1"]
    trace = client.get("/v1/traces", headers=auth).json()[0]
    assert trace["repair_attempts"] == 1
    assert trace["error_code"] is None


def test_business_rule_violation_then_422(client_factory, auth):
    # 修复后仍业务不合法 → 422 business_validation_failed，绝不放行进 Agent Loop
    provider = FakeProvider(
        outcomes=[{"content": INVALID_BUSINESS_PLAN}, {"content": INVALID_BUSINESS_PLAN}]
    )
    client = client_factory(provider=provider)
    r = client.post("/v1/llm", json=chat_body(prompt=TASK_PROMPT), headers=auth)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "business_validation_failed"
    trace = client.get("/v1/traces", headers=auth).json()[0]
    assert trace["failure_layer"] == "l4_schema"
    assert trace["fallback_count"] == 0


def test_valid_task_plan_passes_both_layers(client_factory, auth):
    # 双层校验：供应商约束（jsonschema）+ pydantic 业务规则 同时通过
    provider = FakeProvider(outcomes=[{"content": VALID_TASK_PLAN}])
    client = client_factory(provider=provider)
    r = client.post("/v1/llm", json=chat_body(prompt=TASK_PROMPT), headers=auth)
    assert r.status_code == 200
    assert r.json()["parsed"]["requires_tools"] is False
    assert provider.complete_calls == 1
