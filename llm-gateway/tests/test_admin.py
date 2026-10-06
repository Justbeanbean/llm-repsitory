"""管理接口：/admin/routes（路由+熔断）、/admin/usage、/readyz、/healthz。"""
from conftest import chat_body, net_error
from conftest import FakeProvider as FP


def test_admin_routes_view_and_no_secrets(client_factory, auth):
    client = client_factory()
    r = client.get("/admin/routes", headers=auth)
    assert r.status_code == 200
    body = r.json()
    assert body["models"]["test-model"]["strategy"] == "priority"
    candidates = body["models"]["test-model"]["candidates"]
    assert len(candidates) == 2
    assert candidates[0]["provider_model"] == "fake-model"
    # 安全红线：管理视图绝不包含 api_key
    assert "api_key" not in str(body)
    assert "fake-key" not in str(body)


def test_admin_routes_circuit_status(client_factory, auth):
    provider = FP(default_error=net_error())
    client = client_factory(provider=provider)
    # 两轮全失败 → 熔断打开（阈值 2）
    client.post("/v1/llm", json=chat_body(), headers=auth)
    client.post("/v1/llm", json=chat_body(), headers=auth)
    body = client.get("/admin/routes", headers=auth).json()
    circuits = body["circuits"]
    assert circuits == {"fake:fake-model": "open", "fake:fake-model-b": "open"}


def test_admin_usage_alias(client_factory, auth):
    client = client_factory()
    client.post("/v1/llm", json=chat_body(), headers=auth)
    r = client.get("/admin/usage?limit=10", headers=auth)
    assert r.status_code == 200
    body = r.json()
    assert body["data"] and body["data"][0]["status"] == "success"
    # 与 /v1/traces 等价
    traces = client.get("/v1/traces", headers=auth).json()
    assert body["data"][0]["request_id"] == traces[0]["request_id"]


def test_readyz_and_healthz(client_factory):
    client = client_factory()
    ready = client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert ready.json()["service"] == "llm-gateway"
    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}   # liveness 不检查依赖


def test_admin_requires_auth(client_factory):
    client = client_factory()
    assert client.get("/admin/routes").status_code == 401
    assert client.get("/admin/usage").status_code == 401
