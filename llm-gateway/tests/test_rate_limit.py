"""进程内令牌桶限流：容量耗尽 → 429 + Retry-After。"""
from conftest import chat_body
from llm_gateway.settings import ApiKeySetting


def test_token_bucket_rejects_after_burst(client_factory):
    client = client_factory(
        api_keys=(
            ApiKeySetting(name="tight", key="tight-key", requests_per_second=0.0, burst=2),
        )
    )
    headers = {"Authorization": "Bearer tight-key"}
    assert client.post("/v1/chat/completions", json=chat_body(), headers=headers).status_code == 200
    assert client.post("/v1/chat/completions", json=chat_body(), headers=headers).status_code == 200
    r = client.post("/v1/chat/completions", json=chat_body(), headers=headers)
    assert r.status_code == 429
    assert r.json()["error"]["code"] == "rate_limited"
    assert r.headers["retry-after"] == "1"
