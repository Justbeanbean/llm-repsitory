"""熔断（单进程）：连续失败打开 → 跳过候选 → half_open 探测恢复。"""
from conftest import FakeProvider, chat_body, net_error
from llm_gateway.core.circuit_breaker import CircuitBreaker


def test_circuit_opens_and_skips_candidates(client_factory, auth):
    provider = FakeProvider(default_error=net_error())
    client = client_factory(provider=provider)   # 阈值 2（conftest 默认）

    # 请求 1：c1/c2 各重试 2 次后失败（各计 1 次熔断失败）
    assert client.post("/v1/llm", json=chat_body(), headers=auth).status_code == 502
    assert provider.complete_calls == 4
    # 请求 2：再次全失败 → 两个候选都达阈值，熔断打开
    assert client.post("/v1/llm", json=chat_body(), headers=auth).status_code == 502
    assert provider.complete_calls == 8
    # 请求 3：候选全部被熔断跳过 → 503，且不再打上游
    r = client.post("/v1/llm", json=chat_body(), headers=auth)
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "model_unavailable"
    assert provider.complete_calls == 8


def test_half_open_recovery(client_factory, auth):
    provider = FakeProvider(default_error=net_error())
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.0)
    client = client_factory(provider=provider, breaker=breaker)

    # 阈值 1：一轮后两个候选都打开
    assert client.post("/v1/llm", json=chat_body(), headers=auth).status_code == 502
    # 上游恢复：half_open 放行探测 → 成功 → 熔断重置
    provider.default_error = None
    r = client.post("/v1/llm", json=chat_body(), headers=auth)
    assert r.status_code == 200
    assert provider.complete_calls == 5   # 4（失败轮）+ 1（探测成功）


def test_content_errors_do_not_trip_breaker(client_factory, auth):
    from conftest import empty_content_error

    provider = FakeProvider(default_error=empty_content_error())
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=3600.0)
    client = client_factory(provider=provider, breaker=breaker)

    # L3 内容问题连续失败，但不应打开熔断（内容问题 ≠ 上游故障）
    for _ in range(3):
        assert (
            client.post("/v1/llm", json=chat_body(), headers=auth).status_code == 422
        )
    assert provider.complete_calls == 3   # 每次都正常打到上游，没有被跳过
