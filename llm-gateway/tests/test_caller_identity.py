"""租户身份稳定性：调用方指纹与 yaml key 列表顺序无关。

重排 api_keys 不得导致限流桶 / 并发槽 / metrics by_caller 的租户身份漂移。
"""
from llm_gateway.core.security import authenticate
from llm_gateway.settings import RateLimitConfig, _parse_api_keys


def test_caller_identity_stable_across_key_order():
    keys_a = _parse_api_keys({"api_keys": ["k-one", "k-two"]}, RateLimitConfig(), 4)
    keys_b = _parse_api_keys({"api_keys": ["k-two", "k-one"]}, RateLimitConfig(), 4)

    # 身份名由 key 内容派生，与顺序无关
    names_a = {s.key: s.name for s in keys_a}
    names_b = {s.key: s.name for s in keys_b}
    assert names_a == names_b
    assert all(name.startswith("key-") for name in names_a.values())

    # 鉴权后的调用方指纹（账本 caller_fingerprint / metrics by_caller 的租户键）同样稳定
    for key in ("k-one", "k-two"):
        caller_a = authenticate(f"Bearer {key}", keys_a)
        caller_b = authenticate(f"Bearer {key}", keys_b)
        assert caller_a is not None and caller_b is not None
        assert caller_a.name == caller_b.name
        assert caller_a.fingerprint == caller_b.fingerprint

    # 限流桶 / 并发槽 key（caller:{name}）随之稳定
    assert {f"caller:{c.name}" for c in (authenticate("Bearer k-one", keys_a),)} == {
        f"caller:{c.name}" for c in (authenticate("Bearer k-one", keys_b),)
    }


def test_duplicate_keys_share_single_tenant_identity():
    # 相同 key 重复配置 → 同一身份（共享限流桶与并发槽），而非两个租户
    keys = _parse_api_keys({"api_keys": ["dup", "dup"]}, RateLimitConfig(), 4)
    assert len({s.name for s in keys}) == 1
