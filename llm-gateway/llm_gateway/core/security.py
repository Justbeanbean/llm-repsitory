"""Bearer API Key 鉴权（core）：常量时间比较 + 调用方短指纹。

刻意保留的边界：单进程配置内鉴权；多副本/吊销需外部认证服务。
"""
from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass


@dataclass(frozen=True)
class Caller:
    """已鉴权的调用方身份（账本只保留 fingerprint）。"""

    name: str
    fingerprint: str
    requests_per_second: float
    burst: int
    max_concurrency: int = 4


def _fingerprint(name: str, key: str) -> str:
    return hashlib.sha256(f"{name}:{key}".encode("utf-8")).hexdigest()[:12]


def authenticate(authorization_header: str, api_keys: tuple | list) -> Caller | None:
    """解析 Authorization: Bearer <key>；失败返回 None（由 api 层转 401）。"""
    if not authorization_header.startswith("Bearer "):
        return None
    key = authorization_header[len("Bearer "):].strip()
    if not key:
        return None
    for setting in api_keys:
        if hmac.compare_digest(setting.key, key):
            return Caller(
                name=setting.name,
                fingerprint=_fingerprint(setting.name, setting.key),
                requests_per_second=setting.requests_per_second,
                burst=setting.burst,
                max_concurrency=getattr(setting, "max_concurrency", 4),
            )
    return None
