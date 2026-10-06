"""敏感内容脱敏规则引擎。

执行位置（三层各司其职）：
- api 层：入参侧——日志记录请求前对 headers / 请求 ID 脱敏
- service 层：落日志前——任何要写入日志/trace 的消息内容、变量值必须先过 sanitize/mask_text
- core 层：出口侧兜底——供应商异常 message 先 sanitize 再抛出/记录；
          GatewayError.message 只允许白名单文案，天然无敏感信息
"""
from __future__ import annotations

import hashlib
import re

_API_KEY_RE = re.compile(r"\b(?:sk|rk)-[A-Za-z0-9_\-]{8,}\b")
_BEARER_RE = re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{8,}")
_PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_IDCARD_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")


def mask_secret(value: str, keep: int = 4) -> str:
    """密钥类：保留前 keep 后 keep 位，中间打码。sk-a1***xyz"""
    if len(value) <= keep * 2:
        return "***"
    return f"{value[:keep]}***{value[-keep:]}"


def mask_text(text: str, limit: int = 200) -> str:
    """消息正文：默认不落日志；必须记录时截断 + 哈希指纹。"""
    fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    snippet = text[:limit]
    ellipsis = "…" if len(text) > limit else ""
    return f"{snippet}{ellipsis}(sha256:{fingerprint})"


def sanitize(text: str) -> str:
    """对任意将写入日志/错误的文本做规则脱敏（API Key/Bearer/手机号/邮箱/身份证）。"""
    text = _API_KEY_RE.sub(lambda m: m.group(0)[:4] + "***", text)
    text = _BEARER_RE.sub("Bearer ***", text)
    text = _IDCARD_RE.sub("******************", text)
    text = _PHONE_RE.sub(lambda m: m.group(0)[:3] + "****" + m.group(0)[-4:], text)
    text = _EMAIL_RE.sub(lambda m: m.group(0)[0] + "***@" + m.group(0).split("@", 1)[1], text)
    return text
