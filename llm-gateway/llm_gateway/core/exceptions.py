"""错误与失败分类：全系统唯一事实源。

重试判定、fallback 判定、日志归因、trace 归因、HTTP 映射全部基于 FailureClass，
禁止各层自行 isinstance 判断——这是重试逻辑在各层级间协调一致的保证。
"""
from __future__ import annotations

from enum import Enum


class FailureClass(str, Enum):
    """失败分类，对应上游响应检查的四层模型（L1-L4）。"""

    NETWORK = "network"                # L1: DNS / TCP / TLS / 超时
    RATE_LIMIT = "rate_limit"          # L2: 429
    PROVIDER_SERVER = "provider_5xx"   # L2: 5xx / 408
    PROVIDER_CLIENT = "provider_4xx"   # L2: 其他非 2xx（确定性，不重试不换）
    PROTOCOL = "protocol"              # L3: choices 空 / content None / 结构异常
    SCHEMA = "schema"                  # L4: JSON 解析 / jsonschema 校验失败
    UNKNOWN = "unknown"


# 值得换 provider 的类别：仅非内容问题（L1/L2）
FALLBACK_WORTHY = frozenset(
    {
        FailureClass.NETWORK,
        FailureClass.RATE_LIMIT,
        FailureClass.PROVIDER_SERVER,
    }
)

# 失败类别 → 观测层标签（trace.failure_layer / 日志归因）
FAILURE_LAYER = {
    FailureClass.NETWORK: "l1_network",
    FailureClass.RATE_LIMIT: "l2_http",
    FailureClass.PROVIDER_SERVER: "l2_http",
    FailureClass.PROVIDER_CLIENT: "l2_http",
    FailureClass.PROTOCOL: "l3_protocol",
    FailureClass.SCHEMA: "l4_schema",
    FailureClass.UNKNOWN: "unknown",
}


class GatewayError(Exception):
    """对调用方暴露的稳定错误：错误码 + HTTP 状态 + 失败归因。

    message 只允许白名单文案，禁止包含堆栈、URL、密钥等敏感信息
    （上游原文必须先经 core.masking.sanitize 处理）。
    """

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 502,
        failure_class: FailureClass | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        self.failure_class = failure_class
        self.headers = headers
        super().__init__(message)


class DeadlineExceededError(Exception):
    """Run 预算（调用 deadline）耗尽。

    重试、Fallback 和修复调用共用同一预算；耗尽后立即停止一切尝试（504）。
    """


class ProviderError(Exception):
    """core Adapter 抛出的、已分类的供应商侧失败。

    retryable    → 同模型内是否值得重试（由 retry_async 唯一消费）
    fallback_worthy → 是否值得换 provider（由 service 层唯一消费）
    attempts     → 抛出时本 route 已消耗的尝试次数（由 retry_async 回填）
    status_code  → 上游 HTTP 状态码（PROVIDER_CLIENT 类透传给调用方）
    """

    def __init__(
        self,
        failure_class: FailureClass,
        error_code: str,
        message: str,
        *,
        retryable: bool = False,
        retry_after: float | None = None,
        status_code: int | None = None,
    ) -> None:
        self.failure_class = failure_class
        self.error_code = error_code
        self.message = message
        self.retryable = retryable
        self.retry_after = retry_after
        self.status_code = status_code
        self.attempts = 0
        super().__init__(message)

    @property
    def fallback_worthy(self) -> bool:
        """内容问题（L3/L4）与确定性 4xx 一律不换 provider。"""
        return self.failure_class in FALLBACK_WORTHY
