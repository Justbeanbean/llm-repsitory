"""OpenAI Compatible Adapter：集中处理供应商协议、认证，以及 L1/L2/L3 判定。

- L1/L2：SDK 异常 → _classify_sdk_error 统一分类（消息先脱敏再抛出）
- L3：协议 choice 显式判定（消灭 choices[0] 的 IndexError 地雷）
- usage：三态记账（usage_from_payload），判定点与告警点是同一个函数
- API Key 保留在 Gateway 内，业务侧不接触供应商密钥
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

from llm_gateway.core.exceptions import FailureClass, GatewayError, ProviderError
from llm_gateway.core.logging_setup import get_logger
from llm_gateway.core.masking import sanitize
from llm_gateway.core.models import CandidateConfig, Message, ProviderResult, Usage

logger = get_logger("core.provider")


def _classify_sdk_error(exc: Exception, retry_statuses: tuple[int, ...]) -> ProviderError:
    """L1/L2 判定：SDK 异常 → 已分类 ProviderError。消息脱敏后截断。

    retryable 由 retry_statuses 配置驱动（yaml: retry.retry_statuses）。
    """
    raw = sanitize(str(exc))[:300]
    if isinstance(exc, RateLimitError):
        retry_after: float | None = None
        response = getattr(exc, "response", None)
        header = response.headers.get("retry-after") if response is not None else None
        if header is not None:
            try:
                retry_after = float(header)
            except (TypeError, ValueError):
                retry_after = None
        return ProviderError(
            FailureClass.RATE_LIMIT,
            "rate_limited",
            "上游限流",
            retryable=True,
            retry_after=retry_after,
        )
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return ProviderError(
            FailureClass.NETWORK,
            "network_error",
            f"上游连接失败: {raw}",
            retryable=True,
        )
    if isinstance(exc, APIStatusError):
        status = getattr(exc, "status_code", 0) or 0
        if status == 408:   # 请求超时 → 网络类
            return ProviderError(
                FailureClass.NETWORK, "network_error",
                f"上游请求超时: {raw}", retryable=status in retry_statuses,
            )
        failure_class = FailureClass.PROVIDER_SERVER if status >= 500 else FailureClass.PROVIDER_CLIENT
        return ProviderError(
            failure_class,
            f"provider_{'server' if status >= 500 else 'client'}_error",
            f"上游返回 {status}: {raw}",
            retryable=status in retry_statuses,
            status_code=status,
        )
    return ProviderError(
        FailureClass.UNKNOWN,
        "provider_unknown_error",
        f"未知异常: {raw}",
        retryable=False,
    )


def usage_from_payload(usage: Any) -> Usage:
    """Usage 三态记账：判定点与告警点必须在同一个函数。

    - 上游没回 usage / 部分字段缺失 → tokens 记 0（占位）+ usage_missing=True + warning
    - 上游明确回 0                 → 记 0，不告警（已知为 0 是事实）
    - usage 是账本数据：宽松放行，绝不影响业务成败
    """
    if usage is None:
        logger.warning("usage_missing", extra={"reason": "provider_omitted_usage"})
        return Usage(usage_missing=True)
    input_tokens = getattr(usage, "prompt_tokens", None)
    output_tokens = getattr(usage, "completion_tokens", None)
    if input_tokens is None or output_tokens is None:
        logger.warning(
            "usage_missing",
            extra={
                "reason": "partial_usage",
                "has_input": input_tokens is not None,
                "has_output": output_tokens is not None,
            },
        )
        return Usage(
            input_tokens=input_tokens or 0,
            output_tokens=output_tokens or 0,
            usage_missing=True,
        )
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", None) if details is not None else None
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached or 0,
    )


class OpenAICompatibleProvider:
    """OpenAI Compatible 协议实现。

    retry_statuses：哪些 HTTP 状态码值得同模型重试（yaml: retry.retry_statuses）。
    """

    provider_id = "openai_compatible"

    def __init__(self, retry_statuses: tuple[int, ...] = (408, 409, 429, 500, 502, 503, 504)) -> None:
        self._retry_statuses = tuple(retry_statuses)

    def create_client(self, config: CandidateConfig) -> AsyncOpenAI:
        if not config.api_key:
            raise GatewayError("gateway_misconfigured", "Gateway 模型凭据未配置", 503)
        # SDK 内置重试关闭：重试口径唯一归 core.retry
        return AsyncOpenAI(api_key=config.api_key, base_url=config.base_url, max_retries=0)

    async def complete(
        self,
        config: CandidateConfig,
        messages: list[Message],
        timeout_seconds: float,
        response_schema: dict[str, Any] | None,
    ) -> ProviderResult:
        # 统一请求 → OpenAI Compatible 调用，隔离厂商协议差异
        request_data: dict[str, Any] = {
            "model": config.provider_model,
            "messages": [message.model_dump() for message in messages],
            "timeout": timeout_seconds,
        }
        if response_schema is not None:
            if config.structured_output_mode == "json_schema":
                request_data["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "agent_response",
                        "strict": True,
                        "schema": response_schema,
                    },
                }
            else:
                request_data["response_format"] = {"type": "json_object"}
                request_data["messages"] = [
                    {
                        "role": "system",
                        "content": (
                            "只返回一个合法 JSON 对象，必须严格符合下列 JSON Schema，"
                            "不要返回 Markdown 或额外文字："
                            f"{json.dumps(response_schema, ensure_ascii=False)}"
                        ),
                    },
                    *request_data["messages"],
                ]
        try:
            completion = await (
                self.create_client(config).chat.completions.create(**request_data)
            )
        except GatewayError as exc:
            # 配置类错误（如密钥缺失）：包装为可 fallback 的 ProviderError（换 provider 可绕开）
            raise ProviderError(
                FailureClass.PROVIDER_SERVER, exc.code, exc.message, retryable=False
            ) from exc
        except Exception as exc:
            raise _classify_sdk_error(exc, self._retry_statuses) from exc

        # L3：协议 choice 显式判定（不裸取 choices[0]）
        if not completion.choices:
            raise ProviderError(
                FailureClass.PROTOCOL,
                "empty_content",
                "上游未返回 choices",
                retryable=False,
            )
        message = completion.choices[0].message
        if message.content is None:
            detail = "模型拒绝回答" if message.refusal else "上游未返回文本内容"
            raise ProviderError(
                FailureClass.PROTOCOL,
                "empty_content",
                detail,
                retryable=False,
            )
        # 供应商请求 ID 进入 Trace（上游故障排查）
        return ProviderResult(
            content=message.content,
            usage=usage_from_payload(completion.usage),
            upstream_id=getattr(completion, "id", None),
        )

    def stream(
        self,
        config: CandidateConfig,
        messages: list[Message],
        timeout_seconds: float,
    ) -> AsyncIterator[dict[str, Any]]:
        return self._stream(config, messages, timeout_seconds)

    async def _stream(
        self,
        config: CandidateConfig,
        messages: list[Message],
        timeout_seconds: float,
    ) -> AsyncIterator[dict[str, Any]]:
        # L3 delta 宽松模式：异常块跳过 + warning，不打断流
        # stream_usage：向上游请求流末 usage 事件（choices 为空但带 usage 的尾块）
        request_kwargs: dict[str, Any] = dict(
            model=config.provider_model,
            messages=[message.model_dump() for message in messages],
            stream=True,
            timeout=timeout_seconds,
        )
        if config.stream_usage:
            request_kwargs["stream_options"] = {"include_usage": True}
        try:
            response = await self.create_client(config).chat.completions.create(**request_kwargs)
            async for chunk in response:
                # usage 采集：OpenAI 规范是空 choices 尾块，DeepSeek 则附在带
                # finish_reason 的尾块上——两种形态统一处理，谁带 usage 都收
                if getattr(chunk, "usage", None) is not None:
                    yield {"type": "usage", "usage": usage_from_payload(chunk.usage)}
                # 供应商请求 ID（首个 chunk 携带）→ Trace 上游归因
                if getattr(chunk, "id", None):
                    yield {"type": "meta", "upstream_id": chunk.id}
                if not chunk.choices:
                    logger.warning(
                        "stream_chunk_skipped", extra={"reason": "empty_choices"}
                    )
                    continue
                choice = chunk.choices[0]
                delta = choice.delta.content if choice.delta is not None else None
                if delta:
                    yield {"type": "delta", "delta": delta}
        except GatewayError as exc:
            raise ProviderError(
                FailureClass.PROVIDER_SERVER, exc.code, exc.message, retryable=False
            ) from exc
        except Exception as exc:
            raise _classify_sdk_error(exc, self._retry_statuses) from exc
