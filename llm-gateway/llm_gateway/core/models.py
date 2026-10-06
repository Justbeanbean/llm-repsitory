"""领域模型：三层共享的纯数据结构，不依赖任何上层。

供应商标识三级体系：
  provider        → provider_id，决定用哪个 Adapter（注册表）
  公开模型别名     → 对调用方暴露的唯一入口（/v1/models 列出）
  candidate.alias → 候选内部名 + provider_model（供应商真实模型名，仅 Adapter 内部使用）
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llm_gateway.core.exceptions import FailureClass


class Message(BaseModel):
    """跨模型通用的单条对话消息，隔离供应商消息格式差异。"""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=20_000)


class PromptSelection(BaseModel):
    """只允许调用方选择受控模板及变量，不能提交或覆盖模板正文。

    version="latest"（默认）→ 该模板的最新版本。
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=100)
    version: str = Field(default="latest", min_length=1, max_length=50)
    variables: dict[str, str] = Field(default_factory=dict)


class LLMRequest(BaseModel):
    """统一 Gateway 请求协议，在 HTTP 入口拦截不合法组合和字段。"""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1, max_length=100)
    messages: list[Message] = Field(min_length=1, max_length=100)
    stream: bool = False
    response_schema: dict[str, Any] | None = None
    # None = 未显式指定：由 Prompt Bundle 默认值或全局默认（30s）接管
    timeout_seconds: float | None = Field(default=None, gt=0, le=120)
    prompt: PromptSelection | None = None

    @model_validator(mode="after")
    def check_supported_combination(self) -> "LLMRequest":
        if self.stream and self.response_schema is not None:
            raise ValueError("stream 与 response_schema 不能同时使用")
        return self


class Usage(BaseModel):
    """Token 用量三态记账。

    - 上游明确回 usage 且为 0        → tokens=0, usage_missing=False（已知为 0，不告警）
    - 上游未回 usage / 部分字段缺失  → tokens=0（占位）, usage_missing=True + warning

    usage 是账本数据而非业务结果：不参与成败判定，缺失只告警不失败。
    """

    model_config = ConfigDict(extra="forbid")

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cached_tokens: int = Field(default=0, ge=0)
    usage_missing: bool = False

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            usage_missing=self.usage_missing or other.usage_missing,
        )


class LLMResponse(BaseModel):
    """统一模型调用结果（内部协议；OpenAI 兼容层由 api 层转换）。"""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    model: str
    content: str
    parsed: dict[str, Any] | list[Any] | None = None
    usage: Usage
    latency_ms: int = Field(ge=0)
    attempts: int = Field(ge=1)


class PromptTemplate(BaseModel):
    """Prompt Bundle：版本 + 沙箱模板 + 输入/输出 Schema + 模型与参数默认值。

    - input_schema：变量校验（jsonschema），缺变量/多变量/类型不符都在调用模型前失败
    - output_schema：绑定的 Structured Output Schema（请求未显式指定时生效）
    - business_model：本地业务规则（pydantic 模型注册名），JSON 合法但业务不合法不放行
    - default_model / default_timeout_seconds：模型和参数默认值（记录进 Trace 供审计）
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    system_template: str
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    business_model: str | None = None
    default_model: str | None = None
    default_timeout_seconds: float | None = Field(default=None, gt=0, le=120)

    @property
    def content_hash(self) -> str:
        """模板内容 hash：资产指纹，进 Trace 供版本审计。"""
        return hashlib.sha256(self.system_template.encode("utf-8")).hexdigest()[:12]


def schema_fingerprint(schema: dict[str, Any]) -> str:
    """response_schema 指纹：每次调用可定位使用的 Schema。"""
    canonical = json.dumps(schema, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class Price:
    """单位：美元 / 1M tokens（providers.yaml 的 pricing 段，按供应商模型名查价）。"""

    input_per_million: float
    output_per_million: float
    cached_input_per_million: float = 0.0


@dataclass(frozen=True)
class ProviderConfig:
    """供应商连接配置（yaml providers 段）：api_key 为环境变量替换后的值。

    adapter：实现该供应商协议的 Adapter 注册名（默认 openai_compatible）。
    """

    name: str
    base_url: str
    api_key: str
    timeout_seconds: float = 120.0
    enabled: bool = True
    adapter: str = "openai_compatible"


@dataclass(frozen=True)
class CandidateConfig:
    """一条候选路由（运行时组装完成：routes 项 + ProviderConfig + pricing）。

    api = "both" 支持 chat 与 /v1/responses；"chat" 仅 chat completions。
    """

    alias: str                          # 候选内部名（trace 归因用）
    provider: str                       # 供应商名（providers 段 key，trace/熔断键归因）
    adapter: str                        # Adapter 注册名 → 决定用哪个实现
    provider_model: str                 # 供应商真实模型名，仅 Adapter 内部使用
    base_url: str
    api_key: str                        # 密钥值（来自 providers 段，日志必须脱敏）
    timeout_seconds: float = 120.0      # provider 级默认超时
    api: Literal["chat", "both"] = "both"
    supports_structured_output: bool = True
    structured_output_mode: Literal["json_schema", "json_object"] = "json_schema"
    stream_usage: bool = True           # 流式是否请求上游回传 usage（不兼容端点可在 route 级关闭）
    weight: int = 1                     # 加权轮询权重（priority 策略下表示顺序）
    price: Price = Price(input_per_million=0.0, output_per_million=0.0)
    price_version: str = "v1"           # 价格版本：成本审计可解释（调价前后可对比）


@dataclass(frozen=True)
class ModelRoute:
    """公开模型别名 → 路由策略 + 候选列表。

    strategy=priority  → 按列表顺序：首个为主，其余为 fallback 链
    strategy=weighted  → 加权轮询选主，剩余按权重降序作 fallback 链
    """

    alias: str
    strategy: Literal["priority", "weighted"] = "priority"
    candidates: tuple[CandidateConfig, ...] = ()


@dataclass(frozen=True)
class ProviderResult:
    """Adapter 完成调用的统一返回：内容 + 用量 + 供应商请求 ID。"""

    content: str
    usage: Usage
    upstream_id: str | None = None


class CallTrace(BaseModel):
    """单次调用元数据审计记录（SQLite 账本行）；默认不记录 Prompt 与消息正文。

    Run/Step/Call 关联：调用方通过 X-Run-Id / X-Step-Id 头传入，贯穿日志与账本。
    终态唯一：success / failed / cancelled 三选一，各路径只记一次。
    """

    model_config = ConfigDict(extra="forbid")

    request_id: str
    timestamp: datetime
    caller_fingerprint: str | None = None   # 调用方身份只保留短指纹
    run_id: str | None = None
    step_id: str | None = None
    requested_model: str
    final_candidate: str | None = None
    provider: str | None = None
    upstream_request_id: str | None = None  # 供应商请求 ID（排查上游故障）
    prompt_name: str | None = None
    prompt_version: str | None = None
    prompt_hash: str | None = None          # Prompt 内容 hash（版本审计）
    schema_hash: str | None = None          # 实际使用的 response_schema 指纹
    price_version: str | None = None        # 计价使用的价格版本
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cached_tokens: int = Field(default=0, ge=0)
    usage_missing: bool = False
    cost_usd: float | None = None       # None = 未知（usage 缺失或调用失败时）
    ttft_ms: int | None = None          # 口径：首个业务可见 delta（非连接建立）
    latency_ms: int = Field(default=0, ge=0)
    network_attempts: int = Field(default=0, ge=0)
    repair_attempts: int = Field(default=0, ge=0)
    fallback_count: int = Field(default=0, ge=0)
    attempts: int = Field(default=0, ge=0)
    status: Literal["success", "failed", "cancelled"]
    error_code: str | None = None
    failure_class: FailureClass | None = None
    failure_layer: str | None = None
    route_decisions: list[dict[str, Any]] | None = None   # 可解释路由：候选→决策→理由
