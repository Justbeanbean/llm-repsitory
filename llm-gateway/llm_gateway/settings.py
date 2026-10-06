"""配置加载：providers.yaml → 强类型 Settings。

结构：
  service_name / api_keys（字符串列表，空 = 开发模式不鉴权）/ database_url
  structured_output_retries / retry（max_attempts_per_route、retry_statuses）
  circuit_breaker（cooldown_seconds）/ rate_limit（requests_per_minute、enabled）
  stream_checkpoint / providers（base_url、api_key、timeout、enabled）
  models（strategy: priority | weighted_round_robin；routes: provider/model/api/weight）
  pricing（按供应商模型名，美元 / 1M tokens）
  可选扩展段：prompt_templates / governance / security / log_level

环境变量替换：${VAR} / ${VAR:-default} / ${VAR:default}（bash 风格 `:-`），
仅匹配大写名——模板里的小写占位符（如 ${product_name}）不受影响。
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from llm_gateway.core.exceptions import GatewayError
from llm_gateway.core.models import (
    CandidateConfig,
    ModelRoute,
    Price,
    PromptTemplate,
    ProviderConfig,
)
from llm_gateway.core.retry import RetryPolicy

_ENV_RE = re.compile(r"\$\{([A-Z0-9_]+)(?::(-?)([^}]*))?\}")
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_DEFAULT_RETRY_STATUSES = (408, 409, 429, 500, 502, 503, 504)


def _substitute(value: Any) -> Any:
    if isinstance(value, str):

        def repl(match: re.Match[str]) -> str:
            name, dash, default = match.group(1), match.group(2), match.group(3)
            env = os.getenv(name)
            if env is not None and env != "":
                return env
            if default is not None:
                # ${VAR:-def}：未设置或为空都用默认；${VAR:def}：仅未设置时用默认
                if dash == "-" or env is None:
                    return default
            raise ValueError(f"环境变量 {name} 未设置且无默认值")

        return _ENV_RE.sub(repl, value)
    if isinstance(value, dict):
        return {key: _substitute(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_substitute(item) for item in value]
    return value


@dataclass(frozen=True)
class ApiKeySetting:
    name: str
    key: str
    requests_per_second: float = 10.0
    burst: int = 20
    max_concurrency: int = 4


@dataclass(frozen=True)
class BreakerConfig:
    failure_threshold: int = 5
    recovery_timeout_seconds: float = 30.0


@dataclass(frozen=True)
class RateLimitConfig:
    enabled: bool = True
    requests_per_second: float = 1.0    # 由 requests_per_minute 换算
    burst: int = 10


@dataclass(frozen=True)
class Settings:
    service_name: str = "llm-gateway"
    models: dict[str, ModelRoute] = field(default_factory=dict)
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    templates: dict[tuple[str, str], PromptTemplate] = field(default_factory=dict)
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    retry_statuses: tuple[int, ...] = _DEFAULT_RETRY_STATUSES
    api_keys: tuple[ApiKeySetting, ...] = ()
    rate_limit: RateLimitConfig = RateLimitConfig()
    breaker_config: BreakerConfig = BreakerConfig()
    structured_output_retries: int = 1  # L4 反喂修复次数
    max_caller_concurrency: int = 4     # 调用方默认并发上限（含开发模式匿名调用方）
    cors_allow_origins: tuple[str, ...] = ()   # 空 = 不启用 CORS（服务端 Agent 直连）
    database_path: Path = Path("data/gateway.db")
    stream_checkpoint: dict[str, Any] = field(
        default_factory=lambda: {"enabled": False}
    )
    run_budget_seconds: float = 120.0          # Run 预算（可选扩展段 governance）
    max_concurrency_per_candidate: int = 8     # 每供应商候选并发上限（超限排队）
    allowed_base_url_schemes: tuple[str, ...] = ("http", "https")
    allowed_base_url_hosts: tuple[str, ...] = ()   # 空 = 不限制主机
    log_level: str = "INFO"

    @property
    def auth_required(self) -> bool:
        """api_keys 全空 → 开发模式不鉴权。"""
        return bool(self.api_keys)


def _find_config(path: str | Path | None) -> Path:
    if path:
        return Path(path)
    env = os.getenv("GATEWAY_CONFIG")
    if env:
        return Path(env)
    for candidate in (_PROJECT_ROOT / "providers.yaml", Path.cwd() / "providers.yaml"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("未找到 providers.yaml（可用 GATEWAY_CONFIG 指定路径）")


def _validate_base_url(
    base_url: str,
    schemes: tuple[str, ...],
    hosts: tuple[str, ...],
) -> None:
    """base_url 白名单：协议必须 http/https；声明主机白名单时校验主机。"""
    parsed = urlparse(base_url)
    if parsed.scheme not in schemes:
        raise GatewayError(
            "gateway_misconfigured", f"base_url 协议不受允许: {parsed.scheme}", 503
        )
    if hosts and parsed.hostname not in hosts:
        raise GatewayError(
            "gateway_misconfigured", f"base_url 主机不在白名单: {parsed.hostname}", 503
        )


def _parse_providers(raw: dict) -> dict[str, ProviderConfig]:
    providers: dict[str, ProviderConfig] = {}
    for name, item in raw.get("providers", {}).items():
        providers[name] = ProviderConfig(
            name=name,
            base_url=item["base_url"],
            api_key=str(item.get("api_key", "")),
            timeout_seconds=float(item.get("timeout_seconds", 120)),
            enabled=bool(item.get("enabled", True)),
            adapter=item.get("adapter", "openai_compatible"),
        )
    return providers


def _parse_models(
    raw: dict,
    providers: dict[str, ProviderConfig],
    pricing: dict[str, Price],
    pricing_version: str,
    schemes: tuple[str, ...],
    hosts: tuple[str, ...],
) -> dict[str, ModelRoute]:
    models: dict[str, ModelRoute] = {}
    for alias, item in raw.get("models", {}).items():
        candidates: list[CandidateConfig] = []
        for index, route in enumerate(item.get("routes", [])):
            provider = providers.get(route["provider"])
            if provider is None:
                raise GatewayError(
                    "gateway_misconfigured",
                    f"模型 {alias} 引用了未定义的 provider: {route['provider']}",
                    503,
                )
            if not provider.enabled:
                continue   # 禁用的 provider：候选直接不进入路由链
            _validate_base_url(provider.base_url, schemes, hosts)
            provider_model = route["model"]
            price = pricing.get(provider_model, Price(input_per_million=0.0, output_per_million=0.0))
            candidates.append(
                CandidateConfig(
                    alias=route.get("alias", f"{alias}-{route['provider']}-{index + 1}"),
                    provider=provider.name,
                    adapter=provider.adapter,
                    provider_model=provider_model,
                    base_url=provider.base_url,
                    api_key=provider.api_key,
                    timeout_seconds=provider.timeout_seconds,
                    api=route.get("api", "both"),
                    supports_structured_output=bool(route.get("supports_structured_output", True)),
                    structured_output_mode=route.get("structured_output_mode", "json_schema"),
                    stream_usage=bool(route.get("stream_usage", True)),
                    weight=int(route.get("weight", 1)),
                    price=price,
                    # 价格版本：route 级覆盖 > pricing 段全局版本 > v1（trace 成本审计）
                    price_version=str(route.get("price_version", pricing_version)),
                )
            )
        strategy = item.get("strategy", "priority")
        if strategy == "weighted_round_robin":   # yaml 别名 → 内部策略名
            strategy = "weighted"
        models[alias] = ModelRoute(
            alias=alias, strategy=strategy, candidates=tuple(candidates)
        )
    return models


def _parse_api_keys(raw: dict, rate_limit: RateLimitConfig, max_concurrency: int) -> tuple:
    """api_keys 为纯密钥字符串列表：空/全空 = 开发模式不鉴权。

    身份名由 key 内容派生（稳定指纹），与 yaml 列表顺序无关——
    重排 api_keys 不会导致限流桶 / 并发槽 / metrics by_caller 的租户身份漂移。
    """
    raw_keys = [k for k in raw.get("api_keys", []) if k]
    return tuple(
        ApiKeySetting(
            name=f"key-{hashlib.sha256(key.encode('utf-8')).hexdigest()[:12]}",
            key=key,
            requests_per_second=rate_limit.requests_per_second,
            burst=rate_limit.burst,
            max_concurrency=max_concurrency,
        )
        for key in raw_keys
    )


def load_settings(path: str | Path | None = None) -> Settings:
    raw = _substitute(yaml.safe_load(_find_config(path).read_text(encoding="utf-8")))

    # pricing：按供应商模型名全局定价（美元 / 1M tokens）
    pricing: dict[str, Price] = {
        model: Price(
            input_per_million=float(item.get("input_per_million", 0.0)),
            output_per_million=float(item.get("output_per_million", 0.0)),
            cached_input_per_million=float(item.get("cached_input_per_million", 0.0)),
        )
        for model, item in raw.get("pricing", {}).items()
    }

    # rate_limit：requests_per_minute → 每秒令牌速率
    rate_raw = raw.get("rate_limit", {})
    rate_limit = RateLimitConfig(
        enabled=bool(rate_raw.get("enabled", True)),
        requests_per_second=float(rate_raw.get("requests_per_minute", 60)) / 60.0,
        burst=int(rate_raw.get("burst", 10)),
    )

    retry = raw.get("retry", {})
    retry_policy = RetryPolicy(
        max_attempts=int(retry.get("max_attempts_per_route", retry.get("max_attempts", 3))),
        base_delay=float(retry.get("base_delay_seconds", retry.get("base_delay", 0.2))),
        max_delay=float(retry.get("max_delay_seconds", retry.get("max_delay", 5.0))),
        backoff_factor=float(retry.get("backoff_factor", 2.0)),
        jitter=bool(retry.get("jitter", True)),
    )
    retry_statuses = tuple(int(s) for s in retry.get("retry_statuses", _DEFAULT_RETRY_STATUSES))

    breaker = raw.get("circuit_breaker", {})
    breaker_config = BreakerConfig(
        failure_threshold=int(breaker.get("failure_threshold", 5)),
        recovery_timeout_seconds=float(
            breaker.get("cooldown_seconds", breaker.get("recovery_timeout_seconds", 30.0))
        ),
    )

    security = raw.get("security", {})
    schemes = tuple(security.get("allowed_base_url_schemes", ("http", "https")))
    hosts = tuple(security.get("allowed_base_url_hosts", ()))

    providers = _parse_providers(raw)
    pricing_version = str(raw.get("pricing_version", "v1"))
    models = _parse_models(raw, providers, pricing, pricing_version, schemes, hosts)
    max_caller_concurrency = int(raw.get("governance", {}).get("max_caller_concurrency", 4))
    api_keys = _parse_api_keys(raw, rate_limit, max_caller_concurrency)

    templates: dict[tuple[str, str], PromptTemplate] = {}
    for item in raw.get("prompt_templates", []):
        template = PromptTemplate(
            name=item["name"],
            version=item["version"],
            system_template=item["system_template"],
            input_schema=item.get("input_schema"),
            output_schema=item.get("output_schema"),
            business_model=item.get("business_model"),
            default_model=item.get("default_model"),
            default_timeout_seconds=item.get("default_timeout_seconds"),
        )
        templates[(template.name, template.version)] = template

    governance = raw.get("governance", {})
    cors = raw.get("cors", {})

    return Settings(
        service_name=str(raw.get("service_name", "llm-gateway")),
        models=models,
        providers=providers,
        templates=templates,
        retry_policy=retry_policy,
        retry_statuses=retry_statuses,
        api_keys=api_keys,
        rate_limit=rate_limit,
        breaker_config=breaker_config,
        structured_output_retries=int(raw.get("structured_output_retries", 1)),
        max_caller_concurrency=max_caller_concurrency,
        cors_allow_origins=tuple(cors.get("allow_origins", ())),
        database_path=Path(raw.get("database_url", "data/gateway.db")),
        stream_checkpoint=dict(raw.get("stream_checkpoint", {"enabled": False})),
        run_budget_seconds=float(governance.get("run_budget_seconds", 120.0)),
        max_concurrency_per_candidate=int(governance.get("max_concurrency_per_candidate", 8)),
        allowed_base_url_schemes=schemes,
        allowed_base_url_hosts=hosts,
        log_level=str(raw.get("log_level", "INFO")),
    )
