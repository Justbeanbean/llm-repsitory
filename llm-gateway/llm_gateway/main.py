"""应用装配：中间件、异常处理器、路由注册，以及各层组件构建。

依赖方向：api → service → core（core 不感知任何上层）。
create_app 的可注入参数用于自动化测试（Fake Adapter / 内存账本等）。
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from llm_gateway.api.errors import (
    gateway_error_handler,
    http_exception_handler,
    unhandled_exception_handler,
    validation_error_handler,
)
from llm_gateway.api.middleware import RequestContextMiddleware
from llm_gateway.api.routes import router
from llm_gateway.api.routes_openai import router as openai_router
from llm_gateway.api.routes_prompts import router as prompts_router
from llm_gateway.core.circuit_breaker import CircuitBreaker
from llm_gateway.core.concurrency import ConcurrencyGuard
from llm_gateway.core.exceptions import GatewayError
from llm_gateway.core.logging_setup import init_logging
from llm_gateway.core.providers.openai_provider import OpenAICompatibleProvider
from llm_gateway.core.providers.registry import ProviderRegistry
from llm_gateway.core.providers.selector import ModelSelector
from llm_gateway.core.rate_limit import TokenBucketLimiter
from llm_gateway.service.llm_service import LLMService
from llm_gateway.service.ledger_service import LedgerService
from llm_gateway.service.prompt_service import PromptService
from llm_gateway.service.prompt_store import PromptStore
from llm_gateway.service.validation_service import ValidationService
from llm_gateway.settings import Settings, load_settings


def create_app(
    settings: Settings | None = None,
    *,
    registry: ProviderRegistry | None = None,
    breaker: CircuitBreaker | None = None,
    limiter: TokenBucketLimiter | None = None,
    ledger: LedgerService | None = None,
    concurrency_guard: ConcurrencyGuard | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    init_logging(settings.log_level)

    # 供应商注册表：新供应商 = 实现 Provider Protocol + 此处一行注册
    if registry is None:
        registry = ProviderRegistry()
        registry.register(OpenAICompatibleProvider(settings.retry_statuses))
    breaker = breaker or CircuitBreaker(
        failure_threshold=settings.breaker_config.failure_threshold,
        recovery_timeout=settings.breaker_config.recovery_timeout_seconds,
    )
    limiter = limiter or TokenBucketLimiter()
    ledger = ledger or LedgerService(settings.database_path)
    concurrency_guard = concurrency_guard or ConcurrencyGuard()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # 优雅关闭：SIGTERM → uvicorn 停止接收新请求并等待存量完成后释放账本连接
        yield
        await ledger.close()

    app = FastAPI(title=settings.service_name, version="0.5.0", lifespan=lifespan)
    app.add_middleware(RequestContextMiddleware)
    if settings.cors_allow_origins:
        # CORS 仅在显式配置时启用（浏览器端 Agent 场景）
        from starlette.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_allow_origins),
            allow_methods=["*"],
            allow_headers=["*"],
        )
    app.add_exception_handler(GatewayError, gateway_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
    app.include_router(router)
    app.include_router(openai_router)
    app.include_router(prompts_router)

    # Prompt 版本库：yaml 种子幂等导入 + /v1/prompts API 创建的版本，同一数据源
    prompt_store = PromptStore(settings.database_path, seeds=settings.templates)
    prompt_service = PromptService(prompt_store)
    app.state.settings = settings
    app.state.rate_limiter = limiter
    app.state.ledger = ledger
    app.state.concurrency_guard = concurrency_guard
    app.state.circuit_breaker = breaker
    app.state.prompts = prompt_store
    app.state.prompt_service = prompt_service
    app.state.llm_service = LLMService(
        selector=ModelSelector(settings.models, registry),
        prompt_service=prompt_service,
        validation_service=ValidationService(),
        ledger=ledger,
        retry_policy=settings.retry_policy,
        breaker=breaker,
        run_budget_seconds=settings.run_budget_seconds,
        concurrency_guard=concurrency_guard,
        max_concurrency_per_candidate=settings.max_concurrency_per_candidate,
        max_repair_attempts=settings.structured_output_retries,
        checkpoint_config=settings.stream_checkpoint,
    )
    return app


app = create_app()
