"""HTTP 路由：内部协议 /v1/llm*、审计 /v1/traces、/v1/models、管理 /admin/*、探活 /healthz /readyz。

业务端点统一走 require_caller 依赖（鉴权 + 令牌桶限流）；
/healthz 与 /readyz 保持公开（容器探活）。
"""
from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from llm_gateway.api.deps import require_caller
from llm_gateway.api.schemas import CallTrace, LLMRequest, LLMResponse
from llm_gateway.core.exceptions import GatewayError
from llm_gateway.core.security import Caller

router = APIRouter()


def encode_sse(event: dict) -> str:
    """将统一事件编码为浏览器和 Agent 都可消费的 SSE 格式。"""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post("/v1/llm", response_model=LLMResponse)
async def create_llm_response(
    payload: LLMRequest,
    request: Request,
    _: Caller = Depends(require_caller),
) -> LLMResponse:
    if payload.stream:
        raise HTTPException(
            status_code=400,
            detail={"code": "use_stream_endpoint", "message": "流式请求请使用 /v1/llm/stream"},
        )
    return await request.app.state.llm_service.complete(payload)


@router.post("/v1/llm/stream")
async def create_stream(
    payload: LLMRequest,
    request: Request,
    _: Caller = Depends(require_caller),
) -> StreamingResponse:
    if payload.response_schema is not None:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "unsupported_combination",
                "message": "流式输出不支持 response_schema",
            },
        )
    service = request.app.state.llm_service
    try:
        # 流开始前拦截 400 类错误（模型未知 / 模板缺失 / 变量缺失）
        service.validate_stream_request(payload)
    except GatewayError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc

    async def event_stream() -> AsyncIterator[str]:
        async for event in service.stream(payload):
            yield encode_sse(event)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/v1/traces", response_model=list[CallTrace])
async def list_traces(
    request: Request,
    _: Caller = Depends(require_caller),
    limit: int = 50,
    offset: int = 0,
    model: str | None = None,
    status: str | None = None,
    caller: str | None = None,
    prompt_name: str | None = None,
    since_minutes: int | None = None,
) -> list[CallTrace]:
    """调用审计记录（SQLite 账本）：分页 + 按模型/状态/调用方/Prompt/时间过滤。"""
    return await request.app.state.ledger.recent(
        min(limit, 500), max(offset, 0),
        model=model, status=status, caller=caller,
        prompt_name=prompt_name, since_minutes=since_minutes,
    )


@router.get("/v1/metrics")
async def metrics(
    request: Request,
    _: Caller = Depends(require_caller),
    window_minutes: int | None = None,
) -> dict:
    """观测聚合：P50/P95 TTFT 与 Latency、错误率、429 比例，按模型/调用方/Prompt 版本分组。

    window_minutes：只统计时间窗口内的调用（如 60 = 最近一小时）。
    """
    return await request.app.state.ledger.metrics(window_minutes=window_minutes)


@router.get("/v1/models")
async def list_models(
    request: Request,
    _: Caller = Depends(require_caller),
) -> dict:
    """公开模型别名列表（调用方唯一可见的模型名；需鉴权，带 created 时间戳）。"""
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": alias, "object": "model", "created": now, "owned_by": "llm-gateway"}
            for alias in sorted(request.app.state.settings.models)
        ],
    }


@router.get("/admin/usage")
async def admin_usage(
    request: Request,
    _: Caller = Depends(require_caller),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    model: str | None = None,
    status: str | None = None,
    since_minutes: int | None = None,
) -> dict:
    """用量记录（等价 /v1/traces，返回 {"data": [...]}）。"""
    data = await request.app.state.ledger.recent(
        limit, offset, model=model, status=status, since_minutes=since_minutes
    )
    return {"data": data}


@router.get("/admin/routes")
async def admin_routes(
    request: Request,
    _: Caller = Depends(require_caller),
) -> dict:
    """路由配置 + 熔断状态（管理观测）。绝不包含 api_key。"""
    settings = request.app.state.settings
    models_view = {
        alias: {
            "strategy": route.strategy,
            "candidates": [
                {
                    "alias": c.alias,
                    "provider": c.provider,
                    "adapter": c.adapter,
                    "provider_model": c.provider_model,
                    "api": c.api,
                    "weight": c.weight,
                    "supports_structured_output": c.supports_structured_output,
                    "structured_output_mode": c.structured_output_mode,
                    "stream_usage": c.stream_usage,
                    "timeout_seconds": c.timeout_seconds,
                    "price": {
                        "input_per_million": c.price.input_per_million,
                        "output_per_million": c.price.output_per_million,
                        "cached_input_per_million": c.price.cached_input_per_million,
                    },
                    "price_version": c.price_version,
                }
                for c in route.candidates
            ],
        }
        for alias, route in settings.models.items()
    }
    return {
        "models": models_view,
        "circuits": request.app.state.circuit_breaker.status(),
    }


@router.get("/v1/streams/{request_id}/checkpoint")
@router.get("/v1/stream-checkpoints/{request_id}")
async def get_stream_checkpoint(
    request_id: str,
    request: Request,
    _: Caller = Depends(require_caller),
) -> dict:
    """流式检查点查询（仅 stream_checkpoint.enabled=true 时有数据）。"""
    checkpoint = await request.app.state.ledger.get_checkpoint(request_id)
    if checkpoint is None:
        raise GatewayError("checkpoint_not_found", "流式检查点不存在", 404)
    return checkpoint


@router.get("/healthz", include_in_schema=False)
async def healthz() -> JSONResponse:
    """存活探针（liveness）：进程活着即 200，不检查依赖（避免依赖抖动触发重启风暴）。"""
    return JSONResponse({"status": "ok"})


@router.get("/readyz", include_in_schema=False)
async def readyz(request: Request) -> JSONResponse:
    """就绪探针（readiness）：账本数据库可用才 200（K8s 据此摘除流量）。"""
    ok = await request.app.state.ledger.healthy()
    return JSONResponse(
        status_code=200 if ok else 503,
        content={
            "status": "ready" if ok else "not_ready",
            "service": request.app.state.settings.service_name,
        },
    )
