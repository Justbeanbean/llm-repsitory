"""OpenAI 兼容路由：/v1/chat/completions、/v1/responses。

- 普通/SSE Streaming 都走同一条网关编排链路（service 层）
- 流式在首块前可重试/fallback；首块后只发流内错误事件，不换模型拼接
"""
from __future__ import annotations

import time
from collections.abc import AsyncIterator
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from llm_gateway.api.deps import require_caller
from llm_gateway.api.openai_schemas import ChatCompletionRequest, ResponsesRequest
from llm_gateway.api.routes import encode_sse
from llm_gateway.core.exceptions import GatewayError
from llm_gateway.core.logging_setup import request_id_var
from llm_gateway.core.models import LLMRequest, LLMResponse
from llm_gateway.core.security import Caller

router = APIRouter()

# 流式响应头：禁用代理/CDN 缓冲与改写，保证真实逐块到达客户端（真实 TTFT）
# X-Request-ID 由中间件统一注入，此处不重复
SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",
}


def _prevalidate_stream(service, internal: LLMRequest, *, responses_api: bool = False) -> None:
    """流开始前把 400 类错误拦成 JSON 响应，避免发半截 SSE。"""
    try:
        service.validate_stream_request(internal, responses_api=responses_api)
    except GatewayError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc


# ----------------------------------------------------------------------
# POST /v1/chat/completions
# ----------------------------------------------------------------------

def chat_completion_payload(response: LLMResponse) -> dict:
    return {
        "id": f"chatcmpl-{response.request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": response.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response.content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        },
    }


def chat_chunk_payload(
    completion_id: str,
    model: str,
    delta_content: str | None,
    *,
    role: str | None = None,
    finish_reason: str | None = None,
) -> dict:
    delta: dict = {}
    if role is not None:
        delta["role"] = role
    if delta_content is not None:
        delta["content"] = delta_content
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


@router.post("/v1/chat/completions")
async def create_chat_completion(
    payload: ChatCompletionRequest,
    request: Request,
    _: Caller = Depends(require_caller),
):
    internal = payload.to_llm_request()
    service = request.app.state.llm_service

    if not internal.stream:
        response = await service.complete(internal)
        return chat_completion_payload(response)

    _prevalidate_stream(service, internal)
    completion_id = f"chatcmpl-{request_id_var.get() or uuid4().hex}"

    async def sse() -> AsyncIterator[str]:
        first = True
        async for event in service.stream(internal):
            kind = event["type"]
            if kind == "content.delta":
                chunk = chat_chunk_payload(
                    completion_id, internal.model, event["delta"],
                    role="assistant" if first else None,
                )
                first = False
                yield encode_sse(chunk)
            elif kind == "response.completed":
                yield encode_sse(
                    chat_chunk_payload(completion_id, internal.model, None, finish_reason="stop")
                )
                yield "data: [DONE]\n\n"
            else:  # response.failed：首块后失败，不换模型，只发错误块
                yield encode_sse(
                    {"error": {"code": event["error"], "message": "上游流式失败"}}
                )
                yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream", headers=SSE_HEADERS)


# ----------------------------------------------------------------------
# POST /v1/responses
# ----------------------------------------------------------------------

def responses_payload(response: LLMResponse) -> dict:
    return {
        "id": f"resp-{response.request_id}",
        "object": "response",
        "created_at": int(time.time()),
        "model": response.model,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": response.content, "annotations": []}
                ],
            }
        ],
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        },
    }


@router.post("/v1/responses")
async def create_response(
    payload: ResponsesRequest,
    request: Request,
    _: Caller = Depends(require_caller),
):
    internal = payload.to_llm_request()
    service = request.app.state.llm_service

    if not internal.stream:
        # /v1/responses 协议：候选必须支持 api=both（内部路由提示，调用方不可伪造）
        response = await service.complete(internal, responses_api=True)
        return responses_payload(response)

    _prevalidate_stream(service, internal, responses_api=True)
    response_id = f"resp-{request_id_var.get() or uuid4().hex}"

    async def sse() -> AsyncIterator[str]:
        collected: list[str] = []
        yield encode_sse(
            {
                "type": "response.created",
                "response": {"id": response_id, "model": internal.model, "status": "in_progress"},
            }
        )
        async for event in service.stream(internal, responses_api=True):
            kind = event["type"]
            if kind == "content.delta":
                collected.append(event["delta"])
                yield encode_sse({"type": "response.output_text.delta", "delta": event["delta"]})
            elif kind == "response.completed":
                yield encode_sse(
                    {
                        "type": "response.completed",
                        "response": {
                            "id": response_id,
                            "model": internal.model,
                            "status": "completed",
                            "output": [
                                {
                                    "type": "message",
                                    "role": "assistant",
                                    "status": "completed",
                                    "content": [
                                        {
                                            "type": "output_text",
                                            "text": "".join(collected),
                                            "annotations": [],
                                        }
                                    ],
                                }
                            ],
                        },
                    }
                )
            else:  # response.failed
                yield encode_sse(
                    {
                        "type": "response.failed",
                        "response": {
                            "id": response_id,
                            "status": "failed",
                            "error": {"code": event["error"], "message": "上游流式失败"},
                        },
                    }
                )

    return StreamingResponse(sse(), media_type="text/event-stream", headers=SSE_HEADERS)
