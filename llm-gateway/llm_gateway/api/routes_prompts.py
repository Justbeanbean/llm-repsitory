"""Prompt 管理 API：版本创建 / 列表 / 获取 / 渲染预览。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from llm_gateway.api.deps import require_caller
from llm_gateway.core.exceptions import GatewayError
from llm_gateway.core.models import Message, PromptSelection
from llm_gateway.core.security import Caller
from llm_gateway.service.prompt_store import PromptCreate, PromptRecord, PromptRender

router = APIRouter()


@router.post("/v1/prompts", response_model=PromptRecord, status_code=201)
async def create_prompt(
    payload: PromptCreate,
    request: Request,
    _: Caller = Depends(require_caller),
) -> PromptRecord:
    """创建新版本（版本号自动递增；创建后立即可用于调用）。"""
    return await request.app.state.prompts.create_version(payload)


@router.get("/v1/prompts", response_model=list[PromptRecord])
async def list_prompts(
    request: Request,
    _: Caller = Depends(require_caller),
) -> list[PromptRecord]:
    return await request.app.state.prompts.list()


@router.get("/v1/prompts/{prompt_id}", response_model=PromptRecord)
async def get_prompt(
    prompt_id: str,
    request: Request,
    _: Caller = Depends(require_caller),
    version: str | None = None,
) -> PromptRecord:
    record = await request.app.state.prompts.get(prompt_id, version)
    if record is None:
        raise GatewayError("unknown_prompt_template", "Prompt 模板不存在", 404)
    return record


@router.post("/v1/prompts/{prompt_id}/render")
async def render_prompt(
    prompt_id: str,
    payload: PromptRender,
    request: Request,
    _: Caller = Depends(require_caller),
) -> dict:
    """渲染预览：与正式调用走完全相同的沙箱校验链（缺变量/多变量 400）。"""
    prompt_service = request.app.state.prompt_service
    message: Message = prompt_service.render(
        PromptSelection(
            name=prompt_id,
            version=payload.version or "latest",
            variables=payload.variables,
        )
    )
    template = prompt_service._find_template(prompt_id, payload.version or "latest")
    return {
        "id": prompt_id,
        "version": template.version if template else None,
        "role": message.role,
        "content": message.content,
    }
