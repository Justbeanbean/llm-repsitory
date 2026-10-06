"""L4 校验：JSON 解析 + jsonschema + 本地业务规则（pydantic 二次校验）。

仅当请求带 response_schema（或 Prompt Bundle 绑定 output_schema）时进入本层；
纯文本请求完全跳过。返回 Verdict 而不抛异常——是否反喂重修由 llm_service 编排。

业务规则层：JSON Schema 只能约束结构，约束不了语义
（如 steps 为空数组、completion_criteria 缺失）。JSON 合法但业务不合法
的结果不得进入 Agent Loop —— business_validation_failed。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from jsonschema import ValidationError as JsonSchemaError
from jsonschema import validate
from pydantic import BaseModel, Field, TypeAdapter


class TaskPlan(BaseModel):
    """示例业务模型：结构合法 ≠ 业务合法，语义约束在此层。"""

    goal: str = Field(min_length=1)
    steps: list[str] = Field(min_length=1, max_length=12)
    completion_criteria: list[str] = Field(min_length=1)
    requires_tools: bool


# 本地业务规则注册表：Prompt Bundle 通过 business_model 名引用
BUSINESS_MODELS: dict[str, type[BaseModel]] = {
    "task_plan": TaskPlan,
}


@dataclass(frozen=True)
class ValidationVerdict:
    ok: bool
    error_code: str | None = None    # invalid_json / schema_validation_failed / business_validation_failed
    errors: str | None = None        # 反喂给模型的问题描述（不含原始内容全文）


class ValidationService:
    def check(
        self,
        content: str,
        schema: dict[str, Any],
        *,
        business_model: str | None = None,
    ) -> ValidationVerdict:
        # 1) JSON 合法性
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            return ValidationVerdict(False, "invalid_json", f"输出不是合法 JSON: {exc.msg}")

        # 2) JSON Schema（结构约束，等价于供应商侧的 Structured Output 约束）
        try:
            validate(instance=parsed, schema=schema)
        except JsonSchemaError as exc:
            return ValidationVerdict(False, "schema_validation_failed", f"输出不符合 schema: {exc.message}")

        # 3) 本地业务规则（pydantic）：JSON 合法但业务不合法 → 不放行
        if business_model is not None:
            model_cls = BUSINESS_MODELS.get(business_model)
            if model_cls is not None:
                try:
                    TypeAdapter(model_cls).validate_python(parsed)
                except Exception as exc:  # pydantic ValidationError
                    return ValidationVerdict(
                        False,
                        "business_validation_failed",
                        f"输出不符合业务规则({business_model}): {str(exc)[:200]}",
                    )
        return ValidationVerdict(True)
