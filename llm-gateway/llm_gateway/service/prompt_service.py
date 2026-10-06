"""Prompt Bundle 管理：受控模板库（SQLite 版本库）+ 沙箱渲染 + 请求注入。

- 模板来源：PromptStore（yaml 种子 + /v1/prompts API 创建的版本，同一数据源）
- version="latest"（默认）→ 该 id 的最新版本
- 沙箱渲染边界：
  - 只允许 string.Template 占位符替换（无代码执行、无属性访问）
  - 变量白名单 = 模板声明的占位符：缺变量、多变量都在调用模型前 400 失败
  - input_schema（若声明）对变量做 jsonschema 类型校验（第二道防线）
"""
from __future__ import annotations

from dataclasses import dataclass
from string import Template

from jsonschema import ValidationError as JsonSchemaError
from jsonschema import validate

from llm_gateway.core.exceptions import GatewayError
from llm_gateway.core.models import Message, PromptSelection, PromptTemplate
from llm_gateway.service.prompt_store import PromptStore


@dataclass(frozen=True)
class PromptContext:
    """渲染产物 + Bundle 元数据（供 service 决定 effective schema / timeout / trace）。"""

    messages: list[Message]
    template: PromptTemplate | None = None


class PromptService:
    def __init__(self, store: PromptStore) -> None:
        self._store = store

    def _find_template(self, name: str, version: str) -> PromptTemplate | None:
        return self._store.get_template(name, None if version == "latest" else version)

    def render(self, selection: PromptSelection) -> Message:
        template = self._find_template(selection.name, selection.version)
        if template is None:
            raise GatewayError("unknown_prompt_template", "Prompt 模板不存在", 400)

        required = set(Template(template.system_template).get_identifiers())
        provided = set(selection.variables)
        missing = required - provided
        if missing:
            raise GatewayError(
                "missing_prompt_variable",
                f"缺少 Prompt 变量: {sorted(missing)[0]}",
                400,
            )
        extra = provided - required
        if extra:
            raise GatewayError(
                "unexpected_prompt_variable",
                f"多余的 Prompt 变量: {sorted(extra)[0]}",
                400,
            )
        # input_schema：变量类型校验（Bundle 声明的输入 Schema）
        if template.input_schema is not None:
            self._validate_variables(template, selection.variables)
        content = Template(template.system_template).substitute(selection.variables)
        return Message(role="system", content=content)

    @staticmethod
    def _validate_variables(template: PromptTemplate, variables: dict[str, str]) -> None:
        try:
            validate(instance=variables, schema=template.input_schema)
        except JsonSchemaError as exc:
            raise GatewayError(
                "invalid_prompt_variables",
                f"Prompt 变量不符合输入 Schema: {exc.message[:150]}",
                400,
            ) from exc

    def build_context(
        self,
        messages: list[Message],
        prompt: PromptSelection | None,
    ) -> PromptContext:
        """将模板系统消息统一注入调用上下文，并携带 Bundle 元数据。"""
        if prompt is None:
            return PromptContext(messages=list(messages))
        return PromptContext(
            messages=[self.render(prompt), *messages],
            template=self._find_template(prompt.name, prompt.version),
        )
