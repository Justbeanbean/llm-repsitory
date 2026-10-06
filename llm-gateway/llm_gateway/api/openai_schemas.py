"""OpenAI 兼容协议契约：/v1/chat/completions 与 /v1/responses 的请求模型。

extra="forbid"：未支持的字段在模型调用前明确失败（HTTP 422），静默忽略是事故源。
Gateway 扩展字段 prompt：受控模板注入（沙箱渲染，见 prompt_service）。
"""
from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from llm_gateway.core.exceptions import GatewayError
from llm_gateway.core.models import LLMRequest, Message, PromptSelection


class JsonSchemaRef(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str = "agent_response"
    schema_: dict[str, Any] = Field(alias="schema")
    strict: bool = True


class ResponseFormat(BaseModel):
    """OpenAI response_format：json_schema 触发 L4 本地校验+修复；json_object 直通。"""

    model_config = ConfigDict(extra="forbid")

    type: Literal["json_schema", "json_object"]
    json_schema: JsonSchemaRef | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1, max_length=100)
    messages: list[Message] = Field(min_length=1, max_length=100)
    stream: bool = False
    response_format: ResponseFormat | None = None
    prompt: PromptSelection | None = None   # Gateway 扩展：受控模板注入

    def to_llm_request(self) -> LLMRequest:
        response_schema: dict[str, Any] | None = None
        if self.response_format is not None and self.response_format.type == "json_schema":
            if self.response_format.json_schema is None:
                raise GatewayError(
                    "invalid_response_format", "response_format.json_schema 缺少 schema", 400
                )
            response_schema = self.response_format.json_schema.schema_
        return LLMRequest(
            model=self.model,
            messages=self.messages,
            stream=self.stream,
            response_schema=response_schema,
            prompt=self.prompt,
        )


class TextFormat(BaseModel):
    """OpenAI Responses API text.format。"""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    type: Literal["text", "json_schema"]
    name: str | None = None
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")
    strict: bool = True


class TextConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: TextFormat | None = None


class ResponsesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1, max_length=100)
    input: Union[str, list[Message]] = Field(min_length=1)
    stream: bool = False
    text: TextConfig | None = None
    prompt: PromptSelection | None = None   # Gateway 扩展：受控模板注入

    def to_llm_request(self) -> LLMRequest:
        if isinstance(self.input, str):
            messages = [Message(role="user", content=self.input)]
        else:
            messages = list(self.input)
        response_schema: dict[str, Any] | None = None
        if self.text is not None and self.text.format is not None:
            fmt = self.text.format
            if fmt.type == "json_schema":
                if fmt.schema_ is None:
                    raise GatewayError(
                        "invalid_text_format", "text.format.json_schema 缺少 schema", 400
                    )
                response_schema = fmt.schema_
        return LLMRequest(
            model=self.model,
            messages=messages,
            stream=self.stream,
            response_schema=response_schema,
            prompt=self.prompt,
        )
