"""Prompt 版本库：SQLite 持久化 + yaml 种子导入。

- 版本号 TEXT（"v1"、"v2"…，创建时自动递增到下一个数字版本）
- yaml prompt_templates 启动时幂等导入（相同 (id, version) 不重复）
- "latest" 语义：未指定版本 → 该 id 的最新数字版本
- 读路径同步查询（小表 + SQLite 快，供 PromptService 渲染链路），
  写路径 async（asyncio.to_thread，不阻塞事件循环）
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from llm_gateway.core.models import PromptTemplate


class PromptRecord(BaseModel):
    """Prompt 版本记录（/v1/prompts 的 API 契约）。"""

    model_config = ConfigDict(extra="forbid")

    id: str
    version: str
    role: str = "system"
    template: str
    content_hash: str
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    business_model: str | None = None
    default_model: str | None = None
    default_timeout_seconds: float | None = Field(default=None, gt=0, le=120)
    created_at: datetime


class PromptCreate(BaseModel):
    """创建新版本（版本号自动递增）。"""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=100)
    template: str = Field(min_length=1, max_length=20_000)
    role: str = "system"
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    business_model: str | None = None
    default_model: str | None = None
    default_timeout_seconds: float | None = Field(default=None, gt=0, le=120)


class PromptRender(BaseModel):
    """渲染预览请求（沙箱渲染，走与正式调用完全相同的校验链）。"""

    model_config = ConfigDict(extra="forbid")

    variables: dict[str, str] = Field(default_factory=dict)
    version: str | None = None       # None = 最新版本


_COLUMNS = (
    "id", "version", "role", "template", "content_hash",
    "input_schema", "output_schema", "business_model",
    "default_model", "default_timeout_seconds", "created_at",
)

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS prompts (
    id TEXT NOT NULL,
    version TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'system',
    template TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    input_schema TEXT,
    output_schema TEXT,
    business_model TEXT,
    default_model TEXT,
    default_timeout_seconds REAL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (id, version)
)
"""


def _content_hash(template: str) -> str:
    return hashlib.sha256(template.encode("utf-8")).hexdigest()[:12]


def _dump(schema: dict[str, Any] | None) -> str | None:
    return json.dumps(schema, ensure_ascii=False) if schema is not None else None


class PromptStore:
    def __init__(
        self,
        db_path: str | Path,
        seeds: dict[tuple[str, str], PromptTemplate] | None = None,
    ) -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(_CREATE_SQL)
        if seeds:
            self._seed(seeds)
        self._db.commit()

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _seed(self, seeds: dict[tuple[str, str], PromptTemplate]) -> None:
        """yaml prompt_templates 幂等导入（重启不重复）。"""
        for (name, version), template in seeds.items():
            with self._lock:
                self._db.execute(
                    f"INSERT OR IGNORE INTO prompts ({','.join(_COLUMNS)}) "
                    f"VALUES ({','.join('?' * len(_COLUMNS))})",
                    self._row_of(name, version, template),
                )
        self._db.commit()

    @staticmethod
    def _row_of(name: str, version: str, template: PromptTemplate) -> tuple:
        return (
            name,
            version,
            "system",
            template.system_template,
            _content_hash(template.system_template),
            _dump(template.input_schema),
            _dump(template.output_schema),
            template.business_model,
            template.default_model,
            template.default_timeout_seconds,
            datetime.now(timezone.utc).isoformat(),
        )

    def _next_version(self, prompt_id: str) -> str:
        rows = self._db.execute(
            "SELECT version FROM prompts WHERE id = ?", (prompt_id,)
        ).fetchall()
        numbers = [
            int(v.lstrip("vV")) for (v,) in rows if v.lstrip("vV").isdigit()
        ]
        return f"v{max(numbers, default=0) + 1}"

    @staticmethod
    def _record(row: tuple) -> PromptRecord:
        data = dict(zip(_COLUMNS, row))
        for key in ("input_schema", "output_schema"):
            if data[key]:
                data[key] = json.loads(data[key])
        return PromptRecord(**data)

    @staticmethod
    def _template(row: tuple) -> PromptTemplate:
        data = dict(zip(_COLUMNS, row))
        return PromptTemplate(
            name=data["id"],
            version=data["version"],
            system_template=data["template"],
            input_schema=json.loads(data["input_schema"]) if data["input_schema"] else None,
            output_schema=json.loads(data["output_schema"]) if data["output_schema"] else None,
            business_model=data["business_model"],
            default_model=data["default_model"],
            default_timeout_seconds=data["default_timeout_seconds"],
        )

    # ------------------------------------------------------------------
    # 写路径（async）
    # ------------------------------------------------------------------

    async def create_version(self, payload: PromptCreate) -> PromptRecord:
        def _create() -> PromptRecord:
            with self._lock:
                version = self._next_version(payload.id)
                created_at = datetime.now(timezone.utc).isoformat()
                row = (
                    payload.id, version, payload.role, payload.template,
                    _content_hash(payload.template),
                    _dump(payload.input_schema), _dump(payload.output_schema),
                    payload.business_model, payload.default_model,
                    payload.default_timeout_seconds, created_at,
                )
                self._db.execute(
                    f"INSERT OR REPLACE INTO prompts ({','.join(_COLUMNS)}) "
                    f"VALUES ({','.join('?' * len(_COLUMNS))})",
                    row,
                )
                self._db.commit()
                return PromptRecord(**dict(zip(_COLUMNS, row)))

        return await asyncio.to_thread(_create)

    # ------------------------------------------------------------------
    # 读路径
    # ------------------------------------------------------------------

    async def list(self) -> list[PromptRecord]:
        def _list() -> list[PromptRecord]:
            with self._lock:
                rows = self._db.execute(
                    f"SELECT {','.join(_COLUMNS)} FROM prompts ORDER BY id, created_at"
                ).fetchall()
            return [self._record(row) for row in rows]

        return await asyncio.to_thread(_list)

    async def get(self, prompt_id: str, version: str | None = None) -> PromptRecord | None:
        row = self.get_row(prompt_id, version)
        return self._record(row) if row else None

    def get_row(self, prompt_id: str, version: str | None = None) -> tuple | None:
        """同步读原始行（version=None/"latest" → 最新数字版本）。"""
        with self._lock:
            if version in (None, "latest"):
                return self._db.execute(
                    f"SELECT {','.join(_COLUMNS)} FROM prompts WHERE id = ? "
                    "ORDER BY created_at DESC, rowid DESC LIMIT 1",
                    (prompt_id,),
                ).fetchone()
            return self._db.execute(
                f"SELECT {','.join(_COLUMNS)} FROM prompts WHERE id = ? AND version = ?",
                (prompt_id, version),
            ).fetchone()

    def get_template(self, prompt_id: str, version: str | None = None) -> PromptTemplate | None:
        """供 PromptService 渲染链路（同步，与正式调用走同一数据源）。"""
        row = self.get_row(prompt_id, version)
        return self._template(row) if row else None
