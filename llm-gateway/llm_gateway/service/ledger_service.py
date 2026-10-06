"""SQLite 用量账本 + 观测聚合。

- 调用方身份只保留短指纹；默认不保存 Prompt 与消息正文
- 成本三态：usage 缺失或调用失败 → cost_usd 为 NULL（未知 ≠ 0 美元）
- P50 / P95 Latency、TTFT、错误率与 429 比例按模型 / 调用方 / Prompt 版本聚合
- 进程内 sqlite（WAL）；多副本部署需迁移共享存储（刻意保留的边界）
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_gateway.core.exceptions import FailureClass
from llm_gateway.core.logging_setup import get_logger
from llm_gateway.core.models import CallTrace, CandidateConfig, Usage

logger = get_logger("service.ledger")

_COLUMNS = (
    "request_id",
    "created_at",
    "caller_fingerprint",
    "run_id",
    "step_id",
    "requested_model",
    "final_candidate",
    "provider",
    "upstream_request_id",
    "prompt_name",
    "prompt_version",
    "prompt_hash",
    "schema_hash",
    "price_version",
    "input_tokens",
    "output_tokens",
    "cached_tokens",
    "usage_missing",
    "cost_usd",
    "ttft_ms",
    "latency_ms",
    "network_attempts",
    "repair_attempts",
    "fallback_count",
    "attempts",
    "status",
    "error_code",
    "failure_class",
    "failure_layer",
    "route_decisions",
)

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS call_records (
    request_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    caller_fingerprint TEXT,
    run_id TEXT,
    step_id TEXT,
    requested_model TEXT NOT NULL,
    final_candidate TEXT,
    provider TEXT,
    upstream_request_id TEXT,
    prompt_name TEXT,
    prompt_version TEXT,
    prompt_hash TEXT,
    schema_hash TEXT,
    price_version TEXT,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cached_tokens INTEGER NOT NULL DEFAULT 0,
    usage_missing INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL,
    ttft_ms INTEGER,
    latency_ms INTEGER NOT NULL DEFAULT 0,
    network_attempts INTEGER NOT NULL DEFAULT 0,
    repair_attempts INTEGER NOT NULL DEFAULT 0,
    fallback_count INTEGER NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    error_code TEXT,
    failure_class TEXT,
    failure_layer TEXT,
    route_decisions TEXT
)
"""

_CREATE_CHECKPOINT_SQL = """
CREATE TABLE IF NOT EXISTS stream_checkpoints (
    request_id TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    status TEXT NOT NULL,
    content TEXT,
    updated_at TEXT NOT NULL
)
"""

# 查询索引：traces 过滤与 metrics 时间窗口走索引而非全表扫描
_INDEXES_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_call_records_created_at ON call_records(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_call_records_status ON call_records(status)",
    "CREATE INDEX IF NOT EXISTS idx_call_records_model ON call_records(requested_model)",
)


def calculate_cost(candidate: CandidateConfig, usage: Usage) -> float | None:
    """usage 缺失 → 不计价，返回 None（账本未知）；明确数值才计价。

    cached 部分按 cached_input_per_million 计价（若配置），其余输入按原价。
    """
    if usage.usage_missing:
        return None
    price = candidate.price
    cached = min(usage.cached_tokens, usage.input_tokens)
    billable_input = usage.input_tokens - cached
    return (
        billable_input * price.input_per_million
        + cached * price.cached_input_per_million
        + usage.output_tokens * price.output_per_million
    ) / 1_000_000


def _percentile(values: list[int], p: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((p / 100) * (len(ordered) - 1)))))
    return ordered[index]


class LedgerService:
    """账本读写一律通过 async 接口（asyncio.to_thread）——SQLite 同步 IO 不阻塞事件循环。

    单条写入在请求路径上；若未来写入量增大，可进一步改为内存队列 + 批量落盘。
    """

    def __init__(self, db_path: str | Path = "data/usage.db") -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self._path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(_CREATE_SQL)
        self._db.execute(_CREATE_CHECKPOINT_SQL)
        for index_sql in _INDEXES_SQL:
            self._db.execute(index_sql)
        self._migrate()
        self._db.commit()

    def _migrate(self) -> None:
        """轻量列迁移：旧库自动补齐新增列（ALTER TABLE ADD COLUMN，保留历史数据）。"""
        existing = {row[1] for row in self._db.execute("PRAGMA table_info(call_records)")}
        type_defaults = {
            "TEXT": None,
            "INTEGER": 0,
            "REAL": None,
        }
        for column in _COLUMNS:
            if column in existing:
                continue
            if column in ("input_tokens", "output_tokens", "cached_tokens", "usage_missing",
                          "latency_ms", "network_attempts", "repair_attempts",
                          "fallback_count", "attempts"):
                ddl = "INTEGER NOT NULL DEFAULT 0"
            elif column == "cost_usd":
                ddl = "REAL"
            elif column == "ttft_ms":
                ddl = "INTEGER"
            elif column == "created_at":
                ddl = "TEXT NOT NULL DEFAULT ''"
            elif column in ("request_id", "requested_model", "status"):
                ddl = "TEXT NOT NULL DEFAULT ''"
            else:
                ddl = "TEXT"
            self._db.execute(f"ALTER TABLE call_records ADD COLUMN {column} {ddl}")

    async def record(self, trace: CallTrace) -> None:
        await asyncio.to_thread(self._record_sync, trace)

    def _record_sync(self, trace: CallTrace) -> None:
        values = (
            trace.request_id,
            trace.timestamp.isoformat(),
            trace.caller_fingerprint,
            trace.run_id,
            trace.step_id,
            trace.requested_model,
            trace.final_candidate,
            trace.provider,
            trace.upstream_request_id,
            trace.prompt_name,
            trace.prompt_version,
            trace.prompt_hash,
            trace.schema_hash,
            trace.price_version,
            trace.input_tokens,
            trace.output_tokens,
            trace.cached_tokens,
            int(trace.usage_missing),
            trace.cost_usd,
            trace.ttft_ms,
            trace.latency_ms,
            trace.network_attempts,
            trace.repair_attempts,
            trace.fallback_count,
            trace.attempts,
            trace.status,
            trace.error_code,
            trace.failure_class.value if trace.failure_class else None,
            trace.failure_layer,
            json.dumps(trace.route_decisions, ensure_ascii=False) if trace.route_decisions else None,
        )
        placeholders = ",".join("?" * len(_COLUMNS))
        with self._lock:
            self._db.execute(
                f"INSERT OR REPLACE INTO call_records ({','.join(_COLUMNS)}) VALUES ({placeholders})",
                values,
            )
            self._db.commit()
        logger.info("llm_call_trace", extra=trace.model_dump(mode="json"))

    async def recent(
        self,
        limit: int = 200,
        offset: int = 0,
        *,
        model: str | None = None,
        status: str | None = None,
        caller: str | None = None,
        prompt_name: str | None = None,
        since_minutes: int | None = None,
    ) -> list[CallTrace]:
        return await asyncio.to_thread(
            self._recent_sync, limit, offset, model, status, caller, prompt_name, since_minutes
        )

    def _recent_sync(
        self,
        limit: int,
        offset: int,
        model: str | None,
        status: str | None,
        caller: str | None,
        prompt_name: str | None,
        since_minutes: int | None,
    ) -> list[CallTrace]:
        where, params = self._build_where(model, status, caller, prompt_name, since_minutes)
        with self._lock:
            rows = self._db.execute(
                f"SELECT {','.join(_COLUMNS)} FROM call_records {where} "
                "ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
        traces: list[CallTrace] = []
        for row in rows:
            data = dict(zip(_COLUMNS, row))
            data["timestamp"] = data.pop("created_at")   # DB 列名 → 模型字段名
            data["usage_missing"] = bool(data["usage_missing"])
            if data["failure_class"]:
                data["failure_class"] = FailureClass(data["failure_class"])
            if data["route_decisions"]:
                data["route_decisions"] = json.loads(data["route_decisions"])
            traces.append(CallTrace(**data))
        return traces

    async def close(self) -> None:
        """优雅关闭：应用 lifespan shutdown 时释放数据库连接。"""
        await asyncio.to_thread(self._close_sync)

    def _close_sync(self) -> None:
        with self._lock:
            self._db.close()

    async def healthy(self) -> bool:
        return await asyncio.to_thread(self._healthy_sync)

    def _healthy_sync(self) -> bool:
        try:
            with self._lock:
                self._db.execute("SELECT 1").fetchone()
            return True
        except sqlite3.Error:
            return False

    # ------------------------------------------------------------------
    # 流式检查点（yaml stream_checkpoint.enabled=true 时才写入；默认关闭，
    # 避免在未评估隐私策略前保存模型正文）
    # ------------------------------------------------------------------

    async def save_checkpoint(
        self, *, request_id: str, model: str, status: str, content: str
    ) -> None:
        await asyncio.to_thread(
            self._save_checkpoint_sync,
            request_id=request_id, model=model, status=status, content=content,
        )

    def _save_checkpoint_sync(
        self, *, request_id: str, model: str, status: str, content: str
    ) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO stream_checkpoints "
                "(request_id, model, status, content, updated_at) VALUES (?, ?, ?, ?, ?)",
                (
                    request_id,
                    model,
                    status,
                    content,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self._db.commit()

    async def get_checkpoint(self, request_id: str) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._get_checkpoint_sync, request_id)

    def _get_checkpoint_sync(self, request_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT request_id, model, status, content, updated_at "
                "FROM stream_checkpoints WHERE request_id = ?",
                (request_id,),
            ).fetchone()
        if row is None:
            return None
        return dict(
            zip(("request_id", "model", "status", "content", "updated_at"), row)
        )

    # ------------------------------------------------------------------
    # 观测聚合：P50/P95 TTFT、Latency、错误率、429 比例
    # ------------------------------------------------------------------

    @staticmethod
    def _build_where(
        model: str | None,
        status: str | None,
        caller: str | None,
        prompt_name: str | None,
        since_minutes: int | None,
    ) -> tuple[str, list]:
        clauses, params = [], []
        if model:
            clauses.append("requested_model = ?")
            params.append(model)
        if status:
            clauses.append("status = ?")
            params.append(status)
        if caller:
            clauses.append("caller_fingerprint = ?")
            params.append(caller)
        if prompt_name:
            clauses.append("prompt_name = ?")
            params.append(prompt_name)
        if since_minutes is not None:
            from datetime import timedelta

            cutoff = (datetime.now(timezone.utc) - timedelta(minutes=since_minutes)).isoformat()
            clauses.append("created_at >= ?")
            params.append(cutoff)
        return ("WHERE " + " AND ".join(clauses)) if clauses else "", params

    async def metrics(self, limit: int = 10_000, *, window_minutes: int | None = None) -> dict[str, Any]:
        return await asyncio.to_thread(self._metrics_sync, limit, window_minutes)

    def _metrics_sync(self, limit: int, window_minutes: int | None) -> dict[str, Any]:
        where, params = self._build_where(None, None, None, None, window_minutes)
        with self._lock:
            rows = self._db.execute(
                "SELECT requested_model, caller_fingerprint, prompt_name, prompt_version, "
                "status, error_code, latency_ms, ttft_ms, input_tokens, output_tokens, cost_usd "
                f"FROM call_records {where} ORDER BY created_at DESC LIMIT ?",
                (*params, limit),
            ).fetchall()

        total = len(rows)
        if total == 0:
            return {"total": 0}

        latencies = [r[6] for r in rows if r[6] is not None]
        ttfts = [r[7] for r in rows if r[7] is not None]
        failed = sum(1 for r in rows if r[4] == "failed")
        cancelled = sum(1 for r in rows if r[4] == "cancelled")
        rate_limited = sum(1 for r in rows if r[5] == "rate_limited")

        def _group(key_index: int) -> dict[str, Any]:
            groups: dict[str, list] = {}
            for r in rows:
                groups.setdefault(r[key_index] or "-", []).append(r)
            out: dict[str, Any] = {}
            for key, group in groups.items():
                g_lat = [r[6] for r in group if r[6] is not None]
                g_ttft = [r[7] for r in group if r[7] is not None]
                out[key] = {
                    "calls": len(group),
                    "error_rate": round(
                        sum(1 for r in group if r[4] == "failed") / len(group), 4
                    ),
                    "latency_ms": {"p50": _percentile(g_lat, 50), "p95": _percentile(g_lat, 95)},
                    "ttft_ms": {"p50": _percentile(g_ttft, 50), "p95": _percentile(g_ttft, 95)},
                    "cost_usd_total": round(
                        sum(r[10] for r in group if r[10] is not None), 6
                    ),
                }
            return out

        return {
            "total": total,
            "success": total - failed - cancelled,
            "failed": failed,
            "cancelled": cancelled,
            "error_rate": round(failed / total, 4),
            "rate_limited_ratio": round(rate_limited / total, 4),
            "latency_ms": {"p50": _percentile(latencies, 50), "p95": _percentile(latencies, 95)},
            "ttft_ms": {"p50": _percentile(ttfts, 50), "p95": _percentile(ttfts, 95)},
            "tokens": {
                "input": sum(r[8] for r in rows),
                "output": sum(r[9] for r in rows),
            },
            "cost_usd_total": round(sum(r[10] for r in rows if r[10] is not None), 6),
            "by_model": _group(0),
            "by_caller": _group(1),
            "by_prompt": _prompt_groups(rows),
        }


def _prompt_groups(rows: list) -> dict[str, Any]:
    groups: dict[str, list] = {}
    for r in rows:
        key = f"{r[2] or '-'}@{r[3] or '-'}"
        groups.setdefault(key, []).append(r)
    out: dict[str, Any] = {}
    for key, group in groups.items():
        out[key] = {
            "calls": len(group),
            "tokens": {
                "input": sum(r[8] for r in group),
                "output": sum(r[9] for r in group),
            },
            "cost_usd_total": round(sum(r[10] for r in group if r[10] is not None), 6),
        }
    return out
