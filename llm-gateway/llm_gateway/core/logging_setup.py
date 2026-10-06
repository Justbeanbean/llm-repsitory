"""统一日志底座：JSON 结构化输出 + request_id 上下文贯穿。

用法约定：logger.info("event_name", extra={"layer 内字段": ...})
- message 本身就是事件名（request_completed / retry_scheduled / ...）
- request_id 由 contextvars 自动注入，async 并发安全
"""
from __future__ import annotations

import contextvars
import json
import logging
from datetime import datetime, timezone

# 请求上下文：api 中间件写入 request_id 与 Run/Step 关联；鉴权依赖写入 caller 指纹
# （async 并发安全；每个请求 task 一份拷贝）
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
caller_fingerprint_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "caller_fingerprint", default="-"
)
run_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("run_id", default="-")
step_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("step_id", default="-")

# LogRecord 内建字段（不进入 extras）
_RESERVED = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "layer": record.name,
            "event": record.getMessage(),
            "request_id": request_id_var.get(),
            "caller": caller_fingerprint_var.get(),
            "run_id": run_id_var.get(),
            "step_id": step_id_var.get(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def init_logging(level: str = "INFO") -> None:
    """应用装配时调用一次；对应入口 gateway.py 的启动初始化。"""
    root = logging.getLogger("llm_gateway")
    root.setLevel(level.upper())
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)
    root.propagate = False


def get_logger(layer: str) -> logging.Logger:
    """按层取 logger：get_logger("core") → llm_gateway.core"""
    return logging.getLogger(f"llm_gateway.{layer}")
