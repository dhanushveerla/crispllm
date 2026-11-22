"""Per-request telemetry logging to JSONL.

Records: timestamp, model, input/output token estimates,
TTFT (ms), total latency (ms), proxy RSS (MB).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any

import platformdirs

from crispllm.proxy.sysinfo import get_process_rss_mb

logger = logging.getLogger("crispllm.proxy.telemetry")


@dataclass
class RequestTelemetry:
    """Single request telemetry record."""

    timestamp: str
    model: str
    input_tokens: int
    output_tokens: int
    ttft_ms: float
    total_latency_ms: float
    proxy_rss_mb: float
    stream: bool


def get_rss_mb() -> float:
    """Current process RSS in MB."""
    return get_process_rss_mb()


class TelemetryStore:
    """Persistent SQLite telemetry store for request events."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        if db_path is None:
            data_dir = platformdirs.user_data_dir("crispllm")
            self.db_path: Path | str = Path(data_dir) / "memory.db"
        elif isinstance(db_path, str) and db_path == ":memory:":
            self.db_path = ":memory:"
        else:
            self.db_path = Path(db_path)

        self._is_memory = str(self.db_path) == ":memory:"
        self._lock = threading.RLock()

        if not self._is_memory:
            try:
                Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                self.db_path = ":memory:"
                self._is_memory = True

        try:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        except (sqlite3.OperationalError, OSError):
            self.db_path = ":memory:"
            self._is_memory = True
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)

        self._conn.row_factory = sqlite3.Row

        with self._lock:
            if not self._is_memory:
                self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA busy_timeout=5000;")
            self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS request_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                route_tier TEXT NOT NULL,
                route_reason TEXT NOT NULL,
                model TEXT NOT NULL,
                est_tokens_raw INTEGER NOT NULL,
                est_tokens_sent INTEGER NOT NULL,
                tokens_saved INTEGER NOT NULL,
                compacted_turns INTEGER NOT NULL,
                secrets_redacted INTEGER NOT NULL,
                fallback_used INTEGER NOT NULL,
                latency_ms REAL NOT NULL
            );
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_request_events_ts ON request_events(ts);"
        )
        self._conn.commit()

    def record(
        self,
        route_tier: str,
        route_reason: str,
        model: str,
        est_tokens_raw: int,
        est_tokens_sent: int,
        tokens_saved: int,
        compacted_turns: int,
        secrets_redacted: int,
        fallback_used: int | bool,
        latency_ms: float,
        ts: float | None = None,
    ) -> None:
        if ts is None:
            ts = time.time()
        fallback_int = 1 if fallback_used else 0
        try:
            with self._lock:
                self._conn.execute(
                    """
                    INSERT INTO request_events (
                        ts, route_tier, route_reason, model,
                        est_tokens_raw, est_tokens_sent, tokens_saved,
                        compacted_turns, secrets_redacted, fallback_used, latency_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        ts,
                        route_tier,
                        route_reason,
                        model,
                        est_tokens_raw,
                        est_tokens_sent,
                        tokens_saved,
                        compacted_turns,
                        secrets_redacted,
                        fallback_int,
                        latency_ms,
                    ),
                )
                self._conn.commit()
        except Exception as exc:
            logger.warning("Failed to record request telemetry: %s", exc)

    def summary(self, days: int | None = None) -> dict[str, Any]:
        with self._lock:
            params: list[Any] = []
            where_clause = ""
            if days is not None:
                cutoff = time.time() - (days * 86400.0)
                where_clause = "WHERE ts >= ?"
                params.append(cutoff)

            sql = f"""
                SELECT
                    COUNT(*) AS total_requests,
                    SUM(CASE WHEN route_tier = 'local' THEN 1 ELSE 0 END) AS local_requests,
                    SUM(CASE WHEN route_tier = 'cloud' THEN 1 ELSE 0 END) AS cloud_requests,
                    SUM(CASE WHEN fallback_used > 0 THEN 1 ELSE 0 END) AS fallback_escalations,
                    SUM(est_tokens_raw) AS total_tokens_raw,
                    SUM(est_tokens_sent) AS total_tokens_sent,
                    SUM(tokens_saved) AS total_tokens_saved,
                    SUM(compacted_turns) AS total_compacted_turns,
                    SUM(secrets_redacted) AS total_secrets_redacted,
                    AVG(latency_ms) AS mean_latency_ms
                FROM request_events
                {where_clause};
            """
            cur = self._conn.execute(sql, params)
            row = cur.fetchone()
            if not row or not row["total_requests"]:
                return {
                    "total_requests": 0,
                    "local_requests": 0,
                    "cloud_requests": 0,
                    "local_offload_pct": 0.0,
                    "fallback_escalations": 0,
                    "total_tokens_raw": 0,
                    "total_tokens_sent": 0,
                    "total_tokens_saved": 0,
                    "token_reduction_pct": 0.0,
                    "total_compacted_turns": 0,
                    "total_secrets_redacted": 0,
                    "mean_latency_ms": 0.0,
                }

            total_requests = int(row["total_requests"] or 0)
            local_requests = int(row["local_requests"] or 0)
            cloud_requests = int(row["cloud_requests"] or 0)
            fallback_escalations = int(row["fallback_escalations"] or 0)
            total_tokens_raw = int(row["total_tokens_raw"] or 0)
            total_tokens_sent = int(row["total_tokens_sent"] or 0)
            total_tokens_saved = int(row["total_tokens_saved"] or 0)
            total_compacted_turns = int(row["total_compacted_turns"] or 0)
            total_secrets_redacted = int(row["total_secrets_redacted"] or 0)
            mean_lat = float(row["mean_latency_ms"] or 0.0)

            local_offload_pct = (
                round((local_requests / total_requests) * 100.0, 2)
                if total_requests > 0
                else 0.0
            )
            token_reduction_pct = (
                round((total_tokens_saved / total_tokens_raw) * 100.0, 2)
                if total_tokens_raw > 0
                else 0.0
            )
            mean_latency_ms = round(mean_lat, 2)

            return {
                "total_requests": total_requests,
                "local_requests": local_requests,
                "cloud_requests": cloud_requests,
                "local_offload_pct": local_offload_pct,
                "fallback_escalations": fallback_escalations,
                "total_tokens_raw": total_tokens_raw,
                "total_tokens_sent": total_tokens_sent,
                "total_tokens_saved": total_tokens_saved,
                "token_reduction_pct": token_reduction_pct,
                "total_compacted_turns": total_compacted_turns,
                "total_secrets_redacted": total_secrets_redacted,
                "mean_latency_ms": mean_latency_ms,
            }

    def reset(self) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM request_events;")
            count = cur.rowcount
            self._conn.commit()
            return count

    def close(self) -> None:
        with self._lock:
            if hasattr(self, "_conn") and self._conn is not None:
                self._conn.close()
                self._conn = None

    def __enter__(self) -> TelemetryStore:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class TelemetryLogger:
    """Append-only JSONL telemetry logger."""

    def __init__(self, log_dir: Path) -> None:
        self._log_dir = log_dir
        self._log_file = log_dir / "telemetry.jsonl"
        self._records: list[RequestTelemetry] = []

    def _ensure_dir(self) -> None:
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    def log(self, record: RequestTelemetry) -> None:
        """Append a telemetry record to the JSONL log."""
        self._records.append(record)
        self._ensure_dir()
        try:
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(record)) + "\n")
        except OSError:
            pass

    def recent_summary(self, n: int = 10) -> dict:
        """Summarize the last N requests."""
        recent = self._records[-n:] if self._records else []
        if not recent:
            return {"total_requests": 0}
        return {
            "total_requests": len(self._records),
            "recent_count": len(recent),
            "avg_ttft_ms": round(sum(r.ttft_ms for r in recent) / len(recent), 1),
            "avg_latency_ms": round(
                sum(r.total_latency_ms for r in recent) / len(recent), 1
            ),
            "avg_input_tokens": round(
                sum(r.input_tokens for r in recent) / len(recent), 1
            ),
            "avg_output_tokens": round(
                sum(r.output_tokens for r in recent) / len(recent), 1
            ),
        }


class Timer:
    """Simple context-manager timer for latency measurement."""

    def __init__(self) -> None:
        self.start: float = 0
        self.first_token: float | None = None
        self.end: float = 0

    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        return self

    def mark_first_token(self) -> None:
        if self.first_token is None:
            self.first_token = time.perf_counter()

    def __exit__(self, *_: object) -> None:
        self.end = time.perf_counter()

    @property
    def ttft_ms(self) -> float:
        if self.first_token is None:
            return self.total_ms
        return (self.first_token - self.start) * 1000

    @property
    def total_ms(self) -> float:
        return (self.end - self.start) * 1000
