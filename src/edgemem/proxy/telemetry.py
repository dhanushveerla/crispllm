"""Per-request telemetry logging to JSONL.

Records: timestamp, model, input/output token estimates,
TTFT (ms), total latency (ms), proxy RSS (MB).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil


@dataclass
class RequestTelemetry:
    """Single request telemetry record."""

    timestamp: str  # ISO 8601
    model: str
    input_tokens: int
    output_tokens: int
    ttft_ms: float
    total_latency_ms: float
    proxy_rss_mb: float
    stream: bool


def get_rss_mb() -> float:
    """Current process RSS in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


class TelemetryLogger:
    """Append-only JSONL telemetry logger."""

    def __init__(self, log_dir: Path) -> None:
        self._log_dir = log_dir
        self._log_file = log_dir / "telemetry.jsonl"
        self._records: list[RequestTelemetry] = []

    def _ensure_dir(self) -> None:
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def log(self, record: RequestTelemetry) -> None:
        """Append a telemetry record to the JSONL log."""
        self._records.append(record)
        self._ensure_dir()
        with open(self._log_file, "a") as f:
            f.write(json.dumps(asdict(record)) + "\n")

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
