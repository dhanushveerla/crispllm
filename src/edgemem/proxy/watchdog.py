from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
import time
from typing import Callable

from edgemem.config import ResourcesConfig
from edgemem.proxy.sysinfo import SystemMemoryInfo, get_system_memory_info

logger = logging.getLogger("edgemem.proxy.watchdog")


@dataclass
class WatchdogSnapshot:
    pressure: str
    memory: SystemMemoryInfo | None
    sampled_at: float


class RamPressureWatchdog:
    def __init__(
        self,
        config: ResourcesConfig,
        probe_fn: Callable[[], SystemMemoryInfo | None] = get_system_memory_info,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.probe_fn = probe_fn
        self.clock = clock
        self._lock = threading.Lock()
        self._pressure: str = "normal"
        self._last_sample_time: float = -1.0
        self._cached_info: SystemMemoryInfo | None = None

    @property
    def current_pressure(self) -> str:
        with self._lock:
            return self._pressure

    def _compute_next_pressure(self, available_mb: float) -> str:
        hard_thresh = self.config.hard_free_mb
        soft_thresh = self.config.soft_free_mb
        hysteresis = self.config.hysteresis_mb

        if available_mb < hard_thresh:
            return "hard"

        curr = self._pressure
        if curr == "hard":
            if available_mb >= soft_thresh + hysteresis:
                return "normal"
            if available_mb >= hard_thresh + hysteresis:
                return "soft"
            return "hard"

        if curr == "soft":
            if available_mb >= soft_thresh + hysteresis:
                return "normal"
            return "soft"

        if available_mb < soft_thresh:
            return "soft"

        return "normal"

    def snapshot(self, force: bool = False) -> WatchdogSnapshot:
        now = self.clock()
        with self._lock:
            if not force and self._last_sample_time >= 0.0 and (now - self._last_sample_time < self.config.watchdog_poll_s):
                return WatchdogSnapshot(
                    pressure=self._pressure,
                    memory=self._cached_info,
                    sampled_at=self._last_sample_time,
                )

            if not self.config.enabled:
                self._pressure = "normal"
                self._cached_info = None
                self._last_sample_time = now
                return WatchdogSnapshot(
                    pressure="normal",
                    memory=None,
                    sampled_at=now,
                )

            try:
                info = self.probe_fn()
            except Exception as exc:
                logger.warning("RamPressureWatchdog probe failed (failing open): %s", exc)
                info = None
            self._cached_info = info
            self._last_sample_time = now

            if info is None:
                self._pressure = "normal"
            else:
                self._pressure = self._compute_next_pressure(info.available_mb)

            return WatchdogSnapshot(
                pressure=self._pressure,
                memory=self._cached_info,
                sampled_at=now,
            )
