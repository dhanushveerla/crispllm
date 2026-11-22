from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable
import urllib.parse

import httpx

from edgemem.config import EdgeMemConfig
from edgemem.proxy.budgeter import detect_local_backend

logger = logging.getLogger("edgemem.proxy.unloader")


def get_ollama_generate_url(upstream_base_url: str) -> str:
    parsed = urllib.parse.urlsplit(upstream_base_url)
    scheme = parsed.scheme or "http"
    netloc = parsed.netloc or "127.0.0.1:11434"
    return f"{scheme}://{netloc}/api/generate"


class IdleUnloadWorker:
    def __init__(
        self,
        config: EdgeMemConfig,
        http_client: httpx.AsyncClient | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._http_client = http_client
        self.clock = clock
        self.last_local_activity_at: float = clock()
        self.last_local_model: str | None = None
        self.is_unloaded: bool = True
        self.unload_count: int = 0
        self._running: bool = False
        self._task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    def record_local_activity(self, model: str) -> None:
        self.last_local_activity_at = self.clock()
        self.last_local_model = model
        self.is_unloaded = False

    async def unload_now(self, model: str | None = None) -> bool:
        backend = detect_local_backend(self.config.upstream_base_url, self.config.resources.local_backend)
        if backend != "ollama":
            return False

        target_model = model or self.last_local_model or self.config.model
        if not target_model:
            return False

        generate_url = get_ollama_generate_url(self.config.upstream_base_url)
        payload = {"model": target_model, "keep_alive": 0}

        try:
            if self._http_client is not None:
                resp = await self._http_client.post(generate_url, json=payload, timeout=5.0)
            else:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.post(generate_url, json=payload)

            if resp.status_code == 200:
                self.is_unloaded = True
                self.unload_count += 1
                return True
            return False
        except Exception as exc:
            logger.debug("Ollama unload call failed (fail-open): %s", exc)
            return False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run_loop(self) -> None:
        while self._running:
            try:
                poll_interval = min(max(self.config.resources.idle_unload_s / 4.0, 1.0), 5.0)
                await asyncio.sleep(poll_interval)
                if not self._running:
                    break

                now = self.clock()
                if (
                    not self.is_unloaded
                    and self.last_local_model is not None
                    and (now - self.last_local_activity_at >= self.config.resources.idle_unload_s)
                ):
                    async with self._lock:
                        if not self.is_unloaded and (self.clock() - self.last_local_activity_at >= self.config.resources.idle_unload_s):
                            await self.unload_now()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug("IdleUnloadWorker loop exception: %s", exc)
                await asyncio.sleep(1.0)
