"""SSE streaming utilities for the OpenAI-compatible proxy.

Faithfully passes through upstream SSE chunks without buffering,
preserving delta.tool_calls fragments and finish_reason semantics.

Includes a keepalive heartbeat mechanism for slow-TTFT upstreams
(e.g. Ollama on T1 CPU during long prompt evaluations) so downstream
clients don't drop the idle SSE connection.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

logger = logging.getLogger("crispllm.proxy.streaming")

HEARTBEAT_INTERVAL_S: float = 10.0


def format_openai_error(err_text: str, status_code: int) -> dict[str, Any]:
    """Parse raw upstream error response into an OpenAI-compatible error dict."""
    try:
        parsed = json.loads(err_text)
        if isinstance(parsed, dict) and "error" in parsed:
            err_val = parsed["error"]
            error_obj = dict(err_val) if isinstance(err_val, dict) else {"message": str(err_val)}
        elif (
            isinstance(parsed, list)
            and len(parsed) > 0
            and isinstance(parsed[0], dict)
            and "error" in parsed[0]
        ):
            err_val = parsed[0]["error"]
            error_obj = dict(err_val) if isinstance(err_val, dict) else {"message": str(err_val)}
        elif isinstance(parsed, dict):
            error_obj = dict(parsed)
        else:
            error_obj = {"message": str(parsed)}
    except Exception:
        error_obj = {
            "message": err_text.strip() or f"Upstream error {status_code}",
        }

    if "message" not in error_obj:
        error_obj["message"] = err_text.strip() or f"Upstream error {status_code}"
    if "code" not in error_obj:
        error_obj["code"] = status_code
    if "type" not in error_obj:
        error_obj["type"] = "upstream_error"

    return error_obj


async def iter_sse_lines(response: httpx.Response) -> AsyncIterator[str]:
    """Yield individual SSE lines from an httpx streaming response."""
    buffer = ""
    async for chunk in response.aiter_text():
        buffer += chunk
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            yield line
    if buffer:
        yield buffer


async def stream_sse_passthrough(
    response: httpx.Response,
    on_first_data: Any = None,
) -> AsyncIterator[str]:
    """Stream SSE events from upstream, yielding them faithfully.

    Yields raw SSE-formatted strings: "data: {...}\\n\\n" and final "data: [DONE]\\n\\n".
    Calls on_first_data() (if provided) when the first data chunk arrives.

    Tool-call deltas are passed through byte-identical — no reassembly or mutation.
    """
    if response.status_code >= 400:
        err_bytes = await response.aread()
        err_text = err_bytes.decode("utf-8", errors="replace")
        logger.warning(
            "Upstream stream returned error status=%d: %s",
            response.status_code,
            err_text[:500],
        )
        if on_first_data is not None:
            on_first_data()
        error_obj = format_openai_error(err_text, response.status_code)
        yield f"data: {json.dumps({'error': error_obj})}\n\n"
        return

    first = True
    async for line in iter_sse_lines(response):
        line = line.rstrip("\r")
        if line.startswith("data: "):
            if first and on_first_data is not None:
                on_first_data()
                first = False
            yield line + "\n\n"
        elif line == "":
            continue
        elif line.startswith(":"):
            yield line + "\n"


async def stream_with_heartbeat(
    client: httpx.AsyncClient,
    body: dict[str, Any],
    headers: dict[str, str],
    on_first_data: Any = None,
    heartbeat_interval: float | None = None,
    url: str = "/chat/completions",
    timeout: float | httpx.Timeout | None = None,
) -> AsyncIterator[str]:
    """Open an upstream SSE stream with keepalive heartbeats during TTFT wait.

    Sends `: keepalive\\n\\n` SSE comments every *heartbeat_interval* seconds
    while waiting for the upstream to send its first data, keeping downstream
    HTTP clients (e.g. OpenCode's Bun fetch()) from dropping idle sockets.

    The heartbeat runs concurrently: it covers both the time waiting for the
    upstream TCP+TLS handshake / response headers AND the time waiting for
    the first SSE data line (prompt evaluation on slow CPUs).

    Once the first upstream data line arrives, heartbeats stop and SSE chunks
    are passed through faithfully.
    """
    if heartbeat_interval is None:
        heartbeat_interval = HEARTBEAT_INTERVAL_S

    queue: asyncio.Queue[str | None | BaseException] = asyncio.Queue()
    first_data_event = asyncio.Event()

    def _on_first_data() -> None:
        first_data_event.set()
        if on_first_data is not None:
            on_first_data()

    async def _upstream_reader() -> None:
        """Open the upstream stream and push SSE lines into the queue."""
        try:
            async with client.stream(
                "POST", url, json=body, headers=headers, timeout=timeout
            ) as resp:
                async for sse_line in stream_sse_passthrough(
                    resp, on_first_data=_on_first_data
                ):
                    await queue.put(sse_line)
        except Exception as exc:
            logger.exception("Upstream streaming error")
            await queue.put(exc)
        finally:
            await queue.put(None)

    async def _heartbeat_sender() -> None:
        """Push keepalive comments into the queue until first data arrives."""
        while not first_data_event.is_set():
            try:
                await asyncio.wait_for(
                    first_data_event.wait(),
                    timeout=heartbeat_interval,
                )
                return
            except asyncio.TimeoutError:
                await queue.put(": keepalive\n\n")

    upstream_task = asyncio.ensure_future(_upstream_reader())
    heartbeat_task = asyncio.ensure_future(_heartbeat_sender())

    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        heartbeat_task.cancel()
        if not upstream_task.done():
            upstream_task.cancel()
        try:
            await heartbeat_task
        except (asyncio.CancelledError, Exception):
            pass
        try:
            await upstream_task
        except (asyncio.CancelledError, Exception):
            pass


def estimate_tokens_from_messages(messages: list[dict]) -> int:
    """Rough token estimate from messages (4 chars ≈ 1 token)."""
    if not isinstance(messages, list) or not messages:
        return 0
    total_chars = 0
    for m in messages:
        try:
            if isinstance(m, dict):
                content = m.get("content")
                if isinstance(content, str):
                    total_chars += len(content)
                else:
                    total_chars += len(json.dumps(m, default=str))
            else:
                total_chars += len(str(m))
        except Exception:
            total_chars += 10
    return max(1, total_chars // 4)


def count_output_tokens_from_sse(sse_line: str) -> int:
    """Extract output token count from a single SSE data line.

    Checks both usage.completion_tokens and delta content length.
    """
    if not sse_line.startswith("data: "):
        return 0
    payload = sse_line[6:].strip()
    if payload == "[DONE]":
        return 0
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return 0

    usage = data.get("usage")
    if usage and "completion_tokens" in usage:
        return usage["completion_tokens"]

    return 0
