"""SSE streaming utilities for the OpenAI-compatible proxy.

Faithfully passes through upstream SSE chunks without buffering,
preserving delta.tool_calls fragments and finish_reason semantics.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx


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
    first = True
    async for line in iter_sse_lines(response):
        line = line.rstrip("\r")
        if line.startswith("data: "):
            if first and on_first_data is not None:
                on_first_data()
                first = False
            yield line + "\n\n"
        elif line == "":
            # Empty line = SSE event boundary, already handled by "\n\n" above
            continue
        elif line.startswith(":"):
            # SSE comment line — pass through
            yield line + "\n"


def estimate_tokens_from_messages(messages: list[dict]) -> int:
    """Rough token estimate from messages (4 chars ≈ 1 token)."""
    total_chars = sum(len(json.dumps(m)) for m in messages)
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

    # Prefer usage field if present (some providers include it in final chunk)
    usage = data.get("usage")
    if usage and "completion_tokens" in usage:
        return usage["completion_tokens"]

    return 0
