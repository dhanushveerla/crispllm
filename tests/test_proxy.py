"""Tests for EdgeMem proxy — no live upstream required.

Uses httpx mock transport to simulate an upstream OpenAI-compatible server,
verifying:
  a) Non-streaming chat completion passthrough
  b) Streaming SSE text deltas
  c) Streaming SSE multi-chunk tool_calls delta preservation
  d) MCP Streamable HTTP at POST /mcp
  e) extra_body config injection into upstream payloads
  f) Slow-TTFT keepalive heartbeats
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from starlette.testclient import TestClient

from edgemem.config import EdgeMemConfig, MemoryConfig
from edgemem.proxy.app import create_app

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_config(tmp_path, **overrides) -> EdgeMemConfig:
    """Create a test config pointing at a placeholder upstream."""
    defaults = dict(
        host="127.0.0.1",
        port=8080,
        upstream_base_url="http://mock-upstream",
        model="test-model",
        telemetry_dir=str(tmp_path / "telemetry"),
        memory=MemoryConfig(db_path=str(tmp_path / "memory.db")),
    )
    defaults.update(overrides)
    return EdgeMemConfig(**defaults)


# ── Mock upstream transport ───────────────────────────────────────────────────

# Non-streaming response
NON_STREAM_RESPONSE: dict[str, Any] = {
    "id": "chatcmpl-test-1",
    "object": "chat.completion",
    "model": "test-model",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello from mock upstream!",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}

# SSE text streaming chunks
TEXT_STREAM_CHUNKS = [
    {
        "id": "chatcmpl-stream-1",
        "object": "chat.completion.chunk",
        "model": "test-model",
        "choices": [
            {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}
        ],
    },
    {
        "id": "chatcmpl-stream-1",
        "object": "chat.completion.chunk",
        "model": "test-model",
        "choices": [
            {"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}
        ],
    },
    {
        "id": "chatcmpl-stream-1",
        "object": "chat.completion.chunk",
        "model": "test-model",
        "choices": [
            {"index": 0, "delta": {"content": " world"}, "finish_reason": None}
        ],
    },
    {
        "id": "chatcmpl-stream-1",
        "object": "chat.completion.chunk",
        "model": "test-model",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    },
]

# SSE tool_calls streaming chunks — multi-chunk id, name, incremental arguments
TOOL_CALL_STREAM_CHUNKS = [
    {
        "id": "chatcmpl-tool-1",
        "object": "chat.completion.chunk",
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_abc123",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": ""},
                        }
                    ],
                },
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-tool-1",
        "object": "chat.completion.chunk",
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"arguments": '{"lo'},
                        }
                    ]
                },
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-tool-1",
        "object": "chat.completion.chunk",
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"arguments": 'cation":'},
                        }
                    ]
                },
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-tool-1",
        "object": "chat.completion.chunk",
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"arguments": ' "San Fra'},
                        }
                    ]
                },
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-tool-1",
        "object": "chat.completion.chunk",
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"arguments": 'ncisco"}'},
                        }
                    ]
                },
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-tool-1",
        "object": "chat.completion.chunk",
        "model": "test-model",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
    },
]


def _build_sse_body(chunks: list[dict]) -> str:
    """Build raw SSE body text from a list of chunk dicts."""
    lines = []
    for chunk in chunks:
        lines.append(f"data: {json.dumps(chunk)}\n\n")
    lines.append("data: [DONE]\n\n")
    return "".join(lines)


class MockTransport(httpx.AsyncBaseTransport):
    """Mock transport that returns canned responses based on request body.

    Records all request bodies for assertion in tests.
    """

    def __init__(self) -> None:
        self.recorded_bodies: list[dict] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.recorded_bodies.append(body)
        is_stream = body.get("stream", False)

        # Check if the messages mention "tool" to route to tool_calls response
        messages = body.get("messages", [])
        wants_tools = any("tool" in m.get("content", "").lower() for m in messages)

        if not is_stream:
            return httpx.Response(
                200,
                json=NON_STREAM_RESPONSE,
                headers={"content-type": "application/json"},
            )

        if wants_tools:
            sse_body = _build_sse_body(TOOL_CALL_STREAM_CHUNKS)
        else:
            sse_body = _build_sse_body(TEXT_STREAM_CHUNKS)

        return httpx.Response(
            200,
            content=sse_body.encode(),
            headers={"content-type": "text/event-stream"},
        )


class SlowTTFTTransport(httpx.AsyncBaseTransport):
    """Mock transport that delays the response to simulate slow prompt eval.

    Waits `delay_seconds` before returning SSE data, exercising the
    keepalive heartbeat logic.
    """

    def __init__(self, delay_seconds: float = 0.0) -> None:
        self.delay_seconds = delay_seconds

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)
        sse_body = _build_sse_body(TEXT_STREAM_CHUNKS)
        return httpx.Response(
            200,
            content=sse_body.encode(),
            headers={"content-type": "text/event-stream"},
        )


class UpstreamErrorTransport(httpx.AsyncBaseTransport):
    """Mock transport that returns upstream error responses."""

    def __init__(
        self,
        status_code: int = 404,
        json_data: Any = None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self.json_data = json_data
        self.text = text

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self.json_data is not None:
            content = json.dumps(self.json_data).encode()
            headers = {"content-type": "application/json"}
        else:
            content = self.text.encode()
            headers = {"content-type": "text/plain"}

        return httpx.Response(
            self.status_code,
            content=content,
            headers=headers,
        )


# ── Shared fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def cfg(tmp_path):
    return _make_config(tmp_path)


def _make_app_with_transport(cfg, transport):
    """Create app and patch its lifespan to use a custom transport."""
    application = create_app(cfg)
    original_lifespan = application.router.lifespan_context

    @asynccontextmanager
    async def patched_lifespan(app):
        async with original_lifespan(app):
            old_client = app.state.edgemem_http_client
            await old_client.aclose()
            app.state.edgemem_http_client = httpx.AsyncClient(
                transport=transport,
                base_url="http://mock-upstream/v1",
            )
            yield
            await app.state.edgemem_http_client.aclose()

    application.router.lifespan_context = patched_lifespan
    return application


@pytest.fixture
def mock_transport():
    return MockTransport()


@pytest.fixture
def app(cfg, mock_transport):
    """Create the app with a mock upstream transport."""
    return _make_app_with_transport(cfg, mock_transport)


@pytest.fixture
def client(app):
    """Synchronous test client."""
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ── Test (a): Non-streaming passthrough ──────────────────────────────────────


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "rss_mb" in data


def test_models(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "test-model"


def test_non_streaming_chat(client):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == "Hello from mock upstream!"
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["usage"]["completion_tokens"] == 5


# ── Test (b): Streaming SSE text deltas ──────────────────────────────────────


def test_streaming_text_deltas(client):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    # Parse SSE events
    text_parts = []
    for line in resp.text.split("\n"):
        if line.startswith("data: ") and line.strip() != "data: [DONE]":
            chunk = json.loads(line[6:])
            delta = chunk["choices"][0]["delta"]
            if "content" in delta and delta["content"]:
                text_parts.append(delta["content"])

    assembled = "".join(text_parts)
    assert assembled == "Hello world"


# ── Test (c): Streaming SSE tool_calls delta preservation ────────────────────


def test_streaming_tool_calls_preservation(client):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Use the tool to check weather"}],
            "stream": True,
        },
    )
    assert resp.status_code == 200

    # Reassemble tool call from SSE chunks
    tool_call_id = None
    function_name = None
    arguments_parts: list[str] = []
    finish_reason = None

    for line in resp.text.split("\n"):
        if not line.startswith("data: ") or line.strip() == "data: [DONE]":
            continue
        chunk = json.loads(line[6:])
        choice = chunk["choices"][0]
        delta = choice.get("delta", {})

        if "tool_calls" in delta:
            tc = delta["tool_calls"][0]
            if "id" in tc:
                tool_call_id = tc["id"]
            fn = tc.get("function", {})
            if "name" in fn:
                function_name = fn["name"]
            if "arguments" in fn:
                arguments_parts.append(fn["arguments"])

        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]

    # Verify exact tool call reconstruction
    assert tool_call_id == "call_abc123"
    assert function_name == "get_weather"
    assert finish_reason == "tool_calls"

    full_args = "".join(arguments_parts)
    parsed_args = json.loads(full_args)
    assert parsed_args == {"location": "San Francisco"}


# ── Test (d): MCP Streamable HTTP endpoint ───────────────────────────────────


def test_mcp_endpoint_initialize(client):
    """MCP initialize + tools/list via Streamable HTTP at POST /mcp."""
    # Send an MCP initialize request
    init_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0"},
        },
    }
    resp = client.post(
        "/mcp",
        json=init_request,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    assert resp.status_code == 200, f"MCP init failed: {resp.status_code} {resp.text}"
    init_data = resp.json()
    assert init_data.get("result", {}).get("serverInfo", {}).get("name") == "EdgeMem"

    # Get session ID from response header for subsequent requests
    session_id = resp.headers.get("mcp-session-id")

    # Send initialized notification
    initialized_notif = {
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
    }
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if session_id:
        headers["mcp-session-id"] = session_id
    resp2 = client.post("/mcp", json=initialized_notif, headers=headers)
    # Notifications return 202 or 204 (no response body)
    assert resp2.status_code in (200, 202, 204), (
        f"MCP initialized notification failed: {resp2.status_code}"
    )

    # Now list tools
    tools_request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {},
    }
    headers2 = {"Content-Type": "application/json", "Accept": "application/json"}
    if session_id:
        headers2["mcp-session-id"] = session_id
    resp3 = client.post("/mcp", json=tools_request, headers=headers2)
    assert resp3.status_code == 200, f"tools/list failed: {resp3.status_code} {resp3.text}"
    tools_data = resp3.json()

    tool_names = {t["name"] for t in tools_data["result"]["tools"]}
    assert "edgemem_status" in tool_names
    assert "edgemem_ping" in tool_names
    assert len(tool_names) >= 2


# ── Test (e): extra_body injection ───────────────────────────────────────────


def test_extra_body_injected(tmp_path, mock_transport):
    """extra_body keys from config are merged into the upstream request body."""
    cfg = _make_config(
        tmp_path,
        extra_body={"reasoning_effort": "none", "custom_flag": 42},
    )
    application = _make_app_with_transport(cfg, mock_transport)

    with TestClient(application, raise_server_exceptions=True) as c:
        c.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )

    assert len(mock_transport.recorded_bodies) == 1
    sent = mock_transport.recorded_bodies[0]
    assert sent["reasoning_effort"] == "none"
    assert sent["custom_flag"] == 42


def test_extra_body_client_precedence(tmp_path, mock_transport):
    """Client-sent keys must NOT be overwritten by extra_body."""
    cfg = _make_config(
        tmp_path,
        extra_body={"reasoning_effort": "none"},
    )
    application = _make_app_with_transport(cfg, mock_transport)

    with TestClient(application, raise_server_exceptions=True) as c:
        c.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "reasoning_effort": "high",  # client overrides config
            },
        )

    sent = mock_transport.recorded_bodies[0]
    assert sent["reasoning_effort"] == "high"  # client wins


# ── Test (f): Slow-TTFT keepalive heartbeats ─────────────────────────────────


def test_slow_ttft_emits_keepalive(tmp_path):
    """When upstream is slow, keepalive SSE comments are emitted before data."""
    import edgemem.proxy.streaming as streaming_mod

    # Patch heartbeat interval to something fast for testing
    original_interval = streaming_mod.HEARTBEAT_INTERVAL_S
    streaming_mod.HEARTBEAT_INTERVAL_S = 0.15  # 150ms

    try:
        cfg = _make_config(tmp_path)
        # Upstream delays 0.5s — should emit ~3 keepalives at 150ms interval
        transport = SlowTTFTTransport(delay_seconds=0.5)
        application = _make_app_with_transport(cfg, transport)

        with TestClient(application, raise_server_exceptions=True) as c:
            resp = c.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        raw = resp.text

        # Must contain at least one keepalive comment
        assert ": keepalive" in raw, (
            f"Expected keepalive comments in response, got:\n{raw[:500]}"
        )

        # Count keepalives
        keepalive_count = raw.count(": keepalive")
        assert keepalive_count >= 1, f"Expected >=1 keepalive, got {keepalive_count}"

        # Real data must still arrive after keepalives
        text_parts = []
        for line in raw.split("\n"):
            if line.startswith("data: ") and line.strip() != "data: [DONE]":
                try:
                    chunk = json.loads(line[6:])
                    delta = chunk["choices"][0]["delta"]
                    if "content" in delta and delta["content"]:
                        text_parts.append(delta["content"])
                except (json.JSONDecodeError, KeyError):
                    pass

        assembled = "".join(text_parts)
        assert assembled == "Hello world"
    finally:
        streaming_mod.HEARTBEAT_INTERVAL_S = original_interval


# ── Test (g): Upstream error handling ───────────────────────────────────────


def test_non_streaming_upstream_404_list(tmp_path):
    """Upstream 404 returning a JSON list (like Gemini OpenAI endpoint) is forwarded cleanly."""
    error_list = [{"error": {"code": 404, "message": "not found"}}]
    transport = UpstreamErrorTransport(status_code=404, json_data=error_list)
    cfg = _make_config(tmp_path)
    application = _make_app_with_transport(cfg, transport)

    with TestClient(application, raise_server_exceptions=True) as c:
        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )
    assert resp.status_code == 404
    assert resp.json() == error_list


def test_streaming_upstream_404(tmp_path):
    """Streaming upstream 404 emits an OpenAI SSE error event and does not crash."""
    error_list = [{"error": {"code": 404, "message": "models/xyz not found"}}]
    transport = UpstreamErrorTransport(status_code=404, json_data=error_list)
    cfg = _make_config(tmp_path)
    application = _make_app_with_transport(cfg, transport)

    with TestClient(application, raise_server_exceptions=True) as c:
        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
    assert resp.status_code == 200
    raw = resp.text
    assert "data: " in raw

    error_payload = None
    for line in raw.split("\n"):
        if line.startswith("data: ") and line.strip() != "data: [DONE]":
            chunk = json.loads(line[6:])
            if "error" in chunk:
                error_payload = chunk["error"]
                break

    assert error_payload is not None
    assert error_payload.get("code") == 404
    assert "models/xyz not found" in error_payload.get("message", "")
