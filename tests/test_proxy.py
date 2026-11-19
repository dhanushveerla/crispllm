"""Tests for EdgeMem proxy — no live upstream required.

Uses httpx mock transport to simulate an upstream OpenAI-compatible server,
verifying:
  a) Non-streaming chat completion passthrough
  b) Streaming SSE text deltas
  c) Streaming SSE multi-chunk tool_calls delta preservation
  d) MCP Streamable HTTP at POST /mcp
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import pytest_asyncio
from starlette.testclient import TestClient

from edgemem.config import EdgeMemConfig
from edgemem.proxy.app import create_app

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_config(tmp_path) -> EdgeMemConfig:
    """Create a test config pointing at a placeholder upstream."""
    return EdgeMemConfig(
        host="127.0.0.1",
        port=8080,
        upstream_base_url="http://mock-upstream",
        model="test-model",
        telemetry_dir=str(tmp_path / "telemetry"),
    )


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
    """Mock transport that returns canned responses based on request body."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
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


@pytest.fixture
def cfg(tmp_path):
    return _make_config(tmp_path)


@pytest.fixture
def app(cfg, monkeypatch):
    """Create the app with a mock upstream transport."""
    application = create_app(cfg)

    # Patch the httpx client in lifespan to use our mock transport
    original_lifespan = application.router.lifespan_context

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def patched_lifespan(app):
        async with original_lifespan(app):
            # Replace the HTTP client with one using mock transport
            old_client = app.state.edgemem_http_client
            await old_client.aclose()
            app.state.edgemem_http_client = httpx.AsyncClient(
                transport=MockTransport(),
                base_url="http://mock-upstream/v1",
            )
            yield
            await app.state.edgemem_http_client.aclose()

    application.router.lifespan_context = patched_lifespan
    return application


@pytest.fixture
def client(app):
    """Synchronous test client for non-streaming tests."""
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
