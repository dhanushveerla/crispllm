"""FastAPI application — OpenAI-compatible proxy + embedded MCP server.

Endpoints:
  GET  /health
  GET  /v1/models
  POST /v1/chat/completions  (stream: true/false)
  POST /mcp                  (MCP Streamable HTTP)
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from edgemem.config import EdgeMemConfig
from edgemem.proxy.mcp_server import create_mcp_server, create_mcp_starlette_app
from edgemem.proxy.streaming import (
    count_output_tokens_from_sse,
    estimate_tokens_from_messages,
    stream_sse_passthrough,
)
from edgemem.proxy.telemetry import (
    RequestTelemetry,
    TelemetryLogger,
    Timer,
    get_rss_mb,
)


def create_app(cfg: EdgeMemConfig | None = None) -> FastAPI:
    """Build and return the FastAPI application.

    Args:
        cfg: EdgeMem configuration. Defaults loaded if None.
    """
    if cfg is None:
        from edgemem.config import load_config

        cfg = load_config()

    # Create MCP server and its Starlette sub-app
    mcp_server = create_mcp_server()
    mcp_app = create_mcp_starlette_app(mcp_server)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Manage HTTP client, telemetry logger, and MCP session lifecycle."""
        tel = TelemetryLogger(cfg.telemetry_path)
        client = httpx.AsyncClient(
            base_url=cfg.upstream_base_url,
            timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0),
        )

        # Wire telemetry into MCP tools
        mcp_server._edgemem_set_telemetry(tel, cfg)  # type: ignore[attr-defined]

        app.state.edgemem_config = cfg
        app.state.edgemem_http_client = client
        app.state.edgemem_telemetry = tel

        # Start MCP session manager inside our lifespan
        session_mgr = mcp_server.session_manager
        async with session_mgr.run():
            yield

        await client.aclose()

    app = FastAPI(title="EdgeMem Proxy", version="0.1.0", lifespan=lifespan)

    # --- FastAPI routes (defined BEFORE mounting MCP) -------------------------

    @app.get("/health")
    async def health(request: Request) -> JSONResponse:
        c: EdgeMemConfig = request.app.state.edgemem_config
        return JSONResponse(
            {
                "status": "ok",
                "upstream_base_url": c.upstream_base_url,
                "model": c.model,
                "rss_mb": round(get_rss_mb(), 2),
            }
        )

    @app.get("/v1/models")
    async def list_models(request: Request) -> JSONResponse:
        c: EdgeMemConfig = request.app.state.edgemem_config
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {
                        "id": c.model,
                        "object": "model",
                        "owned_by": "edgemem-proxy",
                    }
                ],
            }
        )

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(request: Request) -> StreamingResponse | JSONResponse:
        c: EdgeMemConfig = request.app.state.edgemem_config
        client: httpx.AsyncClient = request.app.state.edgemem_http_client
        tel: TelemetryLogger = request.app.state.edgemem_telemetry

        body = await request.json()
        is_stream = body.get("stream", False)

        # Build upstream request body — keep system/tools prefix byte-identical
        upstream_body = _build_upstream_body(body, c)
        input_tokens = estimate_tokens_from_messages(body.get("messages", []))

        headers = _upstream_headers(c)

        if is_stream:
            return await _handle_streaming(
                client, upstream_body, headers, input_tokens, c.model, tel
            )
        else:
            return await _handle_non_streaming(
                client, upstream_body, headers, input_tokens, c.model, tel
            )

    # Mount MCP Starlette app — mounted AFTER explicit routes so FastAPI
    # routes match first. Mount path="" so the internal /mcp route resolves
    # to /mcp (not /mcp/mcp).
    app.mount("", mcp_app)

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_upstream_body(body: dict[str, Any], cfg: EdgeMemConfig) -> dict[str, Any]:
    """Build the upstream request body, merging extra_body fields."""
    upstream = dict(body)
    # Ensure model is set
    if "model" not in upstream or not upstream["model"]:
        upstream["model"] = cfg.model
    # Merge extra_body — user request fields take precedence
    if cfg.extra_body:
        for k, v in cfg.extra_body.items():
            if k not in upstream:
                upstream[k] = v
    return upstream


def _upstream_headers(cfg: EdgeMemConfig) -> dict[str, str]:
    """Build headers for the upstream request."""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    api_key = cfg.upstream_api_key
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


async def _handle_non_streaming(
    client: httpx.AsyncClient,
    body: dict[str, Any],
    headers: dict[str, str],
    input_tokens: int,
    model: str,
    tel: TelemetryLogger,
) -> JSONResponse:
    """Forward a non-streaming chat completion and log telemetry."""
    timer = Timer()
    with timer:
        resp = await client.post("/chat/completions", json=body, headers=headers)
        timer.mark_first_token()

    data = resp.json()

    # Extract output tokens
    usage = data.get("usage", {})
    output_tokens = usage.get("completion_tokens", 0)

    tel.log(
        RequestTelemetry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            ttft_ms=round(timer.ttft_ms, 2),
            total_latency_ms=round(timer.total_ms, 2),
            proxy_rss_mb=round(get_rss_mb(), 2),
            stream=False,
        )
    )

    return JSONResponse(data, status_code=resp.status_code)


async def _handle_streaming(
    client: httpx.AsyncClient,
    body: dict[str, Any],
    headers: dict[str, str],
    input_tokens: int,
    model: str,
    tel: TelemetryLogger,
) -> StreamingResponse:
    """Forward a streaming chat completion, passing SSE chunks through faithfully."""
    timer = Timer()
    timer.__enter__()

    async def generate() -> AsyncIterator[str]:
        output_tokens = 0
        try:
            async with client.stream(
                "POST", "/chat/completions", json=body, headers=headers
            ) as resp:
                async for sse_line in stream_sse_passthrough(
                    resp, on_first_data=timer.mark_first_token
                ):
                    output_tokens += count_output_tokens_from_sse(sse_line)
                    yield sse_line
        finally:
            timer.__exit__(None, None, None)
            tel.log(
                RequestTelemetry(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    ttft_ms=round(timer.ttft_ms, 2),
                    total_latency_ms=round(timer.total_ms, 2),
                    proxy_rss_mb=round(get_rss_mb(), 2),
                    stream=True,
                )
            )

    return StreamingResponse(generate(), media_type="text/event-stream")
