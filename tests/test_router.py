from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient
import httpx
import pytest

from edgemem.config import (
    CloudUpstreamConfig,
    EdgeMemConfig,
    MemoryConfig,
    PrivacyConfig,
    RoutingConfig,
)
from edgemem.proxy.app import create_app


class MockDualTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.request_bodies: list[dict[str, Any]] = []
        self.local_error: Exception | int | None = None
        self.custom_stream_lines: list[str] | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        content = await request.aread()
        body = json.loads(content.decode("utf-8")) if content else {}
        self.request_bodies.append(body)

        is_local = (
            "127.0.0.1" in str(request.url)
            or "localhost" in str(request.url)
            or "mock-local" in str(request.url)
        )

        if is_local and self.local_error is not None:
            if isinstance(self.local_error, Exception):
                raise self.local_error
            return httpx.Response(
                self.local_error,
                json={"error": f"error {self.local_error}"},
                headers={"content-type": "application/json"},
            )

        is_stream = body.get("stream", False)
        if is_stream:
            if self.custom_stream_lines is not None:
                async def _custom_stream():
                    for line in self.custom_stream_lines:
                        yield line.encode("utf-8")

                return httpx.Response(
                    200,
                    content=_custom_stream(),
                    headers={"content-type": "text/event-stream"},
                )

            async def _default_stream():
                chunk = {
                    "id": "chatcmpl-stream-test",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "Stream chunk response"},
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"

            return httpx.Response(
                200,
                content=_default_stream(),
                headers={"content-type": "text/event-stream"},
            )

        messages = body.get("messages", [])
        last_msg = messages[-1].get("content", "") if messages else ""
        if isinstance(last_msg, list):
            text_parts = [
                b.get("text", "") for b in last_msg if b.get("type") == "text"
            ]
            last_msg = " ".join(text_parts)

        resp_data = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"Echo: {last_msg}",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18,
            },
        }
        return httpx.Response(
            200,
            json=resp_data,
            headers={"content-type": "application/json"},
        )


def _make_test_setup(
    tmp_path,
    cloud_enabled: bool = True,
    strategy: str = "local_first",
    token_threshold: int = 6000,
    escalate_on_error: bool = True,
    redact_on_cloud_only: bool = True,
    local_base_url: str = "http://mock-local/v1",
    cloud_base_url: str = "http://mock-cloud/v1",
) -> tuple[TestClient, MockDualTransport, EdgeMemConfig]:
    cfg = EdgeMemConfig(
        host="127.0.0.1",
        port=8080,
        upstream_base_url=local_base_url,
        model="local-model",
        telemetry_dir=str(tmp_path / "telemetry"),
        memory=MemoryConfig(db_path=str(tmp_path / "memory.db")),
        cloud=CloudUpstreamConfig(
            enabled=cloud_enabled,
            base_url=cloud_base_url,
            model="cloud-model",
        ),
        routing=RoutingConfig(
            strategy=strategy,
            escalate_token_threshold=token_threshold,
            escalate_on_local_error=escalate_on_error,
        ),
        privacy=PrivacyConfig(
            enabled=True,
            redact_on_cloud_only=redact_on_cloud_only,
        ),
    )
    app = create_app(cfg)
    transport = MockDualTransport()

    from contextlib import asynccontextmanager

    orig_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def patched_lifespan(application):
        async with orig_lifespan(application):
            old = application.state.edgemem_http_client
            await old.aclose()
            application.state.edgemem_http_client = httpx.AsyncClient(
                transport=transport,
                base_url=local_base_url,
            )
            yield
            await application.state.edgemem_http_client.aclose()

    app.router.lifespan_context = patched_lifespan
    client = TestClient(app)
    return client, transport, cfg


def test_backward_compatibility(tmp_path) -> None:
    cfg = EdgeMemConfig(
        host="127.0.0.1",
        port=8080,
        upstream_base_url="http://mock-local/v1",
        model="legacy-model",
        telemetry_dir=str(tmp_path / "telemetry"),
        memory=MemoryConfig(db_path=str(tmp_path / "memory.db")),
    )
    app = create_app(cfg)
    transport = MockDualTransport()
    with TestClient(app) as client:
        app.state.edgemem_http_client = httpx.AsyncClient(
            transport=transport,
            base_url=cfg.upstream_base_url,
        )

        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        health = health_resp.json()
        assert health["status"] == "ok"
        assert health["cloud_enabled"] is False
        assert health["privacy_enabled"] is True
        assert health["local_requests"] == 0
        assert health["cloud_requests"] == 0

        post_resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "ping"}]},
        )
        assert post_resp.status_code == 200
        assert post_resp.headers["X-EdgeMem-Route"] == "local"
        assert post_resp.headers["X-EdgeMem-Route-Reason"] == "default_local"
        assert post_resp.headers["X-EdgeMem-Redacted"] == "0"


def test_routing_rules(tmp_path) -> None:
    client, transport, cfg = _make_test_setup(
        tmp_path,
        cloud_enabled=True,
        strategy="local_first",
        token_threshold=100,
    )
    with client:
        r1 = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "short message"}]},
        )
        assert r1.status_code == 200
        assert r1.headers["X-EdgeMem-Route"] == "local"
        assert r1.headers["X-EdgeMem-Route-Reason"] == "default_local"

        r2 = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "short message"}]},
            headers={"X-EdgeMem-Route": "cloud"},
        )
        assert r2.status_code == 200
        assert r2.headers["X-EdgeMem-Route"] == "cloud"
        assert r2.headers["X-EdgeMem-Route-Reason"] == "header_override"

        r3 = client.post(
            "/v1/chat/completions",
            json={
                "model": "cloud/custom-cloud-model",
                "messages": [{"role": "user", "content": "short message"}],
            },
        )
        assert r3.status_code == 200
        assert r3.headers["X-EdgeMem-Route"] == "cloud"
        assert r3.headers["X-EdgeMem-Route-Reason"] == "model_prefix"

        long_text = "word " * 500
        r4 = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": long_text}]},
        )
        assert r4.status_code == 200
        assert r4.headers["X-EdgeMem-Route"] == "cloud"
        assert r4.headers["X-EdgeMem-Route-Reason"] == "token_threshold"


def test_local_to_cloud_fallback(tmp_path) -> None:
    client, transport, cfg = _make_test_setup(
        tmp_path,
        cloud_enabled=True,
        escalate_on_error=True,
    )
    with client:
        transport.local_error = httpx.ConnectError("Connection refused")
        secret = "sk-proj-1234567890abcdef12345678"

        r1 = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": f"My secret key is {secret}"}
                ]
            },
        )
        assert r1.status_code == 200
        assert r1.headers["X-EdgeMem-Route"] == "cloud"
        assert r1.headers["X-EdgeMem-Route-Reason"] == "local_error_fallback"
        assert r1.headers["X-EdgeMem-Redacted"] == "1"

        data = r1.json()
        assert secret in data["choices"][0]["message"]["content"]
        assert "__EDGEMEM_" not in data["choices"][0]["message"]["content"]

        cloud_req_body = transport.request_bodies[-1]
        assert secret not in json.dumps(cloud_req_body)
        assert "__EDGEMEM_SECRET_1__" in json.dumps(cloud_req_body)

        transport.local_error = 503
        r2 = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Retry message"}]},
        )
        assert r2.status_code == 200
        assert r2.headers["X-EdgeMem-Route"] == "cloud"
        assert r2.headers["X-EdgeMem-Route-Reason"] == "local_error_fallback"

        health = client.get("/health").json()
        assert health["fallback_escalations"] == 2
        assert health["cloud_requests"] == 2


def test_local_404_model_not_found_fallback(tmp_path) -> None:
    client, transport, cfg = _make_test_setup(
        tmp_path,
        cloud_enabled=True,
        escalate_on_error=True,
    )
    with client:
        transport.local_error = 404
        secret = "sk-proj-1234567890abcdef12345678"

        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": f"Secret key: {secret}"}
                ]
            },
        )
        assert resp.status_code == 200
        assert resp.headers["X-EdgeMem-Route"] == "cloud"
        assert resp.headers["X-EdgeMem-Route-Reason"] == "local_error_fallback"
        assert resp.headers["X-EdgeMem-Redacted"] == "1"

        data = resp.json()
        assert secret in data["choices"][0]["message"]["content"]
        assert "__EDGEMEM_" not in data["choices"][0]["message"]["content"]

        cloud_req = transport.request_bodies[-1]
        assert secret not in json.dumps(cloud_req)
        assert "__EDGEMEM_SECRET_1__" in json.dumps(cloud_req)

        health = client.get("/health").json()
        assert health["fallback_escalations"] == 1
        assert health["cloud_requests"] == 1

        transport.local_error = 429
        resp_stream = client.post(
            "/v1/chat/completions",
            json={
                "stream": True,
                "messages": [{"role": "user", "content": "streaming retry"}],
            },
        )
        assert resp_stream.status_code == 200
        assert resp_stream.headers["X-EdgeMem-Route"] == "cloud"
        assert resp_stream.headers["X-EdgeMem-Route-Reason"] == "local_error_fallback"

        health = client.get("/health").json()
        assert health["fallback_escalations"] == 2
        assert health["cloud_requests"] == 2


def test_non_streaming_privacy_round_trip(tmp_path) -> None:
    client, transport, cfg = _make_test_setup(
        tmp_path,
        cloud_enabled=True,
        strategy="cloud_only",
    )
    with client:
        nvapi_key = "nvapi-abcdef1234567890abcdef123456"
        db_pass = "supersecretpass123"
        email = "alice@company.com"
        prompt = (
            f"Here is key {nvapi_key}, "
            f"postgres://user:{db_pass}@host:5432/db, "
            f"and contact {email}."
        )

        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": prompt}]},
        )
        assert resp.status_code == 200
        assert resp.headers["X-EdgeMem-Redacted"] == "3"

        cloud_req = transport.request_bodies[-1]
        req_json = json.dumps(cloud_req)
        assert nvapi_key not in req_json
        assert db_pass not in req_json
        assert email not in req_json

        content = resp.json()["choices"][0]["message"]["content"]
        assert nvapi_key in content
        assert db_pass in content
        assert email in content
        assert "__EDGEMEM_" not in content

        health = client.get("/health").json()
        assert health["secrets_redacted_total"] == 3


def test_streaming_sse_privacy_round_trip(tmp_path) -> None:
    client, transport, cfg = _make_test_setup(
        tmp_path,
        cloud_enabled=True,
        strategy="cloud_only",
    )
    with client:
        secret = "sk-proj-1234567890abcdef12345678"

        transport.custom_stream_lines = [
            f'data: {json.dumps({"id": "1", "choices": [{"index": 0, "delta": {"content": "Your key is __EDGEM"}, "finish_reason": None}]})}\n\n',
            f'data: {json.dumps({"id": "1", "choices": [{"index": 0, "delta": {"content": "EM_SECRET_1_"}, "finish_reason": None}]})}\n\n',
            f'data: {json.dumps({"id": "1", "choices": [{"index": 0, "delta": {"content": "_ verified!"}, "finish_reason": None}]})}\n\n',
            f'data: {json.dumps({"id": "1", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})}\n\n',
            "data: [DONE]\n\n",
        ]

        resp = client.post(
            "/v1/chat/completions",
            json={
                "stream": True,
                "messages": [
                    {"role": "user", "content": f"My secret is {secret}"}
                ],
            },
        )
        assert resp.status_code == 200
        assert resp.headers["X-EdgeMem-Route"] == "cloud"
        assert resp.headers["X-EdgeMem-Redacted"] == "1"

        lines = resp.text.split("\n\n")
        deltas = []
        saw_done = False
        for block in lines:
            if not block.strip():
                continue
            if block == "data: [DONE]":
                saw_done = True
                continue
            assert block.startswith("data: ")
            chunk = json.loads(block[6:])
            text_part = chunk["choices"][0]["delta"].get("content", "")
            if text_part:
                deltas.append(text_part)

        assert saw_done
        reconstructed = "".join(deltas)
        assert reconstructed == f"Your key is {secret} verified!"
        assert "__EDGEMEM_" not in reconstructed


def test_localhost_skip(tmp_path) -> None:
    client, transport, cfg = _make_test_setup(
        tmp_path,
        cloud_enabled=True,
        strategy="local_only",
        redact_on_cloud_only=True,
        local_base_url="http://127.0.0.1:11434/v1",
    )
    with client:
        secret = "nvapi-1234567890abcdef123456"
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": f"Local secret: {secret}"}
                ]
            },
        )
        assert resp.status_code == 200
        assert resp.headers["X-EdgeMem-Route"] == "local"
        assert resp.headers["X-EdgeMem-Redacted"] == "0"

        local_req = transport.request_bodies[-1]
        req_json = json.dumps(local_req)
        assert secret in req_json
        assert "__EDGEMEM_" not in req_json
