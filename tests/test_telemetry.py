from __future__ import annotations

from contextlib import asynccontextmanager
import json
from pathlib import Path
import time
from typing import Any

from click.testing import CliRunner
import httpx
from starlette.testclient import TestClient

from edgemem.cli import cli
from edgemem.config import (
    CloudUpstreamConfig,
    EdgeMemConfig,
    MemoryConfig,
    PrivacyConfig,
    RoutingConfig,
    load_config,
)
from edgemem.proxy.app import create_app
from edgemem.proxy.telemetry import TelemetryStore

MOCK_UPSTREAM_RESPONSE: dict[str, Any] = {
    "id": "chatcmpl-telemetry-test",
    "object": "chat.completion",
    "model": "test-model",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Secret received safely.",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
}

STREAM_CHUNKS = [
    {
        "id": "chatcmpl-stream-test",
        "object": "chat.completion.chunk",
        "model": "test-model",
        "choices": [
            {"index": 0, "delta": {"role": "assistant", "content": "Streamed "}, "finish_reason": None}
        ],
    },
    {
        "id": "chatcmpl-stream-test",
        "object": "chat.completion.chunk",
        "model": "test-model",
        "choices": [
            {"index": 0, "delta": {"content": "response."}, "finish_reason": None}
        ],
    },
    {
        "id": "chatcmpl-stream-test",
        "object": "chat.completion.chunk",
        "model": "test-model",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    },
]


def _build_sse_text(chunks: list[dict[str, Any]]) -> str:
    lines = [f"data: {json.dumps(c)}\n\n" for c in chunks]
    lines.append("data: [DONE]\n\n")
    return "".join(lines)


class MockTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.recorded_bodies: list[dict[str, Any]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.recorded_bodies.append(body)
        is_stream = body.get("stream", False)

        if is_stream:
            content = _build_sse_text(STREAM_CHUNKS)
            return httpx.Response(
                200,
                content=content.encode("utf-8"),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(
            200,
            json=MOCK_UPSTREAM_RESPONSE,
            headers={"content-type": "application/json"},
        )


def _make_test_app(tmp_path: Path, db_file: Path) -> tuple[Any, MockTransport]:
    cfg = EdgeMemConfig(
        host="127.0.0.1",
        port=8080,
        upstream_base_url="http://mock-upstream",
        model="local-model",
        telemetry_dir=str(tmp_path / "telemetry"),
        memory=MemoryConfig(
            enabled=True,
            compact_middle_turns=True,
            compact_threshold_tokens=0,
            compact_min_turn_tokens=0,
            keep_recent_turns=2,
            db_path=str(db_file),
        ),
        privacy=PrivacyConfig(
            enabled=True,
            redact_on_cloud_only=False,
            redact_secrets=True,
        ),
        cloud=CloudUpstreamConfig(
            enabled=True,
            base_url="http://mock-cloud",
            model="cloud-model",
        ),
        routing=RoutingConfig(
            strategy="local_first",
            escalate_token_threshold=10000,
        ),
    )
    transport = MockTransport()
    app = create_app(cfg)
    orig_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def patched_lifespan(application: Any):
        async with orig_lifespan(application):
            old_client = application.state.edgemem_http_client
            await old_client.aclose()
            application.state.edgemem_http_client = httpx.AsyncClient(
                transport=transport,
                base_url="http://mock-upstream/v1",
            )
            yield
            await application.state.edgemem_http_client.aclose()

    app.router.lifespan_context = patched_lifespan
    return app, transport


def test_telemetry_store_unit(tmp_path: Path) -> None:
    db_path = tmp_path / "store_test.db"
    with TelemetryStore(db_path) as store:
        empty_summary = store.summary()
        assert empty_summary["total_requests"] == 0
        assert empty_summary["local_requests"] == 0
        assert empty_summary["cloud_requests"] == 0
        assert empty_summary["local_offload_pct"] == 0.0
        assert empty_summary["total_tokens_raw"] == 0
        assert empty_summary["mean_latency_ms"] == 0.0

        now = time.time()
        store.record(
            route_tier="local",
            route_reason="local_first",
            model="qwen2.5-coder:7b",
            est_tokens_raw=1000,
            est_tokens_sent=800,
            tokens_saved=200,
            compacted_turns=1,
            secrets_redacted=0,
            fallback_used=0,
            latency_ms=50.0,
            ts=now,
        )
        store.record(
            route_tier="cloud",
            route_reason="token_threshold_exceeded",
            model="nvidia/nemotron-3.5-bolt-e4b",
            est_tokens_raw=8000,
            est_tokens_sent=7000,
            tokens_saved=1000,
            compacted_turns=2,
            secrets_redacted=2,
            fallback_used=0,
            latency_ms=150.0,
            ts=now + 1.0,
        )
        store.record(
            route_tier="cloud",
            route_reason="local_failure",
            model="nvidia/nemotron-3.5-bolt-e4b",
            est_tokens_raw=2000,
            est_tokens_sent=2000,
            tokens_saved=0,
            compacted_turns=0,
            secrets_redacted=1,
            fallback_used=1,
            latency_ms=200.0,
            ts=now + 2.0,
        )

        s = store.summary()
        assert s["total_requests"] == 3
        assert s["local_requests"] == 1
        assert s["cloud_requests"] == 2
        assert s["local_offload_pct"] == 33.33
        assert s["fallback_escalations"] == 1
        assert s["total_tokens_raw"] == 11000
        assert s["total_tokens_sent"] == 9800
        assert s["total_tokens_saved"] == 1200
        assert s["token_reduction_pct"] == 10.91
        assert s["total_compacted_turns"] == 3
        assert s["total_secrets_redacted"] == 3
        assert s["mean_latency_ms"] == 133.33

        old_ts = now - (10 * 86400)
        store.record(
            route_tier="local",
            route_reason="local_first",
            model="qwen2.5-coder:7b",
            est_tokens_raw=500,
            est_tokens_sent=500,
            tokens_saved=0,
            compacted_turns=0,
            secrets_redacted=0,
            fallback_used=0,
            latency_ms=40.0,
            ts=old_ts,
        )

        all_summary = store.summary()
        assert all_summary["total_requests"] == 4

        recent_summary = store.summary(days=7)
        assert recent_summary["total_requests"] == 3

        wider_summary = store.summary(days=15)
        assert wider_summary["total_requests"] == 4

        deleted = store.reset()
        assert deleted == 4
        after_reset = store.summary()
        assert after_reset["total_requests"] == 0


def test_telemetry_store_error_handling(tmp_path: Path) -> None:
    db_path = tmp_path / "closed.db"
    store = TelemetryStore(db_path)
    store.close()
    store.record(
        route_tier="local",
        route_reason="test",
        model="test",
        est_tokens_raw=100,
        est_tokens_sent=100,
        tokens_saved=0,
        compacted_turns=0,
        secrets_redacted=0,
        fallback_used=0,
        latency_ms=10.0,
    )


def test_proxy_telemetry_integration(tmp_path: Path) -> None:
    db_file = tmp_path / "integration_memory.db"
    app, transport = _make_test_app(tmp_path, db_file)

    boilerplate = (
        "def generic_boilerplate_function():\n"
        "    a = 1\n"
        "    b = 2\n"
        "    return a + b\n"
    ) * 30

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": f"Turn 1 secret sk-1234567890123456789012345678901234567890\n{boilerplate}",
        },
        {"role": "assistant", "content": f"Turn 2 middle\n{boilerplate}"},
        {"role": "user", "content": f"Turn 3 middle\n{boilerplate}"},
        {"role": "assistant", "content": "Turn 4 recent."},
        {"role": "user", "content": "Turn 5 recent question."},
    ]

    with TestClient(app) as client:
        resp_non_stream = client.post(
            "/v1/chat/completions",
            json={
                "model": "local-model",
                "messages": messages,
                "stream": False,
            },
        )
        assert resp_non_stream.status_code == 200

        resp_stream = client.post(
            "/v1/chat/completions",
            json={
                "model": "local-model",
                "messages": messages,
                "stream": True,
            },
        )
        assert resp_stream.status_code == 200
        stream_lines = list(resp_stream.iter_lines())
        assert any("Streamed" in line for line in stream_lines)

    with TelemetryStore(db_file) as store:
        s = store.summary()
        assert s["total_requests"] == 2
        assert s["local_requests"] == 2
        assert s["total_tokens_raw"] > s["total_tokens_sent"]
        assert s["total_tokens_saved"] > 0
        assert s["token_reduction_pct"] > 0.0
        assert s["total_compacted_turns"] > 0
        assert s["total_secrets_redacted"] >= 2
        assert s["mean_latency_ms"] > 0.0


def test_cli_stats(tmp_path: Path) -> None:
    db_file = tmp_path / "cli_test.db"
    config_file = tmp_path / "cli_config.toml"
    config_file.write_text(f"""\
[server]
host = "127.0.0.1"
port = 8080

[memory]
enabled = true
db_path = "{db_file.as_posix()}"
""", encoding="utf-8")

    with TelemetryStore(db_file) as store:
        store.record(
            route_tier="local",
            route_reason="local_first",
            model="qwen2.5-coder:7b",
            est_tokens_raw=1000,
            est_tokens_sent=800,
            tokens_saved=200,
            compacted_turns=1,
            secrets_redacted=1,
            fallback_used=0,
            latency_ms=45.0,
        )

    runner = CliRunner()
    res_plain = runner.invoke(cli, ["stats", "--config", str(config_file)])
    assert res_plain.exit_code == 0
    assert "Requests:" in res_plain.output
    assert "Total:               1" in res_plain.output
    assert "Context Savings:" in res_plain.output
    assert "Raw Tokens:          1,000" in res_plain.output
    assert "Saved Tokens:        200" in res_plain.output
    assert "Privacy:" in res_plain.output
    assert "Secrets Redacted:    1" in res_plain.output
    assert "Latency:" in res_plain.output

    res_json = runner.invoke(cli, ["stats", "--json", "--config", str(config_file)])
    assert res_json.exit_code == 0
    parsed = json.loads(res_json.output)
    assert parsed["total_requests"] == 1
    assert parsed["total_tokens_saved"] == 200

    res_reset = runner.invoke(cli, ["stats", "--reset", "--config", str(config_file)])
    assert res_reset.exit_code == 0
    assert "Cleared 1 telemetry events." in res_reset.output

    res_after = runner.invoke(cli, ["stats", "--json", "--config", str(config_file)])
    assert res_after.exit_code == 0
    parsed_after = json.loads(res_after.output)
    assert parsed_after["total_requests"] == 0


def test_cli_init_presets(tmp_path: Path) -> None:
    runner = CliRunner()

    presets = ["hybrid", "local", "nvidia", "gemini"]
    for preset in presets:
        target_path = tmp_path / f"edgemem_{preset}.toml"
        res = runner.invoke(cli, ["init", "--preset", preset, "--output", str(target_path)])
        assert res.exit_code == 0
        assert target_path.exists()
        cfg = load_config(target_path)

        if preset == "hybrid":
            assert cfg.upstream_base_url == "http://127.0.0.1:11434/v1"
            assert cfg.model == "qwen2.5-coder:7b"
            assert cfg.cloud.enabled is True
            assert cfg.cloud.base_url == "https://integrate.api.nvidia.com/v1"
            assert cfg.cloud.api_key_env == "NVIDIA_API_KEY"
            assert cfg.cloud.model == "nvidia/nemotron-3.5-bolt-e4b"
            assert cfg.routing.strategy == "local_first"
            assert cfg.routing.escalate_token_threshold == 6000
            assert cfg.routing.escalate_on_local_error is True
            assert cfg.privacy.enabled is True
            assert cfg.privacy.redact_on_cloud_only is True
        elif preset == "local":
            assert cfg.upstream_base_url == "http://127.0.0.1:11434/v1"
            assert cfg.cloud.enabled is False
            assert cfg.routing.strategy == "local_only"
        elif preset == "nvidia":
            assert cfg.upstream_base_url == "https://integrate.api.nvidia.com/v1"
            assert cfg.api_key_env == "NVIDIA_API_KEY"
            assert cfg.model == "nvidia/nemotron-3.5-bolt-e4b"
            assert cfg.cloud.enabled is False
            assert cfg.privacy.enabled is True
        elif preset == "gemini":
            assert cfg.upstream_base_url == "https://generativelanguage.googleapis.com/v1beta/openai/"
            assert cfg.api_key_env == "GEMINI_API_KEY"
            assert cfg.model == "gemini-2.5-flash"
            assert cfg.cloud.enabled is False
            assert cfg.privacy.enabled is True

    existing_file = tmp_path / "edgemem_hybrid.toml"
    res_refuse = runner.invoke(cli, ["init", "--preset", "local", "--output", str(existing_file)])
    assert res_refuse.exit_code != 0
    assert "already exists" in res_refuse.output

    res_force = runner.invoke(cli, ["init", "--preset", "local", "--output", str(existing_file), "--force"])
    assert res_force.exit_code == 0
    cfg_forced = load_config(existing_file)
    assert cfg_forced.routing.strategy == "local_only"
