from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from click.testing import CliRunner
import httpx
from starlette.testclient import TestClient

from edgemem.cli import cli
from edgemem.config import (
    CloudUpstreamConfig,
    EdgeMemConfig,
    MemoryConfig,
    ResourcesConfig,
    RoutingConfig,
    SpeculativeConfig,
)
from edgemem.proxy.app import create_app
from edgemem.proxy.speculative import (
    PredictedOutputsEngine,
    SpeculativeStats,
    build_llama_server_command,
)


class CircuitBreakerMockTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.recorded_requests: list[httpx.Request] = []
        self.recorded_bodies: list[dict[str, Any]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.recorded_requests.append(request)
        body = json.loads(request.content.decode("utf-8")) if request.content else {}
        self.recorded_bodies.append(body)

        if "prediction" in body:
            return httpx.Response(
                status_code=400,
                json={
                    "error": {
                        "message": "Unsupported parameter: prediction",
                        "type": "invalid_request_error",
                        "code": "unsupported_parameter",
                    }
                },
                headers={"content-type": "application/json"},
            )

        is_stream = body.get("stream", False)
        if is_stream:
            chunks = [
                b'data: {"id":"chatcmpl-stream","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Refactored "}}]}\n\n',
                b'data: {"id":"chatcmpl-stream","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"code."}}],"usage":{"prompt_tokens":20,"completion_tokens":2,"completion_tokens_details":{"accepted_prediction_tokens":2,"rejected_prediction_tokens":0}}}\n\n',
                b"data: [DONE]\n\n",
            ]

            async def byte_stream():
                for c in chunks:
                    yield c

            return httpx.Response(
                status_code=200,
                content=byte_stream(),
                headers={"content-type": "text/event-stream"},
            )

        return httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Refactored code result"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 20,
                    "completion_tokens_details": {
                        "accepted_prediction_tokens": 15,
                        "rejected_prediction_tokens": 5,
                    },
                },
            },
            headers={"content-type": "application/json"},
        )


def _sample_python_code() -> str:
    return (
        "def compute_transaction_summary(records: list[dict[str, float]]) -> dict[str, float]:\n"
        "    total_inflow = 0.0\n"
        "    total_outflow = 0.0\n"
        "    for record in records:\n"
        "        amount = record.get('amount', 0.0)\n"
        "        if amount > 0:\n"
        "            total_inflow += amount\n"
        "        else:\n"
        "            total_outflow += abs(amount)\n"
        "    net_balance = total_inflow - total_outflow\n"
        "    return {'inflow': total_inflow, 'outflow': total_outflow, 'net': net_balance}\n"
    )


def test_predicted_outputs_engine_extraction():
    cfg = SpeculativeConfig(predicted_outputs_enabled=True)
    engine = PredictedOutputsEngine(cfg)

    code = _sample_python_code()
    body = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": f"Please refactor the following function to be more concise:\n```python\n{code}```",
            }
        ],
    }

    decision = engine.evaluate(
        body,
        target_is_cloud=True,
        base_url="https://api.openai.com/v1",
        model="gpt-4o",
    )
    assert decision.inject is True
    assert decision.reason == "predicted_outputs"
    assert decision.prediction_payload is not None
    assert decision.prediction_payload["type"] == "content"
    assert decision.prediction_payload["content"] == code.strip()


def test_predicted_outputs_engine_no_edit_intent():
    cfg = SpeculativeConfig(predicted_outputs_enabled=True)
    engine = PredictedOutputsEngine(cfg)

    body = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": "Write a python script from scratch to fetch weather data.",
            }
        ],
    }

    decision = engine.evaluate(
        body,
        target_is_cloud=True,
        base_url="https://api.openai.com/v1",
        model="gpt-4o",
    )
    assert decision.inject is False
    assert decision.reason == "no_edit_intent"


def test_predicted_outputs_engine_code_too_short():
    cfg = SpeculativeConfig(predicted_outputs_enabled=True, min_prediction_chars=120)
    engine = PredictedOutputsEngine(cfg)

    short_code = "def add(a, b):\n    return a + b\n"
    body = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": f"Fix this bug please:\n```python\n{short_code}```",
            }
        ],
    }

    decision = engine.evaluate(
        body,
        target_is_cloud=True,
        base_url="https://api.openai.com/v1",
        model="gpt-4o",
    )
    assert decision.inject is False
    assert decision.reason == "code_too_short"


def test_predicted_outputs_incompatibility_gates():
    cfg = SpeculativeConfig(predicted_outputs_enabled=True)
    engine = PredictedOutputsEngine(cfg)
    code = _sample_python_code()

    base_messages = [
        {"role": "user", "content": f"Refactor this:\n```python\n{code}```"}
    ]

    gates = [
        ({"tools": [{"type": "function", "function": {"name": "test"}}]}, "incompatible_tools"),
        ({"n": 2}, "incompatible_n"),
        ({"logprobs": True}, "incompatible_logprobs"),
        ({"presence_penalty": 0.8}, "incompatible_penalties"),
        ({"frequency_penalty": 0.5}, "incompatible_penalties"),
        ({"response_format": {"type": "json_object"}}, "incompatible_response_format"),
        ({"prediction": {"type": "content", "content": "custom"}}, "caller_provided"),
    ]

    for extra_field, expected_reason in gates:
        body = {"model": "gpt-4o", "messages": list(base_messages), **extra_field}
        decision = engine.evaluate(
            body,
            target_is_cloud=True,
            base_url="https://api.openai.com/v1",
            model="gpt-4o",
        )
        assert decision.inject is False
        assert decision.reason == expected_reason


def test_predicted_outputs_headers_and_target_tier():
    cfg = SpeculativeConfig(predicted_outputs_enabled=False)
    engine = PredictedOutputsEngine(cfg)
    code = _sample_python_code()
    body = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": f"Update this function:\n```python\n{code}```"}
        ],
    }

    decision_local = engine.evaluate(
        body,
        target_is_cloud=False,
        base_url="http://127.0.0.1:11434/v1",
        model="qwen",
    )
    assert decision_local.inject is False
    assert decision_local.reason == "local_target"

    decision_disabled = engine.evaluate(
        body,
        target_is_cloud=True,
        base_url="https://api.openai.com/v1",
        model="gpt-4o",
        header_mode="off",
    )
    assert decision_disabled.inject is False
    assert decision_disabled.reason == "header_disabled"

    decision_header_auto = engine.evaluate(
        body,
        target_is_cloud=True,
        base_url="https://api.openai.com/v1",
        model="gpt-4o",
        header_mode="auto",
    )
    assert decision_header_auto.inject is True
    assert decision_header_auto.reason == "predicted_outputs"


def test_fail_open_circuit_breaker_non_streaming(tmp_path: Path):
    db_file = tmp_path / "spec_test.db"
    cfg = EdgeMemConfig(
        memory=MemoryConfig(enabled=False, db_path=str(db_file)),
        cloud=CloudUpstreamConfig(
            enabled=True,
            base_url="https://api.cloud-provider.com/v1",
            model="cloud-model",
        ),
        routing=RoutingConfig(strategy="cloud_only"),
        resources=ResourcesConfig(enabled=False),
        speculative=SpeculativeConfig(predicted_outputs_enabled=True),
    )

    transport = CircuitBreakerMockTransport()
    custom_client = httpx.AsyncClient(
        transport=transport,
        base_url="https://api.cloud-provider.com/v1",
    )

    app = create_app(cfg)

    code = _sample_python_code()
    body = {
        "model": "cloud-model",
        "messages": [
            {
                "role": "user",
                "content": f"Optimize and refactor this:\n```python\n{code}```",
            }
        ],
    }

    with TestClient(app) as tc:
        app.state.edgemem_http_client = custom_client
        resp1 = tc.post("/v1/chat/completions", json=body)
        assert resp1.status_code == 200
        assert resp1.headers.get("X-EdgeMem-Speculative") == "fallback_retry"
        data1 = resp1.json()
        assert data1["choices"][0]["message"]["content"] == "Refactored code result"
        assert len(transport.recorded_requests) == 2
        assert "prediction" in transport.recorded_bodies[0]
        assert "prediction" not in transport.recorded_bodies[1]

        resp2 = tc.post("/v1/chat/completions", json=body)
        assert resp2.status_code == 200
        assert resp2.headers.get("X-EdgeMem-Speculative") == "skipped:provider_unsupported_cached"
        assert len(transport.recorded_requests) == 3
        assert "prediction" not in transport.recorded_bodies[2]


def test_fail_open_circuit_breaker_streaming(tmp_path: Path):
    db_file = tmp_path / "spec_stream_test.db"
    cfg = EdgeMemConfig(
        memory=MemoryConfig(enabled=False, db_path=str(db_file)),
        cloud=CloudUpstreamConfig(
            enabled=True,
            base_url="https://api.cloud-provider.com/v1",
            model="cloud-model",
        ),
        routing=RoutingConfig(strategy="cloud_only"),
        resources=ResourcesConfig(enabled=False),
        speculative=SpeculativeConfig(predicted_outputs_enabled=True),
    )

    transport = CircuitBreakerMockTransport()
    custom_client = httpx.AsyncClient(
        transport=transport,
        base_url="https://api.cloud-provider.com/v1",
    )

    app = create_app(cfg)

    code = _sample_python_code()
    body = {
        "model": "cloud-model",
        "stream": True,
        "messages": [
            {
                "role": "user",
                "content": f"Refactor this:\n```python\n{code}```",
            }
        ],
    }

    with TestClient(app) as tc:
        app.state.edgemem_http_client = custom_client
        resp = tc.post("/v1/chat/completions", json=body)
        assert resp.status_code == 200
        assert resp.headers.get("X-EdgeMem-Speculative") == "fallback_retry"
        text = resp.text
        assert "Refactored " in text
        assert "code." in text
        assert "[DONE]" in text


def test_speculative_telemetry_snapshot_and_health(tmp_path: Path):
    db_file = tmp_path / "spec_telemetry.db"
    cfg = EdgeMemConfig(
        memory=MemoryConfig(enabled=False, db_path=str(db_file)),
        cloud=CloudUpstreamConfig(
            enabled=True,
            base_url="https://api.cloud-provider.com/v1",
            model="cloud-model",
        ),
        routing=RoutingConfig(strategy="cloud_only"),
        resources=ResourcesConfig(enabled=False),
        speculative=SpeculativeConfig(predicted_outputs_enabled=True),
    )

    transport = CircuitBreakerMockTransport()
    custom_client = httpx.AsyncClient(
        transport=transport,
        base_url="https://api.cloud-provider.com/v1",
    )

    app = create_app(cfg)

    code = _sample_python_code()
    body = {
        "model": "cloud-model",
        "messages": [
            {
                "role": "user",
                "content": f"Please edit this function:\n```python\n{code}```",
            }
        ],
    }

    with TestClient(app) as tc:
        app.state.edgemem_http_client = custom_client
        tc.post("/v1/chat/completions", json=body)

        health_resp = tc.get("/health")
        assert health_resp.status_code == 200
        health_data = health_resp.json()
        assert "speculative" in health_data
        spec = health_data["speculative"]
        assert spec["cloud_injected"] == 1
        assert spec["cloud_fallback_retries"] == 1
        assert spec["cloud_accepted_tokens"] == 15
        assert spec["cloud_rejected_tokens"] == 5
        assert spec["cloud_acceptance_pct"] == 75.0


def test_llama_server_speculative_presets():
    cfg = SpeculativeConfig()

    p_ngram_simple = build_llama_server_command(
        model_path="/models/main.gguf",
        mode="ngram-simple",
        profile="t2-gpu",
        binary="llama-server",
        config=cfg,
    )
    assert p_ngram_simple.effective_mode == "ngram-simple"
    assert p_ngram_simple.estimated_extra_ram_mb == 16
    assert "--spec-type" in p_ngram_simple.args
    idx = p_ngram_simple.args.index("--spec-type")
    assert p_ngram_simple.args[idx + 1] == "ngram-simple"
    assert "--spec-ngram-simple-size-n" in p_ngram_simple.args
    assert "--spec-ngram-simple-size-m" in p_ngram_simple.args
    assert "--spec-ngram-simple-min-hits" in p_ngram_simple.args

    assert "--spec-ngram-size-n" not in p_ngram_simple.args
    assert "--spec-ngram-size-m" not in p_ngram_simple.args
    assert "--spec-ngram-min-hits" not in p_ngram_simple.args

    p_ngram_mod = build_llama_server_command(
        model_path="/models/main.gguf",
        mode="ngram-mod",
        profile="t2-gpu",
        config=cfg,
    )
    assert p_ngram_mod.effective_mode == "ngram-mod"
    assert p_ngram_mod.estimated_extra_ram_mb == 24
    assert "--spec-ngram-mod-n-match" in p_ngram_mod.args
    assert "--spec-ngram-mod-n-min" in p_ngram_mod.args
    assert "--spec-ngram-mod-n-max" in p_ngram_mod.args

    p_draft = build_llama_server_command(
        model_path="/models/main.gguf",
        draft_model_path="/models/draft.gguf",
        mode="draft",
        profile="t2-gpu",
        config=cfg,
    )
    assert p_draft.effective_mode == "draft"
    assert p_draft.estimated_extra_ram_mb == 450
    assert "--spec-type" in p_draft.args
    idx = p_draft.args.index("--spec-type")
    assert p_draft.args[idx + 1] == "draft-simple"
    assert "--model-draft" in p_draft.args
    assert "--spec-draft-n-max" in p_draft.args

    p_hybrid = build_llama_server_command(
        model_path="/models/main.gguf",
        draft_model_path="/models/draft.gguf",
        mode="hybrid",
        profile="t2-gpu",
        config=cfg,
    )
    assert p_hybrid.effective_mode == "hybrid"
    assert p_hybrid.estimated_extra_ram_mb == 466
    idx = p_hybrid.args.index("--spec-type")
    assert p_hybrid.args[idx + 1] == "draft-simple,ngram-simple"


def test_llama_server_ram_pressure_downgrade():
    cfg = SpeculativeConfig()

    p_missing = build_llama_server_command(
        model_path="/models/main.gguf",
        draft_model_path=None,
        mode="draft",
        profile="t2-gpu",
        config=cfg,
    )
    assert p_missing.effective_mode == "ngram-simple"
    assert p_missing.downgrade_reason == "missing_draft_model"
    assert p_missing.estimated_extra_ram_mb == 16

    p_cpu = build_llama_server_command(
        model_path="/models/main.gguf",
        draft_model_path="/models/draft.gguf",
        mode="draft",
        profile="t1-cpu",
        config=cfg,
    )
    assert p_cpu.effective_mode == "ngram-simple"
    assert p_cpu.downgrade_reason == "t1_cpu_budget_guard"

    p_pressure = build_llama_server_command(
        model_path="/models/main.gguf",
        draft_model_path="/models/draft.gguf",
        mode="draft",
        profile="t2-gpu",
        pressure="soft",
        config=cfg,
    )
    assert p_pressure.effective_mode == "ngram-simple"
    assert p_pressure.downgrade_reason == "ram_pressure_guard"


def test_cli_llama_cmd_and_status(tmp_path: Path):
    runner = CliRunner()

    res = runner.invoke(
        cli,
        [
            "llama-cmd",
            "--model",
            "/models/main.gguf",
            "--mode",
            "ngram-simple",
            "--profile",
            "t2-gpu",
            "--port",
            "8081",
        ],
    )
    assert res.exit_code == 0
    assert "Profile:   t2-gpu" in res.output
    assert "Mode:      ngram-simple" in res.output
    assert "Extra RAM: 16 MB" in res.output
    assert "--spec-type ngram-simple" in res.output

    res_downgrade = runner.invoke(
        cli,
        [
            "llama-cmd",
            "--model",
            "/models/main.gguf",
            "--draft-model",
            "/models/draft.gguf",
            "--mode",
            "draft",
            "--profile",
            "t1-cpu",
        ],
    )
    assert res_downgrade.exit_code == 0
    assert "Mode:      ngram-simple" in res_downgrade.output
    assert "Downgrade: t1_cpu_budget_guard" in res_downgrade.output

    health_json = {
        "status": "ok",
        "upstream_base_url": "http://127.0.0.1:11434/v1",
        "model": "qwen",
        "rss_mb": 80.5,
        "speculative": {
            "cloud_injected": 10,
            "cloud_skipped": 5,
            "cloud_fallback_retries": 1,
            "cloud_accepted_tokens": 120,
            "cloud_rejected_tokens": 30,
            "cloud_acceptance_pct": 80.0,
        },
    }

    class MockStatusResponse:
        status_code = 200

        def json(self):
            return health_json

    with patch("httpx.get", return_value=MockStatusResponse()):
        status_res = runner.invoke(cli, ["status"])
        assert status_res.exit_code == 0
        assert "Speculative: cloud_injected=10, retries=1, accepted=120/150 (80.0%)" in status_res.output
