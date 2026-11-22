from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

from bench.run_bench import run_task, summarize_run_results
from crispllm.config import EdgeMemConfig, MemoryConfig, ResourcesConfig, SpeculativeConfig
from crispllm.memory.store import MemoryStore
from crispllm.proxy.app import create_app
from crispllm.proxy.classifier import TaskClassification, TaskClassifier
from crispllm.proxy.middleware import MemoryMiddleware, _get_message_text, estimate_tokens
from crispllm.proxy.speculative import (
    PredictedOutputsEngine,
    PredictionDecision,
    SpeculativeStats,
    _extract_text_from_message,
)
from crispllm.proxy.streaming import estimate_tokens_from_messages
from crispllm.proxy.watchdog import RamPressureWatchdog, WatchdogSnapshot


class TestStressTasksSuiteExpansion:
    def test_twenty_tasks_four_balanced_categories(self):
        stress_file = Path(__file__).parent.parent / "bench" / "stress_tasks.json"
        assert stress_file.exists()
        with open(stress_file, encoding="utf-8") as f:
            tasks = json.load(f)

        assert len(tasks) == 20
        cat_counts: dict[str, int] = {}
        for t in tasks:
            cat = t["category"]
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
            assert len(t["turns"]) >= 6
            file_chars = sum(len(c) for c in t["files"].values())
            turn_chars = sum(len(turn) for turn in t["turns"])
            est_tokens = (file_chars + turn_chars) // 4
            assert 3500 <= est_tokens <= 8500

        expected_categories = {
            "constraint_retention": 5,
            "contradiction_update": 5,
            "noisy_distractor": 5,
            "cross_turn_synthesis": 5,
        }
        assert cat_counts == expected_categories

    def test_dry_run_offline_deterministic_execution(self):
        stress_file = Path(__file__).parent.parent / "bench" / "stress_tasks.json"
        with open(stress_file, encoding="utf-8") as f:
            tasks = json.load(f)

        sample_subset = [tasks[0], tasks[5], tasks[10], tasks[15]]
        results = []
        for t in sample_subset:
            res = run_task(task=t, dry_run=True)
            assert res["passed"] is True
            assert res["error"] == ""
            assert res["usage_source"] == "dry_run"
            assert res["prompt_tokens"] > 0
            assert res["completion_tokens"] > 0
            results.append(res)

        summary = summarize_run_results(results)
        assert summary["sample_count"] == 4
        assert summary["passed_count"] == 4
        assert summary["pass_at_1_pct"] == 100.0


class TestFailOpenMiddlewareHardening:
    def test_get_message_text_safe_with_malformed_inputs(self):
        assert _get_message_text(None) == ""
        assert _get_message_text("plain string") == ""
        assert _get_message_text(12345) == ""
        assert _get_message_text([]) == ""
        assert _get_message_text({}) == ""
        assert _get_message_text({"content": None}) == ""
        assert _get_message_text({"content": 42}) == ""
        assert _get_message_text({"content": "valid text"}) == "valid text"
        assert _get_message_text({"content": [{"type": "text", "text": "part1"}]}) == "part1"
        assert _get_message_text({"content": ["part2", 123]}) == "part2"

    def test_derive_session_id_safe_with_malformed_messages(self):
        store = MemoryStore(":memory:")
        cfg = MemoryConfig(enabled=True)
        mw = MemoryMiddleware(store, cfg)
        sid = mw._derive_session_id([None, "invalid", {}, {"role": "user", "content": "hello"}], {})
        assert isinstance(sid, str) and len(sid) == 16
        store.close()

    def test_process_request_safe_with_database_locked_exception(self):
        store = MemoryStore(":memory:")
        cfg = MemoryConfig(enabled=True, auto_extract=True, compact_middle_turns=True)
        mw = MemoryMiddleware(store, cfg)

        req_body = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Important constraint: always use Python 3.11."},
                {"role": "assistant", "content": "Understood."},
                {"role": "user", "content": "Write settlement logic."},
            ]
        }

        with patch.object(store, "upsert_fact", side_effect=sqlite3.OperationalError("database is locked")):
            mod_body, telemetry = mw.process_request(req_body, {})
            assert "messages" in mod_body

        with patch.object(store, "add_episode", side_effect=sqlite3.OperationalError("database is locked")):
            mod_body, telemetry = mw.process_request(req_body, {})
            assert "messages" in mod_body

        with patch.object(store, "search_facts", side_effect=sqlite3.OperationalError("database is locked")):
            mod_body, telemetry = mw.process_request(req_body, {})
            assert mod_body == req_body
            assert telemetry["facts_extracted"] == 0

        store.close()


class TestFailOpenWatchdogHardening:
    def test_watchdog_probe_exception_fails_open_to_normal(self):
        def failing_probe():
            raise OSError("OS memory probe failed or permission denied")

        cfg = ResourcesConfig(enabled=True, watchdog_poll_s=0.0)
        wd = RamPressureWatchdog(config=cfg, probe_fn=failing_probe)
        snap = wd.snapshot(force=True)
        assert snap.pressure == "normal"
        assert snap.memory is None


class TestFailOpenClassifierHardening:
    def test_classifier_safe_with_non_list_or_malformed_messages(self):
        clf = TaskClassifier()
        res1 = clf.classify(None, est_tokens=100)
        assert res1.tier == "balanced"
        assert "malformed_payload" in res1.reasons

        res2 = clf.classify("not a list", est_tokens=100)
        assert res2.tier == "balanced"

        res3 = clf.classify([None, 123, "string", {}], est_tokens=50)
        assert res3.tier in ("fast", "balanced", "deep")


class TestFailOpenSpeculativeHardening:
    def test_speculative_extract_text_safe_with_malformed(self):
        assert _extract_text_from_message(None) == ""
        assert _extract_text_from_message(12345) == ""
        assert _extract_text_from_message([]) == ""

    def test_speculative_evaluate_safe_with_malformed_payload(self):
        cfg = SpeculativeConfig(predicted_outputs_enabled=True)
        engine = PredictedOutputsEngine(cfg)
        dec1 = engine.evaluate(None, target_is_cloud=True, base_url="https://api.openai.com/v1", model="gpt-4o")
        assert dec1.inject is False
        assert dec1.reason == "malformed_payload"

        dec2 = engine.evaluate({"messages": "not a list"}, target_is_cloud=True, base_url="https://api.openai.com/v1", model="gpt-4o")
        assert dec2.inject is False

    def test_speculative_stats_safe_with_invalid_records(self):
        stats = SpeculativeStats()
        stats.record_usage_payload(None)
        stats.record_usage_payload("string")
        stats.record_usage_payload({"usage": {"completion_tokens_details": {"accepted_prediction_tokens": "not_an_int"}}})
        stats.record_usage_payload({"usage": {"completion_tokens_details": {"accepted_prediction_tokens": True}}})
        stats.record_usage_payload({"usage": {"completion_tokens_details": {"accepted_prediction_tokens": -50}}})
        snap = stats.snapshot()
        assert snap["cloud_accepted_tokens"] == 0


class TestFailOpenStreamingHardening:
    def test_estimate_tokens_from_messages_safe_with_malformed(self):
        assert estimate_tokens_from_messages(None) == 0
        assert estimate_tokens_from_messages("not a list") == 0
        assert estimate_tokens_from_messages([]) == 0
        assert estimate_tokens_from_messages([None, "str", 123, object()]) >= 1


class TestFailOpenChatCompletionsEndpoint:
    def test_malformed_request_body_returns_http_400(self, tmp_path):
        cfg = EdgeMemConfig()
        cfg.memory.db_path = str(tmp_path / "edgemem.db")
        app = create_app(cfg)

        with TestClient(app) as tc:
            resp1 = tc.post(
                "/v1/chat/completions",
                content="not valid json {{{",
                headers={"Content-Type": "application/json"},
            )
            assert resp1.status_code == 400
            assert "error" in resp1.json()

            resp2 = tc.post(
                "/v1/chat/completions",
                json=["array_not_object"],
            )
            assert resp2.status_code == 400
            assert "error" in resp2.json()

    def test_database_locked_fails_open_with_http_200(self, tmp_path):
        import httpx

        cfg = EdgeMemConfig()
        cfg.memory.db_path = str(tmp_path / "edgemem.db")
        app = create_app(cfg)

        async def handler(request: httpx.Request) -> httpx.Response:
            mock_body = {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1234567,
                "model": "qwen2.5-coder:3b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Settlement verified."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 15, "completion_tokens": 5, "total_tokens": 20},
            }
            return httpx.Response(200, json=mock_body)

        mock_transport = httpx.MockTransport(handler)

        with TestClient(app) as tc:
            app.state.edgemem_http_client = httpx.AsyncClient(
                transport=mock_transport, base_url="http://localhost:11434/v1"
            )
            mem_store: MemoryStore = app.state.edgemem_memory_store

            with patch.object(mem_store, "list_facts", side_effect=sqlite3.OperationalError("database is locked")):
                resp = tc.post(
                    "/v1/chat/completions",
                    json={
                        "model": "qwen2.5-coder:3b",
                        "messages": [
                            {"role": "user", "content": "Execute settlement calculation."},
                        ],
                    },
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["choices"][0]["message"]["content"] == "Settlement verified."


class TestVersionSync:
    def test_version_string_and_cli_and_mcp(self):
        from click.testing import CliRunner
        from crispllm import __version__
        from crispllm.cli import cli
        from crispllm.proxy.mcp_server import create_mcp_server

        assert __version__ == "1.1.0"

        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "1.1.0" in result.output

        mcp = create_mcp_server()
        assert mcp.name == "CrispLLM"
        assert mcp.version == "1.1.0"

        doctor_res = runner.invoke(cli, ["doctor"])
        assert doctor_res.exit_code == 0
        assert "CrispLLM Doctor v1.1.0" in doctor_res.output
        assert "[OK] SQLite FTS5" in doctor_res.output
