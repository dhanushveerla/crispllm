from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from crispllm.cli import cli
from crispllm.config import EdgeMemConfig, MemoryConfig
from crispllm.memory.consolidator import (
    ConsolidationReport,
    ConsolidationWorker,
    MemoryConsolidator,
    _lexical_similarity,
)
from crispllm.memory.store import Fact, MemoryStore, compute_effective_salience
from crispllm.proxy.app import create_app
from crispllm.proxy.mcp_server import create_mcp_server
from crispllm.proxy.middleware import MemoryMiddleware


def test_legacy_v1_schema_migration_and_consolidation_runs(tmp_path: Path):
    db_file = tmp_path / "legacy_v1.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute(
        """
        CREATE TABLE facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            scope TEXT NOT NULL,
            kind TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            source_turn INTEGER DEFAULT 0,
            pinned INTEGER DEFAULT 0,
            created_at REAL,
            updated_at REAL,
            UNIQUE(session_id, kind, key)
        );
        """
    )
    conn.execute(
        """
        INSERT INTO facts (session_id, scope, kind, key, value, source_turn, pinned, created_at, updated_at)
        VALUES ('sess_leg', 'session', 'constant', 'PINNED_KEY', 'pinned_val', 1, 1, 100.0, 100.0);
        """
    )
    conn.execute(
        """
        INSERT INTO facts (session_id, scope, kind, key, value, source_turn, pinned, created_at, updated_at)
        VALUES ('sess_leg', 'session', 'note', 'UNPINNED_KEY', 'unpinned_val', 2, 0, 110.0, 110.0);
        """
    )
    conn.commit()
    conn.close()

    with MemoryStore(db_file) as store:
        counts = store.tier_counts()
        assert counts["core"] == 1
        assert counts["episodic"] == 1
        assert counts["archival"] == 0
        assert counts["superseded"] == 0

        pinned_fact = store.get_pinned_facts("sess_leg")
        assert len(pinned_fact) == 1
        assert pinned_fact[0].key == "PINNED_KEY"
        assert pinned_fact[0].tier == "core"
        assert pinned_fact[0].source == "user"
        assert pinned_fact[0].last_accessed_at == 100.0

        all_facts = store.list_facts("sess_leg")
        by_key = {f.key: f for f in all_facts}
        assert by_key["UNPINNED_KEY"].tier == "episodic"
        assert by_key["UNPINNED_KEY"].source == "auto"
        assert by_key["UNPINNED_KEY"].last_accessed_at == 110.0

        cr_cur = store._conn.execute("SELECT COUNT(*) FROM consolidation_runs;")
        assert cr_cur.fetchone()[0] == 0


def test_non_destructive_contradiction_resolution(tmp_path: Path):
    db_file = tmp_path / "contradiction.db"
    with MemoryStore(db_file) as store:
        fid1 = store.upsert_fact(
            "sess_contra", "DATABASE_PORT", "5432", kind="constant", scope="session"
        )
        assert store.stats()["facts_count"] == 1

        search1 = store.search_facts("5432", session_id="sess_contra")
        assert len(search1) == 1
        assert search1[0].value == "5432"
        assert search1[0].status == "active"

        fid2 = store.upsert_fact(
            "sess_contra", "DATABASE_PORT", "5433", kind="constant", scope="session"
        )
        assert fid2 != fid1
        assert store.stats()["facts_count"] == 1

        search_old = store.search_facts("5432", session_id="sess_contra")
        assert len(search_old) == 0

        search_new = store.search_facts("5433", session_id="sess_contra")
        assert len(search_new) == 1
        assert search_new[0].id == fid2
        assert search_new[0].value == "5433"
        assert search_new[0].status == "active"

        history = store.get_fact_history("DATABASE_PORT", session_id="sess_contra")
        assert len(history) == 2
        assert history[0].id == fid1
        assert history[0].value == "5432"
        assert history[0].status == "superseded"
        assert history[0].superseded_by == fid2
        assert history[0].valid_until is not None

        assert history[1].id == fid2
        assert history[1].value == "5433"
        assert history[1].status == "active"
        assert history[1].superseded_by is None


def test_trust_guard_protects_core_and_user_facts(tmp_path: Path):
    db_file = tmp_path / "trust.db"
    with MemoryStore(db_file) as store:
        core_id = store.upsert_fact(
            "sess_trust",
            "ARCHITECTURE",
            "microservices",
            kind="decision",
            tier="core",
            source="user",
            pinned=True,
        )

        attempt_id = store.upsert_fact(
            "sess_trust",
            "ARCHITECTURE",
            "monolith",
            kind="decision",
            tier="episodic",
            source="auto",
            pinned=False,
        )
        assert attempt_id == core_id

        active_facts = store.list_facts("sess_trust")
        assert len(active_facts) == 1
        assert active_facts[0].id == core_id
        assert active_facts[0].value == "microservices"
        assert active_facts[0].tier == "core"

        user_update_id = store.upsert_fact(
            "sess_trust",
            "ARCHITECTURE",
            "modular_monolith",
            kind="decision",
            tier="core",
            source="user",
            pinned=True,
        )
        assert user_update_id != core_id

        history = store.get_fact_history("ARCHITECTURE", session_id="sess_trust")
        assert len(history) == 2
        assert history[0].id == core_id
        assert history[0].status == "superseded"
        assert history[0].superseded_by == user_update_id
        assert history[1].id == user_update_id
        assert history[1].status == "active"


def test_corroboration_in_place(tmp_path: Path):
    db_file = tmp_path / "corroborate.db"
    with MemoryStore(db_file) as store:
        fid1 = store.upsert_fact(
            "sess_corr",
            "MAX_PAYLOAD_MB",
            "25",
            kind="constant",
            salience=1.0,
        )
        initial_fact = store.list_facts("sess_corr")[0]
        assert initial_fact.access_count == 0
        assert initial_fact.salience == 1.0

        fid2 = store.upsert_fact(
            "sess_corr",
            "MAX_PAYLOAD_MB",
            "  25  ",
            kind="constant",
        )
        assert fid2 == fid1

        updated_facts = store.list_facts("sess_corr")
        assert len(updated_facts) == 1
        assert updated_facts[0].id == fid1
        assert updated_facts[0].access_count == 1
        assert pytest.approx(updated_facts[0].salience, 0.001) == 1.15
        assert updated_facts[0].last_accessed_at is not None


def test_compute_effective_salience_formula():
    sal = compute_effective_salience(
        salience=1.0, age_days=0.0, access_count=0, half_life_days=14.0
    )
    assert pytest.approx(sal, 0.001) == 1.0

    sal_half = compute_effective_salience(
        salience=1.0, age_days=14.0, access_count=0, half_life_days=14.0
    )
    assert pytest.approx(sal_half, 0.001) == 0.5

    sal_accessed = compute_effective_salience(
        salience=1.0, age_days=0.0, access_count=10, half_life_days=14.0
    )
    expected = 1.0 * (1.0 + 0.15 * math.log1p(10))
    assert pytest.approx(sal_accessed, 0.001) == expected


def test_middleware_two_phase_injection_and_decay_ranking(tmp_path: Path):
    db_file = tmp_path / "middleware_decay.db"
    now_ts = 1_000_000.0

    with MemoryStore(db_file) as store:
        store.upsert_fact(
            "s1",
            "CORE_RULE",
            "core strict constraint",
            kind="constraint",
            tier="core",
            pinned=True,
            salience=1.0,
        )
        store.upsert_fact(
            "s1",
            "FREQUENT_EPISODIC",
            "frequent episodic memory item",
            kind="note",
            tier="episodic",
            salience=1.5,
        )
        store.upsert_fact(
            "s1",
            "OLD_EPISODIC",
            "old stale episodic memory item",
            kind="note",
            tier="episodic",
            salience=1.0,
        )
        store._conn.execute(
            "UPDATE facts SET access_count = 10, last_accessed_at = ? WHERE key = 'FREQUENT_EPISODIC';",
            (now_ts,),
        )
        store._conn.execute(
            "UPDATE facts SET access_count = 0, last_accessed_at = ? WHERE key = 'OLD_EPISODIC';",
            (now_ts - (30.0 * 86400.0),),
        )
        store._conn.commit()

        cfg = MemoryConfig(enabled=True, max_injected_tokens=200, decay_half_life_days=14.0)
        middleware = MemoryMiddleware(store, cfg)

        req_body = {
            "messages": [
                {"role": "system", "content": "Assistant prompt."},
                {"role": "user", "content": "Tell me about episodic memory items and constraints."},
            ]
        }

        with patch("time.time", return_value=now_ts):
            processed, tel = middleware.process_request(req_body, {"x-edgemem-session": "s1"})

        injected_content = processed["messages"][0]["content"]
        user_content = processed["messages"][1]["content"] if len(processed["messages"]) > 1 else ""
        assert "core strict constraint" in injected_content
        assert "frequent episodic memory item" in (injected_content + user_content)

        accessed_fact = store.search_facts("frequent", session_id="s1")[0]
        assert accessed_fact.access_count == 11


def test_consolidator_sweep_dedup_promotion_demotion(tmp_path: Path):
    db_file = tmp_path / "consolidator.db"
    now_ts = 2_000_000.0

    with MemoryStore(db_file) as store:
        store.upsert_fact(
            "s_sweep", "CACHE_TTL", "300 seconds", kind="constant", salience=1.0
        )
        store.upsert_fact(
            "s_sweep", "CACHE_TTL_DUP", "300 seconds", kind="constant", salience=1.0
        )
        store._conn.execute(
            "UPDATE facts SET key = 'CACHE_TTL' WHERE key = 'CACHE_TTL_DUP';"
        )

        p_id = store.upsert_fact(
            "s_sweep", "PROMOTED_KEY", "high value fact", kind="note", salience=1.5
        )
        store._conn.execute(
            "UPDATE facts SET access_count = 6, last_accessed_at = ? WHERE id = ?;",
            (now_ts, p_id),
        )

        d_id = store.upsert_fact(
            "s_sweep", "STALE_KEY", "forgotten fact", kind="note", salience=0.5
        )
        store._conn.execute(
            "UPDATE facts SET last_accessed_at = ?, updated_at = ? WHERE id = ?;",
            (now_ts - (20.0 * 86400.0), now_ts - (20.0 * 86400.0), d_id),
        )
        store._conn.commit()

        cfg = EdgeMemConfig()
        cfg.memory.promote_access_threshold = 5
        cfg.memory.archive_threshold = 0.25
        cfg.memory.decay_half_life_days = 7.0

        consolidator = MemoryConsolidator(store, cfg)

        dry_report = consolidator.consolidate_once(dry_run=True, now=now_ts)
        assert dry_report.dry_run is True
        assert dry_report.merged == 1
        assert dry_report.promoted == 1
        assert dry_report.demoted == 1

        unmodified_counts = store.tier_counts()
        assert unmodified_counts["episodic"] >= 3
        assert unmodified_counts["archival"] == 0

        real_report = consolidator.consolidate_once(dry_run=False, now=now_ts)
        assert real_report.dry_run is False
        assert real_report.merged == 1
        assert real_report.promoted == 1
        assert real_report.demoted == 1

        final_counts = store.tier_counts()
        assert final_counts["core"] >= 1
        assert final_counts["archival"] == 1
        assert final_counts["superseded"] >= 1

        stale_item = store.list_facts("s_sweep", include_archival=True)
        by_k = {f.key: f for f in stale_item}
        assert by_k["STALE_KEY"].tier == "archival"
        assert by_k["STALE_KEY"].status == "archived"
        assert by_k["PROMOTED_KEY"].tier == "core"

        cr_row = store._conn.execute(
            "SELECT merged_count, promoted_count, demoted_count FROM consolidation_runs;"
        ).fetchone()
        assert cr_row[0] == 1
        assert cr_row[1] == 1
        assert cr_row[2] == 1


@pytest.mark.anyio
async def test_llm_synthesis_fail_open_and_success(tmp_path: Path):
    db_file = tmp_path / "llm_syn.db"
    with MemoryStore(db_file) as store:
        store.add_episode("s_syn", "user", "Please use Python 3.11 features throughout.")
        store.add_episode("s_syn", "assistant", "Acknowledged, targeting Python 3.11.")

        cfg = EdgeMemConfig()
        cfg.memory.llm_synthesis = True

        consolidator = MemoryConsolidator(store, cfg)

        with patch("httpx.AsyncClient.post", side_effect=httpx.ConnectTimeout("timeout")):
            report_timeout = await consolidator.consolidate_with_llm_async()
            assert isinstance(report_timeout, ConsolidationReport)

        fake_resp_bad = MagicMock()
        fake_resp_bad.status_code = 200
        fake_resp_bad.json.return_value = {
            "choices": [{"message": {"content": "Not valid json at all"}}]
        }
        with patch("httpx.AsyncClient.post", return_value=fake_resp_bad):
            report_bad = await consolidator.consolidate_with_llm_async()
            assert isinstance(report_bad, ConsolidationReport)

        fake_resp_ok = MagicMock()
        fake_resp_ok.status_code = 200
        fake_resp_ok.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            [
                                {
                                    "op": "CREATE",
                                    "key": "PYTHON_VERSION",
                                    "value": "3.11",
                                    "tier": "core",
                                }
                            ]
                        )
                    }
                }
            ]
        }
        with patch("httpx.AsyncClient.post", return_value=fake_resp_ok):
            report_ok = await consolidator.consolidate_with_llm_async()
            assert report_ok.mode == "llm_synthesis"
            assert report_ok.created == 1

        created = store.get_pinned_facts("global")
        keys = [f.key for f in created]
        assert "PYTHON_VERSION" in keys


@pytest.mark.anyio
async def test_consolidation_worker_lifecycle_and_clean_shutdown(tmp_path: Path):
    db_file = tmp_path / "worker_lifecycle.db"
    store = MemoryStore(db_file)
    store.upsert_fact("s_w", "WORKER_KEY", "123")

    cfg = EdgeMemConfig()
    cfg.memory.consolidation_every_n_turns = 3
    cfg.memory.consolidation_idle_s = 0.5

    consolidator = MemoryConsolidator(store, cfg)
    worker = ConsolidationWorker(consolidator, cfg)
    worker.start()

    worker.notify_turn()
    worker.notify_turn()
    assert worker.pending_turns == 2
    assert not worker._event.is_set()

    worker.notify_turn()
    assert worker._event.is_set()

    await asyncio.sleep(0.1)

    await worker.stop()
    store.close()

    db_file.unlink()
    assert not db_file.exists()


def test_cli_memory_new_commands_and_ascii(tmp_path: Path):
    db_file = tmp_path / "cli_tier.db"
    cfg_file = tmp_path / "edgemem.toml"
    cfg_file.write_text(
        f"""
[server]
host = "127.0.0.1"
port = 8080

[memory]
enabled = true
db_path = "{db_file.as_posix()}"
""",
        encoding="utf-8",
    )

    runner = CliRunner()

    with MemoryStore(db_file) as store:
        store.upsert_fact("s_cli", "PORT", "8080", tier="core", pinned=True)
        store.upsert_fact("s_cli", "CACHE_KEY", "redis", tier="episodic", pinned=False)
        store.upsert_fact("s_cli", "PORT", "9090", tier="core", pinned=True)

    res_list_core = runner.invoke(
        cli, ["memory", "list", "--tier", "core", "--config", str(cfg_file)]
    )
    assert res_list_core.exit_code == 0
    assert "PORT" in res_list_core.output
    assert "[core/active]" in res_list_core.output
    assert "CACHE_KEY" not in res_list_core.output

    res_history = runner.invoke(
        cli, ["memory", "history", "PORT", "--config", str(cfg_file)]
    )
    assert res_history.exit_code == 0
    assert "8080" in res_history.output
    assert "9090" in res_history.output
    assert "superseded_by:" in res_history.output

    res_search_all = runner.invoke(
        cli, ["memory", "search", "PORT", "--all", "--config", str(cfg_file)]
    )
    assert res_search_all.exit_code == 0
    assert "PORT" in res_search_all.output

    res_consolidate = runner.invoke(
        cli, ["memory", "consolidate", "--dry-run", "--config", str(cfg_file)]
    )
    assert res_consolidate.exit_code == 0
    assert "[DRY-RUN] Consolidation complete" in res_consolidate.output


def test_mcp_and_health_integration_tier_support(tmp_path: Path):
    app, _, _ = _make_test_proxy_app(tmp_path)

    with TestClient(app) as client:
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        health_data = health_resp.json()
        assert "memory_tiers" in health_data
        assert "core" in health_data["memory_tiers"]
        assert "episodic" in health_data["memory_tiers"]
        assert "archival" in health_data["memory_tiers"]
        assert "superseded" in health_data["memory_tiers"]

        init_resp = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1.0"},
                },
            },
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        assert init_resp.status_code == 200
        session_id = init_resp.headers.get("mcp-session-id", "")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if session_id:
            headers["mcp-session-id"] = session_id

        client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            headers=headers,
        )

        rem_resp = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "edgemem_remember",
                    "arguments": {
                        "key": "TIER_TEST_KEY",
                        "value": "core_important_val",
                        "tier": "core",
                    },
                },
            },
            headers=headers,
        )
        assert rem_resp.status_code == 200

        recall_resp = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "edgemem_recall",
                    "arguments": {"query": "TIER_TEST_KEY", "include_archival": True},
                },
            },
            headers=headers,
        )
        assert recall_resp.status_code == 200
        content = json.loads(recall_resp.json()["result"]["content"][0]["text"])
        assert len(content["facts"]) >= 1
        assert content["facts"][0]["key"] == "TIER_TEST_KEY"
        assert content["facts"][0]["tier"] == "core"


def _make_test_proxy_app(tmp_path: Path):
    db_file = tmp_path / "test_app_mem.db"
    tel_dir = tmp_path / "tel"
    tel_dir.mkdir(parents=True, exist_ok=True)
    cfg = EdgeMemConfig()
    cfg.memory.enabled = True
    cfg.memory.db_path = str(db_file)
    cfg.telemetry_dir = str(tel_dir)
    cfg.memory.consolidation_enabled = True

    app = create_app(cfg)
    return app, None, cfg


class _SyntheticMockTransport(httpx.AsyncBaseTransport):
    def __init__(self, response_content: str = "Acknowledged.") -> None:
        self.response_content = response_content

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1234567,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": self.response_content,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
            headers={"content-type": "application/json"},
        )


def _make_mock_proxy_app(tmp_path: Path, response_content: str = "Acknowledged."):
    db_file = tmp_path / "test_app_sec_mem.db"
    tel_dir = tmp_path / "tel_sec"
    tel_dir.mkdir(parents=True, exist_ok=True)
    cfg = EdgeMemConfig()
    cfg.memory.enabled = True
    cfg.memory.db_path = str(db_file)
    cfg.telemetry_dir = str(tel_dir)
    cfg.memory.auto_extract = True
    cfg.memory.compact_middle_turns = True
    cfg.memory.compact_threshold_tokens = 0
    cfg.memory.compact_min_turn_tokens = 0
    cfg.memory.keep_recent_turns = 2

    transport = _SyntheticMockTransport(response_content=response_content)
    app = create_app(cfg)
    orig_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def patched_lifespan(application):
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
    return app, cfg, db_file


def test_security_localhost_route_blocks_secrets_from_memory(tmp_path: Path):
    app, cfg, _ = _make_mock_proxy_app(tmp_path)
    secret_key = "sk-proj-testsecretsupersecret1234567890"
    raw_db_pass = "super_secret_db_pass_999"
    db_url = f"postgres://admin:{raw_db_pass}@localhost:5432/mydb"
    boilerplate = ("line of domain logic\n" * 50)

    messages = [
        {"role": "system", "content": "You are a coding assistant."},
        {
            "role": "user",
            "content": f"Database config constraint: must use {db_url} and API_KEY = '{secret_key}'.\n{boilerplate}",
        },
        {"role": "assistant", "content": f"Acknowledged database config.\n{boilerplate}"},
        {"role": "user", "content": f"Please add a healthcheck route.\n{boilerplate}"},
        {"role": "assistant", "content": f"Healthcheck added.\n{boilerplate}"},
        {"role": "user", "content": "What is our architecture constraint?"},
    ]

    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": cfg.model, "messages": messages, "stream": False},
            headers={"x-edgemem-session": "sec_sess_1"},
        )
        assert resp.status_code == 200

        store = app.state.edgemem_memory_store
        facts = store.list_facts("sec_sess_1", limit=100)
        assert len(facts) >= 1
        for f in facts:
            assert secret_key not in f.value
            assert secret_key not in f.key
            assert raw_db_pass not in f.value
            assert raw_db_pass not in f.key

        episodes = store.list_episodes("sec_sess_1", limit=100)
        assert len(episodes) >= 1
        for ep in episodes:
            assert secret_key not in ep.summary
            assert raw_db_pass not in ep.summary


@pytest.mark.anyio
async def test_security_restored_assistant_secret_never_persisted(tmp_path: Path):
    app, cfg, _ = _make_mock_proxy_app(tmp_path)
    store = app.state.edgemem_memory_store
    restored_key = "sk-live-assistantrestoredsecretkey12345678"
    restored_db_pass = "cloud_db_restored_pass_888"
    restored_url = f"postgres://clouduser:{restored_db_pass}@cloud.db:5432/main"
    boilerplate = ("system operational logic\n" * 50)

    messages = [
        {"role": "system", "content": "Assistant prompt."},
        {"role": "user", "content": f"Setup connection.\n{boilerplate}"},
        {
            "role": "assistant",
            "content": f"Restored credentials: must use API_KEY = '{restored_key}' and {restored_url}.\n{boilerplate}",
        },
        {"role": "user", "content": "Proceed with deployment."},
    ]

    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": cfg.model, "messages": messages, "stream": False},
            headers={"x-edgemem-session": "sec_sess_2"},
        )
        assert resp.status_code == 200

        store = app.state.edgemem_memory_store
        facts = store.list_facts("sec_sess_2", limit=100)
        for f in facts:
            assert restored_key not in f.value
            assert restored_key not in f.key
            assert restored_db_pass not in f.value
            assert restored_db_pass not in f.key

        episodes = store.list_episodes("sec_sess_2", limit=100)
        for ep in episodes:
            assert restored_key not in ep.summary
            assert restored_db_pass not in ep.summary

        consolidator = MemoryConsolidator(store, cfg)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            [
                                {
                                    "op": "CREATE",
                                    "key": "RESTORED_KEY",
                                    "value": restored_key,
                                    "tier": "core",
                                },
                                {
                                    "op": "CREATE",
                                    "key": "DB_URL",
                                    "value": restored_url,
                                    "tier": "core",
                                },
                            ]
                        )
                    }
                }
            ]
        }
        with patch("httpx.AsyncClient.post", return_value=mock_resp):
            cfg.memory.llm_synthesis = True
            await consolidator.consolidate_with_llm_async()

        global_facts = store.list_facts("global", limit=100)
        for gf in global_facts:
            assert restored_key not in gf.value
            assert restored_key not in gf.key
            assert restored_db_pass not in gf.value
            assert restored_db_pass not in gf.key

