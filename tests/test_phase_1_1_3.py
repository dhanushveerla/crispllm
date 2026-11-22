from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any
from unittest.mock import patch

from click.testing import CliRunner
import httpx
from starlette.testclient import TestClient

from crispllm.cli import cli
from crispllm.config import EdgeMemConfig, MemoryConfig, ResourcesConfig, RoutingConfig, load_config
from crispllm.memory.store import MemoryStore
from crispllm.proxy.app import create_app
from crispllm.proxy.budgeter import ContextBudgeter
from crispllm.proxy.classifier import TaskClassification, TaskClassifier
from crispllm.proxy.mcp_server import create_mcp_server
from crispllm.proxy.middleware import MemoryMiddleware
from crispllm.proxy.router import HybridRouter, RouteTarget
from crispllm.proxy.watchdog import RamPressureWatchdog, WatchdogSnapshot
from crispllm.slicer.cache import TagCache
from crispllm.slicer.repomap import RepoMap
from crispllm.slicer.symbols import SymbolDef, SymbolSliceResult, extract_symbols


class MockTransport(httpx.AsyncBaseTransport):
    def __init__(self, canned_response: dict[str, Any] | None = None) -> None:
        self.canned_response = canned_response or {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": "mock-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "OK response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        self.recorded_requests: list[httpx.Request] = []
        self.recorded_bodies: list[dict[str, Any]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.recorded_requests.append(request)
        body = json.loads(request.content.decode("utf-8")) if request.content else {}
        self.recorded_bodies.append(body)
        return httpx.Response(
            status_code=200,
            json=self.canned_response,
            headers={"content-type": "application/json"},
        )


def _make_test_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "test_repo"
    repo.mkdir(parents=True, exist_ok=True)

    auth_py = repo / "auth.py"
    auth_py.write_text(
        "import hashlib\n\n"
        "class TokenManager:\n"
        "    def __init__(self, salt: str = 'secret'):\n"
        "        self.salt = salt\n\n"
        "    def create_token(self, username: str) -> str:\n"
        "        data = f'{username}:{self.salt}'\n"
        "        return hashlib.sha256(data.encode()).hexdigest()\n\n"
        "    def verify_token(self, token: str, username: str) -> bool:\n"
        "        expected = self.create_token(username)\n"
        "        return token == expected\n\n"
        "def helper_hash(val: str) -> str:\n"
        "    return hashlib.md5(val.encode()).hexdigest()\n",
        encoding="utf-8",
    )

    service_py = repo / "service.py"
    service_py.write_text(
        "from auth import TokenManager, helper_hash\n\n"
        "class AuthService:\n"
        "    def __init__(self):\n"
        "        self.manager = TokenManager()\n\n"
        "    def login(self, username: str) -> str:\n"
        "        h = helper_hash(username)\n"
        "        return self.manager.create_token(username)\n",
        encoding="utf-8",
    )
    return repo


def test_config_extensions_defaults_and_toml(tmp_path: Path):
    cfg = EdgeMemConfig()
    assert cfg.routing.task_routing_enabled is False
    assert cfg.routing.fast_model is None
    assert cfg.routing.fast_max_tokens == 1200
    assert cfg.routing.escalate_deep_tasks is True
    assert cfg.memory.prefix_stable_assembly is True
    assert cfg.resources.llama_cache_prompt is True

    toml_path = tmp_path / "custom.toml"
    toml_path.write_text(
        "[routing]\n"
        "task_routing_enabled = true\n"
        "fast_model = 'fast-mini'\n"
        "fast_max_tokens = 900\n"
        "escalate_deep_tasks = false\n\n"
        "[memory]\n"
        "prefix_stable_assembly = false\n\n"
        "[resources]\n"
        "llama_cache_prompt = false\n",
        encoding="utf-8",
    )
    loaded = load_config(toml_path)
    assert loaded.routing.task_routing_enabled is True
    assert loaded.routing.fast_model == "fast-mini"
    assert loaded.routing.fast_max_tokens == 900
    assert loaded.routing.escalate_deep_tasks is False
    assert loaded.memory.prefix_stable_assembly is False
    assert loaded.resources.llama_cache_prompt is False


def test_ast_symbol_extractor(tmp_path: Path):
    repo = _make_test_repo(tmp_path)
    auth_py = repo / "auth.py"

    symbols = extract_symbols(auth_py, "auth.py")
    assert len(symbols) >= 3

    names = {s.name for s in symbols}
    assert "TokenManager" in names
    assert "create_token" in names
    assert "verify_token" in names
    assert "helper_hash" in names

    tm_sym = next(s for s in symbols if s.name == "TokenManager")
    assert tm_sym.kind == "class"
    assert tm_sym.file_path == "auth.py"

    ct_sym = next(s for s in symbols if s.name == "create_token")
    assert ct_sym.qualified_name == "TokenManager.create_token"
    assert ct_sym.kind == "method"
    assert "hashlib" in ct_sym.references or "sha256" in ct_sym.references

    vt_sym = next(s for s in symbols if s.name == "verify_token")
    assert "create_token" in vt_sym.references


def test_repomap_slice_symbol_found_callees_and_callers(tmp_path: Path):
    repo = _make_test_repo(tmp_path)

    with RepoMap(repo_path=repo, max_tokens=1024) as rm:
        res = rm.slice_symbol("create_token", token_budget=800)
        assert res.found is True
        assert res.fallback_used is False
        assert res.target_file == "auth.py"
        assert res.symbol == "create_token"
        assert "# Symbol: TokenManager.create_token" in res.content
        assert res.caller_count >= 1
        assert "# 1-Hop Callers" in res.content
        assert "verify_token" in res.content or "login" in res.content

        res_qname = rm.slice_symbol("TokenManager.create_token", token_budget=800)
        assert res_qname.found is True
        assert res_qname.target_file == "auth.py"

        res_case = rm.slice_symbol("CREATE_TOKEN", token_budget=800)
        assert res_case.found is True


def test_repomap_slice_symbol_not_found_fallback(tmp_path: Path):
    repo = _make_test_repo(tmp_path)

    with RepoMap(repo_path=repo, max_tokens=1024) as rm:
        res = rm.slice_symbol("non_existent_function_xyz", token_budget=600)
        assert res.found is False
        assert res.fallback_used is True
        assert res.target_file is None
        assert res.callee_count == 0
        assert res.caller_count == 0
        assert res.sliced_tokens > 0
        assert len(res.content) > 0


def test_cli_slice_symbol_flag(tmp_path: Path):
    repo = _make_test_repo(tmp_path)
    runner = CliRunner()

    res = runner.invoke(
        cli,
        ["slice", str(repo), "--symbol", "create_token", "--tokens", "500"],
    )
    assert res.exit_code == 0
    assert "TokenManager.create_token" in res.output
    assert "[CrispLLM slice-symbol]" in res.stderr or "[EdgeMem slice-symbol]" in res.stderr
    assert "symbol='create_token' found=True" in res.stderr


def test_mcp_repo_map_focus_symbol(tmp_path: Path):
    repo = _make_test_repo(tmp_path)
    cfg = EdgeMemConfig()
    server = create_mcp_server(config=cfg)

    async def _call():
        tool = server._tool_manager._tools.get("edgemem_repo_map")
        assert tool is not None
        result = await tool.fn(repo_path=str(repo), max_tokens=600, focus_symbol="create_token")
        return result

    data = asyncio.run(_call())
    assert data["found"] is True
    assert data["symbol"] == "create_token"
    assert data["target_file"] == "auth.py"
    assert data["caller_count"] >= 1
    assert "TokenManager.create_token" in data["repo_map"]


def test_task_classifier_heuristics():
    classifier = TaskClassifier()

    fast_res1 = classifier.classify(
        messages=[{"role": "user", "content": "Fix typo in variable name"}],
        est_tokens=20,
        fast_max_tokens=1200,
    )
    assert fast_res1.tier == "fast"
    assert fast_res1.confidence >= 0.8

    fast_res2 = classifier.classify(
        messages=[{"role": "user", "content": "What is the return type of calculate_sum?"}],
        est_tokens=30,
        fast_max_tokens=1200,
    )
    assert fast_res2.tier == "fast"

    deep_res1 = classifier.classify(
        messages=[{"role": "user", "content": "Please conduct an architectural redesign and security audit of the authentication module."}],
        est_tokens=150,
        fast_max_tokens=1200,
    )
    assert deep_res1.tier == "deep"
    assert deep_res1.confidence >= 0.85

    traceback_text = (
        "Traceback (most recent call last):\n"
        "  File 'server.py', line 42, in handle\n"
        "    raise ConnectionRefusedError('Connection dropped')\n"
        "ConnectionRefusedError: Connection dropped\n"
        "Fix this deadlock."
    )
    deep_res2 = classifier.classify(
        messages=[{"role": "user", "content": traceback_text}],
        est_tokens=200,
        fast_max_tokens=1200,
    )
    assert deep_res2.tier == "deep"
    assert "stack_trace_detected" in deep_res2.reasons

    balanced_res = classifier.classify(
        messages=[{"role": "user", "content": "Add a new method `get_status` that returns the status dictionary."}],
        est_tokens=400,
        fast_max_tokens=1200,
    )
    assert balanced_res.tier == "balanced"

    explicit_res = classifier.classify(
        messages=[{"role": "user", "content": "Anything"}],
        est_tokens=100,
        explicit_tier="deep",
    )
    assert explicit_res.tier == "deep"
    assert explicit_res.reasons == ("explicit_header",)

    t0 = time.perf_counter()
    for _ in range(100):
        classifier.classify(messages=[{"role": "user", "content": "Quick check for syntax"}], est_tokens=50)
    dur_ms = (time.perf_counter() - t0) * 1000
    assert dur_ms < 50.0


def test_hybrid_router_task_tier_composition(tmp_path: Path):
    cfg = EdgeMemConfig()
    cfg.routing.task_routing_enabled = True
    cfg.routing.fast_model = "qwen2.5-coder:1.5b"
    cfg.routing.escalate_deep_tasks = True
    cfg.cloud.enabled = True
    cfg.cloud.base_url = "https://api.cloud.example/v1"
    cfg.cloud.model = "cloud-deep-coder"

    router = HybridRouter(cfg)

    fast_body = {"messages": [{"role": "user", "content": "Fix typo in variable"}]}
    target_fast = router.select_target(fast_body, {}, est_prompt_tokens=50)
    assert target_fast.tier == "local"
    assert target_fast.model == "qwen2.5-coder:1.5b"
    assert target_fast.reason == "task_tier_fast"
    assert target_fast.task_tier == "fast"

    deep_body = {"messages": [{"role": "user", "content": "Architect a distributed consensus protocol with formal security audit"}]}
    target_deep = router.select_target(deep_body, {}, est_prompt_tokens=300)
    assert target_deep.tier == "cloud"
    assert target_deep.model == "cloud-deep-coder"
    assert "task_tier_deep" in target_deep.reason
    assert target_deep.task_tier == "deep"

    override_target = router.select_target(deep_body, {"X-EdgeMem-Route": "local"}, est_prompt_tokens=300)
    assert override_target.tier == "local"
    assert override_target.reason == "header_override"

    with patch.object(router.watchdog, "snapshot", return_value=WatchdogSnapshot(pressure="soft", memory=None, sampled_at=time.time())):
        balanced_body = {"messages": [{"role": "user", "content": "Add a simple logging statement in worker"}]}
        target_soft = router.select_target(balanced_body, {}, est_prompt_tokens=500)
        assert target_soft.tier == "local"
        assert target_soft.model == "qwen2.5-coder:1.5b"
        assert target_soft.reason == "soft_pressure_downshift"


def test_prefix_cache_stable_prompt_assembly(tmp_path: Path):
    db_file = tmp_path / "prefix_test.db"
    store = MemoryStore(db_file)
    store.upsert_fact("s1", "SYS_CONSTRAINT_1", "Must use Python 3.11+", kind="constraint", tier="core", pinned=True)
    store.upsert_fact("s1", "SYS_CONSTRAINT_2", "Database must be SQLite WAL mode", kind="constraint", tier="core", pinned=True)
    store.upsert_fact("s1", "EPISODIC_NOTE", "User preferred dark theme", kind="note", tier="episodic", pinned=False)

    cfg = MemoryConfig(enabled=True, prefix_stable_assembly=True, max_injected_tokens=500)
    middleware = MemoryMiddleware(store, cfg)

    turn1 = {
        "messages": [
            {"role": "system", "content": "Base instructions."},
            {"role": "user", "content": "Tell me about dark theme preferences"},
        ]
    }
    processed1, _ = middleware.process_request(turn1, {"x-edgemem-session": "s1"})

    first_content1 = processed1["messages"][0]["content"]
    assert "--- CrispLLM Core Memory ---" in first_content1 or "--- EdgeMem Core Memory ---" in first_content1
    assert "Must use Python 3.11+" in first_content1
    assert "Database must be SQLite WAL mode" in first_content1
    assert "dark theme" not in first_content1

    user_content1 = processed1["messages"][1]["content"]
    assert "<crispllm_context>" in user_content1 or "<edgemem_context>" in user_content1
    assert "dark theme" in user_content1

    turn2 = {
        "messages": [
            {"role": "system", "content": "Base instructions."},
            {"role": "user", "content": "Tell me about dark theme preferences"},
            {"role": "assistant", "content": "Dark theme recorded."},
            {"role": "user", "content": "Now explain SQLite WAL mode"},
        ]
    }
    processed2, _ = middleware.process_request(turn2, {"x-edgemem-session": "s1"})
    first_content2 = processed2["messages"][0]["content"]

    assert first_content1 == first_content2

    store.close()


def test_prefix_cache_hit_tracking_and_headers(tmp_path: Path):
    cfg = EdgeMemConfig()
    cfg.memory.db_path = str(tmp_path / "proxy_prefix.db")
    cfg.memory.prefix_stable_assembly = True

    app = create_app(cfg)
    mock_transport = MockTransport()

    with TestClient(app) as client:
        app.state.edgemem_http_client = httpx.AsyncClient(
            transport=mock_transport,
            base_url="http://mock-upstream/v1",
        )

        headers = {"X-EdgeMem-Session": "test-session-prefix-1"}
        req1 = {
            "messages": [
                {"role": "system", "content": "Fixed system instructions."},
                {"role": "user", "content": "Hello round 1"},
            ]
        }
        resp1 = client.post("/v1/chat/completions", json=req1, headers=headers)
        assert resp1.status_code == 200
        assert resp1.headers.get("X-EdgeMem-Prefix-Stable") == "miss"
        assert "X-EdgeMem-Task-Tier" in resp1.headers

        req2 = {
            "messages": [
                {"role": "system", "content": "Fixed system instructions."},
                {"role": "user", "content": "Hello round 1"},
                {"role": "assistant", "content": "Hello there"},
                {"role": "user", "content": "Hello round 2"},
            ]
        }
        resp2 = client.post("/v1/chat/completions", json=req2, headers=headers)
        assert resp2.status_code == 200
        assert resp2.headers.get("X-EdgeMem-Prefix-Stable") == "hit"

        h_resp = client.get("/health")
        assert h_resp.status_code == 200
        h_data = h_resp.json()
        assert "prefix_cache" in h_data
        assert h_data["prefix_cache"]["hits"] >= 1
        assert h_data["prefix_cache"]["misses"] >= 1
        assert "task_tiers" in h_data


def test_llama_cache_prompt_injection(tmp_path: Path):
    cfg = EdgeMemConfig()
    cfg.memory.db_path = str(tmp_path / "llama_cache.db")
    cfg.resources.llama_cache_prompt = True
    cfg.resources.local_backend = "llama_server"

    app = create_app(cfg)
    mock_transport = MockTransport()

    with TestClient(app) as client:
        app.state.edgemem_http_client = httpx.AsyncClient(
            transport=mock_transport,
            base_url="http://mock-upstream/v1",
        )

        req = {
            "messages": [
                {"role": "user", "content": "Hello llama"},
            ]
        }
        resp = client.post("/v1/chat/completions", json=req)
        assert resp.status_code == 200
        assert len(mock_transport.recorded_bodies) == 1
        body_sent = mock_transport.recorded_bodies[0]
        assert body_sent.get("cache_prompt") is True


def test_generic_callee_filter(tmp_path: Path):
    repo = tmp_path / "generic_repo"
    repo.mkdir(parents=True, exist_ok=True)
    cache_py = repo / "cache.py"
    cache_py.write_text(
        "class Cache:\n"
        "    def get(self, key: str) -> str:\n"
        "        return key\n"
        "    def close(self) -> None:\n"
        "        pass\n",
        encoding="utf-8",
    )
    worker_py = repo / "worker.py"
    worker_py.write_text(
        "def run_worker(data: dict) -> str:\n"
        "    val = data.get('foo')\n"
        "    return str(val)\n",
        encoding="utf-8",
    )
    with RepoMap(repo_path=repo, max_tokens=1024) as rm:
        res = rm.slice_symbol("run_worker", token_budget=800)
        assert res.found is True
        assert "Cache.get" not in res.content
        assert "cache.py" not in res.content
        assert res.callee_count == 0

        res_get = rm.slice_symbol("get", token_budget=800)
        assert res_get.found is True
        assert res_get.target_file == "cache.py"


def test_import_aware_priority(tmp_path: Path):
    repo = tmp_path / "priority_repo"
    repo.mkdir(parents=True, exist_ok=True)
    imported_mod = repo / "imported_mod.py"
    imported_mod.write_text(
        "def imported_func() -> str:\n"
        "    return 'imported'\n",
        encoding="utf-8",
    )
    unimported_mod = repo / "unimported_mod.py"
    unimported_mod.write_text(
        "def unimported_func() -> str:\n"
        "    return 'unimported'\n",
        encoding="utf-8",
    )
    main_py = repo / "main.py"
    main_py.write_text(
        "from imported_mod import imported_func\n\n"
        "def local_helper() -> str:\n"
        "    return 'local'\n\n"
        "def execute_task() -> str:\n"
        "    a = local_helper()\n"
        "    b = imported_func()\n"
        "    c = unimported_func()\n"
        "    return a + b + c\n",
        encoding="utf-8",
    )
    with RepoMap(repo_path=repo, max_tokens=1024) as rm:
        res = rm.slice_symbol("execute_task", token_budget=800)
        assert res.found is True
        callee_section = res.content.split("# 1-Hop Callees")[-1]
        local_idx = callee_section.find("local_helper")
        imported_idx = callee_section.find("imported_func")
        unimported_idx = callee_section.find("unimported_func")
        assert local_idx != -1
        assert imported_idx != -1
        assert local_idx < imported_idx
        if unimported_idx != -1:
            assert imported_idx < unimported_idx


def test_deduplication_and_multi_line_signature(tmp_path: Path):
    repo = tmp_path / "sig_repo"
    repo.mkdir(parents=True, exist_ok=True)
    lib_py = repo / "lib.py"
    lib_py.write_text(
        "def multi_line_target(\n"
        "    first_arg: int,\n"
        "    second_arg: str = 'default',\n"
        ") -> bool:\n"
        "    return True\n",
        encoding="utf-8",
    )
    caller_py = repo / "caller.py"
    caller_py.write_text(
        "from lib import multi_line_target\n\n"
        "def run1() -> None:\n"
        "    multi_line_target(1, 'a')\n"
        "    multi_line_target(2, 'b')\n",
        encoding="utf-8",
    )
    with RepoMap(repo_path=repo, max_tokens=1024) as rm:
        res = rm.slice_symbol("run1", token_budget=800)
        assert res.found is True
        assert "def multi_line_target(first_arg: int, second_arg: str = 'default') -> bool" in res.content
        assert res.callee_count == 1
        callee_section = res.content.split("# 1-Hop Callees")[-1]
        assert callee_section.count("multi_line_target") == 1


def test_caller_hygiene(tmp_path: Path):
    repo = tmp_path / "hygiene_repo"
    repo.mkdir(parents=True, exist_ok=True)
    service_py = repo / "service.py"
    service_py.write_text(
        "class OrderService:\n"
        "    def process_order(self, order_id: str) -> bool:\n"
        "        return len(order_id) > 0\n",
        encoding="utf-8",
    )
    routes_py = repo / "routes.py"
    routes_py.write_text(
        "from service import OrderService\n\n"
        "def dummy_decorator(fn):\n"
        "    return fn\n\n"
        "@dummy_decorator\n"
        "def submit_order_route(order_id: str) -> bool:\n"
        "    svc = OrderService()\n"
        "    return svc.process_order(order_id)\n",
        encoding="utf-8",
    )
    with RepoMap(repo_path=repo, max_tokens=1024) as rm:
        res = rm.slice_symbol("OrderService.process_order", token_budget=800)
        assert res.found is True
        assert "OrderService:" not in res.content.split("# 1-Hop Callers")[-1]
        assert "@dummy_decorator" not in res.content
        assert "submit_order_route" in res.content
        assert "svc.process_order(order_id)" in res.content


def test_warm_cache_reuse(tmp_path: Path):
    repo = _make_test_repo(tmp_path)
    tag_cache = TagCache(db_path=tmp_path / "warm_test.db")
    with RepoMap(repo_path=repo, max_tokens=1024, cache=tag_cache) as rm:
        res1 = rm.slice_symbol("create_token", token_budget=800)
        assert res1.found is True

        with patch("crispllm.slicer.symbols._get_parser") as mock_get_parser:
            t0 = time.perf_counter()
            res2 = rm.slice_symbol("create_token", token_budget=800)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            assert mock_get_parser.call_count == 0
            assert res2.found is True
            assert res2.target_file == "auth.py"
            assert elapsed_ms < 200.0
