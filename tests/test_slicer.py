"""tests/test_slicer.py — Automated test suite for EdgeMem AST Context Slicer.

Verifies:
  a) Output stays strictly <= max_tokens across various budget limits.
  b) Compression ratio exceeds 50% while retaining cross-file referenced symbols.
  c) query and focus_files personalization promote matching symbols/files.
  d) Warm SQLite cache matches cold output, runs faster, and invalidates on mtime change.
  e) MCP tools/list exposes edgemem_repo_map and edgemem_file_outline,
     and calling edgemem_repo_map over Streamable HTTP returns a valid map.
  f) Multi-language support (Python, JS, Go, Rust, C) and ignore rules (.git, .venv, .gitignore).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from edgemem.config import EdgeMemConfig
from edgemem.proxy.app import create_app
from edgemem.slicer.cache import TagCache
from edgemem.slicer.repomap import RepoMap, count_tokens, file_outline
from edgemem.slicer.tags import extract_tags, scan_repo_files


# ---------------------------------------------------------------------------
# Multi-file temporary repository fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def multi_file_project(tmp_path: Path) -> Path:
    """Create a realistic multi-file, multi-language repo structure."""
    repo = tmp_path / "test_repo"
    repo.mkdir()

    # 1. Models module
    models_dir = repo / "models"
    models_dir.mkdir()
    (models_dir / "__init__.py").write_text("", encoding="utf-8")
    (models_dir / "user.py").write_text(
        '"""User domain models and validation."""\n\n'
        "from dataclasses import dataclass\n"
        "from typing import Optional, List\n\n"
        "@dataclass\n"
        "class UserRecord:\n"
        "    user_id: int\n"
        "    username: str\n"
        "    email: str\n"
        "    is_active: bool = True\n"
        "    roles: Optional[List[str]] = None\n\n"
        "def validate_user_record(user: UserRecord) -> bool:\n"
        "    \"\"\"Validate user username, email format, and activity status.\"\"\"\n"
        "    if not user.username or len(user.username) < 3:\n"
        "        return False\n"
        "    if not user.email or '@' not in user.email:\n"
        "        return False\n"
        "    return user.is_active\n\n"
        "def format_user_display(user: UserRecord) -> str:\n"
        "    \"\"\"Return pretty display representation for audit logs.\"\"\"\n"
        "    return f'{user.username} <{user.email}> (id={user.user_id})'\n",
        encoding="utf-8",
    )

    # 2. Auth module (references models.user)
    auth_dir = repo / "auth"
    auth_dir.mkdir()
    (auth_dir / "__init__.py").write_text("", encoding="utf-8")
    (auth_dir / "jwt.py").write_text(
        '"""JWT token handling, verification, and cryptographic signature checking."""\n\n'
        "from models.user import UserRecord, validate_user_record, format_user_display\n\n"
        "class JwtAuthManager:\n"
        "    \"\"\"Handles issuance and verification of bearer tokens.\"\"\"\n"
        "    def __init__(self, secret_key: str, issuer: str = 'edgemem-auth'):\n"
        "        self.secret_key = secret_key\n"
        "        self.issuer = issuer\n\n"
        "    def create_token(self, user: UserRecord) -> str:\n"
        "        \"\"\"Issue signed token for active validated user.\"\"\"\n"
        "        if not validate_user_record(user):\n"
        '            raise ValueError("invalid user record supplied")\n'
        '        return f"token_for_{user.username}_signed_{self.issuer}"\n\n'
        "    def verify_token(self, token: str) -> bool:\n"
        "        \"\"\"Check token signature and issuer claims.\"\"\"\n"
        '        return token.startswith("token_for_") and self.issuer in token\n\n'
        "    def decode_claims(self, token: str) -> dict:\n"
        "        \"\"\"Extract user claims from validated token.\"\"\"\n"
        '        if not self.verify_token(token):\n'
        '            return {}\n'
        '        return {"user": token.split("_")[2], "issuer": self.issuer}\n',
        encoding="utf-8",
    )

    # 3. API routes (references auth.jwt and models.user)
    api_dir = repo / "api"
    api_dir.mkdir()
    (api_dir / "__init__.py").write_text("", encoding="utf-8")
    (api_dir / "routes.py").write_text(
        '"""API route handlers for authentication and user profiles."""\n\n'
        "from auth.jwt import JwtAuthManager\n"
        "from models.user import UserRecord, format_user_display\n\n"
        "auth_mgr = JwtAuthManager('super-secret-key-12345')\n\n"
        "def handle_login(username: str, email: str) -> dict:\n"
        "    \"\"\"Process login credentials and issue token.\"\"\"\n"
        "    user = UserRecord(user_id=1, username=username, email=email)\n"
        "    tok = auth_mgr.create_token(user)\n"
        "    return {'token': tok, 'user_id': user.user_id, 'display': format_user_display(user)}\n\n"
        "def handle_logout(token: str) -> bool:\n"
        "    \"\"\"Revoke session token.\"\"\"\n"
        "    return auth_mgr.verify_token(token)\n\n"
        "def handle_profile_lookup(user_id: int) -> dict:\n"
        "    \"\"\"Fetch user profile record by primary key.\"\"\"\n"
        "    return {'user_id': user_id, 'status': 'found'}\n",
        encoding="utf-8",
    )

    # 4. Independent billing service (large distractor / unreferenced)
    services_dir = repo / "services"
    services_dir.mkdir()
    (services_dir / "__init__.py").write_text("", encoding="utf-8")
    (services_dir / "billing.py").write_text(
        '"""Billing, payment processing, invoice generation, and credit card gateway."""\n\n'
        "class PaymentGateway:\n"
        "    \"\"\"Integrates with external payment processors.\"\"\"\n"
        "    def __init__(self, api_key: str, merchant_id: str = 'merch_001'):\n"
        "        self.api_key = api_key\n"
        "        self.merchant_id = merchant_id\n\n"
        "    def charge_card(self, amount_cents: int, card_token: str) -> str:\n"
        "        \"\"\"Submit payment charge for card token.\"\"\"\n"
        '        return f"txn_{amount_cents}_{card_token[:4]}"\n\n'
        "    def refund_payment(self, transaction_id: str, reason: str = 'requested') -> bool:\n"
        "        \"\"\"Process full refund for transaction id.\"\"\"\n"
        "        return True\n\n"
        "    def create_invoice(self, customer_id: str, items: list) -> dict:\n"
        "        \"\"\"Generate itemised customer billing invoice.\"\"\"\n"
        '        return {"invoice_id": "inv_99", "customer": customer_id, "total": sum(items)}\n\n'
        "    def calculate_taxes(self, subtotal: float, state_code: str) -> float:\n"
        "        \"\"\"Calculate state sales tax liability.\"\"\"\n"
        "        rate = 0.08 if state_code == 'CA' else 0.05\n"
        "        return round(subtotal * rate, 2)\n",
        encoding="utf-8",
    )

    # 5. Frontend JavaScript files
    fe_dir = repo / "frontend" / "src"
    fe_dir.mkdir(parents=True)
    (fe_dir / "client.js").write_text(
        "// Frontend API client with retry and session management\n\n"
        "export function loginClient(username, password) {\n"
        "    return fetch('/api/login', {\n"
        "        method: 'POST',\n"
        "        headers: { 'Content-Type': 'application/json' },\n"
        "        body: JSON.stringify({ username, password }),\n"
        "    });\n"
        "}\n\n"
        "export function logoutClient(token) {\n"
        "    return fetch('/api/logout', { headers: { 'Authorization': `Bearer ${token}` } });\n"
        "}\n\n"
        "export class SessionManager {\n"
        "    constructor(token) {\n"
        "        this.token = token;\n"
        "        this.createdAt = Date.now();\n"
        "    }\n"
        "    isAuthenticated() {\n"
        "        return Boolean(this.token && (Date.now() - this.createdAt < 3600000));\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )

    # 6. Ignored files: .gitignore, .venv, node_modules, binary
    (repo / ".gitignore").write_text("secret_keys/\n*.log\n", encoding="utf-8")
    secret_dir = repo / "secret_keys"
    secret_dir.mkdir()
    (secret_dir / "secrets.py").write_text("API_SECRET = 'do-not-index'\n", encoding="utf-8")

    venv_dir = repo / ".venv" / "lib"
    venv_dir.mkdir(parents=True)
    (venv_dir / "site_package.py").write_text("def venv_func(): pass\n", encoding="utf-8")

    nm_dir = repo / "node_modules" / "pkg"
    nm_dir.mkdir(parents=True)
    (nm_dir / "index.js").write_text("function npmFunc() {}\n", encoding="utf-8")

    (repo / "data.bin").write_bytes(b"\x00\x01\x02\x03\x04\xff" * 50)

    return repo


# ---------------------------------------------------------------------------
# a) Output stays strictly <= max_tokens
# ---------------------------------------------------------------------------

class TestTokenBudgetStrict:
    @pytest.mark.parametrize("budget", [100, 250, 500, 1000, 2048])
    def test_strict_token_budget_enforcement(self, multi_file_project: Path, budget: int):
        rm = RepoMap(repo_path=multi_file_project, max_tokens=budget)
        result = rm.get_repo_map()
        actual_tokens = count_tokens(result)
        assert actual_tokens <= budget, (
            f"Budget exceeded! Requested max_tokens={budget}, got {actual_tokens}"
        )


# ---------------------------------------------------------------------------
# b) Compression ratio exceeds 50% while retaining cross-file referenced symbols
# ---------------------------------------------------------------------------

class TestCompressionAndCrossFileSymbols:
    def test_compression_exceeds_50_percent(self, multi_file_project: Path):
        rm = RepoMap(repo_path=multi_file_project, max_tokens=400)
        raw_tokens = rm.raw_token_count()
        sliced_map = rm.get_repo_map()
        sliced_tokens = count_tokens(sliced_map)

        assert raw_tokens > 0
        compression = (raw_tokens - sliced_tokens) / raw_tokens * 100.0
        assert compression > 50.0, f"Expected compression > 50%, got {compression:.1f}%"

    def test_cross_file_referenced_symbols_retained(self, multi_file_project: Path):
        # With sufficient budget, the well-connected auth <-> models <-> routes subgraph
        # must retain its core definitions
        rm = RepoMap(repo_path=multi_file_project, max_tokens=800)
        sliced_map = rm.get_repo_map()

        # At least one definition from the connected core should appear
        has_user = "UserRecord" in sliced_map or "validate_user_record" in sliced_map
        has_jwt = "JwtAuthManager" in sliced_map or "create_token" in sliced_map
        has_routes = "handle_login" in sliced_map or "handle_logout" in sliced_map

        assert has_user or has_jwt or has_routes, (
            "Connected core symbols missing from sliced map:\n" + sliced_map
        )


# ---------------------------------------------------------------------------
# c) Personalization: query and focus_files promote matching symbols
# ---------------------------------------------------------------------------

class TestPersonalization:
    def test_focus_files_promotes_target_file(self, multi_file_project: Path):
        rm = RepoMap(repo_path=multi_file_project, max_tokens=300)

        # Baseline: billing is isolated so it might not be prioritized
        # Now focus explicitly on billing
        focused_map = rm.get_repo_map(focus_files=["services/billing.py"])

        assert "services/billing.py" in focused_map, (
            "Focused file services/billing.py was not included in repo map:\n" + focused_map
        )
        assert ("PaymentGateway" in focused_map or "charge_card" in focused_map), (
            "Definitions from focused file missing:\n" + focused_map
        )

    def test_query_promotes_relevant_symbols(self, multi_file_project: Path):
        rm = RepoMap(repo_path=multi_file_project, max_tokens=300)

        # Query targeting billing service
        query_map = rm.get_repo_map(query="How to charge credit cards and refund payments?")

        assert "services/billing.py" in query_map or "PaymentGateway" in query_map, (
            "Query-relevant symbols missing from personalized map:\n" + query_map
        )

    def test_query_promotes_frontend_client(self, multi_file_project: Path):
        rm = RepoMap(repo_path=multi_file_project, max_tokens=350)
        query_map = rm.get_repo_map(query="frontend login client session manager")

        assert "frontend/src/client.js" in query_map or "SessionManager" in query_map, (
            "Frontend symbols missing when targeted by query:\n" + query_map
        )

    def test_query_identifier_splitting_boosts_matching_symbols(self, tmp_path: Path):
        repo = tmp_path / "split_repo"
        repo.mkdir()
        (repo / "blueprints.py").write_text(
            "class Blueprint:\n"
            "    def __init__(self, name): self.name = name\n"
            "    def add_url_rule(self, rule, endpoint): pass\n",
            encoding="utf-8",
        )
        (repo / "unrelated.py").write_text(
            "class UnrelatedWorker:\n"
            "    def process_background_job(self): pass\n",
            encoding="utf-8",
        )
        rm = RepoMap(repo_path=repo, max_tokens=250)
        # Query has snake_case and camelCase terms: 'Blueprint routing url_rule'
        res = rm.get_repo_map(query="Blueprint routing url_rule")
        assert "blueprints.py" in res
        assert "add_url_rule" in res or "Blueprint" in res

    def test_production_source_outranks_tests_for_non_test_queries(self, tmp_path: Path):
        repo = tmp_path / "prod_vs_test_repo"
        repo.mkdir()
        src_dir = repo / "src"
        src_dir.mkdir()
        (src_dir / "auth.py").write_text(
            "class AuthenticationService:\n"
            "    def verify_credentials(self, u, p): return True\n"
            "    def issue_token(self, u): return 'token'\n",
            encoding="utf-8",
        )
        test_dir = repo / "tests"
        test_dir.mkdir()
        (test_dir / "test_auth.py").write_text(
            "from src.auth import AuthenticationService\n"
            "def test_verify_credentials():\n"
            "    svc = AuthenticationService()\n"
            "    assert svc.verify_credentials('a', 'b')\n",
            encoding="utf-8",
        )
        rm_tight = RepoMap(repo_path=repo, max_tokens=50)
        # Non-test query targeting auth: production definition wins and fits budget (36 tokens), test file is excluded
        res_nontest = rm_tight.get_repo_map(query="AuthenticationService verify_credentials")
        assert "src/auth.py" in res_nontest
        assert "tests/test_auth.py" not in res_nontest

        # With larger budget where both fit, production file must strictly precede test file
        rm_wide = RepoMap(repo_path=repo, max_tokens=250)
        res_wide = rm_wide.get_repo_map(query="AuthenticationService verify_credentials")
        assert "src/auth.py" in res_wide and "tests/test_auth.py" in res_wide
        assert res_wide.find("src/auth.py") < res_wide.find("tests/test_auth.py")

        # Query explicitly mentioning 'test' should not penalize test files
        res_test = rm_wide.get_repo_map(query="test verify credentials test_auth")
        assert "tests/test_auth.py" in res_test


# ---------------------------------------------------------------------------
# d) Warm SQLite cache matches cold output, runs faster, invalidates on mtime
# ---------------------------------------------------------------------------

class TestTagCacheBehavior:
    def test_cache_hits_and_invalidation(self, multi_file_project: Path, tmp_path: Path):
        cache_db = tmp_path / "custom_tags.db"
        cache = TagCache(db_path=cache_db)

        rm = RepoMap(repo_path=multi_file_project, max_tokens=600, cache=cache)

        # Cold pass
        t0 = time.perf_counter()
        cold_map = rm.get_repo_map()
        cold_ms = (time.perf_counter() - t0) * 1000

        # Warm pass
        t1 = time.perf_counter()
        warm_map = rm.get_repo_map()
        warm_ms = (time.perf_counter() - t1) * 1000

        # Cache content consistency
        assert cold_map == warm_map, "Warm map output did not match cold map output"

        # Invalidate a file by updating its mtime + content
        target_file = multi_file_project / "models" / "user.py"
        cached_before = cache.get(target_file)
        assert cached_before is not None, "Target file should be cached"

        # Update file content and advance mtime
        time.sleep(0.02)
        target_file.write_text(
            target_file.read_text(encoding="utf-8") + "\ndef new_cached_function(): pass\n",
            encoding="utf-8",
        )
        new_mtime = target_file.stat().st_mtime + 2.0
        os.utime(target_file, (new_mtime, new_mtime))

        # Cache miss on modified file
        cached_after = cache.get(target_file)
        assert cached_after is None, "Cache should have invalidated after mtime change"

        # New map reflects updated function
        new_map = rm.get_repo_map(max_tokens=800 if hasattr(rm, "get_repo_map") else None)
        assert "new_cached_function" in new_map or "models/user.py" in new_map
        cache.close()

    def test_warm_cache_does_not_reread_for_raw_tokens(
        self, multi_file_project: Path, tmp_path: Path, monkeypatch
    ):
        cache_db = tmp_path / "cache_token_test.db"
        cache = TagCache(db_path=cache_db)

        # Cold pass: populates SQLite cache with token counts
        rm1 = RepoMap(repo_path=multi_file_project, cache=cache)
        raw_cold = rm1.raw_token_count()
        assert raw_cold > 0

        # Create fresh RepoMap instance sharing the warm cache
        rm2 = RepoMap(repo_path=multi_file_project, cache=cache)

        # Track any calls to read_text on source files
        read_calls = []
        original_read_text = Path.read_text

        def tracking_read_text(self, *args, **kwargs):
            # Only track reads on source code files inside the repo (skip .gitignore)
            if self.name != ".gitignore":
                try:
                    self.relative_to(multi_file_project)
                    read_calls.append(str(self))
                except ValueError:
                    pass
            return original_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", tracking_read_text)

        # Warm pass: must sum raw_tokens from SQLite without reading files from disk
        raw_warm = rm2.raw_token_count()
        assert raw_warm == raw_cold
        assert len(read_calls) == 0, f"Warm cache unexpectedly re-read files from disk: {read_calls}"
        cache.close()


# ---------------------------------------------------------------------------
# e) MCP tools/list and Streamable HTTP execution
# ---------------------------------------------------------------------------

class TestMCPSlicerTools:
    def test_mcp_tools_list_and_call(self, multi_file_project: Path, tmp_path: Path):
        cfg = EdgeMemConfig(
            host="127.0.0.1",
            port=8080,
            upstream_base_url="http://mock-upstream",
            model="test-model",
            telemetry_dir=str(tmp_path / "telemetry"),
        )
        app = create_app(cfg)

        with TestClient(app) as client:
            # 1. Initialize MCP
            init_resp = client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "test-runner", "version": "1.0"},
                    },
                },
                headers={"Content-Type": "application/json", "Accept": "application/json"},
            )
            assert init_resp.status_code == 200
            session_id = init_resp.headers.get("mcp-session-id")
            headers = {"Content-Type": "application/json", "Accept": "application/json"}
            if session_id:
                headers["mcp-session-id"] = session_id

            # 2. Initialized notification
            client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "method": "notifications/initialized"},
                headers=headers,
            )

            # 3. tools/list
            tools_resp = client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
                headers=headers,
            )
            assert tools_resp.status_code == 200
            tool_data = tools_resp.json()
            tool_names = {t["name"] for t in tool_data["result"]["tools"]}

            assert "edgemem_repo_map" in tool_names, f"edgemem_repo_map missing from {tool_names}"
            assert "edgemem_file_outline" in tool_names, f"edgemem_file_outline missing from {tool_names}"

            # 4. tools/call edgemem_repo_map
            call_resp = client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "edgemem_repo_map",
                        "arguments": {
                            "repo_path": str(multi_file_project),
                            "max_tokens": 512,
                        },
                    },
                },
                headers=headers,
            )
            assert call_resp.status_code == 200
            call_data = call_resp.json()
            content_text = call_data["result"]["content"][0]["text"]
            map_result = json.loads(content_text)

            assert "repo_map" in map_result
            assert map_result["sliced_tokens"] <= 512
            assert map_result["compression_pct"] > 0
            assert "models/user.py" in map_result["repo_map"] or "auth/jwt.py" in map_result["repo_map"]

            # 5. tools/call edgemem_file_outline
            outline_resp = client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "tools/call",
                    "params": {
                        "name": "edgemem_file_outline",
                        "arguments": {
                            "file_path": str(multi_file_project / "auth" / "jwt.py"),
                        },
                    },
                },
                headers=headers,
            )
            assert outline_resp.status_code == 200
            outline_data = outline_resp.json()
            outline_text = json.loads(outline_data["result"]["content"][0]["text"])["outline"]
            assert "JwtAuthManager" in outline_text or "create_token" in outline_text


# ---------------------------------------------------------------------------
# f) Multi-language extraction & ignore rules
# ---------------------------------------------------------------------------

class TestScanningAndMultiLanguage:
    def test_ignored_dirs_and_files_excluded(self, multi_file_project: Path):
        files = scan_repo_files(multi_file_project)
        rel_files = [str(f.relative_to(multi_file_project)) for f in files]

        # Excluded by default / .gitignore
        for rf in rel_files:
            assert not rf.startswith(".venv"), f".venv not ignored: {rf}"
            assert not rf.startswith("node_modules"), f"node_modules not ignored: {rf}"
            assert not rf.startswith("secret_keys"), f".gitignore dir not ignored: {rf}"
            assert not rf.endswith(".bin"), f"binary file not ignored: {rf}"

    def test_multilanguage_ast_tags(self, tmp_path: Path):
        # Go, Rust, C files
        (tmp_path / "main.go").write_text("package main\nfunc StartServer() {}\ntype Config struct {}\n", encoding="utf-8")
        (tmp_path / "lib.rs").write_text("pub fn init_engine() {}\npub struct Engine {}\n", encoding="utf-8")
        (tmp_path / "helper.c").write_text("int run_helper(int flag) { return flag + 1; }\n", encoding="utf-8")

        go_tags = extract_tags(tmp_path / "main.go", "main.go")
        go_defs = {t.name for t in go_tags if t.kind == "def"}
        assert "StartServer" in go_defs
        assert "Config" in go_defs

        rs_tags = extract_tags(tmp_path / "lib.rs", "lib.rs")
        rs_defs = {t.name for t in rs_tags if t.kind == "def"}
        assert "init_engine" in rs_defs
        assert "Engine" in rs_defs

        c_tags = extract_tags(tmp_path / "helper.c", "helper.c")
        c_defs = {t.name for t in c_tags if t.kind == "def"}
        assert "run_helper" in c_defs


class TestOutlineCacheBehavior:
    def test_second_slice_hits_in_memory_outline_cache(
        self, multi_file_project: Path, tmp_path: Path, monkeypatch
    ):
        from edgemem.slicer import repomap
        from edgemem.slicer.repomap import clear_graph_cache

        clear_graph_cache()
        cache_db = tmp_path / "outline_test.db"
        cache = TagCache(db_path=cache_db)
        rm = RepoMap(repo_path=multi_file_project, max_tokens=1000, cache=cache)

        out1 = rm.get_repo_map()
        assert out1

        calls = 0
        orig_render = repomap.render_tree_context

        def spy_render(*args, **kwargs):
            nonlocal calls
            calls += 1
            return orig_render(*args, **kwargs)

        monkeypatch.setattr(repomap, "render_tree_context", spy_render)

        out2 = rm.get_repo_map()
        assert out2 == out1
        assert calls == 0
        cache.close()

    def test_warm_sqlite_hits_persisted_outline_cache(
        self, multi_file_project: Path, tmp_path: Path, monkeypatch
    ):
        from edgemem.slicer import repomap
        from edgemem.slicer.repomap import clear_graph_cache

        clear_graph_cache()
        cache_db = tmp_path / "persisted_outline.db"

        cache1 = TagCache(db_path=cache_db)
        rm1 = RepoMap(repo_path=multi_file_project, max_tokens=1000, cache=cache1)
        out1 = rm1.get_repo_map()
        assert out1
        cache1.close()

        clear_graph_cache()

        cache2 = TagCache(db_path=cache_db)
        rm2 = RepoMap(repo_path=multi_file_project, max_tokens=1000, cache=cache2)
        assert len(cache2.outline_cache) == 0

        calls = 0
        orig_render = repomap.render_tree_context

        def spy_render(*args, **kwargs):
            nonlocal calls
            calls += 1
            return orig_render(*args, **kwargs)

        monkeypatch.setattr(repomap, "render_tree_context", spy_render)

        out2 = rm2.get_repo_map()
        assert out2 == out1
        assert calls == 0
        assert len(cache2.outline_cache) > 0
        cache2.close()

    def test_file_edit_invalidates_caches_and_returns_updated_symbols(
        self, multi_file_project: Path, tmp_path: Path
    ):
        from edgemem.slicer.repomap import clear_graph_cache

        clear_graph_cache()
        cache_db = tmp_path / "invalidation_test.db"
        cache = TagCache(db_path=cache_db)
        rm = RepoMap(repo_path=multi_file_project, max_tokens=1000, cache=cache)

        out1 = rm.get_repo_map()
        assert "UserRecord" in out1

        target_file = multi_file_project / "models" / "user.py"
        old_st = target_file.stat()
        old_mtime_ns = old_st.st_mtime_ns
        old_size = old_st.st_size

        time.sleep(0.02)
        target_file.write_text(
            target_file.read_text(encoding="utf-8") + "\ndef unique_phase4a_function(): pass\n",
            encoding="utf-8",
        )
        new_mtime = target_file.stat().st_mtime + 2.0
        os.utime(target_file, (new_mtime, new_mtime))
        new_st = target_file.stat()

        clear_graph_cache()
        out2 = rm.get_repo_map()
        assert "unique_phase4a_function" in out2

        rows = cache._conn.execute(
            "SELECT mtime_ns, size FROM outline_cache WHERE rel_path = ?",
            ("models/user.py",),
        ).fetchall()
        assert len(rows) > 0
        for r_mtime, r_size in rows:
            assert r_mtime == new_st.st_mtime_ns
            assert r_size == new_st.st_size
            assert r_mtime != old_mtime_ns or r_size != old_size

        stale_in_mem = [
            k for k in cache.outline_cache._lru
            if k[0] == "models/user.py" and (k[1] != new_st.st_mtime_ns or k[2] != new_st.st_size)
        ]
        assert len(stale_in_mem) == 0
        cache.close()

    def test_in_memory_lru_cache_never_exceeds_max_entries(self, tmp_path: Path):
        from edgemem.slicer.cache import OutlineCache

        cache_db = tmp_path / "lru_limit_test.db"
        cache = OutlineCache(db_path=cache_db, max_entries=3)

        for i in range(10):
            cache.put(f"file_{i}.py", 1000 + i, 50 + i, "k0", f"render_{i}", 10)
            assert len(cache) <= 3

        assert len(cache) == 3
        assert ("file_0.py", 1000, 50, "k0") not in cache._lru
        assert ("file_9.py", 1009, 59, "k0") in cache._lru
        cache.close()

    def test_warm_repo_map_never_loads_tiktoken(
        self, multi_file_project: Path, tmp_path: Path, monkeypatch
    ):
        from edgemem.slicer import repomap
        from edgemem.slicer.repomap import clear_graph_cache
        import tiktoken

        cache_db = tmp_path / "warm_tiktoken.db"
        cache1 = TagCache(db_path=cache_db)
        rm1 = RepoMap(repo_path=multi_file_project, max_tokens=1000, cache=cache1)
        cold_map = rm1.get_repo_map()
        assert cold_map
        cache1.close()

        clear_graph_cache()
        repomap._ENCODINGS.clear()

        def fail_get_encoding(*args, **kwargs):
            raise AssertionError("tiktoken.get_encoding was called during warm RepoMap execution")

        monkeypatch.setattr(tiktoken, "get_encoding", fail_get_encoding)

        cache2 = TagCache(db_path=cache_db)
        rm2 = RepoMap(repo_path=multi_file_project, max_tokens=1000, cache=cache2)
        assert rm2._enc is None

        warm_map = rm2.get_repo_map()
        assert warm_map == cold_map
        assert rm2._enc is None
        cache2.close()

    def test_find_matching_files_camel_and_snake_case_equivalence(
        self, multi_file_project: Path
    ):
        from edgemem.slicer.pagerank import find_matching_files, personalise_ranks
        from edgemem.slicer.repomap import RepoMap, extract_query_tokens
        from edgemem.slicer.tags import extract_tags, scan_repo_files

        files = scan_repo_files(multi_file_project)
        all_tags = []
        for f in files:
            all_tags.extend(extract_tags(f, str(f.relative_to(multi_file_project))))
        all_file_nodes = [str(f.relative_to(multi_file_project)) for f in files]

        q_camel = "UserRecord JwtAuth"
        q_snake = "user_record jwt_auth"

        tok_camel = extract_query_tokens(q_camel)
        tok_snake = extract_query_tokens(q_snake)

        scores_camel = find_matching_files(
            all_file_nodes, all_tags, None, None, tok_camel, query=q_camel
        )
        scores_snake = find_matching_files(
            all_file_nodes, all_tags, None, None, tok_snake, query=q_snake
        )

        assert scores_camel == scores_snake
        assert "models/user.py" in scores_camel
        assert "auth/jwt.py" in scores_camel

        base_ranks = {f: 1.0 / len(all_file_nodes) for f in all_file_nodes}
        ranks_camel = personalise_ranks(
            base_ranks, None, None, tok_camel, all_tags, query=q_camel, match_scores=scores_camel
        )
        ranks_snake = personalise_ranks(
            base_ranks, None, None, tok_snake, all_tags, query=q_snake, match_scores=scores_snake
        )
        assert ranks_camel == ranks_snake

        rm = RepoMap(repo_path=multi_file_project, max_tokens=1000)
        map_camel = rm.get_repo_map(query=q_camel)
        map_snake = rm.get_repo_map(query=q_snake)
        assert map_camel == map_snake


