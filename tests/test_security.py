from __future__ import annotations

import io
import json
import sys
import types
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from starlette.testclient import TestClient

from crispllm.config import EdgeMemConfig, MemoryConfig, SecurityConfig
from crispllm.proxy.app import create_app
from crispllm.proxy.mcp_server import create_mcp_server
from crispllm.proxy.sysinfo import get_process_rss_mb


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


class MockTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=NON_STREAM_RESPONSE,
            headers={"content-type": "application/json"},
        )


def _make_app(tmp_path: Path, security_cfg: SecurityConfig | None = None):
    cfg = EdgeMemConfig(
        host="127.0.0.1",
        port=8080,
        upstream_base_url="http://mock-upstream",
        model="test-model",
        telemetry_dir=str(tmp_path / "telemetry"),
        memory=MemoryConfig(db_path=str(tmp_path / "memory.db")),
        security=security_cfg or SecurityConfig(),
    )
    application = create_app(cfg)
    original_lifespan = application.router.lifespan_context

    @asynccontextmanager
    async def patched_lifespan(app):
        async with original_lifespan(app):
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


@pytest.fixture(autouse=True)
def _clean_security_env(monkeypatch):
    monkeypatch.delenv("EDGEMEM_AUTH_TOKEN", raising=False)


def test_default_config_picks_up_env_token(monkeypatch):
    monkeypatch.setenv("EDGEMEM_AUTH_TOKEN", "auto-discovered-token")
    cfg = EdgeMemConfig()
    assert cfg.security.auth_token_env == "EDGEMEM_AUTH_TOKEN"
    assert cfg.security.resolve_auth_token() == "auto-discovered-token"


def test_backward_compatibility_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("EDGEMEM_AUTH_TOKEN", raising=False)
    app = _make_app(tmp_path)
    with TestClient(app, raise_server_exceptions=True) as client:
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        health_data = health_resp.json()
        assert health_data["status"] == "ok"
        assert health_data["auth_enabled"] is False
        assert health_data["allow_remote_v1"] is False

        chat_resp = client.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert chat_resp.status_code == 200

        mcp_init = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0"},
            },
        }
        mcp_resp = client.post(
            "/mcp",
            json=mcp_init,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        assert mcp_resp.status_code == 200
        init_data = mcp_resp.json()
        assert init_data.get("result", {}).get("serverInfo", {}).get("name") in ("CrispLLM", "EdgeMem")
        assert init_data.get("result", {}).get("serverInfo", {}).get("version") == "1.1.0"


def test_bearer_auth_via_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("EDGEMEM_AUTH_TOKEN", "super-secret-token")
    sec_cfg = SecurityConfig(auth_token_env="EDGEMEM_AUTH_TOKEN")
    app = _make_app(tmp_path, sec_cfg)

    with TestClient(app, raise_server_exceptions=True) as client:
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        health_data = health_resp.json()
        assert health_data["auth_enabled"] is True
        assert "super-secret-token" not in health_resp.text

        missing_auth_v1 = client.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert missing_auth_v1.status_code == 401
        assert missing_auth_v1.headers.get("www-authenticate") == "Bearer"
        assert missing_auth_v1.json()["error"]["type"] == "unauthorized"

        wrong_auth_v1 = client.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert wrong_auth_v1.status_code == 401

        mcp_init = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0"},
            },
        }
        missing_auth_mcp = client.post(
            "/mcp",
            json=mcp_init,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        assert missing_auth_mcp.status_code == 401
        assert missing_auth_mcp.headers.get("www-authenticate") == "Bearer"

        valid_auth_v1 = client.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer super-secret-token"},
        )
        assert valid_auth_v1.status_code == 200

        valid_auth_mcp = client.post(
            "/mcp",
            json=mcp_init,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": "Bearer super-secret-token",
            },
        )
        assert valid_auth_mcp.status_code == 200


def test_bearer_auth_literal_token(tmp_path):
    sec_cfg = SecurityConfig(auth_token="literal-secret-val")
    app = _make_app(tmp_path, sec_cfg)

    with TestClient(app, raise_server_exceptions=True) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer literal-secret-val"},
        )
        assert resp.status_code == 200


def test_tunnel_v1_isolation(tmp_path):
    sec_cfg = SecurityConfig(
        auth_token="test-token",
        allow_remote_v1=False,
    )
    app = _make_app(tmp_path, sec_cfg)

    with TestClient(app, raise_server_exceptions=True) as client:
        tunnel_headers = [
            {"X-Forwarded-For": "203.0.113.10"},
            {"CF-Connecting-IP": "203.0.113.10"},
            {"Forwarded": "for=203.0.113.10"},
            {"X-Real-IP": "203.0.113.10"},
        ]

        for hdr in tunnel_headers:
            req_headers = {"Authorization": "Bearer test-token"}
            req_headers.update(hdr)
            v1_resp = client.post(
                "/v1/chat/completions",
                json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
                headers=req_headers,
            )
            assert v1_resp.status_code == 403
            data = v1_resp.json()
            assert data["error"]["type"] == "forbidden"
            assert "Remote access to /v1 is disabled" in data["error"]["message"]

        mcp_init = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0"},
            },
        }
        mcp_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": "Bearer test-token",
            "X-Forwarded-For": "203.0.113.10",
        }
        mcp_resp = client.post("/mcp", json=mcp_init, headers=mcp_headers)
        assert mcp_resp.status_code == 200


def test_tunnel_v1_allow_remote_v1(tmp_path):
    sec_cfg = SecurityConfig(
        auth_token="test-token",
        allow_remote_v1=True,
    )
    app = _make_app(tmp_path, sec_cfg)

    with TestClient(app, raise_server_exceptions=True) as client:
        req_headers = {
            "Authorization": "Bearer test-token",
            "X-Forwarded-For": "203.0.113.10",
        }
        v1_resp = client.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
            headers=req_headers,
        )
        assert v1_resp.status_code == 200


@pytest.mark.asyncio
async def test_mcp_allowed_repo_roots(tmp_path):
    allowed_root = tmp_path / "allowed_repo"
    allowed_root.mkdir()
    (allowed_root / "sample.py").write_text("def hello(): return 'world'\n", encoding="utf-8")

    forbidden_root = tmp_path / "forbidden_repo"
    forbidden_root.mkdir()
    (forbidden_root / "secret.py").write_text("API_KEY = 'secret'\n", encoding="utf-8")

    sec_cfg = SecurityConfig(allowed_repo_roots=[str(allowed_root)])
    cfg = EdgeMemConfig(
        host="127.0.0.1",
        port=8080,
        upstream_base_url="http://mock-upstream",
        model="test-model",
        security=sec_cfg,
    )

    server = create_mcp_server(config=cfg)

    res_allowed_raw = await server.call_tool("edgemem_repo_map", {"repo_path": str(allowed_root)})
    res_allowed = json.loads(res_allowed_raw.content[0].text)
    assert "error" not in res_allowed
    assert "repo_map" in res_allowed

    res_forbidden_raw = await server.call_tool("edgemem_repo_map", {"repo_path": str(forbidden_root)})
    res_forbidden = json.loads(res_forbidden_raw.content[0].text)
    assert "error" in res_forbidden
    assert "not within allowed repository roots" in res_forbidden["error"]

    traversal_path = str(allowed_root / ".." / "forbidden_repo")
    res_traversal_raw = await server.call_tool("edgemem_repo_map", {"repo_path": traversal_path})
    res_traversal = json.loads(res_traversal_raw.content[0].text)
    assert "error" in res_traversal

    outline_allowed_raw = await server.call_tool("edgemem_file_outline", {"file_path": str(allowed_root / "sample.py")})
    outline_allowed = json.loads(outline_allowed_raw.content[0].text)
    assert "error" not in outline_allowed
    assert "outline" in outline_allowed

    outline_forbidden_raw = await server.call_tool("edgemem_file_outline", {"file_path": str(forbidden_root / "secret.py")})
    outline_forbidden = json.loads(outline_forbidden_raw.content[0].text)
    assert "error" in outline_forbidden
    assert "not within allowed repository roots" in outline_forbidden["error"]

    outline_etc_raw = await server.call_tool("edgemem_file_outline", {"file_path": "/etc/passwd"})
    outline_etc = json.loads(outline_etc_raw.content[0].text)
    assert "error" in outline_etc


@pytest.mark.asyncio
async def test_mcp_tool_annotations():
    server = create_mcp_server()
    tools = await server.list_tools()
    tool_map = {t.name: t for t in tools}

    repo_map_tool = tool_map["edgemem_repo_map"]
    assert repo_map_tool.annotations is not None
    assert repo_map_tool.annotations.read_only_hint is True
    assert repo_map_tool.annotations.destructive_hint is False

    recall_tool = tool_map["edgemem_recall"]
    assert recall_tool.annotations is not None
    assert recall_tool.annotations.read_only_hint is True
    assert recall_tool.annotations.destructive_hint is False

    remember_tool = tool_map["edgemem_remember"]
    assert remember_tool.annotations is not None
    assert remember_tool.annotations.read_only_hint is False
    assert remember_tool.annotations.destructive_hint is False

    forget_tool = tool_map["edgemem_forget"]
    assert forget_tool.annotations is not None
    assert forget_tool.annotations.read_only_hint is False
    assert forget_tool.annotations.destructive_hint is True


def test_sysinfo_get_process_rss_mb():
    current_rss = get_process_rss_mb()
    assert isinstance(current_rss, float)
    assert current_rss >= 0.0

    linux_status = "Name:\tpython\nVmPeak:\t  100000 kB\nVmRSS:\t   51200 kB\nVmData:\t   20000 kB\n"
    with patch("sys.platform", "linux"):
        with patch("builtins.open", return_value=io.StringIO(linux_status)):
            rss_linux = get_process_rss_mb()
            assert pytest.approx(rss_linux, 0.01) == 50.0

    mock_rusage = MagicMock()
    mock_rusage.ru_maxrss = 104857600
    mock_resource = types.SimpleNamespace(RUSAGE_SELF=0, getrusage=lambda _: mock_rusage)
    with patch("sys.platform", "darwin"):
        with patch.dict(sys.modules, {"resource": mock_resource}):
            rss_darwin = get_process_rss_mb()
            assert pytest.approx(rss_darwin, 0.01) == 100.0

    mock_failing_resource = types.SimpleNamespace(
        RUSAGE_SELF=0, getrusage=MagicMock(side_effect=Exception("no resource"))
    )
    with patch("sys.platform", "linux"):
        with patch("builtins.open", side_effect=OSError("permission denied")):
            with patch.dict(sys.modules, {"resource": mock_failing_resource}):
                rss_fallback = get_process_rss_mb()
                assert rss_fallback == 0.0
