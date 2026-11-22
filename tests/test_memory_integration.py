"""Integration tests for EdgeMem memory middleware, MCP memory tools, and CLI."""

from __future__ import annotations

import inspect
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from click.testing import CliRunner
import httpx
import pytest
from starlette.testclient import TestClient

from edgemem.cli import cli
from edgemem.config import EdgeMemConfig, MemoryConfig
from edgemem.proxy import middleware
from edgemem.proxy.app import create_app

MOCK_UPSTREAM_RESPONSE: dict[str, Any] = {
    "id": "chatcmpl-mem-test",
    "object": "chat.completion",
    "model": "test-model",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Upstream response mock.",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
}


class MockTransport(httpx.AsyncBaseTransport):
    """Captures outgoing request bodies sent to upstream."""

    def __init__(self) -> None:
        self.recorded_bodies: list[dict[str, Any]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.recorded_bodies.append(body)
        return httpx.Response(
            200,
            json=MOCK_UPSTREAM_RESPONSE,
            headers={"content-type": "application/json"},
        )


def _make_app(
    tmp_path: Path,
    memory_enabled: bool = True,
    max_tokens: int = 384,
    compact_threshold_tokens: int = 2000,
    compact_min_turn_tokens: int = 150,
):
    db_file = tmp_path / "test_memory.db"
    cfg = EdgeMemConfig(
        host="127.0.0.1",
        port=8080,
        upstream_base_url="http://mock-upstream",
        model="test-model",
        telemetry_dir=str(tmp_path / "telemetry"),
        memory=MemoryConfig(
            enabled=memory_enabled,
            max_injected_tokens=max_tokens,
            inject_constraints=True,
            auto_extract=True,
            compact_middle_turns=True,
            compact_threshold_tokens=compact_threshold_tokens,
            compact_min_turn_tokens=compact_min_turn_tokens,
            keep_recent_turns=2,
            db_path=str(db_file),
        ),
    )
    transport = MockTransport()
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
    return app, transport, cfg


def test_tiktoken_not_imported_by_middleware() -> None:
    """Ensure tiktoken is never imported by proxy middleware to keep RSS < 85 MB."""
    source = inspect.getsource(middleware)
    assert "import tiktoken" not in source
    assert "from tiktoken" not in source


def test_single_turn_request_no_injection(tmp_path: Path) -> None:
    """Single-turn requests with empty store must add 0 overhead tokens and leave messages untouched."""
    app, transport, _ = _make_app(tmp_path, memory_enabled=True)
    single_turn = [
        {"role": "system", "content": "You are a Python coding assistant."},
        {"role": "user", "content": "Write a function to add two numbers."},
    ]
    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json={"messages": single_turn})
        assert resp.status_code == 200

    assert len(transport.recorded_bodies) == 1
    sent_messages = transport.recorded_bodies[0]["messages"]
    assert sent_messages == single_turn


def test_clean_formatting_and_precedence_note(tmp_path: Path) -> None:
    """Injected memory must omit internal hash keys and include the non-overriding precedence note."""
    app, transport, _ = _make_app(tmp_path, memory_enabled=True)
    messages = [
        {
            "role": "user",
            "content": "Rule: DatabaseConnection must be a singleton. The Cache must be frozen.",
        },
        {
            "role": "assistant",
            "content": "Noted.",
        },
        {
            "role": "user",
            "content": "Implement the cache module.",
        },
    ]
    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json={"messages": messages})
        assert resp.status_code == 200

    assert len(transport.recorded_bodies) == 1
    first_content = transport.recorded_bodies[0]["messages"][0]["content"]

    # Verify precedence note
    assert (
        "Earlier-turn constraints & facts (follow these unless the latest user message "
        "explicitly specifies an exception or return behavior)"
    ) in first_content

    # Verify internal constraint_* keys are NOT leaked
    assert "constraint_" not in first_content
    assert (
        "- DatabaseConnection must be a singleton." in first_content
        or "- The Cache must be frozen." in first_content
    )


def test_middle_turn_compaction_shrinks_prompt_tokens(tmp_path: Path) -> None:
    """Verify 5-turn conversation with distractors compacts them into episodes and shrinks prompt tokens."""
    app, transport, cfg = _make_app(
        tmp_path,
        memory_enabled=True,
        compact_threshold_tokens=0,
        compact_min_turn_tokens=0,
    )

    distractor_q = (
        "Can you explain the detailed history of the Roman empire from 753 BC to the fall of Constantinople? "
        "Also describe the culinary evolution of sourdough pizza through the Mediterranean basin in 500 words."
    )
    distractor_a = (
        "The Roman civilization began as an Italic settlement dated traditionally to 753 BC. "
        "Over centuries it expanded into an empire spanning the Mediterranean. Meanwhile, flatbread traditions "
        "evolved with natural wild yeasts across ancient Greece and Rome."
    )

    messages = [
        # Turn 0: User sets critical constraint
        {"role": "user", "content": "Critical invariant: TransactionRecord must be frozen and immutable."},
        # Turn 1: Assistant
        {"role": "assistant", "content": "Acknowledged. I will enforce that TransactionRecord is frozen."},
        # Turn 2: Distractor User
        {"role": "user", "content": distractor_q},
        # Turn 3: Distractor Assistant
        {"role": "assistant", "content": distractor_a},
        # Turn 4: Final User prompt
        {"role": "user", "content": "Now implement the TransactionRecord commit method."},
    ]

    raw_chars = sum(len(m["content"]) for m in messages)

    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json={"messages": messages})
        assert resp.status_code == 200

    assert len(transport.recorded_bodies) == 1
    sent_messages = transport.recorded_bodies[0]["messages"]

    compacted_chars = sum(len(m["content"]) for m in sent_messages)
    assert compacted_chars < raw_chars

    # Verify placeholder inserted for middle distractors
    contents = [m["content"] for m in sent_messages]
    assert any("[Compacted earlier turn]" in c for c in contents)

    # Verify Turn 1 constraint retained
    sys_msg = sent_messages[0]["content"]
    assert "frozen and immutable" in sys_msg

    # Verify episodes were recorded in SQLite
    from edgemem.memory.store import MemoryStore

    with MemoryStore(cfg.memory.db_path) as store:
        st = store.stats()
        assert st["episodes_count"] >= 2


def test_multi_turn_constraint_injection(tmp_path: Path) -> None:
    """Test multi-turn conversation extracts Turn-1 constraints and injects active memory."""
    app, transport, _ = _make_app(tmp_path, memory_enabled=True, max_tokens=384)

    messages = [
        {
            "role": "user",
            "content": (
                "Requirements:\n"
                "- The AuthToken must be frozen and strictly immutable.\n"
                "- The DatabaseConnection must be a singleton."
            ),
        },
        {
            "role": "assistant",
            "content": "Acknowledged.\n```python\nDEFAULT_EXPIRY = 3600\n```",
        },
        {
            "role": "user",
            "content": "What is the distance to the moon? Tell me a joke.",
        },
        {
            "role": "assistant",
            "content": "The moon is about 384,400 km away.",
        },
        {
            "role": "user",
            "content": "Now implement the token verification method using DEFAULT_EXPIRY.",
        },
    ]

    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json={"messages": messages})
        assert resp.status_code == 200

    assert len(transport.recorded_bodies) == 1
    upstream_body = transport.recorded_bodies[0]
    upstream_messages = upstream_body["messages"]

    # Verify memory block was injected into the first message
    first_msg = upstream_messages[0]
    assert first_msg["role"] == "system"
    content = first_msg["content"]
    assert "[EdgeMem Active Memory]" in content or "--- EdgeMem Core Memory ---" in content
    assert "frozen and strictly immutable" in content
    assert "singleton" in content
    assert "DEFAULT_EXPIRY" in content or "DEFAULT_EXPIRY" in upstream_messages[-1]["content"]


def test_session_isolation_no_leak(tmp_path: Path) -> None:
    """Test that two conversations with different Turn-1 prompts get isolated sessions."""
    app, transport, _ = _make_app(tmp_path, memory_enabled=True)

    # Conversation A
    conv_a = [
        {"role": "user", "content": "Rule for System Alpha: must strictly use RSA encryption."},
        {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": "Write crypto handler for System Alpha."},
    ]

    # Conversation B
    conv_b = [
        {"role": "user", "content": "Rule for System Beta: must strictly use AES-GCM encryption."},
        {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": "Write crypto handler for System Beta."},
    ]

    with TestClient(app) as client:
        resp_a = client.post("/v1/chat/completions", json={"messages": conv_a})
        assert resp_a.status_code == 200

        resp_b = client.post("/v1/chat/completions", json={"messages": conv_b})
        assert resp_b.status_code == 200

    assert len(transport.recorded_bodies) == 2
    body_a = transport.recorded_bodies[0]
    body_b = transport.recorded_bodies[1]

    content_a = body_a["messages"][0]["content"]
    content_b = body_b["messages"][0]["content"]

    assert "RSA" in content_a
    assert "AES-GCM" not in content_a

    assert "AES-GCM" in content_b
    assert "RSA" not in content_b


def test_memory_disabled_forwards_untouched(tmp_path: Path) -> None:
    """Test memory.enabled = False leaves payload completely untouched."""
    app, transport, _ = _make_app(tmp_path, memory_enabled=False)

    messages = [
        {"role": "user", "content": "Constraint: must be immutable."},
        {"role": "assistant", "content": "Understood."},
        {"role": "user", "content": "Generate code."},
    ]

    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json={"messages": messages})
        assert resp.status_code == 200

    assert len(transport.recorded_bodies) == 1
    upstream_messages = transport.recorded_bodies[0]["messages"]
    assert len(upstream_messages) == len(messages)
    for orig, sent in zip(messages, upstream_messages):
        assert orig == sent
        assert "[EdgeMem Active Memory]" not in sent.get("content", "")


def test_health_reports_memory_stats(tmp_path: Path) -> None:
    """Test GET /health reports memory_enabled and facts_count."""
    app, _, _ = _make_app(tmp_path, memory_enabled=True)

    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("memory_enabled") is True
        assert "facts_count" in data
        assert isinstance(data["facts_count"], int)


def test_mcp_memory_tools_roundtrip(tmp_path: Path) -> None:
    """Test edgemem_remember, edgemem_recall, edgemem_list_facts, and edgemem_forget via /mcp."""
    app, _, _ = _make_app(tmp_path, memory_enabled=True)

    with TestClient(app) as client:
        # 1. MCP initialize
        init_req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0"},
            },
        }
        resp = client.post(
            "/mcp",
            json=init_req,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        assert resp.status_code == 200
        session_id = resp.headers.get("mcp-session-id")

        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if session_id:
            headers["mcp-session-id"] = session_id

        # Initialized notification
        client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            headers=headers,
        )

        # 2. tools/list
        tools_resp = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            headers=headers,
        )
        assert tools_resp.status_code == 200
        tools_list = tools_resp.json()["result"]["tools"]
        tool_names = {t["name"] for t in tools_list}

        for required in ("edgemem_remember", "edgemem_recall", "edgemem_list_facts", "edgemem_forget"):
            assert required in tool_names

        # 3. Call edgemem_remember
        rem_call = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "edgemem_remember",
                "arguments": {
                    "key": "GLOBAL_SECRET",
                    "value": "super_secret_token_12345",
                    "kind": "constant",
                    "scope": "global",
                    "pinned": True,
                },
            },
        }
        rem_resp = client.post("/mcp", json=rem_call, headers=headers)
        assert rem_resp.status_code == 200
        res_data = rem_resp.json()["result"]
        content_text = res_data.get("content", [{}])[0].get("text", "")
        remember_out = json.loads(content_text) if content_text else res_data
        assert remember_out.get("status") == "ok"
        fact_id = remember_out.get("fact_id")
        assert fact_id is not None

        # 4. Call edgemem_list_facts
        list_call = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "edgemem_list_facts", "arguments": {}},
        }
        list_resp = client.post("/mcp", json=list_call, headers=headers)
        assert list_resp.status_code == 200
        list_data = json.loads(list_resp.json()["result"]["content"][0]["text"])
        stored_keys = [f["key"] for f in list_data["facts"]]
        assert "GLOBAL_SECRET" in stored_keys

        # 5. Call edgemem_recall
        recall_call = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "edgemem_recall", "arguments": {"query": "GLOBAL_SECRET"}},
        }
        recall_resp = client.post("/mcp", json=recall_call, headers=headers)
        assert recall_resp.status_code == 200
        recall_data = json.loads(recall_resp.json()["result"]["content"][0]["text"])
        assert len(recall_data["facts"]) >= 1
        assert recall_data["facts"][0]["key"] == "GLOBAL_SECRET"

        # 6. Call edgemem_forget
        forget_call = {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {"name": "edgemem_forget", "arguments": {"fact_id": fact_id}},
        }
        forget_resp = client.post("/mcp", json=forget_call, headers=headers)
        assert forget_resp.status_code == 200
        forget_data = json.loads(forget_resp.json()["result"]["content"][0]["text"])
        assert forget_data.get("status") == "ok"

        # 7. Verify deletion in list
        list_resp2 = client.post("/mcp", json=list_call, headers=headers)
        list_data2 = json.loads(list_resp2.json()["result"]["content"][0]["text"])
        stored_keys2 = [f["key"] for f in list_data2["facts"]]
        assert "GLOBAL_SECRET" not in stored_keys2


def test_cli_memory_commands(tmp_path: Path) -> None:
    """Test CLI commands: memory list, memory search, memory clear."""
    db_file = tmp_path / "cli_mem.db"
    config_file = tmp_path / "edgemem.toml"
    config_file.write_text(
        f"""
[server]
host = "127.0.0.1"
port = 8080

[upstream]
base_url = "http://127.0.0.1:11434/v1"

[memory]
enabled = true
db_path = "{db_file.as_posix()}"
""",
        encoding="utf-8",
    )

    from edgemem.memory.store import MemoryStore

    with MemoryStore(db_file) as store:
        store.upsert_fact("sess_test", "API_TIMEOUT", "60", kind="constant", pinned=True)
        store.upsert_fact("sess_test", "MAX_RETRY", "3", kind="constant", pinned=False)

    runner = CliRunner()

    # List facts
    res_list = runner.invoke(cli, ["memory", "list", "--config", str(config_file)])
    assert res_list.exit_code == 0
    assert "API_TIMEOUT" in res_list.output
    assert "MAX_RETRY" in res_list.output

    # Search facts
    res_search = runner.invoke(cli, ["memory", "search", "TIMEOUT", "--config", str(config_file)])
    assert res_search.exit_code == 0
    assert "API_TIMEOUT" in res_search.output

    # Clear facts for session
    res_clear = runner.invoke(cli, ["memory", "clear", "--session", "sess_test", "--config", str(config_file)])
    assert res_clear.exit_code == 0
    assert "Cleared 2 facts" in res_clear.output

    # List after clear
    res_list2 = runner.invoke(cli, ["memory", "list", "--config", str(config_file)])
    assert res_list2.exit_code == 0
    assert "No facts stored." in res_list2.output


def test_middle_turn_generic_boilerplate_is_compacted(tmp_path: Path) -> None:
    """Test verbose middle turn with generic Python/pytest boilerplate is compacted."""
    app, transport, _ = _make_app(
        tmp_path,
        memory_enabled=True,
        compact_threshold_tokens=0,
        compact_min_turn_tokens=0,
    )
    boilerplate_turn = (
        "```python\n"
        "def test_helper(data: dict) -> bool:\n"
        "    if not data:\n"
        "        raise ValueError('invalid data')\n"
        "    config = data.get('config', {})\n"
        "    result = config.get('status')\n"
        "    assert result is not None\n"
        "    return True\n"
        "```\n"
        "This function tests config data items and checks output results using pytest assertions."
    )
    messages = [
        {"role": "user", "content": "Rule: DatabaseConnection must be a singleton."},
        {"role": "assistant", "content": "Acknowledged."},
        {"role": "user", "content": boilerplate_turn},
        {"role": "assistant", "content": "Test helper noted with assert and ValueError handling."},
        {
            "role": "user",
            "content": (
                "Write a Python class using def, ValueError, assert, result, data, and config "
                "to implement DatabaseConnection."
            ),
        },
    ]
    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json={"messages": messages})
        assert resp.status_code == 200

    assert len(transport.recorded_bodies) == 1
    sent_messages = transport.recorded_bodies[0]["messages"]
    assert sent_messages[3]["content"] == "[Compacted earlier turn]"
    assert sent_messages[4]["content"] == "[Compacted earlier turn]"
    assert not any("def test_helper" in m.get("content", "") for m in sent_messages[1:-1])


def test_middle_turn_domain_function_preserved(tmp_path: Path) -> None:
    """Test middle turn defining a specific domain function called in final prompt is preserved."""
    app, transport, _ = _make_app(
        tmp_path,
        memory_enabled=True,
        compact_threshold_tokens=0,
        compact_min_turn_tokens=0,
    )
    domain_function_turn = (
        "Here is the cryptographic helper function:\n"
        "```python\n"
        "def compute_merkle_root(leaf_hashes: list[str]) -> str:\n"
        "    import hashlib\n"
        "    current = list(leaf_hashes)\n"
        "    while len(current) > 1:\n"
        "        next_level = []\n"
        "        for i in range(0, len(current), 2):\n"
        "            left = current[i]\n"
        "            right = current[i + 1] if i + 1 < len(current) else left\n"
        "            next_level.append(hashlib.sha256((left + right).encode()).hexdigest())\n"
        "        current = next_level\n"
        "    return current[0] if current else ''\n"
        "```"
    )
    messages = [
        {"role": "user", "content": "Rule: All blocks must be cryptographically signed."},
        {"role": "assistant", "content": "Acknowledged."},
        {"role": "user", "content": domain_function_turn},
        {"role": "assistant", "content": "compute_merkle_root helper registered."},
        {
            "role": "user",
            "content": "Now implement BlockValidator and call compute_merkle_root to verify block transactions.",
        },
    ]
    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json={"messages": messages})
        assert resp.status_code == 200

    assert len(transport.recorded_bodies) == 1
    sent_messages = transport.recorded_bodies[0]["messages"]
    assert any("def compute_merkle_root" in m.get("content", "") for m in sent_messages)
    assert sent_messages[3]["content"] == domain_function_turn


def test_short_conversation_default_config_skips_compaction(tmp_path: Path) -> None:
    """Test short conversation with default MemoryConfig compacts 0 turns but extracts constraints."""
    app, transport, _ = _make_app(tmp_path, memory_enabled=True)
    messages = [
        {"role": "user", "content": "Rule: TransactionRecord must be strictly immutable and frozen."},
        {"role": "assistant", "content": "Acknowledged. TransactionRecord will be immutable."},
        {
            "role": "user",
            "content": "Please also add a string helper method to sanitize user inputs before logging.",
        },
        {
            "role": "assistant",
            "content": "Added sanitize_input helper method.",
        },
        {
            "role": "user",
            "content": "Now implement TransactionRecord and ensure it respects the Turn 1 rule.",
        },
    ]
    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json={"messages": messages})
        assert resp.status_code == 200

    assert len(transport.recorded_bodies) == 1
    sent_messages = transport.recorded_bodies[0]["messages"]

    assert not any("[Compacted earlier turn]" in m.get("content", "") for m in sent_messages)
    assert any("sanitize user inputs" in m.get("content", "") for m in sent_messages)
    assert any("Added sanitize_input" in m.get("content", "") for m in sent_messages)

    first_content = sent_messages[0]["content"]
    assert "[EdgeMem Active Memory]" in first_content or "--- EdgeMem Core Memory ---" in first_content
    assert "strictly immutable and frozen" in first_content


def test_stress_tasks_compaction_net_token_reduction() -> None:
    """Fast unit test running MemoryMiddleware directly on all 20 stress tasks."""
    from edgemem.memory.store import MemoryStore
    from edgemem.proxy.middleware import MemoryMiddleware, _get_message_text, estimate_tokens

    stress_tasks_file = Path(__file__).parent.parent / "bench" / "stress_tasks.json"
    assert stress_tasks_file.exists()

    with open(stress_tasks_file, encoding="utf-8") as f:
        tasks = json.load(f)

    assert len(tasks) == 20

    for task in tasks:
        store = MemoryStore(":memory:")
        config = MemoryConfig(
            enabled=True,
            compact_middle_turns=True,
            keep_recent_turns=2,
            inject_constraints=True,
            auto_extract=True,
        )
        mid = MemoryMiddleware(store, config)

        file_ctx = "\n".join(f"# {p}\n{c}" for p, c in task.get("files", {}).items())
        sys_msg = "You are an expert Python developer.\n\n" + file_ctx
        raw_messages = [{"role": "system", "content": sys_msg}] + [
            {"role": "user", "content": turn} for turn in task["turns"]
        ]
        raw_tokens = sum(estimate_tokens(_get_message_text(m)) for m in raw_messages)

        body = {"messages": raw_messages}
        mod_body, tel = mid.process_request(body, {})
        mod_tokens = sum(estimate_tokens(_get_message_text(m)) for m in mod_body["messages"])

        assert tel["compacted_turns"] > 0, f"Task {task['id']} compacted 0 turns"
        assert mod_tokens < raw_tokens, (
            f"Task {task['id']} failed to achieve net token reduction: "
            f"raw={raw_tokens}, mod={mod_tokens}"
        )
