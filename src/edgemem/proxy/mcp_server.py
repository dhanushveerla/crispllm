"""Embedded MCP server for EdgeMem.

Exposes MCP tools via Streamable HTTP at POST /mcp,
composed inside the FastAPI app's lifespan.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mcp.server.mcpserver import MCPServer
from starlette.applications import Starlette

if TYPE_CHECKING:
    from edgemem.memory.store import MemoryStore
    from edgemem.proxy.telemetry import TelemetryLogger


def create_mcp_server(memory_store: MemoryStore | None = None) -> MCPServer:
    """Create the MCPServer instance with EdgeMem tools registered."""
    mcp = MCPServer("EdgeMem")

    # Tools reference telemetry and memory store attached via setters or constructor
    _telemetry_ref: dict[str, Any] = {"logger": None, "config": None}
    _memory_ref: dict[str, Any] = {"store": memory_store}

    def set_telemetry(logger: TelemetryLogger, config: Any) -> None:
        _telemetry_ref["logger"] = logger
        _telemetry_ref["config"] = config

    def set_memory(store: MemoryStore) -> None:
        _memory_ref["store"] = store

    def _get_store() -> MemoryStore:
        if _memory_ref["store"] is None:
            from edgemem.memory.store import MemoryStore

            _memory_ref["store"] = MemoryStore()
        return _memory_ref["store"]

    # Attach setters so the app module can wire them up
    mcp._edgemem_set_telemetry = set_telemetry  # type: ignore[attr-defined]
    mcp._edgemem_telemetry_ref = _telemetry_ref  # type: ignore[attr-defined]
    mcp._edgemem_set_memory = set_memory  # type: ignore[attr-defined]
    mcp._edgemem_memory_ref = _memory_ref  # type: ignore[attr-defined]

    @mcp.tool()
    async def edgemem_status() -> dict:
        """Return EdgeMem proxy status: RSS, upstream target, and recent telemetry."""
        from edgemem.proxy.telemetry import get_rss_mb

        ref = _telemetry_ref
        config = ref.get("config")
        logger = ref.get("logger")

        result: dict[str, Any] = {
            "proxy_rss_mb": round(get_rss_mb(), 2),
            "upstream_base_url": getattr(config, "upstream_base_url", "unknown"),
            "model": getattr(config, "model", "unknown"),
        }
        if logger is not None:
            result["telemetry_summary"] = logger.recent_summary()
        else:
            result["telemetry_summary"] = {"note": "telemetry not initialized"}
        return result

    @mcp.tool()
    async def edgemem_ping(message: str) -> dict:
        """Ping the EdgeMem proxy. Echoes the message back with status info."""
        from edgemem.proxy.telemetry import get_rss_mb

        return {
            "echo": message,
            "status": "ok",
            "proxy_rss_mb": round(get_rss_mb(), 2),
        }

    @mcp.tool()
    async def edgemem_remember(
        key: str,
        value: str,
        kind: str = "note",
        scope: str = "global",
        pinned: bool = True,
    ) -> dict:
        """Store a fact, constraint, or note in EdgeMem tiered memory.

        Args:
            key: Unique identifier/name for the fact (e.g. constant name or constraint topic).
            value: The content, signature, or directive text.
            kind: Type of fact: 'constraint', 'constant', 'signature', 'decision', 'note'.
            scope: Accessibility scope: 'global', 'project', or 'session'.
            pinned: Whether this fact is persistently injected into prompts.
        """
        store = _get_store()
        fid = store.upsert_fact(
            session_id="global",
            key=key,
            value=value,
            kind=kind,
            scope=scope,
            pinned=pinned,
        )
        return {"status": "ok", "fact_id": fid, "key": key}

    @mcp.tool()
    async def edgemem_recall(query: str, limit: int = 8) -> dict:
        """Search EdgeMem memory for relevant facts using BM25 FTS5 ranking.

        Args:
            query: Search query or prompt.
            limit: Maximum number of matches to return.
        """
        store = _get_store()
        facts = store.search_facts(query=query, limit=limit)
        return {
            "query": query,
            "facts": [
                {
                    "id": f.id,
                    "session_id": f.session_id,
                    "kind": f.kind,
                    "key": f.key,
                    "value": f.value,
                    "scope": f.scope,
                    "pinned": f.pinned,
                }
                for f in facts
            ],
        }

    @mcp.tool()
    async def edgemem_list_facts(kind: str | None = None, limit: int = 20) -> dict:
        """List stored facts in EdgeMem memory.

        Args:
            kind: Optional filter by kind: 'constraint', 'constant', 'signature', 'decision', 'note'.
            limit: Maximum facts to return.
        """
        store = _get_store()
        facts = store.list_facts(kind=kind, limit=limit)
        return {
            "facts": [
                {
                    "id": f.id,
                    "session_id": f.session_id,
                    "kind": f.kind,
                    "key": f.key,
                    "value": f.value,
                    "scope": f.scope,
                    "pinned": f.pinned,
                }
                for f in facts
            ]
        }

    @mcp.tool()
    async def edgemem_forget(fact_id: int) -> dict:
        """Delete a fact from EdgeMem memory by its ID.

        Args:
            fact_id: The integer ID of the fact to remove.
        """
        store = _get_store()
        deleted = store.delete_fact(fact_id)
        return {"status": "ok" if deleted else "not_found", "fact_id": fact_id}

    @mcp.tool()
    async def edgemem_repo_map(
        repo_path: str,
        max_tokens: int = 1024,
        query: str = "",
        focus_files: list[str] | None = None,
    ) -> dict:
        """
        Return a compact, ranked AST repository map for a given directory.

        The map is sliced to fit within `max_tokens` using PageRank over the
        symbol dependency graph (Aider RepoMap algorithm, Apache-2.0).

        Args:
            repo_path:   Absolute or relative path to the repository root.
            max_tokens:  Hard token budget for the output map (default 1024).
            query:       Natural language query to boost relevant definitions.
            focus_files: List of relative file paths to prioritise in ranking.

        Returns:
            dict with keys:
              - repo_map (str): The formatted repo map text.
              - sliced_tokens (int): Token count of the output map.
              - raw_tokens (int): Total token count of all source files.
              - compression_pct (float): Percentage reduction.
        """
        import time
        from pathlib import Path

        from edgemem.slicer.repomap import RepoMap, count_tokens

        t0 = time.perf_counter()
        rp = Path(repo_path).resolve()
        if not rp.is_dir():
            return {"error": f"Not a directory: {repo_path}"}

        rm = RepoMap(repo_path=rp, max_tokens=max_tokens)
        raw_tokens = rm.raw_token_count()
        repo_map_text = rm.get_repo_map(
            focus_files=focus_files,
            mentioned_symbols=None,
            query=query,
        )
        sliced_tokens = count_tokens(repo_map_text)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        compression_pct = (
            (raw_tokens - sliced_tokens) / raw_tokens * 100.0
            if raw_tokens > 0 else 0.0
        )
        return {
            "repo_map":        repo_map_text,
            "sliced_tokens":   sliced_tokens,
            "raw_tokens":      raw_tokens,
            "compression_pct": round(compression_pct, 2),
            "latency_ms":      round(elapsed_ms, 1),
        }

    @mcp.tool()
    async def edgemem_file_outline(file_path: str) -> dict:
        """
        Return the AST skeleton (class and function signatures) of a single file.

        Uses grep-ast TreeContext for supported languages with a regex fallback
        for unsupported extensions.

        Args:
            file_path: Absolute or relative path to the source file.

        Returns:
            dict with keys:
              - outline (str): Formatted signature skeleton.
              - file_path (str): Resolved absolute path.
              - language (str|None): Detected language or null.
        """
        from pathlib import Path

        from edgemem.slicer.repomap import file_outline
        from edgemem.slicer.tags import _lang_for_path

        fp = Path(file_path).resolve()
        if not fp.is_file():
            return {"error": f"Not a file: {file_path}"}

        lang = _lang_for_path(fp)
        outline_text = file_outline(fp, rel_path=str(fp))
        return {
            "outline":   outline_text,
            "file_path": str(fp),
            "language":  lang,
        }

    return mcp


def create_mcp_starlette_app(mcp: MCPServer) -> Starlette:
    """Create the Starlette sub-app for Streamable HTTP MCP.

    The returned app has an internal route at /mcp, so it must be
    mounted at "" (root) in the parent FastAPI app to avoid path doubling.
    """
    from mcp.server.transport_security import TransportSecuritySettings

    return mcp.streamable_http_app(
        streamable_http_path="/mcp",
        stateless_http=True,
        json_response=True,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=False,
        ),
    )
