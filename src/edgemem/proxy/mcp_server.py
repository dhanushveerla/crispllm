"""Embedded MCP server for EdgeMem.

Exposes MCP tools via Streamable HTTP at POST /mcp,
composed inside the FastAPI app's lifespan.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mcp.server.mcpserver import MCPServer
from starlette.applications import Starlette

if TYPE_CHECKING:
    from edgemem.proxy.telemetry import TelemetryLogger


def create_mcp_server() -> MCPServer:
    """Create the MCPServer instance with EdgeMem tools registered."""
    mcp = MCPServer("EdgeMem")

    # Tools are registered below; the actual implementations reference
    # a TelemetryLogger that gets attached after creation via set_telemetry().
    _telemetry_ref: dict[str, Any] = {"logger": None, "config": None}

    def set_telemetry(logger: TelemetryLogger, config: Any) -> None:
        _telemetry_ref["logger"] = logger
        _telemetry_ref["config"] = config

    # Attach setter so the app module can wire it up
    mcp._edgemem_set_telemetry = set_telemetry  # type: ignore[attr-defined]
    mcp._edgemem_telemetry_ref = _telemetry_ref  # type: ignore[attr-defined]

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
