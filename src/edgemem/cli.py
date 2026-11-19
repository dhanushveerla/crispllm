"""EdgeMem CLI — console entrypoint.

Supports: edgemem serve, edgemem init, edgemem status.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from edgemem.config import EdgeMemConfig, default_toml_content, load_config


@click.group()
@click.version_option(package_name="edgemem")
def cli() -> None:
    """EdgeMem — Lightweight unified memory & context proxy."""


@cli.command()
@click.option("--host", default=None, help="Bind host (overrides config).")
@click.option("--port", default=None, type=int, help="Bind port (overrides config).")
@click.option("--config", "config_path", default=None, type=click.Path(exists=True, path_type=Path))
def serve(host: str | None, port: int | None, config_path: Path | None) -> None:
    """Start the EdgeMem proxy server."""
    cfg = load_config(config_path)
    if host is not None:
        cfg.host = host
    if port is not None:
        cfg.port = port

    import uvicorn

    from edgemem.proxy.app import create_app

    app = create_app(cfg)
    click.echo(f"EdgeMem proxy → {cfg.upstream_base_url} (model: {cfg.model})")
    click.echo(f"Listening on http://{cfg.host}:{cfg.port}")
    click.echo(f"MCP endpoint: http://{cfg.host}:{cfg.port}/mcp")
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")


@cli.command()
@click.option("--path", default="edgemem.toml", type=click.Path(path_type=Path))
def init(path: Path) -> None:
    """Generate a default edgemem.toml config file."""
    if path.exists():
        click.echo(f"Config already exists: {path}", err=True)
        sys.exit(1)
    path.write_text(default_toml_content())
    click.echo(f"Created {path}")


@cli.command()
@click.option("--config", "config_path", default=None, type=click.Path(exists=True, path_type=Path))
def status(config_path: Path | None) -> None:
    """Show EdgeMem proxy status."""
    cfg = load_config(config_path)

    import httpx

    url = f"http://{cfg.host}:{cfg.port}/health"
    try:
        resp = httpx.get(url, timeout=3.0)
        data = resp.json()
        click.echo(f"Status: {data.get('status', 'unknown')}")
        click.echo(f"Upstream: {data.get('upstream_base_url', 'N/A')}")
        click.echo(f"Model: {data.get('model', 'N/A')}")
        click.echo(f"RSS: {data.get('rss_mb', 'N/A')} MB")
    except httpx.ConnectError:
        click.echo(f"EdgeMem not running at {url}", err=True)
        sys.exit(1)
