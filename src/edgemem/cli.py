"""EdgeMem CLI - console entrypoint.

Supports: edgemem serve, edgemem init, edgemem status, edgemem stats, edgemem slice, edgemem memory, edgemem unload.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import click

from edgemem.config import EdgeMemConfig, default_toml_content, load_config


@click.group()
@click.version_option(package_name="edgemem")
def cli() -> None:
    """EdgeMem - Lightweight unified memory & context proxy."""


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
    click.echo(f"EdgeMem proxy -> {cfg.upstream_base_url} (model: {cfg.model})")
    click.echo(f"Listening on http://{cfg.host}:{cfg.port}")
    click.echo(f"MCP endpoint: http://{cfg.host}:{cfg.port}/mcp")
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")


@cli.command()
@click.option(
    "--preset",
    type=click.Choice(["hybrid", "local", "nvidia", "gemini"], case_sensitive=False),
    default="hybrid",
    show_default=True,
    help="Configuration preset (hybrid, local, nvidia, gemini).",
)
@click.option(
    "--output",
    "-o",
    "--path",
    "output",
    default=Path("edgemem.toml"),
    type=click.Path(path_type=Path),
    show_default=True,
    help="Output configuration file path.",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    default=False,
    help="Overwrite existing configuration file.",
)
def init(preset: str, output: Path, force: bool) -> None:
    """Generate an edgemem.toml configuration file."""
    if output.exists() and not force:
        click.echo(
            f"Config file already exists at {output}. Use --force to overwrite.",
            err=True,
        )
        sys.exit(1)

    content = default_toml_content(preset)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8", newline="\n")
    click.echo(f"Created {output} (preset: {preset})")
    click.echo("Start EdgeMem with:")
    preset_lower = preset.lower()
    env_set = "$env:" if sys.platform == "win32" else "export "
    if preset_lower == "nvidia":
        click.echo(f'  {env_set}NVIDIA_API_KEY="<your-key>"')
    elif preset_lower == "gemini":
        click.echo(f'  {env_set}GEMINI_API_KEY="<your-key>"')
    elif preset_lower == "hybrid":
        click.echo(f'  {env_set}NVIDIA_API_KEY="<your-key>"  (optional for cloud fallback)')
    click.echo("  edgemem serve")


@cli.command()
@click.option(
    "--days",
    default=None,
    type=int,
    help="Filter telemetry events to the past N days.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="Output telemetry summary as formatted JSON.",
)
@click.option(
    "--reset",
    is_flag=True,
    default=False,
    help="Reset recorded request telemetry.",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Path to edgemem.toml configuration file.",
)
def stats(
    days: int | None,
    as_json: bool,
    reset: bool,
    config_path: Path | None,
) -> None:
    """Show or reset persistent request telemetry."""
    cfg = load_config(config_path)
    from edgemem.proxy.telemetry import TelemetryStore

    with TelemetryStore(cfg.memory.db_path) as store:
        if reset:
            count = store.reset()
            click.echo(f"Cleared {count} telemetry events.")
            return

        summary = store.summary(days=days)

        if as_json:
            click.echo(json.dumps(summary, indent=2))
            return

        window_str = f"past {days} days" if days is not None else "all time"
        click.echo(f"EdgeMem Telemetry Summary ({window_str})")
        click.echo("=" * 44)
        click.echo("Requests:")
        click.echo(f"  Total:               {summary['total_requests']}")
        click.echo(f"  Local:               {summary['local_requests']}")
        click.echo(f"  Cloud:               {summary['cloud_requests']}")
        click.echo(f"  Local Offload:       {summary['local_offload_pct']:.2f}%")
        click.echo(f"  Fallbacks:           {summary['fallback_escalations']}")
        click.echo("")
        click.echo("Context Savings:")
        click.echo(f"  Raw Tokens:          {summary['total_tokens_raw']:,}")
        click.echo(f"  Sent Tokens:         {summary['total_tokens_sent']:,}")
        click.echo(f"  Saved Tokens:        {summary['total_tokens_saved']:,}")
        click.echo(f"  Token Reduction:     {summary['token_reduction_pct']:.2f}%")
        click.echo(f"  Compacted Turns:     {summary['total_compacted_turns']}")
        click.echo("")
        click.echo("Privacy:")
        click.echo(f"  Secrets Redacted:    {summary['total_secrets_redacted']}")
        click.echo("")
        click.echo("Latency:")
        click.echo(f"  Mean Latency:        {summary['mean_latency_ms']:.2f} ms")


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
        memory_enabled = data.get("memory_enabled", False)
        facts_count = data.get("facts_count", 0)
        click.echo(f"Memory: {'enabled' if memory_enabled else 'disabled'} ({facts_count} facts)")
        privacy_enabled = data.get("privacy_enabled", False)
        secrets_redacted = data.get("secrets_redacted_total", 0)
        click.echo(f"Privacy: {'enabled' if privacy_enabled else 'disabled'} ({secrets_redacted} redacted)")
        auth_enabled = data.get("auth_enabled", False)
        click.echo(f"Auth: {'enabled' if auth_enabled else 'disabled (localhost only)'}")
        local_reqs = data.get("local_requests", 0)
        cloud_reqs = data.get("cloud_requests", 0)
        fallbacks = data.get("fallback_escalations", 0)
        click.echo(f"Routing: local={local_reqs}, cloud={cloud_reqs}, fallbacks={fallbacks}")
        if "system_memory" in data and isinstance(data["system_memory"], dict):
            sm = data["system_memory"]
            avail = sm.get("available_mb")
            total = sm.get("total_mb")
            pressure = sm.get("pressure", "unknown")
            if avail is not None and total is not None:
                click.echo(f"System RAM: {avail:.1f} MB free / {total:.1f} MB ({pressure})")
            else:
                click.echo(f"System RAM: pressure={pressure}")
        if "resources" in data and isinstance(data["resources"], dict):
            res = data["resources"]
            backend = res.get("local_backend", "auto")
            max_c = res.get("effective_max_ctx", "N/A")
            esc_t = res.get("effective_escalate_tokens", "N/A")
            click.echo(f"Context Budget: backend={backend}, max_ctx={max_c}, escalate_at={esc_t} tok")
        if "task_tiers" in data and isinstance(data["task_tiers"], dict):
            tt = data["task_tiers"]
            click.echo(f"Task Tiers: fast={tt.get('fast', 0)}, balanced={tt.get('balanced', 0)}, deep={tt.get('deep', 0)}")
        if "prefix_cache" in data and isinstance(data["prefix_cache"], dict):
            pc = data["prefix_cache"]
            click.echo(f"Prefix Cache: hits={pc.get('hits', 0)}, misses={pc.get('misses', 0)}, hit_rate={pc.get('hit_rate_pct', 0.0):.1f}%")
    except httpx.ConnectError:
        click.echo(f"EdgeMem not running at {url}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--model", default=None, help="Model name to unload (defaults to configured model).")
@click.option("--config", "config_path", default=None, type=click.Path(exists=True, path_type=Path), help="Path to edgemem.toml configuration file.")
def unload(model: str | None, config_path: Path | None) -> None:
    """Manually unload idle local model weights to free RAM/VRAM."""
    cfg = load_config(config_path)
    target_model = model or cfg.model
    backend = cfg.resources.local_backend
    if backend == "auto":
        from edgemem.proxy.budgeter import detect_local_backend

        backend = detect_local_backend(cfg.upstream_base_url, cfg.resources.local_backend)

    if backend != "ollama":
        click.echo(f"Idle unload only supported for Ollama backends (detected: {backend}).")
        return

    import httpx
    from edgemem.proxy.unloader import get_ollama_generate_url

    url = get_ollama_generate_url(cfg.upstream_base_url)
    try:
        resp = httpx.post(url, json={"model": target_model, "keep_alive": 0}, timeout=5.0)
        if resp.status_code == 200:
            click.echo(f"Unloaded model '{target_model}' via Ollama API ({url}).")
        else:
            click.echo(f"Failed to unload '{target_model}': HTTP {resp.status_code}", err=True)
            sys.exit(1)
    except httpx.ConnectError:
        click.echo(f"Ollama server not reachable at {url}", err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(f"Unload error: {exc}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--tokens", default=1024, show_default=True,
              help="Maximum tokens in the output repo map.")
@click.option("--query", default="", show_default=False,
              help="Natural language query to boost relevant definitions.")
@click.option("--focus", "focus_files", multiple=True, metavar="FILE",
              help="Relative paths to focus files (may be repeated).")
@click.option("--symbol", default=None,
              help="Target symbol name or qualified name for 1-hop AST call-graph slice.")
@click.option("--model", default="cl100k_base", show_default=True,
              help="tiktoken encoding for token counting.")
def slice(
    path: Path,
    tokens: int,
    query: str,
    focus_files: tuple[str, ...],
    symbol: str | None,
    model: str,
) -> None:
    """Slice a repository into a compact, ranked AST repo map.

    PATH  Root of the repository (or sub-directory) to slice.

    Prints the repo map to stdout.
    Prints token counts, compression %, and latency to stderr.

    Example:
        edgemem slice . --tokens 2048 --query "authentication" --focus src/auth.py
    """
    from edgemem.slicer.repomap import RepoMap, count_tokens

    t0 = time.perf_counter()

    rm = RepoMap(repo_path=path, max_tokens=tokens, model_name=model)
    try:
        if symbol:
            res = rm.slice_symbol(symbol=symbol, token_budget=tokens)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            click.echo(res.content)
            click.echo(
                f"\n[EdgeMem slice-symbol] symbol='{symbol}' found={res.found} "
                f"file={res.target_file} callees={res.callee_count} callers={res.caller_count} "
                f"sliced={res.sliced_tokens:,} tok raw={res.raw_tokens:,} tok fallback={res.fallback_used} "
                f"latency={elapsed_ms:.0f}ms",
                err=True,
            )
            return

        raw_tokens = rm.raw_token_count()
        repo_map = rm.get_repo_map(
            focus_files=list(focus_files) or None,
            mentioned_symbols=None,
            query=query,
        )
    finally:
        rm.close()

    elapsed_ms = (time.perf_counter() - t0) * 1000
    sliced_tokens = count_tokens(repo_map, model)
    compression_pct = (
        (raw_tokens - sliced_tokens) / raw_tokens * 100.0
        if raw_tokens > 0 else 0.0
    )

    click.echo(repo_map, nl=False)

    click.echo(
        f"\n[EdgeMem slice] raw={raw_tokens:,} tok  sliced={sliced_tokens:,} tok  "
        f"compression={compression_pct:.1f}%  latency={elapsed_ms:.0f}ms",
        err=True,
    )


@cli.group()
def memory() -> None:
    """Manage EdgeMem SQLite tiered memory store."""


@memory.command("list")
@click.option("--kind", default=None, help="Filter facts by kind (constraint, constant, signature, note).")
@click.option("--tier", default=None, type=click.Choice(["core", "episodic", "archival"], case_sensitive=False), help="Filter facts by tier.")
@click.option("--include-superseded", is_flag=True, default=False, help="Include superseded facts.")
@click.option("--limit", default=50, type=int, help="Max facts to return.")
@click.option("--config", "config_path", default=None, type=click.Path(exists=True, path_type=Path))
def memory_list(
    kind: str | None,
    tier: str | None,
    include_superseded: bool,
    limit: int,
    config_path: Path | None,
) -> None:
    """List stored facts in EdgeMem memory."""
    from edgemem.memory.store import MemoryStore

    cfg = load_config(config_path)
    with MemoryStore(cfg.memory.db_path) as store:
        facts = store.list_facts(
            kind=kind,
            tier=tier.lower() if tier else None,
            include_archival=True if (tier and tier.lower() == "archival") else False,
            include_superseded=include_superseded,
            limit=limit,
        )
        if not facts:
            click.echo("No facts stored.")
            return

        click.echo(f"Stored facts ({len(facts)}):")
        for f in facts:
            pin_badge = "[PINNED] " if f.pinned else ""
            click.echo(f"  #{f.id} [{f.kind}] [{f.tier}/{f.status}] {pin_badge}{f.key} = {f.value}")


@memory.command("search")
@click.argument("query")
@click.option("--all", "all_facts", is_flag=True, default=False, help="Include archival and superseded facts.")
@click.option("--limit", default=10, type=int, help="Max results.")
@click.option("--config", "config_path", default=None, type=click.Path(exists=True, path_type=Path))
def memory_search(query: str, all_facts: bool, limit: int, config_path: Path | None) -> None:
    """Search EdgeMem memory using BM25 FTS5 ranking."""
    from edgemem.memory.store import MemoryStore

    cfg = load_config(config_path)
    with MemoryStore(cfg.memory.db_path) as store:
        facts = store.search_facts(
            query,
            include_archival=all_facts,
            include_superseded=all_facts,
            limit=limit,
        )
        if not facts:
            click.echo("No matches found.")
            return

        click.echo(f"Search results for '{query}' ({len(facts)}):")
        for f in facts:
            pin_badge = "[PINNED] " if f.pinned else ""
            click.echo(f"  #{f.id} [{f.kind}] [{f.tier}/{f.status}] {pin_badge}{f.key} = {f.value}")


@memory.command("consolidate")
@click.option("--dry-run", is_flag=True, default=False, help="Calculate consolidation changes without applying them.")
@click.option("--config", "config_path", default=None, type=click.Path(exists=True, path_type=Path))
def memory_consolidate(dry_run: bool, config_path: Path | None) -> None:
    """Run an immediate memory consolidation sweep."""
    from edgemem.memory.consolidator import MemoryConsolidator
    from edgemem.memory.store import MemoryStore

    cfg = load_config(config_path)
    with MemoryStore(cfg.memory.db_path) as store:
        consolidator = MemoryConsolidator(store=store, config=cfg)
        report = consolidator.consolidate_once(dry_run=dry_run)
        prefix = "[DRY-RUN] " if report.dry_run else ""
        click.echo(f"{prefix}Consolidation complete (mode: {report.mode}):")
        click.echo(f"  Merged: {report.merged}")
        click.echo(f"  Superseded: {report.superseded}")
        click.echo(f"  Promoted: {report.promoted}")
        click.echo(f"  Demoted: {report.demoted}")
        click.echo(f"  Created: {report.created}")


@memory.command("history")
@click.argument("key")
@click.option("--session", default=None, help="Filter by session ID.")
@click.option("--config", "config_path", default=None, type=click.Path(exists=True, path_type=Path))
def memory_history(key: str, session: str | None, config_path: Path | None) -> None:
    """Display the versioned supersession timeline of a memory key."""
    from edgemem.memory.store import MemoryStore

    cfg = load_config(config_path)
    with MemoryStore(cfg.memory.db_path) as store:
        history = store.get_fact_history(key, session_id=session)
        if not history:
            click.echo(f"No history found for '{key}'.")
            return

        click.echo(f"History for '{key}' ({len(history)} versions):")
        for f in history:
            superseded_info = f" (superseded_by: #{f.superseded_by})" if f.superseded_by else ""
            click.echo(f"  #{f.id} [{f.kind}] [{f.tier}/{f.status}] {f.key} = {f.value}{superseded_info}")


@memory.command("clear")
@click.option("--session", default=None, help="Clear facts for a specific session ID only.")
@click.option("--config", "config_path", default=None, type=click.Path(exists=True, path_type=Path))
def memory_clear(session: str | None, config_path: Path | None) -> None:
    """Clear facts from EdgeMem memory."""
    from edgemem.memory.store import MemoryStore

    cfg = load_config(config_path)
    with MemoryStore(cfg.memory.db_path) as store:
        if session:
            count = store.clear_session(session)
            click.echo(f"Cleared {count} facts for session '{session}'.")
        else:
            count = store.clear_all()
            click.echo(f"Cleared all {count} facts from memory.")
