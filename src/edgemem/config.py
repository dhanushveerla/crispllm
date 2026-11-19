"""Configuration loading for EdgeMem.

Loads from ./edgemem.toml (project-local) or
platformdirs.user_config_path("edgemem") / "edgemem.toml" (user-global),
with sensible defaults for all values.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import platformdirs


def _default_config_paths() -> list[Path]:
    """Return ordered list of config file paths to search."""
    return [
        Path("edgemem.toml"),
        platformdirs.user_config_path("edgemem") / "edgemem.toml",
    ]


@dataclass
class EdgeMemConfig:
    """Full EdgeMem configuration."""

    # Server
    host: str = "127.0.0.1"
    port: int = 8080

    # Upstream LLM
    upstream_base_url: str = "http://127.0.0.1:11434/v1"
    api_key_env: str = ""  # env var name holding the API key (empty = no auth)
    model: str = "qwen3.5:2b-q4_K_M"

    # Optional extra body fields sent with every upstream request
    # e.g. {"reasoning_effort": "none"}
    extra_body: dict[str, Any] = field(default_factory=dict)

    # Telemetry
    telemetry_dir: str = ""  # empty = use platformdirs default

    @property
    def upstream_api_key(self) -> str | None:
        """Resolve API key from the configured environment variable."""
        if not self.api_key_env:
            return None
        return os.environ.get(self.api_key_env)

    @property
    def telemetry_path(self) -> Path:
        """Resolved telemetry log directory."""
        if self.telemetry_dir:
            return Path(self.telemetry_dir)
        return Path(platformdirs.user_data_dir("edgemem"))


def load_config(path: Path | None = None) -> EdgeMemConfig:
    """Load config from TOML file, falling back to defaults.

    Args:
        path: Explicit path to config file. If None, searches default locations.

    Returns:
        Populated EdgeMemConfig.
    """
    raw: dict[str, Any] = {}

    if path is not None:
        candidates = [path]
    else:
        candidates = _default_config_paths()

    for candidate in candidates:
        try:
            with open(candidate, "rb") as f:
                raw = tomllib.load(f)
            break
        except FileNotFoundError:
            continue

    # Flatten [server] and [upstream] sections
    server = raw.get("server", {})
    upstream = raw.get("upstream", {})
    telemetry = raw.get("telemetry", {})

    return EdgeMemConfig(
        host=server.get("host", EdgeMemConfig.host),
        port=server.get("port", EdgeMemConfig.port),
        upstream_base_url=upstream.get("base_url", EdgeMemConfig.upstream_base_url),
        api_key_env=upstream.get("api_key_env", EdgeMemConfig.api_key_env),
        model=upstream.get("model", EdgeMemConfig.model),
        extra_body=upstream.get("extra_body", {}),
        telemetry_dir=telemetry.get("dir", EdgeMemConfig.telemetry_dir),
    )


def default_toml_content() -> str:
    """Generate default edgemem.toml content for `edgemem init`."""
    return """\
# EdgeMem Configuration
# See https://github.com/edgemem/edgemem for documentation.

[server]
host = "127.0.0.1"
port = 8080

[upstream]
base_url = "http://127.0.0.1:11434/v1"
# api_key_env = "OPENAI_API_KEY"  # env var name holding the API key
model = "qwen3.5:2b-q4_K_M"

# Extra body fields merged into every upstream request.
# Client-sent keys take precedence (never overwritten).
[upstream.extra_body]
reasoning_effort = "none"

[telemetry]
# dir = "/custom/path"  # defaults to platformdirs user_data_dir("edgemem")
"""
