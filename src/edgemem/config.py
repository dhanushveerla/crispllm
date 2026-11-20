"""Configuration loading for EdgeMem.

Loads from ./edgemem.toml (project-local) or
platformdirs.user_config_path("edgemem") / "edgemem.toml" (user-global),
with sensible defaults for all values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import tomllib
from typing import Any

import platformdirs


def _default_config_paths() -> list[Path]:
    """Return ordered list of config file paths to search."""
    return [
        Path("edgemem.toml"),
        platformdirs.user_config_path("edgemem") / "edgemem.toml",
    ]


@dataclass
class MemoryConfig:
    """Configuration for EdgeMem SQLite-backed tiered memory."""

    enabled: bool = True
    max_injected_tokens: int = 384
    inject_constraints: bool = True
    auto_extract: bool = True
    compact_middle_turns: bool = True
    compact_threshold_tokens: int = 2000
    compact_min_turn_tokens: int = 150
    keep_recent_turns: int = 2
    db_path: str | None = None


@dataclass
class CloudUpstreamConfig:
    """Configuration for optional cloud fallback/escalation LLM provider."""

    enabled: bool = False
    base_url: str = ""
    api_key_env: str = ""
    model: str = ""
    extra_body: dict[str, Any] = field(default_factory=dict)
    timeout_s: float = 60.0

    @property
    def api_key(self) -> str | None:
        if not self.api_key_env:
            return None
        return os.environ.get(self.api_key_env)


@dataclass
class RoutingConfig:
    """Configuration for hybrid local-to-cloud request routing."""

    strategy: str = "local_first"
    escalate_token_threshold: int = 6000
    escalate_on_local_error: bool = True
    local_timeout_s: float = 30.0


@dataclass
class PrivacyConfig:
    """Configuration for zero-dependency secret and PII redaction."""

    enabled: bool = True
    redact_on_cloud_only: bool = True
    redact_secrets: bool = True
    redact_emails: bool = True
    redact_ips: bool = False
    extra_patterns: list[str] = field(default_factory=list)


@dataclass
class SecurityConfig:
    """Configuration for pre-release security, authentication, and tunnel isolation."""

    auth_token_env: str = ""
    auth_token: str = ""
    allow_remote_v1: bool = False
    allowed_repo_roots: list[str] = field(default_factory=list)

    def resolve_auth_token(self) -> str:
        """Resolve authentication token from env var or literal fallback."""
        if self.auth_token_env:
            val = os.environ.get(self.auth_token_env)
            if val is not None and val.strip():
                return val.strip()
        if self.auth_token:
            return self.auth_token.strip()
        return ""


@dataclass
class EdgeMemConfig:
    """Full EdgeMem configuration."""

    host: str = "127.0.0.1"
    port: int = 8080

    upstream_base_url: str = "http://127.0.0.1:11434/v1"
    api_key_env: str = ""
    model: str = "qwen3.5:2b-q4_K_M"

    extra_body: dict[str, Any] = field(default_factory=dict)

    telemetry_dir: str = ""

    memory: MemoryConfig = field(default_factory=MemoryConfig)
    cloud: CloudUpstreamConfig = field(default_factory=CloudUpstreamConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)

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
    """Load config from TOML file, falling back to defaults."""
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

    server = raw.get("server", {})
    upstream = raw.get("upstream", {})
    telemetry = raw.get("telemetry", {})
    memory = raw.get("memory", {})
    cloud = raw.get("cloud", {})
    routing = raw.get("routing", {})
    privacy = raw.get("privacy", {})
    security = raw.get("security", {})

    memory_cfg = MemoryConfig(
        enabled=memory.get("enabled", MemoryConfig.enabled),
        max_injected_tokens=memory.get(
            "max_injected_tokens", MemoryConfig.max_injected_tokens
        ),
        inject_constraints=memory.get(
            "inject_constraints", MemoryConfig.inject_constraints
        ),
        auto_extract=memory.get("auto_extract", MemoryConfig.auto_extract),
        compact_middle_turns=memory.get(
            "compact_middle_turns", MemoryConfig.compact_middle_turns
        ),
        compact_threshold_tokens=memory.get(
            "compact_threshold_tokens", MemoryConfig.compact_threshold_tokens
        ),
        compact_min_turn_tokens=memory.get(
            "compact_min_turn_tokens", MemoryConfig.compact_min_turn_tokens
        ),
        keep_recent_turns=memory.get(
            "keep_recent_turns", MemoryConfig.keep_recent_turns
        ),
        db_path=memory.get("db_path", MemoryConfig.db_path),
    )

    cloud_cfg = CloudUpstreamConfig(
        enabled=cloud.get("enabled", CloudUpstreamConfig.enabled),
        base_url=cloud.get("base_url", CloudUpstreamConfig.base_url),
        api_key_env=cloud.get("api_key_env", CloudUpstreamConfig.api_key_env),
        model=cloud.get("model", CloudUpstreamConfig.model),
        extra_body=cloud.get("extra_body", {}),
        timeout_s=float(cloud.get("timeout_s", CloudUpstreamConfig.timeout_s)),
    )

    routing_cfg = RoutingConfig(
        strategy=routing.get("strategy", RoutingConfig.strategy),
        escalate_token_threshold=int(
            routing.get(
                "escalate_token_threshold",
                RoutingConfig.escalate_token_threshold,
            )
        ),
        escalate_on_local_error=routing.get(
            "escalate_on_local_error", RoutingConfig.escalate_on_local_error
        ),
        local_timeout_s=float(
            routing.get("local_timeout_s", RoutingConfig.local_timeout_s)
        ),
    )

    privacy_cfg = PrivacyConfig(
        enabled=privacy.get("enabled", PrivacyConfig.enabled),
        redact_on_cloud_only=privacy.get(
            "redact_on_cloud_only", PrivacyConfig.redact_on_cloud_only
        ),
        redact_secrets=privacy.get("redact_secrets", PrivacyConfig.redact_secrets),
        redact_emails=privacy.get("redact_emails", PrivacyConfig.redact_emails),
        redact_ips=privacy.get("redact_ips", PrivacyConfig.redact_ips),
        extra_patterns=list(privacy.get("extra_patterns", [])),
    )

    security_cfg = SecurityConfig(
        auth_token_env=security.get("auth_token_env", SecurityConfig.auth_token_env),
        auth_token=security.get("auth_token", SecurityConfig.auth_token),
        allow_remote_v1=bool(
            security.get("allow_remote_v1", SecurityConfig.allow_remote_v1)
        ),
        allowed_repo_roots=list(security.get("allowed_repo_roots", [])),
    )

    return EdgeMemConfig(
        host=server.get("host", EdgeMemConfig.host),
        port=server.get("port", EdgeMemConfig.port),
        upstream_base_url=upstream.get(
            "base_url", EdgeMemConfig.upstream_base_url
        ),
        api_key_env=upstream.get("api_key_env", EdgeMemConfig.api_key_env),
        model=upstream.get("model", EdgeMemConfig.model),
        extra_body=upstream.get("extra_body", {}),
        telemetry_dir=telemetry.get("dir", EdgeMemConfig.telemetry_dir),
        memory=memory_cfg,
        cloud=cloud_cfg,
        routing=routing_cfg,
        privacy=privacy_cfg,
        security=security_cfg,
    )


def default_toml_content(preset: str = "hybrid") -> str:
    """Generate default edgemem.toml content for `edgemem init`."""
    preset_lower = preset.lower()
    if preset_lower == "hybrid":
        return """\
[server]
host = "127.0.0.1"
port = 8080

[upstream]
base_url = "http://127.0.0.1:11434/v1"
model = "qwen2.5-coder:7b"

[memory]
enabled = true
max_injected_tokens = 384
inject_constraints = true
auto_extract = true
compact_middle_turns = true
compact_threshold_tokens = 2000
compact_min_turn_tokens = 150
keep_recent_turns = 2

[cloud]
enabled = true
base_url = "https://integrate.api.nvidia.com/v1"
api_key_env = "NVIDIA_API_KEY"
model = "nvidia/nemotron-3.5-bolt-e4b"

[routing]
strategy = "local_first"
escalate_token_threshold = 6000
escalate_on_local_error = true
local_timeout_s = 30.0

[privacy]
enabled = true
redact_on_cloud_only = true
redact_secrets = true
redact_emails = true
redact_ips = false
"""
    if preset_lower == "local":
        return """\
[server]
host = "127.0.0.1"
port = 8080

[upstream]
base_url = "http://127.0.0.1:11434/v1"
model = "qwen2.5-coder:7b"

[memory]
enabled = true
max_injected_tokens = 384
inject_constraints = true
auto_extract = true
compact_middle_turns = true
compact_threshold_tokens = 2000
compact_min_turn_tokens = 150
keep_recent_turns = 2

[cloud]
enabled = false
base_url = ""
api_key_env = ""
model = ""

[routing]
strategy = "local_only"
escalate_token_threshold = 6000
escalate_on_local_error = true
local_timeout_s = 30.0

[privacy]
enabled = true
redact_on_cloud_only = true
redact_secrets = true
redact_emails = true
redact_ips = false
"""
    if preset_lower == "nvidia":
        return """\
[server]
host = "127.0.0.1"
port = 8080

[upstream]
base_url = "https://integrate.api.nvidia.com/v1"
api_key_env = "NVIDIA_API_KEY"
model = "nvidia/nemotron-3.5-bolt-e4b"

[memory]
enabled = true
max_injected_tokens = 384
inject_constraints = true
auto_extract = true
compact_middle_turns = true
compact_threshold_tokens = 2000
compact_min_turn_tokens = 150
keep_recent_turns = 2

[cloud]
enabled = false
base_url = ""
api_key_env = ""
model = ""

[routing]
strategy = "local_only"
escalate_token_threshold = 6000
escalate_on_local_error = true
local_timeout_s = 30.0

[privacy]
enabled = true
redact_on_cloud_only = true
redact_secrets = true
redact_emails = true
redact_ips = false
"""
    if preset_lower == "gemini":
        return """\
[server]
host = "127.0.0.1"
port = 8080

[upstream]
base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
api_key_env = "GEMINI_API_KEY"
model = "gemini-2.5-flash"

[memory]
enabled = true
max_injected_tokens = 384
inject_constraints = true
auto_extract = true
compact_middle_turns = true
compact_threshold_tokens = 2000
compact_min_turn_tokens = 150
keep_recent_turns = 2

[cloud]
enabled = false
base_url = ""
api_key_env = ""
model = ""

[routing]
strategy = "local_only"
escalate_token_threshold = 6000
escalate_on_local_error = true
local_timeout_s = 30.0

[privacy]
enabled = true
redact_on_cloud_only = true
redact_secrets = true
redact_emails = true
redact_ips = false
"""
    raise ValueError(f"Unknown preset: {preset}")
