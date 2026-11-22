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
    decay_half_life_days: float = 14.0
    archive_threshold: float = 0.25
    promote_access_threshold: int = 5
    consolidation_enabled: bool = True
    consolidation_idle_s: float = 30.0
    consolidation_every_n_turns: int = 8
    llm_synthesis: bool = False
    prefix_stable_assembly: bool = True


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
    task_routing_enabled: bool = False
    fast_model: str | None = None
    fast_max_tokens: int = 1200
    escalate_deep_tasks: bool = True


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

    auth_token_env: str = "EDGEMEM_AUTH_TOKEN"
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
class ResourcesConfig:
    """Configuration for active RAM/VRAM watchdog, dynamic context budgeting, and idle model unloading."""

    enabled: bool = True
    local_backend: str = "auto"
    dynamic_num_ctx: bool = True
    ctx_buckets: list[int] = field(default_factory=lambda: [2048, 4096, 8192])
    max_local_ctx: int = 8192
    soft_max_ctx: int = 2048
    hard_min_ctx: int = 2048
    default_output_tokens: int = 512
    soft_free_mb: float = 1024.0
    hard_free_mb: float = 512.0
    hysteresis_mb: float = 64.0
    watchdog_poll_s: float = 2.0
    downshift_model: str | None = None
    idle_unload_enabled: bool = True
    idle_unload_s: float = 300.0
    llama_cache_prompt: bool = True


@dataclass
class SpeculativeConfig:
    """Configuration for Cloud Predicted Outputs and local llama-server speculative presets."""

    predicted_outputs_enabled: bool = False
    min_prediction_chars: int = 120
    max_prediction_chars: int = 24000
    min_prediction_lines: int = 4
    unsupported_provider_ttl_s: float = 1800.0
    local_mode: str = "ngram-simple"
    ngram_simple_size_n: int = 12
    ngram_simple_size_m: int = 32
    ngram_simple_min_hits: int = 1
    ngram_mod_n_match: int = 16
    ngram_mod_n_min: int = 8
    ngram_mod_n_max: int = 32
    draft_model_path: str | None = None
    draft_n_max: int = 4
    draft_p_min: float = 0.75
    draft_gpu_layers: int = 99


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
    resources: ResourcesConfig = field(default_factory=ResourcesConfig)
    speculative: SpeculativeConfig = field(default_factory=SpeculativeConfig)

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
    resources = raw.get("resources", {})
    speculative = raw.get("speculative", {})

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
        decay_half_life_days=float(
            memory.get("decay_half_life_days", MemoryConfig.decay_half_life_days)
        ),
        archive_threshold=float(
            memory.get("archive_threshold", MemoryConfig.archive_threshold)
        ),
        promote_access_threshold=int(
            memory.get("promote_access_threshold", MemoryConfig.promote_access_threshold)
        ),
        consolidation_enabled=bool(
            memory.get("consolidation_enabled", MemoryConfig.consolidation_enabled)
        ),
        consolidation_idle_s=float(
            memory.get("consolidation_idle_s", MemoryConfig.consolidation_idle_s)
        ),
        consolidation_every_n_turns=int(
            memory.get("consolidation_every_n_turns", MemoryConfig.consolidation_every_n_turns)
        ),
        llm_synthesis=bool(
            memory.get("llm_synthesis", MemoryConfig.llm_synthesis)
        ),
        prefix_stable_assembly=bool(
            memory.get("prefix_stable_assembly", MemoryConfig.prefix_stable_assembly)
        ),
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
        task_routing_enabled=bool(
            routing.get("task_routing_enabled", RoutingConfig.task_routing_enabled)
        ),
        fast_model=routing.get("fast_model", RoutingConfig.fast_model),
        fast_max_tokens=int(
            routing.get("fast_max_tokens", RoutingConfig.fast_max_tokens)
        ),
        escalate_deep_tasks=bool(
            routing.get("escalate_deep_tasks", RoutingConfig.escalate_deep_tasks)
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

    resources_cfg = ResourcesConfig(
        enabled=bool(resources.get("enabled", ResourcesConfig.enabled)),
        local_backend=str(resources.get("local_backend", ResourcesConfig.local_backend)),
        dynamic_num_ctx=bool(resources.get("dynamic_num_ctx", ResourcesConfig.dynamic_num_ctx)),
        ctx_buckets=list(resources.get("ctx_buckets", [2048, 4096, 8192])),
        max_local_ctx=int(resources.get("max_local_ctx", ResourcesConfig.max_local_ctx)),
        soft_max_ctx=int(resources.get("soft_max_ctx", ResourcesConfig.soft_max_ctx)),
        hard_min_ctx=int(resources.get("hard_min_ctx", ResourcesConfig.hard_min_ctx)),
        default_output_tokens=int(resources.get("default_output_tokens", ResourcesConfig.default_output_tokens)),
        soft_free_mb=float(resources.get("soft_free_mb", ResourcesConfig.soft_free_mb)),
        hard_free_mb=float(resources.get("hard_free_mb", ResourcesConfig.hard_free_mb)),
        hysteresis_mb=float(resources.get("hysteresis_mb", ResourcesConfig.hysteresis_mb)),
        watchdog_poll_s=float(resources.get("watchdog_poll_s", ResourcesConfig.watchdog_poll_s)),
        downshift_model=resources.get("downshift_model", ResourcesConfig.downshift_model),
        idle_unload_enabled=bool(resources.get("idle_unload_enabled", ResourcesConfig.idle_unload_enabled)),
        idle_unload_s=float(resources.get("idle_unload_s", ResourcesConfig.idle_unload_s)),
        llama_cache_prompt=bool(resources.get("llama_cache_prompt", ResourcesConfig.llama_cache_prompt)),
    )

    speculative_cfg = SpeculativeConfig(
        predicted_outputs_enabled=bool(
            speculative.get(
                "predicted_outputs_enabled",
                SpeculativeConfig.predicted_outputs_enabled,
            )
        ),
        min_prediction_chars=int(
            speculative.get(
                "min_prediction_chars",
                SpeculativeConfig.min_prediction_chars,
            )
        ),
        max_prediction_chars=int(
            speculative.get(
                "max_prediction_chars",
                SpeculativeConfig.max_prediction_chars,
            )
        ),
        min_prediction_lines=int(
            speculative.get(
                "min_prediction_lines",
                SpeculativeConfig.min_prediction_lines,
            )
        ),
        unsupported_provider_ttl_s=float(
            speculative.get(
                "unsupported_provider_ttl_s",
                SpeculativeConfig.unsupported_provider_ttl_s,
            )
        ),
        local_mode=str(
            speculative.get(
                "local_mode",
                SpeculativeConfig.local_mode,
            )
        ),
        ngram_simple_size_n=int(
            speculative.get(
                "ngram_simple_size_n",
                SpeculativeConfig.ngram_simple_size_n,
            )
        ),
        ngram_simple_size_m=int(
            speculative.get(
                "ngram_simple_size_m",
                SpeculativeConfig.ngram_simple_size_m,
            )
        ),
        ngram_simple_min_hits=int(
            speculative.get(
                "ngram_simple_min_hits",
                SpeculativeConfig.ngram_simple_min_hits,
            )
        ),
        ngram_mod_n_match=int(
            speculative.get(
                "ngram_mod_n_match",
                SpeculativeConfig.ngram_mod_n_match,
            )
        ),
        ngram_mod_n_min=int(
            speculative.get(
                "ngram_mod_n_min",
                SpeculativeConfig.ngram_mod_n_min,
            )
        ),
        ngram_mod_n_max=int(
            speculative.get(
                "ngram_mod_n_max",
                SpeculativeConfig.ngram_mod_n_max,
            )
        ),
        draft_model_path=speculative.get(
            "draft_model_path",
            SpeculativeConfig.draft_model_path,
        ),
        draft_n_max=int(
            speculative.get(
                "draft_n_max",
                SpeculativeConfig.draft_n_max,
            )
        ),
        draft_p_min=float(
            speculative.get(
                "draft_p_min",
                SpeculativeConfig.draft_p_min,
            )
        ),
        draft_gpu_layers=int(
            speculative.get(
                "draft_gpu_layers",
                SpeculativeConfig.draft_gpu_layers,
            )
        ),
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
        resources=resources_cfg,
        speculative=speculative_cfg,
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
