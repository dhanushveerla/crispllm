from __future__ import annotations

from dataclasses import dataclass
import urllib.parse

from edgemem.config import EdgeMemConfig
from edgemem.proxy.watchdog import RamPressureWatchdog


def detect_local_backend(upstream_base_url: str, configured: str = "auto") -> str:
    if configured and configured != "auto":
        return configured
    try:
        parsed = urllib.parse.urlsplit(upstream_base_url)
        port = parsed.port
        host_and_path = f"{parsed.netloc}{parsed.path}".lower()
        if port == 11434 or "ollama" in host_and_path:
            return "ollama"
        if "llama" in host_and_path:
            return "llama_server"
        return "openai"
    except Exception:
        return "openai"


@dataclass
class ContextDecision:
    pressure: str
    pressure_ctx_cap: int
    selected_num_ctx: int
    output_headroom: int
    effective_escalate_tokens: int
    should_escalate_cloud: bool
    escalate_reason: str | None
    downshift_model: str | None
    inject_ollama_num_ctx: bool
    trigger_local_unload: bool


class ContextBudgeter:
    def __init__(
        self,
        config: EdgeMemConfig,
        watchdog: RamPressureWatchdog | None = None,
    ) -> None:
        self.config = config
        self.watchdog = watchdog or RamPressureWatchdog(config.resources)

    @property
    def backend(self) -> str:
        return detect_local_backend(self.config.upstream_base_url, self.config.resources.local_backend)

    def pressure_cap(self, pressure: str) -> int:
        res = self.config.resources
        if pressure == "hard":
            return res.hard_min_ctx
        if pressure == "soft":
            return res.soft_max_ctx
        return res.max_local_ctx

    def escalate_threshold(self, pressure: str) -> int:
        res = self.config.resources
        cap = self.pressure_cap(pressure)
        headroom_cap = max(128, cap // 2)
        output_headroom = min(res.default_output_tokens, headroom_cap)
        dynamic_threshold = max(512, cap - output_headroom)
        return min(self.config.routing.escalate_token_threshold, dynamic_threshold)

    def evaluate(
        self,
        prompt_tokens: int,
        requested_max_tokens: int | None = None,
        pressure: str | None = None,
        cloud_available: bool = True,
        local_backend: str | None = None,
    ) -> ContextDecision:
        res = self.config.resources
        if not res.enabled:
            eff_esc = self.config.routing.escalate_token_threshold
            should_esc = bool(cloud_available and prompt_tokens >= eff_esc)
            return ContextDecision(
                pressure="normal",
                pressure_ctx_cap=res.max_local_ctx,
                selected_num_ctx=res.max_local_ctx,
                output_headroom=res.default_output_tokens,
                effective_escalate_tokens=eff_esc,
                should_escalate_cloud=should_esc,
                escalate_reason="token_threshold" if should_esc else None,
                downshift_model=None,
                inject_ollama_num_ctx=False,
                trigger_local_unload=False,
            )

        if pressure is None:
            pressure = self.watchdog.snapshot().pressure

        if local_backend is None:
            local_backend = detect_local_backend(self.config.upstream_base_url, res.local_backend)

        if pressure == "hard":
            pressure_ctx_cap = res.hard_min_ctx
        elif pressure == "soft":
            pressure_ctx_cap = res.soft_max_ctx
        else:
            pressure_ctx_cap = res.max_local_ctx

        headroom_cap = max(128, pressure_ctx_cap // 2)
        req_out = requested_max_tokens if requested_max_tokens and requested_max_tokens > 0 else res.default_output_tokens
        output_headroom = min(req_out, headroom_cap)

        dynamic_threshold = max(512, pressure_ctx_cap - output_headroom)
        effective_escalate_tokens = min(self.config.routing.escalate_token_threshold, dynamic_threshold)

        needed = prompt_tokens + output_headroom
        buckets = sorted(res.ctx_buckets) if res.ctx_buckets else [2048, 4096, 8192]
        candidate_bucket = buckets[-1]
        for b in buckets:
            if b >= needed:
                candidate_bucket = b
                break

        selected_num_ctx = min(max(buckets[0], candidate_bucket), pressure_ctx_cap)

        should_escalate_cloud = False
        escalate_reason: str | None = None
        trigger_local_unload = False
        downshift_model: str | None = None

        if pressure == "hard":
            if cloud_available:
                should_escalate_cloud = True
                escalate_reason = "ram_pressure_hard"
                trigger_local_unload = True
            else:
                should_escalate_cloud = False
                escalate_reason = None
                trigger_local_unload = False
                selected_num_ctx = res.hard_min_ctx
                downshift_model = res.downshift_model
        elif cloud_available and prompt_tokens > effective_escalate_tokens:
            should_escalate_cloud = True
            if prompt_tokens >= self.config.routing.escalate_token_threshold:
                escalate_reason = "token_threshold"
            else:
                escalate_reason = "context_budget_exceeded"
        elif pressure == "soft":
            should_escalate_cloud = False
            downshift_model = res.downshift_model
            selected_num_ctx = min(selected_num_ctx, res.soft_max_ctx)

        inject_ollama = bool(res.dynamic_num_ctx and local_backend == "ollama" and not should_escalate_cloud)

        return ContextDecision(
            pressure=pressure,
            pressure_ctx_cap=pressure_ctx_cap,
            selected_num_ctx=selected_num_ctx,
            output_headroom=output_headroom,
            effective_escalate_tokens=effective_escalate_tokens,
            should_escalate_cloud=should_escalate_cloud,
            escalate_reason=escalate_reason,
            downshift_model=downshift_model,
            inject_ollama_num_ctx=inject_ollama,
            trigger_local_unload=trigger_local_unload,
        )
