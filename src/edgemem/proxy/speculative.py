from __future__ import annotations

from dataclasses import dataclass, field
import os
import re
import shlex
import sys
import threading
import time
from typing import Any, Callable

from edgemem.config import SpeculativeConfig
from edgemem.proxy.watchdog import RamPressureWatchdog

_EDIT_SIGNALS: tuple[str, ...] = (
    "refactor",
    "update",
    "modify",
    "fix",
    "rename",
    "replace",
    "edit",
    "rewrite",
    "patch",
    "convert",
    "change",
    "add docstring",
    "add type hint",
    "add error handling",
    "add logging",
    "migrate",
    "optimize",
)

_GREENFIELD_SIGNALS: tuple[str, ...] = (
    "write from scratch",
    "brainstorm",
    "explain what",
)

_CODE_BLOCK_RE = re.compile(r"```(?:[a-zA-Z0-9_\-\.\+]*\n)?(.*?)```", re.DOTALL)


@dataclass(frozen=True)
class PredictionDecision:
    inject: bool
    reason: str
    prediction_payload: dict[str, Any] | None = None
    char_count: int = 0


def _extract_text_from_message(msg: dict[str, Any]) -> str:
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                txt = p.get("text", "")
                if isinstance(txt, str):
                    parts.append(txt)
        return "".join(parts)
    return str(content or "")


class PredictedOutputsEngine:
    def __init__(
        self,
        config: SpeculativeConfig,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.clock = clock
        self._lock = threading.Lock()
        self._unsupported: dict[tuple[str, str], float] = {}

    def mark_unsupported(self, base_url: str, model: str) -> None:
        key = (base_url.rstrip("/"), model)
        with self._lock:
            self._unsupported[key] = self.clock() + self.config.unsupported_provider_ttl_s

    def is_unsupported(self, base_url: str, model: str) -> bool:
        key = (base_url.rstrip("/"), model)
        now = self.clock()
        with self._lock:
            if key in self._unsupported:
                exp = self._unsupported[key]
                if now < exp:
                    return True
                del self._unsupported[key]
            return False

    def evaluate(
        self,
        body: dict[str, Any],
        target_is_cloud: bool,
        base_url: str,
        model: str,
        header_mode: str | None = None,
    ) -> PredictionDecision:
        if header_mode is not None:
            hm = header_mode.strip().lower()
            if hm in ("off", "false", "0", "no"):
                return PredictionDecision(inject=False, reason="header_disabled")
            forced_enabled = hm in ("auto", "on", "true", "1", "yes")
        else:
            forced_enabled = False

        if not target_is_cloud:
            return PredictionDecision(inject=False, reason="local_target")

        if not forced_enabled and not self.config.predicted_outputs_enabled:
            return PredictionDecision(inject=False, reason="disabled")

        if self.is_unsupported(base_url, model):
            return PredictionDecision(inject=False, reason="provider_unsupported_cached")

        if "prediction" in body:
            return PredictionDecision(inject=False, reason="caller_provided")

        if ("tools" in body and bool(body["tools"])) or ("functions" in body and bool(body["functions"])):
            return PredictionDecision(inject=False, reason="incompatible_tools")

        if body.get("n", 1) is not None:
            try:
                if int(body.get("n", 1)) > 1:
                    return PredictionDecision(inject=False, reason="incompatible_n")
            except (ValueError, TypeError):
                pass

        if bool(body.get("logprobs")) or bool(body.get("top_logprobs")):
            return PredictionDecision(inject=False, reason="incompatible_logprobs")

        pres_pen = body.get("presence_penalty")
        freq_pen = body.get("frequency_penalty")
        if (pres_pen is not None and pres_pen != 0 and pres_pen != 0.0) or (
            freq_pen is not None and freq_pen != 0 and freq_pen != 0.0
        ):
            return PredictionDecision(inject=False, reason="incompatible_penalties")

        if body.get("response_format") is not None:
            return PredictionDecision(inject=False, reason="incompatible_response_format")

        messages = body.get("messages", [])
        if not isinstance(messages, list):
            return PredictionDecision(inject=False, reason="no_user_message")

        user_messages = [m for m in messages if isinstance(m, dict) and m.get("role") == "user"]
        if not user_messages:
            return PredictionDecision(inject=False, reason="no_user_message")

        latest_user_msg = user_messages[-1]
        latest_text = _extract_text_from_message(latest_user_msg)

        text_lower = latest_text.lower()
        has_edit = any(sig in text_lower for sig in _EDIT_SIGNALS)
        has_greenfield = any(gf in text_lower for gf in _GREENFIELD_SIGNALS)
        if not has_edit or (has_greenfield and not has_edit):
            return PredictionDecision(inject=False, reason="no_edit_intent")

        blocks = _CODE_BLOCK_RE.findall(latest_text)
        if not blocks and len(user_messages) >= 2:
            prev_user_text = _extract_text_from_message(user_messages[-2])
            blocks = _CODE_BLOCK_RE.findall(prev_user_text)

        if not blocks:
            return PredictionDecision(inject=False, reason="no_code_block")

        valid_blocks: list[str] = []
        short_found = False
        long_found = False

        for b in blocks:
            code = b.strip()
            char_count = len(code)
            lines = code.splitlines()
            line_count = len(lines)

            if char_count < self.config.min_prediction_chars or line_count < self.config.min_prediction_lines:
                short_found = True
            elif char_count > self.config.max_prediction_chars:
                long_found = True
            else:
                valid_blocks.append(code)

        if valid_blocks:
            selected = max(valid_blocks, key=len)
            return PredictionDecision(
                inject=True,
                reason="predicted_outputs",
                prediction_payload={"type": "content", "content": selected},
                char_count=len(selected),
            )

        if short_found and not long_found:
            return PredictionDecision(inject=False, reason="code_too_short")
        if long_found and not short_found:
            return PredictionDecision(inject=False, reason="code_too_long")
        return PredictionDecision(inject=False, reason="code_too_short")


class SpeculativeStats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.cloud_injected: int = 0
        self.cloud_skipped: int = 0
        self.cloud_fallback_retries: int = 0
        self.cloud_accepted_tokens: int = 0
        self.cloud_rejected_tokens: int = 0
        self.local_draft_tokens: int = 0
        self.local_accepted_tokens: int = 0

    def inc_injected(self) -> None:
        with self._lock:
            self.cloud_injected += 1

    def inc_skipped(self) -> None:
        with self._lock:
            self.cloud_skipped += 1

    def inc_fallback_retry(self) -> None:
        with self._lock:
            self.cloud_fallback_retries += 1

    def record_usage_payload(self, payload: dict[str, Any], is_cloud: bool = True) -> None:
        if not isinstance(payload, dict):
            return
        usage = payload.get("usage")
        timings = payload.get("timings")
        with self._lock:
            if isinstance(usage, dict):
                details = usage.get("completion_tokens_details")
                if isinstance(details, dict):
                    acc = details.get("accepted_prediction_tokens", 0)
                    rej = details.get("rejected_prediction_tokens", 0)
                    if isinstance(acc, (int, float)):
                        self.cloud_accepted_tokens += int(acc)
                    if isinstance(rej, (int, float)):
                        self.cloud_rejected_tokens += int(rej)
            if isinstance(timings, dict):
                draft_n = timings.get("draft_n", 0)
                draft_acc = timings.get("draft_n_accepted", 0)
                if isinstance(draft_n, (int, float)):
                    self.local_draft_tokens += int(draft_n)
                if isinstance(draft_acc, (int, float)):
                    self.local_accepted_tokens += int(draft_acc)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            cloud_total = self.cloud_accepted_tokens + self.cloud_rejected_tokens
            cloud_pct = (
                round(self.cloud_accepted_tokens / cloud_total * 100.0, 1)
                if cloud_total > 0
                else None
            )
            local_pct = (
                round(self.local_accepted_tokens / self.local_draft_tokens * 100.0, 1)
                if self.local_draft_tokens > 0
                else None
            )
            return {
                "cloud_injected": self.cloud_injected,
                "cloud_skipped": self.cloud_skipped,
                "cloud_fallback_retries": self.cloud_fallback_retries,
                "cloud_accepted_tokens": self.cloud_accepted_tokens,
                "cloud_rejected_tokens": self.cloud_rejected_tokens,
                "cloud_acceptance_pct": cloud_pct,
                "local_draft_tokens": self.local_draft_tokens,
                "local_accepted_tokens": self.local_accepted_tokens,
                "local_acceptance_pct": local_pct,
            }


def _render_powershell(args: list[str]) -> str:
    rendered: list[str] = []
    for a in args:
        if not a:
            rendered.append('""')
        elif any(c in a for c in " \t\n\r\"'$`"):
            escaped = a.replace("`", "``").replace('"', '`"')
            rendered.append(f'"{escaped}"')
        else:
            rendered.append(a)
    return " ".join(rendered)


@dataclass
class LlamaSpeculativePreset:
    requested_mode: str
    effective_mode: str
    profile: str
    downgrade_reason: str | None
    args: list[str]
    estimated_extra_ram_mb: int

    def render_shell_command(self, platform_name: str | None = None) -> str:
        plat = platform_name if platform_name is not None else sys.platform
        if plat == "win32":
            return _render_powershell(self.args)
        return shlex.join(self.args)


def build_llama_server_command(
    model_path: str,
    draft_model_path: str | None = None,
    mode: str = "ngram-simple",
    profile: str = "auto",
    binary: str = "llama-server",
    device: str | None = None,
    ctx_size: int | None = None,
    host: str = "127.0.0.1",
    port: int = 8081,
    config: SpeculativeConfig | None = None,
    watchdog: RamPressureWatchdog | None = None,
    available_mb: float | None = None,
    pressure: str | None = None,
) -> LlamaSpeculativePreset:
    cfg = config or SpeculativeConfig()

    avail = available_mb
    press = pressure
    if watchdog is not None:
        snap = watchdog.snapshot()
        if avail is None and snap.memory is not None:
            avail = snap.memory.available_mb
        if press is None:
            press = snap.pressure

    norm_profile = profile.strip().lower()
    if norm_profile == "auto":
        if avail is not None and avail < 2048.0:
            resolved_profile = "t1-cpu"
        else:
            resolved_profile = "t2-gpu"
    else:
        resolved_profile = norm_profile

    req_mode = mode.strip().lower()
    effective_mode = req_mode
    downgrade_reason: str | None = None

    if req_mode in ("draft", "hybrid"):
        actual_draft_path = draft_model_path or cfg.draft_model_path
        if not actual_draft_path:
            effective_mode = "ngram-simple"
            downgrade_reason = "missing_draft_model"
        elif resolved_profile == "t1-cpu":
            effective_mode = "ngram-simple"
            downgrade_reason = "t1_cpu_budget_guard"
        elif press in ("soft", "hard") or (avail is not None and avail < 1024.0):
            effective_mode = "ngram-simple"
            downgrade_reason = "ram_pressure_guard"
        else:
            effective_mode = req_mode
            downgrade_reason = None
    elif req_mode not in ("ngram-simple", "ngram-mod", "none"):
        effective_mode = "ngram-simple"

    profile_args: list[str] = []
    effective_ctx = ctx_size or 4096
    if resolved_profile == "t1-cpu":
        effective_ctx = ctx_size or 2048
        profile_args.extend(["--n-gpu-layers", "0", "--threads", "2", "--threads-batch", "2", "-c", str(effective_ctx)])
    elif resolved_profile == "t2-vulkan":
        vulkan_dev = device or "Vulkan1"
        profile_args.extend([
            "--device",
            vulkan_dev,
            "--n-gpu-layers",
            "99",
            "-c",
            str(effective_ctx),
            "-fa",
            "on",
            "--cache-type-k",
            "q8_0",
            "--cache-type-v",
            "q8_0",
        ])
    else:
        profile_args.extend(["--n-gpu-layers", "99", "-c", str(effective_ctx), "-fa", "on"])

    args: list[str] = [binary, "-m", model_path, "-np", "1", "--host", host, "--port", str(port)]
    args.extend(profile_args)

    actual_draft_path = draft_model_path or cfg.draft_model_path
    extra_ram = 0

    if effective_mode == "ngram-simple":
        extra_ram = 16
        args.extend([
            "--spec-type",
            "ngram-simple",
            "--spec-ngram-simple-size-n",
            str(cfg.ngram_simple_size_n),
            "--spec-ngram-simple-size-m",
            str(cfg.ngram_simple_size_m),
            "--spec-ngram-simple-min-hits",
            str(cfg.ngram_simple_min_hits),
        ])
    elif effective_mode == "ngram-mod":
        extra_ram = 24
        args.extend([
            "--spec-type",
            "ngram-mod",
            "--spec-ngram-mod-n-match",
            str(cfg.ngram_mod_n_match),
            "--spec-ngram-mod-n-min",
            str(cfg.ngram_mod_n_min),
            "--spec-ngram-mod-n-max",
            str(cfg.ngram_mod_n_max),
        ])
    elif effective_mode == "draft" and actual_draft_path:
        extra_ram = 450
        args.extend([
            "--spec-type",
            "draft-simple",
            "--model-draft",
            actual_draft_path,
            "-ngld",
            str(cfg.draft_gpu_layers),
            "--spec-draft-n-max",
            str(cfg.draft_n_max),
            "--spec-draft-p-min",
            str(cfg.draft_p_min),
        ])
    elif effective_mode == "hybrid" and actual_draft_path:
        extra_ram = 466
        args.extend([
            "--spec-type",
            "draft-simple,ngram-simple",
            "--model-draft",
            actual_draft_path,
            "-ngld",
            str(cfg.draft_gpu_layers),
            "--spec-draft-n-max",
            str(cfg.draft_n_max),
            "--spec-draft-p-min",
            str(cfg.draft_p_min),
            "--spec-ngram-simple-size-n",
            str(cfg.ngram_simple_size_n),
            "--spec-ngram-simple-size-m",
            str(cfg.ngram_simple_size_m),
            "--spec-ngram-simple-min-hits",
            str(cfg.ngram_simple_min_hits),
        ])
    elif effective_mode == "none":
        extra_ram = 0

    return LlamaSpeculativePreset(
        requested_mode=req_mode,
        effective_mode=effective_mode,
        profile=resolved_profile,
        downgrade_reason=downgrade_reason,
        args=args,
        estimated_extra_ram_mb=extra_ram,
    )
