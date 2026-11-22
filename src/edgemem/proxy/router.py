from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any, Mapping
import urllib.parse

from edgemem.config import EdgeMemConfig
from edgemem.proxy.budgeter import ContextBudgeter, ContextDecision
from edgemem.proxy.classifier import TaskClassification, TaskClassifier
from edgemem.proxy.watchdog import RamPressureWatchdog


def is_localhost_url(url: str) -> bool:
    try:
        parsed = urllib.parse.urlsplit(url)
        hostname = (parsed.hostname or "").lower()
        return hostname in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
    except Exception:
        return False


@dataclass
class RouteTarget:
    tier: str
    base_url: str
    api_key_env: str
    model: str
    extra_body: dict[str, Any]
    timeout_s: float
    reason: str
    is_external: bool
    decision: ContextDecision | None = None
    task_tier: str = "balanced"

    @property
    def api_key(self) -> str | None:
        if not self.api_key_env:
            return None
        return os.environ.get(self.api_key_env)


class HybridRouter:
    def __init__(
        self,
        config: EdgeMemConfig,
        budgeter: ContextBudgeter | None = None,
        watchdog: RamPressureWatchdog | None = None,
        classifier: TaskClassifier | None = None,
    ) -> None:
        self.config = config
        self.watchdog = watchdog or RamPressureWatchdog(config.resources)
        self.budgeter = budgeter or ContextBudgeter(config, watchdog=self.watchdog)
        self.classifier = classifier or TaskClassifier()

    def _build_local_target(
        self,
        body: dict[str, Any],
        reason: str,
        override_model: str | None = None,
        decision: ContextDecision | None = None,
        task_tier: str = "balanced",
    ) -> RouteTarget:
        model = override_model or body.get("model") or self.config.model
        base_url = self.config.upstream_base_url
        return RouteTarget(
            tier="local",
            base_url=base_url,
            api_key_env=self.config.api_key_env,
            model=model,
            extra_body=dict(self.config.extra_body),
            timeout_s=self.config.routing.local_timeout_s,
            reason=reason,
            is_external=not is_localhost_url(base_url),
            decision=decision,
            task_tier=task_tier,
        )

    def _build_cloud_target(
        self,
        body: dict[str, Any],
        reason: str,
        override_model: str | None = None,
        decision: ContextDecision | None = None,
        task_tier: str = "balanced",
    ) -> RouteTarget:
        model = (
            override_model
            or self.config.cloud.model
            or body.get("model")
            or self.config.model
        )
        return RouteTarget(
            tier="cloud",
            base_url=self.config.cloud.base_url,
            api_key_env=self.config.cloud.api_key_env,
            model=model,
            extra_body=dict(self.config.cloud.extra_body),
            timeout_s=self.config.cloud.timeout_s,
            reason=reason,
            is_external=True,
            decision=decision,
            task_tier=task_tier,
        )

    def select_target(
        self,
        body: dict[str, Any],
        headers: Mapping[str, str],
        est_prompt_tokens: int,
    ) -> RouteTarget:
        explicit_task_tier = headers.get("X-EdgeMem-Task-Tier") or headers.get("x-edgemem-task-tier")
        classification = self.classifier.classify(
            messages=body.get("messages", []),
            est_tokens=est_prompt_tokens,
            fast_max_tokens=self.config.routing.fast_max_tokens,
            explicit_tier=explicit_task_tier,
        )
        active_task_tier = classification.tier

        route_header = headers.get("X-EdgeMem-Route") or headers.get("x-edgemem-route")
        if route_header:
            val = route_header.strip().lower()
            if val == "local":
                return self._build_local_target(body, reason="header_override", task_tier=active_task_tier)
            if val == "cloud" and self.config.cloud.enabled:
                return self._build_cloud_target(body, reason="header_override", task_tier=active_task_tier)

        raw_model = body.get("model", "")
        if isinstance(raw_model, str):
            for prefix in ("cloud/", "escalate/"):
                if raw_model.startswith(prefix):
                    if self.config.cloud.enabled:
                        suffix = raw_model[len(prefix):]
                        cloud_model = self.config.cloud.model or suffix
                        return self._build_cloud_target(
                            body,
                            reason="model_prefix",
                            override_model=cloud_model,
                            task_tier=active_task_tier,
                        )
                    else:
                        return self._build_local_target(body, reason="default_local", task_tier=active_task_tier)

        strategy = self.config.routing.strategy
        if strategy == "cloud_only" and self.config.cloud.enabled:
            return self._build_cloud_target(body, reason="cloud_only", task_tier=active_task_tier)

        decision: ContextDecision | None = None
        if self.config.resources.enabled:
            cloud_avail = bool(self.config.cloud.enabled and strategy != "local_only")
            decision = self.budgeter.evaluate(
                prompt_tokens=est_prompt_tokens,
                requested_max_tokens=body.get("max_tokens"),
                cloud_available=cloud_avail,
            )
            if decision.should_escalate_cloud:
                return self._build_cloud_target(
                    body,
                    reason=decision.escalate_reason or "context_budget_exceeded",
                    decision=decision,
                    task_tier=active_task_tier,
                )

        if self.config.routing.task_routing_enabled or explicit_task_tier:
            if active_task_tier == "deep" and self.config.routing.escalate_deep_tasks and self.config.cloud.enabled and strategy != "local_only":
                reasons_str = ",".join(classification.reasons)
                return self._build_cloud_target(
                    body,
                    reason=f"task_tier_deep:{reasons_str}",
                    decision=decision,
                    task_tier="deep",
                )
            if active_task_tier == "fast" and self.config.routing.fast_model:
                return self._build_local_target(
                    body,
                    reason="task_tier_fast",
                    override_model=self.config.routing.fast_model,
                    decision=decision,
                    task_tier="fast",
                )
            if self.config.resources.enabled:
                pressure = self.watchdog.snapshot().pressure
                if pressure == "soft" and active_task_tier == "balanced":
                    downshift = self.config.routing.fast_model or self.config.resources.downshift_model
                    if downshift:
                        return self._build_local_target(
                            body,
                            reason="soft_pressure_downshift",
                            override_model=downshift,
                            decision=decision,
                            task_tier="balanced",
                        )

        if self.config.resources.enabled and decision is not None:
            return self._build_local_target(
                body,
                reason="default_local",
                override_model=decision.downshift_model,
                decision=decision,
                task_tier=active_task_tier,
            )

        if (
            strategy == "local_first"
            and self.config.cloud.enabled
            and est_prompt_tokens >= self.config.routing.escalate_token_threshold
        ):
            return self._build_cloud_target(body, reason="token_threshold", task_tier=active_task_tier)

        return self._build_local_target(body, reason="default_local", task_tier=active_task_tier)

    def cloud_fallback_target(self, body: dict[str, Any], task_tier: str = "balanced") -> RouteTarget | None:
        if self.config.cloud.enabled and self.config.routing.escalate_on_local_error:
            return self._build_cloud_target(body, reason="local_error_fallback", task_tier=task_tier)
        return None

    def should_redact(self, target: RouteTarget) -> bool:
        if not self.config.privacy.enabled:
            return False
        if not self.config.privacy.redact_on_cloud_only:
            return True
        return target.is_external
