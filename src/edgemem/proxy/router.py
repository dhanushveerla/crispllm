from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any, Mapping
import urllib.parse

from edgemem.config import EdgeMemConfig


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

    @property
    def api_key(self) -> str | None:
        if not self.api_key_env:
            return None
        return os.environ.get(self.api_key_env)


class HybridRouter:
    def __init__(self, config: EdgeMemConfig) -> None:
        self.config = config

    def _build_local_target(self, body: dict[str, Any], reason: str) -> RouteTarget:
        model = body.get("model") or self.config.model
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
        )

    def _build_cloud_target(
        self,
        body: dict[str, Any],
        reason: str,
        override_model: str | None = None,
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
        )

    def select_target(
        self,
        body: dict[str, Any],
        headers: Mapping[str, str],
        est_prompt_tokens: int,
    ) -> RouteTarget:
        route_header = headers.get("X-EdgeMem-Route") or headers.get("x-edgemem-route")
        if route_header:
            val = route_header.strip().lower()
            if val == "local":
                return self._build_local_target(body, reason="header_override")
            if val == "cloud" and self.config.cloud.enabled:
                return self._build_cloud_target(body, reason="header_override")

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
                        )
                    else:
                        return self._build_local_target(body, reason="default_local")

        strategy = self.config.routing.strategy
        if strategy == "cloud_only" and self.config.cloud.enabled:
            return self._build_cloud_target(body, reason="cloud_only")

        if (
            strategy == "local_first"
            and self.config.cloud.enabled
            and est_prompt_tokens >= self.config.routing.escalate_token_threshold
        ):
            return self._build_cloud_target(body, reason="token_threshold")

        return self._build_local_target(body, reason="default_local")

    def cloud_fallback_target(self, body: dict[str, Any]) -> RouteTarget | None:
        if self.config.cloud.enabled and self.config.routing.escalate_on_local_error:
            return self._build_cloud_target(body, reason="local_error_fallback")
        return None

    def should_redact(self, target: RouteTarget) -> bool:
        if not self.config.privacy.enabled:
            return False
        if not self.config.privacy.redact_on_cloud_only:
            return True
        return target.is_external
