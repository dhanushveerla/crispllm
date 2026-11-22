from __future__ import annotations

import hmac
import json
from typing import Any, Awaitable, Callable

from crispllm.config import SecurityConfig

TUNNEL_PROXY_HEADERS: frozenset[str] = frozenset({
    "x-forwarded-for",
    "cf-connecting-ip",
    "forwarded",
    "x-real-ip",
})


class CrispLLMAuthMiddleware:
    def __init__(self, app: Any, config: SecurityConfig) -> None:
        self.app = app
        self.config = config

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "GET").upper()
        if method == "OPTIONS":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path == "/health" and method == "GET":
            await self.app(scope, receive, send)
            return

        headers_dict: dict[str, str] = {}
        for raw_k, raw_v in scope.get("headers", []):
            try:
                headers_dict[raw_k.decode("latin1").lower()] = raw_v.decode("latin1")
            except Exception:
                continue

        if path.startswith("/v1") and not self.config.allow_remote_v1:
            if any(h in headers_dict for h in TUNNEL_PROXY_HEADERS):
                await self._send_json(
                    send,
                    status=403,
                    body={
                        "error": {
                            "message": "Remote access to /v1 is disabled; set [security] allow_remote_v1 = true to enable",
                            "type": "forbidden",
                        }
                    },
                )
                return

        token = self.config.resolve_auth_token()
        if token and (path.startswith("/mcp") or path.startswith("/v1")):
            auth_header = headers_dict.get("authorization", "")
            if not auth_header.lower().startswith("bearer "):
                await self._send_json(
                    send,
                    status=401,
                    body={
                        "error": {
                            "message": "Invalid or missing Bearer token",
                            "type": "unauthorized",
                        }
                    },
                    extra_headers=[(b"www-authenticate", b"Bearer")],
                )
                return

            candidate = auth_header[7:].strip()
            if not hmac.compare_digest(candidate, token):
                await self._send_json(
                    send,
                    status=401,
                    body={
                        "error": {
                            "message": "Invalid or missing Bearer token",
                            "type": "unauthorized",
                        }
                    },
                    extra_headers=[(b"www-authenticate", b"Bearer")],
                )
                return

        await self.app(scope, receive, send)

    @staticmethod
    async def _send_json(
        send: Callable[[dict[str, Any]], Awaitable[None]],
        status: int,
        body: dict[str, Any],
        extra_headers: list[tuple[bytes, bytes]] | None = None,
    ) -> None:
        content = json.dumps(body).encode("utf-8")
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(content)).encode("ascii")),
        ]
        if extra_headers:
            headers.extend(extra_headers)
        await send({
            "type": "http.response.start",
            "status": status,
            "headers": headers,
        })
        await send({
            "type": "http.response.body",
            "body": content,
        })


EdgeMemAuthMiddleware = CrispLLMAuthMiddleware
