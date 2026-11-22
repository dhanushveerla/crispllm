from __future__ import annotations

import copy
from dataclasses import dataclass, field
import json
import re
from typing import Any


@dataclass
class RedactionConfig:
    enabled: bool = True
    redact_secrets: bool = True
    redact_emails: bool = True
    redact_ips: bool = False
    extra_patterns: tuple[str, ...] = ()


@dataclass
class RedactionStats:
    total_redacted: int = 0
    by_category: dict[str, int] = field(default_factory=dict)


class RedactionSession:
    def __init__(self) -> None:
        self._secret_to_placeholder: dict[str, str] = {}
        self._placeholder_to_secret: dict[str, str] = {}
        self._category_counters: dict[str, int] = {}
        self._total_redacted: int = 0
        self._by_category: dict[str, int] = {}

    @property
    def has_redactions(self) -> bool:
        return self._total_redacted > 0

    @property
    def secret_to_placeholder(self) -> dict[str, str]:
        return self._secret_to_placeholder

    @property
    def placeholder_to_secret(self) -> dict[str, str]:
        return self._placeholder_to_secret

    def stats(self) -> RedactionStats:
        return RedactionStats(
            total_redacted=self._total_redacted,
            by_category=dict(self._by_category),
        )

    def get_or_create_placeholder(self, secret: str, category: str) -> str:
        self._total_redacted += 1
        self._by_category[category] = self._by_category.get(category, 0) + 1
        if secret in self._secret_to_placeholder:
            return self._secret_to_placeholder[secret]
        count = self._category_counters.get(category, 0) + 1
        self._category_counters[category] = count
        token = f"__CRISPLLM_{category}_{count}__"
        legacy_token = f"__EDGEMEM_{category}_{count}__"
        self._secret_to_placeholder[secret] = token
        self._placeholder_to_secret[token] = secret
        self._placeholder_to_secret[legacy_token] = secret
        return token


_PEM_PATTERN = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z0-9 ]*PRIVATE KEY-----"
)
_KNOWN_SECRETS_PATTERN = re.compile(
    r"""
    \b(?:
        nvapi-[A-Za-z0-9_-]{20,}
      | AIza[0-9A-Za-z_-]{35}(?![0-9A-Za-z_-])
      | sk-(?:proj-|ant-|live-|test-)?[A-Za-z0-9_-]{20,}
      | (?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}
      | github_pat_[A-Za-z0-9_]{30,}
      | AKIA[0-9A-Z]{16}\b
      | xox[baprs]-[A-Za-z0-9-]{10,}
      | eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}
    )
    """,
    re.VERBOSE,
)
_CRED_URL_PATTERN = re.compile(
    r"((?:postgres|postgresql|mysql|mongodb(?:\+srv)?|redis|amqp)://[^:\s/@]+:)([^@\s/]+)(@)"
)
_BEARER_PATTERN = re.compile(
    r"(?i)(Bearer\s+)([A-Za-z0-9._~+/=-]{16,})"
)
_ENV_PATTERN = re.compile(
    r'(?i)(\b[A-Z0-9_]*(?:API_KEY|SECRET_KEY|SECRET|PASSWORD|PASSWD|ACCESS_TOKEN|AUTH_TOKEN|PRIVATE_KEY)\b\s*[:=]\s*["\']?)([^"\'\s,;]{8,})(["\']?)'
)
_CRISPLLM_PLACEHOLDER_PATTERN = re.compile(r"__(?:CRISPLLM|EDGEMEM)_[A-Z]+_\d+__")
_EDGEMEM_PLACEHOLDER_PATTERN = _CRISPLLM_PLACEHOLDER_PATTERN
_ENV_SKIP_SUBSTRINGS = (
    "your-key-here",
    "changeme",
    "example",
    "dummy",
    "test_key",
    "[redacted_secret]",
)
_SECRET_KEY_PATTERN = re.compile(
    r"(?i)\b[A-Z0-9_]*(?:API_KEY|SECRET_KEY|SECRET|PASSWORD|PASSWD|ACCESS_TOKEN|AUTH_TOKEN|PRIVATE_KEY)\b"
)


def sanitize_for_memory(text: str) -> str:
    if not text or not isinstance(text, str):
        return text

    text = _EDGEMEM_PLACEHOLDER_PATTERN.sub("[REDACTED_SECRET]", text)
    text = _PEM_PATTERN.sub("[REDACTED_SECRET]", text)
    text = _KNOWN_SECRETS_PATTERN.sub("[REDACTED_SECRET]", text)

    def _sub_url(m: re.Match) -> str:
        prefix, password, suffix = m.group(1), m.group(2), m.group(3)
        if password == "[REDACTED_SECRET]":
            return m.group(0)
        return f"{prefix}[REDACTED_SECRET]{suffix}"

    text = _CRED_URL_PATTERN.sub(_sub_url, text)

    def _sub_bearer(m: re.Match) -> str:
        prefix, token = m.group(1), m.group(2)
        if token == "[REDACTED_SECRET]":
            return m.group(0)
        return f"{prefix}[REDACTED_SECRET]"

    text = _BEARER_PATTERN.sub(_sub_bearer, text)

    def _sub_env(m: re.Match) -> str:
        prefix, val, suffix = m.group(1), m.group(2), m.group(3)
        val_lower = val.lower()
        if any(s in val_lower for s in _ENV_SKIP_SUBSTRINGS):
            return m.group(0)
        return f"{prefix}[REDACTED_SECRET]{suffix}"

    text = _ENV_PATTERN.sub(_sub_env, text)
    return text


def sanitize_fact(key: str, value: str) -> tuple[str, str]:
    clean_k = sanitize_for_memory(key)
    clean_v = sanitize_for_memory(value)
    if _SECRET_KEY_PATTERN.search(clean_k):
        val_lower = clean_v.strip().lower()
        if len(val_lower) >= 6 and not any(s in val_lower for s in _ENV_SKIP_SUBSTRINGS):
            clean_v = "[REDACTED_SECRET]"
    return clean_k, clean_v


class SecretRedactor:
    def __init__(self, config: RedactionConfig | None = None) -> None:
        self.config = config or RedactionConfig()
        self._pem_pattern = _PEM_PATTERN
        self._known_secrets_pattern = _KNOWN_SECRETS_PATTERN
        self._cred_url_pattern = _CRED_URL_PATTERN
        self._bearer_pattern = _BEARER_PATTERN
        self._env_pattern = _ENV_PATTERN
        self._email_pattern = re.compile(
            r"\b[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9.-]+\.[A-Za-z]{2,}|localhost)\b"
        )
        self._ipv4_pattern = re.compile(
            r"\b(?:(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])\.){3}(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])\b"
        )
        self._placeholder_restore_pattern = _EDGEMEM_PLACEHOLDER_PATTERN
        self._env_skip_substrings = _ENV_SKIP_SUBSTRINGS
        self._email_skip_suffixes = (
            "@example.com",
            ".example.com",
            "@localhost",
            "@test.local",
            ".test.local",
        )
        self._ipv4_skip = {"127.0.0.1", "0.0.0.0", "255.255.255.255"}
        self._custom_patterns = [
            re.compile(p) for p in self.config.extra_patterns
        ]

    def redact_text(self, text: str, session: RedactionSession) -> str:
        if not text or not self.config.enabled:
            return text

        if self.config.redact_secrets:
            text = self._pem_pattern.sub(
                lambda m: session.get_or_create_placeholder(m.group(0), "PEM"),
                text,
            )

            def _sub_known(m: re.Match) -> str:
                raw = m.group(0)
                if raw.startswith(("__CRISPLLM_", "__EDGEMEM_")):
                    return raw
                return session.get_or_create_placeholder(raw, "SECRET")

            text = self._known_secrets_pattern.sub(_sub_known, text)

            def _sub_url(m: re.Match) -> str:
                prefix, password, suffix = m.group(1), m.group(2), m.group(3)
                if password.startswith(("__CRISPLLM_", "__EDGEMEM_")):
                    return m.group(0)
                placeholder = session.get_or_create_placeholder(password, "SECRET")
                return f"{prefix}{placeholder}{suffix}"

            text = self._cred_url_pattern.sub(_sub_url, text)

            def _sub_bearer(m: re.Match) -> str:
                prefix, token = m.group(1), m.group(2)
                if token.startswith(("__CRISPLLM_", "__EDGEMEM_")):
                    return m.group(0)
                placeholder = session.get_or_create_placeholder(token, "SECRET")
                return f"{prefix}{placeholder}"

            text = self._bearer_pattern.sub(_sub_bearer, text)

            def _sub_env(m: re.Match) -> str:
                prefix, val, suffix = m.group(1), m.group(2), m.group(3)
                if val.startswith(("__CRISPLLM_", "__EDGEMEM_")):
                    return m.group(0)
                val_lower = val.lower()
                if any(s in val_lower for s in self._env_skip_substrings):
                    return m.group(0)
                placeholder = session.get_or_create_placeholder(val, "SECRET")
                return f"{prefix}{placeholder}{suffix}"

            text = self._env_pattern.sub(_sub_env, text)

        if self.config.redact_emails:
            def _sub_email(m: re.Match) -> str:
                email = m.group(0)
                if email.startswith(("__CRISPLLM_", "__EDGEMEM_")):
                    return email
                email_lower = email.lower()
                if any(email_lower.endswith(s) for s in self._email_skip_suffixes):
                    return email
                return session.get_or_create_placeholder(email, "EMAIL")

            text = self._email_pattern.sub(_sub_email, text)

        if self.config.redact_ips:
            def _sub_ip(m: re.Match) -> str:
                ip = m.group(0)
                if ip in self._ipv4_skip:
                    return ip
                return session.get_or_create_placeholder(ip, "IP")

            text = self._ipv4_pattern.sub(_sub_ip, text)

        for custom_pattern in self._custom_patterns:
            def _sub_custom(m: re.Match) -> str:
                raw = m.group(0)
                if raw.startswith(("__CRISPLLM_", "__EDGEMEM_")):
                    return raw
                return session.get_or_create_placeholder(raw, "CUSTOM")

            text = custom_pattern.sub(_sub_custom, text)

        return text

    def redact_messages(
        self, messages: list[dict[str, Any]], session: RedactionSession
    ) -> list[dict[str, Any]]:
        if not messages:
            return []
        copied = copy.deepcopy(messages)
        if not self.config.enabled:
            return copied
        for msg in copied:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = self.redact_text(content, session)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_val = block.get("text")
                        if isinstance(text_val, str):
                            block["text"] = self.redact_text(text_val, session)
            tool_calls = msg.get("tool_calls")
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        fn = tc.get("function")
                        if isinstance(fn, dict):
                            args = fn.get("arguments")
                            if isinstance(args, str):
                                fn["arguments"] = self.redact_text(args, session)
            fn_call = msg.get("function_call")
            if isinstance(fn_call, dict):
                args = fn_call.get("arguments")
                if isinstance(args, str):
                    fn_call["arguments"] = self.redact_text(args, session)
        return copied

    def restore_text(self, text: str, session: RedactionSession) -> str:
        if not text or not session.has_redactions:
            return text
        if "__CRISPLLM_" not in text and "__EDGEMEM_" not in text:
            return text
        mapping = session.placeholder_to_secret
        return self._placeholder_restore_pattern.sub(
            lambda m: mapping.get(m.group(0), m.group(0)),
            text,
        )

    def restore_response(
        self, payload: dict[str, Any], session: RedactionSession
    ) -> dict[str, Any]:
        if not payload:
            return {}
        if not session.has_redactions:
            return copy.deepcopy(payload)
        copied = copy.deepcopy(payload)
        choices = copied.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                message = choice.get("message")
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                if isinstance(content, str):
                    message["content"] = self.restore_text(content, session)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_val = block.get("text")
                            if isinstance(text_val, str):
                                block["text"] = self.restore_text(text_val, session)
                tool_calls = message.get("tool_calls")
                if isinstance(tool_calls, list):
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            fn = tc.get("function")
                            if isinstance(fn, dict):
                                args = fn.get("arguments")
                                if isinstance(args, str):
                                    fn["arguments"] = self.restore_text(args, session)
                fn_call = message.get("function_call")
                if isinstance(fn_call, dict):
                    args = fn_call.get("arguments")
                    if isinstance(args, str):
                        fn_call["arguments"] = self.restore_text(args, session)
        return copied


Redactor = SecretRedactor

