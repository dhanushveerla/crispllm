from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class TaskClassification:
    tier: str
    confidence: float
    reasons: tuple[str, ...]


_DEEP_KEYWORD_RE = re.compile(
    r"\b("
    r"architect|architecture|refactor|redesign|security audit|vulnerability|"
    r"race condition|deadlock|concurrency|thread-safe|threadsafe|memory leak|"
    r"optimize across|optimization|root cause|post-mortem|postmortem|system design|"
    r"migration plan|backward compat|regression test"
    r")\b",
    re.IGNORECASE,
)

_TRACEBACK_RE = re.compile(
    r"(Traceback \(most recent call last\):|[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception): |\bat\s+[A-Za-z0-9_./\\-]+\s+line\s+\d+)",
    re.MULTILINE,
)

_DIFF_RE = re.compile(r"(?:^--- [a-zA-Z0-9_./-]+|\+\+\+ [a-zA-Z0-9_./-]+|@@ -\d+,\d+ \+\d+,\d+ @@)", re.MULTILINE)

_FAST_KEYWORD_RE = re.compile(
    r"\b("
    r"typo|rename|syntax|syntax error|formatting|pep8|lint|one-liner|"
    r"what is|how do i|where is|quick question|quick look|explain this error|"
    r"hello|ping|help"
    r")\b",
    re.IGNORECASE,
)


def _extract_text_from_message(msg: dict[str, Any]) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return ""


class TaskClassifier:
    def classify(
        self,
        messages: list[dict[str, Any]],
        est_tokens: int,
        fast_max_tokens: int = 1200,
        explicit_tier: str | None = None,
    ) -> TaskClassification:
        if explicit_tier:
            val = explicit_tier.strip().lower()
            if val in ("fast", "balanced", "deep"):
                return TaskClassification(tier=val, confidence=1.0, reasons=("explicit_header",))

        last_user_text = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user_text = _extract_text_from_message(m)
                break

        deep_reasons: list[str] = []

        m_kw = _DEEP_KEYWORD_RE.findall(last_user_text)
        if m_kw:
            deep_reasons.append(f"deep_keyword:{','.join(sorted(set(k.lower() for k in m_kw))[:3])}")

        if _TRACEBACK_RE.search(last_user_text):
            deep_reasons.append("stack_trace_detected")

        if _DIFF_RE.search(last_user_text):
            deep_reasons.append("code_diff_detected")

        if len(last_user_text) > 3500 or est_tokens > 4000:
            deep_reasons.append(f"large_context:{est_tokens}_tokens")

        if deep_reasons:
            return TaskClassification(
                tier="deep",
                confidence=0.9,
                reasons=tuple(deep_reasons),
            )

        fast_reasons: list[str] = []
        if est_tokens <= fast_max_tokens and len(last_user_text) < 300:
            m_fast = _FAST_KEYWORD_RE.findall(last_user_text)
            if m_fast:
                fast_reasons.append(f"fast_keyword:{m_fast[0].lower()}")
            elif len(last_user_text.split()) < 15 and "?" in last_user_text:
                fast_reasons.append("short_query")
            elif not last_user_text.strip():
                fast_reasons.append("empty_turn")

        if fast_reasons:
            return TaskClassification(
                tier="fast",
                confidence=0.85,
                reasons=tuple(fast_reasons),
            )

        return TaskClassification(
            tier="balanced",
            confidence=0.7,
            reasons=("default_balanced",),
        )
