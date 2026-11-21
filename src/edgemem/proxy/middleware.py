"""Memory middleware for EdgeMem OpenAI proxy.

Handles session isolation, auto-extraction of facts/constraints, retrieval of
active memory, token budget enforcement without tiktoken, clean non-overriding formatting,
and tiered middle-turn compaction (FR-04).
"""

from __future__ import annotations

import copy
import hashlib
import logging
import re
from typing import Any, Mapping

from edgemem.config import MemoryConfig
from edgemem.memory.extractor import extract_facts_from_messages
from edgemem.memory.store import Fact, MemoryStore

logger = logging.getLogger("edgemem.proxy.memory")

_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "up", "about", "into", "through", "after", "is", "am",
    "are", "was", "were", "be", "been", "being", "have", "has", "had", "do",
    "does", "did", "can", "could", "will", "would", "shall", "should", "may",
    "might", "must", "it", "its", "this", "that", "these", "those", "i", "you",
    "he", "she", "we", "they", "me", "him", "her", "us", "them", "what", "which",
    "who", "whom", "how", "when", "where", "why", "now", "please", "tell",
    "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "def", "class", "return", "import", "from", "self", "none", "true", "false",
    "str", "int", "float", "bool", "dict", "list", "tuple", "set", "optional",
    "callable", "error", "exception", "valueerror", "typeerror", "raise", "try",
    "except", "assert", "pass", "while", "if", "else", "elif", "as",
    "lambda", "yield", "async", "await", "init", "main", "run", "get", "set",
    "test", "tests", "pytest", "passed", "failed", "line", "file", "module",
    "function", "method", "value", "values", "data", "item", "items", "result",
    "results", "output", "input", "check", "valid", "invalid", "code", "python",
    "print", "config", "default", "name", "type", "status", "message", "info",
    "warning", "debug", "log", "logs", "build", "step", "time", "count", "index",
    "key", "keys", "args", "kwargs", "__init__",
})

_GENERIC_CODE_NAMES: frozenset[str] = frozenset({
    "main", "run", "get", "set", "helper", "process", "handle", "validate", "parse",
})


def estimate_tokens(text: str) -> int:
    """Fast zero-allocation token count estimator (~4 characters per token)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def _get_message_text(msg: dict[str, Any]) -> str:
    """Safely extract plain text from a message content field."""
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


def _is_domain_symbol(sym: str) -> bool:
    if len(sym) < 4:
        return False
    if sym.startswith("__") and sym.endswith("__"):
        return False
    if sym.lower().startswith("test_"):
        return False
    sym_lower = sym.lower()
    if sym_lower in _GENERIC_CODE_NAMES or sym_lower in _STOPWORDS:
        return False
    return True


def _has_code_reference(msg_text: str, query_text: str) -> bool:
    symbols = set(re.findall(r"\b(?:def|class)\s+([A-Za-z0-9_]+)", msg_text))
    symbols.update(re.findall(r"^\s*([A-Z][A-Z0-9_]{1,63})\s*=", msg_text, re.MULTILINE))
    domain_symbols = {sym for sym in symbols if _is_domain_symbol(sym)}
    if not domain_symbols:
        return False
    query_identifiers = set(re.findall(r"\b[A-Za-z0-9_]+\b", query_text))
    return bool(domain_symbols.intersection(query_identifiers))


def _extract_domain_tokens(text: str) -> set[str]:
    return {
        tok
        for tok in re.findall(r"[A-Za-z0-9_]+", text.lower())
        if len(tok) >= 3
        and tok not in _STOPWORDS
        and not (tok.startswith("__") and tok.endswith("__"))
    }


def _has_lexical_overlap(msg_text: str, query_text: str) -> bool:
    query_tokens = _extract_domain_tokens(query_text)
    if not query_tokens:
        return False
    msg_tokens = _extract_domain_tokens(msg_text)
    shared = query_tokens.intersection(msg_tokens)
    ratio = len(shared) / len(query_tokens)
    return len(shared) >= 3 and ratio >= 0.25


def _format_fact_line(fact: Fact) -> str:
    """Format fact line without internal constraint hash keys or duplicate variable names."""
    val = fact.value.strip()
    if fact.kind in ("constraint", "note"):
        return f"- {val}"
    if (
        val.startswith(fact.key)
        or val.startswith(f"def {fact.key}")
        or val.startswith(f"async def {fact.key}")
        or val.startswith(f"class {fact.key}")
    ):
        return f"- {val}"
    return f"- {fact.key}: {val}"


def _normalize_fact_value(val: str) -> str:
    """Normalize fact text for duplicate detection."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9_]+", " ", val.lower())).strip()


class MemoryMiddleware:
    """Proxy middleware for transparent tiered memory retrieval, extraction, and compaction."""

    def __init__(self, store: MemoryStore, config: MemoryConfig) -> None:
        self.store = store
        self.config = config

    def _derive_session_id(
        self, messages: list[dict[str, Any]], headers: Mapping[str, str]
    ) -> str:
        """Derive an isolated session ID from headers or initial turn content."""
        for k, v in headers.items():
            if k.lower() == "x-edgemem-session" and v:
                return v.strip()

        system_text = ""
        user_text = ""
        for msg in messages:
            role = msg.get("role")
            if role == "system" and not system_text:
                system_text = _get_message_text(msg)
            elif role == "user" and not user_text:
                user_text = _get_message_text(msg)
            if system_text and user_text:
                break

        seed = f"{system_text}\n---\n{user_text}".encode("utf-8")
        return hashlib.sha256(seed).hexdigest()[:16]

    def process_request(
        self, body: dict[str, Any], headers: Mapping[str, str]
    ) -> tuple[dict[str, Any], dict[str, int]]:
        """Process incoming chat completions request and inject active memory.

        Returns:
            (modified_or_original_body, telemetry_dict)
            telemetry_dict contains "facts_extracted", "memory_injected_tokens", and "compacted_turns".
        """
        if not self.config.enabled:
            return body, {
                "facts_extracted": 0,
                "memory_injected_tokens": 0,
                "compacted_turns": 0,
            }

        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return body, {
                "facts_extracted": 0,
                "memory_injected_tokens": 0,
                "compacted_turns": 0,
            }

        try:
            session_id = self._derive_session_id(messages, headers)
            user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
            if not user_indices:
                return body, {
                    "facts_extracted": 0,
                    "memory_injected_tokens": 0,
                    "compacted_turns": 0,
                }

            last_user_idx = user_indices[-1]
            last_user_text = _get_message_text(messages[last_user_idx])

            if len(user_indices) == 1:
                existing_facts = self.store.list_facts(session_id=session_id, limit=1)
                global_pinned = self.store.get_pinned_facts(
                    session_id=session_id, include_global=True, limit=1
                )
                if not existing_facts and not global_pinned:
                    return body, {
                        "facts_extracted": 0,
                        "memory_injected_tokens": 0,
                        "compacted_turns": 0,
                    }

            facts_extracted = 0
            if self.config.auto_extract and len(user_indices) > 1:
                extracted = extract_facts_from_messages(messages)
                for ef in extracted:
                    self.store.upsert_fact(
                        session_id=session_id,
                        key=ef.key,
                        value=ef.value,
                        kind=ef.kind,
                        scope=ef.scope,
                        source_turn=ef.source_turn,
                        pinned=ef.pinned,
                    )
                facts_extracted = len(extracted)

            candidate_facts: list[Fact] = []
            seen_keys: set[tuple[str, str]] = set()

            if self.config.inject_constraints:
                pinned = self.store.get_pinned_facts(
                    session_id=session_id, include_global=True, limit=20
                )
                for f in pinned:
                    k = (f.kind, f.key)
                    if k not in seen_keys:
                        seen_keys.add(k)
                        candidate_facts.append(f)

            if last_user_text:
                search_res = self.store.search_facts(
                    query=last_user_text,
                    session_id=session_id,
                    include_global=True,
                    limit=10,
                )
                for f in search_res:
                    k = (f.kind, f.key)
                    if k not in seen_keys:
                        seen_keys.add(k)
                        candidate_facts.append(f)

            compacted_turns = 0
            new_messages = copy.deepcopy(messages)

            total_est_tokens = sum(
                estimate_tokens(_get_message_text(m)) for m in messages
            )

            if (
                self.config.compact_middle_turns
                and len(user_indices) >= 3
                and total_est_tokens >= self.config.compact_threshold_tokens
            ):
                first_user_idx = user_indices[0]
                recent_user_threshold = (
                    user_indices[-(self.config.keep_recent_turns - 1)]
                    if self.config.keep_recent_turns > 1
                    else last_user_idx
                )

                start_idx = first_user_idx + 1
                if start_idx < len(new_messages) and new_messages[start_idx].get("role") == "assistant":
                    start_idx += 1

                idx = start_idx
                while idx < recent_user_threshold:
                    m = new_messages[idx]
                    role = m.get("role")
                    if role in ("system", "tool") or "tool_calls" in m:
                        idx += 1
                        continue

                    text = _get_message_text(m)
                    if not text or text == "[Compacted earlier turn]":
                        idx += 1
                        continue

                    is_pair = (
                        role == "user"
                        and idx + 1 < recent_user_threshold
                        and new_messages[idx + 1].get("role") == "assistant"
                        and "tool_calls" not in new_messages[idx + 1]
                        and new_messages[idx + 1].get("content") != "[Compacted earlier turn]"
                    )

                    if is_pair:
                        asst_m = new_messages[idx + 1]
                        asst_text = _get_message_text(asst_m)
                        user_rel = _has_lexical_overlap(text, last_user_text) or _has_code_reference(
                            text, last_user_text
                        )
                        asst_rel = _has_lexical_overlap(asst_text, last_user_text) or _has_code_reference(
                            asst_text, last_user_text
                        )
                        if user_rel or asst_rel:
                            idx += 2
                            continue

                        pair_tokens = estimate_tokens(text) + estimate_tokens(asst_text)
                        if pair_tokens >= self.config.compact_min_turn_tokens:
                            self.store.add_episode(
                                session_id=session_id,
                                role="user",
                                summary=text[:300],
                                raw_tokens=estimate_tokens(text),
                            )
                            self.store.add_episode(
                                session_id=session_id,
                                role="assistant",
                                summary=asst_text[:300],
                                raw_tokens=estimate_tokens(asst_text),
                            )
                            new_messages[idx] = {"role": "user", "content": "[Compacted earlier turn]"}
                            new_messages[idx + 1] = {"role": "assistant", "content": "[Compacted earlier turn]"}
                            compacted_turns += 2

                        idx += 2
                        continue

                    turn_tokens = estimate_tokens(text)
                    if (
                        turn_tokens >= self.config.compact_min_turn_tokens
                        and not _has_lexical_overlap(text, last_user_text)
                        and not _has_code_reference(text, last_user_text)
                    ):
                        self.store.add_episode(
                            session_id=session_id,
                            role=role,
                            summary=text[:300],
                            raw_tokens=turn_tokens,
                        )
                        new_messages[idx] = {"role": role, "content": "[Compacted earlier turn]"}
                        compacted_turns += 1

                    idx += 1

            memory_block = ""
            injected_tokens = 0

            if candidate_facts:
                header = (
                    "[EdgeMem Active Memory]\n"
                    "Earlier-turn constraints & facts (follow these unless the latest user message "
                    "explicitly specifies an exception or return behavior):"
                )

                seen_vals: set[str] = set()
                fact_lines: list[str] = []
                for f in candidate_facts:
                    norm = _normalize_fact_value(f.value)
                    if norm in seen_vals:
                        continue
                    seen_vals.add(norm)
                    fact_lines.append(_format_fact_line(f))

                while fact_lines:
                    test_block = header + "\n" + "\n".join(fact_lines)
                    tok_count = estimate_tokens(test_block)
                    if tok_count <= self.config.max_injected_tokens:
                        break
                    fact_lines.pop()

                if fact_lines:
                    memory_block = header + "\n" + "\n".join(fact_lines)
                    injected_tokens = estimate_tokens(memory_block)

            if not memory_block and compacted_turns == 0:
                return body, {
                    "facts_extracted": facts_extracted,
                    "memory_injected_tokens": 0,
                    "compacted_turns": 0,
                }

            new_body = dict(body)
            if memory_block:
                first_msg = new_messages[0]
                if first_msg.get("role") == "system":
                    first_content = first_msg.get("content")
                    if isinstance(first_content, str):
                        first_msg["content"] = f"{first_content}\n\n{memory_block}"
                    elif isinstance(first_content, list):
                        content_list = list(first_content)
                        content_list.append({"type": "text", "text": f"\n\n{memory_block}"})
                        first_msg["content"] = content_list
                    else:
                        new_messages.insert(0, {"role": "system", "content": memory_block})
                else:
                    new_messages.insert(0, {"role": "system", "content": memory_block})

            new_body["messages"] = new_messages
            return new_body, {
                "facts_extracted": facts_extracted,
                "memory_injected_tokens": injected_tokens,
                "compacted_turns": compacted_turns,
            }

        except Exception as e:
            logger.exception("MemoryMiddleware error, failing open: %s", e)
            return body, {
                "facts_extracted": 0,
                "memory_injected_tokens": 0,
                "compacted_turns": 0,
            }
