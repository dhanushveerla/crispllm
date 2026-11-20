"""Memory middleware for EdgeMem OpenAI proxy.

Handles session isolation, auto-extraction of facts/constraints, retrieval of
active memory, token budget enforcement, and non-destructive injection into upstream prompts.
"""

from __future__ import annotations

import copy
import hashlib
import logging
from typing import Any, Mapping

import tiktoken

from edgemem.config import MemoryConfig
from edgemem.memory.extractor import extract_facts_from_messages
from edgemem.memory.store import Fact, MemoryStore

logger = logging.getLogger("edgemem.proxy.memory")

_ENCODING: tiktoken.Encoding | None = None


def _get_encoding() -> tiktoken.Encoding:
    """Return module-cached tiktoken encoder."""
    global _ENCODING
    if _ENCODING is None:
        _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING


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


class MemoryMiddleware:
    """Proxy middleware for transparent tiered memory retrieval and extraction."""

    def __init__(self, store: MemoryStore, config: MemoryConfig) -> None:
        self.store = store
        self.config = config

    def _derive_session_id(
        self, messages: list[dict[str, Any]], headers: Mapping[str, str]
    ) -> str:
        """Derive an isolated session ID from headers or initial turn content."""
        # Check custom header first
        for k, v in headers.items():
            if k.lower() == "x-edgemem-session" and v:
                return v.strip()

        # Deterministic hash of first system message + first user message
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
            telemetry_dict contains "facts_extracted" and "memory_injected_tokens".
        """
        if not self.config.enabled:
            return body, {"facts_extracted": 0, "memory_injected_tokens": 0}

        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return body, {"facts_extracted": 0, "memory_injected_tokens": 0}

        try:
            session_id = self._derive_session_id(messages, headers)
            facts_extracted = 0

            user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
            if not user_indices:
                return body, {"facts_extracted": 0, "memory_injected_tokens": 0}

            last_user_idx = user_indices[-1]

            # Auto-extract facts from prior conversation turns
            if self.config.auto_extract and last_user_idx > 0:
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

            # Retrieve facts to inject: pinned constraints first, then BM25 search
            candidate_facts: list[Fact] = []
            seen: set[tuple[str, str]] = set()

            if self.config.inject_constraints:
                pinned = self.store.get_pinned_facts(
                    session_id=session_id, include_global=True, limit=20
                )
                for f in pinned:
                    k = (f.kind, f.key)
                    if k not in seen:
                        seen.add(k)
                        candidate_facts.append(f)

            # Query with final user prompt text
            last_user_text = _get_message_text(messages[last_user_idx])
            if last_user_text:
                search_res = self.store.search_facts(
                    query=last_user_text,
                    session_id=session_id,
                    include_global=True,
                    limit=10,
                )
                for f in search_res:
                    k = (f.kind, f.key)
                    if k not in seen:
                        seen.add(k)
                        candidate_facts.append(f)

            if not candidate_facts:
                return body, {
                    "facts_extracted": facts_extracted,
                    "memory_injected_tokens": 0,
                }

            # Format memory block and enforce token budget
            enc = _get_encoding()
            header = "[EdgeMem Active Memory]"
            fact_lines = [f"- {f.key}: {f.value}" for f in candidate_facts]

            while fact_lines:
                test_block = header + "\n" + "\n".join(fact_lines)
                tok_count = len(enc.encode(test_block))
                if tok_count <= self.config.max_injected_tokens:
                    break
                fact_lines.pop()

            if not fact_lines:
                return body, {
                    "facts_extracted": facts_extracted,
                    "memory_injected_tokens": 0,
                }

            memory_block = header + "\n" + "\n".join(fact_lines)
            injected_tokens = len(enc.encode(memory_block))

            # Non-destructive injection
            new_body = dict(body)
            new_messages = copy.deepcopy(messages)

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
            }

        except Exception as e:
            # Fail-open: memory errors must never break user requests
            logger.exception("MemoryMiddleware error, failing open: %s", e)
            return body, {"facts_extracted": 0, "memory_injected_tokens": 0}
