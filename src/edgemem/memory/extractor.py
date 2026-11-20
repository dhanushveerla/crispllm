"""Deterministic pure-Python fact and constraint extractor for EdgeMem.

Extracts architectural constraints and exact code signatures/constants from conversation
turns prior to the final turn without any LLM calls.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import re
from typing import Any

NORM_MARKERS_RE = re.compile(
    r"\b(must|never|always|do\s+not|don't|strictly|only|immutable|frozen|singleton|monotonic|invariant|requirement|constraint|ensure|preserve|forbidden|required)\b",
    re.IGNORECASE,
)

FENCED_CODE_RE = re.compile(r"```(?:[a-zA-Z0-9_\-]+)?\s*\n?(.*?)\n?```", re.DOTALL)
CONST_LINE_RE = re.compile(r"^\s*([A-Z][A-Z0-9_]{1,63})\s*=\s*([^\n#;]+)", re.MULTILINE)
FUNC_LINE_RE = re.compile(
    r"^\s*(?:async\s+)?def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(\(.*?\)(?:\s*->\s*[^:\n]+)?)\s*:",
    re.MULTILINE,
)
CLASS_LINE_RE = re.compile(
    r"^\s*class\s+([a-zA-Z_][a-zA-Z0-9_]*)(?:\s*\((.*?)\))?\s*:",
    re.MULTILINE,
)


@dataclass
class ExtractedFact:
    """Fact extracted from conversation turns."""

    key: str
    value: str
    kind: str  # "constraint", "constant", "signature", etc.
    scope: str = "session"
    source_turn: int = 0
    pinned: bool = False


def _get_message_text(msg: dict[str, Any]) -> str:
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


def _format_func_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
    args_str = ast.unparse(node.args)
    ret_str = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"{prefix}{node.name}({args_str}){ret_str}: ..."


def _format_class_signature(
    node: ast.ClassDef, source_turn: int
) -> tuple[str, list[ExtractedFact]]:
    bases_str = f"({', '.join(ast.unparse(b) for b in node.bases)})" if node.bases else ""
    methods: list[str] = []
    nested_facts: list[ExtractedFact] = []

    for sub in node.body:
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
            mpfx = "async def " if isinstance(sub, ast.AsyncFunctionDef) else "def "
            margs = ast.unparse(sub.args)
            mret = f" -> {ast.unparse(sub.returns)}" if sub.returns else ""
            methods.append(f"    {mpfx}{sub.name}({margs}){mret}: ...")
        elif isinstance(sub, ast.Assign):
            for st in sub.targets:
                if isinstance(st, ast.Name) and st.id.isupper():
                    cval = ast.unparse(sub.value).strip()
                    scoped = f"{node.name}.{st.id}"
                    nested_facts.append(
                        ExtractedFact(
                            key=scoped,
                            value=f"{scoped} = {cval}",
                            kind="constant",
                            source_turn=source_turn,
                            pinned=False,
                        )
                    )

    body = "\n".join(methods[:8]) if methods else "    ..."
    sig = f"class {node.name}{bases_str}:\n{body}"
    return sig, nested_facts


def _extract_from_ast(code: str, source_turn: int) -> list[ExtractedFact]:
    facts: list[ExtractedFact] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return _extract_from_regex(code, source_turn)

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    name = target.id
                    val_str = ast.unparse(node.value).strip()
                    if name.isupper():
                        facts.append(
                            ExtractedFact(
                                key=name,
                                value=f"{name} = {val_str}",
                                kind="constant",
                                source_turn=source_turn,
                                pinned=False,
                            )
                        )
                    elif name[0].isupper():
                        facts.append(
                            ExtractedFact(
                                key=name,
                                value=f"{name} = {val_str}",
                                kind="signature",
                                source_turn=source_turn,
                                pinned=False,
                            )
                        )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            ann_str = ast.unparse(node.annotation).strip()
            val_str = f": {ann_str}" + (f" = {ast.unparse(node.value).strip()}" if node.value else "")
            if name.isupper():
                facts.append(
                    ExtractedFact(
                        key=name,
                        value=f"{name}{val_str}",
                        kind="constant",
                        source_turn=source_turn,
                        pinned=False,
                    )
                )
            elif name[0].isupper() or "TypeAlias" in ann_str:
                facts.append(
                    ExtractedFact(
                        key=name,
                        value=f"{name}{val_str}",
                        kind="signature",
                        source_turn=source_turn,
                        pinned=False,
                    )
                )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sig = _format_func_signature(node)
            facts.append(
                ExtractedFact(
                    key=node.name,
                    value=sig,
                    kind="signature",
                    source_turn=source_turn,
                    pinned=False,
                )
            )
        elif isinstance(node, ast.ClassDef):
            sig, nested = _format_class_signature(node, source_turn)
            facts.append(
                ExtractedFact(
                    key=node.name,
                    value=sig,
                    kind="signature",
                    source_turn=source_turn,
                    pinned=False,
                )
            )
            facts.extend(nested)

    return facts


def _extract_from_regex(text: str, source_turn: int) -> list[ExtractedFact]:
    facts: list[ExtractedFact] = []
    for match in CONST_LINE_RE.finditer(text):
        name = match.group(1).strip()
        val = match.group(2).strip()
        facts.append(
            ExtractedFact(
                key=name,
                value=f"{name} = {val}",
                kind="constant",
                source_turn=source_turn,
                pinned=False,
            )
        )
    for match in FUNC_LINE_RE.finditer(text):
        name = match.group(1).strip()
        params = match.group(2).strip()
        facts.append(
            ExtractedFact(
                key=name,
                value=f"def {name}{params}: ...",
                kind="signature",
                source_turn=source_turn,
                pinned=False,
            )
        )
    for match in CLASS_LINE_RE.finditer(text):
        name = match.group(1).strip()
        bases = match.group(2)
        bases_str = f"({bases.strip()})" if bases else ""
        facts.append(
            ExtractedFact(
                key=name,
                value=f"class {name}{bases_str}: ...",
                kind="signature",
                source_turn=source_turn,
                pinned=False,
            )
        )
    return facts


def _extract_constraints(text: str, source_turn: int) -> list[ExtractedFact]:
    clean_text = FENCED_CODE_RE.sub(" ", text)
    candidates = re.split(r"(?<=[.!?;\n])\s+|\n+", clean_text)
    facts: list[ExtractedFact] = []

    for c in candidates:
        line = c.strip().lstrip("-*•0123456789. ")
        if len(line) < 8 or len(line) > 400 or line.endswith("?"):
            continue
        if NORM_MARKERS_RE.search(line):
            slug_words = re.findall(r"[a-z0-9]+", line.lower())[:6]
            slug = "_".join(slug_words)
            digest = hashlib.sha256(line.encode("utf-8")).hexdigest()[:8]
            key = f"constraint_{slug}_{digest}" if slug else f"constraint_{digest}"
            facts.append(
                ExtractedFact(
                    key=key,
                    value=line,
                    kind="constraint",
                    scope="session",
                    source_turn=source_turn,
                    pinned=True,
                )
            )
    return facts


def extract_facts_from_messages(
    messages: list[dict[str, Any]],
    max_constraints: int = 15,
    max_code_facts: int = 25,
) -> list[ExtractedFact]:
    """Inspect conversation turns prior to the final user message and extract facts.

    Captures:
    1. Normative architectural constraints and invariants from user turns (pinned=True).
    2. Exact constants, type aliases, and signatures from code blocks or raw code (pinned=False).
    """
    if not messages:
        return []

    user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if not user_indices:
        earlier_turns = list(enumerate(messages[:-1])) if len(messages) > 1 else []
    else:
        last_user_idx = user_indices[-1]
        earlier_turns = list(enumerate(messages[:last_user_idx]))

    constraints: list[ExtractedFact] = []
    code_facts: list[ExtractedFact] = []
    seen_constraint_keys: set[str] = set()
    seen_code_keys: set[tuple[str, str]] = set()

    for turn_idx, msg in earlier_turns:
        role = msg.get("role", "")
        text = _get_message_text(msg)
        if not text:
            continue

        # Extract constraints from user turns only
        if role == "user" and len(constraints) < max_constraints:
            extracted_c = _extract_constraints(text, turn_idx)
            for c in extracted_c:
                if c.key not in seen_constraint_keys:
                    seen_constraint_keys.add(c.key)
                    constraints.append(c)
                    if len(constraints) >= max_constraints:
                        break

        # Extract code facts from all roles (fenced blocks first, then un-fenced definitions)
        if len(code_facts) < max_code_facts:
            fenced_blocks = FENCED_CODE_RE.findall(text)
            for block in fenced_blocks:
                facts = _extract_from_ast(block, turn_idx)
                for f in facts:
                    key_pair = (f.kind, f.key)
                    if key_pair not in seen_code_keys:
                        seen_code_keys.add(key_pair)
                        code_facts.append(f)
                        if len(code_facts) >= max_code_facts:
                            break
                if len(code_facts) >= max_code_facts:
                    break

            if len(code_facts) < max_code_facts:
                outside_text = FENCED_CODE_RE.sub(" ", text)
                if any(k in outside_text for k in ("def ", "class ", "=")):
                    regex_facts = _extract_from_regex(outside_text, turn_idx)
                    for f in regex_facts:
                        key_pair = (f.kind, f.key)
                        if key_pair not in seen_code_keys:
                            seen_code_keys.add(key_pair)
                            code_facts.append(f)
                            if len(code_facts) >= max_code_facts:
                                break

    return constraints + code_facts
