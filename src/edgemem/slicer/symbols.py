from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

from edgemem.slicer.tags import (
    _DEF_NODE_TYPES,
    _IDENT_NODE_TYPES,
    _NAME_CHILD_FIELD,
    _find_ident_node,
    _get_parser,
    _is_binary,
    _lang_for_path,
)


@dataclass(frozen=True)
class SymbolDef:
    name: str
    qualified_name: str
    kind: str
    file_path: str
    start_line: int
    end_line: int
    signature: str
    body: str
    references: frozenset[str]


@dataclass
class SymbolSliceResult:
    symbol: str
    found: bool
    target_file: str | None
    callee_count: int
    caller_count: int
    sliced_tokens: int
    raw_tokens: int
    content: str
    fallback_used: bool


_BUILTINS: frozenset[str] = frozenset({
    "self", "cls", "None", "True", "False", "str", "int", "float", "bool",
    "dict", "list", "set", "tuple", "len", "print", "range", "super",
    "isinstance", "issubclass", "type", "return", "def", "class", "async",
    "await", "import", "from", "for", "in", "if", "else", "elif", "while",
    "try", "except", "finally", "with", "as", "pass", "break", "continue",
    "raise", "yield", "lambda", "global", "nonlocal", "assert", "del",
    "object", "Exception", "ValueError", "TypeError", "RuntimeError",
    "KeyError", "IndexError", "AttributeError", "StopIteration", "any", "all",
    "min", "max", "sum", "abs", "round", "map", "filter", "zip", "enumerate",
    "sorted", "reversed", "open", "id", "hash", "callable", "iter", "next",
    "hasattr", "getattr", "setattr", "delattr", "property", "staticmethod",
    "classmethod", "this", "function", "var", "let", "const", "new",
})


def _extract_references_from_node(node, source_bytes: bytes, exclude: set[str]) -> set[str]:
    refs: set[str] = set()
    ident_types = {"identifier", "type_identifier", "property_identifier", "field_identifier", "name"}

    def _walk(n) -> None:
        if n.type in ident_types:
            sym = source_bytes[n.start_byte:n.end_byte].decode("utf-8", errors="replace").strip()
            if sym and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", sym) and len(sym) > 1:
                if sym not in exclude and sym not in _BUILTINS:
                    refs.add(sym)
        for c in n.children:
            _walk(c)

    _walk(node)
    return refs


def _extract_references_regex(text: str, exclude: set[str]) -> set[str]:
    refs: set[str] = set()
    for tok in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{1,}\b", text):
        if tok not in exclude and tok not in _BUILTINS:
            refs.add(tok)
    return refs


def extract_symbols(
    abs_path: Path,
    rel_path: str,
    content: str | None = None,
) -> list[SymbolDef]:
    clean_rel_path = Path(rel_path).as_posix().replace("\\", "/")
    if _is_binary(abs_path):
        return []

    try:
        text = content if content is not None else abs_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    if len(text) < 4:
        return []

    lang = _lang_for_path(abs_path)
    if lang is None:
        return _regex_fallback_symbols(text, clean_rel_path)

    parser = _get_parser(lang)
    if parser is None:
        return _regex_fallback_symbols(text, clean_rel_path)

    try:
        source_bytes = text.encode("utf-8", errors="replace")
        tree = parser.parse(source_bytes)
        def_types = _DEF_NODE_TYPES.get(lang, set())
        name_field = _NAME_CHILD_FIELD.get(lang, "name")

        symbols: list[SymbolDef] = []
        scope_stack: list[str] = []

        def _traverse(node) -> None:
            ntype = node.type
            is_def = ntype in def_types
            pushed_scope = False

            if is_def:
                target = node.child_by_field_name(name_field) or node.child_by_field_name("name") or node
                ident = _find_ident_node(target)
                if ident is not None:
                    name = source_bytes[ident.start_byte:ident.end_byte].decode("utf-8", errors="replace").strip()
                    if name and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
                        start_line = node.start_point[0] + 1
                        end_line = node.end_point[0] + 1
                        body_slice = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
                        first_line = body_slice.splitlines()[0].strip() if body_slice else ""
                        qname = ".".join(scope_stack + [name]) if scope_stack else name

                        kind = "function"
                        if "class" in ntype or "struct" in ntype or "interface" in ntype:
                            kind = "class" if "class" in ntype else ("interface" if "interface" in ntype else "struct")
                        elif scope_stack:
                            kind = "method"
                        elif "async" in ntype:
                            kind = "async_function"

                        exclude = {name, qname}
                        refs = _extract_references_from_node(node, source_bytes, exclude)

                        symbols.append(
                            SymbolDef(
                                name=name,
                                qualified_name=qname,
                                kind=kind,
                                file_path=clean_rel_path,
                                start_line=start_line,
                                end_line=end_line,
                                signature=first_line,
                                body=body_slice,
                                references=frozenset(refs),
                            )
                        )

                        if kind in ("class", "interface", "struct"):
                            scope_stack.append(name)
                            pushed_scope = True

            for child in node.children:
                _traverse(child)

            if pushed_scope:
                scope_stack.pop()

        _traverse(tree.root_node)
        return symbols
    except Exception:
        return _regex_fallback_symbols(text, clean_rel_path)


def _regex_fallback_symbols(text: str, rel_path: str) -> list[SymbolDef]:
    symbols: list[SymbolDef] = []
    lines = text.splitlines()
    scope_stack: list[tuple[int, str]] = []

    for i, line in enumerate(lines):
        indent = len(line) - len(line.lstrip())
        stripped = line.strip()

        while scope_stack and indent <= scope_stack[-1][0] and stripped:
            scope_stack.pop()

        m_func = re.match(r"^(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)", stripped)
        m_cls = re.match(r"^class\s+([A-Za-z_][A-Za-z0-9_]*)", stripped)

        if m_cls:
            cname = m_cls.group(1)
            start_line = i + 1
            body_lines = [line]
            for j in range(i + 1, len(lines)):
                next_line = lines[j]
                if next_line.strip() and (len(next_line) - len(next_line.lstrip())) <= indent:
                    break
                body_lines.append(next_line)
            end_line = start_line + len(body_lines) - 1
            body_text = "\n".join(body_lines)
            qname = ".".join([s[1] for s in scope_stack] + [cname]) if scope_stack else cname
            exclude = {cname, qname}
            refs = _extract_references_regex(body_text, exclude)
            symbols.append(
                SymbolDef(
                    name=cname,
                    qualified_name=qname,
                    kind="class",
                    file_path=rel_path,
                    start_line=start_line,
                    end_line=end_line,
                    signature=stripped,
                    body=body_text,
                    references=frozenset(refs),
                )
            )
            scope_stack.append((indent, cname))

        elif m_func:
            fname = m_func.group(1)
            start_line = i + 1
            body_lines = [line]
            for j in range(i + 1, len(lines)):
                next_line = lines[j]
                if next_line.strip() and (len(next_line) - len(next_line.lstrip())) <= indent:
                    break
                body_lines.append(next_line)
            end_line = start_line + len(body_lines) - 1
            body_text = "\n".join(body_lines)
            kind = "method" if scope_stack else ("async_function" if "async def" in stripped else "function")
            qname = ".".join([s[1] for s in scope_stack] + [fname]) if scope_stack else fname
            exclude = {fname, qname}
            refs = _extract_references_regex(body_text, exclude)
            symbols.append(
                SymbolDef(
                    name=fname,
                    qualified_name=qname,
                    kind=kind,
                    file_path=rel_path,
                    start_line=start_line,
                    end_line=end_line,
                    signature=stripped,
                    body=body_text,
                    references=frozenset(refs),
                )
            )

    return symbols
