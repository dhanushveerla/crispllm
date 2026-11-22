from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Sequence

from crispllm.slicer.tags import (
    _DEF_NODE_TYPES,
    _IDENT_NODE_TYPES,
    _NAME_CHILD_FIELD,
    _find_ident_node,
    _get_parser,
    _is_binary,
    _lang_for_path,
)

GENERIC_CALL_NAMES: frozenset[str] = frozenset({
    "get", "set", "put", "pop", "items", "keys", "values", "update", "clear",
    "copy", "append", "extend", "insert", "remove", "strip", "lstrip", "rstrip",
    "split", "rsplit", "join", "replace", "lower", "upper", "startswith",
    "endswith", "format", "encode", "decode", "read", "write", "close", "open",
    "flush", "len", "str", "int", "float", "bool", "dict", "list", "tuple",
    "isinstance", "issubclass", "getattr", "setattr", "hasattr", "super",
    "print", "next", "iter", "any", "all", "min", "max", "sum", "sorted",
    "enumerate", "zip", "range", "map", "filter", "add", "discard", "find",
    "count", "index", "run", "start", "stop", "send", "dump", "dumps", "load",
    "loads", "match", "search", "sub", "compile", "parse", "exists", "mkdir",
    "unlink", "resolve", "as_posix", "read_text", "write_text", "setdefault",
})

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

_SYMBOL_DEF_NODE_TYPES: dict[str, set[str]] = dict(_DEF_NODE_TYPES)
_SYMBOL_DEF_NODE_TYPES["python"] = {
    "function_definition",
    "async_function_definition",
    "class_definition",
}


@dataclass(frozen=True)
class SymbolDef:
    name: str
    qualified_name: str
    kind: str
    file_path: str
    start_line: int
    end_line: int
    signature: str
    body: str = ""
    references: frozenset[str] = frozenset()
    enclosing_class: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "qualified_name": self.qualified_name,
            "kind": self.kind,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "signature": self.signature,
            "references": sorted(self.references),
            "enclosing_class": self.enclosing_class,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], body: str = "") -> SymbolDef:
        return cls(
            name=d["name"],
            qualified_name=d["qualified_name"],
            kind=d["kind"],
            file_path=d["file_path"],
            start_line=d["start_line"],
            end_line=d["end_line"],
            signature=d["signature"],
            body=body,
            references=frozenset(d.get("references", ())),
            enclosing_class=d.get("enclosing_class"),
        )


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


def _clean_signature(raw: str, max_len: int = 140) -> str:
    cleaned = re.sub(r"\s+", " ", raw).strip()
    cleaned = re.sub(r"\(\s+", "(", cleaned)
    cleaned = re.sub(r",\s*\)", ")", cleaned)
    cleaned = re.sub(r"\s+\)", ")", cleaned)
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3] + "..."


def extract_imports(text: str) -> list[str]:
    imports: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m_from = re.match(r"^from\s+([A-Za-z0-9_.]+)\s+import\s+(.+)$", line)
        if m_from:
            mod = m_from.group(1).lstrip(".")
            if mod:
                imports.add(mod)
                for part in mod.split("."):
                    if part:
                        imports.add(part)
            names_part = m_from.group(2).split("#")[0]
            for item in names_part.split(","):
                item = item.strip()
                if item:
                    sym = item.split(" as ")[0].strip(" ()")
                    if sym and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", sym):
                        imports.add(sym)
            continue
        m_imp = re.match(r"^import\s+(.+)$", line)
        if m_imp:
            names_part = m_imp.group(1).split("#")[0]
            for item in names_part.split(","):
                item = item.strip()
                if item:
                    mod = item.split(" as ")[0].strip()
                    if mod:
                        imports.add(mod)
                        for part in mod.split("."):
                            if part:
                                imports.add(part)
            continue
        m_js = re.match(
            r"^import\s+(?:\{([^}]+)\}|\*\s+as\s+([A-Za-z0-9_]+)|([A-Za-z0-9_]+))\s+from\s+['\"]([^'\"]+)['\"]",
            line,
        )
        if m_js:
            if m_js.group(1):
                for s in m_js.group(1).split(","):
                    s = s.strip().split(" as ")[0].strip()
                    if s:
                        imports.add(s)
            if m_js.group(2):
                imports.add(m_js.group(2).strip())
            if m_js.group(3):
                imports.add(m_js.group(3).strip())
            if m_js.group(4):
                mod_path = Path(m_js.group(4)).stem
                imports.add(mod_path)
    return sorted(imports)


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


def extract_symbols_and_imports(
    abs_path: Path,
    rel_path: str,
    content: str | None = None,
) -> tuple[list[str], list[SymbolDef]]:
    clean_rel_path = Path(rel_path).as_posix().replace("\\", "/")
    if _is_binary(abs_path):
        return [], []

    try:
        text = content if content is not None else abs_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return [], []

    if len(text) < 4:
        return [], []

    imports = extract_imports(text)

    lang = _lang_for_path(abs_path)
    if lang is None:
        return imports, _regex_fallback_symbols(text, clean_rel_path)

    parser = _get_parser(lang)
    if parser is None:
        return imports, _regex_fallback_symbols(text, clean_rel_path)

    try:
        source_bytes = text.encode("utf-8", errors="replace")
        tree = parser.parse(source_bytes)
        def_types = _SYMBOL_DEF_NODE_TYPES.get(lang, set())
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
                        body_node = node.child_by_field_name("body")
                        if body_node is not None and body_node.start_byte > node.start_byte:
                            raw_sig = source_bytes[node.start_byte:body_node.start_byte].decode("utf-8", errors="replace").strip()
                            if raw_sig.endswith(":"):
                                raw_sig = raw_sig[:-1].strip()
                        else:
                            first_line = body_slice.splitlines()[0].strip() if body_slice else ""
                            raw_sig = first_line

                        sig = _clean_signature(raw_sig, max_len=140)
                        qname = ".".join(scope_stack + [name]) if scope_stack else name

                        kind = "function"
                        enclosing_class = scope_stack[-1] if scope_stack else None
                        if "class" in ntype or "struct" in ntype or "interface" in ntype:
                            kind = "class" if "class" in ntype else ("interface" if "interface" in ntype else "struct")
                            enclosing_class = scope_stack[-1] if scope_stack else None
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
                                signature=sig,
                                body=body_slice,
                                references=frozenset(refs),
                                enclosing_class=enclosing_class,
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
        return imports, symbols
    except Exception:
        return imports, _regex_fallback_symbols(text, clean_rel_path)


def extract_symbols(
    abs_path: Path,
    rel_path: str,
    content: str | None = None,
) -> list[SymbolDef]:
    return extract_symbols_and_imports(abs_path, rel_path, content=content)[1]


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
            enclosing = scope_stack[-1][1] if scope_stack else None
            qname = ".".join([s[1] for s in scope_stack] + [cname]) if scope_stack else cname
            exclude = {cname, qname}
            refs = _extract_references_regex(body_text, exclude)

            sig_parts = [stripped]
            for k in range(i + 1, min(i + 10, len(lines))):
                if stripped.endswith(":"):
                    break
                nxt = lines[k].strip()
                sig_parts.append(nxt)
                if nxt.endswith(":"):
                    break
            raw_sig = " ".join(sig_parts)
            if raw_sig.endswith(":"):
                raw_sig = raw_sig[:-1].strip()
            clean_sig = _clean_signature(raw_sig, max_len=140)

            symbols.append(
                SymbolDef(
                    name=cname,
                    qualified_name=qname,
                    kind="class",
                    file_path=rel_path,
                    start_line=start_line,
                    end_line=end_line,
                    signature=clean_sig,
                    body=body_text,
                    references=frozenset(refs),
                    enclosing_class=enclosing,
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
            enclosing = scope_stack[-1][1] if scope_stack else None
            kind = "method" if scope_stack else ("async_function" if "async def" in stripped else "function")
            qname = ".".join([s[1] for s in scope_stack] + [fname]) if scope_stack else fname
            exclude = {fname, qname}
            refs = _extract_references_regex(body_text, exclude)

            sig_parts = [stripped]
            for k in range(i + 1, min(i + 15, len(lines))):
                if stripped.endswith(":") or "):" in stripped:
                    break
                nxt = lines[k].strip()
                sig_parts.append(nxt)
                if nxt.endswith(":") or "):" in nxt:
                    break
            raw_sig = " ".join(sig_parts)
            if raw_sig.endswith(":"):
                raw_sig = raw_sig[:-1].strip()
            clean_sig = _clean_signature(raw_sig, max_len=140)

            symbols.append(
                SymbolDef(
                    name=fname,
                    qualified_name=qname,
                    kind=kind,
                    file_path=rel_path,
                    start_line=start_line,
                    end_line=end_line,
                    signature=clean_sig,
                    body=body_text,
                    references=frozenset(refs),
                    enclosing_class=enclosing,
                )
            )

    return symbols
