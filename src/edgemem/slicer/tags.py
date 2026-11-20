"""edgemem.slicer.tags — Multi-language AST tag extraction via tree-sitter.

Adapts Aider's tag-extraction strategy (Apache-2.0).
Supports Python, JS/TS, Go, Rust, C/C++ with graceful fallback.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

# ---------------------------------------------------------------------------
# Language mapping: file extension → tree-sitter language name
# ---------------------------------------------------------------------------

_EXT_TO_LANG: dict[str, str] = {
    ".py":   "python",
    ".js":   "javascript",
    ".jsx":  "javascript",
    ".ts":   "typescript",
    ".tsx":  "tsx",
    ".go":   "go",
    ".rs":   "rust",
    ".c":    "c",
    ".h":    "c",
    ".cc":   "cpp",
    ".cpp":  "cpp",
    ".cxx":  "cpp",
    ".hpp":  "cpp",
    ".hxx":  "cpp",
    ".cs":   "c_sharp",
    ".rb":   "ruby",
    ".java": "java",
    ".kt":   "kotlin",
    ".swift":"swift",
    ".lua":  "lua",
    ".ex":   "elixir",
    ".exs":  "elixir",
    ".hs":   "haskell",
    ".scala":"scala",
    ".php":  "php",
    ".r":    "r",
    ".toml": None,  # explicitly unsupported (no grammar)
    ".json": None,
    ".yaml": None,
    ".yml":  None,
    ".md":   None,
    ".txt":  None,
}

# Minimum bytes for a file to be considered worth parsing
_MIN_FILE_BYTES = 4

# Binary file detection: if >5% of the first 8 KB is non-text, skip
_BINARY_SAMPLE = 8192
_BINARY_THRESHOLD = 0.05


def _is_binary(path: Path) -> bool:
    try:
        chunk = path.read_bytes()[:_BINARY_SAMPLE]
        non_text = sum(1 for b in chunk if b < 9 or (b > 13 and b < 32) or b == 127)
        return non_text / max(len(chunk), 1) > _BINARY_THRESHOLD
    except OSError:
        return True


def _lang_for_path(path: Path) -> str | None:
    """Return tree-sitter language name or None if unsupported/binary."""
    ext = path.suffix.lower()
    if ext not in _EXT_TO_LANG:
        return None
    return _EXT_TO_LANG[ext]


# ---------------------------------------------------------------------------
# Tag dataclass
# ---------------------------------------------------------------------------

@dataclass
class Tag:
    """A single definition or reference extracted from a source file."""
    rel_path: str           # path relative to repo root
    abs_path: str           # absolute path
    name: str               # symbol name (class, function, variable…)
    line: int               # 0-based line number
    kind: str               # 'def' | 'ref'
    info: str = ""          # optional: return type, signature snippet, etc.


# ---------------------------------------------------------------------------
# tree-sitter query strings per language (definition node types)
# ---------------------------------------------------------------------------

# Queries capture definition nodes; we also capture identifiers as references.
# Format: list of (node_type, capture_name) used in a simple node-walk
# (tree-sitter queries require a compiled language object; we use a simpler
# node-type walk so we don't need language-specific .scm query files).

_DEF_NODE_TYPES: dict[str, set[str]] = {
    "python": {
        "function_definition", "async_function_definition",
        "class_definition", "decorated_definition",
    },
    "javascript": {
        "function_declaration", "function_expression", "arrow_function",
        "class_declaration", "method_definition", "export_statement",
    },
    "typescript": {
        "function_declaration", "function_expression", "arrow_function",
        "class_declaration", "method_definition", "export_statement",
        "interface_declaration", "type_alias_declaration",
    },
    "tsx": {
        "function_declaration", "function_expression", "arrow_function",
        "class_declaration", "method_definition", "export_statement",
        "interface_declaration", "type_alias_declaration",
    },
    "go": {
        "function_declaration", "method_declaration", "type_declaration",
        "type_spec",
    },
    "rust": {
        "function_item", "impl_item", "struct_item", "enum_item",
        "trait_item", "mod_item", "fn_item",
    },
    "c": {
        "function_definition", "declaration", "struct_specifier",
        "enum_specifier",
    },
    "cpp": {
        "function_definition", "declaration", "class_specifier",
        "struct_specifier", "enum_specifier", "template_declaration",
    },
    "c_sharp": {
        "method_declaration", "class_declaration", "interface_declaration",
        "struct_declaration",
    },
    "ruby": {
        "method", "class", "module", "singleton_method",
    },
    "java": {
        "method_declaration", "class_declaration", "interface_declaration",
    },
    "kotlin": {
        "function_declaration", "class_declaration", "object_declaration",
    },
    "swift": {
        "function_declaration", "class_declaration", "struct_declaration",
    },
    "lua": {"function_definition", "local_function"},
    "elixir": {"call"},
    "haskell": {"function"},
    "scala": {"def", "class", "object", "trait"},
    "php": {"function_definition", "method_declaration", "class_declaration"},
    "r": {"binary"},
}

# Identifier node types (used for reference extraction)
_IDENT_NODE_TYPES: dict[str, set[str]] = {
    "python": {"identifier"},
    "javascript": {"identifier", "property_identifier"},
    "typescript": {"identifier", "type_identifier", "property_identifier"},
    "tsx": {"identifier", "type_identifier", "property_identifier"},
    "go": {"identifier", "field_identifier", "type_identifier"},
    "rust": {"identifier", "type_identifier"},
    "c": {"identifier", "type_identifier"},
    "cpp": {"identifier", "type_identifier", "field_identifier"},
    "c_sharp": {"identifier"},
    "ruby": {"identifier", "constant"},
    "java": {"identifier", "type_identifier"},
    "kotlin": {"identifier"},
    "swift": {"identifier"},
    "lua": {"identifier"},
    "elixir": {"identifier"},
    "haskell": {"variable"},
    "scala": {"identifier"},
    "php": {"name"},
    "r": {"identifier"},
}

# Name child field priorities per language for extracting the symbol name
_NAME_CHILD_FIELD: dict[str, str] = {
    "python": "name",
    "javascript": "name",
    "typescript": "name",
    "tsx": "name",
    "go": "name",
    "rust": "name",
    "c": "declarator",
    "cpp": "declarator",
    "c_sharp": "name",
    "ruby": "name",
    "java": "name",
    "kotlin": "name",
    "swift": "name",
    "lua": "name",
    "elixir": "function",
    "haskell": "name",
    "scala": "name",
    "php": "name",
    "r": "left",
}


def _get_parser(lang: str):
    """Return a tree-sitter Parser for the given language, or None."""
    try:
        from tree_sitter_language_pack import get_parser  # type: ignore
        return get_parser(lang)
    except Exception:
        return None


def _find_ident_node(node):
    """Recursively search for the primary identifier node within a declaration."""
    if node.type in ("identifier", "type_identifier", "field_identifier", "name"):
        return node
    for c in node.children:
        # Don't descend into parameters or bodies when locating definition name
        if c.type in ("parameters", "parameter_list", "block", "compound_statement", "statement_block", "field_declaration_list"):
            continue
        res = _find_ident_node(c)
        if res is not None:
            return res
    return None


def _walk_tree(
    node,
    def_types: set[str],
    ident_types: set[str],
    name_field: str,
    source_bytes: bytes,
    rel_path: str,
    abs_path: str,
    tags: list[Tag],
    def_spans: set[tuple[int, int]],
) -> None:
    """Recursive tree walk to collect definition and reference tags."""
    ntype = node.type

    if ntype in def_types:
        target = node.child_by_field_name(name_field) or node.child_by_field_name("name") or node
        ident = _find_ident_node(target)
        if ident is not None:
            sym = source_bytes[ident.start_byte:ident.end_byte].decode("utf-8", errors="replace").strip()
            if sym and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", sym):
                tags.append(Tag(
                    rel_path=rel_path,
                    abs_path=abs_path,
                    name=sym,
                    line=node.start_point[0],
                    kind="def",
                ))
                def_spans.add((ident.start_byte, ident.end_byte))

    elif ntype in ident_types:
        span = (node.start_byte, node.end_byte)
        if span not in def_spans:
            sym = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace").strip()
            if sym and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", sym) and len(sym) > 1:
                tags.append(Tag(
                    rel_path=rel_path,
                    abs_path=abs_path,
                    name=sym,
                    line=node.start_point[0],
                    kind="ref",
                ))

    for child in node.children:
        _walk_tree(
            child,
            def_types,
            ident_types,
            name_field,
            source_bytes,
            rel_path,
            abs_path,
            tags,
            def_spans,
        )


def extract_tags(
    abs_path: Path,
    rel_path: str,
    content: str | None = None,
) -> list[Tag]:
    """
    Extract definition and reference tags from a source file.

    Falls back to a simple regex pass for unsupported extensions or parse errors.
    Returns an empty list for binary files.
    """
    if _is_binary(abs_path):
        return []

    lang = _lang_for_path(abs_path)

    try:
        text = content if content is not None else abs_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    if len(text) < _MIN_FILE_BYTES:
        return []

    tags: list[Tag] = []

    if lang is None:
        # No grammar — return simple regex-based defs for Python at least
        return _regex_fallback_tags(text, rel_path, str(abs_path))

    parser = _get_parser(lang)
    if parser is None:
        return _regex_fallback_tags(text, rel_path, str(abs_path))

    try:
        source_bytes = text.encode("utf-8", errors="replace")
        tree = parser.parse(source_bytes)
        def_types  = _DEF_NODE_TYPES.get(lang, set())
        ident_types = _IDENT_NODE_TYPES.get(lang, set())
        name_field  = _NAME_CHILD_FIELD.get(lang, "name")
        _walk_tree(
            tree.root_node,
            def_types,
            ident_types,
            name_field,
            source_bytes,
            rel_path,
            str(abs_path),
            tags,
            def_spans=set(),
        )
    except Exception:
        return _regex_fallback_tags(text, rel_path, str(abs_path))

    return tags


def _regex_fallback_tags(text: str, rel_path: str, abs_path: str) -> list[Tag]:
    """Simple regex fallback for Python defs when tree-sitter unavailable."""
    tags: list[Tag] = []
    for i, line in enumerate(text.splitlines()):
        m = re.match(r"^\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)", line)
        if m:
            tags.append(Tag(rel_path=rel_path, abs_path=abs_path,
                            name=m.group(1), line=i, kind="def"))
        m2 = re.match(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)", line)
        if m2:
            tags.append(Tag(rel_path=rel_path, abs_path=abs_path,
                            name=m2.group(1), line=i, kind="def"))
    return tags


# ---------------------------------------------------------------------------
# File scanning helpers (shared with repomap.py)
# ---------------------------------------------------------------------------

_IGNORE_DIRS: frozenset[str] = frozenset({
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache",
    ".mypy_cache", "node_modules", "dist", "build", ".tox", ".eggs",
    ".aider", ".idea", ".vscode",
})

_IGNORE_EXTS: frozenset[str] = frozenset({
    ".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib",
    ".exe", ".bin", ".obj", ".o", ".a", ".lib",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".ico", ".svg",
    ".mp3", ".mp4", ".wav", ".avi",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".lock",
})


def _load_gitignore_patterns(repo_root: Path) -> list[re.Pattern]:
    """Parse .gitignore and return compiled glob patterns as regex."""
    gi = repo_root / ".gitignore"
    patterns: list[re.Pattern] = []
    if not gi.exists():
        return patterns
    try:
        for line in gi.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            clean = line.rstrip("/")
            pat = re.escape(clean).replace(r"\*\*", ".*").replace(r"\*", "[^/]*").replace(r"\?", ".")
            if not pat.startswith("/"):
                pat = "(^|.*/)" + pat
            else:
                pat = "^" + pat[1:]
            pat = pat + "(/.*|$)"
            try:
                patterns.append(re.compile(pat))
            except re.error:
                pass
    except OSError:
        pass
    return patterns


def scan_repo_files(
    repo_root: Path,
    extra_ignore_dirs: frozenset[str] | None = None,
) -> list[Path]:
    """
    Walk repo_root, respecting .gitignore and default ignore dirs.
    Returns sorted list of absolute paths to source files.
    """
    ignore_dirs = _IGNORE_DIRS | (extra_ignore_dirs or frozenset())
    gi_patterns = _load_gitignore_patterns(repo_root)
    files: list[Path] = []

    for root, dirs, filenames in __import__("os").walk(repo_root):
        root_path = Path(root)
        # Filter dirs in-place
        dirs[:] = [
            d for d in dirs
            if d not in ignore_dirs
            and not d.startswith(".")
            and not _is_gitignored(root_path / d, repo_root, gi_patterns)
        ]
        for fname in filenames:
            fpath = root_path / fname
            if fpath.suffix.lower() in _IGNORE_EXTS:
                continue
            if _is_gitignored(fpath, repo_root, gi_patterns):
                continue
            if _is_binary(fpath):
                continue
            files.append(fpath)

    return sorted(files)


def _is_gitignored(path: Path, root: Path, patterns: list[re.Pattern]) -> bool:
    try:
        rel = str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return False
    for pat in patterns:
        if pat.search(rel):
            return True
    return False
