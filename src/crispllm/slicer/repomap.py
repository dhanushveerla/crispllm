"""edgemem.slicer.repomap — Token-budgeted repository map via AST + PageRank.

Adapts Aider's RepoMap algorithm (Apache-2.0).  Fully OS-agnostic.
No numpy or scipy.

Public API
----------
RepoMap(repo_path, max_tokens=1024, model_name="cl100k_base")
  .get_repo_map(
      focus_files=None,
      mentioned_symbols=None,
      query="",
  ) -> str

The returned string is a compact, ranked repo map that fits within
`max_tokens` as counted by tiktoken's `cl100k_base` encoder.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import tiktoken
from grep_ast import TreeContext

from crispllm.slicer.cache import OutlineCache, SymbolCache, TagCache
from crispllm.slicer.pagerank import (
    build_symbol_graph,
    compute_pagerank,
    find_matching_files,
    is_test_file,
    personalise_ranks,
)
from crispllm.slicer.symbols import (
    GENERIC_CALL_NAMES,
    SymbolDef,
    SymbolSliceResult,
    extract_symbols,
    extract_symbols_and_imports,
)
from crispllm.slicer.tags import Tag, extract_tags, scan_repo_files

logger = logging.getLogger("crispllm.slicer.repomap")

_GRAPH_CACHE: OrderedDict[tuple[str, str], tuple[list[Tag], int, Any, dict[str, float]]] = OrderedDict()
_MAX_GRAPH_ENTRIES = 4


def clear_graph_cache() -> None:
    _GRAPH_CACHE.clear()


def render_tree_context(
    filename: str,
    code: str,
    def_lines: Sequence[int],
    max_lines: int | None = 35,
) -> str:
    try:
        tc = TreeContext(
            filename=Path(filename).as_posix(),
            code=code,
            color=False,
            verbose=False,
            line_number=False,
            child_context=False,
            last_line=False,
            margin=0,
            show_top_of_file_parent_scope=False,
        )
        tc.add_lines_of_interest(list(def_lines))
        tc.add_context()
        formatted = tc.format()
        if max_lines is not None:
            flines = formatted.splitlines()
            if len(flines) > max_lines:
                formatted = "\n".join(flines[:max_lines])
        return formatted
    except Exception:
        lines = code.splitlines()
        matched = [lines[ln] for ln in def_lines if ln < len(lines)]
        if max_lines is not None and len(matched) > max_lines:
            matched = matched[:max_lines]
        return "\n".join(matched)


_ENCODINGS: dict[str, tiktoken.Encoding] = {}


def _get_enc(model_name: str = "cl100k_base") -> tiktoken.Encoding:
    if model_name not in _ENCODINGS:
        if "TIKTOKEN_CACHE_DIR" not in os.environ:
            from platformdirs import user_cache_dir
            for cand in [
                Path(user_cache_dir("crispllm")) / "tiktoken",
                Path.home() / ".cache" / "tiktoken",
                Path(tempfile.gettempdir()) / "data-gym-cache",
            ]:
                if cand.exists():
                    os.environ["TIKTOKEN_CACHE_DIR"] = str(cand)
                    break
        _ENCODINGS[model_name] = tiktoken.get_encoding(model_name)
    return _ENCODINGS[model_name]


def count_tokens(text: str, model_name: str = "cl100k_base") -> int:
    enc = _get_enc(model_name)
    return len(enc.encode(text, disallowed_special=()))



_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}")


def extract_query_tokens(query: str) -> list[str]:
    """Extract Python-identifier-like tokens from a natural language query."""
    return _IDENT_RE.findall(query)



def file_outline(
    abs_path: Path,
    rel_path: str | None = None,
    cache: OutlineCache | None = None,
) -> str:
    rp = (Path(rel_path).as_posix() if rel_path else Path(abs_path).as_posix()).replace("\\", "/")
    try:
        st = abs_path.stat()
    except OSError:
        return f"# Could not read {rp}\n"

    try:
        content = abs_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return f"# Could not read {rp}\n"

    def_lines = []
    for i, line in enumerate(content.splitlines()):
        stripped = line.strip()
        if stripped.startswith((
            "def ", "async def ", "class ", "fn ", "func ",
            "function ", "pub fn", "pub async", "type ",
            "interface ", "impl ", "struct ", "enum ",
        )):
            def_lines.append(i)

    if not def_lines:
        return "\n".join(content.splitlines()[:20])

    lines_key = f"lines:{','.join(map(str, def_lines))};max_lines=none"
    if cache is not None:
        cached = cache.get(rp, st.st_mtime_ns, st.st_size, lines_key)
        if cached is not None:
            return cached[0]

    formatted = render_tree_context(rp, content, def_lines, max_lines=None)
    if not formatted.strip():
        lines = []
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith(("def ", "async def ", "class ")):
                lines.append(line.rstrip())
        formatted = "\n".join(lines[:40]) if lines else f"# {rp}: (no definitions found)\n"

    if cache is not None:
        tok_cnt = count_tokens(formatted)
        cache.put(rp, st.st_mtime_ns, st.st_size, lines_key, formatted, tok_cnt)

    return formatted


class RepoMap:
    def __init__(
        self,
        repo_path: str | Path,
        max_tokens: int = 1024,
        model_name: str = "cl100k_base",
        cache: TagCache | None = None,
        outline_cache: OutlineCache | None = None,
        max_outline_entries: int = 256,
    ) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.max_tokens = max_tokens
        self.model_name = model_name
        self._cache = cache or TagCache(max_outline_entries=max_outline_entries)
        self.outline_cache = outline_cache or self._cache.outline_cache
        sc = getattr(self._cache, "symbol_cache", None)
        self.symbol_cache: SymbolCache = sc if sc is not None else SymbolCache(conn=getattr(self._cache, "_conn", None))
        self._enc: tiktoken.Encoding | None = None
        self._cached_raw_tokens: int | None = None


    def get_repo_map(
        self,
        focus_files: list[str] | None = None,
        mentioned_symbols: list[str] | None = None,
        query: str = "",
        max_tokens: int | None = None,
    ) -> str:
        t0 = time.perf_counter()

        if focus_files:
            focus_files = [Path(f).as_posix().replace("\\", "/") for f in focus_files]

        all_files = scan_repo_files(self.repo_path)
        if not all_files:
            return ""

        file_stats: list[tuple[str, int, int]] = []
        for fpath in all_files:
            try:
                rel = fpath.relative_to(self.repo_path).as_posix()
            except ValueError:
                rel = fpath.as_posix()
            try:
                st = fpath.stat()
                file_stats.append((rel, st.st_mtime_ns, st.st_size))
            except OSError:
                file_stats.append((rel, 0, 0))

        file_stats.sort()
        hasher = hashlib.sha256()
        for item in file_stats:
            hasher.update(f"{item[0]}:{item[1]}:{item[2]};".encode("utf-8"))
        repo_fingerprint = hasher.hexdigest()
        graph_key = (str(self.repo_path), repo_fingerprint)

        if graph_key in _GRAPH_CACHE:
            all_tags, raw_tok_sum, G, base_ranks = _GRAPH_CACHE[graph_key]
            _GRAPH_CACHE.move_to_end(graph_key)
        else:
            all_tags: list[Tag] = []
            raw_tok_sum = 0

            for fpath in all_files:
                try:
                    rel = fpath.relative_to(self.repo_path).as_posix()
                except ValueError:
                    rel = fpath.as_posix()

                cached = self._cache.get(fpath)
                if cached is not None:
                    tags, tok_cnt = cached
                    all_tags.extend(tags)
                    raw_tok_sum += tok_cnt
                else:
                    try:
                        content = fpath.read_text(encoding="utf-8", errors="replace")
                        tok_cnt = self._count(content)
                    except OSError:
                        content = ""
                        tok_cnt = 0
                    tags = extract_tags(fpath, rel, content=content)
                    all_tags.extend(tags)
                    raw_tok_sum += tok_cnt
                    self._cache.put(fpath, tags, token_count=tok_cnt)

            if not all_tags:
                return ""

            G = build_symbol_graph(all_tags)
            base_ranks = compute_pagerank(G)

            if len(_GRAPH_CACHE) >= _MAX_GRAPH_ENTRIES:
                _GRAPH_CACHE.popitem(last=False)
            _GRAPH_CACHE[graph_key] = (all_tags, raw_tok_sum, G, base_ranks)

        if self._cached_raw_tokens is None:
            self._cached_raw_tokens = raw_tok_sum

        if not all_tags:
            return ""

        query_tokens = extract_query_tokens(query) if query else []

        match_scores = find_matching_files(
            all_files=list(G.nodes()),
            tags=all_tags,
            focus_files=focus_files,
            mentioned_symbols=mentioned_symbols,
            query_tokens=query_tokens,
            query=query,
        )

        query_mentions_test = bool(query and "test" in query.lower())
        focus_has_test = bool(focus_files and any(is_test_file(f) for f in focus_files))
        penalize_tests = not (query_mentions_test or focus_has_test)

        personalization: dict[str, float] | None = None
        weights: dict[str, float] = {}
        for node in G.nodes():
            w = 1.0
            if penalize_tests and is_test_file(node):
                w = 0.15
            if node in match_scores:
                w *= (10.0 * match_scores[node])
            weights[node] = w
        if any(w != 1.0 for w in weights.values()):
            personalization = weights

        if personalization is not None:
            computed_ranks = compute_pagerank(G, personalization=personalization)
        else:
            computed_ranks = base_ranks

        ranks = personalise_ranks(
            computed_ranks,
            focus_files=focus_files,
            mentioned_symbols=mentioned_symbols,
            query_tokens=query_tokens,
            tags=all_tags,
            query=query,
            match_scores=match_scores,
        )

        def_tags_by_file: dict[str, list[Tag]] = {}
        for tag in all_tags:
            if tag.kind == "def":
                def_tags_by_file.setdefault(tag.rel_path, []).append(tag)

        ranked_files = sorted(ranks.items(), key=lambda x: x[1], reverse=True)

        effective_budget = max_tokens if max_tokens is not None else self.max_tokens
        output = self._fill_to_budget(ranked_files, def_tags_by_file, budget=effective_budget)

        if logger.isEnabledFor(logging.DEBUG):
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.debug(
                "RepoMap: %d files, %d tags, %d tokens, %.1f ms",
                len(all_files), len(all_tags), count_tokens(output), elapsed_ms,
            )
        return output

    def _chunk_and_tokens_for_file(
        self,
        rel_path: str,
        def_tags: list[Tag],
        rank: float,
    ) -> tuple[str, int]:
        rel_path = Path(rel_path).as_posix().replace("\\", "/")
        abs_path = self.repo_path / rel_path
        try:
            st = abs_path.stat()
        except OSError:
            return "", 0

        def_lines = sorted({t.line for t in def_tags})[:10]
        lines_key = f"lines:{','.join(map(str, def_lines))};max_lines=35"

        cached = self.outline_cache.get(rel_path, st.st_mtime_ns, st.st_size, lines_key)
        if cached is not None:
            formatted, tok_cnt = cached
            return f"\n{rel_path}:\n{formatted}\n", tok_cnt

        try:
            content = abs_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return "", 0

        formatted = render_tree_context(rel_path, content, def_lines, max_lines=35)
        chunk = f"\n{rel_path}:\n{formatted}\n"
        tok_cnt = self._count(chunk)
        self.outline_cache.put(
            rel_path, st.st_mtime_ns, st.st_size, lines_key, formatted, tok_cnt
        )

        return chunk, tok_cnt

    def _chunk_for_file(
        self,
        rel_path: str,
        def_tags: list[Tag],
        rank: float,
    ) -> str:
        chunk, _ = self._chunk_and_tokens_for_file(rel_path, def_tags, rank)
        return chunk

    def _count(self, text: str) -> int:
        if self._enc is None:
            self._enc = _get_enc(self.model_name)
        return len(self._enc.encode(text, disallowed_special=()))

    def _fill_to_budget(
        self,
        ranked_files: list[tuple[str, float]],
        def_tags_by_file: dict[str, list[Tag]],
        budget: int | None = None,
    ) -> str:
        effective_budget = budget if budget is not None else self.max_tokens

        candidate_chunks: list[tuple[str, int]] = []
        accumulated = 0
        max_render_budget = int(effective_budget * 1.5)

        for rel_path, rank in ranked_files:
            def_tags = def_tags_by_file.get(rel_path)
            if not def_tags:
                continue

            chunk, chunk_toks = self._chunk_and_tokens_for_file(rel_path, def_tags, rank)
            if not chunk.strip():
                continue

            candidate_chunks.append((chunk, chunk_toks))
            accumulated += chunk_toks
            if accumulated >= max_render_budget:
                break

        chunks: list[str] = []
        total = 0

        for chunk, chunk_toks in candidate_chunks:
            if total + chunk_toks > effective_budget:
                if not chunks:
                    chunk = self._truncate_to_budget(chunk, effective_budget)
                    chunks.append(chunk)
                break
            chunks.append(chunk)
            total += chunk_toks

        return "".join(chunks)

    def _count_slice_tokens(self, text: str) -> int:
        if self._enc is not None:
            return len(self._enc.encode(text, disallowed_special=()))
        return max(1, int(len(text) / 3.4))

    def _truncate_to_budget(self, text: str, budget: int, count_fn: Any = None) -> str:
        fn = count_fn if count_fn is not None else self._count
        lines = text.splitlines(keepends=True)
        lo, hi = 0, len(lines)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            candidate = "".join(lines[:mid])
            if fn(candidate) <= budget:
                lo = mid
            else:
                hi = mid - 1
        return "".join(lines[:lo])

    def raw_token_count(self) -> int:
        """Count tokens across all source files, reading from SQLite cache when warm."""
        if self._cached_raw_tokens is not None:
            return self._cached_raw_tokens

        all_files = scan_repo_files(self.repo_path)
        total = 0
        for fpath in all_files:
            cached = self._cache.get(fpath)
            if cached is not None:
                _, tok_cnt = cached
                total += tok_cnt
            else:
                try:
                    text = fpath.read_text(encoding="utf-8", errors="replace")
                    tok_cnt = self._count(text)
                    try:
                        rel = fpath.relative_to(self.repo_path).as_posix()
                    except ValueError:
                        rel = fpath.as_posix()
                    tags = extract_tags(fpath, rel, content=text)
                    self._cache.put(fpath, tags, token_count=tok_cnt)
                    total += tok_cnt
                except Exception:
                    pass

        self._cached_raw_tokens = total
        return total

    def scan_symbols(self) -> list[SymbolDef]:
        all_files = scan_repo_files(self.repo_path)
        all_symbols: list[SymbolDef] = []
        for fpath in all_files:
            try:
                rel = fpath.relative_to(self.repo_path).as_posix()
            except ValueError:
                rel = fpath.as_posix()
            try:
                st = fpath.stat()
                mtime_ns = st.st_mtime_ns
                size_b = st.st_size
            except OSError:
                continue

            cached = self.symbol_cache.get(rel, mtime_ns, size_b)
            if cached is not None:
                _, _, sym_dicts = cached
                for d in sym_dicts:
                    all_symbols.append(SymbolDef.from_dict(d, body=""))
            else:
                try:
                    content = fpath.read_text(encoding="utf-8", errors="replace")
                    tok_cnt = self._count(content)
                except OSError:
                    content = ""
                    tok_cnt = 0
                imports, file_symbols = extract_symbols_and_imports(fpath, rel, content=content)
                sym_dicts = [s.to_dict() for s in file_symbols]
                self.symbol_cache.put(rel, mtime_ns, size_b, tok_cnt, imports, sym_dicts)
                all_symbols.extend(file_symbols)
        return all_symbols

    def slice_symbol(
        self,
        symbol: str,
        token_budget: int = 800,
        include_callers: bool = True,
        include_callees: bool = True,
    ) -> SymbolSliceResult:
        all_files = scan_repo_files(self.repo_path)
        raw_tokens = 0
        all_symbols: list[SymbolDef] = []
        file_imports_map: dict[str, list[str]] = {}
        symbol_def_files: dict[str, set[str]] = {}
        cached_batch = self.symbol_cache.get_all()

        for fpath in all_files:
            try:
                rel = fpath.relative_to(self.repo_path).as_posix()
            except ValueError:
                rel = fpath.as_posix()
            try:
                st = fpath.stat()
                mtime_ns = st.st_mtime_ns
                size_b = st.st_size
            except OSError:
                continue

            key = (rel, mtime_ns, size_b)
            cached = cached_batch.get(key)
            if cached is None:
                cached = self.symbol_cache.get(rel, mtime_ns, size_b)

            if cached is not None:
                tok_cnt, file_imports, sym_dicts = cached
                raw_tokens += tok_cnt
                file_imports_map[rel] = file_imports
                for d in sym_dicts:
                    s_def = SymbolDef.from_dict(d, body="")
                    all_symbols.append(s_def)
                    symbol_def_files.setdefault(s_def.name, set()).add(rel)
            else:
                try:
                    content = fpath.read_text(encoding="utf-8", errors="replace")
                    tok_cnt = self._count_slice_tokens(content)
                except OSError:
                    content = ""
                    tok_cnt = 0
                raw_tokens += tok_cnt
                file_imports, file_symbols = extract_symbols_and_imports(fpath, rel, content=content)
                file_imports_map[rel] = file_imports
                sym_dicts = [s.to_dict() for s in file_symbols]
                self.symbol_cache.put(rel, mtime_ns, size_b, tok_cnt, file_imports, sym_dicts)
                for s_def in file_symbols:
                    all_symbols.append(s_def)
                    symbol_def_files.setdefault(s_def.name, set()).add(rel)

        if self._cached_raw_tokens is None:
            self._cached_raw_tokens = raw_tokens

        target: SymbolDef | None = None
        for s in all_symbols:
            if s.qualified_name == symbol:
                target = s
                break

        if target is None:
            for s in all_symbols:
                if s.name == symbol:
                    target = s
                    break

        if target is None:
            sym_lower = symbol.lower()
            for s in all_symbols:
                if s.name.lower() == sym_lower or s.qualified_name.lower() == sym_lower:
                    target = s
                    break

        if target is None:
            fallback_content = self.get_repo_map(query=symbol, max_tokens=token_budget)
            return SymbolSliceResult(
                symbol=symbol,
                found=False,
                target_file=None,
                callee_count=0,
                caller_count=0,
                sliced_tokens=self._count_slice_tokens(fallback_content),
                raw_tokens=raw_tokens,
                content=fallback_content,
                fallback_used=True,
            )

        target_abs = self.repo_path / target.file_path
        try:
            target_content = target_abs.read_text(encoding="utf-8", errors="replace")
            lines = target_content.splitlines()
            body_lines = lines[target.start_line - 1 : target.end_line]
            target_body = "\n".join(body_lines).strip()
        except Exception:
            target_body = ""

        callees: list[SymbolDef] = []
        if include_callees:
            imported_set = set(file_imports_map.get(target.file_path, []))
            bucket1: list[SymbolDef] = []
            bucket2: list[SymbolDef] = []
            bucket3: list[SymbolDef] = []

            seen_callee_keys: set[tuple[str, str]] = set()
            seen_callee_sigs: set[tuple[str, str]] = set()

            for s in all_symbols:
                if s.file_path == target.file_path and s.qualified_name == target.qualified_name:
                    continue
                if s.name not in target.references and s.qualified_name not in target.references:
                    continue
                if (s.file_path, s.qualified_name) in seen_callee_keys:
                    continue
                if (s.file_path, s.signature) in seen_callee_sigs:
                    continue

                is_generic = s.name in GENERIC_CALL_NAMES and s.name != symbol

                if s.file_path == target.file_path:
                    if not is_generic:
                        seen_callee_keys.add((s.file_path, s.qualified_name))
                        seen_callee_sigs.add((s.file_path, s.signature))
                        bucket1.append(s)
                else:
                    if is_generic:
                        continue
                    file_stem = Path(s.file_path).stem
                    is_imported = (
                        s.name in imported_set
                        or s.qualified_name in imported_set
                        or file_stem in imported_set
                        or any(part in imported_set for part in Path(s.file_path).parts)
                    )
                    if is_imported:
                        seen_callee_keys.add((s.file_path, s.qualified_name))
                        seen_callee_sigs.add((s.file_path, s.signature))
                        bucket2.append(s)
                    elif len(symbol_def_files.get(s.name, set())) == 1:
                        seen_callee_keys.add((s.file_path, s.qualified_name))
                        seen_callee_sigs.add((s.file_path, s.signature))
                        bucket3.append(s)

            bucket1.sort(
                key=lambda s: 0 if (target.enclosing_class and s.enclosing_class == target.enclosing_class) else 1
            )
            callees = bucket1 + bucket2 + bucket3

        callers: list[tuple[SymbolDef, str]] = []
        if include_callers:
            candidate_callers: list[SymbolDef] = []
            seen_caller_keys: set[tuple[str, str]] = set()

            for s in all_symbols:
                if s.file_path == target.file_path and s.qualified_name == target.qualified_name:
                    continue
                if s.file_path == target.file_path and (s.name == target.enclosing_class or s.kind == "class"):
                    continue
                if target.name not in s.references and target.qualified_name not in s.references:
                    continue
                k = (s.file_path, s.qualified_name)
                if k in seen_caller_keys:
                    continue
                seen_caller_keys.add(k)
                candidate_callers.append(s)

            file_lines_cache: dict[str, list[str]] = {}
            for s in candidate_callers:
                if s.file_path not in file_lines_cache:
                    try:
                        cf_text = (self.repo_path / s.file_path).read_text(encoding="utf-8", errors="replace")
                        file_lines_cache[s.file_path] = cf_text.splitlines()
                    except OSError:
                        file_lines_cache[s.file_path] = []
                lines = file_lines_cache[s.file_path]
                snippet = ""
                start_idx = max(0, s.start_line - 1)
                end_idx = min(len(lines), s.end_line)
                for idx in range(start_idx, end_idx):
                    line_str = lines[idx]
                    if target.name in line_str:
                        clean_snippet = re.sub(r"\s+", " ", line_str).strip()
                        if len(clean_snippet) > 120:
                            clean_snippet = clean_snippet[:117] + "..."
                        snippet = clean_snippet
                        break
                callers.append((s, snippet))

        target_budget = max(50, int(token_budget * 0.55))
        callee_budget = int(token_budget * 0.25)
        caller_budget = max(0, token_budget - target_budget - callee_budget)

        target_header = f"# Symbol: {target.qualified_name} ({target.file_path}:{target.start_line}-{target.end_line})"
        full_target_block = f"{target_header}\n{target_body}"
        if self._count_slice_tokens(full_target_block) > target_budget:
            avail_body = max(20, target_budget - self._count_slice_tokens(target_header) - 2)
            target_body = self._truncate_to_budget(target_body, avail_body, count_fn=self._count_slice_tokens)
            full_target_block = f"{target_header}\n{target_body}"

        used_target_toks = self._count_slice_tokens(full_target_block)
        effective_callee_budget = callee_budget + max(0, target_budget - used_target_toks)

        callee_block = ""
        if include_callees and callees:
            lines = [f"# 1-Hop Callees ({len(callees)}):"]
            for c in callees:
                entry = f"{c.file_path}:\n  {c.signature}"
                test_str = "\n".join(lines + [entry])
                if self._count_slice_tokens(test_str) > effective_callee_budget:
                    break
                lines.append(entry)
            if len(lines) > 1:
                callee_block = "\n".join(lines)

        used_callee_toks = self._count_slice_tokens(callee_block) if callee_block else 0
        effective_caller_budget = caller_budget + max(0, effective_callee_budget - used_callee_toks)

        caller_block = ""
        if include_callers and callers:
            lines = [f"# 1-Hop Callers ({len(callers)}):"]
            for c, snippet in callers:
                if snippet:
                    entry = f"{c.file_path}:\n  {c.signature}\n    {snippet}"
                else:
                    entry = f"{c.file_path}:\n  {c.signature}"
                test_str = "\n".join(lines + [entry])
                if self._count_slice_tokens(test_str) > effective_caller_budget:
                    break
                lines.append(entry)
            if len(lines) > 1:
                caller_block = "\n".join(lines)

        sections = [full_target_block]
        if callee_block:
            sections.append(callee_block)
        if caller_block:
            sections.append(caller_block)

        final_content = "\n\n".join(sections).strip()
        sliced_tokens = self._count_slice_tokens(final_content)

        return SymbolSliceResult(
            symbol=symbol,
            found=True,
            target_file=target.file_path,
            callee_count=len(callees),
            caller_count=len(callers),
            sliced_tokens=sliced_tokens,
            raw_tokens=raw_tokens,
            content=final_content,
            fallback_used=False,
        )

    def close(self) -> None:
        if hasattr(self, "_cache") and self._cache is not None:
            self._cache.close()
        if hasattr(self, "outline_cache") and self.outline_cache is not None:
            self.outline_cache.close()
        if hasattr(self, "symbol_cache") and self.symbol_cache is not None:
            self.symbol_cache.close()

    def __enter__(self) -> RepoMap:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
