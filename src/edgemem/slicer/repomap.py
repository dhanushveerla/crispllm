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

from edgemem.slicer.cache import OutlineCache, TagCache
from edgemem.slicer.pagerank import (
    build_symbol_graph,
    compute_pagerank,
    find_matching_files,
    is_test_file,
    personalise_ranks,
)
from edgemem.slicer.tags import Tag, extract_tags, scan_repo_files

logger = logging.getLogger("edgemem.slicer.repomap")

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
                Path(user_cache_dir("edgemem")) / "tiktoken",
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

    def _truncate_to_budget(self, text: str, budget: int) -> str:
        """Truncate text to exactly `budget` tokens using binary search."""
        lines = text.splitlines(keepends=True)
        lo, hi = 0, len(lines)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            candidate = "".join(lines[:mid])
            if self._count(candidate) <= budget:
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

    def close(self) -> None:
        if hasattr(self, "_cache") and self._cache is not None:
            self._cache.close()
        if hasattr(self, "outline_cache") and self.outline_cache is not None:
            self.outline_cache.close()

    def __enter__(self) -> RepoMap:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
