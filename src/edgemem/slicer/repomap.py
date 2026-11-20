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

import logging
import os
import re
import time
from pathlib import Path
from typing import Sequence

import tiktoken
from grep_ast import TreeContext

from edgemem.slicer.cache import TagCache
from edgemem.slicer.pagerank import (
    build_symbol_graph,
    compute_pagerank,
    find_matching_files,
    is_test_file,
    personalise_ranks,
)
from edgemem.slicer.tags import Tag, extract_tags, scan_repo_files

logger = logging.getLogger("edgemem.slicer.repomap")

# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

_ENCODINGS: dict[str, tiktoken.Encoding] = {}


def _get_enc(model_name: str = "cl100k_base") -> tiktoken.Encoding:
    if model_name not in _ENCODINGS:
        if "TIKTOKEN_CACHE_DIR" not in os.environ:
            from platformdirs import user_cache_dir
            for cand in [
                Path(user_cache_dir("edgemem")) / "tiktoken",
                Path.home() / ".cache" / "tiktoken",
                Path("/tmp/data-gym-cache"),
            ]:
                if cand.exists():
                    os.environ["TIKTOKEN_CACHE_DIR"] = str(cand)
                    break
        _ENCODINGS[model_name] = tiktoken.get_encoding(model_name)
    return _ENCODINGS[model_name]


def count_tokens(text: str, model_name: str = "cl100k_base") -> int:
    enc = _get_enc(model_name)
    return len(enc.encode(text, disallowed_special=()))


# ---------------------------------------------------------------------------
# Query → identifier token extraction
# ---------------------------------------------------------------------------

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}")


def extract_query_tokens(query: str) -> list[str]:
    """Extract Python-identifier-like tokens from a natural language query."""
    return _IDENT_RE.findall(query)


# ---------------------------------------------------------------------------
# Single-file outline via grep-ast TreeContext
# ---------------------------------------------------------------------------

def file_outline(abs_path: Path, rel_path: str | None = None) -> str:
    """
    Return an AST skeleton (signatures only) for a single file using grep-ast.
    Falls back to regex extraction on parse errors.
    """
    rp = rel_path or str(abs_path)
    try:
        content = abs_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return f"# Could not read {rp}\n"

    try:
        tc = TreeContext(
            filename=rp,
            code=content,
            color=False,
            verbose=False,
            line_number=False,
            child_context=False,
            last_line=False,
            margin=0,
            show_top_of_file_parent_scope=False,
        )
        # Show all definition lines
        def_lines = []
        for i, line in enumerate(content.splitlines()):
            stripped = line.strip()
            if stripped.startswith(("def ", "async def ", "class ", "fn ", "func ",
                                    "function ", "pub fn", "pub async", "type ",
                                    "interface ", "impl ", "struct ", "enum ")):
                def_lines.append(i)
        if def_lines:
            tc.add_lines_of_interest(def_lines)
            tc.add_context()
            return tc.format()
        # Empty file or no defs — return first 20 lines
        return "\n".join(content.splitlines()[:20])
    except Exception:
        # Regex fallback
        lines = []
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith(("def ", "async def ", "class ")):
                lines.append(line.rstrip())
        return "\n".join(lines[:40]) if lines else f"# {rp}: (no definitions found)\n"


# ---------------------------------------------------------------------------
# Main RepoMap class
# ---------------------------------------------------------------------------

class RepoMap:
    """
    Builds a token-budgeted repo map using PageRank over the symbol graph.

    Parameters
    ----------
    repo_path:   Root of the repository to map.
    max_tokens:  Hard upper bound on output tokens (tiktoken cl100k_base).
    model_name:  tiktoken encoding name (default: cl100k_base).
    cache:       Optional TagCache instance.  Creates a default one if None.
    """

    def __init__(
        self,
        repo_path: str | Path,
        max_tokens: int = 1024,
        model_name: str = "cl100k_base",
        cache: TagCache | None = None,
    ) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.max_tokens = max_tokens
        self.model_name = model_name
        self._cache = cache or TagCache()
        self._enc = _get_enc(model_name)
        self._cached_raw_tokens: int | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_repo_map(
        self,
        focus_files: list[str] | None = None,
        mentioned_symbols: list[str] | None = None,
        query: str = "",
        max_tokens: int | None = None,
    ) -> str:
        """
        Return a compact repo map string within `max_tokens` (or `self.max_tokens`).

        Parameters
        ----------
        focus_files:        Relative paths to files that should rank highest.
        mentioned_symbols:  Symbol names that should boost their defining files.
        query:              Natural language query; identifier tokens extracted
                            and used to boost matching definitions.
        max_tokens:         Optional override for the token budget.
        """
        t0 = time.perf_counter()

        # 1. Scan files
        all_files = scan_repo_files(self.repo_path)
        if not all_files:
            return ""

        # 2. Extract tags (cached) & accumulate raw token counts
        all_tags: list[Tag] = []
        raw_tok_sum = 0

        for fpath in all_files:
            try:
                rel = str(fpath.relative_to(self.repo_path))
            except ValueError:
                rel = str(fpath)

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

        if self._cached_raw_tokens is None:
            self._cached_raw_tokens = raw_tok_sum

        if not all_tags:
            return ""

        # 3. Build graph + PageRank
        G = build_symbol_graph(all_tags)
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

        base_ranks = compute_pagerank(G, personalization=personalization)

        # 4. Post-hoc boost via personalise_ranks (multiplicative)
        ranks = personalise_ranks(
            base_ranks,
            focus_files=focus_files,
            mentioned_symbols=mentioned_symbols,
            query_tokens=query_tokens,
            tags=all_tags,
            query=query,
        )

        # 5. Build ranked file list; collect def tags per file
        def_tags_by_file: dict[str, list[Tag]] = {}
        for tag in all_tags:
            if tag.kind == "def":
                def_tags_by_file.setdefault(tag.rel_path, []).append(tag)

        ranked_files = sorted(ranks.items(), key=lambda x: x[1], reverse=True)

        # 6. Binary-search + greedy fill within max_tokens
        effective_budget = max_tokens if max_tokens is not None else self.max_tokens
        output = self._fill_to_budget(ranked_files, def_tags_by_file, budget=effective_budget)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "RepoMap: %d files, %d tags, %d tokens, %.1f ms",
            len(all_files), len(all_tags), count_tokens(output), elapsed_ms,
        )
        return output

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chunk_for_file(
        self,
        rel_path: str,
        def_tags: list[Tag],
        rank: float,
    ) -> str:
        """Build a dense grep-ast formatted chunk for one file."""
        abs_path = self.repo_path / rel_path
        try:
            content = abs_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

        def_lines = sorted({t.line for t in def_tags})
        # Cap lines per file to avoid one large file consuming the whole budget
        def_lines = def_lines[:10]

        try:
            tc = TreeContext(
                filename=rel_path,
                code=content,
                color=False,
                verbose=False,
                line_number=False,
                child_context=False,
                last_line=False,
                margin=0,
                show_top_of_file_parent_scope=False,
            )
            tc.add_lines_of_interest(def_lines)
            tc.add_context()
            formatted = tc.format()
            flines = formatted.splitlines()
            if len(flines) > 35:
                formatted = "\n".join(flines[:35])
        except Exception:
            # Regex fallback
            lines = content.splitlines()
            formatted = "\n".join(
                lines[ln] for ln in def_lines if ln < len(lines)
            )

        return f"\n{rel_path}:\n{formatted}\n"

    def _count(self, text: str) -> int:
        return len(self._enc.encode(text, disallowed_special=()))

    def _fill_to_budget(
        self,
        ranked_files: list[tuple[str, float]],
        def_tags_by_file: dict[str, list[Tag]],
        budget: int | None = None,
    ) -> str:
        """
        Greedily fill output chunks up to budget (or self.max_tokens).
        Pre-renders only candidate files from the top of the ranked list,
        avoiding TreeContext on files that will never fit into the budget.
        """
        effective_budget = budget if budget is not None else self.max_tokens

        candidate_chunks: list[tuple[str, int]] = []
        accumulated = 0
        max_render_budget = int(effective_budget * 1.5)

        for rel_path, rank in ranked_files:
            def_tags = def_tags_by_file.get(rel_path)
            if not def_tags:
                continue

            chunk = self._chunk_for_file(rel_path, def_tags, rank)
            if not chunk.strip():
                continue

            chunk_toks = self._count(chunk)
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
                        rel = str(fpath.relative_to(self.repo_path))
                    except ValueError:
                        rel = str(fpath)
                    tags = extract_tags(fpath, rel, content=text)
                    self._cache.put(fpath, tags, token_count=tok_cnt)
                    total += tok_cnt
                except Exception:
                    pass

        self._cached_raw_tokens = total
        return total
