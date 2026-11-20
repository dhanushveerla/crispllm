"""edgemem.slicer.pagerank — Pure-Python PageRank over the repo symbol graph.

No numpy or scipy. Zero heavy numeric dependencies.

Adapts Aider's RepoMap graph-building approach (Apache-2.0).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import networkx as nx

from edgemem.slicer.tags import Tag


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_symbol_graph(tags: Iterable[Tag]) -> nx.DiGraph:
    """
    Build a directed graph where:
      - Nodes  = symbol names (strings)
      - Edges  = def_file → ref_file (weighted by shared symbol count)

    Strategy (adapted from Aider RepoMap):
      1. Map each defined symbol → the file(s) that define it.
      2. For every reference in file F to symbol S, add an edge
         from each file that defines S to F, weighted by co-occurrence count.

    This means well-connected files (many symbols referenced by many others)
    accumulate higher PageRank and appear first in the sliced context.
    """
    # symbol_name → set of rel_paths that define it
    defines: dict[str, set[str]] = defaultdict(set)
    # rel_path → set of symbol names referenced
    references: dict[str, set[str]] = defaultdict(set)
    # collect all files
    all_files: set[str] = set()

    for tag in tags:
        all_files.add(tag.rel_path)
        if tag.kind == "def":
            defines[tag.name].add(tag.rel_path)
        elif tag.kind == "ref":
            references[tag.rel_path].add(tag.name)

    G = nx.DiGraph()
    for f in all_files:
        G.add_node(f)

    for src_file, refs in references.items():
        for sym in refs:
            for def_file in defines.get(sym, set()):
                if def_file == src_file:
                    continue
                if G.has_edge(src_file, def_file):
                    G[src_file][def_file]["weight"] += 1
                else:
                    G.add_edge(src_file, def_file, weight=1)

    return G


def compute_pagerank(
    G: nx.DiGraph,
    personalization: dict[str, float] | None = None,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1.0e-6,
) -> dict[str, float]:
    """
    Pure-Python weighted PageRank.  No numpy/scipy.

    Args:
        G:               Directed graph (nodes = file rel_paths).
        personalization: Optional node → weight map (unnormalised).
                         Nodes not in map get weight 0 (sink teleportation
                         goes uniformly to personalization nodes).
        alpha:           Damping factor (default 0.85, Aider default).
        max_iter:        Max power-iteration steps.
        tol:             Convergence L1 tolerance.

    Returns:
        dict mapping node → PageRank score.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return {}
    if n == 1:
        return {nodes[0]: 1.0}

    # Build normalised personalisation vector
    if personalization:
        total_p = sum(personalization.values())
        if total_p <= 0:
            personalization = None

    if personalization:
        total_p = sum(personalization.values())
        p_vec = {node: personalization.get(node, 0.0) / total_p for node in nodes}
    else:
        p_vec = {node: 1.0 / n for node in nodes}

    # Initialise rank vector
    ranks: dict[str, float] = {node: 1.0 / n for node in nodes}

    # Pre-compute out-weight sums
    out_w: dict[str, float] = {}
    for node in nodes:
        out_w[node] = sum(
            G[node][t].get("weight", 1.0) for t in G.successors(node)
        )

    # Power iteration
    for _ in range(max_iter):
        new_ranks: dict[str, float] = {}
        dangling_sum = sum(ranks[node] for node in nodes if out_w[node] == 0.0)

        for node in nodes:
            # Teleportation + dangling redistribution
            rank_val = (1.0 - alpha) * p_vec[node] + alpha * dangling_sum * p_vec[node]
            # In-link contributions
            for src in G.predecessors(node):
                w = G[src][node].get("weight", 1.0)
                rank_val += alpha * ranks[src] * (w / out_w[src])
            new_ranks[node] = rank_val

        # Check convergence
        diff = sum(abs(new_ranks[node] - ranks[node]) for node in nodes)
        ranks = new_ranks
        if diff < n * tol:
            break

    return ranks


import re


def is_test_file(rel_path: str) -> bool:
    """Check if a file path is a test file or test suite."""
    p = rel_path.replace("\\", "/").lower()
    parts = p.split("/")
    filename = parts[-1]
    for part in parts[:-1]:
        if part in ("tests", "test", "__tests__"):
            return True
    if filename.startswith("test_") or filename.startswith("test."):
        return True
    name_no_ext = filename.rsplit(".", 1)[0]
    if name_no_ext.endswith("_test") or name_no_ext.endswith(".spec") or name_no_ext.endswith(".test"):
        return True
    if ".spec." in filename or ".test." in filename:
        return True
    return False


def _split_identifier(ident: str) -> set[str]:
    """Split snake_case and camelCase identifiers into lower-case subwords with simple stems."""
    parts = re.findall(r"[A-Za-z][a-z0-9]*", ident)
    sub = set()
    for p in parts:
        p_low = p.lower()
        if len(p_low) > 1:
            sub.add(p_low)
            if p_low.endswith("s") and len(p_low) > 3:
                sub.add(p_low[:-1])
            if p_low.endswith("ing") and len(p_low) > 4:
                sub.add(p_low[:-3])
    return sub


def find_matching_files(
    all_files: Iterable[str],
    tags: Iterable[Tag],
    focus_files: list[str] | None,
    mentioned_symbols: list[str] | None,
    query_tokens: list[str] | None,
    query: str = "",
) -> dict[str, int]:
    """
    Find files matching focus_files, mentioned_symbols, or query terms.
    Returns mapping of rel_path -> match_count.
    """
    scores: dict[str, int] = defaultdict(int)

    # 1. Direct focus_files
    if focus_files:
        for f in focus_files:
            clean_f = f.replace("\\", "/").strip()
            for af in all_files:
                if af == clean_f or af.endswith("/" + clean_f) or clean_f in af:
                    scores[af] += 5

    # 2. Query tokens / mentioned symbols matching
    target_words: set[str] = set()
    if query:
        raw_words = re.findall(r"[A-Za-z0-9_]+", query)
        for w in raw_words:
            w_low = w.lower()
            if len(w_low) > 1:
                target_words.add(w_low)
                target_words.update(_split_identifier(w))
    if mentioned_symbols:
        for sym in mentioned_symbols:
            target_words.add(sym.lower())
            target_words.update(_split_identifier(sym))
    if query_tokens:
        for tok in query_tokens:
            if len(tok) > 1:
                target_words.add(tok.lower())
                target_words.update(_split_identifier(tok))

    if not target_words:
        return dict(scores)

    # Match against file paths
    for f in all_files:
        path_lower = f.lower()
        path_segments = re.findall(r"[A-Za-z0-9_]+", path_lower)
        path_subwords: set[str] = set()
        for seg in path_segments:
            path_subwords.add(seg)
            path_subwords.update(_split_identifier(seg))

        for tw in target_words:
            if len(tw) >= 3 and (tw in path_lower or any(tw in psw or psw in tw for psw in path_subwords)):
                scores[f] += 2

    # Match against defined and referenced symbols
    for tag in tags:
        sym_lower = tag.name.lower()
        subwords = _split_identifier(tag.name)
        weight = 3 if tag.kind == "def" else 1

        matched = False
        if sym_lower in target_words:
            matched = True
        elif any(w in target_words for w in subwords):
            matched = True
        elif any(tw in sym_lower for tw in target_words if len(tw) >= 3):
            matched = True
        elif any(any(tw in sw or sw in tw for sw in subwords) for tw in target_words if len(tw) >= 3):
            matched = True

        if matched:
            scores[tag.rel_path] += weight

    return dict(scores)


def personalise_ranks(
    base_ranks: dict[str, float],
    focus_files: list[str] | None,
    mentioned_symbols: list[str] | None,
    query_tokens: list[str] | None,
    tags: Iterable[Tag],
    boost_factor: float = 5.0,
    query: str = "",
) -> dict[str, float]:
    """
    Apply focus/query boosts and test-file penalties to PageRank scores.
    """
    match_scores = find_matching_files(
        all_files=list(base_ranks.keys()),
        tags=tags,
        focus_files=focus_files,
        mentioned_symbols=mentioned_symbols,
        query_tokens=query_tokens,
        query=query,
    )

    query_mentions_test = bool(query and "test" in query.lower())
    focus_has_test = bool(focus_files and any(is_test_file(f) for f in focus_files))
    penalize_tests = not (query_mentions_test or focus_has_test)

    boosted: dict[str, float] = {}
    for f, s in base_ranks.items():
        val = s
        if f in match_scores:
            m = match_scores[f]
            val = val * (1.0 + boost_factor * m)

        # Down-weight test files unless query mentions "test" or focus includes test file
        if penalize_tests and is_test_file(f):
            val = val * 0.15

        boosted[f] = val

    total = sum(boosted.values())
    if total <= 0:
        return base_ranks
    return {f: s / total for f, s in boosted.items()}
