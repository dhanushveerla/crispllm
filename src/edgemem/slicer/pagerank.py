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

def _split_identifier(ident: str) -> set[str]:
    """Split snake_case and camelCase identifiers into lower-case subwords."""
    parts = re.findall(r"[A-Za-z][a-z0-9]*", ident)
    return {p.lower() for p in parts if len(p) > 1}


def find_matching_files(
    all_files: Iterable[str],
    tags: Iterable[Tag],
    focus_files: list[str] | None,
    mentioned_symbols: list[str] | None,
    query_tokens: list[str] | None,
) -> dict[str, int]:
    """
    Find files matching focus_files, mentioned_symbols, or query_tokens.
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
    target_words = set()
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
        for tw in target_words:
            if tw in path_lower:
                scores[f] += 2

    # Match against defined and referenced symbols
    for tag in tags:
        sym_lower = tag.name.lower()
        subwords = _split_identifier(tag.name)
        weight = 3 if tag.kind == "def" else 1

        if sym_lower in target_words or any(w in target_words for w in subwords):
            scores[tag.rel_path] += weight

    return dict(scores)


def personalise_ranks(
    base_ranks: dict[str, float],
    focus_files: list[str] | None,
    mentioned_symbols: list[str] | None,
    query_tokens: list[str] | None,
    tags: Iterable[Tag],
    boost_factor: float = 5.0,
) -> dict[str, float]:
    """
    Apply focus/query boosts to PageRank scores.
    """
    match_scores = find_matching_files(
        all_files=list(base_ranks.keys()),
        tags=tags,
        focus_files=focus_files,
        mentioned_symbols=mentioned_symbols,
        query_tokens=query_tokens,
    )

    if not match_scores:
        return base_ranks

    boosted: dict[str, float] = {}
    for f, s in base_ranks.items():
        if f in match_scores:
            m = match_scores[f]
            # Multiplicative boost proportional to match strength
            boosted[f] = s * (1.0 + boost_factor * m)
        else:
            boosted[f] = s

    total = sum(boosted.values())
    if total <= 0:
        return base_ranks
    return {f: s / total for f, s in boosted.items()}
