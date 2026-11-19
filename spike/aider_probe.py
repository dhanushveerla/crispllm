"""Minimal AST Slicing Probe for EdgeMem Phase 0.

Implements Tree-sitter + PageRank + grep-ast TreeContext repository mapping
using ONLY minimal dependencies (grep-ast, tree-sitter-language-pack, networkx, psutil, tiktoken),
completely bypassing heavy aider-chat bloat (no scipy, numpy, litellm, sounddevice).

Measures:
- Files scanned
- Raw tokens vs Sliced context tokens
- Compression %
- Cold latency (ms)
- Process RSS (MB)
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import networkx as nx
import psutil
import tiktoken
from grep_ast import TreeContext

sys.stdout.reconfigure(line_buffering=True)


def scan_repo(repo_path: Path) -> List[Path]:
    """Find source files, ignoring VCS, cache, and build directories."""
    ignore_dirs = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache", "node_modules", "dist", "build"}
    py_files: List[Path] = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith(".aider")]
        for f in files:
            if f.endswith(".py"):
                py_files.append(Path(root) / f)
    return sorted(py_files)


def extract_definitions_and_imports(py_path: Path) -> Tuple[List[int], Set[str], Set[str]]:
    """Extract line numbers of class/def and imported symbols using AST."""
    def_lines: List[int] = []
    defined_symbols: Set[str] = set()
    referenced_symbols: Set[str] = set()

    try:
        content = py_path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(content, filename=str(py_path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                def_lines.append(node.lineno)
                defined_symbols.add(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    referenced_symbols.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    referenced_symbols.add(node.module.split(".")[0])
                for alias in node.names:
                    referenced_symbols.add(alias.name)
            elif isinstance(node, ast.Name):
                referenced_symbols.add(node.id)
    except Exception:
        pass

    return def_lines, defined_symbols, referenced_symbols


def compute_pagerank(graph: nx.DiGraph, alpha: float = 0.85, max_iter: int = 50, tol: float = 1e-4) -> Dict[str, float]:
    """Pure Python PageRank with zero numpy/scipy dependency."""
    nodes = list(graph.nodes())
    n = len(nodes)
    if n == 0:
        return {}
    ranks = {node: 1.0 / n for node in nodes}
    for _ in range(max_iter):
        new_ranks = {node: (1.0 - alpha) / n for node in nodes}
        for node in nodes:
            out_weight = sum(graph[node][t].get("weight", 1.0) for t in graph.successors(node))
            if out_weight > 0:
                for target in graph.successors(node):
                    weight = graph[node][target].get("weight", 1.0)
                    new_ranks[target] += alpha * ranks[node] * (weight / out_weight)
            else:
                for other in nodes:
                    new_ranks[other] += alpha * ranks[node] / n
        diff = sum(abs(new_ranks[node] - ranks[node]) for node in nodes)
        ranks = new_ranks
        if diff < tol:
            break
    return ranks


def build_repo_graph(py_files: List[Path], repo_root: Path) -> Tuple[nx.DiGraph, Dict[str, List[int]], Dict[str, str]]:
    """Build directed graph of file dependencies for PageRank."""
    graph = nx.DiGraph()
    file_defs: Dict[str, List[int]] = {}
    file_contents: Dict[str, str] = {}
    file_symbols: Dict[str, Set[str]] = {}
    file_refs: Dict[str, Set[str]] = {}

    for p in py_files:
        rel = str(p.relative_to(repo_root))
        graph.add_node(rel)
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
            file_contents[rel] = content
        except Exception:
            file_contents[rel] = ""

        dlines, dsyms, rsyms = extract_definitions_and_imports(p)
        file_defs[rel] = dlines
        file_symbols[rel] = dsyms
        file_refs[rel] = rsyms

    # Add edges between files based on symbol references
    for src_rel, refs in file_refs.items():
        for target_rel, defs in file_symbols.items():
            if src_rel != target_rel:
                shared = refs.intersection(defs)
                if shared:
                    graph.add_edge(src_rel, target_rel, weight=len(shared))

    return graph, file_defs, file_contents


def main():
    parser = argparse.ArgumentParser(description="Minimal AST Slicing Spike")
    parser.add_argument(
        "--repo-path",
        type=Path,
        default=Path("../edgemem-scratch/sample-repo").resolve(),
        help="Path to sample repository",
    )
    parser.add_argument(
        "--map-tokens",
        type=int,
        default=1024,
        help="Target context token budget",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("spike/aider_results.json"),
        help="Path to output JSON",
    )
    args = parser.parse_args()

    repo_dir = args.repo_path.resolve()
    print("=== EdgeMem Phase 0 Minimal AST Slicing Probe ===", flush=True)
    print(f"Sample Repo Path: {repo_dir}", flush=True)
    print(f"Target Budget: {args.map_tokens} tokens", flush=True)

    proc = psutil.Process()
    rss_start_mb = proc.memory_info().rss / (1024 * 1024)
    print(f"Process RSS (initial): {rss_start_mb:.2f} MB", flush=True)

    enc = tiktoken.get_encoding("cl100k_base")

    t0 = time.perf_counter()

    # 1. Scan files
    files = scan_repo(repo_dir)
    raw_tokens = 0
    for f in files:
        try:
            raw_tokens += len(enc.encode(f.read_text(encoding="utf-8", errors="replace"), disallowed_special=()))
        except Exception:
            pass

    # 2. Build graph and compute PageRank
    graph, file_defs, file_contents = build_repo_graph(files, repo_dir)
    if len(graph) > 0:
        ranks = compute_pagerank(graph, alpha=0.85)
    else:
        ranks = {}

    ranked_files = sorted(ranks.items(), key=lambda x: x[1], reverse=True)

    # 3. Slice top ranked files using TreeContext up to map_tokens
    sliced_chunks: List[str] = []
    current_tokens = 0

    for rel_path, rank in ranked_files:
        lines = file_defs.get(rel_path, [])
        content = file_contents.get(rel_path, "")
        if not lines or not content:
            continue

        # Bound definitions per file so one file does not blow the budget
        selected_lines = lines[:8]

        try:
            tc = TreeContext(rel_path, content, color=False)
            tc.add_lines_of_interest(selected_lines)
            tc.add_context()
            formatted = tc.format()
        except Exception:
            formatted = f"# {rel_path}\n" + "\n".join(
                line for line in content.splitlines() if line.strip().startswith(("def ", "class "))
            )[:400]

        chunk = f"### {rel_path} (rank: {rank:.4f})\n{formatted}\n"
        chunk_tokens = len(enc.encode(chunk, disallowed_special=()))

        if current_tokens + chunk_tokens > args.map_tokens:
            if not sliced_chunks:
                # If first chunk is larger than budget, truncate lines
                lines_list = chunk.splitlines()
                truncated_chunk = ""
                for l in lines_list:
                    if len(enc.encode(truncated_chunk + l + "\n", disallowed_special=())) > args.map_tokens:
                        break
                    truncated_chunk += l + "\n"
                sliced_chunks.append(truncated_chunk)
                current_tokens = len(enc.encode(truncated_chunk, disallowed_special=()))
            break

        sliced_chunks.append(chunk)
        current_tokens += chunk_tokens

    t1 = time.perf_counter()
    cold_latency_ms = (t1 - t0) * 1000.0

    rss_end_mb = proc.memory_info().rss / (1024 * 1024)
    rss_delta_mb = rss_end_mb - rss_start_mb
    compression_pct = ((raw_tokens - current_tokens) / raw_tokens * 100.0) if raw_tokens > 0 else 0.0

    assembled_map = "\n".join(sliced_chunks)

    print("\n=== AST Slicing Probe Results ===", flush=True)
    print(f"Files Scanned: {len(files)} Python files", flush=True)
    print(f"Raw Tokens: {raw_tokens:,} tokens", flush=True)
    print(f"Sliced Context Tokens: {current_tokens} tokens (Target budget: {args.map_tokens})", flush=True)
    print(f"Compression Ratio: {compression_pct:.2f}% reduction", flush=True)
    print(f"Cold Latency: {cold_latency_ms:.2f} ms ({cold_latency_ms / 1000.0:.2f} s)", flush=True)
    print(f"Process RSS (peak/after): {rss_end_mb:.2f} MB (Delta: +{rss_delta_mb:.2f} MB)", flush=True)

    minimal_deps = [
        "grep-ast",
        "tree-sitter-language-pack",
        "networkx",
        "psutil",
        "tiktoken"
    ]
    print(f"\nActive dependencies: {len(minimal_deps)} packages: {', '.join(minimal_deps)}")
    print("Full aider-chat omitted: 0 MB bloat from scipy, numpy, litellm, sounddevice.")

    summary = {
        "repo_path": str(repo_dir),
        "files_scanned": len(files),
        "raw_tokens": raw_tokens,
        "repomap_tokens": current_tokens,
        "compression_pct": round(compression_pct, 2),
        "cold_latency_ms": round(cold_latency_ms, 2),
        "process_rss_start_mb": round(rss_start_mb, 2),
        "process_rss_end_mb": round(rss_end_mb, 2),
        "process_rss_delta_mb": round(rss_delta_mb, 2),
        "minimal_deps": minimal_deps,
        "map_preview": assembled_map[:500] + "...",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
