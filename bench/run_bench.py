#!/usr/bin/env python3
"""
EdgeMem Universal Benchmark Runner
Talks to any OpenAI-compatible /v1/chat/completions endpoint.

Usage:
  python bench/run_bench.py \
      --base-url http://127.0.0.1:11434/v1 \
      --model qwen2.5-coder:3b \
      --label raw-baseline \
      --category exact_fact
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import textwrap
from pathlib import Path
from statistics import mean, quantiles
from typing import Any

import urllib.request
import urllib.error


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="EdgeMem benchmark runner — OpenAI-compatible endpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              # Run against local Ollama
              python bench/run_bench.py --base-url http://localhost:11434/v1 \\
                  --model qwen2.5-coder:3b --label raw-baseline

              # Run only continuity tasks with EdgeMem proxy
              python bench/run_bench.py --base-url http://localhost:8080/v1 \\
                  --model qwen2.5-coder:3b --label edgemem-v1 --category continuity

              # With health-url for RSS tracking
              python bench/run_bench.py --base-url http://localhost:8080/v1 \\
                  --model any-model --label test --health-url http://localhost:8080/health
        """),
    )
    p.add_argument("--base-url", required=True,
                   help="OpenAI-compatible API base URL (e.g. http://localhost:11434/v1)")
    p.add_argument("--model", required=True,
                   help="Model name to pass in the request body")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY",
                   help="Env var name containing the API key (default: OPENAI_API_KEY; set to '' to skip)")
    p.add_argument("--extra-body-json", default="{}",
                   help="JSON string merged into every request body (e.g. '{\"num_ctx\": 8192}')")
    p.add_argument("--tasks-file", default=None,
                   help="Path to bench_tasks.json (default: bench/bench_tasks.json relative to repo root)")
    p.add_argument("--limit", type=int, default=None,
                   help="Run only the first N tasks (useful for smoke tests)")
    p.add_argument("--category", default=None, choices=["exact_fact", "continuity"],
                   help="Filter tasks by category")
    p.add_argument("--label", default="run",
                   help="Label used in output filenames (e.g. 'raw-baseline', 'edgemem-v1')")
    p.add_argument("--output-dir", default=None,
                   help="Output directory for results (default: bench/results/ relative to repo root)")
    p.add_argument("--health-url", default=None,
                   help="URL of a /health endpoint that returns JSON with an 'rss_mb' field for memory tracking")
    p.add_argument("--timeout-task", type=int, default=10,
                   help="Subprocess timeout in seconds for each test_code run (default: 10)")
    p.add_argument("--timeout-llm", type=int, default=120,
                   help="HTTP timeout in seconds for each LLM call (default: 120)")
    return p


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only — no httpx/requests dependency at bench time)
# ---------------------------------------------------------------------------

def _http_post_json(url: str, payload: dict, api_key: str, timeout: int) -> dict:
    """POST JSON, return parsed response dict. Raises on HTTP error."""
    body = json.dumps(payload).encode()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}" if api_key else "Bearer none",
    }
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _http_get_json(url: str, timeout: int = 5) -> dict | None:
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def extract_code_blocks(text: str) -> str:
    """Extract the last ```python ... ``` block from model output.
    Falls back to the full text if no code block found."""
    pattern = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1].strip()
    # Try without language tag
    pattern2 = re.compile(r"```\s*\n(.*?)```", re.DOTALL)
    matches2 = pattern2.findall(text)
    if matches2:
        return matches2[-1].strip()
    return text.strip()


def extract_all_code_blocks(text: str) -> dict[str, str]:
    """
    Extract filename-prefixed code blocks.
    Format expected: ```python\n# filename: path/to/file.py\n...\n```
    Returns mapping of filename -> code.
    Falls back to single unnamed block.
    """
    pattern = re.compile(
        r"```(?:python)?\s*\n(?:#\s*(?:filename|file|path):\s*([^\n]+)\n)?(.*?)```",
        re.DOTALL | re.IGNORECASE,
    )
    result: dict[str, str] = {}
    for m in pattern.finditer(text):
        fname = (m.group(1) or "").strip()
        code = m.group(2).strip()
        if fname:
            result[fname] = code
        else:
            result.setdefault("__unnamed__", code)
    return result


def run_task(
    task: dict,
    base_url: str,
    model: str,
    api_key: str,
    extra_body: dict,
    timeout_llm: int,
    timeout_task: int,
) -> dict:
    """
    Execute a single benchmark task.
    Returns a result dict.
    """
    task_id = task["id"]
    category = task["category"]
    files = task.get("files", {})
    turns = task["turns"]
    test_code = task["test_code"]
    reference_solution = task["reference_solution"]

    with tempfile.TemporaryDirectory(prefix=f"edgemem_bench_{task_id}_") as tmpdir:
        tmp = Path(tmpdir)

        # Write all context files into the temp dir
        for rel_path, content in files.items():
            dest = tmp / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content)

        # Build conversation: inject file contents as system context
        file_context = "\n\n".join(
            f"### File: `{p}`\n```python\n{c}\n```" for p, c in files.items()
        )
        system_msg = (
            "You are an expert Python developer. "
            "Below are the files in the project. "
            "Implement or fix code as instructed. "
            "Return your code in ```python ... ``` blocks with a comment "
            "`# filename: <relative/path>` at the top of each block so it "
            "is clear which file each block belongs to.\n\n"
            + file_context
        )

        messages: list[dict] = [{"role": "system", "content": system_msg}]
        for turn in turns:
            messages.append({"role": "user", "content": turn})

        # Call LLM — measure TTFT via streaming if available, else single call
        t0 = time.perf_counter()
        ttft_ms: float | None = None
        prompt_tokens: int = 0
        completion_tokens: int = 0
        final_text: str = ""

        try:
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                **extra_body,
            }
            resp = _http_post_json(
                f"{base_url.rstrip('/')}/chat/completions",
                payload,
                api_key,
                timeout_llm,
            )
            t1 = time.perf_counter()
            ttft_ms = (t1 - t0) * 1000  # approximation for non-streaming
            final_text = resp["choices"][0]["message"]["content"]
            usage = resp.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
        except Exception as exc:
            wall_ms = (time.perf_counter() - t0) * 1000
            return {
                "task_id": task_id,
                "category": category,
                "passed": False,
                "error": f"LLM call failed: {exc}",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "ttft_ms": None,
                "wall_ms": wall_ms,
            }

        wall_ms = (time.perf_counter() - t0) * 1000

        # Extract code blocks from model output
        code_blocks = extract_all_code_blocks(final_text)

        # Write extracted code into temp dir (overwriting skeleton files)
        wrote_any = False
        for fname, code in code_blocks.items():
            if fname == "__unnamed__":
                # Try to guess from reference_solution keys
                ref_keys = list(reference_solution.keys())
                if len(ref_keys) == 1:
                    fname = ref_keys[0]
                else:
                    # Write as main.py fallback
                    fname = "solution.py"
            dest = tmp / fname
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(code)
            wrote_any = True

        if not wrote_any:
            # Dump full response as a Python file for debugging
            (tmp / "solution.py").write_text(f"# Model response (no code block found)\n'''\n{final_text}\n'''")

        # Write test_code into temp dir
        test_file = tmp / "_edgemem_test_.py"
        test_file.write_text(test_code)

        # Run test_code in subprocess with timeout
        passed = False
        test_error = ""
        try:
            proc = subprocess.run(
                [sys.executable, str(test_file), str(tmp)],
                capture_output=True,
                text=True,
                timeout=timeout_task,
                cwd=str(tmp),
            )
            if proc.returncode == 0 and "PASS" in proc.stdout:
                passed = True
            else:
                test_error = (proc.stdout + proc.stderr).strip()[:500]
        except subprocess.TimeoutExpired:
            test_error = f"test timed out after {timeout_task}s"
        except Exception as exc:
            test_error = f"subprocess error: {exc}"

    gen_toks_per_s: float | None = None
    wall_s = wall_ms / 1000
    if completion_tokens and wall_s > 0:
        gen_toks_per_s = completion_tokens / wall_s

    return {
        "task_id": task_id,
        "category": category,
        "passed": passed,
        "error": test_error if not passed else "",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_ms": ttft_ms,
        "wall_ms": wall_ms,
        "gen_toks_per_s": gen_toks_per_s,
    }


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------

def _safe_mean(vals: list[float]) -> float | None:
    vals = [v for v in vals if v is not None]
    return mean(vals) if vals else None


def _safe_p95(vals: list[float]) -> float | None:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    if len(vals) < 2:
        return vals[0]
    return quantiles(vals, n=20)[18]  # 95th percentile


def compute_metrics(results: list[dict]) -> dict:
    total = len(results)
    passed = sum(1 for r in results if r["passed"])

    ef = [r for r in results if r["category"] == "exact_fact"]
    ct = [r for r in results if r["category"] == "continuity"]

    ef_pass = sum(1 for r in ef if r["passed"])
    ct_pass = sum(1 for r in ct if r["passed"])

    prompt_toks = [r["prompt_tokens"] for r in results]
    comp_toks   = [r["completion_tokens"] for r in results]
    ttfts       = [r["ttft_ms"] for r in results if r.get("ttft_ms") is not None]
    walls       = [r["wall_ms"] for r in results]
    gen_tps     = [r["gen_toks_per_s"] for r in results if r.get("gen_toks_per_s") is not None]

    return {
        "pass_at_1": {
            "overall":    round(passed / total, 4) if total else 0.0,
            "exact_fact": round(ef_pass / len(ef), 4) if ef else None,
            "continuity": round(ct_pass / len(ct), 4) if ct else None,
            "counts": {
                "total": total, "passed": passed,
                "exact_fact_total": len(ef), "exact_fact_passed": ef_pass,
                "continuity_total": len(ct), "continuity_passed": ct_pass,
            },
        },
        "tokens": {
            "total_prompt":      sum(prompt_toks),
            "total_completion":  sum(comp_toks),
            "mean_prompt":       round(_safe_mean(prompt_toks) or 0, 1),
            "mean_completion":   round(_safe_mean(comp_toks) or 0, 1),
        },
        "latency": {
            "mean_ttft_ms":    round(_safe_mean(ttfts) or 0, 1) if ttfts else None,
            "p95_ttft_ms":     round(_safe_p95(ttfts) or 0, 1) if ttfts else None,
            "mean_gen_toks_s": round(_safe_mean(gen_tps) or 0, 2) if gen_tps else None,
            "total_wall_s":    round(sum(walls) / 1000, 2),
        },
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def build_markdown_report(label: str, model: str, base_url: str,
                           metrics: dict, results: list[dict],
                           rss_mb: float | None) -> str:
    p = metrics["pass_at_1"]
    t = metrics["tokens"]
    l = metrics["latency"]
    c = p["counts"]

    lines = [
        f"# EdgeMem Benchmark Report: `{label}`",
        "",
        f"| Field | Value |",
        f"|---|---|",
        f"| Label | `{label}` |",
        f"| Model | `{model}` |",
        f"| Endpoint | `{base_url}` |",
        f"| Tasks run | {c['total']} |",
        f"| **Pass@1 (overall)** | **{p['overall']:.1%}** ({c['passed']}/{c['total']}) |",
        f"| Pass@1 exact_fact | {p['exact_fact']:.1%} ({c['exact_fact_passed']}/{c['exact_fact_total']}) |" if p['exact_fact'] is not None else "| Pass@1 exact_fact | — |",
        f"| Pass@1 continuity | {p['continuity']:.1%} ({c['continuity_passed']}/{c['continuity_total']}) |" if p['continuity'] is not None else "| Pass@1 continuity | — |",
        f"| Total prompt tokens | {t['total_prompt']:,} |",
        f"| Total completion tokens | {t['total_completion']:,} |",
        f"| Mean prompt tokens | {t['mean_prompt']} |",
        f"| Mean completion tokens | {t['mean_completion']} |",
        f"| Mean TTFT (ms) | {l['mean_ttft_ms']} |" if l['mean_ttft_ms'] is not None else "| Mean TTFT (ms) | — |",
        f"| p95 TTFT (ms) | {l['p95_ttft_ms']} |" if l['p95_ttft_ms'] is not None else "| p95 TTFT (ms) | — |",
        f"| Mean gen tok/s | {l['mean_gen_toks_s']} |" if l['mean_gen_toks_s'] is not None else "| Mean gen tok/s | — |",
        f"| Total wall time (s) | {l['total_wall_s']} |",
        f"| Proxy RSS (MB) | {rss_mb:.1f} |" if rss_mb is not None else "| Proxy RSS (MB) | — |",
        "",
        "## Per-Task Results",
        "",
        "| Task ID | Category | Pass | Prompt tok | Completion tok | Wall (ms) | Error |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        status = "✅" if r["passed"] else "❌"
        err_snippet = (r.get("error") or "")[:60].replace("\n", " ")
        lines.append(
            f"| {r['task_id']} | {r['category']} | {status} | "
            f"{r['prompt_tokens']} | {r['completion_tokens']} | "
            f"{r['wall_ms']:.0f} | {err_snippet} |"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Resolve paths
    repo_root = Path(__file__).parent.parent
    tasks_file = Path(args.tasks_file) if args.tasks_file else (repo_root / "bench" / "bench_tasks.json")
    output_dir = Path(args.output_dir) if args.output_dir else (repo_root / "bench" / "results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # API key
    api_key = ""
    if args.api_key_env:
        api_key = os.environ.get(args.api_key_env, "")

    # Extra body
    try:
        extra_body = json.loads(args.extra_body_json)
    except json.JSONDecodeError as exc:
        sys.exit(f"--extra-body-json is not valid JSON: {exc}")

    # Load tasks
    with open(tasks_file) as f:
        all_tasks: list[dict] = json.load(f)

    # Filter
    if args.category:
        all_tasks = [t for t in all_tasks if t["category"] == args.category]
    if args.limit:
        all_tasks = all_tasks[: args.limit]

    if not all_tasks:
        sys.exit("No tasks to run after filtering.")

    print(f"[EdgeMem Bench] label={args.label!r}  model={args.model!r}  tasks={len(all_tasks)}")
    print(f"  endpoint: {args.base_url}")
    print()

    # Optional: get RSS before run
    rss_before: float | None = None
    if args.health_url:
        hdata = _http_get_json(args.health_url)
        if hdata:
            rss_before = hdata.get("rss_mb")

    results: list[dict] = []
    for i, task in enumerate(all_tasks, 1):
        print(f"  [{i:2d}/{len(all_tasks)}] {task['id']} ({task['category']}) ...", end=" ", flush=True)
        t0 = time.perf_counter()
        result = run_task(
            task=task,
            base_url=args.base_url,
            model=args.model,
            api_key=api_key,
            extra_body=extra_body,
            timeout_llm=args.timeout_llm,
            timeout_task=args.timeout_task,
        )
        elapsed = time.perf_counter() - t0
        status_icon = "✅" if result["passed"] else "❌"
        print(f"{status_icon}  ({elapsed:.1f}s)")
        if not result["passed"] and result.get("error"):
            print(f"         error: {result['error'][:120]}")
        results.append(result)

    # Optional: get RSS after run
    rss_after: float | None = None
    if args.health_url:
        hdata = _http_get_json(args.health_url)
        if hdata:
            rss_after = hdata.get("rss_mb")
    rss_mb = rss_after  # report post-run RSS

    metrics = compute_metrics(results)

    # Print summary
    p = metrics["pass_at_1"]
    l = metrics["latency"]
    c = p["counts"]
    print()
    print(f"{'='*60}")
    print(f"  Pass@1 (overall):    {p['overall']:.1%}  ({c['passed']}/{c['total']})")
    if p["exact_fact"] is not None:
        print(f"  Pass@1 exact_fact:   {p['exact_fact']:.1%}  ({c['exact_fact_passed']}/{c['exact_fact_total']})")
    if p["continuity"] is not None:
        print(f"  Pass@1 continuity:   {p['continuity']:.1%}  ({c['continuity_passed']}/{c['continuity_total']})")
    print(f"  Total wall time:     {l['total_wall_s']}s")
    if l["mean_ttft_ms"] is not None:
        print(f"  Mean TTFT:           {l['mean_ttft_ms']}ms  (p95: {l['p95_ttft_ms']}ms)")
    if l["mean_gen_toks_s"] is not None:
        print(f"  Mean gen tok/s:      {l['mean_gen_toks_s']}")
    if rss_mb is not None:
        print(f"  Proxy RSS (MB):      {rss_mb:.1f}")
    print(f"{'='*60}")

    # Write JSON results
    output_data = {
        "label": args.label,
        "model": args.model,
        "base_url": args.base_url,
        "metrics": metrics,
        "rss_mb": rss_mb,
        "results": results,
    }
    json_path = output_dir / f"{args.label}_results.json"
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n  JSON: {json_path}")

    # Write Markdown report
    md = build_markdown_report(
        label=args.label,
        model=args.model,
        base_url=args.base_url,
        metrics=metrics,
        results=results,
        rss_mb=rss_mb,
    )
    md_path = output_dir / f"{args.label}_report.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"  Markdown: {md_path}")


if __name__ == "__main__":
    main()
