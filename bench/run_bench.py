#!/usr/bin/env python3
"""EdgeMem Universal Benchmark Runner.
Talks to any OpenAI-compatible /v1/chat/completions endpoint.
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
import textwrap
import time
from pathlib import Path
from statistics import mean, median, quantiles
from typing import Any, Mapping

import urllib.error
import urllib.request


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="EdgeMem benchmark runner - OpenAI-compatible endpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python bench/run_bench.py --base-url http://localhost:11434/v1 \\
                  --model qwen2.5-coder:3b --label raw-baseline

              python bench/run_bench.py --base-url http://localhost:8080/v1 \\
                  --model qwen2.5-coder:3b --label edgemem-v1 --category continuity
        """),
    )
    p.add_argument("--base-url", required=True, help="OpenAI-compatible API base URL")
    p.add_argument("--model", required=True, help="Model name to pass in request body")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Env var name containing API key")
    p.add_argument("--extra-body-json", default="{}", help="JSON string merged into request body")
    p.add_argument("--tasks", "--tasks-file", dest="tasks_file", default=None, help="Path to tasks JSON file")
    p.add_argument("--limit", type=int, default=None, help="Run only the first N tasks")
    p.add_argument("--category", default=None, help="Filter tasks by category")
    p.add_argument("--label", default="run", help="Label used in output filenames")
    p.add_argument("--output-dir", default=None, help="Output directory for results")
    p.add_argument("--health-url", default=None, help="URL of health endpoint returning rss_mb")
    p.add_argument("--timeout-task", type=int, default=10, help="Timeout in seconds for test execution")
    p.add_argument("--timeout-llm", type=int, default=120, help="HTTP timeout in seconds for LLM call")
    p.add_argument("--delay", type=float, default=0.0, metavar="SECONDS", help="Delay between tasks in seconds")
    p.add_argument("--max-retries", type=int, default=4, help="Max retry attempts on HTTP 429/503")
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    p.add_argument("--seed", type=int, default=None, help="Random seed for sampling")
    p.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use SSE streaming to decompose TTFT/prefill vs. decode latency (default: True).",
    )
    p.add_argument(
        "--baseline",
        default=None,
        help="Path to baseline JSON results file to print comparison table against.",
    )
    return p


class _RateLimitError(Exception):
    def __init__(self, status: int, retry_after: float | None = None) -> None:
        self.status = status
        self.retry_after = retry_after


def parse_sse_stream(
    lines: Any,
    headers: Any,
    t_start: float,
    clock: Any = time.perf_counter,
) -> dict[str, Any]:
    hdr_dict: dict[str, str] = {}
    if headers:
        if hasattr(headers, "items"):
            hdr_dict = {str(k).lower(): str(v) for k, v in headers.items()}
        elif isinstance(headers, dict):
            hdr_dict = {str(k).lower(): str(v) for k, v in headers.items()}

    proxy_overhead_ms = 0.0
    hdr_overhead = hdr_dict.get("x-edgemem-overhead-ms")
    if hdr_overhead:
        try:
            proxy_overhead_ms = max(0.0, float(hdr_overhead))
        except ValueError:
            proxy_overhead_ms = 0.0

    t_first_token: float | None = None
    accumulated_text: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0
    usage_source = "none"
    timings: dict[str, Any] = {}

    for raw_line in lines:
        if isinstance(raw_line, bytes):
            line = raw_line.decode("utf-8", errors="replace").strip()
        else:
            line = str(raw_line).strip()

        if not line or not line.startswith("data:"):
            continue

        raw_payload = line[5:].strip()
        if raw_payload == "[DONE]":
            break

        try:
            chunk = json.loads(raw_payload)
        except json.JSONDecodeError:
            continue

        choices = chunk.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                delta = choice.get("delta", {})
                if isinstance(delta, dict):
                    has_content = bool(delta.get("content"))
                    has_reasoning = bool(delta.get("reasoning_content"))
                    has_tool_calls = bool(delta.get("tool_calls"))
                    if has_content or has_reasoning or has_tool_calls:
                        if t_first_token is None:
                            t_first_token = clock()
                        if has_content:
                            accumulated_text.append(delta["content"])

        usage = chunk.get("usage")
        if isinstance(usage, dict):
            pt = usage.get("prompt_tokens")
            ct = usage.get("completion_tokens")
            if pt is not None:
                prompt_tokens = int(pt)
            if ct is not None:
                completion_tokens = int(ct)
                usage_source = "api_usage"

        chunk_timings = chunk.get("timings")
        if isinstance(chunk_timings, dict):
            timings = chunk_timings
            if prompt_tokens == 0 and "prompt_n" in timings:
                prompt_tokens = int(timings["prompt_n"])
            if completion_tokens == 0 and "predicted_n" in timings:
                completion_tokens = int(timings["predicted_n"])
                usage_source = "api_usage"

    t_done = clock()
    wall_ms = max(0.0, (t_done - t_start) * 1000.0)

    if t_first_token is not None:
        ttft_total_ms = max(0.0, (t_first_token - t_start) * 1000.0)
        prefill_ms = max(0.0, ttft_total_ms - proxy_overhead_ms)
        decode_ms = max(0.0, (t_done - t_first_token) * 1000.0)
    else:
        ttft_total_ms = None
        prefill_ms = None
        decode_ms = None

    decode_tok_s = None
    if completion_tokens > 0 and decode_ms is not None and decode_ms > 0:
        decode_tok_s = completion_tokens / (decode_ms / 1000.0)

    return {
        "final_text": "".join(accumulated_text),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "wall_ms": wall_ms,
        "proxy_overhead_ms": proxy_overhead_ms,
        "ttft_total_ms": ttft_total_ms,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "decode_tok_s": decode_tok_s,
        "usage_source": usage_source,
        "timings": timings,
    }


def _http_stream_chat(
    url: str,
    payload: dict,
    api_key: str,
    timeout: int,
    max_retries: int = 4,
) -> dict[str, Any]:
    stream_payload = dict(payload)
    stream_payload["stream"] = True
    stream_payload["stream_options"] = {"include_usage": True}
    tried_without_stream_options = False

    base_delay = 2.0
    attempt = 0
    while attempt <= max_retries:
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        body = json.dumps(stream_payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        t_start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp_headers = {k.lower(): v for k, v in resp.headers.items()}
                return parse_sse_stream(resp, resp_headers, t_start)
        except urllib.error.HTTPError as exc:
            if (
                exc.code in (400, 422)
                and not tried_without_stream_options
                and "stream_options" in stream_payload
            ):
                stream_payload.pop("stream_options", None)
                tried_without_stream_options = True
                continue
            if exc.code in (429, 503):
                retry_after: float | None = None
                ra_header = exc.headers.get("Retry-After") or exc.headers.get("retry-after")
                if ra_header:
                    try:
                        retry_after = float(ra_header)
                    except ValueError:
                        retry_after = None

                if attempt >= max_retries:
                    raise _RateLimitError(exc.code, retry_after) from exc

                wait = retry_after if retry_after is not None else base_delay * (2 ** attempt)
                print(
                    f"    [bench] HTTP {exc.code} - retrying in {wait:.1f}s "
                    f"(attempt {attempt + 1}/{max_retries})",
                    file=sys.stderr,
                )
                time.sleep(wait)
                attempt += 1
                continue
            raise

    raise RuntimeError("Unexpected retry loop exit")


def _http_post_json(
    url: str,
    payload: dict,
    api_key: str,
    timeout: int,
    max_retries: int = 4,
) -> tuple[dict, float, float]:
    body = json.dumps(payload).encode("utf-8")
    headers: dict[str, str] = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    base_delay = 2.0
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        t_start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                t_done = time.perf_counter()
                wall_ms = max(0.0, (t_done - t_start) * 1000.0)
                proxy_overhead_ms = 0.0
                hdr_val = resp.headers.get("x-edgemem-overhead-ms") or resp.headers.get("X-EdgeMem-Overhead-Ms")
                if hdr_val:
                    try:
                        proxy_overhead_ms = max(0.0, float(hdr_val))
                    except ValueError:
                        proxy_overhead_ms = 0.0
                return json.loads(resp.read().decode("utf-8")), wall_ms, proxy_overhead_ms
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 503):
                retry_after: float | None = None
                ra_header = exc.headers.get("Retry-After") or exc.headers.get("retry-after")
                if ra_header:
                    try:
                        retry_after = float(ra_header)
                    except ValueError:
                        retry_after = None

                if attempt >= max_retries:
                    raise _RateLimitError(exc.code, retry_after) from exc

                wait = retry_after if retry_after is not None else base_delay * (2 ** attempt)
                print(
                    f"    [bench] HTTP {exc.code} - retrying in {wait:.1f}s "
                    f"(attempt {attempt + 1}/{max_retries})",
                    file=sys.stderr,
                )
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Unexpected retry loop exit")


def _http_get_json(url: str, timeout: int = 5) -> dict | None:
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def extract_code_blocks(text: str) -> str:
    pattern = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1].strip()
    pattern2 = re.compile(r"```\s*\n(.*?)```", re.DOTALL)
    matches2 = pattern2.findall(text)
    if matches2:
        return matches2[-1].strip()
    return text.strip()


def extract_all_code_blocks(text: str) -> dict[str, str]:
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


def build_chat_payload(
    model: str,
    messages: list[dict],
    temperature: float = 0.0,
    seed: int | None = None,
    extra_body: dict | None = None,
    stream: bool = False,
) -> dict:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "temperature": temperature,
    }
    if extra_body:
        payload.update(extra_body)
    if seed is not None:
        payload["seed"] = seed
    return payload


def run_task(
    task: dict,
    base_url: str,
    model: str,
    api_key: str,
    extra_body: dict,
    timeout_llm: int,
    timeout_task: int,
    max_retries: int = 4,
    temperature: float = 0.0,
    seed: int | None = None,
    stream: bool = True,
) -> dict:
    task_id = task["id"]
    category = task["category"]
    files = task.get("files", {})
    turns = task["turns"]
    test_code = task["test_code"]
    reference_solution = task["reference_solution"]

    with tempfile.TemporaryDirectory(prefix=f"edgemem_bench_{task_id}_") as tmpdir:
        tmp = Path(tmpdir)

        for rel_path, content in files.items():
            dest = tmp / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")

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

        t0 = time.perf_counter()
        ttft_total_ms: float | None = None
        prefill_ms: float | None = None
        decode_ms: float | None = None
        decode_tok_s: float | None = None
        proxy_overhead_ms: float = 0.0
        prompt_tokens: int = 0
        completion_tokens: int = 0
        usage_source: str = "none"
        final_text: str = ""
        wall_ms: float = 0.0

        try:
            payload = build_chat_payload(
                model=model,
                messages=messages,
                temperature=temperature,
                seed=seed,
                extra_body=extra_body,
                stream=stream,
            )
            chat_url = f"{base_url.rstrip('/')}/chat/completions"
            if stream:
                stream_res = _http_stream_chat(
                    chat_url,
                    payload,
                    api_key,
                    timeout_llm,
                    max_retries=max_retries,
                )
                final_text = stream_res["final_text"]
                prompt_tokens = stream_res["prompt_tokens"]
                completion_tokens = stream_res["completion_tokens"]
                wall_ms = stream_res["wall_ms"]
                proxy_overhead_ms = stream_res["proxy_overhead_ms"]
                ttft_total_ms = stream_res["ttft_total_ms"]
                prefill_ms = stream_res["prefill_ms"]
                decode_ms = stream_res["decode_ms"]
                decode_tok_s = stream_res["decode_tok_s"]
                usage_source = stream_res["usage_source"]
            else:
                resp, wall_ms, proxy_overhead_ms = _http_post_json(
                    chat_url,
                    payload,
                    api_key,
                    timeout_llm,
                    max_retries=max_retries,
                )
                final_text = resp["choices"][0]["message"]["content"]
                usage = resp.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                usage_source = "api_usage" if "completion_tokens" in usage else "none"
                ttft_total_ms = None
                prefill_ms = None
                decode_ms = None
                decode_tok_s = None
        except Exception as exc:
            wall_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "task_id": task_id,
                "category": category,
                "passed": False,
                "error": f"LLM call failed: {exc}",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "ttft_ms": None,
                "ttft_total_ms": None,
                "wall_ms": wall_ms,
                "proxy_overhead_ms": proxy_overhead_ms,
                "prefill_ms": None,
                "decode_ms": None,
                "decode_tok_s": None,
                "gen_toks_per_s": None,
                "usage_source": usage_source,
            }

        code_blocks = extract_all_code_blocks(final_text)

        wrote_any = False
        for fname, code in code_blocks.items():
            if fname == "__unnamed__":
                ref_keys = list(reference_solution.keys())
                if len(ref_keys) == 1:
                    fname = ref_keys[0]
                else:
                    fname = "solution.py"
            dest = tmp / fname
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(code, encoding="utf-8")
            wrote_any = True

        if not wrote_any:
            (tmp / "solution.py").write_text(f"'''\n{final_text}\n'''", encoding="utf-8")

        test_file = tmp / "_edgemem_test_.py"
        test_file.write_text(test_code, encoding="utf-8")

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

    return {
        "task_id": task_id,
        "category": category,
        "passed": passed,
        "error": test_error if not passed else "",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_ms": ttft_total_ms,
        "ttft_total_ms": ttft_total_ms,
        "wall_ms": wall_ms,
        "proxy_overhead_ms": proxy_overhead_ms,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "decode_tok_s": decode_tok_s,
        "gen_toks_per_s": decode_tok_s,
        "usage_source": usage_source,
    }


def _safe_mean(vals: list[float]) -> float | None:
    filtered = [v for v in vals if v is not None]
    return mean(filtered) if filtered else None


def _safe_median(vals: list[float]) -> float | None:
    filtered = [v for v in vals if v is not None]
    return median(filtered) if filtered else None


def _safe_p95(vals: list[float]) -> float | None:
    filtered = [v for v in vals if v is not None]
    if not filtered:
        return None
    if len(filtered) < 2:
        return filtered[0]
    return quantiles(filtered, n=20)[18]


def summarize_run_results(results: list[dict]) -> dict[str, Any]:
    sample_count = len(results)
    passed_count = sum(1 for r in results if r.get("passed"))
    pass_at_1_pct = round((passed_count / sample_count) * 100.0, 1) if sample_count > 0 else 0.0

    categories: dict[str, dict[str, Any]] = {}
    cat_names: list[str] = []
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in cat_names:
            cat_names.append(cat)
    for cat in cat_names:
        cat_items = [r for r in results if r.get("category") == cat]
        n_cat = len(cat_items)
        passed_cat = sum(1 for r in cat_items if r.get("passed"))
        pct_cat = round((passed_cat / n_cat) * 100.0, 1) if n_cat > 0 else 0.0
        categories[cat] = {
            "n": n_cat,
            "passed": passed_cat,
            "pass_at_1_pct": pct_cat,
            "formatted": f"{pct_cat:.1f}% ({passed_cat}/{n_cat}, n={n_cat})",
        }

    walls = [float(r["wall_ms"]) for r in results if r.get("wall_ms") is not None]
    overheads = [float(r["proxy_overhead_ms"]) for r in results if r.get("proxy_overhead_ms") is not None]
    ttfts = [float(r["ttft_total_ms"]) for r in results if r.get("ttft_total_ms") is not None]
    prefills = [float(r["prefill_ms"]) for r in results if r.get("prefill_ms") is not None]
    decodes = [float(r["decode_ms"]) for r in results if r.get("decode_ms") is not None]
    decode_tps = [float(r["decode_tok_s"]) for r in results if r.get("decode_tok_s") is not None]

    mean_wall = round(mean(walls), 2) if walls else None
    med_wall = round(median(walls), 2) if walls else None

    mean_overhead = round(mean(overheads), 2) if overheads else None
    med_overhead = round(median(overheads), 2) if overheads else None

    mean_ttft = round(mean(ttfts), 2) if ttfts else None
    med_ttft = round(median(ttfts), 2) if ttfts else None

    mean_prefill = round(mean(prefills), 2) if prefills else None
    med_prefill = round(median(prefills), 2) if prefills else None

    mean_decode = round(mean(decodes), 2) if decodes else None
    med_decode = round(median(decodes), 2) if decodes else None

    mean_tok_s = round(mean(decode_tps), 2) if decode_tps else None
    med_tok_s = round(median(decode_tps), 2) if decode_tps else None

    proxy_pct = round((mean_overhead / mean_wall) * 100.0, 1) if mean_wall and mean_overhead is not None else 0.0
    prefill_pct = round((mean_prefill / mean_wall) * 100.0, 1) if mean_wall and mean_prefill is not None else 0.0
    decode_pct = round((mean_decode / mean_wall) * 100.0, 1) if mean_wall and mean_decode is not None else 0.0

    prompt_toks = [int(r["prompt_tokens"]) for r in results if r.get("prompt_tokens") is not None]
    comp_toks = [int(r["completion_tokens"]) for r in results if r.get("completion_tokens") is not None]
    total_prompt = sum(prompt_toks)
    total_comp = sum(comp_toks)
    mean_prompt = round(mean(prompt_toks), 1) if prompt_toks else 0.0
    mean_comp = round(mean(comp_toks), 1) if comp_toks else 0.0

    return {
        "sample_count": sample_count,
        "passed_count": passed_count,
        "pass_at_1_pct": pass_at_1_pct,
        "categories": categories,
        "latency": {
            "n_wall": len(walls),
            "mean_wall_ms": mean_wall,
            "median_wall_ms": med_wall,
            "total_wall_s": round(sum(walls) / 1000.0, 2) if walls else 0.0,
            "n_overhead": len(overheads),
            "mean_proxy_overhead_ms": mean_overhead,
            "median_proxy_overhead_ms": med_overhead,
            "proxy_pct": proxy_pct,
            "n_ttft": len(ttfts),
            "mean_ttft_ms": mean_ttft,
            "median_ttft_ms": med_ttft,
            "n_prefill": len(prefills),
            "mean_prefill_ms": mean_prefill,
            "median_prefill_ms": med_prefill,
            "prefill_pct": prefill_pct,
            "n_decode": len(decodes),
            "mean_decode_ms": mean_decode,
            "median_decode_ms": med_decode,
            "decode_pct": decode_pct,
            "n_decode_tok_s": len(decode_tps),
            "mean_decode_tok_s": mean_tok_s,
            "median_decode_tok_s": med_tok_s,
        },
        "tokens": {
            "n_tokens": len(prompt_toks),
            "total_prompt_tokens": total_prompt,
            "mean_prompt_tokens": mean_prompt,
            "total_completion_tokens": total_comp,
            "mean_completion_tokens": mean_comp,
        },
    }


def compute_metrics(results: list[dict]) -> dict:
    summary = summarize_run_results(results)
    total = summary["sample_count"]
    passed = summary["passed_count"]

    ef = [r for r in results if r.get("category") == "exact_fact"]
    ct = [r for r in results if r.get("category") == "continuity"]
    ef_pass = sum(1 for r in ef if r.get("passed"))
    ct_pass = sum(1 for r in ct if r.get("passed"))

    lat = summary["latency"]
    tok = summary["tokens"]

    raw_ttfts = [r["ttft_total_ms"] for r in results if r.get("ttft_total_ms") is not None]

    return {
        "sample_count": total,
        "passed_count": passed,
        "pass_at_1_pct": summary["pass_at_1_pct"],
        "categories": summary["categories"],
        "pass_at_1": {
            "overall": round(passed / total, 4) if total else 0.0,
            "exact_fact": round(ef_pass / len(ef), 4) if ef else None,
            "continuity": round(ct_pass / len(ct), 4) if ct else None,
            "counts": {
                "total": total,
                "passed": passed,
                "exact_fact_total": len(ef),
                "exact_fact_passed": ef_pass,
                "continuity_total": len(ct),
                "continuity_passed": ct_pass,
            },
        },
        "tokens": {
            "total_prompt": tok["total_prompt_tokens"],
            "total_completion": tok["total_completion_tokens"],
            "mean_prompt": tok["mean_prompt_tokens"],
            "mean_completion": tok["mean_completion_tokens"],
            "n_tokens": tok["n_tokens"],
        },
        "latency": {
            "n_wall": lat["n_wall"],
            "total_wall_s": lat["total_wall_s"],
            "mean_wall_ms": lat["mean_wall_ms"],
            "median_wall_ms": lat["median_wall_ms"],
            "n_overhead": lat["n_overhead"],
            "mean_proxy_overhead_ms": lat["mean_proxy_overhead_ms"],
            "median_proxy_overhead_ms": lat["median_proxy_overhead_ms"],
            "proxy_pct": lat["proxy_pct"],
            "n_ttft": lat["n_ttft"],
            "mean_ttft_ms": lat["mean_ttft_ms"],
            "median_ttft_ms": lat["median_ttft_ms"],
            "p95_ttft_ms": _safe_p95(raw_ttfts),
            "n_prefill": lat["n_prefill"],
            "mean_prefill_ms": lat["mean_prefill_ms"],
            "median_prefill_ms": lat["median_prefill_ms"],
            "prefill_pct": lat["prefill_pct"],
            "n_decode": lat["n_decode"],
            "mean_decode_ms": lat["mean_decode_ms"],
            "median_decode_ms": lat["median_decode_ms"],
            "decode_pct": lat["decode_pct"],
            "n_decode_tok_s": lat["n_decode_tok_s"],
            "mean_decode_tok_s": lat["mean_decode_tok_s"],
            "median_decode_tok_s": lat["median_decode_tok_s"],
            "mean_gen_toks_s": lat["mean_decode_tok_s"],
        },
    }


def format_summary_report(summary: dict, label: str = "") -> str:
    lines = []
    header = f"Benchmark Summary: {label}" if label else "Benchmark Summary"
    lines.append("=" * 66)
    lines.append(f"  {header}")
    lines.append("=" * 66)
    n_total = summary.get("sample_count", 0)
    n_pass = summary.get("passed_count", 0)
    pct = summary.get("pass_at_1_pct", 0.0)
    lines.append(f"  Pass@1 (overall):       {pct:.1f}% ({n_pass}/{n_total}, n={n_total})")

    cats = summary.get("categories", {})
    for cat_name, cat_data in cats.items():
        fmt = cat_data.get("formatted", f"{cat_data.get('pass_at_1_pct', 0.0):.1f}%")
        lines.append(f"  Pass@1 {cat_name:<16}: {fmt}")

    tok = summary.get("tokens", {})
    n_tok = tok.get("n_tokens", 0)
    lines.append(
        f"  Prompt tokens:          total={tok.get('total_prompt_tokens', 0):,}  "
        f"mean={tok.get('mean_prompt_tokens', 0):.1f} (n={n_tok})"
    )
    lines.append(
        f"  Completion tokens:      total={tok.get('total_completion_tokens', 0):,}  "
        f"mean={tok.get('mean_completion_tokens', 0):.1f} (n={n_tok})"
    )

    lat = summary.get("latency", {})
    n_wall = lat.get("n_wall", 0)
    wall_ms_val = lat.get("mean_wall_ms")
    med_wall_val = lat.get("median_wall_ms")
    wall_ms_str = f"{wall_ms_val:.2f}ms" if wall_ms_val is not None else "-"
    med_wall_str = f"{med_wall_val:.2f}ms" if med_wall_val is not None else "-"
    lines.append(
        f"  Total wall time:        {lat.get('total_wall_s', 0.0):.2f}s "
        f"(mean={wall_ms_str}, med={med_wall_str}, n={n_wall})"
    )

    n_over = lat.get("n_overhead", 0)
    if lat.get("mean_proxy_overhead_ms") is not None:
        lines.append(
            f"  Proxy overhead:         mean={lat.get('mean_proxy_overhead_ms', 0.0):.2f}ms, "
            f"med={lat.get('median_proxy_overhead_ms', 0.0):.2f}ms "
            f"({lat.get('proxy_pct', 0.0):.1f}% of wall, n={n_over})"
        )

    n_ttft = lat.get("n_ttft", 0)
    if lat.get("mean_ttft_ms") is not None:
        lines.append(
            f"  Prefill / TTFT:         mean={lat.get('mean_prefill_ms', 0.0):.2f}ms, "
            f"med={lat.get('median_prefill_ms', 0.0):.2f}ms "
            f"({lat.get('prefill_pct', 0.0):.1f}% of wall, n={n_ttft})"
        )
        lines.append(
            f"  Total TTFT:             mean={lat.get('mean_ttft_ms', 0.0):.2f}ms, "
            f"med={lat.get('median_ttft_ms', 0.0):.2f}ms (n={n_ttft})"
        )

    n_dec = lat.get("n_decode", 0)
    if lat.get("mean_decode_ms") is not None:
        lines.append(
            f"  Decode duration:        mean={lat.get('mean_decode_ms', 0.0):.2f}ms, "
            f"med={lat.get('median_decode_ms', 0.0):.2f}ms "
            f"({lat.get('decode_pct', 0.0):.1f}% of wall, n={n_dec})"
        )

    n_tps = lat.get("n_decode_tok_s", 0)
    if lat.get("mean_decode_tok_s") is not None:
        lines.append(
            f"  Decode speed:           mean={lat.get('mean_decode_tok_s', 0.0):.2f} tok/s, "
            f"med={lat.get('median_decode_tok_s', 0.0):.2f} tok/s (n={n_tps})"
        )

    lines.append("=" * 66)
    return "\n".join(lines)


def format_comparison_table(raw_summary: dict, edgemem_summary: dict) -> str:
    def _pct_delta(base: float | None, current: float | None) -> str:
        if base is None or current is None or base == 0:
            return "-"
        delta = ((current - base) / base) * 100.0
        sign = "+" if delta > 0 else ""
        return f"{sign}{delta:.1f}%"

    raw_pass = raw_summary.get("pass_at_1_pct", 0.0)
    raw_n = raw_summary.get("sample_count", 0)
    raw_passed = raw_summary.get("passed_count", 0)

    edge_pass = edgemem_summary.get("pass_at_1_pct", 0.0)
    edge_n = edgemem_summary.get("sample_count", 0)
    edge_passed = edgemem_summary.get("passed_count", 0)

    raw_tok = raw_summary.get("tokens", {})
    edge_tok = edgemem_summary.get("tokens", {})
    raw_prompt = raw_tok.get("total_prompt_tokens", 0)
    edge_prompt = edge_tok.get("total_prompt_tokens", 0)

    raw_lat = raw_summary.get("latency", {})
    edge_lat = edgemem_summary.get("latency", {})
    raw_wall = raw_lat.get("mean_wall_ms")
    edge_wall = edge_lat.get("mean_wall_ms")

    edge_overhead = edge_lat.get("mean_proxy_overhead_ms")
    overhead_str = f"{edge_overhead:.2f} ms" if edge_overhead is not None else "-"

    raw_prefill = raw_lat.get("mean_prefill_ms") or raw_lat.get("mean_ttft_ms")
    edge_prefill = edge_lat.get("mean_prefill_ms") or edge_lat.get("mean_ttft_ms")
    raw_n_ttft = raw_lat.get("n_ttft", 0)
    edge_n_ttft = edge_lat.get("n_ttft", 0)

    raw_decode = raw_lat.get("mean_decode_ms")
    edge_decode = edge_lat.get("mean_decode_ms")
    raw_n_dec = raw_lat.get("n_decode", 0)
    edge_n_dec = edge_lat.get("n_decode", 0)

    raw_tps = raw_lat.get("mean_decode_tok_s")
    edge_tps = edge_lat.get("mean_decode_tok_s")

    raw_pass_str = f"{raw_pass:.1f}% ({raw_passed}/{raw_n}, n={raw_n})"
    edge_pass_str = f"{edge_pass:.1f}% ({edge_passed}/{edge_n}, n={edge_n})"
    raw_wall_str = f"{raw_wall:.2f} ms" if raw_wall is not None else "-"
    edge_wall_str = f"{edge_wall:.2f} ms" if edge_wall is not None else "-"
    raw_prefill_str = f"{raw_prefill:.2f} ms (n={raw_n_ttft})" if raw_prefill is not None else "-"
    edge_prefill_str = f"{edge_prefill:.2f} ms (n={edge_n_ttft})" if edge_prefill is not None else "-"
    raw_decode_str = f"{raw_decode:.2f} ms (n={raw_n_dec})" if raw_decode is not None else "-"
    edge_decode_str = f"{edge_decode:.2f} ms (n={edge_n_dec})" if edge_decode is not None else "-"
    raw_tps_str = f"{raw_tps:.2f} tok/s" if raw_tps is not None else "-"
    edge_tps_str = f"{edge_tps:.2f} tok/s" if edge_tps is not None else "-"

    lines = [
        "+-------------------------------+--------------------------+--------------------------+------------+",
        "| Metric                        | Raw Baseline             | EdgeMem                  | Delta      |",
        "+-------------------------------+--------------------------+--------------------------+------------+",
        f"| Pass@1                        | {raw_pass_str:<24} | {edge_pass_str:<24} | {_pct_delta(raw_pass, edge_pass):<10} |",
        f"| Prompt Tokens                 | {raw_prompt:<24} | {edge_prompt:<24} | {_pct_delta(raw_prompt, edge_prompt):<10} |",
        f"| Total Wall (ms)               | {raw_wall_str:<24} | {edge_wall_str:<24} | {_pct_delta(raw_wall, edge_wall):<10} |",
        f"| EdgeMem Proxy Overhead (ms)   | {'-':<24} | {overhead_str:<24} | {'-':<10} |",
        f"| Prefill / TTFT (ms)           | {raw_prefill_str:<24} | {edge_prefill_str:<24} | {_pct_delta(raw_prefill, edge_prefill):<10} |",
        f"| Decode (ms)                   | {raw_decode_str:<24} | {edge_decode_str:<24} | {_pct_delta(raw_decode, edge_decode):<10} |",
        f"| Decode (tok/s)                | {raw_tps_str:<24} | {edge_tps_str:<24} | {_pct_delta(raw_tps, edge_tps):<10} |",
        "+-------------------------------+--------------------------+--------------------------+------------+",
    ]
    return "\n".join(lines)


def build_markdown_report(
    label: str,
    model: str,
    base_url: str,
    metrics: dict,
    results: list[dict],
    rss_mb: float | None,
    temperature: float = 0.0,
    seed: int | None = None,
) -> str:
    p = metrics["pass_at_1"]
    t = metrics["tokens"]
    l = metrics["latency"]
    c = p["counts"]

    lines = [
        f"# EdgeMem Benchmark Report: `{label}`",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Label | `{label}` |",
        f"| Model | `{model}` |",
        f"| Endpoint | `{base_url}` |",
        f"| Temperature | `{temperature}` |",
    ]
    if seed is not None:
        lines.append(f"| Seed | `{seed}` |")
    lines.extend([
        f"| Tasks run | {c['total']} |",
        f"| **Pass@1 (overall)** | **{p['overall']:.1%}** ({c['passed']}/{c['total']}, n={c['total']}) |",
        f"| Pass@1 exact_fact | {p['exact_fact']:.1%} ({c['exact_fact_passed']}/{c['exact_fact_total']}, n={c['exact_fact_total']}) |"
        if p.get("exact_fact") is not None
        else "| Pass@1 exact_fact | - |",
        f"| Pass@1 continuity | {p['continuity']:.1%} ({c['continuity_passed']}/{c['continuity_total']}, n={c['continuity_total']}) |"
        if p.get("continuity") is not None
        else "| Pass@1 continuity | - |",
        f"| Total prompt tokens | {t['total_prompt']:,} |",
        f"| Total completion tokens | {t['total_completion']:,} |",
        f"| Mean prompt tokens | {t['mean_prompt']} |",
        f"| Mean completion tokens | {t['mean_completion']} |",
        f"| Mean Wall (ms) | {l['mean_wall_ms']} (n={l['n_wall']}) |"
        if l.get("mean_wall_ms") is not None
        else "| Mean Wall (ms) | - |",
        f"| EdgeMem Proxy Overhead (ms) | {l['mean_proxy_overhead_ms']} (n={l['n_overhead']}) |"
        if l.get("mean_proxy_overhead_ms") is not None
        else "| EdgeMem Proxy Overhead (ms) | - |",
        f"| Mean Prefill / TTFT (ms) | {l['mean_prefill_ms']} (n={l['n_prefill']}) |"
        if l.get("mean_prefill_ms") is not None
        else (f"| Mean TTFT (ms) | {l['mean_ttft_ms']} (n={l['n_ttft']}) |" if l.get("mean_ttft_ms") is not None else "| Mean TTFT (ms) | - |"),
        f"| Mean Decode (ms) | {l['mean_decode_ms']} (n={l['n_decode']}) |"
        if l.get("mean_decode_ms") is not None
        else "| Mean Decode (ms) | - |",
        f"| Mean Decode tok/s | {l['mean_decode_tok_s']} (n={l['n_decode_tok_s']}) |"
        if l.get("mean_decode_tok_s") is not None
        else "| Mean Decode tok/s | - |",
        f"| Total wall time (s) | {l['total_wall_s']} |",
        f"| Proxy RSS (MB) | {rss_mb:.1f} |" if rss_mb is not None else "| Proxy RSS (MB) | - |",
        "",
        "## Per-Task Results",
        "",
        "| Task ID | Category | Pass | Prompt tok | Completion tok | Wall (ms) | Overhead (ms) | Prefill (ms) | Decode (ms) | Decode tok/s | Error |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ])
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        err_snippet = (r.get("error") or "")[:60].replace("\n", " ")
        ov_str = f"{r['proxy_overhead_ms']:.1f}" if r.get("proxy_overhead_ms") is not None else "-"
        pf_str = f"{r['prefill_ms']:.1f}" if r.get("prefill_ms") is not None else "-"
        dc_str = f"{r['decode_ms']:.1f}" if r.get("decode_ms") is not None else "-"
        tps_str = f"{r['decode_tok_s']:.1f}" if r.get("decode_tok_s") is not None else "-"
        lines.append(
            f"| {r['task_id']} | {r['category']} | {status} | "
            f"{r['prompt_tokens']} | {r['completion_tokens']} | "
            f"{r['wall_ms']:.0f} | {ov_str} | {pf_str} | {dc_str} | {tps_str} | {err_snippet} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    tasks_file = Path(args.tasks_file) if args.tasks_file else (repo_root / "bench" / "bench_tasks.json")
    output_dir = Path(args.output_dir) if args.output_dir else (repo_root / "bench" / "results")
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = ""
    if args.api_key_env:
        api_key = os.environ.get(args.api_key_env, "")
        if not api_key and (args.api_key_env.startswith("nvapi-") or args.api_key_env.startswith("sk-")):
            api_key = args.api_key_env
        api_key = api_key.strip()

    try:
        extra_body = json.loads(args.extra_body_json)
    except json.JSONDecodeError as exc:
        sys.exit(f"--extra-body-json is not valid JSON: {exc}")

    with open(tasks_file, encoding="utf-8") as f:
        all_tasks: list[dict] = json.load(f)

    if args.category:
        all_tasks = [t for t in all_tasks if t["category"] == args.category]
    if args.limit:
        all_tasks = all_tasks[: args.limit]

    if not all_tasks:
        sys.exit("No tasks to run after filtering.")

    stream_mode_str = "streaming" if args.stream else "non-streaming"
    print(f"[EdgeMem Bench] label={args.label!r}  model={args.model!r}  tasks={len(all_tasks)}  mode={stream_mode_str}")
    print(f"  endpoint: {args.base_url}")
    print()

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
            max_retries=args.max_retries,
            temperature=args.temperature,
            seed=args.seed,
            stream=args.stream,
        )
        elapsed = time.perf_counter() - t0
        status_icon = "PASS" if result["passed"] else "FAIL"
        print(f"{status_icon}  ({elapsed:.1f}s)")
        if not result["passed"] and result.get("error"):
            print(f"         error: {result['error'][:120]}")
        results.append(result)

        if args.delay > 0 and i < len(all_tasks):
            time.sleep(args.delay)

    rss_after: float | None = None
    if args.health_url:
        hdata = _http_get_json(args.health_url)
        if hdata:
            rss_after = hdata.get("rss_mb")
    rss_mb = rss_after

    metrics = compute_metrics(results)
    summary = summarize_run_results(results)

    print()
    print(format_summary_report(summary, label=args.label))

    if args.baseline:
        base_path = Path(args.baseline)
        if base_path.exists():
            with open(base_path, encoding="utf-8") as bf:
                base_data = json.load(bf)
            base_results = base_data.get("results", [])
            base_summary = summarize_run_results(base_results)
            print()
            print(format_comparison_table(base_summary, summary))
        else:
            print(f"  [warn] baseline file not found: {args.baseline}", file=sys.stderr)

    output_data = {
        "label": args.label,
        "model": args.model,
        "base_url": args.base_url,
        "stream": args.stream,
        "temperature": args.temperature,
        "sample_count": summary["sample_count"],
        "metrics": metrics,
        "summary": summary,
        "rss_mb": rss_mb,
        "results": results,
    }
    if args.seed is not None:
        output_data["seed"] = args.seed
    json_path = output_dir / f"{args.label}_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n  JSON: {json_path}")

    md = build_markdown_report(
        label=args.label,
        model=args.model,
        base_url=args.base_url,
        metrics=metrics,
        results=results,
        rss_mb=rss_mb,
        temperature=args.temperature,
        seed=args.seed,
    )
    md_path = output_dir / f"{args.label}_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  Markdown: {md_path}")


if __name__ == "__main__":
    main()
