"""Local Baseline Probe for EdgeMem Phase 0.

Measures Ollama idle RSS, peak RSS, TTFT, generation speed (tok/s),
and streaming + OpenAI-schema tool calling correctness on local LLM test rig.
OS-agnostic, LLM-agnostic, paths managed via pathlib and configurable base_url.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import httpx
import psutil
import tiktoken

sys.stdout.reconfigure(line_buffering=True)


class EfficientOllamaRSSMonitor:
    """Monitors Ollama daemon and llama-server RSS with minimal CPU footprint."""

    def __init__(self, sample_interval_s: float = 0.5):
        self.sample_interval_s = sample_interval_s
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.idle_rss_bytes: int = 0
        self.peak_rss_bytes: int = 0
        self._target_pids: Set[int] = set()

    def _discover_pids(self) -> Set[int]:
        pids = set()
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                name = (proc.info["name"] or "").lower()
                if "ollama" in name or "llama-server" in name:
                    pids.add(proc.info["pid"])
                    for child in proc.children(recursive=True):
                        pids.add(child.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return pids

    def _get_current_rss(self) -> int:
        total = 0
        dead_pids = set()
        for pid in list(self._target_pids):
            try:
                proc = psutil.Process(pid)
                total += proc.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                dead_pids.add(pid)
        self._target_pids.difference_update(dead_pids)
        return total

    def start(self):
        self._target_pids = self._discover_pids()
        self.idle_rss_bytes = self._get_current_rss()
        self.peak_rss_bytes = self.idle_rss_bytes
        self._stop_event.clear()

        def _monitor():
            counter = 0
            while not self._stop_event.is_set():
                counter += 1
                if counter % 4 == 0 or not self._target_pids:
                    self._target_pids = self._discover_pids()

                current = self._get_current_rss()
                if current > self.peak_rss_bytes:
                    self.peak_rss_bytes = current
                time.sleep(self.sample_interval_s)

        self._thread = threading.Thread(target=_monitor, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.5)


def run_streaming_completion(
    client: httpx.Client,
    base_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int = 128,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
    enc: Optional[Any] = None,
) -> Dict[str, Any]:
    """Execute streaming chat completion request with reasoning_effort='none'."""
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
        "reasoning_effort": "none",
        "thinking": False,
    }
    if tools:
        payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice

    url = f"{base_url.rstrip('/')}/chat/completions"

    ttft_ms: Optional[float] = None
    start_time = time.perf_counter()
    first_token_time: Optional[float] = None
    last_token_time: Optional[float] = None

    content_chunks: List[str] = []
    reasoning_chunks: List[str] = []
    tool_calls_accumulator: Dict[int, Dict[str, Any]] = {}
    finish_reason: Optional[str] = None

    with client.stream("POST", url, json=payload, timeout=60.0) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            line = line.strip()
            if not line:
                continue
            if line == "data: [DONE]":
                break
            if not line.startswith("data: "):
                continue

            data_str = line[len("data: ") :]
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            now = time.perf_counter()
            if ttft_ms is None:
                first_token_time = now
                ttft_ms = (now - start_time) * 1000.0
            last_token_time = now

            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            c = delta.get("content")
            if c:
                content_chunks.append(c)
            r = delta.get("reasoning")
            if r:
                reasoning_chunks.append(r)

            if choices[0].get("finish_reason"):
                finish_reason = choices[0].get("finish_reason")

            tc_deltas = delta.get("tool_calls")
            if tc_deltas:
                for tc in tc_deltas:
                    idx = tc.get("index", 0)
                    if idx not in tool_calls_accumulator:
                        tool_calls_accumulator[idx] = {
                            "id": tc.get("id", ""),
                            "type": tc.get("type", "function"),
                            "function": {
                                "name": tc.get("function", {}).get("name", ""),
                                "arguments": tc.get("function", {}).get("arguments", ""),
                            },
                        }
                    else:
                        existing = tool_calls_accumulator[idx]
                        if tc.get("id"):
                            existing["id"] = tc.get("id")
                        fn = tc.get("function", {})
                        if fn.get("name"):
                            existing["function"]["name"] += fn.get("name")
                        if fn.get("arguments"):
                            existing["function"]["arguments"] += fn.get("arguments")

    end_time = time.perf_counter()
    gen_duration = (last_token_time - first_token_time) if (first_token_time and last_token_time) else 0.0
    full_content = "".join(content_chunks)
    full_reasoning = "".join(reasoning_chunks)

    generated_text = full_reasoning + full_content
    if enc is not None:
        token_count = len(enc.encode(generated_text, disallowed_special=()))
    else:
        token_count = len(generated_text.split())

    tok_per_sec = (token_count / gen_duration) if gen_duration > 0 else 0.0

    return {
        "ttft_ms": round(ttft_ms, 2) if ttft_ms else 0.0,
        "generation_duration_s": round(gen_duration, 3),
        "total_duration_s": round(end_time - start_time, 3),
        "generated_tokens": token_count,
        "tok_per_sec": round(tok_per_sec, 2),
        "content_length": len(full_content),
        "reasoning_length": len(full_reasoning),
        "reasoning_empty": len(full_reasoning) == 0,
        "tool_calls": list(tool_calls_accumulator.values()),
        "finish_reason": finish_reason,
        "content_preview": full_content[:100].strip(),
    }


def main():
    parser = argparse.ArgumentParser(description="EdgeMem Local Baseline Probe")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1"),
        help="OpenAI-compatible base URL",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "qwen3.5:2b-q4_K_M"),
        help="Model name",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Max tokens per prompt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("spike/baseline_results.json"),
        help="Path to save results JSON",
    )
    args = parser.parse_args()

    print("=== EdgeMem Phase 0 Baseline Probe ===", flush=True)
    print(f"Target URL: {args.base_url}", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Max Tokens per prompt: {args.max_tokens}", flush=True)
    print("Reasoning Effort: none (thinking disabled)", flush=True)

    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        enc = None

    monitor = EfficientOllamaRSSMonitor(sample_interval_s=0.5)
    monitor.start()

    time.sleep(0.5)
    idle_mb = monitor.idle_rss_bytes / (1024 * 1024)
    print(f"Initial Ollama RSS (idle): {idle_mb:.2f} MB", flush=True)

    test_prompts = [
        {
            "id": "prompt_1",
            "name": "is_palindrome",
            "type": "code",
            "messages": [
                {
                    "role": "user",
                    "content": "Write a concise Python function `is_palindrome(s: str) -> bool`.",
                }
            ],
        },
        {
            "id": "prompt_2",
            "name": "fibonacci_gen",
            "type": "code",
            "messages": [
                {
                    "role": "user",
                    "content": "Write a concise Python generator `fib(n: int)` yielding first n Fibonacci numbers.",
                }
            ],
        },
        {
            "id": "prompt_3",
            "name": "tool_calling_ast",
            "type": "tool_call",
            "messages": [
                {
                    "role": "user",
                    "content": "Call the fetch_file_ast tool for 'src/edgemem/proxy.py' with max_depth 3.",
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "fetch_file_ast",
                        "description": "Extract AST summary of a file given its repository relative path.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "file_path": {
                                    "type": "string",
                                    "description": "Relative file path in repository",
                                },
                                "max_depth": {
                                    "type": "integer",
                                    "description": "Maximum AST depth",
                                },
                            },
                            "required": ["file_path"],
                        },
                    },
                }
            ],
            "tool_choice": "auto",
        },
    ]

    results = []
    streaming_success_all = True
    tool_calling_success = False

    with httpx.Client() as client:
        for idx, p in enumerate(test_prompts, 1):
            print(f"\n[{idx}/3] Running {p['name']} ({p['type']})...", flush=True)
            try:
                res = run_streaming_completion(
                    client=client,
                    base_url=args.base_url,
                    model=args.model,
                    messages=p["messages"],
                    max_tokens=args.max_tokens,
                    tools=p.get("tools"),
                    tool_choice=p.get("tool_choice"),
                    enc=enc,
                )
                res["prompt_id"] = p["id"]
                res["name"] = p["name"]
                res["type"] = p["type"]
                results.append(res)

                print(
                    f"  TTFT: {res['ttft_ms']} ms | Speed: {res['tok_per_sec']} tok/s | Tokens: {res['generated_tokens']} | Duration: {res['generation_duration_s']}s",
                    flush=True,
                )
                print(f"  Reasoning empty/null: {res['reasoning_empty']} (length: {res['reasoning_length']})", flush=True)
                if res["content_preview"]:
                    print(f"  Preview: {res['content_preview'][:70]}...", flush=True)

                if p["type"] == "tool_call":
                    tc = res.get("tool_calls", [])
                    if tc and tc[0]["function"]["name"] == "fetch_file_ast":
                        tool_calling_success = True
                        print(
                            f"  Tool call SUCCESS: {tc[0]['function']['name']} args: {tc[0]['function']['arguments']}",
                            flush=True,
                        )
                    else:
                        print(f"  Tool call status: {tc}", flush=True)
            except Exception as e:
                print(f"  FAILED with error: {e}", flush=True)
                streaming_success_all = False
                results.append({"prompt_id": p["id"], "name": p["name"], "error": str(e)})

    monitor.stop()

    peak_mb = monitor.peak_rss_bytes / (1024 * 1024)
    print("\n=== Baseline Probe Summary ===", flush=True)
    print(f"Ollama Idle RSS: {idle_mb:.2f} MB", flush=True)
    print(f"Ollama Peak RSS: {peak_mb:.2f} MB", flush=True)
    print(f"Streaming clean success: {streaming_success_all}", flush=True)
    print(f"Tool calling clean success: {tool_calling_success}", flush=True)

    valid_results = [r for r in results if "ttft_ms" in r]
    avg_ttft = sum(r["ttft_ms"] for r in valid_results) / len(valid_results) if valid_results else 0.0
    avg_speed = sum(r["tok_per_sec"] for r in valid_results) / len(valid_results) if valid_results else 0.0

    print(f"Average TTFT: {avg_ttft:.2f} ms", flush=True)
    print(f"Average Generation Speed: {avg_speed:.2f} tok/s", flush=True)

    summary_data = {
        "timestamp": time.time(),
        "model": args.model,
        "base_url": args.base_url,
        "reasoning_effort": "none",
        "max_tokens": args.max_tokens,
        "ollama_idle_rss_mb": round(idle_mb, 2),
        "ollama_peak_rss_mb": round(peak_mb, 2),
        "streaming_success": streaming_success_all,
        "tool_calling_success": tool_calling_success,
        "avg_ttft_ms": round(avg_ttft, 2),
        "avg_speed_tok_s": round(avg_speed, 2),
        "prompts": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
