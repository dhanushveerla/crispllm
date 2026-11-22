"""tests/test_bench.py — Grader Self-Test and Latency Decomposition Tests for EdgeMem Benchmark Suite."""
from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch
import urllib.error

import pytest

BENCH_DIR = Path(__file__).parent.parent / "bench"
TASKS_FILE = BENCH_DIR / "bench_tasks.json"
STRESS_TASKS_FILE = BENCH_DIR / "stress_tasks.json"


@pytest.fixture(scope="module")
def tasks() -> list[dict]:
    assert TASKS_FILE.exists(), f"bench_tasks.json not found at {TASKS_FILE}"
    with open(TASKS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return data


@pytest.fixture(scope="module")
def stress_tasks() -> list[dict]:
    assert STRESS_TASKS_FILE.exists(), f"stress_tasks.json not found at {STRESS_TASKS_FILE}"
    with open(STRESS_TASKS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return data


REQUIRED_KEYS = {"id", "category", "files", "turns", "reference_solution", "test_code"}
VALID_CATEGORIES = {"exact_fact", "continuity"}


class TestSchema:
    def test_task_count(self, tasks):
        assert len(tasks) == 20, f"Expected 20 tasks, got {len(tasks)}"

    @pytest.mark.parametrize("idx", range(20))
    def test_task_has_required_keys(self, tasks, idx):
        task = tasks[idx]
        missing = REQUIRED_KEYS - set(task.keys())
        assert not missing, f"Task {idx} missing keys: {missing}"

    @pytest.mark.parametrize("idx", range(20))
    def test_task_id_is_string(self, tasks, idx):
        assert isinstance(tasks[idx]["id"], str) and tasks[idx]["id"], (
            f"Task {idx} id must be non-empty string"
        )

    @pytest.mark.parametrize("idx", range(20))
    def test_task_category_valid(self, tasks, idx):
        cat = tasks[idx]["category"]
        assert cat in VALID_CATEGORIES, f"Task {idx} has invalid category: {cat!r}"

    @pytest.mark.parametrize("idx", range(20))
    def test_task_files_is_dict(self, tasks, idx):
        files = tasks[idx]["files"]
        assert isinstance(files, dict) and files, f"Task {idx} 'files' must be a non-empty dict"
        for path, content in files.items():
            assert isinstance(path, str) and path, f"Task {idx}: file path must be non-empty string"
            assert isinstance(content, str), f"Task {idx}: file content for {path!r} must be string"

    @pytest.mark.parametrize("idx", range(20))
    def test_task_turns_is_list(self, tasks, idx):
        turns = tasks[idx]["turns"]
        assert isinstance(turns, list) and turns, f"Task {idx} 'turns' must be non-empty list"
        for t in turns:
            assert isinstance(t, str) and t, f"Task {idx}: each turn must be non-empty string"

    @pytest.mark.parametrize("idx", range(20))
    def test_task_reference_solution_is_dict(self, tasks, idx):
        ref = tasks[idx]["reference_solution"]
        assert isinstance(ref, dict) and ref, f"Task {idx} 'reference_solution' must be non-empty dict"
        for path, code in ref.items():
            assert isinstance(path, str) and path
            assert isinstance(code, str)

    @pytest.mark.parametrize("idx", range(20))
    def test_task_test_code_is_string(self, tasks, idx):
        tc = tasks[idx]["test_code"]
        assert isinstance(tc, str) and tc, f"Task {idx} 'test_code' must be non-empty string"

    @pytest.mark.parametrize("idx", range(20))
    def test_task_id_unique(self, tasks, idx):
        ids = [t["id"] for t in tasks]
        task_id = tasks[idx]["id"]
        assert ids.count(task_id) == 1, f"Duplicate task id: {task_id!r}"

    def test_continuity_task_multi_turn(self, tasks):
        ct_tasks = [t for t in tasks if t["category"] == "continuity"]
        for t in ct_tasks:
            assert len(t["turns"]) >= 4, (
                f"continuity task {t['id']!r} has only {len(t['turns'])} turns (need >=4)"
            )


class TestCategoryBalance:
    def test_exactly_10_exact_fact(self, tasks):
        ef = [t for t in tasks if t["category"] == "exact_fact"]
        assert len(ef) == 10, f"Expected 10 exact_fact tasks, got {len(ef)}"

    def test_exactly_10_continuity(self, tasks):
        ct = [t for t in tasks if t["category"] == "continuity"]
        assert len(ct) == 10, f"Expected 10 continuity tasks, got {len(ct)}"


def _run_test_in_tmpdir(
    task: dict,
    solution_files: dict[str, str],
    timeout: int = 15,
) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="edgemem_selftest_") as tmpdir:
        tmp = Path(tmpdir)

        for rel_path, content in task["files"].items():
            dest = tmp / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")

        for rel_path, code in solution_files.items():
            dest = tmp / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(code, encoding="utf-8")

        test_file = tmp / "_test_runner_.py"
        test_file.write_text(task["test_code"], encoding="utf-8")

        try:
            proc = subprocess.run(
                [sys.executable, str(test_file), str(tmp)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(tmp),
                encoding="utf-8",
                errors="replace",
            )
            passed = proc.returncode == 0 and "PASS" in proc.stdout
            output = (proc.stdout + proc.stderr).strip()
            return passed, output
        except subprocess.TimeoutExpired:
            return False, f"timeout after {timeout}s"
        except Exception as exc:
            return False, str(exc)


class TestReferenceSolutionsPass:
    @pytest.mark.parametrize("idx", range(20))
    def test_reference_solution_passes(self, tasks, idx):
        task = tasks[idx]
        passed, output = _run_test_in_tmpdir(
            task=task,
            solution_files=task["reference_solution"],
        )
        assert passed, (
            f"Task {task['id']!r} reference_solution FAILED its own test_code.\n"
            f"Output:\n{output[-1200:]}"
        )


BROKEN_PYTHON = "pass\n"


def _make_empty_solution(task: dict) -> dict[str, str]:
    empty: dict[str, str] = {}
    for rel_path in task["reference_solution"]:
        empty[rel_path] = BROKEN_PYTHON
    return empty


class TestBrokenSolutionFails:
    @pytest.mark.parametrize("idx", range(20))
    def test_empty_solution_fails(self, tasks, idx):
        task = tasks[idx]
        broken_sol = _make_empty_solution(task)
        passed, _output = _run_test_in_tmpdir(
            task=task,
            solution_files=broken_sol,
        )
        assert not passed, (
            f"Task {task['id']!r}: empty/broken solution unexpectedly PASSED — "
            f"test_code may not be asserting correctly."
        )


class TestPassAtOneCoverage:
    def test_all_reference_solutions_give_100_pass(self, tasks):
        failures = []
        for task in tasks:
            passed, output = _run_test_in_tmpdir(
                task=task,
                solution_files=task["reference_solution"],
            )
            if not passed:
                failures.append(f"  FAIL: {task['id']!r}\n    {output[-1200:]}")
        assert not failures, (
            f"Reference solutions did not achieve 100% Pass@1:\n" + "\n".join(failures)
        )

    def test_all_empty_solutions_give_0_pass(self, tasks):
        unexpected_passes = []
        for task in tasks:
            broken_sol = _make_empty_solution(task)
            passed, _ = _run_test_in_tmpdir(task=task, solution_files=broken_sol)
            if passed:
                unexpected_passes.append(task["id"])
        assert not unexpected_passes, (
            f"Empty solution unexpectedly passed: {unexpected_passes}"
        )


class TestStressTasksSuite:
    def test_stress_task_count(self, stress_tasks):
        assert len(stress_tasks) == 20, f"Expected 20 stress tasks, got {len(stress_tasks)}"

    @pytest.mark.parametrize("idx", range(20))
    def test_stress_task_schema(self, stress_tasks, idx):
        task = stress_tasks[idx]
        missing = REQUIRED_KEYS - set(task.keys())
        assert not missing, f"Stress task {idx} missing keys: {missing}"
        assert task["category"] in {
            "constraint_retention",
            "contradiction_update",
            "noisy_distractor",
            "cross_turn_synthesis",
            "stress_continuity",
        }, f"Unexpected category {task['category']!r}"
        assert len(task["turns"]) >= 6, f"Expected >= 6 turns, got {len(task['turns'])}"

    @pytest.mark.parametrize("idx", range(20))
    def test_stress_task_prompt_length(self, stress_tasks, idx):
        task = stress_tasks[idx]
        file_chars = sum(len(c) for c in task["files"].values())
        turn_chars = sum(len(t) for t in task["turns"])
        estimated_tokens = (file_chars + turn_chars) // 4
        assert estimated_tokens >= 3500, (
            f"Stress task {task['id']!r} prompt too short: {estimated_tokens} tokens (< 3500)"
        )
        assert estimated_tokens <= 8500, (
            f"Stress task {task['id']!r} prompt too long: {estimated_tokens} tokens (> 8500)"
        )

    @pytest.mark.parametrize("idx", range(20))
    def test_stress_reference_solution_passes(self, stress_tasks, idx):
        task = stress_tasks[idx]
        passed, output = _run_test_in_tmpdir(
            task=task,
            solution_files=task["reference_solution"],
        )
        assert passed, (
            f"Stress task {task['id']!r} reference solution FAILED test_code.\n"
            f"Output:\n{output[-1200:]}"
        )

    @pytest.mark.parametrize("idx", range(20))
    def test_stress_empty_solution_fails(self, stress_tasks, idx):
        task = stress_tasks[idx]
        broken_sol = _make_empty_solution(task)
        passed, _output = _run_test_in_tmpdir(
            task=task,
            solution_files=broken_sol,
        )
        assert not passed, (
            f"Stress task {task['id']!r}: empty/broken solution unexpectedly PASSED."
        )


class TestDeterministicSampling:
    def test_default_payload_temperature_and_seed(self):
        from bench.run_bench import build_chat_payload, build_parser

        parser = build_parser()
        args = parser.parse_args(["--base-url", "http://localhost:8080/v1", "--model", "test-model"])
        assert args.temperature == 0.0
        assert args.seed is None

        payload = build_chat_payload(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            temperature=args.temperature,
            seed=args.seed,
        )
        assert payload["temperature"] == 0.0
        assert "seed" not in payload

    def test_custom_payload_temperature_and_seed(self):
        from bench.run_bench import build_chat_payload, build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--base-url", "http://localhost:8080/v1",
            "--model", "test-model",
            "--temperature", "0.7",
            "--seed", "42",
        ])
        assert args.temperature == 0.7
        assert args.seed == 42

        payload = build_chat_payload(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            temperature=args.temperature,
            seed=args.seed,
        )
        assert payload["temperature"] == 0.7
        assert payload["seed"] == 42


class TestProxyTimingHeaders:
    def test_proxy_timing_headers_non_streaming(self, tmp_path: Path):
        from crispllm.config import EdgeMemConfig
        from crispllm.proxy.app import create_app
        from starlette.testclient import TestClient

        cfg = EdgeMemConfig()
        cfg.memory.db_path = str(tmp_path / "timing_test.db")
        cfg.telemetry_dir = str(tmp_path / "telemetry")
        cfg.upstream_base_url = "http://localhost:11434/v1"
        app = create_app(cfg)

        with patch("httpx.AsyncClient.post") as mock_post:
            fake_response = MagicMock()
            fake_response.status_code = 200
            fake_response.headers = {"content-type": "application/json"}
            fake_response.json.return_value = {
                "id": "chatcmpl-test",
                "choices": [{"message": {"role": "assistant", "content": "Hi"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
            }

            async def fake_post_async(*args: Any, **kwargs: Any) -> Any:
                return fake_response

            mock_post.side_effect = fake_post_async

            with TestClient(app) as client:
                resp = client.post(
                    "/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "hello"}]},
                )
                assert resp.status_code == 200
                assert "X-EdgeMem-Overhead-Ms" in resp.headers
                overhead = float(resp.headers["X-EdgeMem-Overhead-Ms"])
                assert overhead >= 0.0
                assert "Server-Timing" in resp.headers
                assert f"edgemem_proxy;dur={overhead:.2f}" in resp.headers["Server-Timing"]

    def test_proxy_timing_headers_streaming(self, tmp_path: Path):
        from crispllm.config import EdgeMemConfig
        from crispllm.proxy.app import create_app
        from starlette.testclient import TestClient

        cfg = EdgeMemConfig()
        cfg.memory.db_path = str(tmp_path / "timing_stream_test.db")
        cfg.telemetry_dir = str(tmp_path / "telemetry")
        cfg.upstream_base_url = "http://localhost:11434/v1"
        app = create_app(cfg)

        async def fake_stream_lines():
            yield "data: " + json.dumps({"choices": [{"delta": {"content": "Hello"}}]}) + "\n\n"
            yield "data: " + json.dumps({"choices": [{"delta": {"content": " world"}}]}) + "\n\n"
            yield "data: [DONE]\n\n"

        with patch("crispllm.proxy.app.stream_with_heartbeat") as mock_stream:
            async def fake_stream_fn(*args: Any, **kwargs: Any) -> Any:
                on_first_data = kwargs.get("on_first_data")
                if on_first_data:
                    on_first_data()
                async for line in fake_stream_lines():
                    yield line

            mock_stream.side_effect = fake_stream_fn

            with TestClient(app) as client:
                resp = client.post(
                    "/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "hello"}], "stream": True},
                )
                assert resp.status_code == 200
                assert "X-EdgeMem-Overhead-Ms" in resp.headers
                overhead = float(resp.headers["X-EdgeMem-Overhead-Ms"])
                assert overhead >= 0.0
                assert "Server-Timing" in resp.headers
                assert f"edgemem_proxy;dur={overhead:.2f}" in resp.headers["Server-Timing"]
                assert "data: " in resp.text
                assert "Hello" in resp.text


class TestStreamingLatencyDecomposition:
    def test_parse_sse_stream_exact_math(self):
        from bench.run_bench import parse_sse_stream

        lines = [
            b"data: " + json.dumps({"choices": [{"delta": {"role": "assistant", "content": ""}}]}).encode("utf-8") + b"\n",
            b"data: " + json.dumps({"choices": [{"delta": {"content": "Hello"}}]}).encode("utf-8") + b"\n",
            b"data: " + json.dumps({"choices": [{"delta": {"content": " world"}}]}).encode("utf-8") + b"\n",
            b"data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": 15, "completion_tokens": 10}}).encode("utf-8") + b"\n",
            b"data: [DONE]\n",
        ]

        times = [100.05, 100.20]
        time_iter = iter(times)

        def mock_clock() -> float:
            return next(time_iter)

        headers = {"X-EdgeMem-Overhead-Ms": "12.50"}
        t_start = 100.0

        res = parse_sse_stream(lines, headers, t_start=t_start, clock=mock_clock)

        assert res["final_text"] == "Hello world"
        assert res["proxy_overhead_ms"] == 12.50
        assert res["ttft_total_ms"] == pytest.approx(50.0, rel=1e-3)
        assert res["wall_ms"] == pytest.approx(200.0, rel=1e-3)
        assert res["prefill_ms"] == pytest.approx(res["ttft_total_ms"] - 12.50, rel=1e-3)
        assert res["decode_ms"] == pytest.approx(res["wall_ms"] - res["ttft_total_ms"], rel=1e-3)
        expected_tok_s = 10.0 / (res["decode_ms"] / 1000.0)
        assert res["decode_tok_s"] == pytest.approx(expected_tok_s, rel=1e-3)
        assert res["prompt_tokens"] == 15
        assert res["completion_tokens"] == 10
        assert res["usage_source"] == "api_usage"


class TestStreamOptionsFallback:
    def test_stream_options_400_fallback_retry(self):
        from bench.run_bench import _http_stream_chat

        first_call = True

        def fake_urlopen(req: Any, timeout: int = 120) -> Any:
            nonlocal first_call
            data = json.loads(req.data.decode("utf-8"))
            if first_call:
                first_call = False
                assert "stream_options" in data
                raise urllib.error.HTTPError(
                    url="http://mock/chat/completions",
                    code=400,
                    msg="stream_options not supported",
                    hdrs={},
                    fp=io.BytesIO(b'{"error": "unsupported stream_options"}'),
                )
            assert "stream_options" not in data
            assert data["stream"] is True
            fake_resp = MagicMock()
            fake_resp.headers = {"X-EdgeMem-Overhead-Ms": "2.0"}
            fake_resp.__iter__.return_value = [
                b"data: " + json.dumps({"choices": [{"delta": {"content": "ok"}}]}).encode("utf-8") + b"\n",
                b"data: [DONE]\n",
            ]
            fake_resp.__enter__.return_value = fake_resp
            fake_resp.__exit__.return_value = None
            return fake_resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            res = _http_stream_chat(
                "http://mock/chat/completions",
                {"model": "test", "messages": []},
                api_key="",
                timeout=10,
            )
            assert res["final_text"] == "ok"
            assert res["proxy_overhead_ms"] == 2.0


class TestNoStreamMode:
    def test_no_stream_mode_timing_splits(self, tmp_path: Path):
        from bench.run_bench import run_task

        task = {
            "id": "test_ns",
            "category": "exact_fact",
            "files": {"helper.py": "def add(a, b): return a + b\n"},
            "turns": ["Return code block with add."],
            "reference_solution": {"helper.py": "def add(a, b): return a + b\n"},
            "test_code": "print('PASS')\n",
        }

        fake_resp = {
            "choices": [{"message": {"content": "```python\n# filename: helper.py\ndef add(a, b): return a + b\n```"}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10},
        }

        with patch("bench.run_bench._http_post_json", return_value=(fake_resp, 150.0, 5.0)):
            res = run_task(
                task=task,
                base_url="http://mock/v1",
                model="test",
                api_key="",
                extra_body={},
                timeout_llm=10,
                timeout_task=10,
                stream=False,
            )
            assert res["passed"] is True
            assert res["wall_ms"] == 150.0
            assert res["proxy_overhead_ms"] == 5.0
            assert res["ttft_total_ms"] is None
            assert res["prefill_ms"] is None
            assert res["decode_ms"] is None
            assert res["decode_tok_s"] is None
            assert res["prompt_tokens"] == 20
            assert res["completion_tokens"] == 10


class TestMetricsReportingAndSampleCounts:
    def test_summarize_and_format_reports(self):
        from bench.run_bench import (
            format_comparison_table,
            format_summary_report,
            summarize_run_results,
        )

        results = [
            {
                "task_id": "t1",
                "category": "exact_fact",
                "passed": True,
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "wall_ms": 500.0,
                "proxy_overhead_ms": 10.0,
                "ttft_total_ms": 100.0,
                "prefill_ms": 90.0,
                "decode_ms": 400.0,
                "decode_tok_s": 125.0,
            },
            {
                "task_id": "t2",
                "category": "continuity",
                "passed": False,
                "prompt_tokens": 200,
                "completion_tokens": 80,
                "wall_ms": 800.0,
                "proxy_overhead_ms": 20.0,
                "ttft_total_ms": 200.0,
                "prefill_ms": 180.0,
                "decode_ms": 600.0,
                "decode_tok_s": 133.33,
            },
        ]

        summary = summarize_run_results(results)
        assert summary["sample_count"] == 2
        assert summary["passed_count"] == 1
        assert summary["pass_at_1_pct"] == 50.0

        cats = summary["categories"]
        assert "exact_fact" in cats
        assert cats["exact_fact"]["formatted"] == "100.0% (1/1, n=1)"
        assert "continuity" in cats
        assert cats["continuity"]["formatted"] == "0.0% (0/1, n=1)"

        lat = summary["latency"]
        assert lat["n_wall"] == 2
        assert lat["n_ttft"] == 2
        assert lat["n_decode"] == 2
        assert lat["mean_wall_ms"] == 650.0
        assert lat["median_wall_ms"] == 650.0
        assert lat["mean_proxy_overhead_ms"] == 15.0
        assert lat["mean_prefill_ms"] == 135.0
        assert lat["mean_decode_ms"] == 500.0

        report = format_summary_report(summary, label="test-run")
        assert "n=2" in report
        assert "n=1" in report
        assert report.isascii()

        raw_results = [
            {
                "task_id": "t1",
                "category": "exact_fact",
                "passed": True,
                "prompt_tokens": 500,
                "completion_tokens": 50,
                "wall_ms": 900.0,
                "proxy_overhead_ms": 0.0,
                "ttft_total_ms": 500.0,
                "prefill_ms": 500.0,
                "decode_ms": 400.0,
                "decode_tok_s": 125.0,
            },
            {
                "task_id": "t2",
                "category": "continuity",
                "passed": False,
                "prompt_tokens": 1000,
                "completion_tokens": 80,
                "wall_ms": 1400.0,
                "proxy_overhead_ms": 0.0,
                "ttft_total_ms": 800.0,
                "prefill_ms": 800.0,
                "decode_ms": 600.0,
                "decode_tok_s": 133.33,
            },
        ]
        raw_summary = summarize_run_results(raw_results)
        table = format_comparison_table(raw_summary, summary)
        assert "Raw Baseline" in table
        assert "CrispLLM" in table or "EdgeMem" in table
        assert "Pass@1" in table
        assert "Prefill / TTFT (ms)" in table
        assert table.isascii()
