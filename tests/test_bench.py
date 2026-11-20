"""
tests/test_bench.py — Grader Self-Test for EdgeMem Benchmark Suite

Verifies:
  1. All 20 tasks have valid schema
  2. Balanced 10/10 category split (exact_fact / continuity)
  3. Every task's reference_solution passes its own test_code (100% Pass@1)
  4. An empty/broken solution fails every test_code (0% Pass@1)
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BENCH_DIR = Path(__file__).parent.parent / "bench"
TASKS_FILE = BENCH_DIR / "bench_tasks.json"
STRESS_TASKS_FILE = BENCH_DIR / "stress_tasks.json"


@pytest.fixture(scope="module")
def tasks() -> list[dict]:
    assert TASKS_FILE.exists(), f"bench_tasks.json not found at {TASKS_FILE}"
    with open(TASKS_FILE) as f:
        data = json.load(f)
    return data


@pytest.fixture(scope="module")
def stress_tasks() -> list[dict]:
    assert STRESS_TASKS_FILE.exists(), f"stress_tasks.json not found at {STRESS_TASKS_FILE}"
    with open(STRESS_TASKS_FILE) as f:
        data = json.load(f)
    return data


# ---------------------------------------------------------------------------
# 1. Schema validation
# ---------------------------------------------------------------------------

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
        assert isinstance(tasks[idx]["id"], str) and tasks[idx]["id"], \
            f"Task {idx} id must be non-empty string"

    @pytest.mark.parametrize("idx", range(20))
    def test_task_category_valid(self, tasks, idx):
        cat = tasks[idx]["category"]
        assert cat in VALID_CATEGORIES, \
            f"Task {idx} has invalid category: {cat!r}"

    @pytest.mark.parametrize("idx", range(20))
    def test_task_files_is_dict(self, tasks, idx):
        files = tasks[idx]["files"]
        assert isinstance(files, dict) and files, \
            f"Task {idx} 'files' must be a non-empty dict"
        for path, content in files.items():
            assert isinstance(path, str) and path, \
                f"Task {idx}: file path must be non-empty string"
            assert isinstance(content, str), \
                f"Task {idx}: file content for {path!r} must be string"

    @pytest.mark.parametrize("idx", range(20))
    def test_task_turns_is_list(self, tasks, idx):
        turns = tasks[idx]["turns"]
        assert isinstance(turns, list) and turns, \
            f"Task {idx} 'turns' must be non-empty list"
        for t in turns:
            assert isinstance(t, str) and t, \
                f"Task {idx}: each turn must be non-empty string"

    @pytest.mark.parametrize("idx", range(20))
    def test_task_reference_solution_is_dict(self, tasks, idx):
        ref = tasks[idx]["reference_solution"]
        assert isinstance(ref, dict) and ref, \
            f"Task {idx} 'reference_solution' must be non-empty dict"
        for path, code in ref.items():
            assert isinstance(path, str) and path
            assert isinstance(code, str)

    @pytest.mark.parametrize("idx", range(20))
    def test_task_test_code_is_string(self, tasks, idx):
        tc = tasks[idx]["test_code"]
        assert isinstance(tc, str) and tc, \
            f"Task {idx} 'test_code' must be non-empty string"

    @pytest.mark.parametrize("idx", range(20))
    def test_task_id_unique(self, tasks, idx):
        ids = [t["id"] for t in tasks]
        task_id = tasks[idx]["id"]
        assert ids.count(task_id) == 1, f"Duplicate task id: {task_id!r}"

    def test_continuity_task_multi_turn(self, tasks):
        """continuity tasks must have >= 4 turns."""
        ct_tasks = [t for t in tasks if t["category"] == "continuity"]
        for t in ct_tasks:
            assert len(t["turns"]) >= 4, \
                f"continuity task {t['id']!r} has only {len(t['turns'])} turns (need >=4)"


# ---------------------------------------------------------------------------
# 2. Category balance
# ---------------------------------------------------------------------------

class TestCategoryBalance:
    def test_exactly_10_exact_fact(self, tasks):
        ef = [t for t in tasks if t["category"] == "exact_fact"]
        assert len(ef) == 10, f"Expected 10 exact_fact tasks, got {len(ef)}"

    def test_exactly_10_continuity(self, tasks):
        ct = [t for t in tasks if t["category"] == "continuity"]
        assert len(ct) == 10, f"Expected 10 continuity tasks, got {len(ct)}"


# ---------------------------------------------------------------------------
# Helpers for running test_code
# ---------------------------------------------------------------------------

def _run_test_in_tmpdir(
    task: dict,
    solution_files: dict[str, str],  # relative path -> python source
    timeout: int = 15,
) -> tuple[bool, str]:
    """
    Write context files + solution files into a temp dir,
    then run task['test_code'] as a subprocess.
    Returns (passed: bool, output: str).
    """
    with tempfile.TemporaryDirectory(prefix="edgemem_selftest_") as tmpdir:
        tmp = Path(tmpdir)

        # Write context files (skeleton / supporting files)
        for rel_path, content in task["files"].items():
            dest = tmp / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content)

        # Overwrite with solution files
        for rel_path, code in solution_files.items():
            dest = tmp / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(code)

        # Write test_code
        test_file = tmp / "_test_runner_.py"
        test_file.write_text(task["test_code"])

        try:
            proc = subprocess.run(
                [sys.executable, str(test_file), str(tmp)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(tmp),
            )
            passed = proc.returncode == 0 and "PASS" in proc.stdout
            output = (proc.stdout + proc.stderr).strip()
            return passed, output
        except subprocess.TimeoutExpired:
            return False, f"timeout after {timeout}s"
        except Exception as exc:
            return False, str(exc)


# ---------------------------------------------------------------------------
# 3. Reference solutions pass their own test_code (100% sanity check)
# ---------------------------------------------------------------------------

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
            f"Output:\n{output[:800]}"
        )


# ---------------------------------------------------------------------------
# 4. Empty/broken solution fails (0% sanity check)
# ---------------------------------------------------------------------------

BROKEN_PYTHON = "# intentionally broken — no implementations\npass\n"


def _make_empty_solution(task: dict) -> dict[str, str]:
    """
    Return solution files filled with no-op stubs.
    For each file in reference_solution, return a file that:
    - defines any class/function names present but with `pass` bodies
    - will almost certainly fail the test assertions
    """
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


# ---------------------------------------------------------------------------
# 5. Overall Pass@1 coverage assertion (meta-test)
# ---------------------------------------------------------------------------

class TestPassAtOneCoverage:
    def test_all_reference_solutions_give_100_pass(self, tasks):
        """Meta-test: iterate all tasks; every reference solution must pass."""
        failures = []
        for task in tasks:
            passed, output = _run_test_in_tmpdir(
                task=task,
                solution_files=task["reference_solution"],
            )
            if not passed:
                failures.append(f"  FAIL: {task['id']!r}\n    {output[:200]}")
        assert not failures, (
            f"Reference solutions did not achieve 100% Pass@1:\n" + "\n".join(failures)
        )

    def test_all_empty_solutions_give_0_pass(self, tasks):
        """Meta-test: empty stubs must fail every task's test_code."""
        unexpected_passes = []
        for task in tasks:
            broken_sol = _make_empty_solution(task)
            passed, _ = _run_test_in_tmpdir(task=task, solution_files=broken_sol)
            if passed:
                unexpected_passes.append(task["id"])
        assert not unexpected_passes, (
            f"Empty solution unexpectedly passed: {unexpected_passes}"
        )


# ---------------------------------------------------------------------------
# 6. Stress tasks suite verification
# ---------------------------------------------------------------------------

class TestStressTasksSuite:
    def test_stress_task_count(self, stress_tasks):
        assert len(stress_tasks) == 6, f"Expected 6 stress tasks, got {len(stress_tasks)}"

    @pytest.mark.parametrize("idx", range(6))
    def test_stress_task_schema(self, stress_tasks, idx):
        task = stress_tasks[idx]
        missing = REQUIRED_KEYS - set(task.keys())
        assert not missing, f"Stress task {idx} missing keys: {missing}"
        assert task["category"] == "stress_continuity", f"Unexpected category {task['category']!r}"
        assert len(task["turns"]) >= 6, f"Expected >= 6 turns, got {len(task['turns'])}"

    @pytest.mark.parametrize("idx", range(6))
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

    @pytest.mark.parametrize("idx", range(6))
    def test_stress_reference_solution_passes(self, stress_tasks, idx):
        task = stress_tasks[idx]
        passed, output = _run_test_in_tmpdir(
            task=task,
            solution_files=task["reference_solution"],
        )
        assert passed, (
            f"Stress task {task['id']!r} reference solution FAILED test_code.\n"
            f"Output:\n{output[:1000]}"
        )

    @pytest.mark.parametrize("idx", range(6))
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
