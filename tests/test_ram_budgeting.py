from __future__ import annotations

import asyncio
import ctypes
from ctypes import wintypes
import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

from edgemem.config import EdgeMemConfig, ResourcesConfig
from edgemem.proxy.app import create_app
from edgemem.proxy.budgeter import ContextBudgeter, detect_local_backend
from edgemem.proxy.sysinfo import (
    MEMORYSTATUSEX,
    SystemMemoryInfo,
    get_system_memory_info,
)
from edgemem.proxy.unloader import IdleUnloadWorker, get_ollama_generate_url
from edgemem.proxy.watchdog import RamPressureWatchdog


def test_sysinfo_linux_proc_meminfo() -> None:
    meminfo_content = (
        "MemTotal:       16384000 kB\n"
        "MemFree:         4096000 kB\n"
        "MemAvailable:    8192000 kB\n"
        "Buffers:          512000 kB\n"
        "Cached:          2048000 kB\n"
    )

    with patch("sys.platform", "linux"):
        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", MagicMock(return_value=io.StringIO(meminfo_content))):
                info = get_system_memory_info()
                assert info is not None
                assert info.total_mb == 16000.0
                assert info.available_mb == 8000.0
                assert info.used_percent == 50.0
                assert info.source == "proc_meminfo"


def test_sysinfo_linux_proc_meminfo_fallback_available() -> None:
    meminfo_content = (
        "MemTotal:        8192000 kB\n"
        "MemFree:         2048000 kB\n"
        "Buffers:         1024000 kB\n"
        "Cached:          1024000 kB\n"
    )

    with patch("sys.platform", "linux"):
        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", MagicMock(return_value=io.StringIO(meminfo_content))):
                info = get_system_memory_info()
                assert info is not None
                assert info.total_mb == 8000.0
                assert info.available_mb == 4000.0
                assert info.used_percent == 50.0
                assert info.source == "proc_meminfo"


def test_sysinfo_windows_memorystatusex_struct() -> None:
    stat = MEMORYSTATUSEX()
    assert ctypes.sizeof(stat) == 64
    stat.dwLength = ctypes.sizeof(stat)
    stat.dwMemoryLoad = 45
    stat.ullTotalPhys = 8 * 1024 * 1024 * 1024
    stat.ullAvailPhys = 4 * 1024 * 1024 * 1024

    mock_kernel32 = MagicMock()

    def fake_status(byref_stat: Any) -> int:
        target = byref_stat._obj
        target.dwMemoryLoad = 50
        target.ullTotalPhys = 8 * 1024 * 1024 * 1024
        target.ullAvailPhys = 4 * 1024 * 1024 * 1024
        return 1

    mock_kernel32.GlobalMemoryStatusEx.side_effect = fake_status

    with patch("sys.platform", "win32"):
        info = get_system_memory_info(kernel32=mock_kernel32)
        assert info is not None
        assert info.total_mb == 8192.0
        assert info.available_mb == 4096.0
        assert info.used_percent == 50.0
        assert info.source == "win32_global_memory_status"


def test_sysinfo_darwin_vm_stat() -> None:
    vm_stat_output = (
        "Mach Virtual Memory Statistics: (page size of 4096 bytes)\n"
        "Pages free:                         100000.\n"
        "Pages active:                       500000.\n"
        "Pages inactive:                      50000.\n"
        "Pages speculative:                   50000.\n"
        "Pages wired down:                   300000.\n"
    )

    with patch("sys.platform", "darwin"):
        with patch("subprocess.run") as mock_run:
            mock_sysctl = MagicMock()
            mock_sysctl.returncode = 0
            mock_sysctl.stdout = "8589934592\n"

            mock_vm = MagicMock()
            mock_vm.returncode = 0
            mock_vm.stdout = vm_stat_output

            def side_effect(cmd: list[str], **kwargs: Any) -> MagicMock:
                if "sysctl" in cmd:
                    return mock_sysctl
                return mock_vm

            mock_run.side_effect = side_effect
            info = get_system_memory_info()
            assert info is not None
            assert info.total_mb == 8192.0
            expected_avail = (200000 * 4096) / (1024.0 * 1024.0)
            assert info.available_mb == pytest.approx(expected_avail, rel=1e-3)
            assert info.source == "darwin_vm_stat"


def test_sysinfo_fail_open_returns_none() -> None:
    with patch("sys.platform", "unknown_os"):
        info = get_system_memory_info()
        assert info is None


def test_watchdog_pressure_hysteresis() -> None:
    cfg = ResourcesConfig(
        enabled=True,
        soft_free_mb=1000.0,
        hard_free_mb=500.0,
        hysteresis_mb=100.0,
        watchdog_poll_s=1.0,
    )

    simulated_available = 2000.0
    current_time = 100.0

    def mock_probe() -> SystemMemoryInfo:
        return SystemMemoryInfo(
            total_mb=8000.0,
            available_mb=simulated_available,
            used_percent=((8000.0 - simulated_available) / 8000.0) * 100.0,
            source="mock",
        )

    def mock_clock() -> float:
        return current_time

    dog = RamPressureWatchdog(cfg, probe_fn=mock_probe, clock=mock_clock)

    snap = dog.snapshot()
    assert snap.pressure == "normal"
    assert snap.memory is not None

    simulated_available = 800.0
    snap_cached = dog.snapshot()
    assert snap_cached.pressure == "normal"

    current_time += 1.5
    snap = dog.snapshot()
    assert snap.pressure == "soft"

    simulated_available = 400.0
    current_time += 1.5
    snap = dog.snapshot()
    assert snap.pressure == "hard"

    simulated_available = 550.0
    current_time += 1.5
    snap = dog.snapshot()
    assert snap.pressure == "hard"

    simulated_available = 650.0
    current_time += 1.5
    snap = dog.snapshot()
    assert snap.pressure == "soft"

    simulated_available = 1050.0
    current_time += 1.5
    snap = dog.snapshot()
    assert snap.pressure == "soft"

    simulated_available = 1150.0
    current_time += 1.5
    snap = dog.snapshot()
    assert snap.pressure == "normal"


def test_watchdog_fail_open_and_disabled() -> None:
    cfg = ResourcesConfig(enabled=False)
    dog = RamPressureWatchdog(cfg)
    assert dog.snapshot().pressure == "normal"

    cfg_enabled = ResourcesConfig(enabled=True)
    dog_none = RamPressureWatchdog(cfg_enabled, probe_fn=lambda: None)
    assert dog_none.snapshot().pressure == "normal"


def test_context_budgeter_bucketing() -> None:
    cfg = EdgeMemConfig()
    cfg.resources.enabled = True
    cfg.resources.dynamic_num_ctx = True
    cfg.resources.ctx_buckets = [2048, 4096, 8192]
    cfg.resources.max_local_ctx = 8192
    cfg.resources.default_output_tokens = 512

    budgeter = ContextBudgeter(cfg)

    d1 = budgeter.evaluate(
        prompt_tokens=500,
        requested_max_tokens=None,
        pressure="normal",
        cloud_available=True,
        local_backend="ollama",
    )
    assert d1.selected_num_ctx == 2048
    assert d1.pressure_ctx_cap == 8192
    assert d1.should_escalate_cloud is False

    d2 = budgeter.evaluate(
        prompt_tokens=2500,
        requested_max_tokens=512,
        pressure="normal",
        cloud_available=True,
        local_backend="ollama",
    )
    assert d2.selected_num_ctx == 4096
    assert d2.should_escalate_cloud is False

    d3 = budgeter.evaluate(
        prompt_tokens=5000,
        requested_max_tokens=512,
        pressure="normal",
        cloud_available=True,
        local_backend="ollama",
    )
    assert d3.selected_num_ctx == 8192
    assert d3.should_escalate_cloud is False


def test_context_budgeter_hard_pressure_escalates_and_unloads() -> None:
    cfg = EdgeMemConfig()
    cfg.resources.enabled = True
    cfg.resources.hard_min_ctx = 2048
    budgeter = ContextBudgeter(cfg)

    dec = budgeter.evaluate(
        prompt_tokens=300,
        requested_max_tokens=256,
        pressure="hard",
        cloud_available=True,
        local_backend="ollama",
    )
    assert dec.should_escalate_cloud is True
    assert dec.escalate_reason == "ram_pressure_hard"
    assert dec.trigger_local_unload is True


def test_context_budgeter_hard_pressure_local_fallback() -> None:
    cfg = EdgeMemConfig()
    cfg.resources.enabled = True
    cfg.resources.hard_min_ctx = 2048
    cfg.resources.downshift_model = "qwen2.5-coder:0.5b"
    budgeter = ContextBudgeter(cfg)

    dec = budgeter.evaluate(
        prompt_tokens=300,
        requested_max_tokens=256,
        pressure="hard",
        cloud_available=False,
        local_backend="ollama",
    )
    assert dec.should_escalate_cloud is False
    assert dec.selected_num_ctx == 2048
    assert dec.downshift_model == "qwen2.5-coder:0.5b"
    assert dec.trigger_local_unload is False


def test_context_budgeter_soft_pressure_caps() -> None:
    cfg = EdgeMemConfig()
    cfg.resources.enabled = True
    cfg.resources.soft_max_ctx = 2048
    cfg.resources.downshift_model = "qwen2.5-coder:1.5b"
    budgeter = ContextBudgeter(cfg)

    dec = budgeter.evaluate(
        prompt_tokens=400,
        requested_max_tokens=256,
        pressure="soft",
        cloud_available=True,
        local_backend="ollama",
    )
    assert dec.should_escalate_cloud is False
    assert dec.pressure_ctx_cap == 2048
    assert dec.selected_num_ctx == 2048
    assert dec.downshift_model == "qwen2.5-coder:1.5b"


def test_context_budgeter_backend_detection() -> None:
    assert detect_local_backend("http://localhost:11434/v1", "auto") == "ollama"
    assert detect_local_backend("http://127.0.0.1:11434", "auto") == "ollama"
    assert detect_local_backend("http://localhost:8080/v1", "auto") == "openai"
    assert detect_local_backend("http://localhost:11434", "llama_server") == "llama_server"


def test_get_ollama_generate_url() -> None:
    assert get_ollama_generate_url("http://localhost:11434/v1") == "http://localhost:11434/api/generate"
    assert get_ollama_generate_url("http://127.0.0.1:11434") == "http://127.0.0.1:11434/api/generate"


@pytest.mark.asyncio
async def test_idle_unload_worker_execution() -> None:
    cfg = EdgeMemConfig()
    cfg.resources.idle_unload_s = 2.0
    cfg.model = "qwen2.5-coder:1.5b"

    mock_client = MagicMock()
    mock_post = MagicMock()
    fake_resp = MagicMock()
    fake_resp.status_code = 200

    future: asyncio.Future[Any] = asyncio.Future()
    future.set_result(fake_resp)
    mock_post.return_value = future
    mock_client.post = mock_post

    worker = IdleUnloadWorker(cfg, http_client=mock_client)
    worker.record_local_activity("qwen2.5-coder:1.5b")

    success = await worker.unload_now()
    assert success is True
    assert worker.unload_count == 1
    mock_post.assert_called_once()
    call_args, call_kwargs = mock_post.call_args
    assert call_kwargs["json"] == {"model": "qwen2.5-coder:1.5b", "keep_alive": 0}


def test_proxy_e2e_ollama_options_injection(tmp_path: Path) -> None:
    cfg = EdgeMemConfig()
    cfg.memory.db_path = str(tmp_path / "test_budget.db")
    cfg.telemetry_dir = str(tmp_path / "telemetry")
    cfg.upstream_base_url = "http://localhost:11434/v1"
    cfg.model = "qwen2.5-coder:1.5b"
    cfg.resources.enabled = True
    cfg.resources.dynamic_num_ctx = True
    cfg.resources.ctx_buckets = [2048, 4096, 8192]

    app = create_app(cfg)

    with patch("httpx.AsyncClient.post") as mock_post:
        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.headers = {"content-type": "application/json"}
        fake_response.json.return_value = {
            "id": "chatcmpl-test",
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        async def fake_post_async(*args: Any, **kwargs: Any) -> Any:
            return fake_response

        mock_post.side_effect = fake_post_async

        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen2.5-coder:1.5b",
                    "messages": [{"role": "user", "content": "ping"}],
                },
            )
            assert resp.status_code == 200
            assert resp.headers["X-EdgeMem-Route"] == "local"

            assert mock_post.called
            call_kwargs = mock_post.call_args[1]
            sent_json = call_kwargs["json"]
            assert "options" in sent_json
            assert sent_json["options"]["num_ctx"] == 2048

            health_resp = client.get("/health")
            assert health_resp.status_code == 200
            health_data = health_resp.json()
            assert "system_memory" in health_data
            assert "resources" in health_data
            assert health_data["resources"]["local_backend"] == "ollama"
            assert health_data["resources"]["effective_max_ctx"] == 8192
            assert health_data["resources"]["last_num_ctx"] == 2048


def test_proxy_e2e_cloud_options_stripping(tmp_path: Path) -> None:
    cfg = EdgeMemConfig()
    cfg.memory.db_path = str(tmp_path / "test_cloud_strip.db")
    cfg.telemetry_dir = str(tmp_path / "telemetry")
    cfg.upstream_base_url = "http://localhost:11434/v1"
    cfg.model = "qwen2.5-coder:1.5b"
    cfg.cloud.enabled = True
    cfg.cloud.base_url = "http://mock-cloud/v1"
    cfg.cloud.model = "cloud-model"
    cfg.routing.escalate_token_threshold = 0

    app = create_app(cfg)

    with patch("httpx.AsyncClient.post") as mock_post:
        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.headers = {"content-type": "application/json"}
        fake_response.json.return_value = {
            "id": "chatcmpl-cloud",
            "choices": [{"message": {"role": "assistant", "content": "Cloud reply"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        async def fake_post_async(*args: Any, **kwargs: Any) -> Any:
            return fake_response

        mock_post.side_effect = fake_post_async

        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen2.5-coder:1.5b",
                    "messages": [{"role": "user", "content": "ping"}],
                    "options": {"num_ctx": 4096},
                },
            )
            assert resp.status_code == 200
            assert resp.headers["X-EdgeMem-Route"] == "cloud"

            assert mock_post.called
            call_kwargs = mock_post.call_args[1]
            sent_json = call_kwargs["json"]
            assert "options" not in sent_json or "num_ctx" not in sent_json.get("options", {})
