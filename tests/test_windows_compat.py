from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
from pathlib import Path, PureWindowsPath
import sys
import tomllib
from unittest.mock import MagicMock, patch
import pytest
from click.testing import CliRunner

from crispllm.cli import cli
from crispllm.memory.store import MemoryStore
from crispllm.proxy.mcp_server import is_within_allowed_roots
from crispllm.proxy.sysinfo import PROCESS_MEMORY_COUNTERS, _get_windows_process_rss_mb, get_process_rss_mb
from crispllm.proxy.telemetry import TelemetryStore
from crispllm.slicer.cache import OutlineCache, TagCache
from crispllm.slicer.repomap import RepoMap
from crispllm.slicer.tags import Tag, extract_tags


def test_process_memory_counters_struct_fields():
    fields = dict(PROCESS_MEMORY_COUNTERS._fields_)
    assert fields["cb"] == wintypes.DWORD
    assert fields["PageFaultCount"] == wintypes.DWORD
    assert fields["PeakWorkingSetSize"] == ctypes.c_size_t
    assert fields["WorkingSetSize"] == ctypes.c_size_t
    assert fields["QuotaPeakPagedPoolUsage"] == ctypes.c_size_t
    assert fields["QuotaPagedPoolUsage"] == ctypes.c_size_t
    assert fields["QuotaPeakNonPagedPoolUsage"] == ctypes.c_size_t
    assert fields["QuotaNonPagedPoolUsage"] == ctypes.c_size_t
    assert fields["PagefileUsage"] == ctypes.c_size_t
    assert fields["PeakPagefileUsage"] == ctypes.c_size_t


def test_windows_rss_helper_mocked_and_native():
    if sys.platform == "win32":
        native_rss = get_process_rss_mb()
        assert isinstance(native_rss, float)
        assert native_rss > 0.0

    mock_kernel32 = MagicMock()
    mock_kernel32.GetCurrentProcess.return_value = 12345

    def fake_k32_info(handle, byref_counters, cb):
        target = ctypes.cast(byref_counters, ctypes.POINTER(PROCESS_MEMORY_COUNTERS)).contents
        target.WorkingSetSize = 64 * 1024 * 1024
        return 1

    mock_kernel32.K32GetProcessMemoryInfo = fake_k32_info
    rss = _get_windows_process_rss_mb(kernel32=mock_kernel32)
    assert rss == 64.0
    assert mock_kernel32.GetCurrentProcess.restype == wintypes.HANDLE
    assert fake_k32_info.argtypes == [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    assert fake_k32_info.restype == wintypes.BOOL

    mock_kernel32_fallback = MagicMock(spec=["GetCurrentProcess"])
    mock_kernel32_fallback.GetCurrentProcess.return_value = 54321
    mock_psapi = MagicMock()

    def fake_psapi_info(handle, byref_counters, cb):
        target = ctypes.cast(byref_counters, ctypes.POINTER(PROCESS_MEMORY_COUNTERS)).contents
        target.WorkingSetSize = 32 * 1024 * 1024
        return 1

    mock_psapi.GetProcessMemoryInfo = fake_psapi_info
    rss_fallback = _get_windows_process_rss_mb(kernel32=mock_kernel32_fallback, psapi=mock_psapi)
    assert rss_fallback == 32.0
    assert fake_psapi_info.argtypes == [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    assert fake_psapi_info.restype == wintypes.BOOL


def test_slicer_posix_normalization_and_utf8(tmp_path: Path):
    sub = tmp_path / "pkg" / "nested"
    sub.mkdir(parents=True)

    utf8_file = sub / "café.py"
    utf8_file.write_text(
        "class CaféManager:\n    def brew_espresso(self) -> str:\n        return '☕'\n",
        encoding="utf-8",
    )

    non_utf8_file = sub / "legacy_cp1252.py"
    non_utf8_file.write_bytes(b"# Windows cp1252 source\ndef test_fn(): return '\xe9\xe8\xe0'\n")

    tags_utf8 = extract_tags(utf8_file, "pkg\\nested\\café.py")
    assert len(tags_utf8) > 0
    assert tags_utf8[0].rel_path == "pkg/nested/café.py"

    tags_legacy = extract_tags(non_utf8_file, "pkg\\nested\\legacy_cp1252.py")
    assert isinstance(tags_legacy, list)

    tag = Tag(rel_path="dir\\sub\\file.py", abs_path="/abs/dir/sub/file.py", name="sym", line=1, kind="def")
    assert tag.rel_path == "dir/sub/file.py"

    cache_db = tmp_path / "outline_norm.db"
    with OutlineCache(db_path=cache_db) as oc:
        oc.put("src\\auth.py", 100, 200, "k1", "outline text", 15)
        res = oc.get("src/auth.py", 100, 200, "k1")
        assert res is not None
        assert res[0] == "outline text"

        oc.put("src/auth2.py", 100, 200, "k2", "outline text 2", 20)
        res2 = oc.get("src\\auth2.py", 100, 200, "k2")
        assert res2 is not None
        assert res2[0] == "outline text 2"

    with RepoMap(repo_path=tmp_path, max_tokens=500) as rm:
        repo_map = rm.get_repo_map(focus_files=["pkg\\nested\\café.py"])
        assert "pkg/nested/café.py" in repo_map
        assert "pkg\\nested" not in repo_map


def test_allowed_repo_roots_path_jail(tmp_path: Path):
    root_a = tmp_path / "repo_a"
    root_b = tmp_path / "repo_b"
    root_a.mkdir()
    root_b.mkdir()

    file_a = root_a / "sub" / "file.py"
    file_a.parent.mkdir()
    file_a.write_text("print('hello')\n", encoding="utf-8")

    file_outside = tmp_path / "outside.py"
    file_outside.write_text("print('outside')\n", encoding="utf-8")

    allowed = [str(root_a)]
    assert is_within_allowed_roots(file_a, allowed) is True
    assert is_within_allowed_roots(root_a, allowed) is True
    assert is_within_allowed_roots(file_outside, allowed) is False
    assert is_within_allowed_roots(root_b, allowed) is False
    assert is_within_allowed_roots(file_a, []) is True


def test_stores_clean_sqlite_teardown_and_file_deletion(tmp_path: Path):
    mem_db = tmp_path / "test_teardown_mem.db"
    store = MemoryStore(mem_db)
    store.upsert_fact("sess_td", "PORT", "9000", kind="constant")
    store.close()
    mem_db.unlink()
    assert not mem_db.exists()

    tel_db = tmp_path / "test_teardown_tel.db"
    tel = TelemetryStore(tel_db)
    tel.record(
        route_tier="local",
        route_reason="test",
        model="test-model",
        est_tokens_raw=50,
        est_tokens_sent=50,
        tokens_saved=0,
        compacted_turns=0,
        secrets_redacted=0,
        fallback_used=0,
        latency_ms=5.0,
    )
    tel.close()
    tel_db.unlink()
    assert not tel_db.exists()

    rm_db = tmp_path / "test_teardown_rm.db"
    cache = TagCache(db_path=rm_db)
    rm = RepoMap(repo_path=tmp_path, cache=cache)
    rm.raw_token_count()
    rm.close()
    rm_db.unlink()
    assert not rm_db.exists()


def test_cli_output_pure_ascii_and_init_newline(tmp_path: Path):
    runner = CliRunner()

    res_help = runner.invoke(cli, ["--help"])
    assert res_help.exit_code == 0
    res_help.output.encode("ascii")

    res_serve = runner.invoke(cli, ["serve", "--help"])
    assert res_serve.exit_code == 0
    res_serve.output.encode("ascii")

    cfg_file = tmp_path / "edgemem.toml"
    res_init = runner.invoke(cli, ["init", "--output", str(cfg_file)])
    assert res_init.exit_code == 0
    res_init.output.encode("ascii")

    raw_bytes = cfg_file.read_bytes()
    assert b"\r\n" not in raw_bytes
    assert b"\n" in raw_bytes
    decoded_text = raw_bytes.decode("utf-8")
    assert "[server]" in decoded_text

    res_stats = runner.invoke(cli, ["stats", "--config", str(cfg_file)])
    assert res_stats.exit_code == 0
    res_stats.output.encode("ascii")


def test_toml_windows_path_escaping(tmp_path: Path):
    windows_path_str = r"C:\Users\runneradmin\AppData\Local\Temp\test.db"
    posix_path_str = PureWindowsPath(windows_path_str).as_posix()
    valid_toml = f'db_path = "{posix_path_str}"\n'
    parsed = tomllib.loads(valid_toml)
    assert parsed["db_path"] == "C:/Users/runneradmin/AppData/Local/Temp/test.db"

    local_posix_toml = f'db_path = "{(tmp_path / "test.db").as_posix()}"\n'
    parsed_local = tomllib.loads(local_posix_toml)
    assert "test.db" in parsed_local["db_path"]

    invalid_toml = f'db_path = "{windows_path_str}"\n'
    with pytest.raises(tomllib.TOMLDecodeError):
        tomllib.loads(invalid_toml)


def test_cli_init_windows_environment_hints(tmp_path: Path):
    runner = CliRunner()
    target_win = tmp_path / "win_cfg.toml"
    with patch("sys.platform", "win32"):
        res_win = runner.invoke(cli, ["init", "--output", str(target_win), "--preset", "nvidia"])
        assert res_win.exit_code == 0
        assert '$env:NVIDIA_API_KEY="<your-key>"' in res_win.output
        assert "export NVIDIA_API_KEY" not in res_win.output

    target_linux = tmp_path / "linux_cfg.toml"
    with patch("sys.platform", "linux"):
        res_linux = runner.invoke(cli, ["init", "--output", str(target_linux), "--preset", "nvidia"])
        assert res_linux.exit_code == 0
        assert 'export NVIDIA_API_KEY="<your-key>"' in res_linux.output
        assert "$env:" not in res_linux.output

