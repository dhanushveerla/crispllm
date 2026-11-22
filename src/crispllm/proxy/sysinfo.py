from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import re
import subprocess
import sys
from typing import Any


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_uint32),
        ("dwMemoryLoad", ctypes.c_uint32),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    ]


@dataclass
class SystemMemoryInfo:
    total_mb: float
    available_mb: float
    used_percent: float
    source: str


_DARWIN_TOTAL_MEM_BYTES: int | None = None


def _get_linux_system_memory() -> SystemMemoryInfo | None:
    try:
        data: dict[str, float] = {}
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) >= 2:
                    k = parts[0].strip()
                    v_str = parts[1].strip().split()[0]
                    try:
                        data[k] = float(v_str)
                    except ValueError:
                        pass
        total_kb = data.get("MemTotal")
        if total_kb is None or total_kb <= 0:
            return None
        avail_kb = data.get("MemAvailable")
        if avail_kb is None:
            free_kb = data.get("MemFree", 0.0)
            buffers_kb = data.get("Buffers", 0.0)
            cached_kb = data.get("Cached", 0.0)
            sreclaimable_kb = data.get("SReclaimable", 0.0)
            avail_kb = free_kb + buffers_kb + cached_kb + sreclaimable_kb

        total_mb = round(total_kb / 1024.0, 2)
        available_mb = round(avail_kb / 1024.0, 2)
        used_percent = round(((total_mb - available_mb) / total_mb) * 100.0, 2) if total_mb > 0 else 0.0
        return SystemMemoryInfo(
            total_mb=total_mb,
            available_mb=available_mb,
            used_percent=used_percent,
            source="proc_meminfo",
        )
    except Exception:
        return None


def _get_windows_system_memory(kernel32: Any = None) -> SystemMemoryInfo | None:
    try:
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        if kernel32 is None:
            kernel32 = ctypes.windll.kernel32
        fn = getattr(kernel32, "GlobalMemoryStatusEx")
        fn.argtypes = [ctypes.c_void_p]
        fn.restype = wintypes.BOOL
        ret = fn(ctypes.byref(stat))
        if not ret:
            return None
        total_mb = round(float(stat.ullTotalPhys) / (1024.0 * 1024.0), 2)
        available_mb = round(float(stat.ullAvailPhys) / (1024.0 * 1024.0), 2)
        used_percent = round(float(stat.dwMemoryLoad), 2)
        return SystemMemoryInfo(
            total_mb=total_mb,
            available_mb=available_mb,
            used_percent=used_percent,
            source="win32_global_memory_status",
        )
    except Exception:
        return None


def _get_darwin_system_memory(total_mem_bytes: int | None = None) -> SystemMemoryInfo | None:
    global _DARWIN_TOTAL_MEM_BYTES
    try:
        if total_mem_bytes is not None:
            total_bytes = total_mem_bytes
        else:
            if _DARWIN_TOTAL_MEM_BYTES is None:
                res = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    timeout=1.5,
                    capture_output=True,
                    text=True,
                )
                if res.returncode == 0 and res.stdout.strip():
                    _DARWIN_TOTAL_MEM_BYTES = int(res.stdout.strip())
                else:
                    return None
            total_bytes = _DARWIN_TOTAL_MEM_BYTES

        vm_res = subprocess.run(
            ["vm_stat"],
            timeout=1.5,
            capture_output=True,
            text=True,
        )
        if vm_res.returncode != 0 or not vm_res.stdout:
            return None

        page_size = 4096
        free_pages = 0
        inactive_pages = 0
        speculative_pages = 0

        for line in vm_res.stdout.splitlines():
            line_str = line.strip()
            if "page size of" in line_str:
                match = re.search(r"page size of\s+([0-9]+)\s+bytes", line_str)
                if match:
                    page_size = int(match.group(1))
            elif line_str.startswith("Pages free:"):
                val = line_str.split(":")[1].strip().rstrip(".")
                free_pages = int(val)
            elif line_str.startswith("Pages inactive:"):
                val = line_str.split(":")[1].strip().rstrip(".")
                inactive_pages = int(val)
            elif line_str.startswith("Pages speculative:"):
                val = line_str.split(":")[1].strip().rstrip(".")
                speculative_pages = int(val)

        avail_bytes = (free_pages + inactive_pages + speculative_pages) * page_size
        total_mb = round(float(total_bytes) / (1024.0 * 1024.0), 2)
        available_mb = round(float(avail_bytes) / (1024.0 * 1024.0), 2)
        used_percent = round(((total_mb - available_mb) / total_mb) * 100.0, 2) if total_mb > 0 else 0.0
        return SystemMemoryInfo(
            total_mb=total_mb,
            available_mb=available_mb,
            used_percent=used_percent,
            source="darwin_vm_stat",
        )
    except Exception:
        return None


def get_system_memory_info(
    kernel32: Any = None, total_mem_bytes: int | None = None
) -> SystemMemoryInfo | None:
    try:
        plat = sys.platform
        if plat.startswith("linux"):
            return _get_linux_system_memory()
        if plat == "win32":
            return _get_windows_system_memory(kernel32=kernel32)
        if plat == "darwin":
            return _get_darwin_system_memory(total_mem_bytes=total_mem_bytes)
        return None
    except Exception:
        return None



def _get_windows_process_rss_mb(kernel32: Any = None, psapi: Any = None) -> float:
    try:
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        if kernel32 is None:
            kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        handle = kernel32.GetCurrentProcess()
        try:
            fn = getattr(kernel32, "K32GetProcessMemoryInfo")
        except AttributeError:
            if psapi is None:
                psapi = ctypes.windll.psapi
            fn = getattr(psapi, "GetProcessMemoryInfo")
        fn.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        fn.restype = wintypes.BOOL
        ret = fn(handle, ctypes.byref(counters), counters.cb)
        if ret:
            return float(counters.WorkingSetSize) / (1024.0 * 1024.0)
        return 0.0
    except Exception:
        return 0.0


def get_process_rss_mb() -> float:
    try:
        platform = sys.platform
        if platform == "darwin":
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF)
            return float(usage.ru_maxrss) / (1024.0 * 1024.0)

        if platform == "win32":
            return _get_windows_process_rss_mb()

        try:
            with open("/proc/self/status", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            return float(parts[1]) / 1024.0
        except Exception:
            pass

        try:
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF)
            return float(usage.ru_maxrss) / 1024.0
        except Exception:
            pass

        return 0.0
    except Exception:
        return 0.0
