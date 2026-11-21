from __future__ import annotations

import ctypes
from ctypes import wintypes
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
