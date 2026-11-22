"""CrispLLM — Lightweight unified memory & context proxy."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("crispllm")
except PackageNotFoundError:
    __version__ = "1.1.1"
