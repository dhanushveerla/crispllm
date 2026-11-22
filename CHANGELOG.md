# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.1] - 2026-11-23

### Fixed
- **PyPI PEP 740 Provenance Attestations**: Updated GitHub Actions publishing workflow (`.github/workflows/publish.yml`) to use `pypa/gh-action-pypi-publish@release/v1` with `attestations: true` and OIDC trusted publishing, ensuring PyPI displays verified digital provenance on releases.
- **Secret-Into-Memory Isolation Gate**: Enforced that `MemoryMiddleware` redacts messages before invoking fact and constraint extraction (`extract_facts_from_messages`). Sanitized secrets and placeholders are blocked from entering SQLite tables (`facts`, `episodes`, `memory_fts`) across both non-streaming JSON and SSE streaming lifecycles.
- **Darwin Memory Mock Isolation**: Isolated cached total RAM global in `get_system_memory_info()` to ensure deterministic test execution across Linux, macOS, and Windows CI runners.

### Added
- **Security Policy (`SECURITY.md`)**: Published supported version lifecycle (`1.1.x`), private vulnerability reporting instructions, and explicit guarantees/limitations of the privacy redaction model.
- **Competitive Comparison Matrix**: Added an honest architectural comparison in `README.md` contrasting CrispLLM against Mem0, Letta (MemGPT), and Aider, clarifying memory tradeoffs (low RAM vs. vector fuzzy search).

### Changed
- Bumped package version to `1.1.1`.
- Added inline sample sizes (`n=3`, `n=20`) and hardware tier context to all empirical benchmark tables in `README.md`.
- Documented deprecation of legacy `EdgeMem` compatibility shims scheduled for removal in `v1.3.0`.

## [1.1.0] - 2026-11-22

### Added
- **3-OS Platform Parity & Continuous Integration (Phase 1.1.0)**:
  - Full native support for Windows, macOS, and Linux across all subpackages.
  - Windows memory measurement via `ctypes` Win32 `K32GetProcessMemoryInfo` with zero extra dependencies.
  - POSIX-normalized AST path hashing to eliminate cache collisions across operating systems.
  - GitHub Actions matrix testing on `ubuntu-latest`, `macos-latest`, and `windows-latest`.
- **3-Tier Synthetic Memory & Async Consolidation (Phase 1.1.1)**:
  - Deterministic SQLite WAL memory hierarchy with Core, Episodic, and Archival tiers.
  - BM25 full-text search (FTS5) retrieval with time-decay recency scoring.
  - Background asynchronous consolidation worker pruning superseded and low-salience facts.
  - Deterministic secret-sanitization gate preventing raw secrets or credentials from entering SQLite storage.
- **Active RAM/VRAM Reduction & Dynamic Context Budgeting (Phase 1.1.2)**:
  - `RamPressureWatchdog`: Standard library 3-OS memory pressure monitor (`normal`, `soft`, `hard`) using `/proc/meminfo` on Linux, `sysctl` on macOS, and Win32 `GlobalMemoryStatusEx` on Windows.
  - `ContextBudgeter`: Unified ADR-03 context controller managing dynamic `num_ctx` bucketing (`[2048, 4096, 8192]`) and derived routing escalation thresholds.
  - `IdleUnloadWorker`: Async idle watchdog automatically releasing local model weights after inactivity (`keep_alive: 0` on Ollama, slot release on `llama-server`).
  - Verified on Lenovo LOQ Windows 11 hardware: Peak System RSS capped under 2.74 GB with steady-state proxy RSS under 81 MB.
- **1-Hop AST Symbol Slicing & Prefix-Cache Prompt Assembly (Phase 1.1.3)**:
  - 1-hop AST symbol and call-graph slicer (`crispllm slice --symbol`, MCP `focus_symbol`): warm lookup in 47 ms on the CrispLLM repository with 99.7% token compression.
  - Zero-LLM task-tier heuristic router (< 0.1 ms latency, 0 LLM calls) classifying requests into `fast`, `balanced`, or `deep`.
  - Prefix-cache-stable prompt assembly: consolidated `messages[0]` system prompt, byte-stable Core facts, user Episodic suffix, and session prefix tracking (`X-CrispLLM-Prefix-Stable: hit|miss`).
  - Verified Lenovo LOQ TTFT reduction: 2,441 ms -> 800 ms (3.05x speedup, -67.2% latency, 3/3 cache hits).
- **Speculative Token Generation & Preset Engine (Phase 1.1.4)**:
  - Automatic Cloud Predicted Outputs injection with edit-intent detection and strict parameter gating (`tools`, `n > 1`, `logprobs`, `presence_penalty`, `response_format`).
  - Automatic fail-open circuit breaker on HTTP 400/422 with immediate fallback retry and cached provider exclusion.
  - `llama-server` speculative presets: `ngram-simple`, `ngram-mod`, `draft`, and `hybrid` supporting current llama.cpp `b12631+` CLI flags via `crispllm llama-cmd`.
  - Verified Lenovo LOQ speculative speedups on Qwen2.5-Coder-3B Q4_K_M: T1-CPU 11.64 -> 35.71 tok/s (3.07x, 84.4% acceptance, 0 MB extra RAM) and T2-Vulkan 29.02 -> 82.52 tok/s (2.84x, 84.3% acceptance, 0 MB extra RAM).
- **Expanded Deterministic Stress Benchmark & Hardening Audit (Phase 1.1.5)**:
  - Long-context stress benchmark suite expanded from `n=6` to `n=20` deterministic multi-turn engineering tasks across 4 balanced categories: `constraint_retention`, `contradiction_update`, `noisy_distractor`, and `cross_turn_synthesis` (5 tasks each).
  - Deterministic offline `--dry-run` execution mode in `bench/run_bench.py`.
  - Complete fail-open hardening across middleware, watchdog, classifier, speculative engine, and streaming estimators.
  - Two-job automated PyPI Trusted Publishing workflow using `uv build` and `uv publish`.

### Changed
- Bumped package version from `1.0.0` to `1.1.0`.
- Synchronized CLI `--version` and MCP `serverInfo.version` to single source of truth in `crispllm.__version__`.
- Updated PyPI distribution metadata, URLs, keywords, and classifiers in `pyproject.toml`.

## [1.0.0] - 2026-11-20

### Added
- Local-first OpenAI-compatible `/v1/chat/completions` proxy on port 8080.
- Streamable HTTP Model Context Protocol (MCP) server at `/mcp`.
- Deterministic SQLite FTS5 memory store with exact constraint tracking.
- Algorithmic middle-turn conversation compaction for multi-turn sessions.
- Reversible zero-dependency secret and PII redaction engine with SSE stream restoration.
- Repository AST mapping and PageRank symbol prioritization (`crispllm.slicer`).
- Proactive token-threshold routing and automatic local-to-cloud fallback on 5xx/429/404.
- Benchmark suite (`bench/run_bench.py`) with telemetry logging.
- Bearer token authentication gate for local and remote access.
