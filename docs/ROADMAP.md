# EdgeMem Architecture Roadmap & ADR Traceability

## 1. Architecture Decision Records (ADRs)

### ADR-01: Deterministic 3-Tier SQLite Synthetic Memory vs. Neural Gist Compressor (FR-04 Pivot)
- **Status:** Approved & Implemented (v1.0.0)
- **Context:**
  The original FR-04 specification envisioned a trained recurrent neural gist compressor converting earlier conversation turns into continuous latent vectors stored in `.edgemem.bin`.
- **Decision:**
  Pivot from a trained recurrent neural gist compressor to a deterministic 3-tier synthetic memory hierarchy (Core, Episodic, Archival) backed by embedded SQLite WAL mode with BM25 FTS5 retrieval, salience-decay scoring, and background asynchronous consolidation.
- **Rationale:**
  1. **Cross-Model Independence:** Neural gist vectors are inherently pinned to the hidden-state geometry and tokenizer of a single specific model checkpoint. They cannot cross model boundaries, survive local-to-cloud escalation, or transfer across model upgrades. Deterministic text-space synthetic memory remains 100% LLM-agnostic.
  2. **Auditability & Safety:** Synthetic facts and episodes stored as human-readable text can be inspected, searched, edited, or purged via CLI and MCP tools (`edgemem memory`, `edgemem_recall`). Latent vectors are opaque black boxes prone to uninspectable hallucination drift.
  3. **Zero GPU/Training Overhead:** Training and hosting recurrent neural compressors requires substantial GPU VRAM and specialized execution kernels, directly violating our resource-constrained target hardware budget.
  4. **Empirical Verification:** The 3-tier synthetic architecture achieved +5.0 pp overall Pass@1 improvement on the standard benchmark suite (`n=20`) and 3x Pass@1 improvement (16.7% -> 50.0%) on the long-context stress suite (`n=6`).

### ADR-02: Embedded SQLite FTS5 & Minimal Tree-Sitter vs. External Frameworks (Phase 0 Spike)
- **Status:** Approved & Implemented (Phase 0 / v1.0.0)
- **Context:**
  Phase 0 explored integrating existing external stacks: Letta/Mem0 for memory management and full `aider-chat` for AST repository mapping.
- **Decision:**
  Build custom, zero-external-dependency embedded SQLite memory (`edgemem.memory`) and vendor a minimal Tree-sitter + NetworkX AST repo-mapper (`edgemem.slicer`), rejecting standalone Letta/Mem0 server runtimes and full Aider package dependencies.
- **Rationale:**
  1. **Strict RAM Ceiling:** On the PRD T1 target (<= 3.0 GB total system RSS), a local 2-3B model consumes ~2.05 GB RSS, leaving < 950 MB for the proxy, OS caches, and client IDEs. Letta requires Docker/PostgreSQL/embedding services drawing 800–1,200 MB RSS. Custom SQLite draws < 15 MB RSS, keeping the entire EdgeMem proxy under 85 MB RSS.
  2. **Latency & Longevity:** Letta introduces 2–3 internal LLM round-trips per turn for memory maintenance, which would add 60–120s of delay on CPU hardware and rapidly exhaust free-tier cloud quotas.
  3. **Decoupled Architecture:** `aider-chat` carries over 100 dependencies (`scipy`, `numpy`, `sounddevice`, `pillow`, etc.) tightly bound to its terminal UI loop. Extracting the core PageRank AST ranking logic directly requires only 5 packages and eliminates ~103 unneeded dependencies.

### ADR-03: Unified Dynamic Context Budgeting Between Router and Controller (Phase 1.1.2)
- **Status:** Approved & Implemented (v1.1.2)
- **Context:**
  Currently, `HybridRouter` calculates prompt token escalation thresholds using a static configuration (`escalate_token_threshold`), while the runtime context window can be adjusted separately via the `num_ctx` controller.
- **Decision:**
  Implement `ContextBudgeter` which dynamically derives `HybridRouter`'s escalation threshold directly from the active `num_ctx` controller, establishing a single source of truth for the context budget.
- **Rationale:**
  1. Eliminates configuration drift where prompt compaction threshold and local model context exhaustion cutoffs diverge.
  2. Ensures that whenever `num_ctx` is dynamically constrained under memory pressure, `HybridRouter` escalates to cloud backends before local KV-cache overflows occur.

---

## 2. Functional Requirements (FR) Traceability Matrix

| Requirement ID | Description | Status | Implementing Module(s) | Verification Evidence |
|---|---|---|---|---|
| **FR-01a** | OpenAI-compatible proxy (`/v1`) & Streamable HTTP MCP server (`/mcp`) | Done | `edgemem.proxy.app`, `edgemem.proxy.mcp_server`, `edgemem.proxy.auth` | `tests/test_proxy.py`, `tests/test_security.py` |
| **FR-01b** | Native Anthropic & Gemini schema adapters | Planned (1.1.x) | `edgemem.proxy.adapters` | Schema mapping specifications in PRD Section 4 |
| **FR-02** | AST context slicer with PageRank symbol ranking & 1-hop AST call-graph slicer (`focus_symbol`) | Done (v1.1.3) | `edgemem.slicer.repomap`, `edgemem.slicer.symbols`, `edgemem.slicer.pagerank`, `edgemem.slicer.tags` | `tests/test_slicer.py`, `tests/test_phase_1_1_3.py`, Flask 83-file benchmark (99.2% compression) |
| **FR-03** | Complexity & hybrid router with Zero-LLM task-tier routing (`fast`/`balanced`/`deep`) | Done (v1.1.3) | `edgemem.proxy.router`, `edgemem.proxy.classifier` | `tests/test_router.py`, `tests/test_phase_1_1_3.py` |
| **FR-04** | Tiered memory manager & compaction | Pivoted & Done | `edgemem.memory.store`, `edgemem.proxy.middleware`, `edgemem.memory.consolidator` | `tests/test_synthetic_memory.py`, `tests/test_memory_integration.py` |
| **FR-05** | Exact fact store with prefix-cache-stable prompt assembly | Done (v1.1.3) | `edgemem.memory.store`, `edgemem.proxy.middleware`, `edgemem.proxy.app` | `tests/test_synthetic_memory.py`, `tests/test_phase_1_1_3.py` |
| **FR-06** | Hardware-aware inference tuning & active RAM/VRAM reduction | Done (v1.1.2) | `edgemem.proxy.watchdog`, `edgemem.proxy.budgeter`, `edgemem.proxy.unloader`, `edgemem.proxy.sysinfo` | `tests/test_ram_budgeting.py`, verified Lenovo LOQ benchmarks |
| **FR-07** | Reversible secret & PII redaction gate | Done | `edgemem.privacy.redactor`, `edgemem.privacy.stream_restorer` | `tests/test_privacy.py`, `tests/test_synthetic_memory.py` |
| **FR-08** | Reproducible benchmark harness & telemetry | Done | `bench/run_bench.py`, `edgemem.proxy.telemetry` | `bench/baselines/`, `tests/test_telemetry.py` |
| **FR-09** | Speculative token generation (Cloud Predicted Outputs & local llama-server presets) | Done (v1.1.4) | `edgemem.proxy.speculative`, `edgemem.proxy.app`, `edgemem.cli` | `tests/test_phase_1_1_4.py` |

---

## 3. Metric Precision, Hardware Tiering & Physics of Latency

### Hardware Validation Tiers
- **PRD Total System Peak RSS Ceiling:** `<= 3.0 GB` (encompassing operating system baseline, local quantized 2–3B LLM, and EdgeMem proxy runtime).
- **EdgeMem Proxy-Only RSS:** Measured empirically at **79.7 MB – 81.8 MB** (well below the 85.0 MB proxy-specific allocation target). Full system RSS with local Ollama runtime will be validated end-to-end in Phase 1.1.2.
- **Hardware Tier Definitions:**
  - **T1 Floor (Target):** Dual-core CPU-only, 8 GB RAM, zero GPU. (Pinned validation using 2-core CPU pinning scheduled for Phase 1.1.2).
  - **T2 Mid (Empirical Testbed):** Lenovo LOQ `83FQ` (Intel Core i5, 7.7 GB usable RAM, Intel Arc A530M 4 GB VRAM). Reclassified accurately as T2 mid-tier validation.
  - **T3 High:** Entry discrete GPU (>= 8 GB VRAM) with speculative decoding validation.

### Lenovo LOQ Benchmark Measurements (Phase 1.1.2 & 1.1.3)
Measurements conducted on Windows 11 Home, Lenovo LOQ `83FQ` (Intel Core i5-12450HX, 7.7 GB usable RAM, Intel Arc A530M 4 GB VRAM):
- **T1-Bloated (Unbudgeted Baseline):** Unconstrained Ollama (`num_ctx=8192` default KV cache) with `qwen2.5-coder:1.5b` reached Peak System RSS of **4.18 GB** (~54% of physical RAM), breaching the <= 3.0 GB constraint during multi-turn generation.
- **T1-Budgeted (ContextBudgeter + Watchdog):** Dynamic context bucketing (`ctx_buckets=[2048, 4096, 8192]`) and watchdog-driven context capping reduced Peak System RSS to **2.74 GB** (well within the <= 3.0 GB ceiling). EdgeMem proxy RSS held steady at **80.4 MB**.
- **T2-Vulkan (Intel Arc A530M Offload):** Offloading weights to the discrete GPU utilized **1.82 GB VRAM**, reducing Host System RAM consumption of the local model to **1.08 GB**.
- **Idle Auto-Unload:** Idle unloader worker triggered after configured timeout (`idle_unload_s = 300.0`), issuing `keep_alive: 0` to release model weights and returning system available RAM to baseline within 2.1 seconds.
- **Phase 1.1.3 Prefix-Caching Benchmarks:**
  - **T1-CPU:** Latency reduced from **2441.0 ms -> 800.2 ms** (a **3.05x speedup / -67.2% reduction**), with cached tokens increasing from `139 -> 814` tokens.
  - **T2-Vulkan (Arc A530M):** Latency reduced from **2469.5 ms -> 773.4 ms** (a **3.19x speedup / -68.7% reduction**), with cached tokens increasing from `139 -> 812` tokens and `3/3` prefix cache hits.
  - **AST Symbol Slicing (`slice_symbol`):** Warm execution latency of **47 ms** on the EdgeMem repository with **99.7% token compression** relative to full repository dump.

### Sample Sizes & Pass@1 Benchmarks
- **Standard Suite:** `n=20` synthetic coding tasks (10 exact-fact constraints + 10 conversational continuity). Measured: Raw 75.0% (15/20) vs. EdgeMem 80.0% (16/20), delta +5.0 pp.
- **Stress Suite:** `n=6` multi-turn long-context tasks (~7.2k tokens/task, temperature=0.0). Measured: Raw 16.7% (1/6) vs. EdgeMem 50.0% (3/6), delta 3x improvement. The suite will expand to `n >= 20` tasks prior to v1.1.0 general release.

### Physics of Prompt Reduction vs. Wall-Time Delta
On the long-context stress benchmark (`n=6`), prompt tokens decreased by **-87.9%** (from 43,099 to 5,195 tokens), whereas total cloud wall-time decreased by **-34.3%** (from 99.24s to 65.16s).
This delta reflects LLM inference physics:
1. **Prefill Phase Acceleration:** Middle-turn context compaction slashes prompt evaluation (prefill) latency by eliminating boilerplate tokens before they reach the model attention heads.
2. **Decode Phase Invariance:** Autoregressive generation (token-by-token decode) consumes the majority of wall-clock time and runs at a fixed tokens/second throughput regardless of prompt brevity.
3. **Network Baseline:** Base HTTP round-trip latency and cloud provider queuing remain constant across both compressed and uncompressed payloads.

---

## 4. Milestone Schedule

- **v1.0.0 (Current):** Unified proxy, deterministic 3-tier memory, reversible secret redaction, PageRank AST repo map, and MCP server.
- **v1.1.0 (Upcoming):** Expansion of stress benchmark suite to `n >= 20`, native Gemini adapter, and dynamic context budgeting.
- **v1.1.2 (Hardware Integration & RAM Budgeting):** Unified `ContextBudgeter`, 3-OS RAM-pressure watchdog, and idle model auto-unload. Complete.
- **v1.1.3 (AST Symbol Slicing, Task-Tier Routing & Prefix Caching):**
  - **1-Hop AST Symbol & Call-Graph Slicer:** `SymbolDef` + `SymbolSliceResult` extraction with `mtime`-cached symbols; 0-hop body (~55%), 1-hop callees (~25%), 1-hop callers (~20%) budgeting with graceful query repo-map fallback. Exposed via CLI `edgemem slice --symbol` and MCP `edgemem_repo_map(focus_symbol=...)`.
  - **Zero-LLM Task-Tier Expert Routing:** Deterministic heuristic classifier (< 0.1 ms latency, 0 LLM calls) assigning `fast`, `balanced`, or `deep` tiers; composed into `HybridRouter` with explicit route/watchdog/budgeter precedence and `X-EdgeMem-Task-Tier` header.
  - **Prefix-Cache-Stable Prompt Assembly:** Consolidated `messages[0]` system prompt + byte-stable Core facts sorted by ID, turn Episodic facts attached to latest user turn under `<edgemem_context>`, SHA-256 session prefix tracking (`X-EdgeMem-Prefix-Stable: hit|miss`), and `llama_server` cache_prompt injection. Complete.
- **v1.1.4 (Speculative Token Generation & Preset Engine):**
  - **Cloud Predicted Outputs Injection:** Deterministic edit-intent classification and candidate extraction for OpenAI-compatible cloud endpoints. Incompatible parameter gating (`tools`, `n > 1`, `logprobs`, `presence_penalty`, `response_format`).
  - **Fail-Open Circuit Breaker:** Automatic retry on HTTP 400/422 with stripped prediction and 1800s TTL provider caching.
  - **llama-server Speculative Presets:** Exact llama.cpp b12631+ CLI support for `ngram-simple`, `ngram-mod`, `draft`, and `hybrid` with automated RAM-pressure safety guards and OS-safe shell command generation (`edgemem llama-cmd`).
  - **Speculative Acceptance Telemetry:** Tracking cloud and local speculative acceptance in `/health` and `edgemem status`. Complete.
