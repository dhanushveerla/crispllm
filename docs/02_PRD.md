# Product Requirements Document (PRD) v2.0
## EdgeMem — Lightweight Unified Memory & Context Proxy for Weak-Hardware AI Coding

> Revision of the original "Unified Edge Inference & Hybrid Memory Engine" concept. See `01_EXISTING_SOLUTIONS_AND_POSITIONING.md` for why this version integrates rather than reinvents several components.

---

## 1. Product Overview

| Attribute | Specification |
|---|---|
| **Product Name** | EdgeMem — Unified Memory & Context Proxy |
| **Vision** | Let students, hackathon builders, and resource-constrained developers run AI coding agents efficiently — locally or via free-tier cloud APIs — without re-sending full history/files every turn, and without requiring expensive hardware. |
| **Target Users** | Developers on 8–16 GB RAM laptops, no/weak GPU, using free-tier cloud AI quotas, working across multiple IDEs/CLIs/agents. |
| **Core Differentiator** | A single local proxy that applies tiered memory + AST-aware slicing **under the same tiering policy and token budget** whether the backend is a local quantized model or a cloud API (ChatGPT, Claude, Gemini) — the mechanism differs per tier (see NFR on gist-vector scope), but the token-budget outcome and architecture do not. |

### Honest scope correction from v1.0
- We are **not** claiming "zero accuracy loss" for the memory layer — it is explicitly lossy-for-continuity, exact-for-facts.
- We are **not** building a custom C++20/Rust inference runtime — we integrate llama.cpp.
- We are **not** building AST slicing from scratch — we integrate/reference Aider's repo-map approach.
- We **are** building: the tiered memory manager, the unification proxy, the hardware-aware tuning layer, and the retrieval/eviction policy — the actual gap identified in the landscape review.

## 2. Success Metrics

| Metric | Target |
|---|---|
| Input token reduction per turn | ≥ 80% (from ~4,000+ down to ≤ 500–800) |
| Local generation speedup (where applicable) | 1.5x–3x on supported hardware, via integrated speculative decoding |
| Memory footprint | Engine + 2–3B model + memory state ≤ 3.0 GB peak RSS on CPU-only tier |
| Pass@1 / task fidelity | No measurable regression on exact-fact tasks (via retrieval); documented, bounded degradation (<5%) on continuity-dependent tasks (via gist tier) — **measured, not assumed** |
| Free-tier cloud session longevity | ≥ 3x more turns before hitting rate limits, vs. naive full-context resend |

### Target hardware tiers (benchmarked against in Phase 1 and Phase 4)

| Tier | Profile | Primary validation target |
|---|---|---|
| T1 — CPU floor | Dual-core CPU, 8 GB RAM, no GPU | The ≤3.0GB RSS and metrics above are measured here first; this is the hard floor the product must remain usable on. |
| T2 — CPU mid | Quad-core+ CPU, 16 GB RAM, no discrete GPU | Confirms headroom and multi-file refactor performance once the floor is cleared. |
| T3 — Entry GPU | 4–8 GB VRAM laptop GPU (e.g. 4060-class) + CPU | Validates that EAGLE/Medusa speculative decoding gains materialize once batched verification is available — this is also where the n-gram lookup fallback (see Implementation Plan, Phase 4) gets compared against EAGLE directly. |

T1 is the must-pass tier; T2/T3 are stretch validation to confirm the design doesn't *regress* on better hardware.

## 3. System Architecture (Revised)

```
 USER / IDE / CLI / Browser / Agent
            |
            v
 +-----------------------------+
 | Complexity Router (<5ms)    |  -- classify: trivial / standard / deep
 +-----------------------------+
      |                  |
      v                  v
 +-------------+   +--------------------+
 | Hot Tier    |   | AST / Retrieval    |
 | (verbatim,  |   | Store (facts: sigs,|
 | last N turns|   | diffs, decisions)  |
 +-------------+   +--------------------+
      |                  |
      v                  v
 +-----------------------------+
 | Warm Tier: neural gist       |
 | (local model, pinned) OR     |
 | rolling text summary (cloud) |
 +-----------------------------+
      |
      v
 +-----------------------------+
 | Cold Tier: coarse long-term  |
 | summary (decayed, evicted)   |
 +-----------------------------+
            |
            v
 +-----------------------------+
 | Compact Payload Assembler    |
 +-----------------------------+
      |                  |
      v                  v
 +-------------+   +-------------------+
 | Local llama.|   | Sanitized Cloud   |
 | cpp runtime |   | Proxy (OpenAI/    |
 | (quantized) |   | Anthropic/Gemini  |
 |             |   | schema-compatible)|
 +-------------+   +-------------------+
```

## 4. Functional Requirements

| ID | Feature | Requirement | Priority | Build vs. Integrate |
|---|---|---|---|---|
| FR-01a | Local API/MCP Proxy (OpenAI schema) | HTTP server exposing OpenAI-compatible `/v1/chat/completions` (including streaming tool-call deltas) + MCP endpoints so any IDE/CLI/agent connects without code changes. Cloud backends reachable via their OpenAI-compatible endpoints where available (e.g. Gemini's `/v1beta/openai/`). | **P0 (v1.0)** | **Build** (thin layer) |
| FR-01b | Native Anthropic/Gemini Schema Adapters | Dedicated (non-OpenAI-compatible) Anthropic and Gemini schema translation, for clients that need native-format requests rather than the OpenAI-compatible fallback. | **P1 (v1.1, Phase 4)** | **Build**, deferred |
| FR-02 | AST Context Slicer | Extract only relevant function/file slices (≤30 lines) per turn. | P0 | **Integrate** (reference Aider's repo-map; reimplement minimal subset with Tree-sitter) |
| FR-03 | Complexity Router | Classify prompts <5ms into tiers; route local vs. cloud. | P0 | **Build** (small, novel glue logic) |
| FR-04 | Tiered Memory Manager (`.edgemem.bin`) | Hot (verbatim) / Warm (bounded neural gist, **local-model-pinned only**) / Cold (decayed summary) with recurrent updates and eviction. **When the Complexity Router escalates to a cloud backend, the Warm Tier MUST fall back to text-space compression (structured rolling summary), never the neural gist vectors** — gist vectors live in a specific model's hidden-state space and cannot cross a closed API boundary or transfer to a different model family. | P0 | **Build**, but reuse Letta's memory-tiering logic/patterns rather than reinventing the OS-paging model |
| FR-05 | Retrieval/Fact Store | Exact facts (signatures, diffs, decisions) retrievable on demand, never compressed into the neural tier. | P0 | **Integrate** (simple embedding index — e.g., sqlite + FAISS/sqlite-vec) |
| FR-06 | Hardware-Aware Inference | Quantized local model execution with speculative decoding where hardware supports it. | P1 | **Integrate** (llama.cpp + existing EAGLE/Medusa support) |
| FR-07 | Sanitized Cloud Escalation | Strip secrets/.env before sending compressed capsules to any cloud API. | P1 | **Build** (small, security-critical, must be custom) |
| FR-08 | Benchmark Harness | Before/after telemetry: tokens/turn, tok/s, RSS, Pass@1 on real hardware. | P0 (for validation) | **Build** |

## 5. Non-Functional Requirements

- **Honesty constraint:** No claim of "zero accuracy loss" for the neural gist tier in any documentation or marketing. Claim "bounded-RAM lossy continuity, retrieval-exact facts."
- **Zero mandatory infrastructure:** No required external database server; default to embedded/local storage (SQLite, local files) so a solo developer can run everything on one machine.
- **Hardware floor:** Must remain usable (not necessarily fast) on a dual-core CPU, 8GB RAM, no GPU.
- **Gist-vector scope:** The neural Warm Tier is explicitly scoped to **one pinned local base model**. It is not a cross-backend feature. Any claim of "identical memory behavior across local and cloud" refers to the *tiering architecture and token-budget outcome*, not the neural gist mechanism specifically — the cloud path always uses text-space summarization.
- **License hygiene:** Any integrated component (Letta, Aider, llama.cpp) must be used in compliance with its license; this PRD assumes MIT/Apache-class dependencies only.

## 6. Out of Scope for v1

- Custom C++20/Rust inference runtime (use llama.cpp).
- Cross-model latent-bridge re-encoding (drop entirely; re-run cheap retrieval/summarization instead when switching base models).
- Full multi-agent parallel sandboxing (defer to Phase 4+, hardware-dependent).

## 7. Risks

| Risk | Mitigation |
|---|---|
| Neural gist tier underperforms or misleads users about fidelity | Benchmark explicitly (Pass@1 before/after) before claiming any number; ship with honest framing from day one. |
| Duplicating Letta/Mem0/Aider functionality wastes effort | Treat Phase 0 as an integration spike: try wiring Letta + Aider's repo-map together before writing any custom memory/slicing code. |
| Scope creep into full custom runtime | Explicitly out-of-scope (Section 6); revisit only if llama.cpp proves insufficient. |
| Letta's own runtime overhead (Python + DB + embedding model, plus its internal memory-management tool calls) eats into both the 3GB RAM budget and the rate-limit-longevity success metric | Measure Letta's idle RSS and per-turn internal round-trips during the Phase 0 spike before deciding to depend on it; be ready to borrow only its paging *pattern* in lightweight SQLite if the full framework is too heavy. |
| Provider schema divergence (OpenAI/Anthropic/Gemini streaming tool-call deltas) breaks IDE agent workflows that depend on them | Test streaming tool-call translation explicitly in Phase 1, not just plain chat completions; narrow initial scope to one schema + MCP before adding the other two (see Implementation Plan). |
| Hardware-dependent benchmarks (Peak RSS, speculative-decoding speedup) measured only on Colab's GPU runtime don't transfer to the dual-core CPU target, since CPU speculative decoding is compute-bound rather than memory-bandwidth-bound | Use Colab only for Pass@1/accuracy sweeps and synthetic data generation; measure RSS and speculative-decoding gains on real target-class hardware before publishing any number. |
