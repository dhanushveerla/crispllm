# Implementation Plan — Using Your Current Toolset

## Your resources, mapped to roles

| Tool | What you have | Best role in this project |
|---|---|---|
| **Google Antigravity (Pro)** | Agent-first IDE, Gemini 3.1 Pro (1M token context), multi-agent orchestration (Mission Control), built-in browser automation, MCP support, higher quota than free tier | Primary dev environment: orchestrating multi-file builds, running the benchmark harness end-to-end, browser-based verification of any local web UI/dashboard you build, and testing your MCP server directly (Antigravity is already an MCP host). |
| **OpenCode (free tier)** | Open-source terminal CLI agent, MIT license, BYOK, supports local models, MCP client | Your **reference implementation sandbox** — run it against your in-progress EdgeMem proxy to test the "universal proxy" claim (FR-01a) directly, since it's provider-agnostic by design. Also good for lightweight, fast iteration without Antigravity's IDE overhead. |
| **Google Colab (free)** | Free GPU/TPU runtime, ephemeral | Training/evaluation work: calibrating GGUF importance-matrix (imatrix) quantization, running the Pass@1 benchmark suite with GPU, prototyping the recurrent gist-memory module (small-scale) before running it locally. **Not for hardware-dependent telemetry** (RSS, tok/s, speculative-decoding gains) — those must be measured on real target-class hardware. Not for persistent services — Colab sessions expire. |
| **2nd Google account (Plus/Google AI Pro)** | Higher Gemini API quota, likely Google AI Studio access | Use as your **secondary cloud backend** for testing the Sanitized Cloud Escalation path (FR-07) and for generating synthetic training data (multi-turn coding trajectories) for the gist-memory model without touching your primary account's quota. |

## Phase 0 — Integration Spike (Week 1)
**Goal:** Confirm the "don't rebuild, integrate" plan is viable before writing custom code.

1. In **Antigravity**, clone and run **Letta** locally; point a toy conversation at it and confirm tiered memory (core/recall/archival) behaves as expected on your target low-RAM profile. **Specifically measure:** (a) Letta's idle RSS footprint (its Python runtime + DB layer + embedding model can plausibly add 800MB–1.2GB on top of your 2.0–2.5GB model budget — this alone could blow the 3.0GB ceiling), and (b) how many *extra* internal LLM round-trips Letta's own agent makes per turn to manage its memory, since those calls eat directly into the "3x more turns before rate-limit" success metric. If either number is too high, the decision is to borrow Letta's paging **pattern** in a lightweight SQLite implementation rather than depend on the full framework — don't default to importing it wholesale.
2. In **OpenCode**, try Aider's repo-map approach (or Aider itself) against a sample multi-file repo; confirm AST-based slicing quality is good enough to reuse rather than reimplement.
3. Document findings in a short `INTEGRATION_SPIKE.md`: what Letta/Aider give you for free, what's missing for your 8GB-RAM target.

**Exit criteria:** A go/no-go decision — if Letta + Aider cover 70%+ of FR-02/FR-04, pivot effort to the proxy + hardware-tuning layer only.

## Phase 1 — Benchmark Harness + Proxy MVP (Weeks 2–3)
**Tool split:**
- **Antigravity**: scaffold the local HTTP proxy (FR-01a). **Narrow the real build target**: implement OpenAI-compatible `/v1/chat/completions` + MCP endpoints first, including **streaming tool-call delta translation** (test this explicitly, since OpenCode and Antigravity both depend on streaming tool calls for file edits/terminal execution, and this is where provider schemas diverge most). Add Anthropic and Gemini schema adapters as a follow-up task once the OpenAI path is proven end-to-end — don't try to ship all three schemas inside the two-week window.
- **Colab**: build `bench_tasks.json` (20-task benchmark suite) and run **Pass@1 accuracy sweeps** using a free GPU runtime. For quantization, calibrate **GGUF importance-matrix (imatrix) quantization (Q4_K_M or IQ4_XS)** — not AWQ, which targets GPU-oriented runtimes like vLLM/ExLlama and isn't the right fit for llama.cpp's CPU path. **Caveat: treat Colab's TTFT/tok/s/RSS numbers as accuracy-only reference data, not hardware telemetry** — GPU-measured speculative decoding gains don't transfer to a CPU target, since CPU speculative decoding is typically compute-bound rather than memory-bandwidth-bound and may show little gain or even regress latency. Peak RSS and tok/s must be re-measured on real target-class hardware (see Phase 4).
- **OpenCode**: smoke-test the proxy by pointing OpenCode's BYOK config at `localhost:8080/v1` and confirming a real agent workflow — including streaming tool calls for a file edit — runs through it unmodified.

**Deliverable:** working proxy (OpenAI schema + MCP first) that OpenCode can drive unmodified, a Pass@1 baseline from Colab, and a clear flag that RSS/speed numbers from this phase are provisional until re-measured on target hardware.

## Phase 2 — AST/Retrieval Store + Sanitized Cloud Escalation (Weeks 4–5)
- **Antigravity**: implement the retrieval store (embeddings + Tree-sitter AST index, SQLite-backed) and the secrets-redaction layer (FR-07) — use Antigravity's browser-automation to verify the sanitized proxy visually (e.g., confirm a `.env` value never leaves the sandbox by inspecting outgoing requests).
- **2nd Google account (Plus)**: route escalation-path tests through Gemini via this account's higher quota, so testing doesn't eat into your primary account's limits. Since the Gemini schema adapter is deferred past Phase 1, **use Gemini's OpenAI-compatible endpoint** (`https://generativelanguage.googleapis.com/v1beta/openai/`) for this phase's escalation tests — it speaks the OpenAI schema your proxy already supports, so you get real Gemini-backed escalation testing in Week 4 without writing a custom Gemini translator yet.
- **OpenCode**: run a real multi-file refactor task through the full pipeline (slicer → retrieval → cloud escalation) as an end-to-end integration test.

**Deliverable:** facts are retrieved exactly, secrets never leak, cloud escalation path runs under the same tiering policy and token budget as local (via the OpenAI-compatible Gemini endpoint for now).

## Phase 3 — Tiered Memory + Neural Gist Tier (Weeks 6–8)
**This is the research-heavy phase — scope it as its own workstream, not a side task.**
- **Colab (GPU)**: train the small recurrent gist-compressor (start from the Gist Tokens paper's recipe, not from scratch) on synthetic multi-turn coding trajectories.
- **2nd Google account (Plus)**: use Gemini via Google AI Studio to *generate* the synthetic training trajectories (multi-turn coding conversations) cheaply, since this needs volume, not quality-of-reasoning.
- **Antigravity**: integrate the trained gist module into the warm tier of the memory manager; wire up hot/warm/cold decay and eviction. **Hard constraint: the trained gist vectors are pinned to the exact base model they were trained against** (e.g., a gist model trained for Qwen2.5-3B will not work on Llama-3.2-3B or any cloud API). Implement the Warm Tier's cloud fallback **now, not later** — whenever the Complexity Router escalates to a cloud backend, swap to a text-space rolling summary (plain structured text, not vectors) instead of the neural gist. Build this fallback path first and treat the neural gist as a local-only optimization layered on top, so a cloud-escalation bug can't silently send garbage vectors to an API that can't use them.
- **Benchmark immediately**: run the Phase 1 harness again with the gist tier enabled vs. disabled, on real target hardware — measure the actual Pass@1 delta before claiming any number publicly.

**Deliverable:** working tiered memory with a *measured*, documented fidelity cost, a working text-based cloud fallback, and an explicit "local-model-pinned" label on the neural gist component.

## Phase 4 — Hardware Tuning + Speculative Decoding (Weeks 9–10)
- Integrate llama.cpp with existing EAGLE/Medusa support rather than writing a custom runtime.
- **Target-class local hardware (not Colab — per the Phase 1 caveat, GPU-measured thresholds don't transfer to CPU)**: run the actual threshold-tuning for when to enable speculative decoding vs. fall back to linear decoding, measured on your real dual-core/8GB target. While testing llama.cpp on this tier, also benchmark llama.cpp's built-in **n-gram prompt lookup decoding** (`--spec-type ngram-mod` or equivalent) alongside EAGLE-3 draft heads: n-gram lookup needs zero extra model RAM, though it mainly helps on repetitive/templated edits (e.g., renames, boilerplate) rather than diverse generation — so it's worth having as a cheap fallback mode the router can select for that task type, not a universal replacement for EAGLE.
- **Antigravity**: final integration pass, using Mission Control's multi-agent mode to run build + test + browser-verification steps in parallel.
- **OpenCode**: final end-to-end smoke test as the "neutral third-party client" check — if it works unmodified through OpenCode, your proxy is genuinely provider-agnostic.

**Deliverable:** v1.0 — proxy + tiered memory + AST retrieval + hardware-aware inference, benchmarked across the T1/T2/T3 hardware tiers defined in PRD Section 2, with every claim backed by a number you actually measured (not a Colab-only estimate).

## Sequencing notes
- **Don't start Phase 3 (the hard ML part) until Phases 0–2 are solid.** The integration spike (Phase 0) may remove a large fraction of planned work if Letta/Aider cover more than expected — find that out before investing in custom training.
- **Colab's session limits** mean nothing long-running should live there — use it purely for training/benchmark runs, not as a persistent server.
- **Keep both Google accounts' usage separated by role** (primary = main dev/testing, secondary/Plus = quota-heavy synthetic data generation and escalation-path testing) so you don't hit rate limits on your main account mid-development.
