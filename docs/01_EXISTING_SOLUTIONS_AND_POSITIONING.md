# Existing Solutions & Where EdgeMem Actually Fits

## TL;DR
**No part of this idea is novel in isolation.** Every mechanism (gist-vector memory, AST-aware context slicing, speculative decoding, quantization) already exists as mature, often open-source, software. **What doesn't exist yet** is a single lightweight tool that combines them specifically for *very* weak hardware (dual-core CPU, 8GB RAM) and unifies local + cloud AI behind one proxy. That's the actual opportunity — not reinventing memory systems, but integrating proven pieces for an underserved hardware tier.

---

## 1. Memory / context compression — already solved by several projects

| Project | What it does | Relevance |
|---|---|---|
| **MemGPT / Letta** (UC Berkeley, arXiv:2310.08560) | Treats the context window as OS virtual memory: core memory (fixed scratchpad), recall memory (paged in on demand), archival memory (long-term). Recursively summarizes evicted turns. | This **is** the tiered hot/warm/cold architecture we designed — already built, open source, production-ready as the Letta framework. |
| **Mem0** | Extracts and retrieves salient facts rather than storing raw transcripts. Published LoCoMo benchmark: ~90%+ token cost reduction, 91% lower p95 latency vs. full context. | Proves the "facts via retrieval, not compression" half of our design at scale. |
| **Gist Tokens** (Mu et al., NeurIPS 2023) | The actual research behind compressing a prompt into a small number of learned "gist" vectors that a frozen LLM can condition on. | This is the real academic precedent for the PRD's "64 vectors" idea — it's a known technique with a real training recipe, not something to invent from scratch. |
| **ReadAgent, SelfMem** (research) | Gist memory with selective lookup of full original text when detail is needed. | Validates the hot/warm split conceptually. |
| **LLMLingua, memahead/Headroom, ContextBudget** | Token-level and plan-aware prompt compression, budget-constrained. | Could be used as a library rather than rebuilt. |
| **StreamingLLM, H2O** | KV-cache eviction strategies for bounded GPU memory during generation. | Relevant to the "shrink the KV cache" claim in the original doc — already implemented in inference engines. |

**Conclusion:** Don't build a new memory architecture from scratch. Either (a) use Letta/Mem0 directly as the memory backend, or (b) build a thin, hardware-aware wrapper that's *tuned for extremely constrained devices* in a way these general-purpose tools aren't optimized for.

## 2. AST-aware context slicing — already a standard feature

Aider, Cursor, Continue, Cline, and Sourcegraph Cody all already implement "send only the relevant function/file slice, not the whole repo." Aider in particular is open source, terminal-based, and already does repo-map + AST-aware context selection extremely well.

**Conclusion:** Don't rebuild this. Either integrate Aider's repo-map approach directly, or treat it as a solved reference implementation to learn from.

## 3. Local inference efficiency — already mature libraries

- **llama.cpp** — quantization (GGUF), CPU/GPU dispatch (AVX2, CUDA, Metal, Vulkan) — exactly what the original doc described building from scratch.
- **AWQ** — activation-aware quantization, already implemented and widely used.
- **EAGLE / Medusa** — speculative decoding draft heads, already implemented in several inference stacks (vLLM, some llama.cpp forks).

**Conclusion:** This layer is "integrate, don't reinvent." Writing a custom C++20/Rust runtime from scratch (as the original PRD proposed) is almost certainly wasted effort — llama.cpp already does this, is actively maintained, and supports the exact hardware dispatch the doc wanted.

## 4. The actual gap

Nobody currently ships **one coherent, lightweight tool** that:
1. Runs comfortably on *very* weak hardware (dual-core CPU, 8GB RAM) — most memory/agent tooling above assumes a server, a database, or at least a decent GPU.
2. Unifies **local model + any cloud AI (ChatGPT, Claude, Gemini)** behind the same local proxy, so the memory/slicing benefits apply no matter which backend answers the prompt.
3. Is simple enough for a solo developer or student to self-host with zero infrastructure (no vector DB server, no cloud dependency for the memory layer itself).

This is a legitimate, narrower product position: **"Letta-style tiered memory + Aider-style AST slicing, packaged as a single local proxy, tuned for the weakest hardware the existing tools don't optimize for."**

## 5. Recommendation

- **Build vs. integrate:** Prefer integrating Letta (memory) and Aider's repo-map (slicing) as libraries/reference implementations where licensing allows, rather than rewriting them. Spend original engineering effort on the **hardware-tuning, unification layer, and the proxy** — that's the actual gap.
- **Naming/marketing honesty:** Don't claim "first of its kind" — claim "the first to package this specific combination for 8GB-RAM-class hardware," which is defensible and still compelling.
- **Differentiation to lead with:** bounded RAM guarantee on commodity hardware + the same tiering policy and token budget whether the backend is a local 2–3B model or a free-tier cloud API (the memory *mechanism* differs per tier — see PRD FR-04 — but the architecture and outcome don't).
