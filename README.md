# EdgeMem

> Lightweight, unified memory & context proxy for AI coding on resource-constrained hardware.

EdgeMem is a local-first, zero-bloat OpenAI-compatible LLM proxy and Model Context Protocol (MCP) server. It equips small local coding models (Ollama, vLLM, llama.cpp) and cloud APIs with tiered architectural memory, automatic middle-turn context compaction, reversible zero-dependency secret redaction, and intelligent local-to-cloud escalation — maintaining a **sub-85 MB EdgeMem proxy RSS footprint (not including local LLM weights)** with **zero vector databases, zero embedding models, and zero extra LLM calls** on the request hot path.

### Supported Platforms
- **Tier 1 (Fully Supported)**: **Linux and macOS (Python 3.11+)**
- **Experimental**: Windows (WorkingSetSize RSS measurement via `ctypes` Win32 API)

---

## Client Compatibility

EdgeMem provides two interfaces on port 8080: an OpenAI-compatible `/v1` proxy and a Streamable HTTP `/mcp` server.

| Client Category | Example Clients / Tools | `/v1` OpenAI Proxy<br>*(Memory, Compaction, Redaction, Routing)* | `/mcp` Server<br>*(AST `repo_map`, `remember`, `recall`)* | Integration Notes |
|---|---|---|---|---|
| **Coding Agents** | OpenCode, Cline, Roo Code, Continue, Aider | **Full Support** | **Full Support** | Dual integration: set LLM `base_url` to `/v1` and register `/mcp` server. |
| **Desktop IDEs & Apps** | Cursor, Windsurf, Claude Desktop, Claude Code | **Partial / Optional** | **Full Support** | Full MCP tool access; `/v1` proxy supported where custom OpenAI base URLs can be set. |
| **Python / JS SDKs** | Official `openai` Python/Node SDKs, LangChain, LlamaIndex | **Full Support** | N/A | Drop-in `base_url="http://127.0.0.1:8080/v1"` with arbitrary `api_key`. |
| **Hosted Web AIs** | Claude.ai Web, ChatGPT Web, Manus | **Unsupported** *(Web UIs cannot route completions through a local `/v1` proxy)* | **Full Support** *(Remote HTTPS MCP)* | Expose `/mcp` via secure HTTPS tunnel with Bearer auth; `/v1` stays blocked by default. |

---

## Request Lifecycle

```
  ┌─────────────────────────────────────────────────────────┐
  │            Client (OpenCode / Cursor / curl)            │
  └────────────────────────────┬────────────────────────────┘
                               │ POST /v1/chat/completions
                               ▼
  ┌─────────────────────────────────────────────────────────┐
  │      Memory & Threshold Compaction (SQLite FTS5)        │
  │  • Session-isolated constraints & architectural facts   │
  │  • Algorithmic middle-turn boilerplate compaction       │
  └────────────────────────────┬────────────────────────────┘
                               │
                               ▼
  ┌─────────────────────────────────────────────────────────┐
  │          Reversible Secret / PII Redactor               │
  │  • High-entropy API keys, auth tokens & passwords       │
  │  • In-memory session placeholder restoration map        │
  └────────────────────────────┬────────────────────────────┘
                               │
                               ▼
  ┌─────────────────────────────────────────────────────────┐
  │            Hybrid Local-to-Cloud Router                 │
  │  • Proactive prompt token threshold check               │
  │  • Automatic cloud escalation on local 5xx / 429 / 404  │
  └──────────────┬───────────────────────────┬──────────────┘
                 │                           │
           (Local Ollama)               (Cloud NIM)
                 │                           │
                 └─────────────┬─────────────┘
                               │ SSE / JSON Stream
                               ▼
  ┌─────────────────────────────────────────────────────────┐
  │                 SSE Stream Restorer                     │
  │  • Pass-through SSE delta chunk streaming               │
  │  • Replaces redaction placeholders with real secrets    │
  └────────────────────────────┬────────────────────────────┘
                               │
                               ▼
  ┌─────────────────────────────────────────────────────────┐
  │            Client (OpenCode / Cursor / curl)            │
  └─────────────────────────────────────────────────────────┘
```

---

## Verified Benchmarks

All metrics are measured empirically using `bench/run_bench.py` and committed in `bench/baselines/`.

### 1. Standard Suite (`n=20`, `nvidia/nemotron-3.5-bolt-e4b`)
Evaluation across 20 synthetic coding tasks (10 exact constraint adherence + 10 conversational continuity):

| Metric | Raw Baseline (`phase1b`) | EdgeMem Proxy (`phase2b`) | Delta / Impact |
|---|---|---|---|
| **Pass@1 (Overall, n=20)** | **75.0%** (15/20) | **80.0%** (16/20) | **+5.0 pp** |
| Pass@1 exact_fact (n=10) | 90.0% (9/10) | 90.0% (9/10) | Parity |
| Pass@1 continuity (n=10) | 60.0% (6/10) | 70.0% (7/10) | **+10.0 pp** |
| Total Prompt Tokens | 8,056 | 9,004 | Injected architectural facts |
| Total Completion Tokens | 5,867 | 5,056 | -13.8% tokens generated |
| Total Wall Time | 73.24 s | 61.08 s | **-16.6% faster** |
| Mean TTFT | 3,662.2 ms | 3,054.1 ms | **-608.1 ms latency** |
| Mean Generation tok/s | 78.89 | 80.45 | Consistent throughput |
| **EdgeMem Proxy RSS** | — | **79.7 MB** | **< 85 MB target (proxy only)** |

### 2. Long-Context Stress Suite (`n=6`, `temperature=0.0`, ~7.2k tokens/task)
Evaluation under multi-turn dialogue with large middle-turn code boilerplate:

| Metric | Raw Baseline (`stress-raw-temp0`) | EdgeMem Proxy (`stress-edgemem-temp0`) | Delta / Impact |
|---|---|---|---|
| **Pass@1 (Overall, n=6)** | **16.7%** (1/6) | **50.0%** (3/6) | **3x improvement** |
| Total Prompt Tokens | 43,099 | 5,195 | **-87.9% reduction** (-37,904 tokens) |
| Mean Prompt Tokens | 7,183.2 | 865.8 | -87.9% prompt overhead |
| Total Completion Tokens | 5,683 | 4,908 | Compact, accurate answers |
| Total Wall Time | 99.24 s | 65.16 s | **-34.3% faster** (-34.08 s) |
| Mean TTFT | 16,540.7 ms | 10,860.3 ms | **-5,680.4 ms TTFT** |
| Mean Generation tok/s | 64.57 | 75.97 | +17.7% generation speed |
| **EdgeMem Proxy RSS** | — | **81.8 MB** | **< 85 MB target (proxy only)** |

While context compaction slashes prompt tokens by -87.9% during prefill, total wall-time savings reach -34.3% because autoregressive output generation and baseline network latency remain constant across turns. See [docs/ROADMAP.md](file:///home/dhanush/projects/edgemem/docs/ROADMAP.md) for full prefill-vs-decode latency analysis, hardware tiering, and ADRs.

### 3. AST Context Slicer (Flask Repository)
Tree-sitter symbol extraction, PageRank centrality ranking, and incremental AST outline caching:

| Metric | Full Repository | EdgeMem Sliced RepoMap | Delta / Impact |
|---|---|---|---|
| Total Tokens | 264,845 | 1,021 | **99.6% compression** |
| In-Process MCP Latency (warm) | — | 130 ms | Sub-second symbol navigation |
| Runtime Memory Load (warm) | — | Zero `tiktoken` RAM | Lazy encoding & disk-backed cache |

---

## 60-Second Quickstart

### 1. Installation

Requires Python 3.11+. Install using `uv`:

```bash
git clone https://github.com/example/edgemem.git
cd edgemem
uv sync
```

### 2. Initialize Configuration

Generate an `edgemem.toml` file tailored to your stack:

```bash
# Preset choices: hybrid (default), local, nvidia, gemini
uv run edgemem init --preset hybrid
```

### 3. Start the Proxy Server

```bash
uv run edgemem serve
```
EdgeMem proxy listens on `http://127.0.0.1:8080/v1` and hosts an MCP endpoint at `http://127.0.0.1:8080/mcp`.

### 4. Connect Your Client

#### OpenCode (`opencode.json`)
Point OpenCode directly to the EdgeMem proxy and register the MCP server:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "api_base": "http://127.0.0.1:8080/v1",
  "model": "qwen2.5-coder:7b",
  "mcp_servers": {
    "edgemem": {
      "type": "http",
      "url": "http://127.0.0.1:8080/mcp"
    }
  }
}
```

#### Cursor / Cline MCP (`~/.cursor/mcp.json` or Cline Settings)
Register EdgeMem's Streamable HTTP MCP endpoint:

```json
{
  "mcpServers": {
    "edgemem": {
      "url": "http://127.0.0.1:8080/mcp"
    }
  }
}
```

#### Claude Desktop (`claude_desktop_config.json`)
```json
{
  "mcpServers": {
    "edgemem": {
      "url": "http://127.0.0.1:8080/mcp"
    }
  }
}
```

#### Standard OpenAI Python SDK
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="qwen2.5-coder:7b",
    messages=[
        {"role": "system", "content": "You are an expert systems programmer."},
        {"role": "user", "content": "Explain WAL mode in SQLite."},
    ],
)
print(response.choices[0].message.content)
```

#### Connecting Web AIs via Remote MCP (Claude Web, ChatGPT Web, Manus)
Hosted web applications cannot route completions through a local `/v1` proxy, but they can connect directly to EdgeMem's MCP tools via a secure HTTPS tunnel:

1. **Activate Bearer Authentication**:
   Export your secret token in your shell environment (or set `auth_token` in `edgemem.toml`):
   ```bash
   export EDGEMEM_AUTH_TOKEN="your-secure-token-here"
   ```

2. **Configure Allowed Repository Roots (Optional Jail)**:
   Restrict file access in `edgemem.toml` to prevent traversal outside your designated workspace:
   ```toml
   [security]
   allowed_repo_roots = ["/home/user/projects/my-repo"]
   ```

3. **Start the EdgeMem Proxy**:
   ```bash
   uv run edgemem serve
   ```

4. **Expose the Port via HTTPS Tunnel**:
   Using Cloudflare Tunnel (`cloudflared`):
   ```bash
   cloudflared tunnel --url http://127.0.0.1:8080
   ```

5. **Register MCP in the Web Client**:
   - **Server URL**: `https://<tunnel-subdomain>.trycloudflare.com/mcp`
   - **HTTP Header**: `Authorization: Bearer your-secure-token-here`

> [!NOTE]
> By default, `allow_remote_v1 = false` blocks any requests to `/v1/*` carrying tunnel/proxy headers (`x-forwarded-for`, `cf-connecting-ip`, etc.) with HTTP 403 Forbidden. This ensures an HTTPS tunnel opened for `/mcp` cannot inadvertently expose your paid upstream LLM endpoints.

---

## CLI Reference

| Command | Options / Arguments | Description |
|---|---|---|
| `edgemem init` | `[--preset hybrid\|local\|nvidia\|gemini]`<br>`[--output PATH]`<br>`[--force]` | Generate a ready-to-run `edgemem.toml` config file. Refuses to overwrite existing files unless `--force` is provided. |
| `edgemem serve` | `[--host HOST]`<br>`[--port PORT]`<br>`[--config PATH]` | Start the EdgeMem OpenAI-compatible proxy and MCP Streamable HTTP server. |
| `edgemem status` | `[--config PATH]` | Query proxy `/health` to check upstream status, memory fact counts, privacy redactions, and current RSS. |
| `edgemem stats` | `[--days N]`<br>`[--json]`<br>`[--reset]`<br>`[--config PATH]` | Read persistent SQLite request telemetry. Displays total requests, local offload %, context savings, privacy counts, and mean latency. |
| `edgemem slice` | `<PATH>`<br>`[--tokens N]`<br>`[--query TEXT]`<br>`[--focus FILE]`<br>`[--model ENC]` | Parse a codebase with tree-sitter, compute PageRank centrality, and emit a compact AST repo map to stdout. |
| `edgemem memory` | `list [--kind KIND] [--limit N]`<br>`search <QUERY> [--limit N]`<br>`clear [--session SESS]` | Inspect, BM25 search, or purge stored architectural facts in SQLite memory. |

---

## Reproducing the Benchmarks

All benchmarks run via `bench/run_bench.py` against any OpenAI-compatible endpoint.

### 1. Standard 20-Task Suite
In one terminal, start EdgeMem:
```bash
uv run edgemem serve
```

In a second terminal, execute the benchmark:
```bash
python bench/run_bench.py \
  --base-url http://127.0.0.1:8080/v1 \
  --model nvidia/nemotron-3.5-bolt-e4b \
  --label phase2b-final-standard \
  --health-url http://127.0.0.1:8080/health \
  --tasks bench/bench_tasks.json
```

### 2. 6-Task Long-Context Stress Suite
Evaluate EdgeMem under long multi-turn dialogue with deterministic sampling (`temperature=0.0`):
```bash
python bench/run_bench.py \
  --base-url http://127.0.0.1:8080/v1 \
  --model nvidia/nemotron-3.5-bolt-e4b \
  --label stress-edgemem-temp0 \
  --health-url http://127.0.0.1:8080/health \
  --tasks bench/stress_tasks.json \
  --temperature 0.0
```

To run the raw unproxied baseline directly against the cloud upstream:
```bash
python bench/run_bench.py \
  --base-url https://integrate.api.nvidia.com/v1 \
  --model nvidia/nemotron-3.5-bolt-e4b \
  --label stress-raw-temp0 \
  --api-key-env NVIDIA_API_KEY \
  --tasks bench/stress_tasks.json \
  --temperature 0.0
```

---

## Architecture & Roadmap

See [docs/ROADMAP.md](file:///home/dhanush/projects/edgemem/docs/ROADMAP.md) for Architecture Decision Records (ADR-01, ADR-02, ADR-03), PRD FR-01 through FR-08 traceability, and hardware tier benchmarks.

---

## License

MIT License (Copyright (c) 2026 Dhanush Veerla). See [LICENSE](file:///home/dhanush/projects/edgemem/LICENSE) for details.
