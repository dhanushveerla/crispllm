# Security Policy

## Supported Versions

CrispLLM maintains active security patches and vulnerability mitigations for the following versions:

| Version | Supported | Notes |
|---|---|---|
| `1.1.x` | :white_check_mark: Yes | Current stable release track |
| `1.0.x` | :x: No | Deprecated; upgrade to `>= 1.1.1` |
| `< 1.0.0` | :x: No | Alpha / pre-release milestones |

---

## Reporting a Vulnerability

If you discover a security vulnerability or potential privacy flaw in CrispLLM:

1. **Private GitHub Security Advisory (Preferred):**
   Submit an advisory report at [GitHub Security Advisories](https://github.com/dhanushveerla/crispllm/security/advisories/new).
2. **Direct Email:**
   Email the maintainer at `dhamansai5@gmail.com` (or `dhanushveerla@gmail.com`) with the subject line `[SECURITY] CrispLLM Vulnerability Report`.
   - Please include reproduction steps, affected versions, and impact analysis.
   - We acknowledge reports within 48 hours and coordinate a coordinated disclosure release within 7 business days.

Please do not open public GitHub issues or public discussions for sensitive security disclosures.

---

## Privacy & Redaction Security Model

CrispLLM includes a deterministic, zero-dependency privacy redaction engine (`crispllm.privacy`) designed to prevent credentials, secrets, and PII from leaking to upstream cloud LLMs.

### What CrispLLM Guarantees

- **Outbound Secret Masking:** Known API key patterns (OpenAI `sk-proj-`, Anthropic `sk-ant-`, Google AI `AIza`, GitHub `ghp_` / `github_pat_`, AWS `AKIA`, Slack `xox*`, JWTs), PEM private keys, database connection URLs with embedded credentials, Bearer tokens, and env assignments are replaced with deterministic opaque placeholders (`__CRISPLLM_*__`) before cloud egress.
- **Inbound Streaming & Non-Streaming Restoration:** Completed responses and live SSE delta chunks restore placeholders back to the caller's original values before returning data to local development tools.
- **Secret-Into-Memory Isolation:** Memory extraction (`crispllm.memory`) operates strictly on redacted message representations. Raw secrets are blocked from entering SQLite storage (`facts`, `episodes`, `memory_fts`).
- **Zero Remote Telemetry:** CrispLLM does not collect, log, or transmit telemetry or prompt contents to external servers. All telemetry is stored in local SQLite databases.

### What CrispLLM Does Not Guarantee

- **Arbitrary Unstructured PII / High-Entropy Strings:** Strings without identifiable token prefixes, structural assignment patterns, or configured `extra_patterns` are not redacted.
- **Adversarial Steganography:** The redactor is a fast deterministic lexical filter (< 2 ms latency) and does not deploy an LLM to de-obfuscate deliberately fragmented or steganographic prompts designed to bypass regexes.
- **Local Loopback Routes:** Requests directed to local backends (e.g., `llama-server` or Ollama on localhost) are not redacted by default unless `redact_on_cloud_only` is set to `false`.
- **Local Host Compromise:** CrispLLM runs with local user permissions and does not sandbox against root or malicious local processes with direct filesystem access to `.crispllm/memory.db`.
