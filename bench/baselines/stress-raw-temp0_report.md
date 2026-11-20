# EdgeMem Benchmark Report: `stress-raw-temp0`

| Field | Value |
|---|---|
| Label | `stress-raw-temp0` |
| Model | `nvidia/nemotron-3.5-bolt-e4b` |
| Endpoint | `https://integrate.api.nvidia.com/v1` |
| Temperature | `0.0` |
| Tasks run | 6 |
| **Pass@1 (overall)** | **16.7%** (1/6) |
| Pass@1 exact_fact | — |
| Pass@1 continuity | — |
| Total prompt tokens | 43,099 |
| Total completion tokens | 5,683 |
| Mean prompt tokens | 7183.2 |
| Mean completion tokens | 947.2 |
| Mean TTFT (ms) | 16540.7 |
| p95 TTFT (ms) | 29459.7 |
| Mean gen tok/s | 64.57 |
| Total wall time (s) | 99.24 |
| Proxy RSS (MB) | — |

## Per-Task Results

| Task ID | Category | Pass | Prompt tok | Completion tok | Wall (ms) | Error |
|---|---|---|---|---|---|---|
| stress_01 | stress_continuity | ❌ | 7050 | 375 | 25637 | Traceback (most recent call last):   File "/tmp/edgemem_benc |
| stress_02 | stress_continuity | ❌ | 7233 | 1004 | 12455 | Traceback (most recent call last):   File "/tmp/edgemem_benc |
| stress_03 | stress_continuity | ❌ | 7256 | 1203 | 19757 | Traceback (most recent call last):   File "/tmp/edgemem_benc |
| stress_04 | stress_continuity | ❌ | 7058 | 1144 | 18229 | Traceback (most recent call last):   File "/tmp/edgemem_benc |
| stress_05 | stress_continuity | ❌ | 7180 | 776 | 9329 | Traceback (most recent call last):   File "/tmp/edgemem_benc |
| stress_06 | stress_continuity | ✅ | 7322 | 1181 | 13837 |  |
