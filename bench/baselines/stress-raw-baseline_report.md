# EdgeMem Benchmark Report: `stress-raw-baseline`

| Field | Value |
|---|---|
| Label | `stress-raw-baseline` |
| Model | `nvidia/nemotron-3.5-bolt-e4b` |
| Endpoint | `https://integrate.api.nvidia.com/v1` |
| Tasks run | 6 |
| **Pass@1 (overall)** | **16.7%** (1/6) |
| Pass@1 exact_fact | — |
| Pass@1 continuity | — |
| Total prompt tokens | 43,099 |
| Total completion tokens | 5,754 |
| Mean prompt tokens | 7183.2 |
| Mean completion tokens | 959 |
| Mean TTFT (ms) | 13789.3 |
| p95 TTFT (ms) | 21172.4 |
| Mean gen tok/s | 70.63 |
| Total wall time (s) | 82.74 |
| Proxy RSS (MB) | — |

## Per-Task Results

| Task ID | Category | Pass | Prompt tok | Completion tok | Wall (ms) | Error |
|---|---|---|---|---|---|---|
| stress_01 | stress_continuity | ❌ | 7050 | 335 | 4383 | Traceback (most recent call last):   File "/tmp/edgemem_benc |
| stress_02 | stress_continuity | ❌ | 7233 | 1458 | 19784 | Traceback (most recent call last):   File "/tmp/edgemem_benc |
| stress_03 | stress_continuity | ❌ | 7256 | 942 | 11627 | Traceback (most recent call last):   File "/tmp/edgemem_benc |
| stress_04 | stress_continuity | ❌ | 7058 | 956 | 17018 | Traceback (most recent call last):   File "/tmp/edgemem_benc |
| stress_05 | stress_continuity | ❌ | 7180 | 1275 | 17648 | Traceback (most recent call last):   File "/tmp/edgemem_benc |
| stress_06 | stress_continuity | ✅ | 7322 | 788 | 12277 |  |
