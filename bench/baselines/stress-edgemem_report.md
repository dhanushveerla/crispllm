# EdgeMem Benchmark Report: `stress-edgemem`

| Field | Value |
|---|---|
| Label | `stress-edgemem` |
| Model | `nvidia/nemotron-3.5-bolt-e4b` |
| Endpoint | `http://127.0.0.1:8080/v1` |
| Tasks run | 6 |
| **Pass@1 (overall)** | **50.0%** (3/6) |
| Pass@1 exact_fact | — |
| Pass@1 continuity | — |
| Total prompt tokens | 30,895 |
| Total completion tokens | 5,298 |
| Mean prompt tokens | 5149.2 |
| Mean completion tokens | 883 |
| Mean TTFT (ms) | 10943.2 |
| p95 TTFT (ms) | 20935.8 |
| Mean gen tok/s | 79.68 |
| Total wall time (s) | 65.66 |
| Proxy RSS (MB) | 80.1 |

## Per-Task Results

| Task ID | Category | Pass | Prompt tok | Completion tok | Wall (ms) | Error |
|---|---|---|---|---|---|---|
| stress_01 | stress_continuity | ✅ | 3761 | 398 | 5151 |  |
| stress_02 | stress_continuity | ❌ | 7644 | 999 | 12284 | Traceback (most recent call last):   File "/tmp/edgemem_benc |
| stress_03 | stress_continuity | ❌ | 7613 | 615 | 8801 | Traceback (most recent call last):   File "/tmp/edgemem_benc |
| stress_04 | stress_continuity | ✅ | 4067 | 705 | 8458 |  |
| stress_05 | stress_continuity | ❌ | 3562 | 1509 | 17853 | Traceback (most recent call last):   File "/tmp/edgemem_benc |
| stress_06 | stress_continuity | ✅ | 4248 | 1072 | 13111 |  |
