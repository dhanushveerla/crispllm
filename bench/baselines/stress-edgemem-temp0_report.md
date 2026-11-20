# EdgeMem Benchmark Report: `stress-edgemem-temp0`

| Field | Value |
|---|---|
| Label | `stress-edgemem-temp0` |
| Model | `nvidia/nemotron-3.5-bolt-e4b` |
| Endpoint | `http://127.0.0.1:8080/v1` |
| Temperature | `0.0` |
| Tasks run | 6 |
| **Pass@1 (overall)** | **50.0%** (3/6) |
| Pass@1 exact_fact | — |
| Pass@1 continuity | — |
| Total prompt tokens | 5,195 |
| Total completion tokens | 4,908 |
| Mean prompt tokens | 865.8 |
| Mean completion tokens | 818 |
| Mean TTFT (ms) | 10860.3 |
| p95 TTFT (ms) | 19866.0 |
| Mean gen tok/s | 75.97 |
| Total wall time (s) | 65.16 |
| Proxy RSS (MB) | 81.8 |

## Per-Task Results

| Task ID | Category | Pass | Prompt tok | Completion tok | Wall (ms) | Error |
|---|---|---|---|---|---|---|
| stress_01 | stress_continuity | ✅ | 560 | 350 | 4936 |  |
| stress_02 | stress_continuity | ❌ | 964 | 1176 | 17481 | Traceback (most recent call last):   File "/tmp/edgemem_benc |
| stress_03 | stress_continuity | ✅ | 839 | 549 | 6864 |  |
| stress_04 | stress_continuity | ❌ | 834 | 689 | 8493 | Traceback (most recent call last):   File "/tmp/edgemem_benc |
| stress_05 | stress_continuity | ❌ | 1030 | 1099 | 13811 | Traceback (most recent call last):   File "/tmp/edgemem_benc |
| stress_06 | stress_continuity | ✅ | 968 | 1045 | 13578 |  |
