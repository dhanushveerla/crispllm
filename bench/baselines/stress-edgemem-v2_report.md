# EdgeMem Benchmark Report: `stress-edgemem-v2`

| Field | Value |
|---|---|
| Label | `stress-edgemem-v2` |
| Model | `nvidia/nemotron-3.5-bolt-e4b` |
| Endpoint | `http://127.0.0.1:8080/v1` |
| Tasks run | 6 |
| **Pass@1 (overall)** | **66.7%** (4/6) |
| Pass@1 exact_fact | — |
| Pass@1 continuity | — |
| Total prompt tokens | 5,195 |
| Total completion tokens | 4,809 |
| Mean prompt tokens | 865.8 |
| Mean completion tokens | 801.5 |
| Mean TTFT (ms) | 11204.0 |
| p95 TTFT (ms) | 14171.4 |
| Mean gen tok/s | 69.86 |
| Total wall time (s) | 67.22 |
| Proxy RSS (MB) | 79.8 |

## Per-Task Results

| Task ID | Category | Pass | Prompt tok | Completion tok | Wall (ms) | Error |
|---|---|---|---|---|---|---|
| stress_01 | stress_continuity | ✅ | 560 | 301 | 9063 |  |
| stress_02 | stress_continuity | ❌ | 964 | 1079 | 14133 | Traceback (most recent call last):   File "/tmp/edgemem_benc |
| stress_03 | stress_continuity | ✅ | 839 | 603 | 9442 |  |
| stress_04 | stress_continuity | ❌ | 834 | 851 | 10499 | Traceback (most recent call last):   File "/tmp/edgemem_benc |
| stress_05 | stress_continuity | ✅ | 1030 | 845 | 10014 |  |
| stress_06 | stress_continuity | ✅ | 968 | 1130 | 14073 |  |
