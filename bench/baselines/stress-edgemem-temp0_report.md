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
| Total prompt tokens | 4,635 |
| Total completion tokens | 4,781 |
| Mean prompt tokens | 772.5 |
| Mean completion tokens | 796.8 |
| Mean TTFT (ms) | 11935.0 |
| p95 TTFT (ms) | 16636.0 |
| Mean gen tok/s | 81.86 |
| Total wall time (s) | 70.43 |
| Proxy RSS (MB) | 81.8 |

## Per-Task Results

| Task ID | Category | Pass | Prompt tok | Completion tok | Wall (ms) | Error |
|---|---|---|---|---|---|---|
| stress_01 | stress_continuity | ❌ | 0 | 0 | 10759 | LLM call failed: HTTP Error 500: Internal Server Error |
| stress_02 | stress_continuity | ❌ | 964 | 1136 | 16212 | Traceback (most recent call last):   File "/tmp/edgemem_benc |
| stress_03 | stress_continuity | ✅ | 839 | 699 | 7590 |  |
| stress_04 | stress_continuity | ❌ | 834 | 673 | 7952 | Traceback (most recent call last):   File "/tmp/edgemem_benc |
| stress_05 | stress_continuity | ✅ | 1030 | 1289 | 15606 |  |
| stress_06 | stress_continuity | ✅ | 968 | 984 | 12316 |  |
