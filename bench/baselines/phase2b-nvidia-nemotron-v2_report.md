# EdgeMem Benchmark Report: `phase2b-nvidia-nemotron-v2`

| Field | Value |
|---|---|
| Label | `phase2b-nvidia-nemotron-v2` |
| Model | `nvidia/nemotron-3.5-bolt-e4b` |
| Endpoint | `http://127.0.0.1:8080/v1` |
| Tasks run | 20 |
| **Pass@1 (overall)** | **80.0%** (16/20) |
| Pass@1 exact_fact | 90.0% (9/10) |
| Pass@1 continuity | 70.0% (7/10) |
| Total prompt tokens | 8,874 |
| Total completion tokens | 5,142 |
| Mean prompt tokens | 443.7 |
| Mean completion tokens | 257.1 |
| Mean TTFT (ms) | 3323.0 |
| p95 TTFT (ms) | 9398.7 |
| Mean gen tok/s | 78.88 |
| Total wall time (s) | 66.46 |
| Proxy RSS (MB) | 79.7 |

## Per-Task Results

| Task ID | Category | Pass | Prompt tok | Completion tok | Wall (ms) | Error |
|---|---|---|---|---|---|---|
| ef_01 | exact_fact | ✅ | 431 | 253 | 8358 |  |
| ef_02 | exact_fact | ✅ | 307 | 104 | 1424 |  |
| ef_03 | exact_fact | ✅ | 314 | 100 | 1424 |  |
| ef_04 | exact_fact | ✅ | 506 | 132 | 1728 |  |
| ef_05 | exact_fact | ✅ | 451 | 180 | 2227 |  |
| ef_06 | exact_fact | ✅ | 371 | 347 | 4135 |  |
| ef_07 | exact_fact | ❌ | 387 | 191 | 2253 | Traceback (most recent call last):   File "/tmp/edgemem_benc |
| ef_08 | exact_fact | ✅ | 283 | 78 | 1123 |  |
| ef_09 | exact_fact | ✅ | 348 | 150 | 1889 |  |
| ef_10 | exact_fact | ✅ | 364 | 196 | 2397 |  |
| ct_01 | continuity | ✅ | 523 | 342 | 3970 |  |
| ct_02 | continuity | ✅ | 485 | 178 | 2161 |  |
| ct_03 | continuity | ❌ | 626 | 269 | 3154 | Traceback (most recent call last):   File "/tmp/edgemem_benc |
| ct_04 | continuity | ✅ | 509 | 281 | 3291 |  |
| ct_05 | continuity | ✅ | 459 | 624 | 6948 |  |
| ct_06 | continuity | ❌ | 483 | 223 | 2741 | Traceback (most recent call last):   File "/tmp/edgemem_benc |
| ct_07 | continuity | ✅ | 502 | 182 | 2247 |  |
| ct_08 | continuity | ✅ | 500 | 167 | 2075 |  |
| ct_09 | continuity | ❌ | 532 | 847 | 9454 | Traceback (most recent call last):   File "/tmp/edgemem_benc |
| ct_10 | continuity | ✅ | 493 | 298 | 3460 |  |
