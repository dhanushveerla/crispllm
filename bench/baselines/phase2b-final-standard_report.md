# EdgeMem Benchmark Report: `phase2b-final-standard`

| Field | Value |
|---|---|
| Label | `phase2b-final-standard` |
| Model | `nvidia/nemotron-3.5-bolt-e4b` |
| Endpoint | `http://127.0.0.1:8080/v1` |
| Tasks run | 20 |
| **Pass@1 (overall)** | **80.0%** (16/20) |
| Pass@1 exact_fact | 90.0% (9/10) |
| Pass@1 continuity | 70.0% (7/10) |
| Total prompt tokens | 9,004 |
| Total completion tokens | 5,056 |
| Mean prompt tokens | 450.2 |
| Mean completion tokens | 252.8 |
| Mean TTFT (ms) | 3054.1 |
| p95 TTFT (ms) | 7913.6 |
| Mean gen tok/s | 80.45 |
| Total wall time (s) | 61.08 |
| Proxy RSS (MB) | 79.7 |

## Per-Task Results

| Task ID | Category | Pass | Prompt tok | Completion tok | Wall (ms) | Error |
|---|---|---|---|---|---|---|
| ef_01 | exact_fact | ✅ | 431 | 253 | 3511 |  |
| ef_02 | exact_fact | ✅ | 307 | 104 | 1502 |  |
| ef_03 | exact_fact | ✅ | 314 | 100 | 1271 |  |
| ef_04 | exact_fact | ✅ | 506 | 132 | 1736 |  |
| ef_05 | exact_fact | ✅ | 451 | 180 | 2333 |  |
| ef_06 | exact_fact | ✅ | 371 | 333 | 3910 |  |
| ef_07 | exact_fact | ❌ | 387 | 246 | 2983 | Traceback (most recent call last):   File "/tmp/edgemem_benc |
| ef_08 | exact_fact | ✅ | 283 | 76 | 1110 |  |
| ef_09 | exact_fact | ✅ | 348 | 157 | 2040 |  |
| ef_10 | exact_fact | ✅ | 364 | 236 | 2937 |  |
| ct_01 | continuity | ✅ | 549 | 313 | 3686 |  |
| ct_02 | continuity | ✅ | 504 | 178 | 2355 |  |
| ct_03 | continuity | ❌ | 645 | 286 | 3502 | Traceback (most recent call last):   File "/tmp/edgemem_benc |
| ct_04 | continuity | ✅ | 518 | 281 | 3247 |  |
| ct_05 | continuity | ✅ | 474 | 475 | 5333 |  |
| ct_06 | continuity | ❌ | 491 | 234 | 2752 | Traceback (most recent call last):   File "/tmp/edgemem_benc |
| ct_07 | continuity | ✅ | 508 | 183 | 2308 |  |
| ct_08 | continuity | ✅ | 505 | 157 | 1915 |  |
| ct_09 | continuity | ❌ | 545 | 726 | 8049 | Traceback (most recent call last):   File "/tmp/edgemem_benc |
| ct_10 | continuity | ✅ | 503 | 406 | 4601 |  |
