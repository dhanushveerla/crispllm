# EdgeMem Benchmark Report: `phase1b-nvidia-nemotron-3.5-bolt-e4b`

| Field | Value |
|---|---|
| Label | `phase1b-nvidia-nemotron-3.5-bolt-e4b` |
| Model | `nvidia/nemotron-3.5-bolt-e4b` |
| Endpoint | `http://127.0.0.1:8080/v1` |
| Tasks run | 20 |
| **Pass@1 (overall)** | **75.0%** (15/20) |
| Pass@1 exact_fact | 90.0% (9/10) |
| Pass@1 continuity | 60.0% (6/10) |
| Total prompt tokens | 8,056 |
| Total completion tokens | 5,867 |
| Mean prompt tokens | 402.8 |
| Mean completion tokens | 293.4 |
| Mean TTFT (ms) | 3662.2 |
| p95 TTFT (ms) | 8581.5 |
| Mean gen tok/s | 78.89 |
| Total wall time (s) | 73.24 |
| Proxy RSS (MB) | 77.2 |

## Per-Task Results

| Task ID | Category | Pass | Prompt tok | Completion tok | Wall (ms) | Error |
|---|---|---|---|---|---|---|
| ef_01 | exact_fact | ✅ | 431 | 253 | 3145 |  |
| ef_02 | exact_fact | ✅ | 307 | 104 | 1369 |  |
| ef_03 | exact_fact | ✅ | 314 | 100 | 1409 |  |
| ef_04 | exact_fact | ✅ | 506 | 132 | 1639 |  |
| ef_05 | exact_fact | ✅ | 451 | 657 | 7424 |  |
| ef_06 | exact_fact | ✅ | 371 | 337 | 3834 |  |
| ef_07 | exact_fact | ❌ | 387 | 192 | 3227 | Traceback (most recent call last):   File "/tmp/edgemem_benc |
| ef_08 | exact_fact | ✅ | 283 | 78 | 1042 |  |
| ef_09 | exact_fact | ✅ | 348 | 157 | 1906 |  |
| ef_10 | exact_fact | ✅ | 364 | 245 | 2865 |  |
| ct_01 | continuity | ✅ | 446 | 329 | 3991 |  |
| ct_02 | continuity | ✅ | 423 | 178 | 2359 |  |
| ct_03 | continuity | ❌ | 512 | 267 | 3535 | Traceback (most recent call last):   File "/tmp/edgemem_benc |
| ct_04 | continuity | ✅ | 410 | 285 | 3487 |  |
| ct_05 | continuity | ❌ | 413 | 582 | 7377 | Traceback (most recent call last):   File "/tmp/edgemem_benc |
| ct_06 | continuity | ❌ | 413 | 215 | 2506 | Traceback (most recent call last):   File "/tmp/edgemem_benc |
| ct_07 | continuity | ✅ | 388 | 531 | 6021 |  |
| ct_08 | continuity | ✅ | 447 | 191 | 3274 |  |
| ct_09 | continuity | ❌ | 465 | 666 | 8642 | Traceback (most recent call last):   File "/tmp/edgemem_benc |
| ct_10 | continuity | ✅ | 377 | 368 | 4191 |  |
