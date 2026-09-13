# HummingFlight Benchmark Report (L4)

| Field | Value |
|---|---|
| Project version | `2026.9.0` |
| Platform | `Windows-10-10.0.19044-SP0` |
| Python | `3.11.15` |
| Timestamp (UTC) | `2026-09-13T04:12:38Z` |

## Config

| Item | Value |
|---|---|
| Model directory | E:\glm-5.2-bf16 |
| Prompt tokens | 3,7,12 |
| Max tokens | 4 |

## Results

| Metric | Value |
|---|---|
| generated_tokens | 4 |
| total_sec | 5875.710399 s |
| speed_tok_per_s | 0.000681 tok/s |

4] [INFO ] Index build time: 0.261351 s
[10:34:38.074] [INFO ] Raw token ids provided (--prompt-tokens); tokenizer skipped
[10:34:38.074] [INFO ] Scheduler ready: LRU=16091 MB, IOCP workers=4
[10:34:38.074] [INFO ] Scheduler ready: LRU 16091 MB, IOCP workers 4
[10:34:38.074] [INFO ] Initializing forward engine (MLA + MoE, AVX2 + OpenMP)...
[10:34:38.074] [INFO ] DSA indexer: ENABLED
[10:34:38.194] [INFO ] Loaded model.norm.weight
[10:34:38.194] [INFO ] OpenMP thread count: 16
[10:34:38.194] [INFO ] Forward engine ready: 78 layers, 6144 dim
[10:34:38.195] [INFO ] Input: raw 3 tokens
[10:34:38.195] [INFO ] Encoded: 3 tokens
[10:34:38.195] [INFO ] max-tokens = 4
[11:26:18.784] [INFO ] [1/4] token=537 (3100.588380 s)
[11:36:47.637] [INFO ] [2/4] token=537 (3729.441881 s)
[11:52:27.820] [INFO ] [3/4] token=537 (4669.624084 s)
[12:12:33.905] [INFO ] [4/4] token=537 (5875.709685 s)
[12:12:33.906] [INFO ] === Generation complete ===
[12:12:33.906] [INFO ] Generated 4 tokens, time 5875.710399 s
[12:12:33.906] [INFO ] Speed: 0.000681 tok/s
[12:12:33.906] [INFO ] Output ids: 537,537,537,537
[12:12:33.912] [INFO ] Scheduler shutdown: LRU 16080 MB used, hit rate 57%
[12:12:33.912] [INFO ] Done

## L4 gate decision

- `PASS` **L4 performance benchmark** -- 4 tokens @ 0.00 tok/s
## Summary

- **Total checks**: 1
- **Passed**: 1
- **Failed**: 0
- **Skipped**: 0
- **Verification rate**: `100.00%` of runnable checks passed
