# HummingFlight Benchmark Report (L4)

| Field | Value |
|---|---|
| Project version | `2026.9.1` |
| Platform | `Windows-10-10.0.19044-SP0` |
| Python | `3.13.5` |
| Timestamp (UTC) | `2026-09-15T22:52:07Z` |

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
| total_sec | 12030.37161 s |
| speed_tok_per_s | 0.000332 tok/s |

## Engine telemetry (I/O & cache)

| Metric | Value |
|---|---|
| CPU utilization | 1 |
| GPU utilization (expert stream) | 0.00% |
| PCIe utilization | 0.00% |
| Expert LRU hit rate | 20.97% |
| Expert LRU misses | 2845 |
| Expert disk (sync) loads | 2845 |
| Prefetch delivery rate | 126.04% |
| Prefetch waste rate | 69.92% |
| Random I/O runs | 3840 |
| I/O read requests | 3840 |
| Disk bytes read | 277.88 GiB |
| Sequential bytes | 277.88 GiB |
| Bytes / token (decode) | 74591500000.0 B |
| Sequential bytes / token (decode) | 74591500000.0 B |
| HDD latency (avg) | 3718.48 ms |
| RAM bandwidth (compute) | 39.3 MB/s |
| PCIe transfer | 0.000 GB/s |
| Expert reuse distance | 108.7 tok |

2/4] token=29661 (8994.422937 s)
[06:25:43.728] [INFO ] [3/4] token=90 (10451.422978 s)
[06:52:02.677] [INFO ] [4/4] token=77 (12030.371547 s)
[06:52:02.677] [INFO ] === Generation complete ===
[06:52:02.677] [INFO ] Generated 4 tokens, time 12030.371610 s
[06:52:02.677] [INFO ] Speed: 0.000332 tok/s
[06:52:02.679] [INFO ] STATS: {"generated_sec":12030.4,"cpu_utilization":1,"gpu_utilization":0,"pcie_utilization":0,"gpu_active_ms":0,"pcie_bytes":0,"expert_lookups":3600,"expert_lru_hits":755,"expert_lru_misses":2845,"expert_disk_loads":2845,"expert_cache_hit_rate":0.209722,"expert_cache_miss_rate":0.790278,"prefetch_submitted":3840,"prefetch_inserted":4840,"prefetch_wasted":2685,"prefetch_delivery_rate":1.26042,"prefetch_waste_rate":0.699219,"disk_bytes_read":298366009344,"sequential_bytes":298366009344,"random_io_count":3840,"io_read_count":3840,"bytes_read_per_token":7.45915e+10,"sequential_bytes_per_token":7.45915e+10,"hdd_latency_ms":3718.48,"ram_bandwidth_mbps":39.2952,"pcie_transfer_gbps":0,"expert_reuse_distance_tok":108.692}
[06:52:02.679] [INFO ] Output ids: 16,29661,90,77
[06:52:02.681] [INFO ] Scheduler shutdown: LRU 16056 MB used, hit rate 42%
[06:52:02.681] [INFO ] Done

## L4 gate decision

- `PASS` **L4 performance benchmark** -- 4 tokens @ 0.00 tok/s
## Summary

- **Total checks**: 1
- **Passed**: 1
- **Failed**: 0
- **Skipped**: 0
- **Verification rate**: `100.00%` of runnable checks passed
