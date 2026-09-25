# HummingFlight Benchmark Report (L4)

| Field | Value |
|---|---|
| Project version | `2026.9.2` |
| Platform | `Windows-10-10.0.19044-SP0` |
| Python | `3.13.5` |
| Timestamp (UTC) | `2026-09-23T13:06:19Z` |

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
| total_sec | 17685.303505 s |
| speed_tok_per_s | 0.000226 tok/s |

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
| Prefetch waste rate | 66.33% |
| Random I/O runs | 3840 |
| I/O read requests | 3840 |
| Disk bytes read | 277.88 GiB |
| Sequential bytes | 277.88 GiB |
| Bytes / token (decode) | 74591500000.0 B |
| Sequential bytes / token (decode) | 74591500000.0 B |
| HDD latency (avg) | 3770.83 ms |
| RAM bandwidth (compute) | 26.7 MB/s |
| PCIe transfer | 0.000 GB/s |
| Expert reuse distance | 108.7 tok |

/4] token=29661 (12043.258429 s)
[19:58:31.410] [INFO ] [3/4] token=90 (13627.743894 s)
[21:06:08.969] [INFO ] [4/4] token=77 (17685.303138 s)
[21:06:08.969] [INFO ] === Generation complete ===
[21:06:08.969] [INFO ] Generated 4 tokens, time 17685.303505 s
[21:06:08.969] [INFO ] Speed: 0.000226 tok/s
[21:06:08.972] [INFO ] STATS: {"generated_sec":17685.3,"cpu_utilization":1,"gpu_utilization":0,"pcie_utilization":0,"gpu_active_ms":0,"pcie_bytes":0,"expert_lookups":3600,"expert_lru_hits":755,"expert_lru_misses":2845,"expert_disk_loads":2845,"expert_cache_hit_rate":0.209722,"expert_cache_miss_rate":0.790278,"prefetch_submitted":3840,"prefetch_inserted":4840,"prefetch_wasted":2547,"prefetch_delivery_rate":1.26042,"prefetch_waste_rate":0.663281,"disk_bytes_read":298366009344,"sequential_bytes":298366009344,"random_io_count":3840,"io_read_count":3840,"bytes_read_per_token":7.45915e+10,"sequential_bytes_per_token":7.45915e+10,"hdd_latency_ms":3770.83,"ram_bandwidth_mbps":26.7305,"pcie_transfer_gbps":0,"expert_reuse_distance_tok":108.692}
[21:06:08.972] [INFO ] Output ids: 16,29661,90,77
[21:06:08.977] [INFO ] Scheduler shutdown: LRU 38592 MB used, hit rate 42%
[21:06:08.977] [INFO ] Done

## L4 gate decision

- `PASS` **L4 performance benchmark** -- 4 tokens @ 0.00 tok/s
## Summary

- **Total checks**: 1
- **Passed**: 1
- **Failed**: 0
- **Skipped**: 0
- **Verification rate**: `100.00%` of runnable checks passed
