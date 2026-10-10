# HummingFlight Benchmark Report (L4)

| Field | Value |
|---|---|
| Project version | `2026.9.3` |
| Platform | `Windows-10-10.0.19044-SP0` |
| Python | `3.11.15` |
| Timestamp (UTC) | `2026-10-10T12:45:42Z` |

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
| total_sec | 6922.836242 s |
| speed_tok_per_s | 0.000578 tok/s |

## Engine telemetry (I/O & cache)

| Metric | Value |
|---|---|
| CPU utilization | 1 |
| GPU utilization (expert stream) | 0.00% |
| PCIe utilization | 0.00% |
| Expert LRU hit rate | 66.11% |
| Expert LRU misses | 1220 |
| Expert disk (sync) loads | 1220 |
| Prefetch delivery rate | 123.91% |
| Prefetch waste rate | 0.00% |
| Routed-pred staged (gate survivors) | 16 |
| Routed-pred consumed | 0 |
| Routed-pred waste rate | 100.00% |
| Demand keys staged | 3617 |
| Demand wait timeouts | 0 |
| Sync (mmap) fallback loads | 0 |
| Sync (mmap) fallback bytes | 0.00 B |
| Sync (mmap) stall | 0 ms |
| Demand wait (forward thread) | 3038670 ms |
| Staging skipped (LRU hit) | 0 |
| Loop look-ahead issued (O12/O13b) | 42 |
| First-expert wait / layer (O13c) | 3759.69 ms |
| Reads in flight @ first wait (O13c) | 8.7261 |
| Experts in flight @ first wait (O13c) | 10.9376 |
| Total disk bytes (async+sync) | 285.82 GiB |
| Random I/O runs | 4007 |
| I/O read requests | 4007 |
| Disk bytes read | 285.82 GiB |
| Sequential bytes | 285.82 GiB |
| Bytes / token (decode) | 76724300000.0 B |
| Sequential bytes / token (decode) | 76724300000.0 B |
| HDD latency (avg) | 4009.55 ms |
| RAM bandwidth (compute) | 68.3 MB/s |
| PCIe transfer | 0.000 GB/s |
| Expert reuse distance | 108.7 tok |

tilization":0,"pcie_utilization":0,"gpu_active_ms":0,"pcie_bytes":0,"expert_lookups":3600,"expert_lru_hits":2380,"expert_lru_misses":1220,"expert_disk_loads":1220,"expert_cache_hit_rate":0.661111,"expert_cache_miss_rate":0.338889,"prefetch_submitted":4007,"prefetch_inserted":4965,"prefetch_wasted":0,"prefetch_delivery_rate":1.23908,"prefetch_waste_rate":0,"routed_pred_staged":16,"routed_pred_consumed":0,"routed_pred_waste_rate":1,"demand_keys_staged":3617,"demand_wait_timeout":0,"sync_loads":0,"sync_bytes":0,"sync_stall_ms":0,"demand_wait_ms":3.03867e+06,"stage_skip_lru":0,"loop_prefetch_issued":42,"first_expert_wait_ms":3759.69,"reads_in_flight_at_first_wait":8.72606,"experts_in_flight_at_first_wait":10.9376,"total_disk_bytes":306897223680,"disk_bytes_read":306897223680,"sequential_bytes":306897223680,"random_io_count":4007,"io_read_count":4007,"bytes_read_per_token":7.67243e+10,"sequential_bytes_per_token":7.67243e+10,"hdd_latency_ms":4009.55,"ram_bandwidth_mbps":68.2865,"pcie_transfer_gbps":0,"expert_reuse_distance_tok":108.692}
[20:45:38.679] [INFO ] Output ids: 16,29661,90,77
[20:45:38.682] [INFO ] Scheduler shutdown: LRU 38592 MB used, hit rate 2%
[20:45:38.682] [INFO ] Done

## L4 gate decision

- `PASS` **L4 performance benchmark** -- 4 tokens @ 0.00 tok/s
## Summary

- **Total checks**: 1
- **Passed**: 1
- **Failed**: 0
- **Skipped**: 0
- **Verification rate**: `100.00%` of runnable checks passed
## Run comparison (standalone L4, same config `--prompt-tokens 3,7,12 --max-tokens 4`)

| Metric | R5 | R6 | release | R6-revert | unlabelled | R4 | R5-widemerge | R3 | R2 | R1 | Baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|
| total_sec | **6684.179613 s** | 6781.376757 s | 6922.836242 s | 6984.419813 s | 6995.504731 s | 7103.63227 s | 7219.38121 s | 7572.95 s | 8667.82 s | 9849.07 s | 12030.37 s |
| speed_tok_per_s | **0.000598** | 0.000590 | 0.000578 | 0.000573 | 0.000572 | 0.000563 | 0.000554 | 0.000528 | 0.000461 | 0.000406 | 0.000332 |
| Total disk bytes (whole run) | **285.82 GiB** | 285.82 GiB | 285.82 GiB | 285.82 GiB | 285.82 GiB | 279.63 GiB | 285.82 GiB | 318.52 GiB | — | — | — |
| Prefetch waste | 0.0 % | 0.0 % | 0.0 % | 0.0 % | 0.0 % | 0.0 % | 0.0 % | 8.9 % | 31.7 % | 66.3 % | 69.9 % |
| Generated output | **`16,29661,90,77`** | `16,29661,90,77` | `16,29661,90,77` | `16,29661,90,77` | `16,29661,90,77` | `16,29661,90,77` | `16,29661,90,77` | `16,29661,90,77` | `16,29661,90,77` | `16,29661,90,77` | `16,29661,90,77` |

> Output bit-identical across all 11 runs.

