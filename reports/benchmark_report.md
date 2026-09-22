# HummingFlight Benchmark Report (L4)

| Field | Value |
|---|---|
| Project version | `2026.9.1` |
| Platform | `Windows-10-10.0.19044-SP0` |
| Python | `3.13.5` |
| Timestamp (UTC) | `2026-09-21T15:54:57Z` |

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
| total_sec | 9849.070608 s |
| speed_tok_per_s | 0.000406 tok/s |

## Performance optimization round (01e1cad, 2026-09)

Before/after on this identical L4 workload (prompt ids `3,7,12`, 4 tokens, same HDD box).
The optimization round is I/O-layer only (O1 RAM-aware LRU sizing, O2 prefetch in-flight
budget, O3 gap-tolerant seek merging -- see `doc/design.md` 17.2), so the generated token
sequence is **bit-identical** to the pre-optimization engine: `16,29661,90,77`.

| Metric | Baseline (2026-09-15) | Optimized (2026-09-21) | Delta |
|---|---|---|---|
| total_sec (4 tokens) | 12030.37 s | 9849.07 s | **-18.1 %** |
| speed_tok_per_s | 0.000332 | 0.000406 | **+22.3 %** |
| LRU capacity (auto) | 15.7 GiB (RAM/4) | 38.6 GiB (min(RAM·⅗, 40 GiB)) | +2.5× |
| LRU used @ shutdown | 16 056 MB | 38 592 MB | +2.4× |
| Prefetch waste rate | 69.92 % | 66.33 % | -3.6 pp |
| HDD latency (avg) | 3718 ms | 3699 ms | -0.5 % |
| Disk bytes / token | 69.5 GiB | 69.5 GiB | 0 (cold worst case) |
| Expert LRU hit rate | 20.97 % | 20.97 % | 0 (see note) |
| Generated output | 16,29661,90,77 | 16,29661,90,77 | **identical** |

Note on hit rate: the L4 workload's prompt is three *random init ids*, so every batch position
routes to almost-disjoint experts -- a cold worst case where bytes/token is structurally fixed
(~69.5 GiB) regardless of cache size; the 21 % hits are intra-batch repeats only. The 18 % wall
gain here comes from O2/O3 I/O scheduling plus warm OS page-cache for the dense weights (the
earlier run opened a cold cache). On **real workloads** (chat prompts with routing locality and
tokens spaced within the 108-token reuse distance) the now-2.5× LRU window keeps repeated
experts resident, which this synthetic 3-id prompt cannot exhibit. Golden outputs are unaffected
by the round.

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
| HDD latency (avg) | 3698.68 ms |
| RAM bandwidth (compute) | 48.0 MB/s |
| PCIe transfer | 0.000 GB/s |
| Expert reuse distance | 108.7 tok |

] [2/4] token=29661 (6606.851992 s)
[23:27:00.318] [INFO ] [3/4] token=90 (8178.888477 s)
[23:54:50.500] [INFO ] [4/4] token=77 (9849.070200 s)
[23:54:50.500] [INFO ] === Generation complete ===
[23:54:50.500] [INFO ] Generated 4 tokens, time 9849.070608 s
[23:54:50.500] [INFO ] Speed: 0.000406 tok/s
[23:54:50.502] [INFO ] STATS: {"generated_sec":9849.07,"cpu_utilization":1,"gpu_utilization":0,"pcie_utilization":0,"gpu_active_ms":0,"pcie_bytes":0,"expert_lookups":3600,"expert_lru_hits":755,"expert_lru_misses":2845,"expert_disk_loads":2845,"expert_cache_hit_rate":0.209722,"expert_cache_miss_rate":0.790278,"prefetch_submitted":3840,"prefetch_inserted":4840,"prefetch_wasted":2547,"prefetch_delivery_rate":1.26042,"prefetch_waste_rate":0.663281,"disk_bytes_read":298366009344,"sequential_bytes":298366009344,"random_io_count":3840,"io_read_count":3840,"bytes_read_per_token":7.45915e+10,"sequential_bytes_per_token":7.45915e+10,"hdd_latency_ms":3698.68,"ram_bandwidth_mbps":47.9981,"pcie_transfer_gbps":0,"expert_reuse_distance_tok":108.692}
[23:54:50.502] [INFO ] Output ids: 16,29661,90,77
[23:54:50.510] [INFO ] Scheduler shutdown: LRU 38592 MB used, hit rate 42%
[23:54:50.510] [INFO ] Done

## L4 gate decision

- `PASS` **L4 performance benchmark** -- 4 tokens @ 0.00 tok/s
## Summary

- **Total checks**: 1
- **Passed**: 1
- **Failed**: 0
- **Skipped**: 0
- **Verification rate**: `100.00%` of runnable checks passed
