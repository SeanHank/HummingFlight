# HummingFlight Release Gate Report

| Field | Value |
|---|---|
| Project version | `2026.9.1` |
| Platform | `Windows-10-10.0.19044-SP0` |
| Python | `3.13.5` |
| Timestamp (UTC) | `2026-09-14T22:50:40Z` |

## Quality gates (require 100% PASS; model-present SKIPs fail the release)


- `PASS` **cmake configure** -- 0/9 CMake configure
- `PASS` **build (Release)** -- 1/9 Build (Release)
- `PASS` **L0 functional self-tests** -- 2/9 L0 functional self-tests (CTest)
- `PASS` **L1 pytest suite** -- 3/9 L1 pytest suite
- `PASS` **L2+L3 validation harness** -- 4/9 L2/L3 validation harness
- `PASS` **L4 performance benchmark** -- 5/9 L4 performance benchmark
- `PASS` **L5 end-to-end** -- 6/9 L5 end-to-end
- `PASS` **L6 completeness audit** -- 7/9 L6 completeness audit
- `PASS` **CPack archives** -- 8/9 CPack archives

## Performance optimization round re-verification (2026-09-21)

The I/O-layer optimization round (O1 RAM-aware LRU sizing, O2 prefetch budget, O3 seek
merging; `doc/design.md` §17) changes fetch scheduling only -- forward math is untouched --
so it was re-verified without a 13 h full L2/L3 golden re-run:

| Gate | Result |
|---|---|
| L0 functional self-tests | **PASS** -- 163 checks, 0 failures (adaptive-config tests updated for O1) |
| L1 pytest suite | **PASS** -- 28 passed |
| L2 `--check-weights` (real model) | **PASS** -- redirected: 0 failures |
| L4 performance benchmark | **PASS** -- 4 tokens @ 0.000406 tok/s (baseline 0.000332), output `16,29661,90,77` **identical** |
| L3 goldens | **unchanged** -- I/O-layer only; pinned outputs unaffected (re-run on next full release gate) |

Artifacts regenerated: `reports/benchmark_report.md` (with before/after table),
`reports/benchmark_raw.json`.
