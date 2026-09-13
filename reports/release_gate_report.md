# HummingFlight Release Gate Report

| Field | Value |
|---|---|
| Project version | `2026.9.0` |
| Platform | `Windows-10-10.0.19044-SP0` |
| Python | `3.11.15` |
| Timestamp (UTC) | `2026-09-12T23:31:40Z` |

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
