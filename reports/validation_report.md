# HummingFlight Validation Report (L0..L6)

| Field | Value |
|---|---|
| Project version | `2026.9.3` |
| Platform | `Windows-10-10.0.19044-SP0` |
| Python | `3.11.15` |
| Timestamp (UTC) | `2026-10-09T23:58:36Z` |

## Environment

| Item | Value |
|---|---|
| Source commit-reproducible inputs | deterministic self-tests + sha256-pinned goldens |
| Version | 2026.9.3 |
| Engine binary | C:\Users\Administrator\Documents\PyCharm\HummingFlight\build\Release\glm.exe |

## L0 functional

- `PASS` **L0 version string** -- 2026.9.3
- `PASS` **L0 functional self-tests** -- glm.exe --self-test
- `PASS` **L1 pytest suite** -- delegated: `pytest tests/ -q` (enforced by package.py and CI)
## L2 model validation

| Item | Value |
|---|---|
| Model directory | E:\glm-5.2-bf16 |

- `PASS` **L2 model structural validation** -- 292 lines
    [07:58:36.782] [INFO ] Found 282 shard files
    [07:58:36.906] [INFO ] Using index.json for fast startup (59585 tensors mapped)
    [07:58:36.923] [INFO ] safetensors parsed: E:\glm-5.2-bf16\model-00057-of-00282.safetensors (211 of 211 tensors accepted)
    [07:58:36.924] [INFO ] safetensors parsed: E:\glm-5.2-bf16\model-00241-of-00282.safetensors (211 of 211 tensors accepted)
    [07:58:36.925] [INFO ] safetensors parsed: E:\glm-5.2-bf16\model-00173-of-00282.safetensors (213 of 213 tensors accepted)
    [07:58:36.926] [INFO ] safetensors parsed: E:\glm-5.2-bf16\model-00118-of-00282.safetensors (213 of 213 tensors accepted)
    [07:58:36.926] [INFO ] safetensors parsed: E:\glm-5.2-bf16\model-00213-of-00282.safetensors (213 of 213 tensors accepted)
    [07:58:36.927] [INFO ] safetensors parsed: E:\glm-5.2-bf16\model-00125-of-00282.safetensors (213 of 213 tensors accepted)
    [07:58:36.928] [INFO ] safetensors parsed: E:\glm-5.2-bf16\model-00051-of-00282.safetensors (213 of 213 tensors accepted)
    [07:58:36.929] [INFO ] safetensors parsed: E:\glm-5.2-bf16\model-00002-of-00282.safetensors (212 of 212 tensors accepted)
    [07:58:36.930] [INFO ] safetensors parsed: E:\glm-5.2-bf16\model-00189-of-00282.safetensors (211 of 211 tensors accepted)
    [07:58:36.931] [INFO ] safetensors parsed: E:\glm-5.2-bf16\model-00080-of-00282.safetensors (213 of 213 tensors accepted)
- `PASS` **L2 index.json parse** -- 59585 tensors mapped
- `PASS` **L2 shard files exist** -- 282 shards present
| layers detected | 79 |
- `PASS` **L2 embedding present**
- `PASS` **L2 lm_head present**
- `PASS` **L2 DSA indexer weights present** -- 21/21 full layers
- `PASS` **L2b tokenizer encode** -- 4 ids
- `PASS` **L2b tokenizer decode** -- 'Hello, world!'
## L3 golden inference outputs

- `PASS` **L3 golden greeting.json** -- 4 tokens matched
- `PASS` **L3 golden smoke.json** -- 1 tokens matched
## L4 performance benchmark

- `PASS` **L4 performance benchmark** -- C:\Users\Administrator\.conda\envs\humming_flight\python.exe C:\Users\Administrator\Documents\PyCharm\HummingFlight\scripts\benchmark.py ...
## L5 end-to-end

- `PASS` **L5 end-to-end** -- C:\Users\Administrator\.conda\envs\humming_flight\python.exe C:\Users\Administrator\Documents\PyCharm\HummingFlight\scripts\run_e2e.py ...
## L6 completeness audit

- `PASS` **L6 completeness audit** -- C:\Users\Administrator\.conda\envs\humming_flight\python.exe C:\Users\Administrator\Documents\PyCharm\HummingFlight\scripts\audit_completeness.py ...
## Reproducibility

| Artifact | SHA-256 |
|---|---|
| CMakeLists.txt | 841591a004e0173e51bea1b8df07dedc481d9a1f14aff05fbc3059f9cd9092cc |
| src/version.h | 2608e829f67ae799f67236aaf3fffa5e722eadcd22b000e0db28293ffa9d3203 |
| tools/tokenizer_server.py | e561c35b7cede9d463d0ebdd9a1b69e6837b3afd10873046a958b0d47c7a87fd |

- `PASS` **100% gate: all quality gates PASS**
## Summary

- **Total checks**: 17
- **Passed**: 17
- **Failed**: 0
- **Skipped**: 0
- **Verification rate**: `100.00%` of runnable checks passed
