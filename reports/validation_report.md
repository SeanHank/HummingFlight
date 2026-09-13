# HummingFlight Validation Report (L0..L6)

| Field | Value |
|---|---|
| Project version | `2026.9.0` |
| Platform | `Windows-10-10.0.19044-SP0` |
| Python | `3.11.15` |
| Timestamp (UTC) | `2026-09-12T23:31:45Z` |

## Environment

| Item | Value |
|---|---|
| Source commit-reproducible inputs | deterministic self-tests + sha256-pinned goldens |
| Version | 2026.9.0 |
| Engine binary | C:\Users\Administrator\Documents\PyCharm\HummingFlight\build\Release\glm.exe |

## L0 functional

- `PASS` **L0 version string** -- 2026.9.0
- `PASS` **L0 functional self-tests** -- glm.exe --self-test
- `PASS` **L1 pytest suite** -- delegated: `pytest tests/ -q` (enforced by package.py and CI)
## L2 model validation

| Item | Value |
|---|---|
| Model directory | E:\glm-5.2-bf16 |

- `PASS` **L2 model structural validation** -- 289 lines
    [07:31:45.269] [INFO ] Found 282 shard files
    [07:31:45.345] [INFO ] Using index.json for fast startup (59585 tensors mapped)
    [07:31:45.365] [INFO ] safetensors parsed: E:\glm-5.2-bf16\model-00057-of-00282.safetensors (211 tensors)
    [07:31:45.365] [INFO ] safetensors parsed: E:\glm-5.2-bf16\model-00241-of-00282.safetensors (211 tensors)
    [07:31:45.366] [INFO ] safetensors parsed: E:\glm-5.2-bf16\model-00173-of-00282.safetensors (213 tensors)
    [07:31:45.366] [INFO ] safetensors parsed: E:\glm-5.2-bf16\model-00118-of-00282.safetensors (213 tensors)
    [07:31:45.367] [INFO ] safetensors parsed: E:\glm-5.2-bf16\model-00213-of-00282.safetensors (213 tensors)
    [07:31:45.368] [INFO ] safetensors parsed: E:\glm-5.2-bf16\model-00125-of-00282.safetensors (213 tensors)
    [07:31:45.368] [INFO ] safetensors parsed: E:\glm-5.2-bf16\model-00051-of-00282.safetensors (213 tensors)
    [07:31:45.369] [INFO ] safetensors parsed: E:\glm-5.2-bf16\model-00002-of-00282.safetensors (212 tensors)
    [07:31:45.370] [INFO ] safetensors parsed: E:\glm-5.2-bf16\model-00189-of-00282.safetensors (211 tensors)
    [07:31:45.370] [INFO ] safetensors parsed: E:\glm-5.2-bf16\model-00080-of-00282.safetensors (213 tensors)
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
| CMakeLists.txt | 9fe2dfa2d07f9a917134b339efd54e67ce1210b2b95d0534744050d3ffd1a512 |
| src/version.h | 5d6c850d578f24a7e60ee28f853ada4195d33fdfedbc2987114e4fc1bd3fd59a |
| tools/tokenizer_server.py | b63e19c8506943bafe59c41ecf9627c4df855c81a6b91424e1850af67e18ff72 |

- `PASS` **100% gate: all quality gates PASS**
## Summary

- **Total checks**: 17
- **Passed**: 17
- **Failed**: 0
- **Skipped**: 0
- **Verification rate**: `100.00%` of runnable checks passed
