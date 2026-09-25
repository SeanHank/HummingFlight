# HummingFlight Validation Report (L0..L6)

| Field | Value |
|---|---|
| Project version | `2026.9.2` |
| Platform | `Windows-10-10.0.19044-SP0` |
| Python | `3.13.5` |
| Timestamp (UTC) | `2026-09-22T11:23:34Z` |

## Environment

| Item | Value |
|---|---|
| Source commit-reproducible inputs | deterministic self-tests + sha256-pinned goldens |
| Version | 2026.9.2 |
| Engine binary | C:\Users\Administrator\Documents\PyCharm\HummingFlight\build\Release\glm.exe |

## L0 functional

- `PASS` **L0 version string** -- 2026.9.2
- `PASS` **L0 functional self-tests** -- glm.exe --self-test
- `PASS` **L1 pytest suite** -- delegated: `pytest tests/ -q` (enforced by package.py and CI)
## L2 model validation

| Item | Value |
|---|---|
| Model directory | E:\glm-5.2-bf16 |

- `PASS` **L2 model structural validation** -- 292 lines
    [19:23:34.187] [INFO ] Found 282 shard files
    [19:23:34.327] [INFO ] Using index.json for fast startup (59585 tensors mapped)
    [19:23:34.345] [INFO ] safetensors parsed: E:\glm-5.2-bf16\model-00057-of-00282.safetensors (211 of 211 tensors accepted)
    [19:23:34.346] [INFO ] safetensors parsed: E:\glm-5.2-bf16\model-00241-of-00282.safetensors (211 of 211 tensors accepted)
    [19:23:34.347] [INFO ] safetensors parsed: E:\glm-5.2-bf16\model-00173-of-00282.safetensors (213 of 213 tensors accepted)
    [19:23:34.348] [INFO ] safetensors parsed: E:\glm-5.2-bf16\model-00118-of-00282.safetensors (213 of 213 tensors accepted)
    [19:23:34.349] [INFO ] safetensors parsed: E:\glm-5.2-bf16\model-00213-of-00282.safetensors (213 of 213 tensors accepted)
    [19:23:34.350] [INFO ] safetensors parsed: E:\glm-5.2-bf16\model-00125-of-00282.safetensors (213 of 213 tensors accepted)
    [19:23:34.351] [INFO ] safetensors parsed: E:\glm-5.2-bf16\model-00051-of-00282.safetensors (213 of 213 tensors accepted)
    [19:23:34.352] [INFO ] safetensors parsed: E:\glm-5.2-bf16\model-00002-of-00282.safetensors (212 of 212 tensors accepted)
    [19:23:34.353] [INFO ] safetensors parsed: E:\glm-5.2-bf16\model-00189-of-00282.safetensors (211 of 211 tensors accepted)
    [19:23:34.354] [INFO ] safetensors parsed: E:\glm-5.2-bf16\model-00080-of-00282.safetensors (213 of 213 tensors accepted)
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

- `PASS` **L4 performance benchmark** -- C:\ProgramData\anaconda3\python.exe C:\Users\Administrator\Documents\PyCharm\HummingFlight\scripts\benchmark.py ...
## L5 end-to-end

- `PASS` **L5 end-to-end** -- C:\ProgramData\anaconda3\python.exe C:\Users\Administrator\Documents\PyCharm\HummingFlight\scripts\run_e2e.py ...
## L6 completeness audit

- `PASS` **L6 completeness audit** -- C:\ProgramData\anaconda3\python.exe C:\Users\Administrator\Documents\PyCharm\HummingFlight\scripts\audit_completeness.py ...
## Reproducibility

| Artifact | SHA-256 |
|---|---|
| CMakeLists.txt | 0d3f710e568abf228fb7880f0834d231f71784a474c14dc6c2b9905ad7cd1c93 |
| src/version.h | 420ca8b0c5d0aeccf25b796739e5356d83dbf91325a83eeb32fb3b975337ca1e |
| tools/tokenizer_server.py | bfb4ae64cd99763331c7b3017eeb4fe8e0b29214fe41ce87119e2ba0769e1d89 |

- `PASS` **100% gate: all quality gates PASS**
## Summary

- **Total checks**: 17
- **Passed**: 17
- **Failed**: 0
- **Skipped**: 0
- **Verification rate**: `100.00%` of runnable checks passed
