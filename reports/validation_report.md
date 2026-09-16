# HummingFlight Validation Report (L0..L6)

| Field | Value |
|---|---|
| Project version | `2026.9.1` |
| Platform | `Windows-10-10.0.19044-SP0` |
| Python | `3.13.5` |
| Timestamp (UTC) | `2026-09-14T22:52:09Z` |

## Environment

| Item | Value |
|---|---|
| Source commit-reproducible inputs | deterministic self-tests + sha256-pinned goldens |
| Version | 2026.9.1 |
| Engine binary | C:\Users\Administrator\Documents\PyCharm\HummingFlight\build\Release\glm.exe |

## L0 functional

- `PASS` **L0 version string** -- 2026.9.1
- `PASS` **L0 functional self-tests** -- glm.exe --self-test
- `PASS` **L1 pytest suite** -- delegated: `pytest tests/ -q` (enforced by package.py and CI)
## L2 model validation

| Item | Value |
|---|---|
| Model directory | E:\glm-5.2-bf16 |

- `PASS` **L2 model structural validation** -- 292 lines
    [06:52:09.646] [INFO ] Found 282 shard files
    [06:52:09.784] [INFO ] Using index.json for fast startup (59585 tensors mapped)
    [06:52:09.802] [INFO ] safetensors parsed: E:\glm-5.2-bf16\model-00057-of-00282.safetensors (211 of 211 tensors accepted)
    [06:52:09.803] [INFO ] safetensors parsed: E:\glm-5.2-bf16\model-00241-of-00282.safetensors (211 of 211 tensors accepted)
    [06:52:09.804] [INFO ] safetensors parsed: E:\glm-5.2-bf16\model-00173-of-00282.safetensors (213 of 213 tensors accepted)
    [06:52:09.805] [INFO ] safetensors parsed: E:\glm-5.2-bf16\model-00118-of-00282.safetensors (213 of 213 tensors accepted)
    [06:52:09.806] [INFO ] safetensors parsed: E:\glm-5.2-bf16\model-00213-of-00282.safetensors (213 of 213 tensors accepted)
    [06:52:09.807] [INFO ] safetensors parsed: E:\glm-5.2-bf16\model-00125-of-00282.safetensors (213 of 213 tensors accepted)
    [06:52:09.808] [INFO ] safetensors parsed: E:\glm-5.2-bf16\model-00051-of-00282.safetensors (213 of 213 tensors accepted)
    [06:52:09.809] [INFO ] safetensors parsed: E:\glm-5.2-bf16\model-00002-of-00282.safetensors (212 of 212 tensors accepted)
    [06:52:09.810] [INFO ] safetensors parsed: E:\glm-5.2-bf16\model-00189-of-00282.safetensors (211 of 211 tensors accepted)
    [06:52:09.811] [INFO ] safetensors parsed: E:\glm-5.2-bf16\model-00080-of-00282.safetensors (213 of 213 tensors accepted)
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
| CMakeLists.txt | 0d2f0878a5b06d87b6adaebcda636039a4ff3f08bf8f1c06742e14e87bc13d18 |
| src/version.h | d3d4dc235be63d2220fb03b58a74be2573ddc5be6d9ac573db99800e515d3019 |
| tools/tokenizer_server.py | 190149a1fd882841fd7e849a623b9728c84240fa43c77c2dc6e6ced708f9e7be |

- `PASS` **100% gate: all quality gates PASS**
## Summary

- **Total checks**: 17
- **Passed**: 17
- **Failed**: 0
- **Skipped**: 0
- **Verification rate**: `100.00%` of runnable checks passed
