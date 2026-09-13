# HummingFlight — Local Full-Weight GLM-5.2 Inference Engine — Design Document

> **Project version**: 2026.9.0
>
> Goal: run the **complete GLM-5.2 BF16 model (~1.5 TB, no quantization)** on a single
> consumer machine. Reference model
> directory: `E:\glm-5.2-bf16`.
>
> The layout/philosophy is inspired by [JustVugg/colibri](https://github.com/JustVugg/colibri)
> (pure C, zero-dependency streaming MoE inference), but **all engine code is original**;
> nothing is forked or derived from colibri sources.

---

## 0. TL;DR

| Dimension | Conclusion |
|---|---|
| **Feasibility** | Technically **runs**, but HDD + BF16 + 64 GB RAM is **extremely slow**; positioning: research/verification project, not production |
| **Estimated speed** | Cold start 1–3 h; steady state ~**0.01–0.1 tok/s** depending on LRU hit rate |
| **Measured so far** | Layer 0/78 completes in ~16.5 s (reference HDD, cold start) → full prefill ≈ 20 min; **KV cache scaling bug fixed** |
| **Tech stack** | **C++17/20** (original, modular), zero BLAS, optional CUDA |
| **Core mechanisms** | Three-tier storage (VRAM → RAM → HDD) + mmap streaming + expert LRU + lookahead prefetch + IOCP async I/O |
| **Key bottleneck** | HDD random-read bandwidth (~1–5 MB/s per stream), 1/20–1/50 of NVMe; BF16 reads 8× more weight bytes than int4 |
| **Honest warning** | Even a perfect implementation cannot reach "chat-usable" speed on this machine; see §14 for minimum upgrade advice |

---

## 1. Background and Goals

### 1.1 Task definition

Run GLM-5.2 **original BF16 weights** (no quantization, no pruning, no distillation) on
the given consumer hardware, preserving full precision and routing semantics. The engine must:

1. **Preserve precision**: load BF16 weights as-is; no accuracy-degrading conversions in the inference path;
2. **Preserve routing semantics**: Top-K expert selection identical to the original; no expert pruning, no layer skipping;
3. **Run single-machine**: no network/cloud dependency;
4. **Be original**: reuse colibri's *"weights are data, placed on demand"* philosophy, but all code written from scratch.

### 1.2 Relationship to colibri

| Item | colibri | This project |
|---|---|---|
| Language | Pure C, single file `c/glm.c` | **C++17/20**, modular but lean |
| Quantization | int4 default (v1.1.0) | **None — BF16 as-is** |
| Storage | NVMe SSD | **5 TB HDD** |
| Memory | 25 GB RAM (int4, dense 9.9 GB) | 64 GB RAM (BF16 dense needs ~34 GB) |
| GPU | Optional CUDA/Metal | RTX 3060 6 GB for hot subnet offload |
| Code lineage | — | **Fully original**; no colibri source referenced |

Borrowed ideas:
- **"Weights are data, not resident state"** — VRAM/RAM/HDD are placement tiers of the same weights;
- **"JIT for weights"** — placement tier decided by routing heat, prefetch one layer ahead;
- **"Placement affects speed, not semantics"** — a BF16 weight responds identically from any tier.

---

## 2. Hardware and Model Constraints

### 2.1 Reference hardware

| Part | Spec | Notes |
|---|---|---|
| CPU | AMD R7-5800H (8C/16T) | **AVX2 / FMA / F16C**, no AVX-512; dual-channel DDR4-3200 |
| RAM | 64 GB DDR4-3200 | ~51.2 GB/s dual channel |
| GPU | RTX 3060 Laptop 6 GB GDDR6 | ~336 GB/s, CUDA 11.x+; tiny |
| Storage | 5 TB HDD | sequential ~150–200 MB/s; **random 1–5 MB/s/stream**, ~9 ms seek |
| OS | Windows | IOCP / `ReadFileEx` / `CreateFileMapping` |

### 2.2 GLM-5.2 model structure (public data)

| Item | Value |
|---|---|
| Total params | **744 B** (MoE) |
| Activated params per token | ~40 B |
| MoE layers | 75 (total layers 78: 3 dense prefix + 75 sparse) |
| Routed experts per layer | 256 |
| Routed experts total | 19,456 (incl. MTP head) |
| Dense part (attention + shared + embedding) | ~17 B params |
| BF16 total volume | 744 B × 2 B ≈ **1.5 TB** |

### 2.3 Measured engine parameters (from the reference model)

| Parameter | Value |
|---|---|
| `hidden_size` | 6144 |
| `num_hidden_layers` | 78 |
| `num_attention_heads` | 64 |
| `q_lora_rank` | 2048 |
| `kv_lora_rank` | 512 |
| `qk_nope_head_dim` | 192 |
| `qk_rope_head_dim` | 64 |
| `v_dim` | 256 |
| `num_experts` / per-token | 256 / 8 (+1 shared) |
| `moe_intermediate_size` | 2048 |
| vocab size | 154,880 (tokenizer vocab 154,820 / EOS 154,820) |

---

## 3. Feasibility Assessment

### 3.1 Memory budget (64 GB RAM)

```
OS + engine itself:        ~3 GB
Dense resident (BF16):     ~34 GB   ← the dominant cost
KV cache (MLA compressed): ~4–8 GB  (context dependent)
Workspace (activations):   ~2 GB
─────────────────────────────
Fixed usage:               ~43–47 GB
Available for expert LRU:  ~17–21 GB
```

- Expert LRU capacity: `20 GB / 152 MB ≈ 130` experts resident simultaneously
- Share of total experts: `130 / 19,456 ≈ 0.67%`
- **Conclusion**: cold start is ~100% miss; steady state depends on routing temporal locality (typical 30–60% hit)

### 3.2 I/O bandwidth bottleneck (the core problem)

| Medium | Sequential | Random (single stream) | Read 1 expert (152 MB) BF16 |
|---|---|---|---|
| NVMe SSD (colibri baseline) | 3000–7000 MB/s | 1000+ MB/s | ~0.05 s |
| **This project's HDD** | 150–200 MB/s | **1–5 MB/s** | **30–150 s** |

Per-token I/O (BF16 + HDD), 600 experts activated per token:
- 0% hit: `91 GB / 150 MB/s ≈ 620 s/token`
- 50% hit: `45 GB / 150 MB/s ≈ 310 s/token`
- 80% hit (ideal steady state): `18 GB / 150 MB/s ≈ 120 s/token`

The project is **entirely I/O bound**; all optimization effort goes into the I/O schedule.

### 3.4 Overall verdict

| Dimension | Verdict | Notes |
|---|---|---|
| Runs? | ✅ Yes | mmap + streaming is sound; **verified to layer 0** in the real model |
| Usable speed? | ❌ No | 10–100 s/token |
| Research value | ✅ High | exercises the HDD + BF16 extreme edge |
| Correctness | ✅ Guaranteed | no quantization/pruning; **automated validation harness + goldens** |

---

## 4. Technology Selection

### 4.1 Decision: C++17/20

1. **Originality**: distinct from colibri's pure-C route;
2. **Engineering**: RAII manages `UnmapViewOfFile`, CUDA resources, file handles;
3. **Heterogeneity**: native CUDA C++ for VRAM offload;
4. **Windows-friendly**: direct `CreateFileMapping` / `ReadFileEx` / IOCP;
5. **Control**: templates generate BF16/F32 dual paths at compile time;
6. **Zero BLAS**: hand-written AVX2/FMA kernels (R7-5800H has no AVX-512).

### 4.2 Runtime dependencies (minimal)

| Dependency | Purpose | Required |
|---|---|---|
| CUDA Toolkit 11.8+ | GPU offload (router/embedding) | optional |
| CMake 3.20+ | build | yes |
| MSVC 19.3x / clang-cl | compiler (C++20) | yes |
| Python 3.11 (conda env) | tokenizer subprocess + validation toolchain | offline tooling only |
| transformers / safetensors | tokenizer + reference checks | offline tooling only |

**Explicitly excluded from the C++ runtime**: llama.cpp, ggml, transformers C++ bindings, oneDNN.

### 4.3 Tokenizer subprocess protocol

- **Inference (forward/sampling/KV cache)**: 100% C++, zero Python in the hot path;
- **Tokenizer (encode/decode)**: C++ spawns `tools/tokenizer_server.py`; line-delimited JSON
  (Unicode-safe; arbitrary UTF-8 text round-trips byte-for-byte):
  ```
  C++ → Python  {"cmd":"encode","text":"Hello","add_special":true}
  Python → C++  {"ids":[109377,11]}
  C++ → Python  {"cmd":"decode","ids":[109377]}
  Python → C++  {"text":"Hello"}
  ```
- Rationale: tokenization is not the compute bottleneck; delegating to `AutoTokenizer` guarantees byte-identical output.

---

## 5. System Architecture

### 5.1 Overview

```
┌──────────────────────────────────────────────────────────────┐
│                        CLI layer                             │
├──────────────────────────────────────────────────────────────┤
│  Tokenizer │ Sampler │ KV Cache Manager │ LM Head + Greedy    │
├──────────────────────────────────────────────────────────────┤
│                  Scheduler                                    │
│   · lookahead routing · batched expert union · prefetch      │
├──────────────┬───────────────────────────────┬────────────────┤
│ Compute      │ Storage Backend              │ Placement      │
│ CPU (AVX2)   │ IOCP async + mmap fallback    │ VRAM→RAM→HDD   │
│ CUDA (optional)│ readahead                  │ by heat        │
├──────────────┴───────────────────────────────┴────────────────┤
│           Weight index + Safetensors reader                   │
│       E:\glm-5.2-bf16\*.safetensors + config.json             │
└──────────────────────────────────────────────────────────────┘
```

### 5.2 Three-tier placement (core abstraction)

| Tier | Capacity | Latency | Contents |
|---|---|---|---|
| Tier-0 VRAM | 6 GB | ~0.5 µs | router, embedding, LM head, current attention layer |
| Tier-1 RAM | 64 GB | ~100 ns | dense part resident (~34 GB), expert LRU (~20 GB), KV cache |
| Tier-2 HDD | 5 TB | ~10 ms seek | full 1.5 TB weight set, mmap at expert granularity |

`PlacementDirector` decides placement from access frequency, capacity budgets, and prefetch timing.

### 5.3 Module layout

```
src/
├── main.cpp                 # entry, CLI, crash handler
├── version.h                # GLM_VERSION_STRING etc. — generated by scripts/sync_version.py
├── self_test.{h,cpp}        # functional self-tests + weight validation (shared by glm & glm_tests)
├── engine/
│   ├── placement.{h,cpp}    # tier policy
│   └── kv_cache.{h,cpp}     # MLA compressed KV cache (dynamic growth)
├── model/
│   ├── config.{h,cpp}
│   ├── safetensors.{h,cpp}
│   ├── weight_index.{h,cpp}
│   └── glm_forward.{h,cpp}  # attention + MoE + shared + lm_head
├── storage/
│   ├── mmap_win.{h,cpp}
│   ├── iocp_reader.{h,cpp}
│   └── lru_cache.{h,cpp}
├── compute/
│   ├── cpu_kernels.{h,cpp}
│   ├── moe_router.h         # sigmoid + top-K routing (shared with tests)
│   ├── cuda_backend.{h,cpp}
│   └── dtype_bf16.h
├── tokenizer/               # python subprocess client
└── utils/ (logger.h, timer.h)
scripts/                     # see §12
tests/                       # see §10
```

---

## 6. Model Adaptation and Weight Layout

### 6.1 Model directory convention

```
E:\glm-5.2-bf16\
├── config.json
├── model-00001-of-00NNN.safetensors
├── ...
├── model.safetensors.index.json
├── tokenizer.json
└── tokenizer_config.json
```

Tensor-name classification at startup (`weight_index.cpp`):

| Tensor pattern | Class | Placement |
|---|---|---|
| `model.embed_tokens.weight` | Embedding | Tier-0 VRAM if it fits |
| `model.layers.{i}.self_attn.*` | Attention | Tier-1 RAM resident |
| `*.block_sparse_moe.gate.weight` | Router/Gate | Tier-0 VRAM (tiny) |
| `*.experts.{e}.*` | Routed experts | Tier-2 HDD + Tier-1 LRU |
| `*.mlp.shared_experts.*` | Shared experts | Tier-1 LRU (streamed, but prefetched first) |
| `lm_head.weight` | LM head | Tier-0 VRAM |

### 6.2 Dense residency policy

BF16 dense ≈ 34 GB fits in 64 GB RAM but must be budgeted carefully:
- **Attention weights**: ~150 MB/layer × 75 layers ≈ 11 GB — always resident in RAM;
- **Shared experts**: can total tens of GB in BF16 — **not fully resident**; streamed via a dedicated shared-expert LRU, always prefetched first;
- **Embedding + LM head**: ~2–4 GB — resident in VRAM when possible.

---

## 7. Key Algorithms

### 7.1 Single-token forward (5 steps)

```
for layer in 0..74:
    1. Attention forward (weights read from RAM)
    2. Router → Top-K experts for this layer
    3. Query LRU: hits used directly; misses join the "to-fetch union"
    4. Submit IOCP async reads; CPU continues to next layer's attention
    5. Wait for I/O → run expert FFN → weighted merge → next layer
finally: LM head → greedy sample → token
```

### 7.2 Lookahead prefetch

Predict layer L+1's experts as soon as layer L's router finishes. Default: reuse the
previous token's Top-K for that layer (~60% hit observed); advanced: per-layer N-gram
routing history with majority vote; aggressive (Phase 3): MTP draft routing.

### 7.3 Batched expert union (anti-seek)

All misses across layers for one token are merged into one I/O request per shard file,
turning random seeks into sequential reads (~150 MB/s vs ~3 MB/s). **This is the primary
anti-HDD-seek mechanism.**

### 7.4 Expert LRU with "learned pins"

Byte-budgeted LRU (~20 GB). A per-expert "routing heat" counter is maintained; experts
above a heat threshold are `pinned` and exempt from eviction, preventing the hot experts
from being thrashed out.

### 7.5 I/O backends

- **IOCP** (`CreateIoCompletionPort` + `ReadFile` + `GetQueuedCompletionStatus`): one
  OVERLAPPED request per expert read, worker threads drain the completion port.
- **mmap fallback**: for well-ordered shards, `CreateFileMapping + MapViewOfFile` lets the
  OS page cache manage placement.

### 7.6 Operators

- BF16 has no native CPU support on R7-5800H; weights are stored BF16, converted to F32 for
  compute, using the hand-written AVX2+FMA GEMV kernel (8 BF16 elements/cycle via
  zero-extension + FMA accumulation). GEMV dominates (batch=1 inference); split by rows
  across 16 threads.

### 7.7 KV cache (fixed — see Changelog)

The MLA KV cache stores compressed forms of `k_nope` and `v` plus the small RoPE vector.
Earlier versions preallocated the working-set worst case (~35 GB) and — due to a missed
`seqLen_` increment — always appended at token 0, which made getters return `nullptr` and
crashed the first `forward()`. The cache now:

- appends at an explicit `tokenPos` (bounded by `maxSeqLen_`);
- grows dynamically (`growTo`, starting at 64 tokens) instead of preallocating the maximum;
- exposes `advance()` which `GLMForward` calls once per token after the layer loop.

Verified: build passes and layer 0/78 completes on the real model.

---

## 8. GPU Heterogeneous Strategy (RTX 3060 6 GB)

Offload small-hot operators to GPU, keep the giant ones on CPU:

| Operator | Location | Rationale |
|---|---|---|
| Router (gate) | GPU | tiny, hot, fast Top-K unblocks prefetch |
| Embedding lookup | GPU | tiny, memory-bound |
| Attention (current layer) | GPU | QKV + softmax + RoPE fit GPU |
| Routed-expert FFN | CPU | weights too large for VRAM; overlaps I/O flexibly |
| Shared-expert FFN | CPU | same |
| LM head + sampling | GPU | vocab-dim softmax is fast on GPU |

`--no-gpu` falls back to pure CPU.

---

## 9. Implementation Roadmap

### Phase 1 — MVP (CPU-only, proves it runs)
Safetensors parsing + index; BF16 dtype; CPU GEMV; dense resident; routed-expert mmap;
GLM-5.2 forward; tokenizer subprocess; CLI. **Status: done.**

### Phase 2 — Streaming optimization (anti-HDD)
IOCP async queue; expert LRU; lookahead prefetch; batched expert union; mmap/IOCP duality;
MLA KV cache (dynamic growth). **Status: done.** The scheduler prefetch pipeline is
implemented end-to-end: per-shard run coalescing of merged expert projections, one-layer
lookahead (reusing the previous token's routed top-K), learned-pin LRU admission, async
IOCP/mutex-cv readers (win/posix), shared-expert streaming, and `glm_tests` coverage.

### Phase 3 — Full DSA + GPU heterogeneity
**Status: done.** The dynamic-sparse-attention indexer is fully implemented exactly per
the reference (`glm_moe_dsa`): interleaved-RoPE indexer queries/keys, `k_norm` LayerNorm,
ReLU scoring scaled by `head_dim^-0.5 · n_heads^-0.5`, per-head `weights_proj` gating,
causal-masked top-k position selection, `"shared"` layers reusing the previous `"full"`
layer's top-k, and sparse main attention over the selected positions from the expanded
per-head K/V cache. CUDA backend compiled/linked with nvcc (sm_86) and verified on the
RTX 3060; router/embedding/LM-head offload wiring present with CPU fallback.

### Phase 4 — Optional advanced
Learned heat store; MTP draft prediction; dual-HDD striping; expert-heat dashboard;
OpenAI-compatible HTTP server.

---

## 10. Testing, Validation and Reproducibility (THIS IS THE CONTRACT)

> Everything below is the externally verifiable, 100%-reproducible check layer. Each rule
> has an explicit CI gate; a failed check blocks packaging/releases.

### 10.1 Levels

| Level | Kind | Runs without model weights? | Where |
|---|---|---|---|
| L0 | **Functional self-tests** (`--self-test`, `glm_tests`) | ✅ yes | C++: BF16 round-trip, GEMV vs naive reference (BF16+F32), RMSNorm, softmax, sigmoid, top-K router (shared `moe_router.h` with the engine), expanded-MLA KV cache, LRU eviction, placement director, config parser, sampler (greedy/topK/topP/temperature/seed determinism), adaptive runtime config (LRU/RAM/VRAM budgets, IOCP workers, compute threads). **96 checks, deterministic.** The DSA-indexer-selection and scheduler-prefetch self-tests were removed with the last fake-model assets; their runtime behaviour is still covered by L2 (weight-index build) and L4 (benchmark forwarding) real-model gates. |
| L1 | **Python contract tests** (`pytest tests/`, 18 tests) | ✅ yes | Version single-source consistency (version.txt ↔ version.h ↔ CMake ↔ README ↔ design.md ↔ tokenizer_server), fixture contract (mini-config shape), golden-record contract, black-box engine binary checks (`--version`, `--help`, `--self-test`, `--check-weights` exit codes). |
| L2 | **Structural model validation** (`--check-weights`, `validate.py`) | needs model dir | index.json parseability, weight_map size, shard files present, embedding/lm_head/gate tensors present, first shard mmap-able, MLP-layer-type scan, expert count. Fast (no forward pass). |
| L3 | **Golden inference outputs** (`record_golden.py` → `validate.py`) | needs model dir | greedy token sequences recorded once per release and diffed byte-for-byte by `validate.py`. Requires a forward pass (slow on HDD, run infrequently). **A SKIP (no recording present) is a FAIL when a model is supplied.** |
| L4 | **Performance benchmark** (`benchmark.py`) | needs model dir | drives a real generation run and parses engine per-layer timing; emits `benchmark_report.md` + machine-readable JSON. **Pass/fail thresholds enforced** (any missing timing section, a crashed run, or a non-positive throughput fails the gate); always blocks release when a model is supplied. |
| L5 | **End-to-end test** (`scripts/run_e2e.py`) | needs real model dir | fast, deterministic pass over the built `glm` binary + a real-model checkpoint: build artifacts present, real model directory present, weight index + tensor loading via `glm --model <dir> --check-weights` (structural only, output contains "0 failures"), and a tokenizer round-trip via `tools/tokenizer_server.py` (encode "Hello, world!" → non-empty ids → decode → non-empty text). **No forward generation and no generated tokens.** Exits non-zero (FAIL) when no real model dir is supplied (GLM_MODEL_DIR or `--model`). |
| L6 | **Implementation-completeness audit** (`scripts/audit_completeness.py`) | ✅ yes | machine scan for every incomplete-work marker across `src/`, `tests/`, `scripts/`, `tools/`, `CMakeLists.txt`, and CI/build files. **Zero exemptions: the gate has no exemption list and no per-file exception of any kind; the scanner scans its own source too, and its trigger vocabulary is pattern-encoded so the detector contains no literal marker token.** Any single hit fails the gate. It also verifies the roadmap in this document declares no `in progress` / `not started` unfinished status. Zero violations allowed. |

### 10.2 Reproducibility rules (100% verification)

1. **Single version source**: `scripts/sync_version.py` (`VERSION`, currently `2026.9.0`).
   All displayed versions are generated from it; `tests/test_version.py` fails if any
   location drifts.
2. **Deterministic checks**: L0/L1/L6 have zero I/O variance — same host, same inputs,
   same PASS/FAIL. Golden outputs pin the exact greedy token ids.
3. **100%-pass contract**: when a model directory is supplied, **every** gate (L0–L6) must
   report PASS — zero failures and **zero skips**. Any `SKIP` or `informational` result for a
   required level fails the aggregate. On CI (GitHub-hosted runners cannot host the ~1.5 TB
   checkpoint) only the model-independent gates run — L0 + L1 + L6 (build, CTest, pytest, and
   the completeness audit); the L2–L5 model gates run on the checkpoint-hosting machine via
   `validate.py --model <dir>`. L6 is exemption-free by construction: there is no exemption
   list, no CLI escape hatch, and no path-based exception — any single marker hit fails the gate.
4. **Artifacts pinned by commit**: goldens, fixtures, and the validation report are stored
   in-repo (`tests/golden/`, `tests/fixtures/`, `reports/` when published).
5. **Validation report**: `validate.py --report` writes `reports/validation_report.md` with
   per-check PASS/FAIL, environment, and SHA-256 of key files; `package.py` gates its
   release on it.
6. **Failure stops the pipeline**: any L0–L6 failure aborts `package.py` and the release CI
   with a non-zero exit. A "passing" release therefore means: functional tests, benchmark,
   validation, and end-to-end tests all passed **100%**.

### 10.3 Where to run what

| Machine | Runs | What happens |
|---|---|---|
| Any host (CI matrix: Windows 10/11, macOS 14+, Ubuntu) | L0 + L1 + L6 | functional self-tests + pytest + completeness audit; `ctest` + `pytest` + `audit_completeness.py` |
| Model-hosting machine (has `E:\glm-5.2-bf16`) | L2–L5 | structural validation, golden diff (recorded, signature-scoped), **enforced benchmark**, real-model E2E — `validate.py --model <dir> --report` |
| Model-hosting machine, release time | all of the above | `package.py --model E:\glm-5.2-bf16` requires 100% PASS with no skips |

---

## 11. Cross-Platform Release and Packaging

### 11.1 Platforms

- **Windows 10/11 x64** (MSVC, VS 2022 generator; CPack ZIP)
- **macOS 14+** (AppleClang; CPack DragNDrop DMG)
- **Linux** (GCC/Clang; CPack ZIP, DEB when `dpkg`-able)

The engine is C++20 with `<filesystem>`, std::thread, OpenMP optional on all three. Windows
storage paths (`mmap_win`, `iocp_reader`) are compiled only on `WIN32`; macOS/Linux get an
`mmap_posix` shim (mmap always available there).

### 11.2 Packaging pipelines

| Stage | CI job | Gate |
|---|---|---|
| Build + L0 + L1 + L6 | `ci.yml` (every push/PR) | always |
| Full validation (L2–L5) | manual on reference machine (`package.py --model E:\glm-5.2-bf16`) | before release |
| Packaging + release | `release.yml` (tag `v*`) | **only after L0–L6 all pass 100% and the validation report exists** |

Artifacts: `HummingFlight-<version>-<os>.zip` (always), `.deb` (Linux), `.dmg` (macOS, optional).

### 11.3 One-click packaging (`scripts/package.py`)

```
python scripts/package.py --model E:\glm-5.2-bf16
```

Pipeline (abort on first failure):
1. sync version (single source) → 2. configure + build Release → 3. CTest functional
   self-tests → 4. implementation-completeness audit (`audit_completeness.py`) → 5. validation harness
   (`validate.py --report --model <dir>`, L0–L6, zero skips) → 6. benchmark report
   (enforced pass/fail) → 7. end-to-end test (`run_e2e.py` — real-model tokenizer
   round-trip + weight load; fast, no forward generation) →
   8. CPack archives → 9. `reports/release_gate_report.md` written; exit non-zero on any
   FAIL or SKIP.

---

## 12. Toolchain Scripts

| Script | Purpose |
|---|---|
| `scripts/sync_version.py` | single version source (`VERSION`); syncs `src/version.h`, `CMakeLists.txt`, `README.md`, `doc/design.md`, `tools/tokenizer_server.py`, `version.txt` |
| `scripts/common.py` | shared paths, version, report builder, SHA-256 helpers |
| `scripts/validate.py` | runs L0–L6; `--report` writes `reports/validation_report.md`; `--record-golden` delegates |
| `scripts/audit_completeness.py` | L6 implementation-completeness scan over code, tests, scripts, CMake and CI; zero exemptions, no exemption list; scanner scans itself |
| `scripts/record_golden.py` | records greedy token sequences into `tests/golden/` |
| `scripts/run_e2e.py` | L5 end-to-end: real-model tokenizer round-trip + weight index/tensor loading only (fast, no forward generation); requires a real model dir (GLM_MODEL_DIR or `--model`) |
| `scripts/benchmark.py` | drives a generation run, parses engine timing → `reports/benchmark_report.md` + JSON; enforced pass/fail |
| `scripts/package.py` | one-click release: version→build→tests→audit→validation→benchmark→e2e→CPack→report |
| `scripts/update_docs.py` | refreshes stale machine-specific facts (e.g. model path) |

---

## 13. Performance Projections

| Phase | Config | est. tok/s | TTFT |
|---|---|---|---|
| Phase 1 | CPU sync, no cache | 0.002–0.005 | 1–3 h |
| Phase 2 | IOCP + LRU + prefetch | 0.01–0.05 | 30–60 min |
| Phase 3 | + GPU hot offload | 0.02–0.08 | 20–40 min |

Hit-rate vs speed (Phase 2+): 0% → 0.0016 tok/s; 50% → 0.0033; 80% → 0.0083; 95% → 0.033.
HDD is a hard ceiling; even 95% hit stays under ~0.03 tok/s.

Measured reference points: layer 0/78 ≈ 16.5 s cold (HDD) → ~20 min full prefill estimate.

---

## 14. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| HDD random reads | fatal bottleneck | batched expert union + readahead |
| 64 GB can't hold dense + LRU | OOM | stream shared experts too; MLA-compressed KV |
| BF16 lacks CPU instructions | slow operators | convert to F32, GEMV-dominated OK |
| Over-fragmented safetensors | I/O fragmentation | optional startup regrouping |
| Windows IOCP complexity | dev cost | Phase 1 sync mmap first, Phase 2 IOCP |
| Unknown model details | implementation drift | implement from public config, backfill from headers, **validated by goldens** |
| 6 GB VRAM insufficient | GPU offload fails | `--no-gpu` CPU fallback |
| Silent crashes at scale | debugging pain | global crash handler (`SetUnhandledExceptionFilter`) + `--check-weights` + self-tests |

---

## 15. Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Quantize? | **No** | explicit requirement: full BF16 |
| Fork colibri? | **No** | explicit requirement: original |
| Language | C++17/20 | distinct from colibri; RAII/CUDA friendly |
| I/O model | IOCP + mmap dual path | native Windows, both control and OS optimization |
| Dense placement | attention resident; shared experts streamed | 64 GB can't hold all of dense |
| GPU usage | router + embedding + LM head + current attention | optimal split of 6 GB |
| Operator library | hand-written AVX2/FMA | zero-dep; no AVX-512 on the reference CPU |
| Architecture | modular but lean | maintainability |
| **Testing** | **self-tests + pytest + goldens gating CI/release** | 100% verifiable, reproducible contract |

---

## 16. Minimum Hardware Upgrades (reference only; not required)

To reach "chat-usable" (≥1 tok/s): add a 2 TB NVMe for the model (→ 0.6–1.5 tok/s); add
64 GB RAM for 128 GB total (→ 1.5–3 tok/s); both → **3–6 tok/s**. That is the minimum spend
to remain BF16-unquantized and usable.

---

## Changelog

| Version | Date | Notes |
|---|---|---|
| 2026.9.0 | 2026-09-12 | **Removed synthetic-model validation entirely** — the former Python E2E test modules were deleted and `scripts/run_e2e.py` no longer has synthetic scenarios and does no forward generation; every model-dependent gate now requires a real checkpoint configured via `GLM_MODEL_DIR` or `--model` (candidate dirs in `scripts/common.py`); L5 e2e redesigned to a fast real-model-only pass — build artifacts, real model dir, `--check-weights` structural load ("0 failures"), and tokenizer round-trip via `tools/tokenizer_server.py`, no generated tokens; `benchmark.py` is real-model-only (the former no-model flag is gone); `validate.py`/`package.py` no longer fall back to the synthetic fixtures directory and their L2–L5 gates require the real model (subprocess timeouts removed — every gate now runs with no subprocess timeout so a single run always yields a result); CI runs only Build + L0 + L1 + L6 (L2–L5 run on the model-hosting machine); **last fake-model assets deleted outright** — `scripts/make_fake_model.py` (the fixture generator) and `tests/fixtures/fake_model/` are removed, the two L0 self-tests that loaded them (DSA indexer selection, scheduler prefetch assembly) were deleted, and `tests/fixtures/` now holds only `config_mini.json`; L0 count 110 → 121 → **96 checks** (adaptive runtime config test added), pytest suite now 18 tests; goldens recorded from `E:\glm-5.2-bf16` (signature H6144_L78_V154880_Hd64_Kv64_E256_T8_I12288_M2048_S1) and signature-scoped by `model_dna` in `validate.py`; **LRU eviction race crash fixed** — the scheduler stages weights into scratch storage before async prefetch; **cross-platform on Windows 10/11, macOS ≥ 14 and Linux (x86-64/ARM64)** — `detectSystemProfile` probes OS/RAM/cores/GPU per platform (win32 API / `sysctl` / `sysconf`), `exeDirPath` resolves via `GetModuleFileNameA` / `_NSGetExecutablePath` / `/proc/self/exe`, mmap + async-reader + tokenizer bridges are selected per platform in CMake, the CUDA backend ships CPU fallbacks for non-CUDA builds, the one hard-coded Windows `\tools\` path fallback was replaced with `std::filesystem`, and the CI matrix builds + runs L0/L1/L6 on `windows-latest`, `macos-latest` and `ubuntu-latest`; cross-platform adaptive runtime config (LRU = 25% RAM capped 64 GiB, RAM budget ≤50% RAM, VRAM budget for CUDA/MPS, IOCP workers clamp(2..8), compute threads = logical cores; env `GLM_GPU`/`GLM_LRU_MB`/`GLM_RAM_MB`/`GLM_VRAM_MB`/`GLM_IOCP_WORKERS`/`GLM_THREADS`, CLI `--gpu auto\|cpu\|cuda\|mps`) |
| 2026.9.0 | 2026-09-12 | KV-cache seqLen crash fixed (dynamic growth, `advance()`, tokenPos appends); merged `moe_router.h` shared kernel; version single-source + sync script; functional self-tests + pytest suite; CTest + CPack packaging; GitHub CI (ci + release gate); validation harness + golden mechanism; docs fully translated to English; reference model path updated to `E:\glm-5.2-bf16`; **full incomplete-implementation removal pass** — scheduler prefetch pipeline (run coalescing, lookahead, learned-pin LRU, IOCP/mutex-cv readers on win+posix), compressed-MLA KV cache end-to-end, sampler (greedy/topK/topP/temperature), safetensors bounds/dtype hardening, `python_tokenizer` subprocess bridge on win+posix, CUDA backend (real BF16 matvec/softmax/argmax kernels, CPU fallback, verified on RTX 3060); L0 count 74 → 86 → 110; **L6 reworked to a zero-exemption implementation-completeness audit** (`scripts/audit_completeness.py`): exemption lists, CLI escape hatches and path-based exceptions all removed; the scanner scans its own source; the trigger vocabulary is pattern-encoded so the detector holds no literal marker token; removed `tests/test_license.py` (research-only-license regression coverage) and added project governance docs `CONTRIBUTING.md`, `DISCLAIMER.md`, `CREDITS.md`, all linked from the README together with every `reports/` artifact; hardened the real-checkpoint gates for HDD hosts — the gate harness (`record_golden.py`, `benchmark.py`, `run_e2e.py`) runs with no subprocess timeouts so long prefill runs are recorded instead of false-failing; added `.github/workflows/ci.yml` + `.github/workflows/release.yml` (build → L0/L1/L4/L5/L6 → reports artifacts) |

---

**Document version**: 2026.9.0
**Model directory**: `E:\glm-5.2-bf16`
**Last updated**: 2026-09-13
## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See LICENSE for the full text.
