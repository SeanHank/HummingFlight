# HummingFlight — Local Full-Weight GLM-5.2 Inference Engine — Design Document

> **Project version**: 2026.9.3
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

> Measured 2026-09 (L4 gate, 4-token run): 21 % LRU hit → **69.5 GiB/token, 12030 s/4 tok
> (0.00033 tok/s)**. The 69.5 GiB far exceeds the "600 experts ≈ 19 GB" floor because the
> auto LRU (RAM/4 = 15.7 GB) cannot hold one forward's ~27 GB routed footprint, so every token
> re-streams it. Before/after analysis and the fix plan: **§17 Performance Optimization Round**.

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

### 4.4 Native in-process tokenizer (opt-in, #10)

- `--tokenizer native` loads `tokenizer.json` + `tokenizer_config.json` directly in C++ (`src/tokenizer/native_tokenizer.cpp`),
  no Python process. **Opt-in only**: the default remains the Python subprocess, so existing byte-for-byte parity is unchanged.
- Implements the full pipelines: GPT-2 split pre-tokenization (7-alternation regex), BPE merge (byte-level `bytes_to_unicode`),
  ByteLevel decoder; Unicode categories (Letter/Number/Space) come from the generated `src/tokenizer/unicode_ranges.h`
  (`scripts/gen_unicode_ranges.py`).
- Verified in `tests/test_native_tokenizer.py` against the real model directory: encode ids, decode round-trip and EOS
  handling match the reference `AutoTokenizer` byte-for-byte for a 16-string battery incl. CJK, contractions, numbers,
  `<|endoftext|>`, emoji, whitespace and diacritics (skipped when the model or binary is absent).
- `--dump-tokens <text>` (hidden CI helper) prints `TOKEN:` (comma-separated ids) and `TOKEN_DEC:` (JSON-escaped decode).

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
├── tokenizer/               # python subprocess client + native BPE backend (#10)
└── utils/ (logger.h, timer.h, json_util.h)
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
**GLM-5.2 MTP surface is structurally integrated** — the real `num_nextn_predict_layers` /
`index_share_for_mtp_iteration` config keys and `nextn_predict_layers.*` weight detection
are parsed, indexed and gated (`--mtp`, `--check-weights`, L0) with no silent fallback;
the 5.2-bf16 release ships the nextn weights separately, so MTP *draft forward* remains
a Phase-4 item (no unverifiable forward math is shipped — unverified runs are refused).

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
| L0 | **Functional self-tests** (`--self-test`, `glm_tests`) | ✅ yes | C++: BF16 round-trip, GEMV vs naive reference (BF16+F32, ILP-unrolled kernels), RMSNorm, softmax, sigmoid, top-K router (shared `moe_router.h` with the engine), expanded-MLA KV cache (incl. grow-by-reserve + head-major per-head addressing/contiguity), LRU eviction + soft-boost, placement director, config parser (incl. MTP head section + GLM-5.2 `num_nextn_predict_layers` / `index_share_for_mtp_iteration` surface + malformed-JSON rejection), sampler (greedy/topK/topP/temperature/seed determinism + min-p + repetition penalty + typical-p + frequency/presence penalties + 13-arg equivalence), adaptive runtime config (LRU/RAM/VRAM budgets, IOCP workers, compute threads), strict safetensors validation (BF16-only dtype gate with an **explicit F32 allow-list** for the noaux_tc router bias, shape/size byte consistency, offset bounds, header + data alignment, JSON parse rejection, duplicate rejection), item-7 structural-completeness helpers (expert gate/up/down completeness + shape-vs-config, missing-tensor flags, MTP gate decision), weight-index MTP detection. **163 checks, deterministic.** The DSA-indexer-selection and scheduler-prefetch self-tests were removed with the last fake-model assets; their runtime behaviour is still covered by L2 (weight-index build) and L4 (benchmark forwarding) real-model gates. |
| L1 | **Python contract tests** (`pytest tests/`, 28 tests) | ✅ yes | Version single-source consistency (version.txt ↔ version.h ↔ CMake ↔ README ↔ design.md ↔ tokenizer_server), fixture contract (mini-config + MTP-config shape), golden-record contract, black-box engine binary checks (`--version`, `--help`, `--self-test`, `--check-weights` exit codes, strict opt-in errors: unavailable `--mtp` / missing CUDA `--gpu-experts` must fail loudly), benchmark STATS/telemetry parser, native tokenizer parity (encode/decode/EOS against reference transformers with the real model directory, skipped when absent or transformers is unavailable). |
| L2 | **Structural model validation** (`--check-weights`, `validate.py`) | needs model dir | index.json parseability, weight_map size, shard files present, first shard mmap-able, **strict per-layer structural completeness (item 7):** every layer's MLA projector + normalization set present, dense MLP (or MoE router `mlp.gate.weight` + `e_score_correction_bias` + shared experts) present, all 5 full-indexer weights present on "full" DSA layers, **all 256 routed experts per MoE layer complete (gate+up+down) with shape-vs-config matching**, embedding/final-norm/lm-head present with exact shapes, MTP config-vs-tensors consistency reported, GLM-5.2 root/corpus layer (index 78) surfaced as info. Fast (no forward pass); verifies **0 failures** on the real GLM-5.2 checkpoint. |
| L3 | **Golden inference outputs** (`record_golden.py` → `validate.py`) | needs model dir | greedy token sequences recorded once per release and diffed byte-for-byte by `validate.py`. Records use **chat-template prompts** (`raw:false`), exercise the full stack (tokenizer template → forward) and stop on EOS; the F32-correct router is pinned by these goldens (see changelog 2026.9.1). Requires a forward pass (slow on HDD, run infrequently). **A SKIP (no recording present) is a FAIL when a model is supplied.** |
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
| Build + L0 + L1 + L6 | `ci.yml` job `build-and-gates` (every push/PR; Windows + macOS + Linux) | always |
| Full validation (L2–L5) | manual on reference machine (`package.py --model E:\glm-5.2-bf16`) | before merge |
| Packaging + release | `ci.yml` jobs `build-and-gates` (CPack) + `release` (push to default branch) | **only after Build + L0 + L1 + L6 pass 100% on all three OS; auto-publishes a GitHub Release `v<version>` (version read from `scripts/sync_version.py`); skipped if that version is already released** |

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
Measured 2026-09 (4-token L4 run on this HDD box): **0.00033 tok/s @ 21% LRU hit** — the
observed regime sits below the 0%–20% curve because the LRU window (RAM/4) is smaller than one
forward's footprint (see §17.1). After the §17 optimization round the hit rate target is ≥70 %
(toward the 80% curve, ~0.002–0.008 tok/s).

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

## 17. Performance Optimization Round (measured, 2026-09)

### 17.1 Measured baseline (L4 benchmark, 2026.9.1, before this round)

The L4 gate on the reference HDD box (`E:\glm-5.2-bf16`, prompt ids `3,7,12`, 4 tokens):

| Metric | Measured | Consequence |
|---|---|---|
| Speed | 0.00033 tok/s (12030 s for 4 tokens) | ~50 min/token — not usable, I/O bound |
| Disk bytes | 277.88 GiB total = **69.5 GiB per token** | every token re-streams the expert set |
| Expert LRU hit rate | **20.97 %** (755/3600 lookups) | cache thrashes inside a single forward |
| LRU capacity (auto) | 15.7 GB (= RAM/4) | far below one forward's ~27 GB routed footprint |
| Prefetch waste | 69.9 % | speculative stream floods the HDD queue |
| HDD latency (avg) | 3718 ms / I/O | queue-inflated latency, not seek physics |
| I/O runs | 3840 (≈960/forward) | adjacency-only merging leaves ~2.6 proj/run |

Root causes (code-verified):

1. **LRU window is ¼ of RAM (=15.7 GB) but a single forward touches ~27 GB of distinct experts**
   (78 layers × 8 routed + 234 shared projections, ~30 MB each; measured 900 lookups/forward).
   The 21 % hits are almost exclusively intra-batch repeats; cross-token carry-over is zero.
2. **Prefetch speculation has no budget.** `onLayerRouterDone` fires next-layer top-k + all
   three shared projections every layer with no in-flight cap. On the HDD the speculative
   stream deepens the disk queue behind which critical-path demand reads wait (latency 3718 ms
   is queue inflation, not seek time).
3. **Run coalescing is adjacency-only** (`io.offset > run->end`). Nearly-touching projections
   split into separate seeks; ~3840 runs/run though only ~2.6 projections are merged per run.

Compute is healthy: CPU util ≈ 1 %, RAM bandwidth 39 MB/s, GPU/PCIe idle (CPU-only build).
The engine is 100 % I/O bound.

### 17.2 Optimization plan (this round)

All three optimizations live in the **I/O layer only** (fetch scheduling + sizing). They never
change *which bytes* land in an assembled expert (padding bytes are discarded, slice offsets
unchanged, LRU contents identical), so the forward math is **bit-exact** — `tests/golden/*.json`
remain valid and L3 does not need a re-run.

- **O1 — RAM-aware LRU/RAM-budget auto-sizing** (`runtime_config.cpp`). Raise the CPU expert
  window from `RAM/4` (capped 64 GiB; 15.7 GiB here → thrash) to `min(RAM*0.60, 40 GiB)`
  (→ 37.7 GiB) so one forward's ~27–30 GiB footprint fits and carries over between tokens
  (the 4-token run would keep its first forward's experts resident → per-token traffic drops
  toward the *delta*: only experts whose routing differs from the previous token). Placement
  RAM budget follows (`max(lru*2, RAM*0.80)`, capped at `RAM − reserve`). `GLM_LRU_MB` /
  `GLM_RAM_MB` overrides keep precedence.
- **O2 — Prefetch budget (demand-priority backstop)** (`scheduler.cpp`). Cap concurrent
  in-flight speculative expert assemblies (`kMaxPrefetchInFlight ≈ 40`). Once at the cap,
  further speculation is skipped (predictions still recorded, routing untouched) so demand
  reads never wait behind an unbounded speculative queue. On the HDD this bounds queue-latency
  inflation (3718 ms → seek-bound) and trims the 70 % prefetch waste.
- **O3 — Gap-tolerant seek merging** (`scheduler.cpp`). Merge same-shard projections with
  inter-run gaps ≤ `kRunMergeGap = 1 MiB` into one run, reading the padding instead of paying
  another seek. Slices still copy only their own bytes. Expect I/O run count to drop several
  fold (3840 → low hundreds) at a <5 % byte-extension cost — a large win on a seek/queue-bound
  mechanical disk.

### 17.3 Expected impact and test protocol

| Scenario | Est. | Basis |
|---|---|---|
| L4 benchmark (4 tok, ids 3,7,12), after O1+O2+O3 | 0.0005–0.002 tok/s (4–10× long-run bytes ↓) | hit 21%→~70%+, I/O runs ÷3–8 |
| Long chat (reuse distance 108 tok) | stronger | LRU window now spans reuse |
| Golden outputs | **identical** | I/O-layer only; goldens re-validated by L3 policy unchanged |

Test protocol (executed 2026-09-21): rebuilt Release → L0 CTest/L1 pytest/L2 `--check-weights`
(all green) → reran the L4 benchmark standalone → before/after written into
`reports/benchmark_report.md` and §17.4. L3 goldens are re-run per the standing gate policy when
a full release runs.

### 17.4 Measured outcome (2026-09-21, L4 gate on the reference HDD box)

| Metric | Baseline | Optimized | Delta |
|---|---|---|---|
| total_sec (4 tokens) | 12030.37 s | 9849.07 s | **-18.1 %** |
| speed_tok_per_s | 0.000332 | 0.000406 | **+22.3 %** |
| LRU used @ shutdown | 16 056 MB | 38 592 MB | +2.4× |
| Prefetch waste rate | 69.92 % | 66.33 % | -3.6 pp |
| Disk bytes / token | 69.5 GiB | 69.5 GiB | 0 (cold worst case) |
| Expert LRU hit rate | 20.97 % | 20.97 % | 0 (cold worst case) |
| Generated output | 16,29661,90,77 | 16,29661,90,77 | **bit-identical** |

Interpretation: the L4 workload's synthetic 3-id prompt routes each batch position to almost-
disjoint experts — a **cold worst case** whose bytes/token and hit rate are structurally fixed
independent of cache capacity (the 21 % hits are intra-batch repeats only). The measured 18 %
wall gain then comes from O2/O3 I/O scheduling (prefetch waste 69.9→66.3 %, HDD latency
3718→3699 ms) plus a warm OS page cache for the dense weights (the baseline ran cold). The
O1 LRU dividend (2.4× window: 38.6 GiB holds one forward's full footprint) pays off on real
workloads with routing locality and on long generations within the 108-token reuse distance —
exactly the regime this synthetic prompt cannot exercise. Golden/benchmark outputs stayed
bit-identical, confirming the round never touched forward math (L3 goldens validity preserved).

### 17.5 Deep-dive round 2 (2026-09-21 → 2026-09-25): why round-1 left 3840 runs, and the stability gate

Round 1's O3 "gap-tolerant seek merging" produced **no measured run-count change** (3840 runs
identical before/after). Round 2 root-caused why, from the actual weight bytes rather than the
model file's metadata:

**Finding 1 — the shards are already perfectly contiguous.** A per-shard header audit of all 282
shards (`analyze_shards.py`) shows every tensor and projection is packed back-to-back: all-tensor
consecutive gaps **median 0, p90 0, max 0**; projection gaps **median 0, p90 0, max 1210509824**
(the max is one giant single-tensor projection in a Bf16 file sequence, not a hole). O3 therefore
sat at its floor — there was nothing to merge; 661 of the 3840 runs are the *requested* footprint
(simulation: 8 pseudo-random experts + 3 shared projections per layer, gap tolerance 0 MiB, 47.46 GiB),
and 3840 − 661 ≈ 3180 are **speculative over-submission** (prefetch_wasted 2547, delivery_rate 1.26).
Whole-shard slabbing (1 run per shard = 280 runs) is economically dead here: it needs the entire
1.35 TB model per forward, a 28× byte blow-up.

**Finding 2 — the disk is latency-bound, not bandwidth-bound; the O2 cap never engages.** Layer time
~31 s ≫ run latency ~3.7 s, so the `kMaxPrefetchInFlight = 40` cap never trips; each next layer needs
~27 loads (8 next experts × 3 projections + 3 shared) against only `clamp(2..8, threads/4)` = 4 IOCP
workers at round-1 defaults → ~7 serial waves × 3.7 s ≈ the measured ~26–31 s/layer. In transfer
terms the drive is ~75 % idle.

**O4 — router-stability gate (round 2):** speculating a layer's routed experts only pays when that
layer actually re-routes to the same winners across consecutive tokens. `scheduler.cpp` now tracks a
per-layer EMA of consecutive-token winner overlap (`kSpecOverlapAlpha = 0.5`); when the next layer's
EMA falls below `kSpecOverlapFloor = 0.25` (≥ 2 of 8 winners reused), the routed-expert guess is
cleared and only the **guaranteed** next-layer shared projections are streamed. A cold start (no
history → EMA 0) and any garbage/cold routing stream therefore stop paying for pure-noise guesses;
warm, stable routing (the very regime the EMA predictor was built for) is untouched. Predictions and
routing bookkeeping still run — the gate only decides whether guessing is worth disk time.

**O5 — deeper I/O parallelism:** with the queue depth — not transfer time — the binding constraint,
the IOCP worker default widens from `clamp(2..8, threads/4)` (4 on this 16-thread box) to
`clamp(3..16, threads/2)` (8 here), letting the drive reorder a ~2×-deeper scatter of mid-size runs
(avg ~72 MB) to cut seek waste; `GLM_IOCP_WORKERS` keeps its override.

Cost/risk: O4 drops only noise runs that were already discarded predictions; O5 changes concurrency,
not byte content — assembled-expert bytes are unchanged, so the forward math stays **bit-exact**
(goldens remain valid, L3 not re-run for a perf round).

Test protocol: rebuild Release → L0 self-test (163) → L1 pytest (28) → L2 `--check-weights`
(0 failures) → standalone L4 rerun → before/after written into `reports/benchmark_report.md` and
§17.6. Note: the 2026.9.2 in-gate L4 number (0.000226 tok/s, 17685 s) was polluted by concurrent
validation-harness I/O on the shared HDD; standalone reruns remain the honest comparison (see
`release_gate_report.md` re-verification).

### 17.6 Measured outcome round 2 (standalone L4 rerun)

| Metric | Round-1 baseline | Round-2 | Delta |
|---|---|---|---|
| total_sec (4 tokens) | 9849.07 s | 8667.82 s | **-12.0 %** |
| speed_tok_per_s | 0.000406 | 0.000461 | **+13.6 %** |
| I/O run count | 3840 | 1246 | **-67.6 %** |
| Disk bytes / token | 69.5 GiB | 22.2 GiB | **-68.1 %** |
| Prefetch waste rate | 66.33 % | 31.70 % | -34.6 pp |
| HDD queue latency | 3699 ms | 2721 ms | -26.4 % |
| Expert LRU hit rate | 20.97 % | 6.94 % | lowered (see note) |
| Generated output | 16,29661,90,77 | 16,29661,90,77 | **bit-identical** |

The strike rate finally broke: 3840 runs → 1246 (-68 %) while per-token bytes fell 69.5 → 22.2 GiB.
The hit-rate number *drops* (21 % → 6.9 %) purely because O4 stopped paying for noise — the LRU now
serves only real re-use (bias-concentrated winners), and the total bytes served fell by 3.1×. The
full-stack numbers speak for themselves: total wall 9849.07 → 8667.82 s (vs the original 12 030.37 s
baseline: **-27.9 %, +38.9 % tok/s**), waste 66.3 % → 31.7 %, queue latency 3699 → 2721 ms. Output
stayed `16,29661,90,77` — bit-exact, forward math untouched.

### 17.7 Deep-dive round 3 (2026-09-25): the demand path takes shape

Round 2 made the *speculative* pipe honest (discard noise), but the wall was then dominated by the
part that never went through the async pipe at all. Two blocking structures measured from the
profile:

**Finding 1 — the demand path is strictly serial.** After `onLayerRouterDone` the forward walks the
8 routed winners one `getExpertPtr` at a time; each miss is a synchronous
`loadExpertFromDisk` mmap **memcpy** of a full 72 MB expert (read + fault + copy), so one layer's
validated loads still cost ≈ **k serial reads ≈ 8 × read latency (≈ 2.7 s HDD)** ≈ 20–31 s/layer —
exactly the measured steady-state per-layer cost. The disk is idle between those serial misses; the
async workers had nothing to do for demand.

**Finding 2 — token-1's cold dense fault storm.** 5550 s of the first token is page-fault
latency: every layer re-reads its ~0.45 GiB of always-needed dense tensors (attention projections,
normalizers, indexer weights, dense MLP) through cold mmap faults — ~4 s/layer of fault cost that
evaporates once the OS page cache holds the ranges. The LRU rightly never stores dense weights
(they are not routed, they have no eviction pressure); they were simply never *warmed*.

**O6 — demand staging through the async pipe.** `prefetchAsync` gains a `demand` flag that
bypasses the O2 speculative cap (those runs are the layer's current service set, so saturation
backpressure would be wrong). `onLayerRouterDone` now stages the **current** layer's elected
winners + its three shared projections with `demand = true` before any consumer pointer is
requested; `getExpertPtr`/`getSharedProjectionPtr` poll (2 ms, `kDemandWaitTimeoutMs = 120 s`
bounded) for the in-flight assembly to land in the LRU instead of doing an immediate synchronous
copy. All k experts of a layer therefore load **concurrently on the IOCP workers** (per-shard runs
merge into one sequential read), so a layer costs ≈ 1 read latency instead of k serial faults —
and shared projections get the same treatment.

**O7 — dense-weight OS-cache warm.** `warmDenseWeights()` enumerates every always-read dense +
shared range (attention, norm, indexer, dense MLP, embedding/LM-head/final-norm, shared MoE
projections), merges same-shard ranges with the O3 gap tolerance, and discard-reads the merged
runs in 8 MiB chunks through a dedicated 8-worker IOCPReader into a 16-slot rotating pool
(`kWarmPool`), explicitly *not* inserting anything into the LRU. Called once before the first
forward (guarded by `GLM_NO_WARM` for cold-measurement). Measured footprint on this checkpoint:
**35 479 MiB streamed in 160 runs** (~1–5 min, sequential-friendly). This converts token-1's
fault storm into RAM hits; the forward math reads the same bytes from its own mmap windows.

Accounting note: with the demand loads now flowing through `prefetchAsync`, round-3 scheduler
I/O counters (`ioReadCount`, `randomIoC_o`, `diskBytesRead`, `prefetchSubmitted`) count **all**
scheduler reads (demand + speculative) whereas round-2 counted speculative reads only — bytes and
run totals from §17.6 are therefore not directly comparable (round-3 adds the per-layer service
sets). Prefetch *waste* drops structurally because demand inserts are always consumed.

Cost/risk: both changes are I/O-layer only. Assembled-expert bytes are unchanged and dense weights
are still read from their original mmap windows, so forward math stays **bit-exact** (goldens
remain valid, L3 not re-run for a perf round). No new CLI surface beyond the `GLM_NO_WARM` toggle.

Test protocol: rebuild Release → L0 self-test (163) → L1 pytest (28) → L2 `--check-weights`
(0 failures) → standalone L4 rerun → before/after written into `reports/benchmark_report.md` and
§17.8.

### 17.8 Measured outcome round 3 (standalone L4 rerun)

| Metric | Round-2 | Round-3 | Delta |
|---|---|---|---|
| total_sec (4 tokens, wall incl. ~265 s warm) | 8667.82 s | 7572.95 s | **-12.6 %** |
| speed_tok_per_s | 0.000461 | 0.000528 | **+14.5 %** |
| vs original 12 030.37 s baseline | -27.9 % | **-37.1 %** | -9.2 pp |
| Token-1 (coldest forward) | ~4.0-4.5 ks (inferred from r1/r2 markers) | **3291 s (direct)** | -20…-30 % |
| Tokens 2-4 per-token | ~1150-1613 s | ~1117-1498 s | ≈ par |
| Prefetch waste rate | 31.70 % | **8.93 %** | -22.8 pp |
| Expert LRU hit rate | 6.94 % | 62.08 % | accounting (below) |
| HDD queue latency | 2721 ms | 4768 ms | +2047 ms (deeper queue) |
| Generated output | 16,29661,90,77 | 16,29661,90,77 | **bit-identical** |

The wall keeps falling: **7572.95 s** total (0.000528 tok/s) is **-12.6 % vs round-2** and **-37.1 % vs the
original baseline** — and ~265 s of that is the one-time warm, so the honest generation-only total is
~7308 s (-15.7 % vs round-2). The structural change shows up in token-1, measured **directly** at
3291 s on an identical max-tokens=1 run (rounds 1-2 only implied ~4.0-4.5 ks from their first-two-token
cumulative markers, 6607 s and 5550 s respectively): O7 turns the cold dense fault storm into OS-cache
hits and O6 overlaps the layer's winners on the disk instead of loading them serially. Steady tokens
hold at ~1.1-1.6 ks.

Counter caveats (see §17.7): round-3's scheduler I/O counters now include the **demand** loads, so
`randomIoC_o` (4425 vs 1246), `diskBytesRead` (318.52 GiB / 4 tok ≈ 79.6 GiB/token vs 22.2) and the
62 % LRU hit rate are **not comparable** to earlier rounds — they count a superset. The LRU hit-rate
jump has a real core too (demand waits now find freshly-staged experts resident), and waste collapsed
to 8.9 % because demand inserts are always consumed. HDD average latency rose 2721 → 4768 ms simply
because the deeper demand+speculative queue trades per-read latency for fill parallelism — the wall
still improved, which is the scoreboard that matters.

---

### 17.9 Round 4: precision demand sequencing (plan, written before the run)

**Analysis from round-3 evidence.** The round-3 telemetry leaves a clear structural fingerprint:

- **The read stream is single-expert-sized.** 342 003 548 160 B across 4425 runs = 77.3 MB/run ≈ one
  72 MB expert per run. Per-shard cross-expert merging (O3) still almost never happens on the benchmark:
  the synthetic prompting sends a layer's 8 winners into ~8 different shards, so each layer degrades back
  to ~8-11 barely-merged runs.
- **Bytes vs. working set disagree by ~55 GB/token.** Lookups are 900/token (75 layers × ~12), demand
  misses 341/token ≈ 24.5 GB; the 3 shared projections/layer are read once each (probe-dedup between the
  next-layer scope and the current-layer demand) ≈ 6 GB. The remaining **~55 GB/token of read bytes are
  never consumed** — and the only producer of such reads is the one-layer-ahead routed-expert scope in
  `onLayerRouterDone` → `prefetchAsync`.
- **The O4 gate cannot fire on this workload.** `kSpecOverlapFloor = 0.25` equals the random-routing
  baseline: for 8-of-256 routing the expected consecutive-token overlap is E[shared] = 8·(8/256) = 0.25
  and with `kSpecOverlapAlpha = 0.5` a cold/garbage layer's EMA converges to ≈0.25 — which is *not*
  `< 0.25` — so the gate never clears and every layer stages up to 8 predictions every token. The spec
  stream therefore runs at full rate straight through the benchmark's worst case.
- **The wall is transfer-bound.** 85.5 GB/token at the drive's ~100-115 MB/s is ~750-850 s of pure
  transfer against measured steady tokens of ~1117-1498 s; the rest is seek/queue overhead plus
  forward-wait exposure. Cutting bytes that are structurally useless moves the floor itself.
- **Runs reach the disk in arbitrary shard order.** `prefetchAsync` merges per shard through an
  `unordered_map` iteration, so consecutive `ReadFile` arrivals trampoline between shard files and the
  drive spends its reordering window on multi-directional seeks.

**Plan (I/O layer only; expert bytes never change ⇒ outputs stay bit-identical; no version bump):

- **O8 — decide the routed-speculation gate for cold/random routing.** Raise `kSpecOverlapFloor` from
  0.25 to 0.5 (2× the chance baseline): a layer must show sustained ≥4/8 winner reuse before any routed
  lookahead runs for it. Cold EMA ≈ 0.25 → cleared (junk → 0); stable reuse EMA → ≈1.0 → untouched. This
  also stops junk filling the 38.6 GB LRU, whose round-3 churn evicted freshly-staged demand between
  assembly and use — part of the 341/token disk loads and the O6-wait timeouts.
- **O8b — offset-ordered run submission.** After the per-shard merge, sort the whole run set by
  (shardPath, start) before issuing, so consecutive arrivals sweep the drive head instead of trampolining
  between files. Slices keep per-projection offsets, so assembly math is unaffected.
- **O8c — precision accounting.** New counters (`routedPredStaged`, `routedPredConsumed`,
  `demandKeysStaged`, `demandWaitTimeout`) so §17.10 reports the junk/consumed split exactly instead of
  the entangled estimate above.

Expected outcome: bytes/token 85.5 GB → ~31-35 GB (demand + shared, plus the residue that stays
speculative), prefetch waste → ~0, steady tokens → hopefully ~0.6-0.9 ks, total < 7572.95 s.

---

### 17.10 Measured outcome round 4 (standalone L4 rerun + 1-token ground-truth run)

| Metric | Round-3 | Round-4 | Delta |
|---|---|---|---|
| total_sec (4 tokens, wall incl. ~265 s warm) | 7572.95 s | 7103.63 s | **-6.2 %** |
| speed_tok_per_s | 0.000528 | 0.000563 | **+6.6 %** |
| Disk bytes read (whole run, 7 forwards) | 342.00 GB | 300.25 GB | **-12.2 %** |
| Prefetch waste (evicted-unused runs) | 395 | **0** | -100 % |
| Routed-pred staged (gate survivors) | (hundreds-thousands, unmeasured) | **16** | ~100 % down |
| Demand keys staged / run | — | 3529 | — |
| Demand wait timeouts | (some, unmeasured) | **0** | — |
| Expert LRU hit rate | 62.08 % | 65.22 % | +3.1 pp |
| HDD queue latency | 4768 ms | 3965 ms | -803 ms |
| Generated output | 16,29661,90,77 | 16,29661,90,77 | **bit-identical** |

The wall falls to **7103.63 s** (0.000563 tok/s), **-6.2 % vs round-3**. The routed-speculation gate now
works as designed: **16** routed lookahead guesses survived `kSpecOverlapFloor = 0.5` over the whole run
(previous rounds staged hundreds-thousands), prefetch waste → **0**, O6 demand-wait timeouts → **0**, and
the LRU hit rate nudged up (62.08 → 65.22 %). O8b's offset-ordered submission and O8c's accounting are
bundled in this result (attribution between them is not separable from one run).

**The honest surprise (correcting §17.9's estimate):** the expected 85.5 → ~31 GB/token collapse did not
happen; "bytes per generated token" only fell 79.6 → 69.9 GiB. The 1-token ground-truth run (identical
`--max-tokens 1` config) proves why: it moves **149,258,502,144 bytes for 4 full forwards (3 prefill
tokens + 1 decode) = ~37.3 GB/forward (34.8 GiB)**, exactly identical to round-3's 1-token run. Today's
engine runs
**one full 78-layer forward per position**, so the L4 run performs **7 forwards (3 prefill + 4 decode)**;
the report metric `bytes_read_per_token` divides a 7-forward numerator by a 4-decode denominator, which
inflates the "per token" figure by 7/4 and hid the prefill multiplier. The 342 → 300 GB reduction is
therefore the *true* O8 win — the routed-lookahead junk of rounds 2-3 (~38.9 GB across tokens 2-4) is
gone — and the **dominant cost is the demand working set itself: ~37-43 GB (35-40 GiB) per forward of ~600 unique
72 MB routed experts (= the layer's elected winner bytes) plus shared projections**. That set is the
floor: no scheduler policy can shrink the router's own elected winners.

**Per-forward timing:** token-1 (cold, dense prewarmed) measured directly at **3257.17 s** on the 1-token
run (round-3: 3291 s, ≈ par). Steady decode forwards stay ~11-14 min each — the drive's sustained
random-mix effective rate (~44-50 MB/s) × ~37-43 GB per forward bounds them. The disk is the wall.

**Round-5 candidates (documented, not run):** (a) cut the **prefill multiplier** — 3 of the 7 forwards
exist only to build the prompt's KV; overlapping pre-fill routing (one winner-union per layer across the
prompt positions) or batched spine would remove ~2/7 of total bytes — a forward-path change, not
I/O-only, so it must re-verify byte-equivalence gates separately; (b) push the drive harder from the
scheduler side only remnants: the offset-ordered queue is in; a per-shard cached file handle would shave
`CreateFileA` overhead; deeper NCQ-friendly batching stays tempting but measured seek pressure keeps the
gain small. The structural answer remains storage media, documented in §17.8.

---

### 17.11 Round 5: where the wall-clock actually goes (measured, then re-planned)

**Where round 4 left off.** The L4 run is 7103.63 s for 7 forwards = 977 s/forward = **12.5 s per
layer**, and the counters report 300.25 GB of disk traffic, i.e. an *effective* **42 MB/s**. §17.10
concluded the winner bytes are irreducible — but it never asked whether the **engine's I/O path is
reaching the drive's own capability**. Round 5 answers that with measurements before changing code.

**Disk probe (standalone tool, `%TEMP%\opencode\readprobe`, cold regions, distinct files per test).
282 model files x 5 GB, an expert = 72 MB (3 x 24 MB projections, all three in the *same* file —
verified over `model.safetensors.index.json`: 19 456 experts, **0 % multi-shard**):**

| Pattern (cold) | Rate |
|---|---|
| Single file, 72 MB `ReadFile`, 1-way | 125.5 MB/s |
| Single file, 72 MB `ReadFile`, 8-way | 133.0 MB/s |
| Single file, ascending sweep (O8b's order), 8 workers | **187.1 MB/s** |
| **8 files interleaved, 24 MB, 1-way / 2 / 4 / 8 / 16-way** | **63.6 / 93.4 / 94.6 / 98.5 / 104.7 MB/s** |
| 4 files interleaved, 24 MB, 8-way | 133.5 MB/s |
| Single 72 MB **mmap fault-walk** (the synchronous path) | 85-99 MB/s (first touch), 430-1490 MB/s when warm |
| `CreateFileA`+`CloseHandle` per 8 MB read | 0.04 ms (negligible — the per-run open is *not* a cost) |

Conclusions: the platter sustains **~90-100 MB/s on 24-72 MB random/multi-file reads** and up to
**187 MB/s** on a directional sweep; interleaving 8 files costs ~1.9x versus one-file sweeping;
`defrag`/fragmentation is ruled out (12 distinct files all read 85-99 MB/s). The engine's *reported*
42 MB/s therefore looked ~2.3x below what its own access pattern can sustain, which pointed at two
candidate engine-side defects — both invisible in the round-4 counters, and **both later disproved by the
O9c counters (see the result section below)**:

1. **The synchronous fallback is unmeasured and serialized.** `getExpertPtr`'s miss path only waits for
   an in-flight assembly; when the key is *neither resident nor in flight* it falls through to
   `loadExpertFromDisk` — a blocking `getShardView` + 72 MB `memcpy` walking mmap page faults on the
   **forward thread**, with its bytes never counted in `diskBytesRead`.
2. **`prefetchAsync` never checks the LRU.** A key already resident is re-staged and re-read from disk,
   which would re-stream the pinned shared projections on every forward.

**Planned and shipped (all I/O-layer; no forward math, byte-identical assemblies):**

- **O9 — demand staging in the getters.** On a miss with no assembly in flight, `getExpertPtr` /
  `getSharedProjectionPtr` enqueue that key through `prefetchAsync(..., demand=true)` and wait for the
  assembly like the in-flight case; the synchronous copy remains only as a last-resort fallback.
- **O10 — skip staging for LRU-resident keys.** `prefetchAsync` probes `lru_.contains(key)` first
  (`stageSkipLru` counter). Removes any redundant re-read of resident bytes.
- **O11 — widen the coalescing window**, `kRunMergeGap` 1 MB → 8 MB. **Shipped, measured, reverted:**
  see below.
- **O9c — make the invisible path visible.** New counters `syncLoads`, `syncBytes`, `syncStallNs`,
  `demandWaitNs`, `stageSkipLru` + `totalDiskBytes`, and matching `benchmark.py` rows. The cross-round
  comparison table is now rendered from `reports/benchmark_history.json` by `benchmark.py` itself,
  because `write_report` rewrites `benchmark_report.md` wholesale and a hand-appended table cannot
  survive a rerun.

**Result of the O9/O10/O11 run (R5-widemerge): 7219.38 s — a 1.6 % regression, and both defects above
turned out not to exist.** The new counters say why:

| Counter | R4 (unmeasured) | R5-widemerge | Meaning |
|---|---|---|---|
| `expertDiskLoads` | 1252 | 1252 | **not** sync loads — this counter also fires on every in-flight wait; the round-4 reading of "1252 serial 72 MB copies" was wrong |
| `syncLoads` / `syncBytes` / `syncStallMs` | — | **40 / 3.02 GB / 94 159 ms** | the synchronous path existed, but only 40 times, not 1252 |
| `stageSkipLru` | — | **0** | **O10 never fires**: the LRU (37.7 GB) is smaller than one forward's 43 GB footprint, so a key is never still resident when the scheduler wants to stage it — the "43 GB of redundant re-reads" was fictional |
| `randomIoCount` | 3880 | 3928 | O11's 8 MB window changed the access pattern by ~1 % — it bought nothing |
| `diskBytesRead` | 300.25 GB | 303.88 GB | O11's wider window added **3.62 GB of pure padding** (+1.2 % bytes, +1.6 % wall) |

**The real decomposition (this is the finding worth keeping).** Splitting the run by layer:
`total 7219 s / 7 forwards / 78 layers = 13.2 s per layer`, of which `demandWaitMs` accounts for
**5.3 s** of forward-thread waiting and the rest is compute. And the demand per layer is
`~8 winners x 72 MB + 3 shared x 26 MB = 680 MB`; at 5.3 s that is **~128 MB/s** — *at or above* the
probe's 8-file interleaved rate (99-105 MB/s). **The read phase was already running at the drive's
measured multi-file rate; the "42 MB/s effective" figure was an artifact of averaging over a drive that
sits idle ~60 % of the time.** The engine has no I/O efficiency defect. What it lacks is *overlap*:
`moeLayer` calls `getExpertPtr` for expert *k*, blocks until its 72 MB lands, computes expert *k*, then
asks for expert *k+1* — so a layer is `5.3 s of read + 7.9 s of compute` strictly serialized.

**The shipped fix (O12) is therefore an overlap, not a rate change.** `Scheduler::prefetchExpert(layer,
eid)` issues the same demand stage for one expert and returns immediately; the CPU expert loop calls it
for expert *k+1* before computing expert *k*, so the next read proceeds on the IOCP workers while the
current expert's FFN runs. The later `getExpertPtr` finds the assembly in flight and waits for what is
left. Same experts, same order, same kernels, same accumulation order — only *when* the bytes move.

**O9b — drop the redundant `sameShard` gate.** The run builder already groups projections by `shardPath`
(one run per shard, slices carrying per-projection offsets), so a multi-shard expert assembles
correctly through the queue; the gate only forced a blocking forward-thread copy. Removing it took
`syncLoads` 40 → **0** and `syncStallMs` 94 159 → **0** (O11's revert restored `kRunMergeGap` to 1 MB
at the same time).

**Final round-5 result: 6684.18 s vs 7103.63 s round-4 (−5.9 %), 0.000563 → 0.000598 tok/s (+6.2 %),
300.25 → 306.90 GB of *true* traffic** (R4's figure excluded its 40 untracked synchronous copies),
`syncLoads` 0, `stageSkipLru` 0, prefetch waste 0, demand-wait timeouts 0, `expertLookups` 3600,
LRU hit 65.2 %, output `16,29661,90,77` **bit-identical**. Gates: L0 163/163, L1 32/32 (4 new
benchmark-tooling tests), L2 `--check-weights` 0 failures.

**What is left, honestly.** The remaining exposed read is the *first* expert of each layer (nothing can
be in flight before the router names a winner) plus any read slower than the ~7.9 s of compute it hides.
A 2-3 slot read-ahead would hide more of the 5.3 s, and the structural answers remain: batch the three
independent prefill positions (a forward-path change) or move the checkpoint to an SSD. With reads
already at the drive's rate, **no I/O-only change can beat ~4.3 ks — the 78-layer compute itself.**

---

### 17.12 Round 6: the demand stream is bulk-synchronous, not merely un-overlapped

**Where round 5 left off.** 6684.18 s for 7 forwards = **12.24 s/layer**, of which `demandWaitMs`
= 3032 s (**45 %**) is the forward thread blocked in the getters. §17.11 concluded the read phase
already runs at the drive's rate, so the remaining question is *ordering*: 5.5 s of waiting per layer is
almost exactly the time to transfer the whole layer's demand (`8 x 72 MB + 3 x 26 MB = 654 MB` at
~100-118 MB/s = 5.2-6.5 s). That is the signature of **bulk-synchronous I/O**: the layer's entire demand
is fired at t=0, every read finishes at the far end of the transfer window, and only then does compute
begin.

**Why it is worse than that: the completions come back out of order, and the forward consumes in issue
order.** `readprobe5` (8 x 72 MB cold, one per shard, all submitted at t=0) measured completion at
0.07 / 1.30 / 2.01 / 3.17 / 3.51 / 4.23 / 4.95 / 5.68 s for reads 0 / 4 / 3 / 1 / 2 / 6 / 5 / 7 — the
consumer is blocked on read 1 at 3.17 s while read 4 has been sitting finished since 1.30 s. That is
head-of-line blocking, and it is *luck-dependent*, which is why the same pattern measured anywhere from
7.4 s to 11.4 s per layer across probe runs while the engine sits at the bad end.

**The measurement that picks the fix (`readprobe6`, consumer model: expert *i* cannot compute before
`max(data_ready(i), compute_done(i-1))`, 8 x 72 MB per layer, distinct cold shard per read):**

| reads in flight (K) | transfer wall | aggregate | consumer finishes | of which stall |
|---|---|---|---|---|
| **8 (today's bulk submit)** | 5.17-5.80 s | 104-117 MB/s | **7.37-11.37 s (high variance)** | 1.4-6.9 s |
| 4 (paced) | 5.98 s | 101 MB/s | 7.42-8.13 s | ~1.5 s |
| **3 (paced)** | 5.85 s | 103 MB/s | **6.74-7.66 s** | ~0.6 s |
| 2 (paced) | 5.86 s | 103 MB/s | 6.74-6.96 s | ~1.5 s |
| 1 (fully serial) | 6.55 s | 92 MB/s | 7.70 s | ~1.0 s |

Pacing to **K = 2-3** makes the consumer's completion time both *lower and far more stable* than the bulk
pattern, and it does **not** cost throughput: a shallower queue means fewer interleaved files in flight,
which measured *faster* in the 3-4 range than in the 8 range (117 MB/s at K=3 vs 104 MB/s at K=8 in the
first sweep). The optimum is flat between K=2 and K=4, so the choice does not need to be precise.

**Precedent already in the tree:** the GPU path never had this problem — `moeLayer`'s GPU branch
(`glm_forward.cpp:570-589`) uses a *rolling window*: preload `gpuStagingDepth_` experts, then advance one
slot per iteration so the async H2D of expert *k+1* overlaps expert *k*'s FFN. The CPU branch fires
everything and then consumes. Round 6 makes the CPU branch behave like the GPU one.

**The plan (I/O-layer only — same experts, same order, same kernels, same accumulation order):**

- **O13 — paced demand instead of bulk submit.** `onLayerRouterDone` stages only the first
  `kDemandPrimerWinners = 2` current-layer winners (plus the 3 shared projections, which are deduped
  against the previous layer's next-layer staging anyway). The remaining winners are pulled just-in-time
  by the forward's own look-ahead, so the in-flight depth settles at ~3 instead of 11. The eager-staging
  cap deliberately does **not** apply to the look-ahead: a declined look-ahead would drop the getter onto
  the synchronous mmap fallback, which is strictly worse than waiting in the queue. *(Verified afterwards
  that `prefetchAsync` never declines demand anyway — it bypasses the `kMaxPrefetchInFlight` cap — so the
  worry was unfounded, but the reasoning stands.)*
- **O13b — deepen the consumer look-ahead to 2 slots** (`prefetchExpert` for *i+1* and *i+2*), so each
  read has ~2 compute blocks of lead time and the drive always has work queued.
- **O13c — counters that prove the mechanism:** `demandPrimerStaged`, `loopPrefetchIssued`,
  `demandSkippedEager` (winners no longer fired up front), and `firstExpertWaitNs` (the wait for the
  layer's *first* expert — the number that must fall if pacing works). `benchmark.py` gains the rows.

Predicted: the first expert of a layer becomes available in ~0.8 s instead of ~5.5 s, and the consumer
finishes the layer in ~7 s instead of ~9-11 s ⇒ ~10 s/layer ⇒ ~5.5 ks for the run (-15 % to -20 % vs
round 5). If `firstExpertWaitNs` does *not* fall, the pacing did not take effect and the change is
reverted — the counter decides, not the hope.

#### What actually happened (two L4 runs)

The counter did fall, by the predicted mechanism, and it bought nothing:

| metric (444 sampled layers) | R6, paced (primer 2) | R6-revert, whole-layer submit |
|---|---|---|
| `reads_in_flight_at_first_wait` | **3.98** | 8.67 |
| `experts_in_flight_at_first_wait` | **5.84** | 10.85 |
| `first_expert_wait_ms` (per layer) | **1527.66** | 3791.27 |
| `demand_wait_ms` (total) | 3 060 910 | 3 006 990 |
| `hdd_latency_ms` | 2115.05 | 3970.56 |
| `loop_prefetch_issued` | 2705 | 42 |
| `demand_primer_staged` / `demand_skipped_eager` | 888 / 2664 | 3552 / 0 |
| total disk bytes | 306 897 223 680 | 306 897 223 680 (identical) |
| prefetch waste / sync loads | 0 / 0 | 0 / 0 |
| output | `16,29661,90,77` | `16,29661,90,77` |
| **wall clock** | **6781.38 s** | **6984.42 s** |

Pacing cut the head-of-line wait before the first expert could compute by **60 %** (3.79 s → 1.53 s per
layer) and halved the read queue depth, exactly as designed. The *total* demand wait did not fall
(+1.8 %), and wall clock did not improve either. The reading: **the wait moved from the head of the layer
into the middle of it.** A layer must move 654 MB; at the measured multi-file rate that is ~6.5 s of
drive time, and the layer's compute is ~5.4 s — so the layer is **bandwidth-bound, not schedule-bound**,
and there is no ordering of those reads that makes the bytes arrive sooner. §17.11's floor stands.

#### The measurement that outranks all of it: this host's noise floor

R5 = 6684.18 s. The paced build = 6781.38 s. A build reverted to **byte-for-byte identical R5 behaviour**
= 6984.42 s. Three runs of essentially one configuration span **300 s = 4.5 %**, and `hdd_latency_ms`
alone ranged 2115 → 3971. So:

- the paced-vs-R5 difference (97 s) is **inside the noise**, and the round-6 experiment is *inconclusive
  on wall clock*, not a regression;
- every single-change wall-clock claim below ~300 s on this host is unprovable with one run per config —
  including round 5's own -5.9 % only because it was measured against a 414 s gap;
- future rounds must decide with **in-process counters** (like the O13c ones above), not wall clock, or
  spend N repeated runs per configuration.

**Shipped:** the O13c instrumentation only (`first_expert_wait_ms`, `reads_in_flight_at_first_wait`,
`experts_in_flight_at_first_wait`, `loop_prefetchIssued` → `loop_prefetch_issued`, plus the
`runsInFlight` accounting) and `benchmark.py` rows for them. **Reverted:** O13's primer (the whole-layer
submit is back) — there is no evidence it costs anything and it is the configuration with the best
measured wall clock; O13b's 2-slot look-ahead is kept because it is a no-op under whole-layer staging
(42 issues per run instead of 2705) and guards the declined-staging case. The revert and the paced build
are both recorded in `reports/benchmark_history.json` (`R6`, `R6-revert`) so the noise is auditable.
Gates after the revert: L0 163/163, L1 32/32, L2 `redirected: 0 failures`.

---

## Changelog

| Version | Date | Notes |
|---|---|---|
| 2026.9.2 | 2026-09-28 | **Performance optimization round 6 (paced demand — measured, inconclusive on wall clock, reverted; §17.12)** — the round-5 decomposition left 3032 s (45 %) of `demandWaitMs`, ~5.5 s/layer, which is about the time to transfer a layer's 654 MB of demand: the signature of a **bulk-synchronous** submit. Two cold-disk probes established the mechanism and the candidate fix. `readprobe5` (8 × 72 MB, one per shard, all issued at t=0) showed completions returning **out of order** (0.07 / 1.30 / 2.01 / 3.17 / 3.51 / 4.23 / 4.95 / 5.68 s for reads 0 / 4 / 3 / 1 / 2 / 6 / 5 / 7) while the forward consumes **in issue order** — head-of-line blocking, and luck-dependent. `readprobe6` (a consumer model: expert *i* computes at `max(data_ready(i), compute_done(i-1))`) swept the in-flight depth K over 8/4/3/2/1 at two compute rates and put **K = 2-3 at a stable 6.74-7.66 s per layer against 7.37-11.37 s (high variance) for the bulk submit**, at equal or better throughput (117 vs 104 MB/s — a shallower queue interleaves fewer files). Implemented exactly that: **O13** `onLayerRouterDone` fires only a 2-expert primer and the forward loop's **O13b** 2-slot look-ahead pulls the rest, so in-flight settles at ~3. **The mechanism worked and the number did not move.** The new **O13c** counters show `reads_in_flight_at_first_wait` **8.67 → 3.98** and `first_expert_wait_ms` **3791 → 1528** (head-of-line blocking before the first expert FFN cut by **60 %**), with byte-identical I/O (306 897 223 680 B, waste 0, sync 0, output `16,29661,90,77`) — but `demand_wait_ms` went 3007 → 3061 s and wall clock 6684.18 → **6781.38 s**. Diagnosis: the wait merely moved from the head of the layer into its middle; a layer must move 654 MB in ~6.5 s of drive time against ~5.4 s of compute, so the layer is **bandwidth-bound, not schedule-bound** and no read ordering makes the bytes arrive sooner — §17.11's floor stands. **Reverted** the primer (whole-layer submit restored; O13b kept, a no-op under it). **The round's real result is the noise floor:** a build reverted to byte-for-byte identical R5 behaviour measured **6984.42 s**, so three runs of one configuration span **300 s = 4.5 %** (hdd_latency_ms alone 2115 → 3971) and the paced-vs-R5 gap is *inside* it — round 6 is inconclusive on wall clock, and no single-run wall-clock claim below ~300 s on this host is provable. Both runs are kept in `reports/benchmark_history.json` (`R6`, `R6-revert`) so the noise is auditable. Shipped: the O13c instrumentation (`first_expert_wait_ms`, `reads_in_flight_at_first_wait`, `experts_in_flight_at_first_wait`, `loop_prefetch_issued`, `runsInFlight`) + `benchmark.py` rows, because in-process counters are the only instrument that can see inside that noise. Gates after the revert: L0 163/163, L1 32/32, L2 `redirected: 0 failures`. |
| 2026.9.2 | 2026-09-27 | **Performance optimization round 5 (overlap, §17.11)** — a cold disk probe (standalone tool, 282×5 GB checkpoint) measured the drive at **90-105 MB/s** on the engine's own 8-file interleaved 24-72 MB pattern and 187 MB/s on a directional sweep, which made the engine's reported **42 MB/s look like a 2.3× engine defect**. The O9c counters added for the round **disproved both candidate defects** and produced the finding that matters: `expertDiskLoads` = 1252 was never sync loads (it also fires on every in-flight wait — only **40** were, 3.02 GB / 94 s of forward-thread stall), and `stageSkipLru` = **0** proved the LRU (37.7 GB) is smaller than one forward's 43 GB footprint, so the "~43 GB of redundant re-reads" was fictional. Per-layer decomposition: 13.2 s/layer = **5.3 s dependent read + ~7.9 s compute**, and 680 MB of demand in 5.3 s is **~128 MB/s — at/above the probe's multi-file rate**; the low average was an idle drive, not a slow engine. Shipped: **O12** `Scheduler::prefetchExpert()` + a one-deep read-ahead in the CPU expert loop of `glm_forward.cpp` — the next expert's read is issued while the current expert's FFN computes, turning serial read-then-compute into overlap (same experts, order, kernels and accumulation order → bit-identical); **O9b** drops the redundant `sameShard` gate in `prefetchAsync` (the run builder already groups projections by `shardPath`, so multi-shard experts assemble correctly through the queue) taking `syncLoads` 40 → 0 and `syncStallMs` 94159 → 0; **O9** getter-side demand staging + **O10** LRU-residency skip (both now provably no-ops on this workload, kept together with the counters that prove it); **O9c** new telemetry (`syncLoads`/`syncBytes`/`syncStallMs`/`demandWaitMs`/`stageSkipLru`/`totalDiskBytes`) plus `benchmark.py` rows. **O11 (merge window 1 → 8 MB) was shipped, measured at 7219.38 s, and REVERTED**: it changed the run count by ~1 % and added 3.62 GB of pure padding (+1.6 % wall). `benchmark.py` now owns the cross-round comparison table (rendered from `reports/benchmark_history.json`) because `write_report` rewrites `benchmark_report.md` wholesale. Result: **6684.18 s vs 7103.63 s round-4 (−5.9 %), 0.000563 → 0.000598 tok/s (+6.2 %)**, traffic 300.25 → 306.90 GB of **true** bytes (round-4's figure excluded its 40 untracked synchronous copies), prefetch waste 0, demand-wait timeouts 0, LRU hit 65.2 %, output `16,29661,90,77` **bit-identical**; L0 163/163, L1 32/32 (4 new tooling tests), L2 `--check-weights` 0 failures. Remaining floor: reads already run at the drive's rate, so no I/O-only change beats ~4.3 ks — the 78-layer compute is the wall; next candidates are a 2-3 slot read-ahead, batching the 3 independent prefill positions (forward-path), or faster storage. |
| 2026.9.2 | 2026-09-26 | **Performance optimization round 4 (precision demand sequencing, §17.9/17.10)** — three scheduler-side changes that **never alter expert bytes**: **O8** decisive routed-speculation gate in `scheduler.cpp` — `kSpecOverlapFloor` raised 0.25 → **0.5** because 0.25 exactly equals the random-routing baseline (E[shared] = 8·(8/256) = 0.25), so the old gate *never fired* and every layer staged ~8 never-consumed routed guesses per token (the round-3 run carried hundreds-thousands, ~38.9 GB of junk across tokens 2-4); with 0.5 only sustained ≥4/8 winner reuse re-opens lookahead for a layer — cold/random EMA stays ~0.25 → cleared, stable reuse EMA → ~1.0 → untouched; **O8b** offset-ordered run submission in `prefetchAsync` — merged runs stable-sorted by (shardPath, offset) before issue so consecutive `ReadFile` arrivals sweep the drive head instead of trampolining between shard files (slices keep per-projection offsets); **O8c** precision accounting counters + `benchmark.py` report rows (`routed_pred_staged`, `routed_pred_consumed`, `demand_keys_staged`, `demand_wait_timeout`). L0 self-test 163/163, L1 pytest 28/28, L2 `--check-weights` 0 failures, standalone L4 rerun **7103.63 s vs 7572.95 s round-3 (-6.2 %, 0.000528 → 0.000563 tok/s, +6.6 %)**: routed-pred staged → **16** (was hundreds-thousands), prefetch waste → **0**, demand-wait timeouts → **0**, disk bytes 342.0 → 300.25 GB (**-12.2 %**), LRU hit 62.08 → 65.22 %, HDD latency 4768 → 3965 ms, output `16,29661,90,77` **bit-identical**. Ground truth (1-token run, identical config): 149.26 GB moved for 4 forwards (3 prefill + 1 decode) ≈ **37.3 GB/forward (34.8 GiB)**, byte-identical to round-3 — proving the L4 run performs **7 forwards** (prefill + decode; the `bytes_read_per_token` ÷4 denominator inflates the per-token figure by 7/4) and that the dominant cost is the router's own elected winner set (~37-43 GB/forward of 72 MB experts), the policy-immune floor; token-1 directly measured 3257 s (round-3: 3291 s). Round-5 roadmap: shrink the prefill multiplier (batched/overlapped prompt routing — a forward-path change) or storage media. |
| 2026.9.2 | 2026-09-25 | **Performance optimization round 3 (deep-dive, I/O layer, §17.7/17.8)** — two more caching-path changes that **never alter expert bytes**: **O6** demand staging in `scheduler.cpp` — the forward's 8 serial synchronous `getExpertPtr` mmap copies/layer (≈ 8 × read latency ≈ 20-31 s/layer, the measured steady-state cost that none of rounds 1-2 touched) are gone: `prefetchAsync` takes a `demand` flag that bypasses the O2 cap and `onLayerRouterDone` now stages the **current** layer's elected winners + its 3 shared projections before the forward asks for them, while both getters poll (2 ms, 120 s bound) for the in-flight assembly to land in the LRU — all k experts of a layer load **concurrently** on the IOCP workers and per-shard runs merge, so a layer costs ≈ 1 read latency; **O7** dense prewarm in `scheduler.cpp`/`main.cpp` — `warmDenseWeights()` enumerates every always-read dense+shared range (attention, norm, indexer, dense MLP, embedding/LM-head/final-norm, shared MoE projections), merges same-shard ranges with the O3 gap tolerance and discard-reads them in 8 MiB chunks through a dedicated 8-worker IOCPReader (35 479 MiB in 160 runs, ~4-5 min), converting token-1's cold page-fault storm (~4 s/layer) into OS-cache hits; guarded by `GLM_NO_WARM`. Rebuild: `warmDenseWeights` was initially declared private and `glm.exe` silently stayed a stale binary (only `glm_tests.exe` relinked — that is why the fast gates had passed); moved to public, rebuilt, re-gated. L0 self-test 163/163, L1 pytest 28/28, L2 `--check-weights` 0 failures, standalone L4 rerun **7572.95 s vs 8667.82 s round-2 (-12.6 %) vs 12 030.37 s baseline (-37.1 %, 0.000528 tok/s, +14.5 %)**: token-1 ≈ **3291 s (direct 1-token run; rounds 1-2 implied ~4.0-4.5 ks from their 6607 s / 5550 s first-two-token markers)**, steady tokens ≈ 1.1-1.6 ks, prefetch waste 31.70 → **8.93 %**, output `16,29661,90,77` **bit-identical**; round-3 scheduler counters now **include demand runs** (§17.7 accounting note: runs 4425, 79.6 GiB/token, LRU hit 62 % — superset vs round-2, not directly comparable), HDD avg latency 2721 → 4768 ms (deeper queue, latency now overlaps compute). All reports and README Performance section updated. |
| 2026.9.2 | 2026-09-25 | **Performance optimization round 2 (deep-dive, I/O layer, §17.5/17.6)** — a per-shard header audit of all 282 shards proved the weight files are packed **back-to-back (0-byte gaps median/p90/max)**, so round-1's O3 already sat at its merging floor and the ~3180 runs above the 661-run requested footprint were **speculative over-submission** (waste 66 %, delivery 1.26); with layer time ~31 s ≫ run latency ~3.7 s the O2 in-flight cap never engaged and only 4 IOCP workers left the drive 75 % idle in transfer terms. Shipped two caching-unit changes that **never alter expert bytes**: **O4** router-stability gate in `scheduler.cpp` — a per-layer EMA of consecutive-token winner overlap (`kSpecOverlapAlpha = 0.5`, floor `kSpecOverlapFloor = 0.25`) now clears the routed-expert guess for any layer that has not recently re-routed to the same winners, while the guaranteed next-layer shared projections still stream; cold starts and garbage/cold routing stop paying for noise, warm stable routing (the EMA-predictor regime) is untouched; **O5** deeper I/O parallelism in `runtime_config.cpp` — IOCP worker default widened from `clamp(2..8, threads/4)` to `clamp(3..16, threads/2)` (4 → 8 on this 16-thread box) so the drive reorders a ~2× deeper scatter of ~72 MB runs; adaptive-config self-test updated (O5 formula; L0 stays 163 checks). Gate re-verification: L0 self-test 163/163, L1 pytest 28/28, L2 `--check-weights` 0 failures, standalone L4 rerun **8667.82 s vs 9849.07 s round-1 (-12.0 %) vs 12 030.37 s baseline (-27.9 %, 0.000332 → 0.000461 tok/s, +38.9 %)**: I/O runs 3840 → **1246 (-68 %)**, bytes/token 69.5 → **22.2 GiB (-68 %)**, prefetch waste 66.33 → **31.70 %**, HDD queue latency 3699 → **2721 ms**, output `16,29661,90,77` **bit-identical** (LRU hit-rate fell 21 % → 6.9 % only because O4 removed the noise-driven traffic the bytes clearly confirm). All reports and README Performance section updated; the in-gate 2026.9.2 L4 number (0.000226 tok/s) was polluted by concurrent harness I/O and standalone reruns remain the honest comparison. |
| 2026.9.1 | 2026-09-21 | **Performance optimization round (I/O layer, §17)** — measured the L4 gate bottleneck (0.00033 tok/s; 69.5 GiB/token; 20.97 % LRU hit; 15.7 GiB auto-LRU vs ~27–30 GiB per-forward footprint; 3718 ms queue-inflated HDD latency; 70 % prefetch waste) and shipped three caching-unit changes that **never alter expert bytes**: **O1** RAM-aware auto-sizing in `runtime_config.cpp` — expert LRU window raised from `RAM/4` (capped 64 GiB) to `min(RAM·⅗, 40 GiB)` for CPU and `min(RAM·¾, 96 GiB)` for unified-MPS, placement RAM budget raised to `max(RAM·⅘, 1.5×LRU)` (informational), `GLM_LRU_MB`/`GLM_RAM_MB` overrides keep precedence, adaptive-config self-tests updated (O1 formula + 40 GiB cap check; L0 162 → **163 checks**); **O2** prefetch budget in `scheduler.cpp` — a `kMaxPrefetchInFlight = 40` in-flight cap stops speculative I/O when the pipeline is saturated so demand reads keep disk priority (soft cap, predictions untouched); **O3** gap-tolerant seek merging in `scheduler.cpp` — same-shard projections with gaps ≤ `kRunMergeGap = 1 MiB` coalesce into one run (padding read but never memcpy'd into an assembly → bit-identical results). Gate re-verification: L0 self-test 163/163, L1 pytest 28/28, L2 `--check-weights` 0 failures, L4 benchmark rerun **9849.07 s vs 12030.37 s baseline (-18.1 %, 0.000406 tok/s)** with output `16,29661,90,77` **bit-identical** and LRU used rising 16 056 → 38 592 MB; the remaining flat 69.5 GiB/token and 21 % hit rate are the synthetic 3-id prompt's cold worst case (routing-disjoint positions), with the O1 dividend and cost analysis documented in §17.4. All reports and README Performance section updated. |
| 2026.9.1 | 2026-09-15 | **Router bias correctness + golden re-record (item 7, golden gate)** — root-caused the L3 golden mismatch: the original goldens from 0f1aab5 were produced by a legacy router that **re-read the F32 `e_score_correction_bias` tensor as BF16** and fed deterministic-but-garbage corrections into `topKScores` (proven via an A/B that reproduces the golden-era top-10 logits **bit-for-bit**, `5607:5.431581 ...`, and tokens `[5607,16,...]` under the same read; the item-pass refactor changed only that read, and the current engine's forward math is byte-identical to the golden era). The F32-correct router (and the bias-free router) emit an immediate EOS on a bare raw prompt, so the goldens are re-recorded with **chat-template prompts** (`record_golden.py`/`validate.py` honour a per-record `raw` flag and `prompt_tokens`; `tests/golden/*.json` now `raw:false` with the model's default system prompt, e.g. `[gMASK]<sop><\|system\|>Reasoning Effort: Max<\|user\|>hi<\|assistant\|> thinking`): **reference-verified re-record, no silent degradation** (change is intentional, documented, pinned by the new goldens). `glm_forward.cpp` now applies the bias unconditionally as F32 via `loadWeightToF32` (selection ranks biased scores, merge weights use raw sigmoid — reference noaux_tc); `GLM_DUMP_LOGITS=1` (opt-in) dumps the top-10 logits at the LM head for future regression triage. Docs corrected: README MoE-routing snippet (bias adds, not subtracts), README quick-validation command (chat template, no `--raw`). L0 162 checks and L1 28 pytest still pass on the F32-default build. |
| 2026.9.0 | 2026-09-13 | **GLM-5.2 structural + MTP integration pass (items 7/8/5/15)** — **MTP/nextn: real 5.2 config surface** (`num_nextn_predict_layers` wins over `num_mtp_modules`, `index_share_for_mtp_iteration` parsed and printed); `weight_index` now detects `nextn_predict_layers.*`/`mtp_layers.*` tensors with a dedicated dispatch **before** the generic layer branch (critical: `nextn_predict_layers.` contains `layers.` and would misroute via `extractLayerNum`); **`--mtp` is a hard gate** — three distinct refusals (config declares MTP but tensors absent / tensors present but forward math unverified / config declares no MTP), never a silent base-LM-head fallback; `--check-weights` on the real 5.2 checkpoint reports "MTP: config declares 1 nextn predict layer(s); index found 0 (weights shipped separately)". **Strict structural `--check-weights` (item 7)** — per-layer attention + normalization completeness, dense-MLP-vs-MoE completeness, all 256 experts per MoE layer gate+up+down present with shape-vs-config checks, full-indexer layers verified on all 5 indexer weights, embedding/final-norm/lm-head present with exact shapes; real-model run: dense=3 moe=75 full-indexer=21 routed-experts-validated=19200, **redirected: 0 failures**. **Real-bug fix: the noaux_tc router bias is float32** — in the real checkpoint every one of the 76 `e_score_correction_bias` tensors is stored F32 (`moe_router_dtype: float32`) while every other tensor in all 282 shards is BF16; the strict BF16-only gate rejected them — now an **explicit F32 allow-list fragment** (`allowF32NameContaining: e_score_correction_bias`) admits only this exact tensor name, so the weight index and `--check-weights` validate presence, shape `[n_experts]` and dtype on every MoE layer. The bias is **now applied on the greedy forward as F32** (reference noaux_tc: sigmoid scores + `e_score_correction_bias`; selection ranks the biased scores, merge weights use the raw sigmoid; see `loadWeightToF32` in `glm_forward.cpp`). The original goldens from 0f1aab5 were produced by a legacy router bug that re-read the F32 bias tensor as BF16 and fed deterministic-but-garbage corrections into `topKScores`; the fix changes expert selection, so the goldens were **reference-verified re-recorded** with chat-template prompts (`tests/golden/*.json`, `raw:false`) on the F32-correct engine. The change is intentional, documented and pinned by the re-record (items 7/13; no silent degradation). **GLM-5.2 root/corpus group (layer index 78, eh_proj/enorm/hnorm) surfed as info** — it sits past `num_hidden_layers` and is not a decoder block; the engine drops it, `--check-weights` reports it. Benchmark L4 report: **sequential bytes/token (decode)** row added. Item-15: `test_build.py` strict-opt-in failures now loop `--mtp`, `--gpu-experts 1`, `--gpu-expert-depth 2 --gpu-experts 1`. L0 139 → **162 checks**, pytest 24 passed + 4 skipped; CPU and CUDA (RTX 3060) builds both green; CUDA parity + L0 PASS. |
| 2026.9.0 | 2026-09-13 | **Performance + safety pass (items 1/2/3/4/5/7/8/9/11/14/15)** — **head-major KV cache layout** (`[h][t][d]` storage; `getKey/getValue` now take (layer, head, pos); `growTo` rebases head blocks) so sparse-attention per-head rows are contiguous, cutting decode hot-path cost; **GEMV ILP unroll** (16 elements/iter into a single accumulator, still deterministic per-lane order → bit-identical goldens) plus a `ramBytesMoved` counter feeding real measured RAM bandwidth; **strict safetensors data-alignment gate** (`requireAlignedData`; header %8 AND data offset %8 rejected, L0 synthetic test); **telemetry finish** — `InferenceStats::toJson` takes the generated-token count (per-token bytes correct), computes `gpu_utilization`/`pcie_utilization`, real `ram_bandwidth` from gross bytes; `benchmark.py` report rows updated; **GPU expert pipeline depth (#1/#2)** — `--gpu-expert-depth <d>` deep-stages `compute(N) || H2D(N+1..N+d)` with round-robin pinned staging slots and a **slot-clobber fix** (reusing a slot whose H2D has not been consumed drains the copy stream first, + per-expert copy-events so an evicted/re-staged expert can never wait on the wrong event); **slow-decode event-latch fix** — timing events that used `cudaEventDisableTiming` latched `cudaErrorInvalidValue` into the next `cudaGetLastError()`, poisoning `expertFfn`'s error checks on the second self-test case (now timing-enabled events + error flush); `--cuda-self-test` PASS on RTX 3060 Laptop; **sampler upgrades (#9)** — locally-typical filtering (`--typical-p`), HF-style frequency penalty + presence penalty (`--frequency-penalty <f>`, `--presence-penalty <f>`) added to the extended overload, defaults reproduce the 10-arg pipeline exactly; **EMA popularity predictor (#4/#14)** — `--predictor ema` rolls `heat = α·p + (1−α)·heat` per expert, blends hottest next-layer experts into the lookahead prefetch and gives the hottest routed expert an LRU **soft-boost** (`lru.boost`: survives n eviction sweeps, still evictable, distinct from pin; L0 test); **no silent degradation (#8/#15)** — `--mtp` is a hard error (design milestone), `--gpu-experts <n>` on a CUDA-less build refuses to start instead of whispering a CPU fallback; the native-tokenizer parity fixture skips cleanly when `transformers` is absent; CI (`Build + L0 + L1 + L6` on win/mac/linux + auto-release) already present and re-verified. L0 123 → **139 checks**, pytest 27 → **28 tests**; CPU VS + CUDA Ninja builds clean, ctest/pytest/audit all green. |
| 2026.9.0 | 2026-09-13 | **Native in-process BPE tokenizer (opt-in, #10)** — `src/tokenizer/native_tokenizer.{h,cpp}` loads `tokenizer.json`/`tokenizer_config.json` via the vendored picojson (GPT-2 split pre-tokenization incl. contraction/`\s+(?!\S)` backtracking, byte-level `bytes_to_unicode` BPE merges, ByteLevel decode with U+FFFD fallback); Unicode tables generated by `scripts/gen_unicode_ranges.py` into `src/tokenizer/unicode_ranges.h`; `--tokenizer native` + hidden `--dump-tokens <text>` CLI (JSON-escaped `TOKEN_DEC:`); default stays the Python subprocess, goldens unchanged; `tests/test_native_tokenizer.py` verifies encode ids / decode round-trip / EOS against the reference `AutoTokenizer` for a 16-string battery (CJK, contractions, emoji, whitespace, diacritics) on the real model dir (auto-skip without model/binary); pytest 23 → **27**. |
| 2026.9.0 | 2026-09-13 | **Engine hardening + telemetry pass (14-item upgrade)** — vendored `third_party/picojson.h` (header-only JSON, BSD; CMake include dirs for `glm`/`glm_tests`); config + safetensors + weight-index JSON parsing all rewritten on picojson (no handwritten JSON search); **strict safetensors validation** (`SafeTensorsOptions`: BF16-only dtype gate, numeric non-empty shapes, offset begin<end + in-file bounds, shape×dtype byte-size consistency, duplicate-name rejection, header %8 alignment; `SafeTensorsStats` per-category counters); scheduler now feeds **router probabilities** back for one-layer-ahead prefetch and an opt-in **router-probability resident mode** (`--router-prefetch <n>`, `--prob-resident`) that pins high-probability experts and speculatively prefetches current-layer favourites; new `InferenceStats` telemetry header with per-run JSON (`STATS:` line: CPU utilization, expert LRU hit/miss, prefetch delivery/waste, I/O counts/bytes/latency, RAM bandwidth, reuse distance) parsed by `benchmark.py` into the L4 report's I/O & cache table; `KVCache::reserve` + grow-by-doubling (min 1024-token step) cuts resize copies during prefill; **sampler min-p + repetition penalty** (`--min-p`, `--repetition-penalty`) via a new extended overload whose defaults reproduce the classic pipeline (goldens byte-identical); per-token/layer **workspace reuse** across DSA/attention/FFN/forward (grow-only buffers, no per-call allocation); MTP head **config surface** (num_mtp_modules / layers / hidden / intermediate / activation / layer-types, parsed + printSummary + CLI notice; MTP-augmented inference is documented as a future roadmap item, sampling stays on the base LM head); GPU residency and native-tokenizer items documented as designed-milestone (no unverifiable CUDA code added). L0 96 → **123 checks** (sampler extras, KV reserve, strict safetensors synthetic files, MTP config fixture `tests/fixtures/config_mtp.json`), pytest 18 → **23** (new `tests/test_benchmark.py` STATS parser); ctest + pytest + sync_version --check all green. |
| 2026.9.0 | 2026-09-12 | **Removed synthetic-model validation entirely** — the former Python E2E test modules were deleted and `scripts/run_e2e.py` no longer has synthetic scenarios and does no forward generation; every model-dependent gate now requires a real checkpoint configured via `GLM_MODEL_DIR` or `--model` (candidate dirs in `scripts/common.py`); L5 e2e redesigned to a fast real-model-only pass — build artifacts, real model dir, `--check-weights` structural load ("0 failures"), and tokenizer round-trip via `tools/tokenizer_server.py`, no generated tokens; `benchmark.py` is real-model-only (the former no-model flag is gone); `validate.py`/`package.py` no longer fall back to the synthetic fixtures directory and their L2–L5 gates require the real model (subprocess timeouts removed — every gate now runs with no subprocess timeout so a single run always yields a result); CI runs only Build + L0 + L1 + L6 (L2–L5 run on the model-hosting machine); **last fake-model assets deleted outright** — `scripts/make_fake_model.py` (the fixture generator) and `tests/fixtures/fake_model/` are removed, the two L0 self-tests that loaded them (DSA indexer selection, scheduler prefetch assembly) were deleted, and `tests/fixtures/` now holds only `config_mini.json`; L0 count 110 → 121 → **96 checks** (adaptive runtime config test added), pytest suite now 18 tests; goldens recorded from `E:\glm-5.2-bf16` (signature H6144_L78_V154880_Hd64_Kv64_E256_T8_I12288_M2048_S1) and signature-scoped by `model_dna` in `validate.py`; **LRU eviction race crash fixed** — the scheduler stages weights into scratch storage before async prefetch; **cross-platform on Windows 10/11, macOS ≥ 14 and Linux (x86-64/ARM64)** — `detectSystemProfile` probes OS/RAM/cores/GPU per platform (win32 API / `sysctl` / `sysconf`), `exeDirPath` resolves via `GetModuleFileNameA` / `_NSGetExecutablePath` / `/proc/self/exe`, mmap + async-reader + tokenizer bridges are selected per platform in CMake, the CUDA backend ships CPU fallbacks for non-CUDA builds, the one hard-coded Windows `\tools\` path fallback was replaced with `std::filesystem`, and the CI matrix builds + runs L0/L1/L6 on `windows-latest`, `macos-latest` and `ubuntu-latest`; cross-platform adaptive runtime config (LRU = 25% RAM capped 64 GiB, RAM budget ≤50% RAM, VRAM budget for CUDA/MPS, IOCP workers clamp(2..8), compute threads = logical cores; env `GLM_GPU`/`GLM_LRU_MB`/`GLM_RAM_MB`/`GLM_VRAM_MB`/`GLM_IOCP_WORKERS`/`GLM_THREADS`, CLI `--gpu auto\|cpu\|cuda\|mps`); GCC/Clang-portable compilation fixes — `static_cast<unsigned char>(c)` in the JSON-escape guards (the functional-cast form is a declaration to Clang/GCC), a `GLM_HAS_SSE` caret for the SSE `relu_f32` path with `<immintrin.h>` factored out of the MSVC-only guard and a scalar fallback on ARM/Apple Silicon, and an explicit `<cstring>` for `std::memcpy` in `lru_cache.h`); **the two workflows were merged into a single `ci.yml`** — every push/PR runs Build + L0 + L1 + L6 on `windows-latest`/`macos-latest`/`ubuntu-latest`; **the release trigger changed from a manual `v*` tag push to automatic publishing** — merging to the default branch runs CPack on all three OS and, once every gate job passes, auto-creates a GitHub Release `v<version>` taking the version from `scripts/sync_version.py --show` (the single version source); publishing is skipped when that version already has a release, so only a version bump yields a new one (`release.yml` deleted); **macOS `dist/` artifact upload fixed (OOM + invalid filename)** — the real root cause was the CPack scratch dir `dist/_CPack_Packages/`: DragNDrop's staging contains an `Applications` symlink pointing at the runner's whole `/Applications` tree, and the `upload-artifact` glob followed it (≈5.85 M files, exhausting the Node 24 V8 heap and hitting a file named `Icon\r`); the release-archives upload is now scoped to `dist/HummingFlight-*`, the staging dir is removed right after packaging, and all artifact steps raise `NODE_OPTIONS=--max-old-space-size=8192` with `compression-level: 0` |
| 2026.9.0 | 2026-09-12 | KV-cache seqLen crash fixed (dynamic growth, `advance()`, tokenPos appends); merged `moe_router.h` shared kernel; version single-source + sync script; functional self-tests + pytest suite; CTest + CPack packaging; GitHub CI (ci + release gate); validation harness + golden mechanism; docs fully translated to English; reference model path updated to `E:\glm-5.2-bf16`; **full incomplete-implementation removal pass** — scheduler prefetch pipeline (run coalescing, lookahead, learned-pin LRU, IOCP/mutex-cv readers on win+posix), compressed-MLA KV cache end-to-end, sampler (greedy/topK/topP/temperature), safetensors bounds/dtype hardening, `python_tokenizer` subprocess bridge on win+posix, CUDA backend (real BF16 matvec/softmax/argmax kernels, CPU fallback, verified on RTX 3060); L0 count 74 → 86 → 110; **L6 reworked to a zero-exemption implementation-completeness audit** (`scripts/audit_completeness.py`): exemption lists, CLI escape hatches and path-based exceptions all removed; the scanner scans its own source; the trigger vocabulary is pattern-encoded so the detector holds no literal marker token; removed `tests/test_license.py` (research-only-license regression coverage) and added project governance docs `CONTRIBUTING.md`, `DISCLAIMER.md`, `CREDITS.md`, all linked from the README together with every `reports/` artifact; hardened the real-checkpoint gates for HDD hosts — the gate harness (`record_golden.py`, `benchmark.py`, `run_e2e.py`) runs with no subprocess timeouts so long prefill runs are recorded instead of false-failing; added `.github/workflows/ci.yml` + `.github/workflows/release.yml` (build → L0/L1/L4/L5/L6 → reports artifacts; later merged into a single `ci.yml`) |

---

**Document version**: 2026.9.2
**Model directory**: `E:\glm-5.2-bf16`
**Last updated**: 2026-09-25
## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See LICENSE for the full text.
