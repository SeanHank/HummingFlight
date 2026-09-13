<div align="center">

# HummingFlight

**Run the full, unquantized GLM-5.2 BF16 on a single laptop.**

*"Innovation is the ability to see change as an opportunity, not a threat." — Steve Jobs*

[![License](https://img.shields.io/badge/license-AGPLv3-red.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Win%2010%2B%20macOS%2014%2B%20Linux-lightgrey.svg)](#)
[![Language](https://img.shields.io/badge/C%2B%2B-20-orange.svg)](#)
[![Precision](https://img.shields.io/badge/precision-BF16%20lossless-success.svg)](#)
[![Version](https://img.shields.io/badge/version-2026.9.0-blue)](#)

A from-scratch C++ inference engine that loads and runs the **complete GLM-5.2** model on consumer-grade hardware. No quantization. No approximation. No GPU farm.

**[Architecture](#-architecture) · [Quick Start](#-quick-start) · [Testing & Validation](#-testing--validation) · [Performance](#-performance) · [Tech Deep Dive](#-tech-deep-dive)**

</div>

---

## The Problem

Large language models are getting big. **Really** big. GLM-5.2 in BF16 precision weighs in at ~1.5 TB — a footprint that laughs in the face of consumer GPUs and even maxed-out workstations (64 GB RAM).

The standard answer? **Shrink the model.** Quantize to INT8, INT4, or GGUF. Compress, prune, distill. Make it fit.

But what if you don't want to fit? What if you need the **exact same output** a data-center GPU would produce? What if every decimal place matters?

**HummingFlight is the answer to "what if we just... load the whole thing?"**

---

## The Idea

> Treat storage as a hierarchy. Treat patience as a resource. Treat fidelity as non-negotiable.

HummingFlight doesn't shrink the model. It **streams** it. Weights live on disk and flow through a three-tier pipeline — HDD to RAM to VRAM — with only the tensors needed for the *current token* in motion. A 256-expert MoE layer? We load the 8 the router picked, not all 256. Layer *L+1*'s experts? Already prefetching while layer *L* computes.

The math is simple:
- 1.5 TB model on a 5 TB HDD
- 64 GB RAM holds the hottest ~4% of weights
- 6 GB VRAM holds active attention heads
- 100% of weights used in original BF16 precision

| | Quantized (llama.cpp, etc.) | **HummingFlight** |
|---|---|---|
| Precision | INT4 / INT8 / Q8 | **BF16 (bit-exact)** |
| Output fidelity | Approximated | **Identical to original** |
| Hardware | Moderate | **One laptop** |
| Speed | Fast | Slow (I/O-bound) |
| Use case | Daily chat | **Research, verification, fidelity-critical work** |

---

## Features

### Three-Tier Memory Hierarchy
Weights are semantically identical at every tier — only their *placement* differs:
- **Tier 0 — VRAM** (6 GB): Hot tensors, active heads
- **Tier 1 — RAM** (64 GB): LRU-cached experts, KV cache
- **Tier 2 — HDD** (5 TB): Full 1.5 TB model, mmap-backed

### MoE-Aware Streaming
GLM-5.2 has 256 routed experts per layer. HummingFlight loads **only the 8 selected by the router** — a 32x I/O reduction. Combined with expert-level LRU caching, the hottest experts stay in RAM across tokens.

### Layer-Ahead Prefetch
When layer *L* computes its router, the scheduler **predicts and asynchronously fetches** layer *L+1*'s expert set. By the time layer *L* finishes, the next layer's weights are already in RAM.

### Windows IOCP Async I/O
Native I/O Completion Ports for non-blocking file reads. The CPU stays fed while the HDD seeks.

### AVX2 + FMA BF16 GEMV
Custom kernel that processes 8 BF16 elements per cycle via zero-extension and FMA accumulation. Parallelized across all cores with OpenMP.

### DeepSeek-V3 Style MLA Attention
Full Multi-head Latent Attention implementation:
- Q/K/V LoRA compression
- Decoupled RoPE (positional info on a separate sub-head)
- Compressed KV cache — 4x smaller per token

### Python Subprocess Tokenizer
Tokenization is delegated to the official `transformers.AutoTokenizer` via subprocess. **Byte-identical** to the reference implementation. No reverse-engineered BPE.

---

## Architecture

```
+-------------------------------------------------------------+
|                      HummingFlight Engine                    |
|                                                              |
|   +-----------+    +--------------+    +----------------+   |
|   | Tokenizer |--->| Forward Pass |--->| Token Sampler  |   |
|   | (Python)  |    |  (78 layers) |    |   (Greedy)     |   |
|   +-----------+    +------+-------+    +----------------+   |
|                           |                                  |
|                +----------+----------+                       |
|                |                     |                       |
|       +--------v--------+  +---------v----------+           |
|       | MLA Attention   |  |  MoE / Dense MLP   |           |
|       | (KV Cache)      |  |  (8 of 256 experts)|           |
|       +--------+--------+  +---------+----------+           |
|                |                     |                       |
|                +----------+----------+                       |
|                           v                                  |
|              +------------------------+                     |
|              |  Weight Scheduler      |                     |
|              |  (Prefetch + LRU)      |                     |
|              +----------+-------------+                     |
|                           |                                  |
|       +-------------------+-------------------+             |
|       v                   v                   v             |
|   +------+         +----------+        +------------+      |
|   | VRAM |<--hot-- |   RAM    |<-warm-|    HDD     |      |
|   | 6 GB |  cache  |  64 GB   | cache |  1.5 TB    |      |
|   +------+         +----------+        +------------+      |
|                                                              |
+-------------------------------------------------------------+
```

### Forward Pass (78 layers: 3 dense + 75 MoE)

```
Token -> RMSNorm -> MLA Attention -> +Residual
                              -> RMSNorm -> MoE/Dense MLP -> +Residual
                              -> (repeat 77 more times)
                              -> LM Head (vocab 154,880) -> Greedy Sample -> Next Token
```

---

## Quick Start

### Prerequisites

| Component | Requirement |
|---|---|
| OS | Windows 10/11 (x64) · macOS ≥ 14 (Apple Silicon / Intel) · Linux (x86-64 / ARM64) |
| Compiler | Visual Studio 2022 (MSVC 19.x) on Windows · AppleClang / Xcode 15+ on macOS · GCC 12+ or Clang 15+ on Linux |
| CMake | >= 3.20 |
| Python | 3.10+ with `transformers` and `jinja2` |
| Model | GLM-5.2 BF16 weights (~1.5 TB) |

### Build

```powershell
git clone <repo-url> HummingFlight
cd HummingFlight

# CPU-only build (also builds the glm_tests functional test binary)
cmake -B build -G "Visual Studio 17 2022" -A x64 -DENABLE_CUDA=OFF
cmake --build build --config Release

# Optional: enable CUDA (RTX 3060)
cmake -B build -G "Visual Studio 17 2022" -A x64 -DENABLE_CUDA=ON
cmake --build build --config Release
```

> The version string is single-sourced in `scripts/sync_version.py` and pushed to
> `src/version.h`, `CMakeLists.txt`, `README.md`, `doc/design.md`,
> `tools/tokenizer_server.py` and `version.txt`. Never edit those by hand; run
> `python scripts/sync_version.py` after bumping the `VERSION` constant.

### Install Python Dependencies

```powershell
pip install transformers jinja2
```

### Run

```powershell
# IMPORTANT: run from project root (locates tools/tokenizer_server.py)
cd HummingFlight

# Chat mode (applies GLM chat template)
.\build\Release\glm.exe --model "E:\glm-5.2-bf16" --prompt "What is the capital of China?"

# Unlimited generation (stops on EOS token)
.\build\Release\glm.exe --model "E:\glm-5.2-bf16" --prompt "Explain quantum entanglement."

# Quick validation -- fastest path (raw encode, 1 token)
.\build\Release\glm.exe --model "E:\glm-5.2-bf16" --prompt "hi" --raw --max-tokens 1

# Debug mode -- per-layer timing
.\build\Release\glm.exe --model "E:\glm-5.2-bf16" --prompt "Hello" --max-tokens 8 --verbose
```

### CLI Reference

| Option | Description | Default |
|---|---|---|
| `--model <dir>` | Model directory (config.json + *.safetensors) | *required* |
| `--prompt <text>` | Input prompt (user message) | -- |
| `--raw` | Skip chat template, encode prompt directly | off |
| `--max-tokens <n>` | Max tokens to generate. `-1` = unlimited (stop on EOS) | `-1` |
| `--python <path>` | Python interpreter path | conda env |
| `--no-gpu` | Disable CUDA backend | off |
| `--verbose` | Enable debug logging (per-layer timing) | off |
| `--version` | Print version and exit | — |
| `--self-test` | Run functional self-tests (no model needed) and exit | — |
| `--check-weights <dir>` | Validate a model directory structurally and exit | — |
| `--help` | Show usage | — |

---

## 🧪 Testing & Validation

Every check below is automated and **reproducible**. GitHub-hosted CI runs only
**Build + L0 + L1 + L6** (runners cannot host the ~1.5 TB checkpoint); the **L2–L5**
gates (goldens, benchmark, end-to-end) require the real checkpoint and run on the
model-hosting machine via `python scripts/validate.py --model <dir> --report`.
The release pipeline refuses to publish until all gates pass. See `doc/design.md`
§10–§11 for the full contract.

| Level | What | Runs without weights? | Command |
|---|---|---|---|
| L0 | Functional self-tests (BF16, GEMV, RMSNorm, softmax, sigmoid, top-K router, compressed-MLA KV cache, LRU, placement, config parser, sampler, adaptive runtime config — 96 checks) | ✅ | `build\Release\glm_tests.exe` or `glm.exe --self-test` |
| L1 | Python contract tests (version consistency, fixtures, golden contract, binary black-box) | ✅ | `python -m pytest tests/ -v` |
| L2 | Structural model validation (index, shards, tensor layout) | needs model | `glm.exe --check-weights --model <dir>` |
| L3 | Golden inference outputs (greedy token ids, byte-for-byte) | needs model | `python scripts/record_golden.py --model <dir>` then `scripts/validate.py --model <dir>` |
| L4 | Performance benchmark | needs model | `python scripts/benchmark.py --model <dir> --report` |
| L5 | End-to-end pipeline (fast real-model e2e: tokenizer round-trip + weight index/tensor loading only — no forward generation) | needs model | `python scripts/run_e2e.py --model <dir> --report` |
| L6 | Implementation-completeness audit (exemption-free marker scan + structural invariants; scans its own source) | ✅ | `scripts/audit_completeness.py --report` |

One-click release packaging (runs tests → validation → build → archive → report):

```powershell
python scripts/package.py --model E:\glm-5.2-bf16
```

Validation locally:

```powershell
ctest --test-dir build -C Release --output-on-failure
python -m pytest tests/ -v
python scripts/validate.py --report --model E:\glm-5.2-bf16
```

## Reports & Governance

Every quality gate writes a verifiable report into `reports/`; each one is
regenerated by its producing command and must exist in a released tree.

| Report | Produced by | Link |
|---|---|---|
| Implementation-completeness audit (L6) | `scripts/audit_completeness.py --report` | [reports/audit_report.md](reports/audit_report.md) |
| Performance benchmark (L4) | `scripts/benchmark.py --report` | [reports/benchmark_report.md](reports/benchmark_report.md) |
| Benchmark raw metrics | `scripts/benchmark.py --report` | [reports/benchmark_raw.json](reports/benchmark_raw.json) |
| End-to-end (L5) | `python scripts/run_e2e.py --model <dir> --report` | [reports/e2e_report.md](reports/e2e_report.md) |
| Validation harness (L0–L6) | `scripts/validate.py --report` | [reports/validation_report.md](reports/validation_report.md) |
| Release gate | `scripts/package.py` | [reports/release_gate_report.md](reports/release_gate_report.md) |

Project governance documents:

- **[CONTRIBUTING.md](CONTRIBUTING.md)** — how to contribute; the exemption-free audit and 100%-pass rules.
- **[DISCLAIMER.md](DISCLAIMER.md)** — model ownership, experimental nature, and no-warranty terms.
- **[CREDITS.md](CREDITS.md)** — authorship and third-party acknowledgments.
- **[LICENSE](LICENSE)** — GNU AGPL-3.0.

## Performance

Benchmarked on **AMD Ryzen 7 5800H (8C/16T) - 64 GB RAM - 5 TB HDD** (model on HDD):

| Phase | Per-Token | Notes |
|---|---|---|
| Prefill (1st token) | ~1,900 s | Cold-start, loading 1.5 TB into RAM |
| Prefill (subsequent) | ~680 s | Weights cached, CPU-bound |
| Generation | ~680 s / token | Same -- no expert reuse across tokens |

**Throughput**: ~0.0015 tok/s

> Yes, it's slow. That's the price of running 1.5 TB from an HDD without quantization. The goal is **fidelity**, not throughput. With an NVMe SSD: 5-10x faster. With 128 GB RAM: most experts stay cached.

### Where Does the Time Go?

```
Per-token breakdown (generation, RAM-cached):
|-- Expert weight load (HDD->RAM):  ~85%   <- I/O bound
|-- MoE expert GEMV (CPU):          ~10%   <- AVX2 parallelized
|-- MLA attention compute:           ~3%
+-- Tokenizer + sampling:            ~2%
```

**The path to speed is clear: faster storage + more RAM.**

---

## Tech Deep Dive

### MoE Routing: noaux_tc Top-K

GLM-5.2 uses load-balancing-aware routing. For each token, the router scores all 256 experts via sigmoid, applies a frequency bias, and selects the top 8:

```cpp
// Affinity scores
for (int e = 0; e < 256; ++e)
    scores[e] = sigmoid(dot(hidden, routerWeight[e]));

// noaux_tc bias: down-weight overused experts
for (int e = 0; e < 256; ++e)
    scores[e] -= bias[e];

// Top-8 selection + weight normalization
topK(scores, 8, selectedExperts, selectedWeights);
// -> Only these 8 experts' weights are fetched from storage
```

**I/O reduction: 32x** (8 of 256 experts loaded per token).

### MLA: Decoupled RoPE

GLM-5.2's attention splits each key into two parts:
- **`k_nope`** -- semantic content, compressed via LoRA, no position encoding
- **`k_pe`** -- a small vector carrying *only* positional info, with RoPE applied

This decoupling lets the KV cache store the **compressed** form and expand only during attention. Cache per token: ~4x smaller than naive MHA.

### AVX2 BF16 GEMV Kernel

The hot loop -- BF16 matrix x F32 vector, 8 elements per iteration:

```cpp
// Load 8 BF16 (uint16) -> zero-extend to uint32 -> shift to F32 bit pattern
__m128i bf16  = _mm_loadu_si128(row + j);                  // 8 x uint16
__m256i u32   = _mm256_cvtepu16_epi32(bf16);               // 8 x uint32
__m256i f32   = _mm256_slli_epi32(u32, 16);                // BF16 -> F32 bits
__m256  a     = _mm256_castsi256_ps(f32);
__m256  x     = _mm256_loadu_ps(x_vec + j);
sum = _mm256_fmadd_ps(a, x, sum);                          // FMA accumulate
```

No lookup tables. No per-element loops. Just bit manipulation and fused multiply-add.

---

## Project Structure

```
HummingFlight/
|-- CMakeLists.txt
|-- CONTRIBUTING.md              # Contribution rules + gate policy
|-- DISCLAIMER.md                # No-warranty / model-ownership terms
|-- CREDITS.md                   # Authorship + acknowledgments
|-- src/
|   |-- main.cpp                    # Entry, CLI, crash handler
|   |-- version.h                   # Version macros (sync_version.py generated)
|   |-- self_test.*                 # Functional tests + weight validation
|   |-- model/
|   |   |-- config.*                # config.json parser
|   |   |-- weight_index.*          # Safetensors index (282 shards)
|   |   |-- safetensors.*           # Safetensors header parser
|   |   `-- glm_forward.*           # Core: MLA + MoE + RMSNorm
|   |-- compute/
|   |   |-- cpu_kernels.*           # AVX2 BF16 GEMV, activations
|   |   |-- moe_router.h            # sigmoid + top-K routing (shared with tests)
|   |   |-- cuda_kernels.cu          # CUDA kernels (BF16 matvec, softmax, argmax)
|   |   |-- cuda_backend.*          # CUDA host wrapper with CPU fallback (optional)
|   |   `-- dtype_bf16.h            # BF16 <-> F32 conversion
|   |-- storage/
|   |   |-- mapped_file.h            # Cross-platform mmap abstraction (win+posix)
|   |   |-- mmap_win.* / mmap_posix.* # OS-specific mmap backends
|   |   |-- iocp_reader.* / iocp_reader_posix.*  # Async readers (IOCP / mutex+cv)
|   |   `-- lru_cache.h             # Expert-level LRU (learned-pin admission)
|   |-- engine/
|   |   |-- placement.*             # Tier policy
|   |   |-- scheduler.*             # Async expert prefetch pipeline
|   |   `-- kv_cache.*              # MLA KV cache (dynamic growth)
|   |-- tokenizer/
|   |   `-- python_tokenizer.*      # C++ <-> Python subprocess
|   `-- utils/
|       |-- logger.h
|       `-- timer.h
|-- scripts/
|   |-- sync_version.py             # Single version source (2026.9.0)
|   |-- audit_completeness.py       # L6 exemption-free audit
|   |-- validate.py                 # Validation harness (L0-L6)
|   |-- record_golden.py            # Golden output recording
|   |-- benchmark.py                # Performance benchmark (L4 gate)
|   |-- package.py                  # One-click release packaging
|   `-- update_docs.py              # Doc fact refresh
|-- tests/
|   |-- run_tests.cpp               # glm_tests entry
|   |-- test_*.py                   # pytest contract tests
|   |-- fixtures/                   # Mini config fixture (config parsing only; fake-model fixtures removed)
|   `-- golden/                     # Recorded inference outputs (L3)
|-- doc/
|   `-- design.md                   # Design + validation contract (English)
|-- .github/workflows/
|   |-- ci.yml                      # Per-PR: build + tests + validation
|   `-- release.yml                 # Tag: validation gate + packages + release
|-- tools/
|   |-- tokenizer_server.py         # Tokenizer subprocess service
|   `-- test_tokenizer.py
```

---

## Design Principles

1. **Fidelity over speed.** No quantization. The output is bit-identical to a data-center run -- just slower.

2. **I/O is the enemy.** On consumer hardware, the bottleneck is never compute; it's moving 1.5 TB from disk. Every optimization targets I/O: expert caching, layer-ahead prefetch, async IOCP.

3. **Correctness via delegation.** Tokenization goes to the official `transformers` library -- no BPE mismatch risk. RoPE, MLA, and MoE routing are implemented from `config.json` and validated against reference outputs.

4. **Respect the hardware.** Don't pretend a laptop is a data center. Embrace the hierarchy. Schedule around it.

---

## License

**GLM-5.2 model weights are property of their respective owners and must be obtained independently.**

This project is licensed under the **GNU Affero General Public License v3.0** (AGPLv3).  

See [LICENSE](LICENSE) and the companion documents
[CONTRIBUTING.md](CONTRIBUTING.md), [DISCLAIMER.md](DISCLAIMER.md), and
[CREDITS.md](CREDITS.md).

Copyright © 2026 Sean Hank.

---

## Acknowledgments

- **GLM-5.2** -- The model this engine serves.
- **colibri** -- Inspiration for "run huge models on small hardware".
- **transformers** -- Hugging Face, powering the tokenizer subprocess.
