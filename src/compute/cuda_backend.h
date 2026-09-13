#pragma once

#if defined(GLM_ENABLE_CUDA) && GLM_ENABLE_CUDA
#define GLM_HAVE_CUDA 1
#else
#define GLM_HAVE_CUDA 0
#endif

#include "compute/dtype_bf16.h"
#include <cstddef>
#include <string>

// CUDA backend (design doc Phase 3). The engine offloads the latency-sensitive
// dense ops (embedding lookup, router gate, LM head) to an RTX 3060 6GB
// (sm_86). Every op has a CPU fallback so a missing/failed device never blocks
// inference.
//
// When built without -DENABLE_CUDA the host wrappers compile to the CPU
// implementations and gpuAvailable() returns false.
namespace glm {

// Initialize CUDA device, returns true on success.
bool cudaInit();

// Release CUDA resources.
void cudaShutdown();

// Returns GPU name and memory (MB) (empty / 0 when CUDA unavailable).
std::string cudaDeviceName();
size_t cudaDeviceMemoryMB();

// True after a successful cudaInit().
bool gpuAvailable();

// ---- Offloaded ops (GPU preferred, CPU fallback; returns true on success) ----

// Embedding row lookup: out[i] = emb[tokenId][i] (BF16 -> F32).
bool gpuEmbeddingLookup(const BFloat16* emb, int tokenId, float* out, int hidden);

// y = W @ x, W row-major [M x K] BF16, x/y F32.
bool gpuMatvecBf16(const BFloat16* W, const float* x, float* y, int M, int K);

// In-place softmax over an n-vector.
bool gpuSoftmax(float* x, int n);

// Argmax index (or -1 on failure).
int gpuArgmax(const float* x, int n);

// ---- GPU expert FFN residency cache (#1/#2, opt-in --gpu-experts <n>) ----
// Offloads the routed-expert FFN to the GPU. Weights stay BF16 on the host;
// the cache keeps up to `capacity` experts resident in VRAM keyed by
// (layer, expertId) and streams x/out over the bus per layer. Staging uses a
// deep pinned-memory + async H2D pipeline (`stagingDepth` slots): the N+1 and
// N+2 experts' copies overlap the current expert's compute, exactly the
// compute(N) || H2D(N+1) || H2D(N+2) pattern of item 2. Every op degrades
// gracefully: a missing device or any CUDA error reports false and the caller
// keeps the CPU implementation (the CPU path is the correctness fallback, and
// CLI-requested GPU offload that cannot start is a hard error at startup, not
// a silent degradation).

// Configure the cache capacity in number of experts (0 = disabled) and the
// staging pipeline depth (number of experts copied ahead, >= 1). Allocates
// pinned host staging + device buffers. Safe to call once after cudaInit().
bool gpuExpertCacheConfig(int capacity, int stagingDepth = 2);

// Release all expert-cache resources.
void gpuExpertCacheShutdown();

// Returns true when the cache is enabled and the device is available.
bool gpuExpertsEnabled();

// Stage one expert for compute. `w` points at a transient host buffer holding
// [gate | up | down] (BF16, contiguous, 6 * hidden * inter * sizeof(uint16_t)).
// The bytes are copied synchronously into a (round-robin) pinned staging slot
// and uploaded asynchronously (stagingDepth-deep), so the caller may reuse `w`
// right away. No-op when the expert is already resident (LRU touch only).
// Returns false on device error.
bool gpuExpertStage(int layer, int expertId, const BFloat16* w, int hidden, int inter);

// Run the FFN of a staged expert: out = down @ (silu(x @ gate^T) .* (x @ up^T)).
// Waits for the expert's pending H2D copy, executes on the compute stream and
// copies the result back. Returns false if not resident or on device error
// (caller falls back to CPU).
bool gpuExpertFFN(int layer, int expertId, const float* x, float* out,
                  int hidden, int inter);

// GPU expert FFN numeric parity check (device vs the host reference). Returns
// true when CUDA is unavailable (skipped) or when every generated sample
// matches the CPU result within tolerance.
bool cudaExpertSelfTest();

} // namespace glm