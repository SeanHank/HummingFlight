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

} // namespace glm