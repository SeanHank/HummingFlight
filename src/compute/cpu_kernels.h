#pragma once

#include "compute/dtype_bf16.h"
#include <cstdint>
#include <span>
#include <vector>

namespace glm {

// CPU kernel interface (AVX2/FMA hand-written kernels, no BLAS dependency)
// Phase 1 implementation: GEMV (the main operator for batch=1 inference)

// y = alpha * A @ x + beta * y
// A: [M, K] row-major, x: [K], y: [M]
// A stored as BF16, converted to F32 during computation
void gemv_bf16_f32(const BFloat16* A, const float* x, float* y,
                   int M, int K, float alpha = 1.0f, float beta = 0.0f);

// y = alpha * A @ x + beta * y (A is F32)
void gemv_f32(const float* A, const float* x, float* y,
              int M, int K, float alpha = 1.0f, float beta = 0.0f);

// Element-wise ReLU / SiLU / GELU
void relu_f32(float* x, int n);
void silu_f32(float* x, int n);
void gelu_f32(float* x, int n);

// Softmax (along last dim)
void softmax_f32(float* x, int n);

// Returns the number of CPU threads available (for parallel splitting)
int cpuThreadCount();

} // namespace glm
