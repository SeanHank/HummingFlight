#include "compute/cpu_kernels.h"
#include "engine/inference_stats.h"
#include "utils/logger.h"
#include <cmath>
#include <thread>

#if defined(_MSC_VER)
#include <immintrin.h>
#define GLM_HAS_SSE 1
#elif defined(__SSE2__)
#include <immintrin.h>
#define GLM_HAS_SSE 1
#else
// MSVC without immintrin never occurs on the supported x64 host; GCC/Clang on
// ARM (Apple Silicon, ARM64 Linux) skip SSE entirely and use the scalar path.
#define GLM_HAS_SSE 0
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace glm {

// Helper: convert 8 BF16 (128-bit) to a __m256 of F32 (zero-extend + shift).
#if defined(_MSC_VER) || defined(__SSE2__)
static inline __m256 bf16x8toF32(const BFloat16* p) {
    __m128i bf16_bits = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    __m256i u32_bits = _mm256_cvtepu16_epi32(bf16_bits);
    __m256i f32_bits = _mm256_slli_epi32(u32_bits, 16);
    return _mm256_castsi256_ps(f32_bits);
}
#endif

// ---------- AVX2 BF16 GEMV kernel ----------
// y = alpha * A @ x + beta * y
// A: [M, K] row-major BF16, x: [K] F32, y: [M] F32
// R7-5800H supports AVX2 + FMA, processes 16 BF16 -> 16 F32 -> 2 FMA ops per
// loop iteration (two independent 8-wide lanes into ONE accumulator so the
// per-lane accumulation order—and therefore the fp32 result—is bit-identical
// to the previous single-8-wide loop, only with more instruction-level
// parallelism to hide FMA latency).
void gemv_bf16_f32(const BFloat16* A, const float* x, float* y,
                   int M, int K, float alpha, float beta) {
    auto& stats = InferenceStats::instance();
    stats.ramBytesMoved += uint64_t(M) * uint64_t(K) * 2 + uint64_t(K) * 4 + uint64_t(M) * 4;
#ifdef _MSC_VER
    // AVX2 implementation: process 8 BF16 elements per lane, 16 per iteration.
    const int K8 = K & ~7;  // Round K down to a multiple of 8

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; ++i) {
        const BFloat16* row = A + int64_t(i) * K;
        __m256 sum = _mm256_setzero_ps();

        // Main loop: 16 BF16 (two 8-wide FMA groups) at a time.
        int j = 0;
        for (; j + 8 < K8; j += 16) {
            sum = _mm256_fmadd_ps(bf16x8toF32(row + j), _mm256_loadu_ps(x + j), sum);
            sum = _mm256_fmadd_ps(bf16x8toF32(row + j + 8), _mm256_loadu_ps(x + j + 8), sum);
        }
        for (; j < K8; j += 8) {
            sum = _mm256_fmadd_ps(bf16x8toF32(row + j), _mm256_loadu_ps(x + j), sum);
        }

        // Horizontal sum
        __m128 hi = _mm256_extractf128_ps(sum, 1);
        __m128 lo = _mm256_castps256_ps128(sum);
        __m128 sum128 = _mm_add_ps(hi, lo);
        // hadd twice: [s0+s1, s2+s3, s0+s1, s2+s3] -> [s0+s1+s2+s3, ...]
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float dot = _mm_cvtss_f32(sum128);

        // Tail handling (K is not a multiple of 8)
        for (; j < K; ++j) {
            dot += row[j].toF32() * x[j];
        }

        y[i] = alpha * dot + beta * y[i];
    }
#else
    // Non-MSVC fallback: scalar
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; ++i) {
        const BFloat16* row = A + int64_t(i) * K;
        float dot = 0.0f;
        for (int j = 0; j < K; ++j) {
            dot += row[j].toF32() * x[j];
        }
        y[i] = alpha * dot + beta * y[i];
    }
#endif
}

// ---------- GEMV: F32 matrix x F32 vector (AVX2 + OpenMP) ----------
void gemv_f32(const float* A, const float* x, float* y,
              int M, int K, float alpha, float beta) {
    auto& stats = InferenceStats::instance();
    stats.ramBytesMoved += uint64_t(M) * uint64_t(K) * 4 + uint64_t(K) * 4 + uint64_t(M) * 4;
#ifdef _MSC_VER
    const int K8 = K & ~7;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; ++i) {
        const float* row = A + int64_t(i) * K;
        __m256 sum = _mm256_setzero_ps();
        int j = 0;
        for (; j + 8 < K8; j += 16) {
            sum = _mm256_fmadd_ps(_mm256_loadu_ps(row + j), _mm256_loadu_ps(x + j), sum);
            sum = _mm256_fmadd_ps(_mm256_loadu_ps(row + j + 8), _mm256_loadu_ps(x + j + 8), sum);
        }
        for (; j < K8; j += 8) {
            sum = _mm256_fmadd_ps(_mm256_loadu_ps(row + j), _mm256_loadu_ps(x + j), sum);
        }
        __m128 hi = _mm256_extractf128_ps(sum, 1);
        __m128 lo = _mm256_castps256_ps128(sum);
        __m128 sum128 = _mm_add_ps(hi, lo);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float dot = _mm_cvtss_f32(sum128);
        for (; j < K; ++j) dot += row[j] * x[j];
        y[i] = alpha * dot + beta * y[i];
    }
#else
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; ++i) {
        const float* row = A + int64_t(i) * K;
        float dot = 0.0f;
        for (int j = 0; j < K; ++j) dot += row[j] * x[j];
        y[i] = alpha * dot + beta * y[i];
    }
#endif
}

// ---------- Activation functions ----------
void relu_f32(float* x, int n) {
#if GLM_HAS_SSE
    __m128 zero = _mm_setzero_ps();
    int i = 0;
    for (; i + 3 < n; i += 4) {
        __m128 v = _mm_loadu_ps(x + i);
        v = _mm_max_ps(v, zero);
        _mm_storeu_ps(x + i, v);
    }
#else
    int i = 0;
#endif
    for (; i < n; ++i) if (x[i] < 0.0f) x[i] = 0.0f;
}

void silu_f32(float* x, int n) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        x[i] = x[i] / (1.0f + std::exp(-x[i]));
    }
}

void gelu_f32(float* x, int n) {
    constexpr float c = 0.7978845608f; // sqrt(2/pi)
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        float v = x[i];
        float inner = c * (v + 0.044715f * v * v * v);
        x[i] = 0.5f * v * (1.0f + std::tanh(inner));
    }
}

// ---------- Softmax ----------
void softmax_f32(float* x, int n) {
    float maxval = x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] > maxval) maxval = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - maxval);
        sum += x[i];
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; ++i) {
        x[i] *= inv;
    }
}

int cpuThreadCount() {
#ifdef _OPENMP
    int n = omp_get_max_threads();
    if (n > 0) return n;
#endif
    int hw = int(std::thread::hardware_concurrency());
    return hw > 0 ? hw : 1;
}

} // namespace glm
