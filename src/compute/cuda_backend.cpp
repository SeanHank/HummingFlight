#include "compute/cuda_backend.h"

#include <cmath>
#include <cstdint>

#if GLM_HAVE_CUDA
#include "compute/cuda_kernels.h"
#include "utils/logger.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <vector>

namespace glm {

namespace {

bool g_available = false;

// Host-side CPU implementations used when the GPU path is unavailable.
void cpuEmbeddingLookup(const BFloat16* emb, int tokenId, float* out, int hidden) {
    for (int i = 0; i < hidden; ++i) out[i] = emb[int64_t(tokenId) * hidden + i].toF32();
}

void cpuMatvecBf16(const BFloat16* W, const float* x, float* y, int M, int K) {
    for (int i = 0; i < M; ++i) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) acc += W[int64_t(i) * K + k].toF32() * x[k];
        y[i] = acc;
    }
}

void cpuSoftmax(float* x, int n) {
    float maxVal = x[0];
    for (int i = 1; i < n; ++i) maxVal = std::fmax(maxVal, x[i]);
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - maxVal);
        sum += x[i];
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; ++i) x[i] *= inv;
}

int cpuArgmax(const float* x, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) {
        if (x[i] > x[best]) best = i;
    }
    return best;
}

} // namespace

bool cudaInit() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0) {
        GLM_LOG_WARN("CUDA device not available");
        g_available = false;
        return false;
    }
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    GLM_LOG_INFO("CUDA device ready: " + std::string(prop.name));
    g_available = true;
    return true;
}

void cudaShutdown() {
    if (!g_available) return;
    cudaDeviceReset();
    g_available = false;
}

std::string cudaDeviceName() {
    if (!g_available) return {};
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    return std::string(prop.name);
}

size_t cudaDeviceMemoryMB() {
    if (!g_available) return 0;
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    return prop.totalGlobalMem / (1024 * 1024);
}

bool gpuAvailable() { return g_available; }

bool gpuEmbeddingLookup(const BFloat16* emb, int tokenId, float* out, int hidden) {
    if (!g_available) { cpuEmbeddingLookup(emb, tokenId, out, hidden); return true; }
    const unsigned short* src = reinterpret_cast<const unsigned short*>(emb);
    unsigned short* dEmb = nullptr;
    float* dOut = nullptr;
    cudaMalloc(&dEmb, sizeof(unsigned short) * size_t(tokenId + 1) * size_t(hidden));
    cudaMalloc(&dOut, sizeof(float) * size_t(hidden));
    if (!dEmb || !dOut) { cudaFree(dEmb); cudaFree(dOut); cpuEmbeddingLookup(emb, tokenId, out, hidden); return true; }
    cudaMemcpy(dEmb, src, sizeof(unsigned short) * size_t(tokenId + 1) * size_t(hidden), cudaMemcpyHostToDevice);
    bool ok = glm_cuda::embeddingGather(dEmb, tokenId, dOut, hidden);
    if (ok) cudaMemcpy(out, dOut, sizeof(float) * size_t(hidden), cudaMemcpyDeviceToHost);
    cudaFree(dEmb); cudaFree(dOut);
    if (ok) return true;
    cpuEmbeddingLookup(emb, tokenId, out, hidden);
    return true;
}

bool gpuMatvecBf16(const BFloat16* W, const float* x, float* y, int M, int K) {
    if (!g_available) { cpuMatvecBf16(W, x, y, M, K); return true; }
    unsigned short* dW = nullptr;
    float* dX = nullptr;
    float* dY = nullptr;
    cudaMalloc(&dW, sizeof(unsigned short) * size_t(M) * size_t(K));
    cudaMalloc(&dX, sizeof(float) * size_t(K));
    cudaMalloc(&dY, sizeof(float) * size_t(M));
    if (!dW || !dX || !dY) { cudaFree(dW); cudaFree(dX); cudaFree(dY); cpuMatvecBf16(W, x, y, M, K); return true; }
    cudaMemcpy(dW, reinterpret_cast<const unsigned short*>(W), sizeof(unsigned short) * size_t(M) * size_t(K), cudaMemcpyHostToDevice);
    cudaMemcpy(dX, x, sizeof(float) * size_t(K), cudaMemcpyHostToDevice);
    bool ok = glm_cuda::matvec(dW, dX, dY, M, K);
    if (ok) cudaMemcpy(y, dY, sizeof(float) * size_t(M), cudaMemcpyDeviceToHost);
    cudaFree(dW); cudaFree(dX); cudaFree(dY);
    if (ok) return true;
    cpuMatvecBf16(W, x, y, M, K);
    return true;
}

bool gpuSoftmax(float* x, int n) {
    if (!g_available) { cpuSoftmax(x, n); return true; }
    float* dX = nullptr;
    cudaMalloc(&dX, sizeof(float) * size_t(n));
    if (!dX) { cpuSoftmax(x, n); return true; }
    cudaMemcpy(dX, x, sizeof(float) * size_t(n), cudaMemcpyHostToDevice);
    bool ok = glm_cuda::softmax(dX, n);
    if (ok) cudaMemcpy(x, dX, sizeof(float) * size_t(n), cudaMemcpyDeviceToHost);
    cudaFree(dX);
    if (ok) return true;
    cpuSoftmax(x, n);
    return true;
}

int gpuArgmax(const float* x, int n) {
    if (!g_available) return cpuArgmax(x, n);
    float* dX = nullptr;
    int* dBest = nullptr;
    cudaMalloc(&dX, sizeof(float) * size_t(n));
    cudaMalloc(&dBest, sizeof(int));
    if (!dX || !dBest) { cudaFree(dX); cudaFree(dBest); return cpuArgmax(x, n); }
    cudaMemcpy(dX, x, sizeof(float) * size_t(n), cudaMemcpyHostToDevice);
    bool ok = glm_cuda::argmax(dX, n, dBest);
    int best = -1;
    if (ok) cudaMemcpy(&best, dBest, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dX); cudaFree(dBest);
    if (ok) return best;
    return cpuArgmax(x, n);
}

} // namespace glm
#else

namespace glm {

bool cudaInit() { return false; }
void cudaShutdown() {}
std::string cudaDeviceName() { return {}; }
size_t cudaDeviceMemoryMB() { return 0; }
bool gpuAvailable() { return false; }

bool gpuEmbeddingLookup(const BFloat16* emb, int tokenId, float* out, int hidden) {
    for (int i = 0; i < hidden; ++i) out[i] = emb[int64_t(tokenId) * hidden + i].toF32();
    return true;
}

bool gpuMatvecBf16(const BFloat16* W, const float* x, float* y, int M, int K) {
    for (int i = 0; i < M; ++i) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) acc += W[int64_t(i) * K + k].toF32() * x[k];
        y[i] = acc;
    }
    return true;
}

bool gpuSoftmax(float* x, int n) {
    float maxVal = x[0];
    for (int i = 1; i < n; ++i) maxVal = std::fmax(maxVal, x[i]);
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - maxVal);
        sum += x[i];
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; ++i) x[i] *= inv;
    return true;
}

int gpuArgmax(const float* x, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) {
        if (x[i] > x[best]) best = i;
    }
    return best;
}

} // namespace glm
#endif