// GLM-5.2 GPU compute kernels (CUDA). Compiled only when ENABLE_CUDA=ON.
// Design doc Phase 3: the router, embedding lookup and LM head are small but
// latency-sensitive dense ops on an RTX 3060 6GB (sm_86).
//
// Data layout: weights stay BF16 on disk/RAM; the host copies the small working
// tensors (embedding row, router weights per layer, LM head) onto the GPU.
// Every kernel is designed to degrade gracefully: host wrappers fall back to
// the CPU implementation when a call fails.

#include <cuda_runtime.h>

namespace glm_cuda {

namespace {

// BF16 (raw u16) -> float, matching the CPU BFloat16::toF32() semantics
// (sign:1, exp:8, mantissa:7, stored as upper 16 bits of an f32).
__device__ __forceinline__ float bf16ToF32(unsigned short b) {
    return __uint_as_float((unsigned)b << 16);
}

__global__ void embeddingGatherKernel(const unsigned short* emb, int tokenId,
                                      float* out, int hidden) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden) return;
    out[i] = bf16ToF32(emb[(size_t)tokenId * hidden + i]);
}

// y = W @ x  (W stored row-major [M x K], BF16; x and y F32).
__global__ void matvecKernel(const unsigned short* W, const float* x, float* y,
                            int M, int K) {
    __shared__ float partial[256];
    int row = blockIdx.x;
    if (row >= M) return;
    int tid = threadIdx.x;

    float acc = 0.0f;
    for (int k = tid; k < K; k += blockDim.x) {
        acc += bf16ToF32(W[(size_t)row * K + k]) * x[k];
    }
    partial[tid] = acc;
    __syncthreads();
    int stride = blockDim.x >> 1;
    while (stride > 0) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
        stride >>= 1;
    }
    if (tid == 0) y[row] = partial[0];
}

// In-place softmax over a single n-vector (single block).
__global__ void softmaxKernel(float* x, int n) {
    __shared__ float sMax;
    __shared__ float sExp;
    int tid = threadIdx.x;

    float localMax = tid < n ? x[tid] : -1e30f;
    for (int i = tid + blockDim.x; i < n; i += blockDim.x) {
        localMax = fmaxf(localMax, x[i]);
    }
    __shared__ float sreduced[256];
    sreduced[tid] = localMax;
    __syncthreads();
    int stride = blockDim.x >> 1;
    while (stride > 0) {
        if (tid < stride) sreduced[tid] = fmaxf(sreduced[tid], sreduced[tid + stride]);
        __syncthreads();
        stride >>= 1;
    }
    if (tid == 0) sMax = sreduced[0];
    __syncthreads();

    float localSum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        localSum += __expf(x[i] - sMax);
    }
    sreduced[tid] = localSum;
    __syncthreads();
    stride = blockDim.x >> 1;
    while (stride > 0) {
        if (tid < stride) sreduced[tid] += sreduced[tid + stride];
        __syncthreads();
        stride >>= 1;
    }
    if (tid == 0) sExp = sreduced[0];
    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x) {
        x[i] = __expf(x[i] - sMax) / sExp;
    }
}

__global__ void argmaxKernel(const float* x, int n, int* best) {
    __shared__ float sreduced[256];
    __shared__ int sindex[256];
    int tid = threadIdx.x;

    float localBest = -1e30f;
    int localIdx = 0;
    for (int i = tid; i < n; i += blockDim.x) {
        if (x[i] > localBest) {
            localBest = x[i];
            localIdx = i;
        }
    }
    sreduced[tid] = localBest;
    sindex[tid] = localIdx;
    __syncthreads();
    int stride = blockDim.x >> 1;
    while (stride > 0) {
        if (tid < stride) {
            if (sreduced[tid + stride] > sreduced[tid]) {
                sreduced[tid] = sreduced[tid + stride];
                sindex[tid] = sindex[tid + stride];
            }
        }
        __syncthreads();
        stride >>= 1;
    }
    if (tid == 0) best[0] = sindex[0];
}

// Silu + elementwise multiply (in place on act): act[i] = silu(act[i]) * up[i].
__global__ void siluScaleKernel(float* act, const float* up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = act[i];
    act[i] = (g / (1.0f + __expf(-g))) * up[i];
}

// matvecKernel on an explicit stream (stream-safe variant).
__global__ void matvecStreamKernel(const unsigned short* W, const float* x, float* y,
                                   int M, int K) {
    __shared__ float partial[256];
    int row = blockIdx.x;
    if (row >= M) return;
    int tid = threadIdx.x;

    float acc = 0.0f;
    for (int k = tid; k < K; k += blockDim.x) {
        acc += bf16ToF32(W[(size_t)row * K + k]) * x[k];
    }
    partial[tid] = acc;
    __syncthreads();
    int stride = blockDim.x >> 1;
    while (stride > 0) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
        stride >>= 1;
    }
    if (tid == 0) y[row] = partial[0];
}

} // namespace

// ---------- Device helpers with failure reporting ----------

int lastError() {
    return int(cudaGetLastError());
}

bool embeddingGather(const unsigned short* d_emb, int tokenId, float* d_out, int hidden) {
    embeddingGatherKernel<<<(hidden + 127) / 128, 128>>>(d_emb, tokenId, d_out, hidden);
    return cudaGetLastError() == cudaSuccess;
}

// Row-major BF16 weight matrix on device: launch grid of M blocks, one per row.
bool matvec(const unsigned short* d_W, const float* d_x, float* d_y, int M, int K) {
    matvecKernel<<<M, 256>>>(d_W, d_x, d_y, M, K);
    return cudaGetLastError() == cudaSuccess;
}

bool softmax(float* d_x, int n) {
    softmaxKernel<<<1, 256>>>(d_x, n);
    return cudaGetLastError() == cudaSuccess;
}

bool argmax(const float* d_x, int n, int* d_best) {
    argmaxKernel<<<1, 256>>>(d_x, n, d_best);
    return cudaGetLastError() == cudaSuccess;
}

bool expertFfn(const unsigned short* d_gateW, const unsigned short* d_upW,
               const unsigned short* d_downW, const float* d_x, float* d_out,
               float* d_act, float* d_up, int hidden, int inter,
               cudaStream_t stream) {
    // out = down @ (silu(x @ gate^T) .* (x @ up^T))
    matvecStreamKernel<<<inter, 256, 0, stream>>>(d_gateW, d_x, d_act, inter, hidden);
    if (cudaGetLastError() != cudaSuccess) return false;
    matvecStreamKernel<<<inter, 256, 0, stream>>>(d_upW, d_x, d_up, inter, hidden);
    if (cudaGetLastError() != cudaSuccess) return false;
    siluScaleKernel<<<(inter + 255) / 256, 256, 0, stream>>>(d_act, d_up, inter);
    if (cudaGetLastError() != cudaSuccess) return false;
    matvecStreamKernel<<<hidden, 256, 0, stream>>>(d_downW, d_act, d_out, hidden, inter);
    return cudaGetLastError() == cudaSuccess;
}

} // namespace glm_cuda