#pragma once

// Device-kernel entry points (implemented in cuda_kernels.cu, compiled by nvcc).
// These return false on device error; the host wrapper falls back to the CPU.
#if defined(GLM_HAVE_CU) && GLM_HAVE_CU
namespace glm_cuda {

bool embeddingGather(const unsigned short* d_emb, int tokenId, float* d_out, int hidden);
bool matvec(const unsigned short* d_W, const float* d_x, float* d_y, int M, int K);
bool softmax(float* d_x, int n);
bool argmax(const float* d_x, int n, int* d_best);

// Routed-expert FFN for a single token:
//   out = down @ (silu(x @ gate^T) .* (x @ up^T))
// gate/up: [inter x hidden] row-major BF16, down: [hidden x inter] BF16.
// d_act/d_up are per-expert activation scratch (inter floats each); d_x/d_out
// are shared input/output buffers (hidden floats). Runs on `stream`. See the
// double-buffered host pipeline in cuda_backend.cpp.
bool expertFfn(const unsigned short* d_gateW, const unsigned short* d_upW,
               const unsigned short* d_downW, const float* d_x, float* d_out,
               float* d_act, float* d_up, int hidden, int inter,
               cudaStream_t stream);

} // namespace glm_cuda
#endif