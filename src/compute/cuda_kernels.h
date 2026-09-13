#pragma once

// Device-kernel entry points (implemented in cuda_kernels.cu, compiled by nvcc).
// These return false on device error; the host wrapper falls back to the CPU.
#if defined(GLM_HAVE_CU) && GLM_HAVE_CU
namespace glm_cuda {

bool embeddingGather(const unsigned short* d_emb, int tokenId, float* d_out, int hidden);
bool matvec(const unsigned short* d_W, const float* d_x, float* d_y, int M, int K);
bool softmax(float* d_x, int n);
bool argmax(const float* d_x, int n, int* d_best);

} // namespace glm_cuda
#endif