#pragma once

#include "compute/cuda_backend.h"
#include "compute/dtype_bf16.h"
#include "engine/kv_cache.h"
#include "engine/scheduler.h"
#include "model/weight_index.h"
#include <vector>

namespace glm {

// GLM-5.2 forward pass (DeepSeek-V3 style MLA + MoE + GlmMoeDsa indexer)
// Architecture: 78 layers (first 3 dense MLP, last 75 MoE)
//   Each layer = RMSNorm -> MLA Attention -> Residual -> RMSNorm -> MLP(dense/MoE) -> Residual
//
// DSA (dynamic sparse attention): on "full" indexer layers the DSA indexer
// computes, per query, a top-k selection over history positions (reference
// glm_moe_dsa semantics: interleaved RoPE, k_norm LayerNorm, ReLU scoring,
// per-head weights_proj gating, head_dim/head-count scaling). "shared" layers
// reuse the selection of the previous "full" layer. Main attention runs only
// over the selected positions (non-selected masked), using the expanded
// per-head KV cache.
//
// Storage: when a Scheduler is attached (attachScheduler), routed/shared experts
// are served from the LRU/IOCP pipeline (lookahead prefetch + batched union);
// otherwise the engine falls back to direct mmap views for tests and hardware
// without the storage pipeline.
class GLMForward {
public:
    explicit GLMForward(WeightIndex& idx) : index_(&idx) {}

    // Initialize: load embedding + lm_head + final norm into RAM, allocate KV cache
    bool init();

    // Attach the scheduler pipeline (optional; enables LRU/IOCP expert serving).
    void attachScheduler(Scheduler* s) { scheduler_ = s; }
    Scheduler* scheduler() const { return scheduler_; }

    // Opt-in GPU expert FFN (#1/#2, --gpu-experts <n>). Off by default so the
    // CPU pipeline (and golden outputs) is unchanged. Falls back to the CPU
    // implementation per layer whenever the device is absent or a call fails.
    // `capacity` bounds the VRAM-resident expert window (number of experts);
    // `stagingDepth` is how many experts are staged ahead of compute (deep
    // staging: compute(N) || H2D(N+1) ... || H2D(N+stagingDepth)).
    void enableGpuExperts(int capacity, int stagingDepth = 2);
    bool gpuExpertsEnabled() const { return gpuExperts_; }

    // Single token forward (incremental inference)
    // inputIds: full input sequence (including history), returns next token id.
    // When logitsOut is non-null the (unnormalized) logits vector over the whole
    // vocab are written there for sampling instead of returning the argmax.
    int forward(const std::vector<int>& inputIds, std::vector<float>* logitsOut = nullptr);

    // Number of tokens currently processed (KV cache length)
    int currentSeqLen() const { return seqLen_; }

    // Reset KV cache (start a new conversation)
    void reset();

    // Get reference to KV cache (for external inspection)
    const KVCache& kvCache() const { return kvCache_; }

    // Last DSA top-k position selection applied by `layer` during the most
    // recent forward step (for tests/inspection; empty until the first step).
    const std::vector<int>& lastTopk(int layer) const {
        static const std::vector<int> kEmpty;
        if (layer < 0 || layer >= int(lastTopkByLayer_.size())) return kEmpty;
        return lastTopkByLayer_[layer];
    }

private:
    // MLA attention (single layer, single token, incremental) with the full
    // DSA indexer path. `selSet` carries the shared top-k selection: "full"
    // layers overwrite it, "shared" layers consume it.
    void attentionLayer(int layer, const float* hiddenStates, float* output,
                        std::vector<int>& selSet);

    // LayerNorm (indexer k_norm: weight + bias, eps 1e-6).
    void layerNorm(const TensorLocation& w, const TensorLocation& b, float* x, int n);

    // DSA indexer for the current token: append this token's indexer key to the
    // KV cache and compute the top-k position selection (descending order).
    void dsaIndexer(int layer, const float* hiddenStates, const float* qResid,
                    std::vector<int>& selSet);

    // RMSNorm
    void rmsNorm(const TensorLocation& w, float* x, int n);

    // Dense MLP (gate * silu(up)) @ down
    void denseMlp(const LayerWeights& lw, const float* input, float* output);

    // MoE layer: router -> Top-K(noaux_tc+sigmoid) -> expert FFN -> shared + weighted merge
    void moeLayer(int layer, const float* input, float* output);

    // Single expert FFN from raw contiguous [gate|up|down] BF16 pointers
    void expertFFNRaw(const BFloat16* gateW, const BFloat16* upW, const BFloat16* downW,
                      const float* hiddenStates, float* output, int hiddenSize, int interSize);

    // Single expert FFN (fallback path via direct weights)
    void expertFFN(const ExpertWeights& ew, const float* hiddenStates,
                   float* output, int hiddenSize, int interSize);

    // Shared Experts FFN from three raw projection pointers (scheduler path)
    void sharedExpertFFNRaw(const BFloat16* gateW, const BFloat16* upW, const BFloat16* downW,
                            const float* input, float* output, int gateInter, int hidden);

    // Shared Experts FFN (fallback path via direct weights)
    void sharedExpertFFN(const LayerWeights& lw, const float* input, float* output);

    // RoPE (decoupled, interleave mode)
    void applyRoPE(float* x, int dim, int pos, int offset, bool interleave);

    // Load numel elements of a tensor into a float buffer, honoring the tensor's
    // declared dtype (BF16, or F32 -- GLM-5.2 stores the noaux_tc router
    // score-correction bias as float32 per config moe_router_dtype=float32).
    bool loadWeightToF32(const TensorLocation& loc, float* buf, int numel);

    // Raw mmap view of a tensor's data (dtype-agnostic).
    const uint8_t* getWeightRaw(const TensorLocation& loc);

    // Get BF16 pointer directly from TensorLocation (mmap view)
    const BFloat16* getWeightPtr(const TensorLocation& loc);

    // Final norm + LM head. Greedy argmax or full logits output.
    int lmHeadAndSample(const float* hiddenStates, std::vector<float>* logitsOut);

    // Preallocated per-token scratch buffers (DSA + expert pipeline; item 12:
    // workspace reuse). Reused across forward() calls and grow-only, so the hot
    // path never allocates per token/layer. Numerics are identical to the
    // previous per-call temporaries (same vectors, same data layout).
    struct Workspace {
        // DSA indexer
        std::vector<float> iq, iK, wVec;
        std::vector<std::pair<float, int>> ranked;
        // MLA attention
        std::vector<float> qResid, qB, qNope, qRope, kvA, kvLatent, kRope,
                          expandedNew, kNew, vNew, attnOut, qCur, scores;
        // Router / FFN (shared across dense/expert/shared: sized to the largest
        // intermediate dimension; calls are sequential so reuse is safe)
        std::vector<float> routerScores, bias, gate, up, mid, moeOut, expertOut, sharedOut;
        // forward() layer + lm head scratch
        std::vector<float> hiddenStates, residual, attnOutL, mlpOut, normed;
    };
    Workspace ws_;

    // Grow-only ensure helper: keeps the buffer at least n floats; no shrink.
    static void ensureFloat(std::vector<float>& v, size_t n) {
        if (v.size() < n) v.resize(n);
    }

    WeightIndex* index_;
    Scheduler* scheduler_ = nullptr;
    bool initialized_ = false;
    bool dsaEnabled_ = false;
    bool gpuExperts_ = false;
    int gpuStagingDepth_ = 2;
    int seqLen_ = 0;

    // Expanded per-head KV + indexer key cache
    KVCache kvCache_;

    // Per-layer last DSA selection (test/inspection hook)
    std::vector<std::vector<int>> lastTopkByLayer_;

    // RAM-resident weights
    std::vector<float> finalNorm_;
};

} // namespace glm