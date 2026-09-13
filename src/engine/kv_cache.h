#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

namespace glm {

// Expanded MLA KV + DSA indexer key cache (GLM-5.2 / GlmMoeDsa).
//
// Per token per layer we store the EXPANDED per-head key and value produced by
// kv_b_proj, exactly like the reference implementation caches key_states /
// value_states (the sparse-attention path cannot re-derive selected positions
// on demand, so expanded K/V is cached):
//   key:   [numHeads * qkHeadDim]           k[h] = [k_nope(192) | k_rope(64)]
//   value: [numHeads * vHeadDim]            v[h] = kv_b_proj rows (v dims)
//   idxK:  [indexHeadDim]                   DSA indexer key (full layers only)
//
// Memory/token/layer = numHeads*(qkHeadDim + vHeadDim) + indexHeadDim floats,
// mirroring the reference expanded layout (previously a compressed-latent cache
// was used; the reference materialized expanded K/V and so do we).
//
// Buffers grow dynamically (growTo) instead of preallocating the worst case.

class KVCache {
public:
    KVCache() = default;

    // Pre-allocate for expected sequence length.
    void init(int numLayers, int numHeads, int qkHeadDim, int vHeadDim,
              int indexHeadDim, int maxSeqLen = 4096);

    // Append current token's expanded key / value / indexer key at an explicit
    // token position. key: [numHeads*qkHeadDim], value: [numHeads*vHeadDim],
    // indexKey: [indexHeadDim].
    void appendKey(int layer, int tokenPos, const float* key);
    void appendValue(int layer, int tokenPos, const float* value);
    void appendIndexKey(int layer, int tokenPos, const float* indexKey);

    // Get pointer to cached data for token t at layer l (t < maxSeqLen).
    const float* getKey(int layer, int tokenPos) const;
    const float* getValue(int layer, int tokenPos) const;
    const float* getIndexKey(int layer, int tokenPos) const;

    // Advance sequence length by one (call once per token after all layers appended).
    void advance() { if (seqLen_ < maxSeqLen_) ++seqLen_; }

    int seqLen() const { return seqLen_; }
    int maxSeqLen() const { return maxSeqLen_; }
    int numLayers() const { return numLayers_; }

    void clear();

    // Dimensional accessors.
    int numHeads() const { return numHeads_; }
    int qkHeadDim() const { return qkTotal_; }
    int vHeadDim() const { return vDim_; }
    int indexHeadDim() const { return indexDim_; }

private:
    struct LayerBuffer {
        std::vector<float> key;      // contiguous [maxSeqLen * numHeads * qkHeadDim]
        std::vector<float> value;    // contiguous [maxSeqLen * numHeads * vHeadDim]
        std::vector<float> indexKey; // contiguous [maxSeqLen * indexHeadDim]
    };

    int numLayers_ = 0;
    int numHeads_ = 0;
    int qkTotal_ = 0;
    int vDim_ = 0;
    int indexDim_ = 0;
    int maxSeqLen_ = 0;
    int seqLen_ = 0;

    int layerKeyStride_ = 0;
    int layerValueStride_ = 0;
    int layerIndexStride_ = 0;

    std::vector<LayerBuffer> layerBuffers_;

    void growTo(int tokenPos);
};

} // namespace glm