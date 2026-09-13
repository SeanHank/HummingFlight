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
// HEAD-MAJOR STORAGE (item 3: decode KV construction cost). Key/value buffers
// are laid out [numHeads][maxSeqLen][headDim] so a head's history across the
// selected DSA positions is a single contiguous run of cache lines:
//   index(h, t, d) = h * (maxSeqLen * headDim) + t * headDim + d
// The hot sparse-attention inner loop therefore walks selected positions
// sequentially per head instead of leaping a full token stride per position
// (token-major layout caused one 17 KB jump per selected position). Values are
// unchanged numerically - this is a pure memory-layout change.
//
// Buffers grow dynamically (growTo) instead of preallocating the worst case.

class KVCache {
public:
    KVCache() = default;

    // Pre-allocate for expected sequence length.
    void init(int numLayers, int numHeads, int qkHeadDim, int vHeadDim,
              int indexHeadDim, int maxSeqLen = 4096);

    // Grow-only reserve for a known horizon (prompt + decode budget) so the
    // per-token append path never reallocates mid-run (item: KV decode-cost
    // reduction). No-op when the current capacity already covers tokenCount.
    void reserve(int tokenCount);

    // Append current token's expanded key / value / indexer key at an explicit
    // token position. key: [numHeads*qkHeadDim], value: [numHeads*vHeadDim],
    // indexKey: [indexHeadDim].
    void appendKey(int layer, int tokenPos, const float* key);
    void appendValue(int layer, int tokenPos, const float* value);
    void appendIndexKey(int layer, int tokenPos, const float* indexKey);

    // Get pointer to cached data for head h / token t at layer l (head-major:
    // head h's token history is contiguous, so sparse attention reads cache
    // lines sequentially across the selected positions). Returns nullptr on
    // out-of-range.
    const float* getKey(int layer, int head, int tokenPos) const;
    const float* getValue(int layer, int head, int tokenPos) const;
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
        // Head-major [numHeads * maxSeqLen * headDim] float buffers.
        std::vector<float> key;      // [h][t][d], headDim = qkHeadDim
        std::vector<float> value;    // [h][t][d], headDim = vHeadDim
        std::vector<float> indexKey; // token-major [maxSeqLen * indexHeadDim]
    };

    int numLayers_ = 0;
    int numHeads_ = 0;
    int qkTotal_ = 0;
    int vDim_ = 0;
    int indexDim_ = 0;
    int maxSeqLen_ = 0;
    int seqLen_ = 0;

    // Head-h block stride inside a head-major key/value buffer (floats):
    // maxSeqLen * headDim. Used to rebase offsets when the buffer grows.
    size_t headStride_ = 0;
    int layerIndexStride_ = 0;

    std::vector<LayerBuffer> layerBuffers_;

    void growTo(int tokenPos);
};

} // namespace glm