#include "engine/kv_cache.h"

namespace glm {

void KVCache::init(int numLayers, int numHeads, int qkHeadDim, int vHeadDim,
                   int indexHeadDim, int maxSeqLen) {
    numLayers_ = numLayers;
    numHeads_ = numHeads;
    qkTotal_ = qkHeadDim;
    vDim_ = vHeadDim;
    indexDim_ = indexHeadDim;
    maxSeqLen_ = maxSeqLen;
    seqLen_ = 0;

    headStride_ = size_t(maxSeqLen) * size_t(qkHeadDim);
    layerIndexStride_ = indexHeadDim;

    layerBuffers_.resize(numLayers);
    for (auto& buf : layerBuffers_) {
        buf.key.resize(size_t(numHeads) * size_t(maxSeqLen) * size_t(qkHeadDim));
        buf.value.resize(size_t(numHeads) * size_t(maxSeqLen) * size_t(vHeadDim));
        buf.indexKey.resize(size_t(maxSeqLen) * size_t(layerIndexStride_));
    }
}

void KVCache::reserve(int tokenCount) {
    if (tokenCount <= maxSeqLen_) return;
    growTo(tokenCount - 1);
}

void KVCache::growTo(int tokenPos) {
    int need = tokenPos + 1;
    if (need <= maxSeqLen_) return;
    // Double from the current capacity (amortized O(1) per token) with a
    // minimum step of 1024 tokens so a long decode does not repeatedly resize
    // (+ copy) multi-GB buffers during the first tokens of a run.
    int newCap = maxSeqLen_ == 0 ? 64 : maxSeqLen_;
    int floor = 1024;
    while (newCap < need) {
        newCap = std::max(newCap * 2, floor);
        floor *= 2;
    }
    // Head-major rebase: each head's block grows from oldStride to newStride.
    const size_t oldKeyHeadStride = headStride_;
    const size_t oldValueHeadStride = size_t(maxSeqLen_) * size_t(vDim_);
    const size_t newKeyHeadStride = size_t(newCap) * size_t(qkTotal_);
    const size_t newValueHeadStride = size_t(newCap) * size_t(vDim_);
    for (auto& buf : layerBuffers_) {
        std::vector<float> nk(size_t(numHeads_) * newKeyHeadStride, 0.0f);
        std::vector<float> nv(size_t(numHeads_) * newValueHeadStride, 0.0f);
        for (int h = 0; h < numHeads_; ++h) {
            std::memcpy(nk.data() + size_t(h) * newKeyHeadStride,
                        buf.key.data() + size_t(h) * oldKeyHeadStride,
                        oldKeyHeadStride * sizeof(float));
            std::memcpy(nv.data() + size_t(h) * newValueHeadStride,
                        buf.value.data() + size_t(h) * oldValueHeadStride,
                        oldValueHeadStride * sizeof(float));
        }
        buf.key.swap(nk);
        buf.value.swap(nv);
        buf.indexKey.resize(size_t(newCap) * size_t(layerIndexStride_));
    }
    headStride_ = newKeyHeadStride;
    maxSeqLen_ = newCap;
}

void KVCache::appendKey(int layer, int tokenPos, const float* key) {
    if (layer < 0 || layer >= numLayers_ || tokenPos < 0) return;
    growTo(tokenPos);
    auto& buf = layerBuffers_[layer];
    for (int h = 0; h < numHeads_; ++h) {
        std::memcpy(buf.key.data() + size_t(h) * headStride_ +
                        size_t(tokenPos) * size_t(qkTotal_),
                    key + size_t(h) * size_t(qkTotal_),
                    size_t(qkTotal_) * sizeof(float));
    }
}

void KVCache::appendValue(int layer, int tokenPos, const float* value) {
    if (layer < 0 || layer >= numLayers_ || tokenPos < 0) return;
    growTo(tokenPos);
    auto& buf = layerBuffers_[layer];
    for (int h = 0; h < numHeads_; ++h) {
        std::memcpy(buf.value.data() + size_t(h) * size_t(maxSeqLen_) * size_t(vDim_) +
                        size_t(tokenPos) * size_t(vDim_),
                    value + size_t(h) * size_t(vDim_),
                    size_t(vDim_) * sizeof(float));
    }
}

void KVCache::appendIndexKey(int layer, int tokenPos, const float* indexKey) {
    if (layer < 0 || layer >= numLayers_ || tokenPos < 0) return;
    growTo(tokenPos);
    auto& buf = layerBuffers_[layer];
    std::memcpy(buf.indexKey.data() + size_t(tokenPos) * size_t(layerIndexStride_),
                indexKey, size_t(layerIndexStride_) * sizeof(float));
}

const float* KVCache::getKey(int layer, int head, int tokenPos) const {
    if (layer < 0 || layer >= numLayers_ || head < 0 || head >= numHeads_ ||
        tokenPos < 0 || tokenPos >= maxSeqLen_) return nullptr;
    return layerBuffers_[layer].key.data() + size_t(head) * headStride_ +
           size_t(tokenPos) * size_t(qkTotal_);
}

const float* KVCache::getValue(int layer, int head, int tokenPos) const {
    if (layer < 0 || layer >= numLayers_ || head < 0 || head >= numHeads_ ||
        tokenPos < 0 || tokenPos >= maxSeqLen_) return nullptr;
    return layerBuffers_[layer].value.data() + size_t(head) * size_t(maxSeqLen_) * size_t(vDim_) +
           size_t(tokenPos) * size_t(vDim_);
}

const float* KVCache::getIndexKey(int layer, int tokenPos) const {
    if (layer < 0 || layer >= numLayers_ || tokenPos < 0 || tokenPos >= maxSeqLen_) return nullptr;
    return layerBuffers_[layer].indexKey.data() + size_t(tokenPos) * size_t(layerIndexStride_);
}

void KVCache::clear() {
    seqLen_ = 0;
}

} // namespace glm