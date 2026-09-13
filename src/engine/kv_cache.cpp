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

    layerKeyStride_ = numHeads * qkTotal_;
    layerValueStride_ = numHeads * vDim_;
    layerIndexStride_ = indexHeadDim;

    layerBuffers_.resize(numLayers);
    for (auto& buf : layerBuffers_) {
        buf.key.resize(size_t(maxSeqLen) * size_t(layerKeyStride_));
        buf.value.resize(size_t(maxSeqLen) * size_t(layerValueStride_));
        buf.indexKey.resize(size_t(maxSeqLen) * size_t(layerIndexStride_));
    }
}

void KVCache::growTo(int tokenPos) {
    int need = tokenPos + 1;
    if (need <= maxSeqLen_) return;
    int newCap = maxSeqLen_ == 0 ? 64 : maxSeqLen_;
    while (newCap < need) newCap *= 2;
    for (auto& buf : layerBuffers_) {
        buf.key.resize(size_t(newCap) * size_t(layerKeyStride_));
        buf.value.resize(size_t(newCap) * size_t(layerValueStride_));
        buf.indexKey.resize(size_t(newCap) * size_t(layerIndexStride_));
    }
    maxSeqLen_ = newCap;
}

void KVCache::appendKey(int layer, int tokenPos, const float* key) {
    if (layer < 0 || layer >= numLayers_ || tokenPos < 0) return;
    growTo(tokenPos);
    auto& buf = layerBuffers_[layer];
    std::memcpy(buf.key.data() + size_t(tokenPos) * size_t(layerKeyStride_),
                key, size_t(layerKeyStride_) * sizeof(float));
}

void KVCache::appendValue(int layer, int tokenPos, const float* value) {
    if (layer < 0 || layer >= numLayers_ || tokenPos < 0) return;
    growTo(tokenPos);
    auto& buf = layerBuffers_[layer];
    std::memcpy(buf.value.data() + size_t(tokenPos) * size_t(layerValueStride_),
                value, size_t(layerValueStride_) * sizeof(float));
}

void KVCache::appendIndexKey(int layer, int tokenPos, const float* indexKey) {
    if (layer < 0 || layer >= numLayers_ || tokenPos < 0) return;
    growTo(tokenPos);
    auto& buf = layerBuffers_[layer];
    std::memcpy(buf.indexKey.data() + size_t(tokenPos) * size_t(layerIndexStride_),
                indexKey, size_t(layerIndexStride_) * sizeof(float));
}

const float* KVCache::getKey(int layer, int tokenPos) const {
    if (layer < 0 || layer >= numLayers_ || tokenPos < 0 || tokenPos >= maxSeqLen_) return nullptr;
    return layerBuffers_[layer].key.data() + size_t(tokenPos) * size_t(layerKeyStride_);
}

const float* KVCache::getValue(int layer, int tokenPos) const {
    if (layer < 0 || layer >= numLayers_ || tokenPos < 0 || tokenPos >= maxSeqLen_) return nullptr;
    return layerBuffers_[layer].value.data() + size_t(tokenPos) * size_t(layerValueStride_);
}

const float* KVCache::getIndexKey(int layer, int tokenPos) const {
    if (layer < 0 || layer >= numLayers_ || tokenPos < 0 || tokenPos >= maxSeqLen_) return nullptr;
    return layerBuffers_[layer].indexKey.data() + size_t(tokenPos) * size_t(layerIndexStride_);
}

void KVCache::clear() {
    seqLen_ = 0;
}

} // namespace glm