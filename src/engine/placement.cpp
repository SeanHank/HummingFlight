#include "engine/placement.h"

namespace glm {

void PlacementDirector::recordAccess(int layer, int expertId) {
    uint64_t key = heatKey(layer, expertId);
    accessCounts_[key]++;
}

PlacementTier PlacementDirector::decideExpertPlacement(int layer, int expertId) const {
    int count = getAccessCount(layer, expertId);

    // Dynamic thresholds based on accumulated heat:
    // >100 accesses -> VRAM (extremely hot, only a few experts fit in 6GB)
    // >10 accesses  -> RAM (warm, LRU cache)
    // else          -> HDD (cold, stream on demand)
    if (count > 100) return PlacementTier::VRAM;
    if (count > 10)  return PlacementTier::RAM;
    return PlacementTier::HDD;
}

PlacementTier PlacementDirector::decideExpertPlacement(uint64_t expertKey, int accessCount) const {
    // Legacy overload
    if (accessCount > 100) return PlacementTier::VRAM;
    if (accessCount > 10)  return PlacementTier::RAM;
    return PlacementTier::HDD;
}

PlacementTier PlacementDirector::decideDensePlacement(const std::string& name, size_t sz) const {
    // Router/Gate -> VRAM (very small)
    if (name.find("gate") != std::string::npos) return PlacementTier::VRAM;
    // Embedding / LM Head -> VRAM
    if (name.find("embed_tokens") != std::string::npos || name.find("lm_head") != std::string::npos)
        return PlacementTier::VRAM;
    // Attention -> RAM resident
    if (name.find("self_attn") != std::string::npos) return PlacementTier::RAM;
    // Shared experts -> RAM (stream if total exceeds budget)
    if (name.find("shared_experts") != std::string::npos) return PlacementTier::RAM;
    return PlacementTier::RAM;
}

int PlacementDirector::getAccessCount(int layer, int expertId) const {
    uint64_t key = heatKey(layer, expertId);
    auto it = accessCounts_.find(key);
    return it != accessCounts_.end() ? it->second : 0;
}

void PlacementDirector::decayHeat(float factor) {
    for (auto& [key, count] : accessCounts_) {
        count = int(count * factor);
    }
}

void PlacementDirector::resetHeat() {
    accessCounts_.clear();
}

} // namespace glm
