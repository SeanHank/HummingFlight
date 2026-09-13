#pragma once

#include "storage/lru_cache.h"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace glm {

// Placement tiers (inspired by colibri: VRAM / RAM / HDD are different placement tiers for the same weight)
enum class PlacementTier {
    HDD = 0,    // Tier-2: full weights (HDD-backed checkpoint)
    RAM = 1,    // Tier-1: LRU cache + Dense resident
    VRAM = 2,   // Tier-0: hot subgraph (CUDA / MPS when enabled)
};

// Placement director: decides weight placement tier based on routing heat / capacity budget / prefetch timing
// Maintains per-expert access frequency for dynamic tier promotion/demotion
class PlacementDirector {
public:
    void setVramBudget(size_t bytes) { vramBudget_ = bytes; }
    void setRamBudget(size_t bytes) { ramBudget_ = bytes; }

    size_t vramBudget() const { return vramBudget_; }
    size_t ramBudget() const { return ramBudget_; }

    // Record an expert access (called on router hit)
    void recordAccess(int layer, int expertId);

    // Decide expert placement tier (based on accumulated heat)
    PlacementTier decideExpertPlacement(int layer, int expertId) const;

    // Legacy overload (backward compatible)
    PlacementTier decideExpertPlacement(uint64_t expertKey, int accessCount) const;

    // Decide dense weight placement tier
    PlacementTier decideDensePlacement(const std::string& tensorName, size_t byteSize) const;

    // Get access count for an expert
    int getAccessCount(int layer, int expertId) const;

    // Periodic decay of access counts (call every N tokens)
    void decayHeat(float factor = 0.9f);

    // Reset all heat data
    void resetHeat();

private:
    size_t vramBudget_ = 6ULL * 1024 * 1024 * 1024;  // 6GB
    size_t ramBudget_ = 64ULL * 1024 * 1024 * 1024;   // 64GB

    // Per-expert access count: key = layer * 65536 + expert_id
    std::unordered_map<uint64_t, int> accessCounts_;

    static uint64_t heatKey(int layer, int expertId) {
        return uint64_t(layer) * 65536 + uint64_t(expertId);
    }
};

} // namespace glm
