#pragma once

#include "engine/placement.h"
#include "model/weight_index.h"
#include "storage/iocp_reader.h"
#include "storage/lru_cache.h"
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

namespace glm {

// Core scheduler (inspired by colibri's "JIT for weights")
// Responsibilities: one-layer-ahead prefetch + expert union batch fetch +
// prefetch/compute overlap + LRU eviction + learned pins (routing heat).
//
// Design (doc/design.md §7): every routed expert activated by the router is
// staged through the byte-budgeted ExpertLRU. Loads come either from the LRU
// (hits) or from batched asynchronous IOCP reads grouped by shard file so HDD
// seeks collapse (batched expert union). Shared experts are streamed through
// the same LRU using the pseudo-expert ids below and are prefetched one layer
// ahead together with the routed Top-K.

// Reserved expert-id slots for the shared experts of each layer (routed
// experts use ids in [0, numLocalExperts)).
enum : uint16_t {
    kSharedGateProjSlot = 0xFF00,  // shared_experts.gate_proj.weight
    kSharedUpProjSlot   = 0xFF01,  // shared_experts.up_proj.weight
    kSharedDownProjSlot = 0xFF02,  // shared_experts.down_proj.weight
};

class Scheduler {
public:
    Scheduler();

    // Initialize with weight index reference (needed for expert I/O).
    // lruCapacityBytes bounds the resident expert window; iocpWorkers scales
    // the async disk-read pool. When ramBudgetBytes/vramBudgetBytes are 0 the
    // legacy heuristic (LRU + 34 GiB RAM) is kept.
    bool init(WeightIndex& idx, size_t lruCapacityBytes, int iocpWorkers = 2,
              size_t ramBudgetBytes = 0, size_t vramBudgetBytes = 0);

    void shutdown();

    // Block until every submitted prefetch read completes (test / teardown helper).
    void waitForPrefetch() { iocp_.waitAll(); }

    // Called after layer L's router computation: record L's Top-K and prefetch
    // the predicted experts for layer L+1 (lookahead).
    void onLayerRouterDone(int layer, const std::vector<int>& topKExperts);

    // Load expert weights: LRU cache first, batched IOCP fallback, sync read
    // last. Returns true on success, writes to destBuffer.
    bool loadExpert(int layer, int expertId, void* destBuffer, size_t bufSize);

    // Pointer to expert weights staged in a scheduler-owned scratch buffer
    // (contents are a locked copy of the LRU entry, so async prefetch eviction
    // can never free memory while the compute path reads it). Layout:
    // [gate_proj | up_proj | down_proj], each BF16. Returns nullptr on error.
    const uint8_t* getExpertPtr(int layer, int expertId);

    // Shared-expert projection access (staged through the LRU as pseudo-experts).
    // Each slot copies into its own scratch, so all three pointers from a single
    // layer stay valid simultaneously.
    const uint8_t* getSharedProjectionPtr(int layer, uint16_t slot);

    // Expert LRU statistics.
    double lruHitRate() const;
    size_t lruUsedBytes() const;
    size_t lruCapacityBytes() const;

    ExpertLRU& lru() { return lru_; }

private:
    // Predict next layer's expert set (default: reuse previous token's Top-K at that layer).
    std::vector<int> predictNextLayerExperts(int nextLayer, const std::vector<int>& currentTopK);

    // Async prefetch expert set via IOCP (batched per shard, correct offsets).
    void prefetchAsync(const std::vector<std::pair<int, int>>& experts);

    // Synchronously load full expert [gate|up|down] from disk via mmap.
    bool loadExpertFromDisk(int layer, int expertId, BufferHandle& out);

    // True when key is resident in the LRU or already in an in-flight prefetch.
    bool probe(ExpertKey key);

    ExpertLRU lru_;
    IOCPReader iocp_;
    PlacementDirector placement_;
    WeightIndex* index_ = nullptr;

    // Previous token's Top-K per layer (prediction basis).
    std::vector<std::vector<int>> lastTopK_;

    // Compute-side staging buffers. getExpertPtr/getSharedProjectionPtr return
    // pointers into these; contents are copied from the LRU/disk under lock so
    // eviction by prefetch workers can never invalidate in-use memory.
    std::vector<uint8_t> exScratch_;        // one full [gate|up|down] expert
    std::vector<uint8_t> shScratch_[3];     // one per shared slot (gate/up/down)

    // Keys currently in flight (guarded; touched by IOCP workers and the caller).
    std::mutex inFlightMtx_;
    std::unordered_set<ExpertKey, ExpertKeyHash> prefetchedInFlight_;
};

} // namespace glm