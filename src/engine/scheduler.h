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

    // Router-probability aware variant (new overload; the 2-arg overload
    // delegates with empty probabilities). `probs` holds the router weights of
    // the elected experts (same order/length as topKExperts). Enables #14:
    // speculative extra prefetch (`prefetchExtraDepth`), probability-weighted
    // residency pins and per-layer probability tracking.
    void onLayerRouterDone(int layer, const std::vector<int>& topKExperts,
                           const std::vector<float>& probs);

    // Configure the router-probability upgrades. prefetchExtraDepth > 0 enables
    // speculative extra experts per layer; probPriority promotes experts whose
    // router probability is ~max for their layer to residency pins.
    void setRouterPrefetch(int prefetchExtraDepth, bool probPriority);

    // Configure the EMA popularity predictor (#4): keeps an exponential moving
    // average of each expert's routing heat (sum of probabilities per routing
    // event, or 1.0 when probabilities are unavailable). With `enable` the
    // lookahead predictor falls back to the top-N hottest experts of the next
    // layer, and the hottest expert of each routed layer gets an LRU soft-boost
    // (#14). `alpha` in (0,1] is the EMA window; <= 0 means 0.1. Off by default
    // (goldens preserved).
    void setPopularityPredictor(float alpha, bool enable);

    // Load expert weights: LRU cache first, batched IOCP fallback, sync read
    // last. Returns true on success, writes to destBuffer.
    bool loadExpert(int layer, int expertId, void* destBuffer, size_t bufSize);

    // Pointer to expert weights staged in a scheduler-owned scratch buffer
    // (contents are a locked copy of the LRU entry, so async prefetch eviction
    // can never free memory while the compute path reads it). Layout:
    // [gate_proj | up_proj | down_proj], each BF16. Returns nullptr on error.
    const uint8_t* getExpertPtr(int layer, int expertId);

    // O12 (doc/design.md 17.11): non-blocking read-ahead for one expert. Issues
    // the same demand-stage prefetchAsync would issue on a getter miss, but
    // returns immediately, so the caller can overlap the disk read with its own
    // compute and let the later getExpertPtr() find the assembly in flight.
    // Purely a *when* change: identical bytes, identical order, no math touched.
    void prefetchExpert(int layer, int expertId);

    // Shared-expert projection access (staged through the LRU as pseudo-experts).
    // Each slot copies into its own scratch, so all three pointers from a single
    // layer stay valid simultaneously.
    const uint8_t* getSharedProjectionPtr(int layer, uint16_t slot);

    // Expert LRU statistics.
    double lruHitRate() const;
    size_t lruUsedBytes() const;
    size_t lruCapacityBytes() const;

    // O7: warm the always-read dense tensor ranges into the OS page cache with
    // sequential discard reads before the first forward (kills the cold page-fault
    // storm that dominated token-1). No-op when the index is unavailable.
    void warmDenseWeights();

    ExpertLRU& lru() { return lru_; }

private:
    // Predict next layer's expert set (default: reuse previous token's Top-K at that layer).
    std::vector<int> predictNextLayerExperts(int nextLayer, const std::vector<int>& currentTopK);

    // Async prefetch expert set via IOCP (batched per shard, correct offsets).
    // `demand = true` marks the current layer's routed winners (guaranteed to be
    // consumed immediately): it bypasses the O2 speculative cap so the layer's
    // loads overlap on the disk instead of degrading to serial mmap faults (O6).
    void prefetchAsync(const std::vector<std::pair<int, int>>& experts, bool demand = false);

    // Synchronously load full expert [gate|up|down] from disk via mmap.
    bool loadExpertFromDisk(int layer, int expertId, BufferHandle& out);

    // True when key is resident in the LRU or already in an in-flight prefetch.
    bool probe(ExpertKey key);

    // O13c: size of the in-flight expert-key set (sampled at each layer's first
    // getter wait to report the queue depth that pacing controls).
    int64_t inFlightKeysSnapshot() const;

    ExpertLRU lru_;
    IOCPReader iocp_;
    PlacementDirector placement_;
    WeightIndex* index_ = nullptr;

    // Previous token's Top-K per layer (prediction basis).
    std::vector<std::vector<int>> lastTopK_;
    std::vector<float> overlapEma_;

    // Router probabilities of the previous token per layer (same order).
    std::vector<std::vector<std::pair<int, float>>> lastTopKProbs_;

    // Router-probability upgrade knobs (#14/#13). Off by default so byte-for-byte
    // golden outputs are preserved.
    int prefetchExtraDepth_ = 0;
    bool probPriority_ = false;
    std::vector<float> layerMaxProb_;

    // EMA popularity predictor (#4, opt-in). heat(key) = alpha*p + (1-alpha)*heat.
    bool emaPredictor_ = false;
    float emaAlpha_ = 0.1f;
    std::unordered_map<ExpertKey, float, ExpertKeyHash> emaHeat_;

    // Step counter for expert reuse-distance telemetry (incremented once per
    // onLayerRouterDone call = one routing event).
    uint64_t routingStep_ = 0;
    // O13c (doc/design.md 17.12): routing step whose first getter wait has
    // already been timed, so only the layer's FIRST wait feeds
    // InferenceStats::firstExpertWaitNs (the pre-compute head-of-line cost).
    uint64_t demandFirstWaitStep_ = UINT64_MAX;
    std::unordered_map<ExpertKey, uint64_t, ExpertKeyHash> lastUseStep_;

    // O8c (doc/design.md 17.9): experts named by routed lookahead guesses that
    // survived the spec gate. Erased when a real demand lookup consumes them, so
    // the report can separate junk reads (never referenced) from delivered ones.
    // Accessed only from the generation thread (callers of onLayerRouterDone /
    // getExpertPtr / getSharedProjectionPtr), so no lock is needed.
    std::unordered_set<ExpertKey, ExpertKeyHash> routedPredKeys_;

    // Compute-side staging buffers. getExpertPtr/getSharedProjectionPtr return
    // pointers into these; contents are copied from the LRU/disk under lock so
    // eviction by prefetch workers can never invalidate in-use memory.
    std::vector<uint8_t> exScratch_;        // one full [gate|up|down] expert
    std::vector<uint8_t> shScratch_[3];     // one per shared slot (gate/up/down)

    // Keys currently in flight (guarded; touched by IOCP workers and the caller).
    mutable std::mutex inFlightMtx_;
    std::unordered_set<ExpertKey, ExpertKeyHash> prefetchedInFlight_;

    // Record a routing event for reuse-distance / LRU telemetry (expert consumed).
    void noteRoutingUse(int layer, int expertId);
};

} // namespace glm