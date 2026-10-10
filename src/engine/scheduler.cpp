#include "engine/scheduler.h"
#include "engine/inference_stats.h"
#include "utils/logger.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>

namespace glm {

namespace {

constexpr int kPinThreshold = 5;  // routing accesses before an LRU learn-in (learned pin)

// O2 (doc/design.md 17.2): speculative prefetch budget. Once this many expert
// assemblies are being streamed no further speculative I/O is issued, so the
// critical-path demand reads never wait behind an unbounded speculative queue
// (on HDD this bounded the measured 3718 ms queue-latency inflation and trims
// the ~70% prefetch waste). Soft cap; a stray concurrent caller may overshoot
// by one batch, which is harmless.
constexpr size_t kMaxPrefetchInFlight = 40;

// O3 (doc/design.md 17.2): maximum inter-projection gap (bytes) tolerated when
// coalescing same-shard reads into one run. Projections separated by <= this are
// merged into a single I/O request; the padding is read but never copied into an
// assembly, so merged runs stay byte-identical and only the seek count drops.
// O11 (doc/design.md 17.11) tried 8 MB here on the theory that fewer/longer runs
// would sweep the drive better; measured on the full 4-token L4 run it bought
// nothing (3928 runs vs 3880 at 1 MB, i.e. no change in the access pattern) and
// cost 3.62 GB of pure padding (+1.2 % bytes, +1.6 % wall), so the window stays
// at 1 MB. The real lever was elsewhere: the layer's read phase was already
// running at the drive's measured multi-file rate.
constexpr uint64_t kRunMergeGap = 1ULL * 1024 * 1024;

// O4 (doc/design.md 17.5): routine winner-overlap floor and EMA blend for the
// per-layer stability gate. Speculating the next layer's routed experts only
// pays when consecutive tokens re-route to shared winners (router is stable).
// On a cold start or a garbage stream the overlap EMA stays ~0 and the expert
// guesses would be pure waste, so O4 clears the predicted set (shared
// projections below are still streamed: they are guaranteed next-layer inputs).
// O8 (doc/design.md 17.9): the floor must sit clearly above the random-routing
// baseline (E[shared] = 8*(8/256) = 0.25 for 8-of-256). With the old 0.25 floor
// a cold/garbage layer's EMA converged to ~0.25, the gate never fired and every
// layer staged ~8 never-consumed routed guesses every token (~55 GB/token in the
// benchmark worst case). 0.5 = 2x the chance baseline: only sustained >=4/8
// winner reuse re-opens routed lookahead for a layer.
constexpr float kSpecOverlapFloor = 0.5f;     // requires sustained, non-chance reuse
constexpr float kSpecOverlapAlpha = 0.5f;     // EMA blend per routing event

// O6 (doc/design.md 17.7): how long a demand getter waits for its (just-scheduled,
// guaranteed) current-layer prefetch assembly to land in the LRU before falling
// back to the synchronous mmap copy. Bounded so a failed I/O never deadlocks.
constexpr int kDemandWaitTimeoutMs = 120000;

// Round 6 note (doc/design.md 17.12): the demand stream was deliberately NOT paced.
// readprobe6's consumer model put K=2-3 reads in flight at a stable 6.7-7.7 s per layer
// against 7.4-11.4 s for the whole-layer submit, so onLayerRouterDone was changed to fire
// only a 2-expert primer and let the forward's look-ahead pull the rest. In the engine that
// did what the model said (reads_in_flight_at_first_wait 8.67 -> 3.98, first_expert_wait_ms
// 3791 -> 1528, i.e. 60 % less blocking before the first expert can compute) and changed
// nothing else (byte-identical 306897223680 B, waste 0, sync 0) -- but the total demand wait
// did not fall (3007 s -> 3061 s) and neither did wall clock, because the two measured runs
// turned out to sit inside this host's +/-4.5 % run-to-run noise (6684.18 s for R5, 6781.38 s
// paced, 6984.42 s for a REVERT to identical R5 behaviour). The layer is bandwidth-bound,
// not schedule-bound: its demand bytes exceed what the drive can deliver inside the compute
// window, so there is no scheduling win left to take here. Reverted; the O13c instrumentation
// is kept because it is the only way to see inside that noise floor.

// O7 (doc/design.md 17.7): dense-prewarm read chunk and in-flight buffer pool.
// 16 x 8 MiB keep the async discard reads deep without huge allocations.
constexpr size_t kWarmChunk = 8ULL * 1024 * 1024;
constexpr int kWarmPool = 16;

TensorLocation sharedSlotLoc(const LayerWeights& lw, uint16_t slot) {
    switch (slot) {
        case kSharedGateProjSlot: return lw.sharedGateProj;
        case kSharedUpProjSlot:   return lw.sharedUpProj;
        default:                  return lw.sharedDownProj;
    }
}

// Per-expert assembly buffer, assembled from possibly several shard runs.
// Lives until the prefetch callback completes (owned by the captured copies).
class Assembly {
public:
    ExpertKey key;
    size_t total = 0;
    std::shared_ptr<BufferHandle> handle;
    std::atomic<int> remaining{0};
    std::atomic<bool> inserted{false};
};

// A projection slice that is part of a large sequential run.
struct Slice {
    std::shared_ptr<Assembly> assem;  // owning assembly
    size_t expertOffset;              // byte offset inside [gate|up|down]
    size_t size;
    uint64_t offsetInRun;
};

// A contiguous merged read over one shard.
struct Run {
    std::string shard;
    uint64_t start;
    uint64_t end;
    std::vector<Slice> slices;
};

// (shard) -> flat list of projections for that shard.
struct ProjectionIO {
    std::shared_ptr<Assembly> assem;
    uint16_t slot;
    uint64_t offset;
    uint64_t size;
    size_t expertOffset;
};

} // namespace

Scheduler::Scheduler() : lru_(20ULL * 1024 * 1024 * 1024) {}

bool Scheduler::init(WeightIndex& idx, size_t lruCapacityBytes, int iocpWorkers,
                     size_t ramBudgetBytes, size_t vramBudgetBytes) {
    index_ = &idx;
    lru_.resize(lruCapacityBytes);
    if (ramBudgetBytes == 0) ramBudgetBytes = lruCapacityBytes + 34ULL * 1024 * 1024 * 1024;
    placement_.setRamBudget(ramBudgetBytes);
    if (vramBudgetBytes > 0) placement_.setVramBudget(vramBudgetBytes);
    if (!iocp_.start(iocpWorkers)) {
        GLM_LOG_WARN("IOCP start failed, falling back to synchronous reads");
    }

    int numLayers = idx.config.numLayers;
    lastTopK_.resize(numLayers);
    overlapEma_.resize(numLayers, 0.0f);
    lastTopKProbs_.resize(numLayers);
    layerMaxProb_.assign(numLayers, 0.0f);

    GLM_LOG_INFO("Scheduler ready: LRU=" + std::to_string(lruCapacityBytes / (1024 * 1024)) +
                 " MB, IOCP workers=" + std::to_string(iocpWorkers));
    return true;
}

void Scheduler::setRouterPrefetch(int prefetchExtraDepth, bool probPriority) {
    prefetchExtraDepth_ = std::max(0, prefetchExtraDepth);
    probPriority_ = probPriority;
    if (prefetchExtraDepth_ > 0 || probPriority_) {
        GLM_LOG_INFO("Scheduler router-probability mode: extra-prefetch=" +
                     std::to_string(prefetchExtraDepth_) + ", prob-priority-residency=" +
                     std::to_string(probPriority_ ? 1 : 0));
    }
}

void Scheduler::setPopularityPredictor(float alpha, bool enable) {
    emaPredictor_ = enable;
    emaAlpha_ = (alpha > 0.0f && alpha <= 1.0f) ? alpha : 0.1f;
    if (emaPredictor_) {
        GLM_LOG_INFO("Scheduler EMA popularity predictor: alpha=" +
                     std::to_string(emaAlpha_) + ", LRU soft-boost for hottest experts");
    }
}

void Scheduler::shutdown() {
    iocp_.waitAll();
    iocp_.stop();
    GLM_LOG_INFO("Scheduler shutdown: LRU " + std::to_string(lru_.usedBytes() / (1024 * 1024)) +
                 " MB used, hit rate " + std::to_string(int(lru_.hitRate() * 100)) + "%");
}

bool Scheduler::probe(ExpertKey key) {
    if (lru_.contains(key)) return true;
    std::lock_guard<std::mutex> guard(inFlightMtx_);
    return prefetchedInFlight_.count(key) != 0;
}

std::vector<int> Scheduler::predictNextLayerExperts(int nextLayer, const std::vector<int>& currentTopK) {
    // Base predictor (kept unchanged when the EMA upgrade is off): the previous
    // token's routing at the same layer, else a weak inter-layer carry-over.
    std::vector<int> reuse = currentTopK;
    bool haveReuse = false;
    if (nextLayer >= 0 && nextLayer < int(lastTopK_.size()) && !lastTopK_[nextLayer].empty()) {
        reuse = lastTopK_[nextLayer];
        haveReuse = true;
    }

    if (!emaPredictor_) return reuse;

    // EMA popularity blend (#4): start from the reuse set, then fill any
    // remaining slots with the hottest experts of the next layer by their
    // EM-averaged routing heat.
    int numNeed = int(reuse.size());
    std::unordered_set<int> seen(reuse.begin(), reuse.end());
    std::vector<int> hot;
    {
        std::vector<std::pair<float, int>> candidates;
        for (const auto& [key, heat] : emaHeat_) {
            if (key.layer != uint32_t(nextLayer)) continue;
            if (key.expert_id >= index_->config.numLocalExperts) continue;
            if (seen.count(key.expert_id)) continue;
            if (heat > 0.0f) candidates.emplace_back(heat, key.expert_id);
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        for (const auto& [heat, eid] : candidates) {
            (void)heat;
            if (int(hot.size()) + int(reuse.size()) >= numNeed) break;
            seen.insert(eid);
            hot.push_back(eid);
        }
    }
    if (hot.empty() && !haveReuse) return reuse;
    std::vector<int> merged = reuse;
    merged.insert(merged.end(), hot.begin(), hot.end());
    return merged;
}

void Scheduler::noteRoutingUse(int layer, int expertId) {
    ExpertKey key{uint32_t(layer), uint16_t(expertId)};
    auto& stats = InferenceStats::instance();
    auto it = lastUseStep_.find(key);
    if (it != lastUseStep_.end()) {
        if (routingStep_ > it->second) {
            stats.reuseDistanceSum += routingStep_ - it->second;
            ++stats.reuseDistanceCount;
        }
    }
    lastUseStep_[key] = routingStep_;
}

void Scheduler::onLayerRouterDone(int layer, const std::vector<int>& topKExperts) {
    onLayerRouterDone(layer, topKExperts, {});
}

void Scheduler::onLayerRouterDone(int layer, const std::vector<int>& topKExperts,
                                  const std::vector<float>& probs) {
    if (!index_) return;

    routingStep_++;
    if (layer >= 0 && layer < int(lastTopK_.size())) {
        // O4: roll the consecutive-token winner overlap EMA before the previous
        // winners of this layer are overwritten. A low EMA means the layer's
        // routing is unstable, so its next-token winners cannot be predicted.
        if (layer < int(overlapEma_.size()) && !lastTopK_[layer].empty() && !topKExperts.empty()) {
            int shared = 0;
            for (int e : topKExperts) {
                if (std::find(lastTopK_[layer].begin(), lastTopK_[layer].end(), e) !=
                    lastTopK_[layer].end())
                    ++shared;
            }
            float cur = float(shared) /
                        float(std::max(lastTopK_[layer].size(), topKExperts.size()));
            overlapEma_[layer] =
                kSpecOverlapAlpha * cur + (1.0f - kSpecOverlapAlpha) * overlapEma_[layer];
        }
        lastTopK_[layer] = topKExperts;
    }

    // Record per-layer probability state (item 14). Weights come in the same
    // order as topKExperts or empty (legacy 2-arg call).
    std::vector<std::pair<int, float>> probPairs;
    probPairs.reserve(topKExperts.size());
    for (size_t i = 0; i < topKExperts.size(); ++i) {
        float p = (i < probs.size()) ? probs[i] : 0.0f;
        probPairs.emplace_back(topKExperts[i], p);
        if (probPriority_ && layer >= 0 && layer < int(layerMaxProb_.size())) {
            layerMaxProb_[layer] = std::max(layerMaxProb_[layer], p);
        }
        noteRoutingUse(layer, topKExperts[i]);
    }
    if (layer >= 0 && layer < int(lastTopKProbs_.size())) lastTopKProbs_[layer] = std::move(probPairs);

    // EMA popularity roll (#4): heat = alpha*signal + (1-alpha)*heat, where the
    // signal is the router probability or 1.0 for occurrence-only calls. The
    // hottest expert of the routed layer earns an LRU soft-boost (#14).
    if (emaPredictor_ && layer >= 0) {
        float a = (emaAlpha_ > 0.0f && emaAlpha_ <= 1.0f) ? emaAlpha_ : 0.1f;
        ExpertKey bestKey{uint32_t(layer), uint16_t(0)};
        float bestHeat = -1.0f;
        for (const auto& eid : topKExperts) {
            if (eid < 0 || eid >= index_->config.numLocalExperts) continue;
            ExpertKey k{uint32_t(layer), uint16_t(eid)};
            float& h = emaHeat_[k];
            h = a * 1.0f + (1.0f - a) * h;  // occurrence signal; prob-weighting kept per-layer below
            if (h > bestHeat) { bestHeat = h; bestKey = k; }
        }
        if (bestHeat > 0.0f) lru_.boost(bestKey, 1);
    }

    int nextLayer = layer + 1;
    if (nextLayer < 0 || nextLayer >= index_->config.numLayers) return;

    std::vector<int> predicted = predictNextLayerExperts(nextLayer, topKExperts);

    // O4 (doc/design.md 17.5): only keep the routed-expert guesses when the next
    // layer has demonstrably re-used winners across recent tokens. Else the
    // prediction is cold carry-over noise; the shared projections below are
    // still prefetched (they are guaranteed inputs of the next forward, so those
    // runs are never waste).
    if (nextLayer >= 0 && nextLayer < int(overlapEma_.size()) &&
        overlapEma_[nextLayer] < kSpecOverlapFloor) {
        predicted.clear();
    }

    // O8c (doc/design.md 17.9): count the routed lookahead guesses that survive
    // the gate and remember which experts they named, so the report can show
    // exactly how much disk traffic is speculative-but-unreferenced junk. The
    // set is erased by a real demand lookup that consumes a staged guess.
    if (!predicted.empty()) {
        auto& rstats = InferenceStats::instance();
        rstats.routedPredStaged += predicted.size();
        for (int eid : predicted) {
            if (eid >= 0 && eid < index_->config.numLocalExperts)
                routedPredKeys_.insert(ExpertKey{uint32_t(nextLayer), uint16_t(eid)});
        }
    }

    // Speculative extra prefetch (#14): pull the next-most-probable experts of
    // the CURRENT layer (this token, d == routed winners) unless the estimate
    // for the next layer already covers them. Off by default (prefetchExtraDepth_).
    if (prefetchExtraDepth_ > 0 && layer >= 0 && layer < int(lastTopKProbs_.size())) {
        std::unordered_set<int> seen(predicted.begin(), predicted.end());
        std::vector<std::pair<float, int>> candidates;
        for (const auto& [eid, p] : lastTopKProbs_[layer]) {
            if (p > 0.0f && seen.count(eid) == 0) candidates.emplace_back(p, eid);
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        int added = 0;
        for (const auto& [p, eid] : candidates) {
            if (added >= prefetchExtraDepth_) break;
            predicted.push_back(eid);
            seen.insert(eid);
            ++added;
        }
    }

    // Probability-weighted residency (#13): experts whose router probability is
    // within 10% of their layer's maximum stay resident once loaded (learned pins
    // fire earlier: access threshold 2 instead of kPinThreshold).
    if (probPriority_ && layer >= 0 && layer < int(layerMaxProb_.size()) &&
        layerMaxProb_[layer] > 0.0f) {
        for (const auto& [eid, p] : lastTopKProbs_[layer]) {
            if (p >= 0.9f * layerMaxProb_[layer] &&
                placement_.getAccessCount(layer, eid) >= 2) {
                lru_.pin(ExpertKey{uint32_t(layer), uint16_t(eid)});
            }
        }
    }

    std::vector<std::pair<int, int>> scope;
    for (int eid : predicted) {
        if (eid >= 0 && eid < index_->config.numLocalExperts) {
            scope.emplace_back(nextLayer, eid);
        }
    }
    // Shared experts of the next layer are streamed too, so prefetch them ahead.
    scope.emplace_back(nextLayer, int(kSharedGateProjSlot));
    scope.emplace_back(nextLayer, int(kSharedUpProjSlot));
    scope.emplace_back(nextLayer, int(kSharedDownProjSlot));

    // O6 (doc/design.md 17.7): stage the CURRENT layer's routed winners + shared
    // projections through the async pipe with demand priority so the per-layer
    // loads overlap on the disk instead of serial mmap page-fault reads. probe()
    // dedups against what the previous layer already staged these keys with.
    //
    // Round 6 (doc/design.md 17.12) TRIED firing only a 2-expert primer here and
    // pulling the rest from the forward loop's look-ahead, to break the
    // submit/consume coupling. It worked mechanically and won nothing: see the
    // note above. Left as the whole-layer submit.
    if (layer >= 0) {
        std::vector<std::pair<int, int>> demand;
        demand.reserve(topKExperts.size() + 3);
        for (int eid : topKExperts) {
            if (eid >= 0 && eid < index_->config.numLocalExperts)
                demand.emplace_back(layer, eid);
        }
        demand.emplace_back(layer, int(kSharedGateProjSlot));
        demand.emplace_back(layer, int(kSharedUpProjSlot));
        demand.emplace_back(layer, int(kSharedDownProjSlot));
        prefetchAsync(demand, /*demand=*/true);
    }

    if (!scope.empty()) prefetchAsync(scope);
}

void Scheduler::prefetchAsync(const std::vector<std::pair<int, int>>& experts, bool demand) {
    if (!index_) return;

    // O2 backpressure: skip SPECULATION while the prefetch pipeline is already
    // saturated. Demand staging (O6, demand=true) bypasses the cap -- those runs
    // are the layer's current service set and the forward is waiting on them.
    if (!demand) {
        std::lock_guard<std::mutex> guard(inFlightMtx_);
        if (prefetchedInFlight_.size() >= kMaxPrefetchInFlight) return;
    }

    std::unordered_map<std::string, std::vector<ProjectionIO>> byShard;
    std::unordered_map<ExpertKey, std::shared_ptr<Assembly>, ExpertKeyHash> assemblies;

    for (const auto& [layer, eidRaw] : experts) {
        if (layer < 0 || layer >= index_->config.numLayers) continue;
        uint16_t eid = uint16_t(eidRaw);

        ExpertKey key{uint32_t(layer), eid};
        if (probe(key)) continue;

        // O10 (doc/design.md 17.11): a key that is already resident needs no
        // staging at all. Without this check every forward re-reads the pinned
        // shared projections (3 per layer, ~26 MB each) from disk although the
        // bytes are in RAM -- ~6 GB/forward of pure redundant traffic in the
        // round-4 run. The getter will hit the LRU; no assembled byte changes.
        if (lru_.contains(key)) {
            ++InferenceStats::instance().stageSkipLru;
            continue;
        }

        const LayerWeights& lw = index_->layers[layer];

        struct Proj { const TensorLocation* loc; };
        std::vector<Proj> projs;
        if (eid >= kSharedGateProjSlot) {
            if (eid != kSharedGateProjSlot && eid != kSharedUpProjSlot && eid != kSharedDownProjSlot) continue;
            const TensorLocation* sloc[1] = { nullptr };
            switch (eid) {
                case kSharedGateProjSlot: sloc[0] = &lw.sharedGateProj; break;
                case kSharedUpProjSlot:   sloc[0] = &lw.sharedUpProj;   break;
                default:                  sloc[0] = &lw.sharedDownProj; break;
            }
            projs.push_back({sloc[0]});
        } else {
            if (eid >= index_->config.numLocalExperts) continue;
            const auto& ew = index_->routedExperts[layer][eid];
            projs.push_back({&ew.gateProj});
            projs.push_back({&ew.upProj});
            projs.push_back({&ew.downProj});
        }

        // Compute total and per-projection expert offsets. O9b (doc/design.md
        // 17.11): a same-shard requirement used to send multi-shard experts down
        // the synchronous fallback. The run builder below already groups
        // projections by shardPath (one run per shard, slices carry per-projection
        // offsets), so a multi-shard expert assembles correctly through the queue;
        // the gate only cost blocking forward-thread copies -- 40 of them and
        // 94 s of stall in the measured round-5 run. Only a genuinely empty
        // expert is declined now.
        size_t total = 0;
        for (const auto& p : projs) total += p.loc->byteSize;
        if (total == 0) continue;  // nothing to read for this key

        if (demand) ++InferenceStats::instance().demandKeysStaged;  // O8c

        auto assem = std::make_shared<Assembly>();
        assem->key = key;
        assem->total = total;
        assem->handle = std::make_shared<BufferHandle>(total);
        assem->remaining.store(int(projs.size()));
        assemblies[key] = assem;

        size_t expertOffset = 0;
        for (const auto& p : projs) {
            if (p.loc->byteSize == 0) {
                --assem->remaining;  // nothing to load for this slot
                expertOffset += 0;
                continue;
            }
            byShard[p.loc->shardPath].push_back({assem, eid, p.loc->byteOffset, p.loc->byteSize, expertOffset});
            expertOffset += p.loc->byteSize;
        }
    }

    // Mark keys in flight before submitting (probe guard under the same mutex).
    {
        std::lock_guard<std::mutex> guard(inFlightMtx_);
        for (const auto& [key, assem] : assemblies) prefetchedInFlight_.insert(key);
    }

    // Merge projections per shard into sequential runs (overlap/adjacency merge).
    std::vector<std::shared_ptr<Run>> runs;
    for (auto& [shard, ios] : byShard) {
        std::sort(ios.begin(), ios.end(), [](const auto& a, const auto& b) {
            return a.offset < b.offset;
        });
        std::shared_ptr<Run> run = std::make_shared<Run>();
        run->shard = shard;
        for (const auto& io : ios) {
            // O3: coalesce any projection whose gap to the current run is within
            // kRunMergeGap; the padding bytes are read but never memcpy'd into an
            // assembly, so expert bytes and forward math stay bit-identical.
            if (run->slices.empty() || io.offset > run->end + kRunMergeGap) {
                // New run starts.
                run = std::make_shared<Run>();
                run->shard = shard;
                run->start = io.offset;
                run->end = io.offset + io.size;
                run->slices.push_back({io.assem, io.expertOffset, size_t(io.size), 0});
                runs.push_back(run);
            } else {
                // Extend or reuse the current run.
                if (io.offset + io.size > run->end) run->end = io.offset + io.size;
                run->slices.push_back({io.assem, io.expertOffset, size_t(io.size),
                                       io.offset - run->start});
            }
        }
    }

    // O8b (doc/design.md 17.9): submit runs in (shard, offset) order so each
    // batch reaches the drive as a directional sweep. The unordered_map iteration
    // order below would trampoline between shard files and hand the drive's
    // reordering window multi-directional seeks. Slices keep per-projection
    // offsets, so assembly bytes and forward math are unchanged.
    std::stable_sort(runs.begin(), runs.end(), [](const auto& a, const auto& b) {
        if (a->shard != b->shard) return a->shard < b->shard;
        return a->start < b->start;
    });

    // Wasted-decrement compensation: projections whose size was zero already
    // decremented `remaining`; runs only carry positive-size projections.

    for (auto& run : runs) {
        uint64_t runSize = run->end - run->start;
        if (runSize == 0 || run->slices.empty()) continue;

        auto buf = std::make_shared<std::vector<uint8_t>>(size_t(runSize));
        auto slices = run->slices;
        std::string shard = run->shard;
        uint64_t start = run->start;
        const auto submitTime = std::chrono::steady_clock::now();

        auto& stats = InferenceStats::instance();
        ++stats.prefetchSubmitted;
        ++stats.ioReadCount;
        ++stats.randomIoCount;
        ++stats.runsInFlight;  // O13c: drive concurrency sample
        stats.diskBytesRead += runSize;
        stats.sequentialBytes += runSize;

        ReadRequest req;
        req.filePath = shard;
        req.offset = start;
        req.size = runSize;
        req.destBuffer = buf->data();

        iocp_.submitRead(req, [this, buf, slices, shard, start, submitTime](bool success, size_t bytesRead) {
            (void)shard;
            auto& stats = InferenceStats::instance();
            --stats.runsInFlight;  // O13c
            stats.ioLatencyNsSum += uint64_t(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - submitTime).count());
            if (!success) return;
            for (const auto& s : slices) {
                if (bytesRead >= s.offsetInRun + s.size) {
                    std::memcpy(static_cast<uint8_t*>(s.assem->handle->ptr()) + s.expertOffset,
                                buf->data() + s.offsetInRun, s.size);
                }
                if (--s.assem->remaining == 0) {
                    // Only one thread observes the zero transition.
                    if (!s.assem->inserted.exchange(true)) {
                        std::lock_guard<std::mutex> guard(inFlightMtx_);
                        BufferHandle h(std::move(*s.assem->handle));
                        if (lru_.insert(s.assem->key, std::move(h), /*fromPrefetch=*/true)) {
                            ++stats.prefetchInserted;
                        }
                        prefetchedInFlight_.erase(s.assem->key);
                    }
                }
            }
        });
    }

    GLM_LOG_DEBUG("Prefetch submitted: " + std::to_string(assemblies.size()) +
                  " experts across " + std::to_string(runs.size()) + " shard runs");
}

bool Scheduler::loadExpertFromDisk(int layer, int expertId, BufferHandle& out) {
    if (!index_) return false;
    if (layer < 0 || layer >= index_->config.numLayers) return false;
    if (expertId < 0 || expertId >= index_->config.numLocalExperts) return false;

    const auto& ew = index_->routedExperts[layer][expertId];
    uint64_t totalBytes = ew.totalBytes();
    if (totalBytes == 0) return false;

    out = BufferHandle(totalBytes);
    uint8_t* dest = static_cast<uint8_t*>(out.ptr());

    const TensorLocation* locs[3] = { &ew.gateProj, &ew.upProj, &ew.downProj };
    for (const TensorLocation* loc : locs) {
        if (loc->byteSize > 0) {
            const uint8_t* view = index_->getShardView(loc->shardPath);
            if (!view) return false;
            std::memcpy(dest, view + loc->byteOffset, loc->byteSize);
            dest += loc->byteSize;
        }
    }
    return true;
}

bool Scheduler::loadExpert(int layer, int expertId, void* destBuffer, size_t bufSize) {
    ExpertKey key{uint32_t(layer), uint16_t(expertId)};

    if (auto* cached = lru_.acquire(key)) {
        if (cached->size <= bufSize) {
            std::memcpy(destBuffer, cached->ptr(), cached->size);
            placement_.recordAccess(layer, expertId);
            return true;
        }
        return false;
    }
    placement_.recordAccess(layer, expertId);

    BufferHandle loaded;
    if (!loadExpertFromDisk(layer, expertId, loaded)) return false;

    if (loaded.size > bufSize) return false;
    std::memcpy(destBuffer, loaded.ptr(), loaded.size);

    if (lru_.insert(key, std::move(loaded))) {
        if (placement_.getAccessCount(layer, expertId) >= kPinThreshold) lru_.pin(key);
    }
    return true;
}

int64_t Scheduler::inFlightKeysSnapshot() const {
    std::lock_guard<std::mutex> guard(inFlightMtx_);
    return int64_t(prefetchedInFlight_.size());
}

void Scheduler::prefetchExpert(int layer, int expertId) {
    if (layer < 0 || layer >= index_->config.numLayers) return;
    if (expertId < 0 || expertId >= index_->config.numLocalExperts) return;
    ExpertKey key{uint32_t(layer), uint32_t(expertId)};
    {
        std::lock_guard<std::mutex> guard(inFlightMtx_);
        if (prefetchedInFlight_.count(key)) return;  // already loading
    }
    if (lru_.contains(key)) return;                 // already resident
    ++InferenceStats::instance().loopPrefetchIssued;  // O13c: look-ahead submission
    prefetchAsync({{layer, expertId}}, /*demand=*/true);
}

const uint8_t* Scheduler::getExpertPtr(int layer, int expertId) {
    if (!index_) return nullptr;
    if (layer < 0 || layer >= index_->config.numLayers) return nullptr;
    if (expertId < 0 || expertId >= index_->config.numLocalExperts) return nullptr;

    ExpertKey key{uint32_t(layer), uint16_t(expertId)};
    placement_.recordAccess(layer, expertId);

    const uint64_t bytes = index_->routedExperts[layer][expertId].totalBytes();
    if (bytes == 0) return nullptr;
    if (exScratch_.size() < bytes) exScratch_.resize(size_t(bytes));

    auto& stats = InferenceStats::instance();
    ++stats.expertLookups;
    if (lru_.copyTo(key, exScratch_.data(), exScratch_.size())) {
        ++stats.expertLruHits;
        if (routedPredKeys_.erase(key)) ++stats.routedPredConsumed;  // O8c
        if (placement_.getAccessCount(layer, expertId) >= kPinThreshold) lru_.pin(key);
        return exScratch_.data();
    }

    ++stats.expertLruMisses;

    // O6 (doc/design.md 17.7): this winner was just staged by the current layer's
    // demand prefetch. Wait (bounded) for the async assembly to land in the LRU
    // instead of immediately doing a synchronous mmap copy -- all k experts of a
    // layer load concurrently on the IOCP workers, so one layer costs ~one read
    // latency instead of k serial page-fault reads.
    //
    // O9 (doc/design.md 17.11): when NO assembly is in flight (neither staged by
    // the layer's demand set nor a routed guess) the old code fell straight to
    // loadExpertFromDisk, i.e. a blocking 72 MB mmap page-fault walk on this
    // forward thread at the disk probe's measured ~90 MB/s -- with its bytes
    // invisible to diskBytesRead. Round 4 took that path 1252 times (21 % of all
    // lookups, ~90 GB). Stage the key on the demand path instead and wait for
    // the merged, offset-ordered read; the synchronous copy stays only as the
    // last-resort fallback when staging is declined (multi-shard expert).
    if (!probe(key)) {
        prefetchAsync({{layer, expertId}}, /*demand=*/true);
    }
    if (probe(key)) {
        ++stats.expertDiskLoads;
        // O13c: the FIRST getter wait of each layer is the time the forward
        // cannot start any expert FFN yet -- the number pacing must shrink.
        const bool firstOfLayer = (demandFirstWaitStep_ != routingStep_);
        if (firstOfLayer) {
            demandFirstWaitStep_ = routingStep_;
            stats.firstWaitRunsInFlight += uint64_t(std::max<int64_t>(0, stats.runsInFlight.load()));
            stats.firstWaitKeysInFlight += uint64_t(std::max<int64_t>(0, inFlightKeysSnapshot()));
        }
        const auto waitStart = std::chrono::steady_clock::now();
        const auto deadline = waitStart + std::chrono::milliseconds(kDemandWaitTimeoutMs);
        while (std::chrono::steady_clock::now() < deadline) {
            if (lru_.copyTo(key, exScratch_.data(), exScratch_.size())) {
                stats.demandWaitNs += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                  std::chrono::steady_clock::now() - waitStart).count());
                if (firstOfLayer) {
                    stats.firstExpertWaitNs += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                             std::chrono::steady_clock::now() - waitStart).count());
                    ++stats.layerFirstWaitCount;
                }
                if (routedPredKeys_.erase(key)) ++stats.routedPredConsumed;  // O8c
                if (placement_.getAccessCount(layer, expertId) >= kPinThreshold) lru_.pin(key);
                return exScratch_.data();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        stats.demandWaitNs += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::steady_clock::now() - waitStart).count());
        if (firstOfLayer) {
            stats.firstExpertWaitNs += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                     std::chrono::steady_clock::now() - waitStart).count());
            ++stats.layerFirstWaitCount;
        }
        ++stats.demandWaitTimeout;  // O8c: failed to land in time; sync re-read below
        ++stats.expertDiskLoads;  // timed out; the sync path below re-reads it
    }

    // Synchronous fallback: only reached when the async path could not serve the
    // key (staging declined, or the O6 wait timed out).
    ++stats.expertDiskLoads;
    ++stats.syncLoads;  // O9c
    stats.syncBytes += bytes;
    const auto syncStart = std::chrono::steady_clock::now();
    BufferHandle loaded;
    if (!loadExpertFromDisk(layer, expertId, loaded)) return nullptr;
    stats.syncStallNs += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::steady_clock::now() - syncStart).count());
    if (loaded.size != bytes) return nullptr;
    std::memcpy(exScratch_.data(), loaded.ptr(), loaded.size);

    if (lru_.insert(key, std::move(loaded), /*fromPrefetch=*/false)) {
        if (placement_.getAccessCount(layer, expertId) >= kPinThreshold) lru_.pin(key);
    }
    return exScratch_.data();
}

const uint8_t* Scheduler::getSharedProjectionPtr(int layer, uint16_t slot) {
    if (!index_) return nullptr;
    if (layer < 0 || layer >= index_->config.numLayers) return nullptr;

    const TensorLocation loc = sharedSlotLoc(index_->layers[layer], slot);
    if (loc.byteSize == 0) return nullptr;

    std::vector<uint8_t>& scratch = shScratch_[slot - kSharedGateProjSlot];
    if (scratch.size() < loc.byteSize) scratch.resize(size_t(loc.byteSize));

    ExpertKey key{uint32_t(layer), slot};
    placement_.recordAccess(layer, int(slot));

    if (lru_.copyTo(key, scratch.data(), scratch.size())) {
        return scratch.data();
    }

    // O6 (doc/design.md 17.7): shared projections are staged for the current
    // layer too; wait (bounded) for the async assembly instead of a sync copy.
    // O9 (doc/design.md 17.11): stage it on the demand path when nothing is in
    // flight, so this path also stops blocking the forward thread on a
    // page-faulting mmap copy whose bytes never reached diskBytesRead.
    if (!probe(key)) {
        prefetchAsync({{layer, int(slot)}}, /*demand=*/true);
    }
    if (probe(key)) {
        const auto waitStart = std::chrono::steady_clock::now();
        const auto deadline = waitStart + std::chrono::milliseconds(kDemandWaitTimeoutMs);
        while (std::chrono::steady_clock::now() < deadline) {
            if (lru_.copyTo(key, scratch.data(), scratch.size())) {
                InferenceStats::instance().demandWaitNs += uint64_t(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - waitStart).count());
                return scratch.data();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        InferenceStats::instance().demandWaitNs += uint64_t(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - waitStart).count());
        ++InferenceStats::instance().demandWaitTimeout;  // O8c
    }

    auto& syncStats = InferenceStats::instance();
    ++syncStats.syncLoads;  // O9c
    syncStats.syncBytes += loc.byteSize;
    const auto syncStart = std::chrono::steady_clock::now();
    const uint8_t* view = index_->getShardView(loc.shardPath);
    if (!view) return nullptr;
    std::memcpy(scratch.data(), view + loc.byteOffset, loc.byteSize);
    syncStats.syncStallNs += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now() - syncStart).count());

    BufferHandle loaded(loc.byteSize);
    std::memcpy(loaded.ptr(), view + loc.byteOffset, loc.byteSize);
    if (lru_.insert(key, std::move(loaded))) lru_.pin(key);  // shared experts are always hot
    return scratch.data();
}

double Scheduler::lruHitRate() const { return lru_.hitRate(); }
size_t Scheduler::lruUsedBytes() const { return lru_.usedBytes(); }
size_t Scheduler::lruCapacityBytes() const { return lru_.capacityBytes(); }

void Scheduler::warmDenseWeights() {
    // O7 (doc/design.md 17.7): every forward re-reads the DENSE (non-routed,
    // non-LRU) tensors through getWeightPtr mmap faults. On a cold OS page cache
    // that fault storm dominates token-1 (measured ~4s/layer extra on the HDD that
    // evaporates once resident). Warming the ranges once with big sequential reads
    // pre-populates the file cache, so the first forward's faults hit RAM.
    if (!index_) return;

    auto add = [](std::vector<std::pair<std::string, std::pair<uint64_t, uint64_t>>>& out,
                  const TensorLocation& t) {
        if (t.byteSize > 0 && !t.shardPath.empty())
            out.push_back({t.shardPath, {t.byteOffset, t.byteSize}});
    };

    std::vector<std::pair<std::string, std::pair<uint64_t, uint64_t>>> ranges;
    ranges.reserve(64);
    for (const auto& lw : index_->layers) {
        add(ranges, lw.inputLayernorm);
        add(ranges, lw.postAttentionLayernorm);
        add(ranges, lw.qAProj);
        add(ranges, lw.qALayernorm);
        add(ranges, lw.qBProj);
        add(ranges, lw.kvAProjWithMqa);
        add(ranges, lw.kvALayernorm);
        add(ranges, lw.kvBProj);
        add(ranges, lw.oProj);
        add(ranges, lw.indexWqB);
        add(ranges, lw.indexWk);
        add(ranges, lw.indexWeightsProj);
        add(ranges, lw.indexKNormW);
        add(ranges, lw.indexKNormB);
        add(ranges, lw.mlpGate);
        add(ranges, lw.mlpUp);
        add(ranges, lw.mlpDown);
        add(ranges, lw.gateBias);
        add(ranges, lw.sharedGateProj);
        add(ranges, lw.sharedUpProj);
        add(ranges, lw.sharedDownProj);
    }
    add(ranges, index_->embedTokens);
    add(ranges, index_->lmHead);
    add(ranges, index_->finalNorm);

    // Group per shard, sort by offset, merge with the same gap tolerance as O3.
    std::unordered_map<std::string, std::vector<std::pair<uint64_t, uint64_t>>> byShard;
    for (const auto& [shard, rr] : ranges) byShard[shard].push_back(rr);

    struct WarmRun { std::string shard; uint64_t start; uint64_t end; };
    std::vector<WarmRun> runs;
    uint64_t totalWarm = 0;
    for (auto& [shard, v] : byShard) {
        std::sort(v.begin(), v.end());
        for (const auto& [off, sz] : v) {
            if (!runs.empty() && runs.back().shard == shard &&
                off <= runs.back().end + kRunMergeGap) {
                if (off + sz > runs.back().end) runs.back().end = off + sz;
            } else {
                runs.push_back({shard, off, off + sz});
            }
        }
    }
    for (const auto& r : runs) totalWarm += r.end - r.start;
    if (runs.empty()) return;

    // Discard-read each merged run in fixed chunks through a dedicated reader,
    // keeping the queue deep (pool of kWarmPool buffers, flushed every wrap).
    IOCPReader warm;
    if (!warm.start(8)) return;

    std::vector<uint8_t> pool(size_t(kWarmPool) * kWarmChunk);
    size_t submitted = 0;
    GLM_LOG_INFO("Warm: streaming " + std::to_string(totalWarm / (1024 * 1024)) +
                 " MiB of dense+shared weights into the OS cache (" +
                 std::to_string(runs.size()) + " runs)");
    auto submit = [&](const std::string& shard, uint64_t off, uint64_t sz) {
        size_t idx = submitted % size_t(kWarmPool);
        ReadRequest req{shard, off, sz, pool.data() + idx * kWarmChunk};
        warm.submitRead(req, [](bool, size_t) {});
        ++submitted;
        if (submitted % size_t(kWarmPool) == 0) warm.waitAll();
    };

    for (const auto& r : runs) {
        for (uint64_t pos = r.start; pos < r.end; pos += kWarmChunk) {
            submit(r.shard, pos, std::min(kWarmChunk, r.end - pos));
        }
    }
    warm.waitAll();
    warm.stop();
    GLM_LOG_INFO("Warm: done (" + std::to_string(submitted) + " reads)");
}

} // namespace glm