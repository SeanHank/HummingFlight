#include "engine/scheduler.h"
#include "utils/logger.h"
#include <algorithm>
#include <atomic>
#include <cstring>
#include <memory>
#include <unordered_map>

namespace glm {

namespace {

constexpr int kPinThreshold = 5;  // routing accesses before an LRU learn-in (learned pin)

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

    GLM_LOG_INFO("Scheduler ready: LRU=" + std::to_string(lruCapacityBytes / (1024 * 1024)) +
                 " MB, IOCP workers=" + std::to_string(iocpWorkers));
    return true;
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
    if (nextLayer >= 0 && nextLayer < int(lastTopK_.size()) && !lastTopK_[nextLayer].empty()) {
        return lastTopK_[nextLayer];  // previous token's routing at the same layer
    }
    return currentTopK;               // weak inter-layer fallback
}

void Scheduler::onLayerRouterDone(int layer, const std::vector<int>& topKExperts) {
    if (!index_) return;
    if (layer >= 0 && layer < int(lastTopK_.size())) lastTopK_[layer] = topKExperts;

    int nextLayer = layer + 1;
    if (nextLayer < 0 || nextLayer >= index_->config.numLayers) return;

    auto predicted = predictNextLayerExperts(nextLayer, topKExperts);

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

    if (!scope.empty()) prefetchAsync(scope);
}

void Scheduler::prefetchAsync(const std::vector<std::pair<int, int>>& experts) {
    if (!index_) return;

    std::unordered_map<std::string, std::vector<ProjectionIO>> byShard;
    std::unordered_map<ExpertKey, std::shared_ptr<Assembly>, ExpertKeyHash> assemblies;

    for (const auto& [layer, eidRaw] : experts) {
        if (layer < 0 || layer >= index_->config.numLayers) continue;
        uint16_t eid = uint16_t(eidRaw);

        ExpertKey key{uint32_t(layer), eid};
        if (probe(key)) continue;

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

        // Compute total and per-projection expert offsets; require one shard.
        size_t total = 0;
        std::string singleShard;
        bool sameShard = true;
        for (const auto& p : projs) {
            total += p.loc->byteSize;
            if (singleShard.empty()) singleShard = p.loc->shardPath;
            else if (p.loc->shardPath != singleShard) sameShard = false;
        }
        if (total == 0 || !sameShard) continue;  // multi-shard experts: sync path

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
            if (run->slices.empty() || io.offset > run->end) {
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

    // Wasted-decrement compensation: projections whose size was zero already
    // decremented `remaining`; runs only carry positive-size projections.

    for (auto& run : runs) {
        uint64_t runSize = run->end - run->start;
        if (runSize == 0 || run->slices.empty()) continue;

        auto buf = std::make_shared<std::vector<uint8_t>>(size_t(runSize));
        auto slices = run->slices;
        std::string shard = run->shard;
        uint64_t start = run->start;

        ReadRequest req;
        req.filePath = shard;
        req.offset = start;
        req.size = runSize;
        req.destBuffer = buf->data();

        iocp_.submitRead(req, [this, buf, slices, shard, start](bool success, size_t bytesRead) {
            (void)shard;
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
                        lru_.insert(s.assem->key, std::move(h));
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

const uint8_t* Scheduler::getExpertPtr(int layer, int expertId) {
    if (!index_) return nullptr;
    if (layer < 0 || layer >= index_->config.numLayers) return nullptr;
    if (expertId < 0 || expertId >= index_->config.numLocalExperts) return nullptr;

    ExpertKey key{uint32_t(layer), uint16_t(expertId)};
    placement_.recordAccess(layer, expertId);

    const uint64_t bytes = index_->routedExperts[layer][expertId].totalBytes();
    if (bytes == 0) return nullptr;
    if (exScratch_.size() < bytes) exScratch_.resize(size_t(bytes));

    if (lru_.copyTo(key, exScratch_.data(), exScratch_.size())) {
        if (placement_.getAccessCount(layer, expertId) >= kPinThreshold) lru_.pin(key);
        return exScratch_.data();
    }

    BufferHandle loaded;
    if (!loadExpertFromDisk(layer, expertId, loaded)) return nullptr;
    if (loaded.size != bytes) return nullptr;
    std::memcpy(exScratch_.data(), loaded.ptr(), loaded.size);

    if (lru_.insert(key, std::move(loaded))) {
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

    const uint8_t* view = index_->getShardView(loc.shardPath);
    if (!view) return nullptr;
    std::memcpy(scratch.data(), view + loc.byteOffset, loc.byteSize);

    BufferHandle loaded(loc.byteSize);
    std::memcpy(loaded.ptr(), view + loc.byteOffset, loc.byteSize);
    if (lru_.insert(key, std::move(loaded))) lru_.pin(key);  // shared experts are always hot
    return scratch.data();
}

double Scheduler::lruHitRate() const { return lru_.hitRate(); }
size_t Scheduler::lruUsedBytes() const { return lru_.usedBytes(); }
size_t Scheduler::lruCapacityBytes() const { return lru_.capacityBytes(); }

} // namespace glm