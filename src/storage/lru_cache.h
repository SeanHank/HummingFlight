#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace glm {

// Expert-level LRU cache (with heat statistics)
// Capacity counted in bytes, evicts the tail (Least Recently Used).
// Pinned entries (high routing heat, "learned pins") are exempt from eviction.

struct ExpertKey {
    uint32_t layer;
    uint16_t expert_id;

    bool operator==(const ExpertKey& o) const {
        return layer == o.layer && expert_id == o.expert_id;
    }
};

struct ExpertKeyHash {
    size_t operator()(const ExpertKey& k) const {
        return (size_t(k.layer) << 16) | k.expert_id;
    }
};

// BufferHandle: memory buffer holding expert weights
struct BufferHandle {
    std::unique_ptr<uint8_t[]> data;
    size_t size = 0;

    BufferHandle() = default;
    BufferHandle(size_t sz) : data(std::make_unique<uint8_t[]>(sz)), size(sz) {}
    void* ptr() const { return data.get(); }
};

class ExpertLRU {
public:
    explicit ExpertLRU(size_t capacityBytes) : capacity_(capacityBytes), used_(0) {}

    // Reset capacity (clears cache).
    void resize(size_t newCapacity) {
        std::lock_guard<std::mutex> lock(mtx_);
        capacity_ = newCapacity;
        lruOrder_.clear();
        map_.clear();
        used_ = 0;
        hitCount_ = 0;
        missCount_ = 0;
    }

    // Query: on hit promote to front and return a pointer; on miss return nullptr.
    BufferHandle* acquire(ExpertKey k) {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = map_.find(k);
        if (it == map_.end()) {
            ++missCount_;
            return nullptr;
        }
        ++hitCount_;
        it->second.used = true;  // compute actually consumed this entry
        lruOrder_.splice(lruOrder_.begin(), lruOrder_, it->second.listIter);
        return &it->second.handle;
    }

    // Copy the entry for k into an external buffer while holding the lock,
    // so eviction cannot free the source mid-copy. Returns false on miss or
    // when dest is too small; promotes the entry on hit.
    bool copyTo(ExpertKey k, void* dest, size_t destSize) {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = map_.find(k);
        if (it == map_.end()) {
            ++missCount_;
            return false;
        }
        ++hitCount_;
        it->second.used = true;  // compute actually consumed this entry
        lruOrder_.splice(lruOrder_.begin(), lruOrder_, it->second.listIter);
        if (destSize < it->second.handle.size) return false;
        const uint8_t* src = static_cast<const uint8_t*>(it->second.handle.ptr());
        std::memcpy(dest, src, it->second.handle.size);
        return true;
    }

    // Insert (or replace) the entry for k. Returns false when the handle is too
    // large to ever fit or when every existing entry is pinned (bounded eviction).
    // fromPrefetch marks entries that arrived via the lookahead pipeline so the
    // wasted-prefetch counter can detect entries evicted before first use.
    bool insert(ExpertKey k, BufferHandle handle, bool fromPrefetch = false) {
        std::lock_guard<std::mutex> lock(mtx_);
        size_t sz = handle.size;
        if (sz > capacity_) return false;

        // Replace an existing entry: drop the old byte contribution first.
        auto existing = map_.find(k);
        if (existing != map_.end()) {
            used_ -= existing->second.handle.size;
            lruOrder_.erase(existing->second.listIter);
            map_.erase(existing);
        }

        // Evict until the new entry fits (bounded; stops when all pinned).
        size_t guard = lruOrder_.size();
        while (used_ + sz > capacity_ && !lruOrder_.empty() && guard-- > 0) {
            if (!evictOne()) break;
        }
        if (used_ + sz > capacity_) return false;

        auto& entry = map_[k];
        entry.handle = std::move(handle);
        entry.prefetched = fromPrefetch;
        entry.used = false;
        lruOrder_.push_front(k);
        entry.listIter = lruOrder_.begin();
        used_ += sz;
        return true;
    }

    // Mark as pinned (excluded from eviction).
    void pin(ExpertKey k) {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = map_.find(k);
        if (it != map_.end()) it->second.pinned = true;
    }

    // Soft-priority boost (item 14): promote to the front of the recency order
    // and grant `protectCycles` eviction survivals. Unlike pin(), the entry is
    // still evictable — the boost only makes it outlive `protectCycles` eviction
    // sweeps. Shared/pinned entries are unaffected. No-op when not resident.
    void boost(ExpertKey k, int protectCycles = 1) {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = map_.find(k);
        if (it == map_.end()) return;
        it->second.protect = std::max(it->second.protect, protectCycles);
        lruOrder_.splice(lruOrder_.begin(), lruOrder_, it->second.listIter);
    }

    bool contains(ExpertKey k) const {
        std::lock_guard<std::mutex> lock(mtx_);
        return map_.find(k) != map_.end();
    }

    double hitRate() const {
        size_t hits = hitCount_.load();
        size_t misses = missCount_.load();
        size_t total = hits + misses;
        return total == 0 ? 0.0 : double(hits) / double(total);
    }

    // Number of prefetched entries evicted before the compute path ever touched
    // them (wasted-prefetch telemetry). Sampled at report time.
    size_t evictedPrefetchedUnused() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return evictedPrefetchedUnused_;
    }

    size_t usedBytes() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return used_;
    }
    size_t capacityBytes() const { return capacity_; }

private:
    struct Entry {
        BufferHandle handle;
        bool pinned = false;
        bool prefetched = false;  // inserted by the lookahead pipeline
        bool used = false;        // consumed by the compute path at least once
        int protect = 0;          // soft-priority eviction survivals (boost())
        std::list<ExpertKey>::iterator listIter;
    };

    // Evict the least-recently-used evictable entry: pinned entries are never
    // evicted, softly-boosted entries survive the sweep n times (consuming one
    // protection per sweep). Returns true when an entry was evicted, false when
    // only protected/pinned entries remain.
    bool evictOne() {
        for (auto it = lruOrder_.rbegin(); it != lruOrder_.rend(); ++it) {
            auto& entry = map_[*it];
            if (entry.pinned) continue;
            if (entry.protect > 0) {
                --entry.protect;  // survive this sweep
                continue;
            }
            if (entry.prefetched && !entry.used) ++evictedPrefetchedUnused_;
            used_ -= entry.handle.size;
            map_.erase(*it);
            lruOrder_.erase(std::next(it).base());
            return true;
        }
        return false;
    }

    size_t capacity_;
    size_t used_;
    size_t evictedPrefetchedUnused_ = 0;
    std::list<ExpertKey> lruOrder_;
    std::unordered_map<ExpertKey, Entry, ExpertKeyHash> map_;
    mutable std::mutex mtx_;

    std::atomic<size_t> hitCount_{0};
    std::atomic<size_t> missCount_{0};
};

} // namespace glm