#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <sstream>
#include <string>

namespace glm {

// Runtime telemetry counters (benchmark upgrade). Instrumented by the scheduler
// (expert LRU / prefetch / IO), the forward path (compute) and main.cpp (CPU
// time). At the end of a run main.cpp serializes the counters as a single
// `STATS: {...}` JSON line; scripts/benchmark.py parses it into the L4 report's
// I/O & cache telemetry table.
class InferenceStats {
public:
    // Expert LRU / disk fallback
    std::atomic<uint64_t> expertLookups{0};
    std::atomic<uint64_t> expertLruHits{0};
    std::atomic<uint64_t> expertLruMisses{0};
    std::atomic<uint64_t> expertDiskLoads{0};     // synchronous mmap fallback loads

    // Prefetch pipeline
    std::atomic<uint64_t> prefetchSubmitted{0};   // merged IO runs submitted
    std::atomic<uint64_t> prefetchInserted{0};    // assemblies landed in the LRU
    std::atomic<uint64_t> prefetchWasted{0};      // prefetched entry evicted before first use

    // IO telemetry (bytes moved through the prefetch pipeline)
    std::atomic<uint64_t> diskBytesRead{0};
    std::atomic<uint64_t> sequentialBytes{0};     // bytes inside merged sequential runs
    std::atomic<uint64_t> randomIoCount{0};       // merged run count (random access events)
    std::atomic<uint64_t> ioReadCount{0};         // individual read requests
    std::atomic<uint64_t> ioLatencyNsSum{0};      // cumulative async read latency

    // Expert reuse distance (tokens between consecutive uses, summed + count)
    std::atomic<uint64_t> reuseDistanceSum{0};
    std::atomic<uint64_t> reuseDistanceCount{0};

    // Compute-path RAM traffic (item 5: real RAM bandwidth). Bumped by the
    // CPU kernels with the exact bytes each op streams (weights + activations),
    // so ram_bandwidth is measured, not inferred from disk bytes.
    std::atomic<uint64_t> ramBytesMoved{0};

    // GPU expert pipeline telemetry (item 5). gpuActiveNs accumulates the time
    // the device is actually busy inside the expert FFN streaming path;
    // pcieBytes counts every byte crossed over the bus (expert weight uploads,
    // per-use x/out transfers).
    std::atomic<uint64_t> gpuActiveNs{0};
    std::atomic<uint64_t> pcieBytes{0};

    static InferenceStats& instance() { static InferenceStats s; return s; }

    void resetAll() {
        expertLookups = 0;
        expertLruHits = 0;
        expertLruMisses = 0;
        expertDiskLoads = 0;
        prefetchSubmitted = 0;
        prefetchInserted = 0;
        prefetchWasted = 0;
        diskBytesRead = 0;
        sequentialBytes = 0;
        randomIoCount = 0;
        ioReadCount = 0;
        ioLatencyNsSum = 0;
        reuseDistanceSum = 0;
        reuseDistanceCount = 0;
        ramBytesMoved = 0;
        gpuActiveNs = 0;
        pcieBytes = 0;
    }

    // cpuTimeNs = process CPU time for the run; elapsedSec = wall time;
    // generatedTokens = number of generated tokens (decode steps). Emits a
    // flat JSON object (no newlines) safe to embed in a log line.
    std::string toJson(uint64_t cpuTimeNs, double elapsedSec,
                       uint64_t generatedTokens = 0) const {
        const double gpuBusy = gpuActiveNs.load() / 1e9;
        // PCIe reference: 16 GB/s (PCIe 3.0 x16, RTX 3060-class upstream link).
        constexpr double kPcieRefBytePerSec = 16e9;
        std::ostringstream s;
        s << "{\"generated_sec\":" << elapsedSec
          << ",\"cpu_utilization\":" << (elapsedSec > 0.0 ? double(cpuTimeNs) / 1e9 / elapsedSec : 0.0)
          << ",\"gpu_utilization\":" << (elapsedSec > 0.0 ? std::min(gpuBusy / elapsedSec, 1.0) : 0.0)
          << ",\"pcie_utilization\":" << (elapsedSec > 0.0 ? std::min(double(pcieBytes.load()) / (elapsedSec * kPcieRefBytePerSec), 1.0) : 0.0)
          << ",\"gpu_active_ms\":" << gpuBusy * 1e3
          << ",\"pcie_bytes\":" << pcieBytes.load()
          << ",\"expert_lookups\":" << expertLookups.load()
          << ",\"expert_lru_hits\":" << expertLruHits.load()
          << ",\"expert_lru_misses\":" << expertLruMisses.load()
          << ",\"expert_disk_loads\":" << expertDiskLoads.load()
          << ",\"expert_cache_hit_rate\":" << hitRate(expertLruHits.load(), expertLruMisses.load())
          << ",\"expert_cache_miss_rate\":" << missRate(expertLruHits.load(), expertLruMisses.load())
          << ",\"prefetch_submitted\":" << prefetchSubmitted.load()
          << ",\"prefetch_inserted\":" << prefetchInserted.load()
          << ",\"prefetch_wasted\":" << prefetchWasted.load()
          << ",\"prefetch_delivery_rate\":" << ratio(prefetchInserted.load(), prefetchSubmitted.load())
          << ",\"prefetch_waste_rate\":" << ratio(prefetchWasted.load(), prefetchSubmitted.load())
          << ",\"disk_bytes_read\":" << diskBytesRead.load()
          << ",\"sequential_bytes\":" << sequentialBytes.load()
          << ",\"random_io_count\":" << randomIoCount.load()
          << ",\"io_read_count\":" << ioReadCount.load()
          << ",\"bytes_read_per_token\":" << (generatedTokens > 0 ? double(diskBytesRead.load()) / generatedTokens : 0.0)
          << ",\"sequential_bytes_per_token\":" << (generatedTokens > 0 ? double(sequentialBytes.load()) / generatedTokens : 0.0)
          << ",\"hdd_latency_ms\":" << avgLatencyMs()
          << ",\"ram_bandwidth_mbps\":" << (elapsedSec > 0.0 ? double(ramBytesMoved.load()) / elapsedSec / 1e6 : 0.0)
          << ",\"pcie_transfer_gbps\":" << (elapsedSec > 0.0 ? double(pcieBytes.load()) / elapsedSec / 1e9 : 0.0)
          << ",\"expert_reuse_distance_tok\":" << avgInt(reuseDistanceSum.load(), reuseDistanceCount.load())
          << "}";
        return s.str();
    }

private:
    static double hitRate(uint64_t hits, uint64_t misses) {
        uint64_t total = hits + misses;
        return total == 0 ? 0.0 : double(hits) / double(total);
    }
    static double missRate(uint64_t hits, uint64_t misses) {
        uint64_t total = hits + misses;
        return total == 0 ? 0.0 : double(misses) / double(total);
    }
    static double ratio(uint64_t a, uint64_t b) {
        return b == 0 ? 0.0 : double(a) / double(b);
    }
    static double avgInt(uint64_t sum, uint64_t count) {
        return count == 0 ? 0.0 : double(sum) / double(count);
    }
    double avgLatencyMs() const {
        uint64_t cnt = ioReadCount.load();
        if (cnt == 0) return 0.0;
        return double(ioLatencyNsSum.load()) / double(cnt) / 1e6;
    }
};

} // namespace glm