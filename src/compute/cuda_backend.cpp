#include "compute/cuda_backend.h"
#include "engine/inference_stats.h"

#include <cmath>
#include <cstdint>

#if GLM_HAVE_CUDA
#include "utils/logger.h"

#include <cuda_runtime.h>
#include "compute/cuda_kernels.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <random>
#include <unordered_map>
#include <vector>

namespace glm {

namespace {

bool g_available = false;

// Host-side CPU implementations used when the GPU path is unavailable.
void cpuEmbeddingLookup(const BFloat16* emb, int tokenId, float* out, int hidden) {
    for (int i = 0; i < hidden; ++i) out[i] = emb[int64_t(tokenId) * hidden + i].toF32();
}

void cpuMatvecBf16(const BFloat16* W, const float* x, float* y, int M, int K) {
    for (int i = 0; i < M; ++i) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) acc += W[int64_t(i) * K + k].toF32() * x[k];
        y[i] = acc;
    }
}

void cpuSoftmax(float* x, int n) {
    float maxVal = x[0];
    for (int i = 1; i < n; ++i) maxVal = std::fmax(maxVal, x[i]);
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - maxVal);
        sum += x[i];
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; ++i) x[i] *= inv;
}

int cpuArgmax(const float* x, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) {
        if (x[i] > x[best]) best = i;
    }
    return best;
}

size_t g_expertWeightBytes(int hidden, int inter) {
    // [gate | up | down], each BF16 hidden x inter -> 6 * hidden * inter bytes.
    return size_t(hidden) * size_t(inter) * 6;
}

} // namespace

bool cudaInit() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0) {
        GLM_LOG_WARN("CUDA device not available");
        g_available = false;
        return false;
    }
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    GLM_LOG_INFO("CUDA device ready: " + std::string(prop.name));
    g_available = true;
    return true;
}

void cudaShutdown() {
    gpuExpertCacheShutdown();
    if (!g_available) return;
    cudaDeviceReset();
    g_available = false;
}

std::string cudaDeviceName() {
    if (!g_available) return {};
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    return std::string(prop.name);
}

size_t cudaDeviceMemoryMB() {
    if (!g_available) return 0;
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    return prop.totalGlobalMem / (1024 * 1024);
}

bool gpuAvailable() { return g_available; }

bool gpuEmbeddingLookup(const BFloat16* emb, int tokenId, float* out, int hidden) {
    if (!g_available) { cpuEmbeddingLookup(emb, tokenId, out, hidden); return true; }
    const unsigned short* src = reinterpret_cast<const unsigned short*>(emb);
    unsigned short* dEmb = nullptr;
    float* dOut = nullptr;
    cudaMalloc(&dEmb, sizeof(unsigned short) * size_t(tokenId + 1) * size_t(hidden));
    cudaMalloc(&dOut, sizeof(float) * size_t(hidden));
    if (!dEmb || !dOut) { cudaFree(dEmb); cudaFree(dOut); cpuEmbeddingLookup(emb, tokenId, out, hidden); return true; }
    cudaMemcpy(dEmb, src, sizeof(unsigned short) * size_t(tokenId + 1) * size_t(hidden), cudaMemcpyHostToDevice);
    bool ok = glm_cuda::embeddingGather(dEmb, tokenId, dOut, hidden);
    if (ok) cudaMemcpy(out, dOut, sizeof(float) * size_t(hidden), cudaMemcpyDeviceToHost);
    cudaFree(dEmb); cudaFree(dOut);
    if (ok) return true;
    cpuEmbeddingLookup(emb, tokenId, out, hidden);
    return true;
}

bool gpuMatvecBf16(const BFloat16* W, const float* x, float* y, int M, int K) {
    if (!g_available) { cpuMatvecBf16(W, x, y, M, K); return true; }
    unsigned short* dW = nullptr;
    float* dX = nullptr;
    float* dY = nullptr;
    cudaMalloc(&dW, sizeof(unsigned short) * size_t(M) * size_t(K));
    cudaMalloc(&dX, sizeof(float) * size_t(K));
    cudaMalloc(&dY, sizeof(float) * size_t(M));
    if (!dW || !dX || !dY) { cudaFree(dW); cudaFree(dX); cudaFree(dY); cpuMatvecBf16(W, x, y, M, K); return true; }
    cudaMemcpy(dW, reinterpret_cast<const unsigned short*>(W), sizeof(unsigned short) * size_t(M) * size_t(K), cudaMemcpyHostToDevice);
    cudaMemcpy(dX, x, sizeof(float) * size_t(K), cudaMemcpyHostToDevice);
    bool ok = glm_cuda::matvec(dW, dX, dY, M, K);
    if (ok) cudaMemcpy(y, dY, sizeof(float) * size_t(M), cudaMemcpyDeviceToHost);
    cudaFree(dW); cudaFree(dX); cudaFree(dY);
    if (ok) return true;
    cpuMatvecBf16(W, x, y, M, K);
    return true;
}

bool gpuSoftmax(float* x, int n) {
    if (!g_available) { cpuSoftmax(x, n); return true; }
    float* dX = nullptr;
    cudaMalloc(&dX, sizeof(float) * size_t(n));
    if (!dX) { cpuSoftmax(x, n); return true; }
    cudaMemcpy(dX, x, sizeof(float) * size_t(n), cudaMemcpyHostToDevice);
    bool ok = glm_cuda::softmax(dX, n);
    if (ok) cudaMemcpy(x, dX, sizeof(float) * size_t(n), cudaMemcpyDeviceToHost);
    cudaFree(dX);
    if (ok) return true;
    cpuSoftmax(x, n);
    return true;
}

int gpuArgmax(const float* x, int n) {
    if (!g_available) return cpuArgmax(x, n);
    float* dX = nullptr;
    int* dBest = nullptr;
    cudaMalloc(&dX, sizeof(float) * size_t(n));
    cudaMalloc(&dBest, sizeof(int));
    if (!dX || !dBest) { cudaFree(dX); cudaFree(dBest); return cpuArgmax(x, n); }
    cudaMemcpy(dX, x, sizeof(float) * size_t(n), cudaMemcpyHostToDevice);
    bool ok = glm_cuda::argmax(dX, n, dBest);
    int best = -1;
    if (ok) cudaMemcpy(&best, dBest, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dX); cudaFree(dBest);
    if (ok) return best;
    return cpuArgmax(x, n);
}

// ---------- GPU expert FFN residency cache (#1/#2) ----------
// Offloads the routed-expert FFN to the GPU with a double-buffered async H2D
// staging pipeline: the caller streams one expert ahead, so the next expert's
// pinned-memory upload overlaps the current expert's compute. Experts stay
// resident in VRAM (capacity-sized LRU) so repeated routing hits avoid the bus.
// Every entry point reports failure instead of throwing, so the forward path
// can fall back to the CPU implementation and goldens stay byte-identical.

namespace {

// One VRAM-resident expert: weights + per-expert activation scratch.
struct DeviceExpert {
    unsigned short* dW = nullptr;   // [gate | up | down], 6*hidden*inter uint16
    float* dAct = nullptr;          // inter floats (gate output / silu result)
    float* dUp = nullptr;           // inter floats (up output)
    size_t weightBytes = 0;
    int lastUse = 0;
    bool inUse = false;
};

// Ping-pong staging slot: pinned host buffer + a per-copy completion event.
struct StagingSlot {
    unsigned short* host = nullptr;
    cudaEvent_t copyDone = nullptr;
};

class ExpertGpuCache {
public:
    int requestedCapacity = 4;
    int capacity = 0;
    int hidden = 0;
    int inter = 0;
    bool configured = false;

    cudaStream_t copyStream = nullptr;
    cudaStream_t computeStream = nullptr;
    float* dX = nullptr;    // shared input  (hidden floats)
    float* dOut = nullptr;  // shared output (hidden floats)

    std::vector<DeviceExpert> experts;
    std::vector<StagingSlot> slots;
    std::vector<char> slotPending;  // H2D from this slot not yet consumed by ffn
    int slotCount = 2;
    int nextStageSlot = 0;          // round-robin staging cursor

    std::unordered_map<uint64_t, int> mapKeyToSlot;          // (layer,eid) -> device slot
    std::unordered_map<uint64_t, bool> uploadPending;         // H2D not yet consumed by ffn
    std::unordered_map<uint64_t, cudaEvent_t> uploadEvent;    // per-expert copy-done event
    std::unordered_map<uint64_t, int> uploadStageSlot;        // staging slot owning that event
    int lruClock = 0;

    static uint64_t key(int layer, int eid) {
        return (uint64_t(uint32_t(layer)) << 16) | uint64_t(uint16_t(eid));
    }

    bool alloc(int cap, int h, int i) {
        capacity = cap;
        hidden = h;
        inter = i;
        experts.assign(size_t(cap), DeviceExpert{});
        slots.assign(size_t(slotCount), StagingSlot{});
        slotPending.assign(size_t(slotCount), 0);
        nextStageSlot = 0;
        const size_t one = g_expertWeightBytes(h, i);
        size_t actBytes = sizeof(float) * size_t(i);

        bool ok = cudaStreamCreate(&copyStream) == cudaSuccess &&
                  cudaStreamCreate(&computeStream) == cudaSuccess;
        for (int s = 0; s < slotCount && ok; ++s) {
            ok = cudaHostAlloc(&slots[s].host, one, cudaHostAllocDefault) == cudaSuccess &&
                 cudaEventCreateWithFlags(&slots[s].copyDone, cudaEventDisableTiming) == cudaSuccess;
        }
        for (int e = 0; e < cap && ok; ++e) {
            ok = cudaMalloc(&experts[e].dW, one) == cudaSuccess &&
                 cudaMalloc(&experts[e].dAct, actBytes) == cudaSuccess &&
                 cudaMalloc(&experts[e].dUp, actBytes) == cudaSuccess;
            if (ok) {
                experts[e].weightBytes = one;
                experts[e].lastUse = 0;
                experts[e].inUse = false;
            }
        }
        ok = ok && cudaMalloc(&dX, sizeof(float) * size_t(h)) == cudaSuccess &&
                   cudaMalloc(&dOut, sizeof(float) * size_t(h)) == cudaSuccess;
        if (!ok) {
            release();
            return false;
        }
        mapKeyToSlot.reserve(size_t(cap) + 16);
        uploadPending.reserve(size_t(cap) + 16);
        uploadEvent.reserve(size_t(cap) + 16);
        configured = true;
        return true;
    }

    void release() {
        for (auto& e : experts) {
            cudaFree(e.dW);
            cudaFree(e.dAct);
            cudaFree(e.dUp);
        }
        experts.clear();
        for (auto& s : slots) {
            if (s.host) cudaFreeHost(s.host);
            if (s.copyDone) cudaEventDestroy(s.copyDone);
        }
        slots.clear();
        if (dX) cudaFree(dX);
        if (dOut) cudaFree(dOut);
        if (copyStream) cudaStreamDestroy(copyStream);
        if (computeStream) cudaStreamDestroy(computeStream);
        copyStream = nullptr;
        computeStream = nullptr;
        dX = dOut = nullptr;
        mapKeyToSlot.clear();
        uploadPending.clear();
        configured = false;
        capacity = 0;
    }

    // Round-robin pinned staging slot claim. Reusing a slot whose previous
    // async H2D has not been consumed would read garbled bytes (the host buffer
    // is overwritten before the queued copy executes). To guarantee the byte
    // stream is correct we drain any earlier copy on the copy stream before
    // reusing the slot — this only happens when staging moves faster than the
    // H2D pipeline drains.
    int claimStageSlot(cudaStream_t copyStream) {
        const int s = nextStageSlot;
        nextStageSlot = (nextStageSlot + 1) % std::max(1, slotCount);
        if (slotPending[size_t(s)] != 0) {
            if (cudaStreamSynchronize(copyStream) != cudaSuccess) return -1;
            for (char& p : slotPending) p = 0;
        }
        return s;
    }

    // Compute has consumed the H2D of the given slot (after the wait).
    void releaseStageSlot(int s) {
        if (s >= 0 && s < int(slotPending.size())) slotPending[size_t(s)] = 0;
    }

    // Returns true when the slot's host staging is unsafe to reuse.
    bool stageSlotPending(int s) const {
        return s >= 0 && s < int(slotPending.size()) && slotPending[size_t(s)] != 0;
    }

    int findSlot(int layer, int eid) const {
        auto it = mapKeyToSlot.find(key(layer, eid));
        return it != mapKeyToSlot.end() ? it->second : -1;
    }

    // Resident-slot lookup or LRU admission (eviction of the least-used entry).
    int acquireSlot(int layer, int eid, bool& needUpload) {
        const uint64_t k = key(layer, eid);
        auto it = mapKeyToSlot.find(k);
        if (it != mapKeyToSlot.end()) {
            experts[it->second].lastUse = ++lruClock;
            needUpload = false;
            return it->second;
        }
        int victim = -1;
        uint64_t evictedKey = 0;
        if (int(mapKeyToSlot.size()) >= capacity) {
            int oldest = std::numeric_limits<int>::max();
            for (int e = 0; e < capacity; ++e) {
                if (experts[e].lastUse < oldest) {
                    oldest = experts[e].lastUse;
                    victim = e;
                }
            }
        } else {
            for (int e = 0; e < capacity; ++e) {
                if (!experts[e].inUse) { victim = e; break; }
            }
            if (victim < 0) victim = 0;
        }
        for (auto mit = mapKeyToSlot.begin(); mit != mapKeyToSlot.end();) {
            if (mit->second == victim) {
                evictedKey = mit->first;
                mit = mapKeyToSlot.erase(mit);
            } else {
                ++mit;
            }
        }
        uploadPending.erase(evictedKey);
        uploadEvent.erase(evictedKey);
        uploadStageSlot.erase(evictedKey);
        DeviceExpert& d = experts[victim];
        d.inUse = true;
        d.lastUse = ++lruClock;
        mapKeyToSlot[k] = victim;
        needUpload = true;
        return victim;
    }
};

ExpertGpuCache* g_expertCache = nullptr;

} // namespace

bool gpuExpertCacheConfig(int capacity, int stagingDepth) {
    if (!g_available || capacity <= 0) return false;
    if (g_expertCache) {
        g_expertCache->release();
        delete g_expertCache;
        g_expertCache = nullptr;
    }
    g_expertCache = new ExpertGpuCache();
    g_expertCache->requestedCapacity = std::max(1, capacity);
    g_expertCache->slotCount = std::max(1, stagingDepth);
    return true;
}

void gpuExpertCacheShutdown() {
    if (g_expertCache) {
        g_expertCache->release();
        delete g_expertCache;
        g_expertCache = nullptr;
    }
}

bool gpuExpertsEnabled() {
    return g_available && g_expertCache != nullptr;
}

bool gpuExpertStage(int layer, int expertId, const BFloat16* w, int hidden, int inter) {
    if (!gpuExpertsEnabled()) return false;
    ExpertGpuCache& c = *g_expertCache;
    if (!c.configured) {
        if (!c.alloc(c.requestedCapacity, hidden, inter)) return false;
    }
    if (c.hidden != hidden || c.inter != inter || c.capacity == 0) return false;

    bool needUpload = false;
    const int slot = c.acquireSlot(layer, expertId, needUpload);
    if (!needUpload) return true;  // already resident; LRU touched

    // Claim a pinned staging slot (round-robin; GPU-copy-safe against reuse).
    const int stageSlot = c.claimStageSlot(c.copyStream);
    if (stageSlot < 0) return false;

    // Copy the transient host block into pinned staging, then upload it
    // asynchronously (deep staging: the N+1 .. N+stagingDepth copies overlap
    // the current expert's compute).
    StagingSlot& s = c.slots[size_t(stageSlot)];
    std::memcpy(s.host, w, c.experts[slot].weightBytes);
    cudaError_t e = cudaMemcpyAsync(c.experts[slot].dW, s.host,
                                    c.experts[slot].weightBytes,
                                    cudaMemcpyHostToDevice, c.copyStream);
    if (e != cudaSuccess) return false;
    cudaEventRecord(s.copyDone, c.copyStream);
    c.slotPending[size_t(stageSlot)] = 1;
    const uint64_t k = ExpertGpuCache::key(layer, expertId);
    c.uploadPending[k] = true;
    c.uploadEvent[k] = s.copyDone;
    c.uploadStageSlot[k] = stageSlot;
    InferenceStats::instance().pcieBytes += c.experts[slot].weightBytes;  // H2D
    return true;
}

bool gpuExpertFFN(int layer, int expertId, const float* x, float* out,
                  int hidden, int inter) {
    if (!gpuExpertsEnabled()) return false;
    ExpertGpuCache& c = *g_expertCache;
    if (!c.configured || c.hidden != hidden || c.inter != inter) return false;
    const int slot = c.findSlot(layer, expertId);
    if (slot < 0) return false;

    const uint64_t k = ExpertGpuCache::key(layer, expertId);
    auto pit = c.uploadPending.find(k);
    if (pit != c.uploadPending.end() && pit->second) {
        auto eit = c.uploadEvent.find(k);
        if (eit != c.uploadEvent.end()) {
            if (cudaStreamWaitEvent(c.computeStream, eit->second) != cudaSuccess) {
                return false;
            }
        }
        auto sit = c.uploadStageSlot.find(k);
        if (sit != c.uploadStageSlot.end()) c.releaseStageSlot(sit->second);
        pit->second = false;
    }

    DeviceExpert& d = c.experts[slot];
    if (cudaMemcpyAsync(c.dX, x, sizeof(float) * size_t(hidden),
                        cudaMemcpyHostToDevice, c.computeStream) != cudaSuccess) {
        return false;
    }
    auto& stats = InferenceStats::instance();
    stats.pcieBytes += sizeof(float) * size_t(hidden);  // x H2D

    const size_t half = size_t(hidden) * size_t(inter);
    // Timing events (with timing enabled -- DisableTiming events make
    // cudaEventElapsedTime fail, whose error would LATCH and poison the next
    // cudaGetLastError()) bracket the FFN on the compute stream.
    cudaEvent_t start = nullptr, stop = nullptr;
    cudaEventCreateWithFlags(&start, 0) == cudaSuccess ? void() : void();
    cudaEventCreateWithFlags(&stop, 0) == cudaSuccess ? void() : void();
    bool timed = start && stop &&
                 cudaEventRecord(start, c.computeStream) == cudaSuccess;
    bool ok = glm_cuda::expertFfn(d.dW, d.dW + half, d.dW + 2 * half,
                                  c.dX, c.dOut, d.dAct, d.dUp,
                                  hidden, inter, c.computeStream);
    if (timed && ok && cudaEventRecord(stop, c.computeStream) == cudaSuccess) {
        cudaStreamSynchronize(c.computeStream);
        float ms = 0.0f;
        if (cudaEventElapsedTime(&ms, start, stop) == cudaSuccess) {
            stats.gpuActiveNs += uint64_t(ms * 1e6);
        }
    } else if (!ok) {
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        return false;
    }
    if (start) cudaEventDestroy(start);
    if (stop) cudaEventDestroy(stop);
    // Flush any latched error so a later expertFfn's cudaGetLastError() never
    // misreports this call's state (async errors are sticky until queried).
    (void)cudaGetLastError();
    if (cudaMemcpy(out, c.dOut, sizeof(float) * size_t(hidden),
                  cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    stats.pcieBytes += sizeof(float) * size_t(hidden);  // out D2H
    return true;
}

bool cudaExpertSelfTest() {
    if (!g_available) return true;  // no device: skipped

    const std::pair<int, int> cases[] = { {64, 32}, {6144, 2048} };
    bool allOk = true;
    for (const auto& cse : cases) {
        const int h = cse.first, i = cse.second;
        const size_t half = size_t(h) * size_t(i);
        std::vector<BFloat16> w(3 * half);
        std::mt19937 rng(1);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& v : w) v = BFloat16::fromF32(dist(rng));

        std::vector<float> x(static_cast<size_t>(h)), yGpu(static_cast<size_t>(h)),
                           yRef(static_cast<size_t>(h));
        for (auto& v : x) v = 0.7f * dist(rng);

        ExpertGpuCache* prev = g_expertCache;
        g_expertCache = new ExpertGpuCache();
        bool ok = g_expertCache->alloc(2, h, i);
        GLM_LOG_INFO("[slt] case " + std::to_string(h) + "x" + std::to_string(i) +
                     " alloc=" + std::to_string(int(ok)));
        ok = ok && gpuExpertStage(1, 3, w.data(), h, i);
        GLM_LOG_INFO("[slt] stage 3 ok=" + std::to_string(int(ok)));
        ok = ok && gpuExpertStage(1, 7, w.data(), h, i);
        GLM_LOG_INFO("[slt] stage 7 ok=" + std::to_string(int(ok)));
        ok = ok && gpuExpertFFN(1, 3, x.data(), yGpu.data(), h, i);
        GLM_LOG_INFO("[slt] ffn 3 ok=" + std::to_string(int(ok)));

        // Host reference using the same fp32 math as the CPU forward path.
        std::vector<float> g(static_cast<size_t>(i)), u(static_cast<size_t>(i)),
                           m(static_cast<size_t>(i));
        cpuMatvecBf16(w.data(), x.data(), g.data(), i, h);
        cpuMatvecBf16(w.data() + half, x.data(), u.data(), i, h);
        for (int r = 0; r < i; ++r) m[r] = (g[r] / (1.0f + std::exp(-g[r]))) * u[r];
        cpuMatvecBf16(w.data() + 2 * half, m.data(), yRef.data(), h, i);

        bool caseOk = true;
        float worst = 0.0f;
        for (int k = 0; k < h; ++k) {
            if (std::fabs(yRef[k]) > 1e-8f) {
                const float rel = std::fabs(yGpu[k] - yRef[k]) / std::fabs(yRef[k]);
                if (rel > worst) worst = rel;
                // fp32 tree-vs-serial accumulation across 2048-term dot products
                // reaches ~1e-3; the engine's golden tolerance is 1e-2.
                if (rel > 1e-2f) caseOk = false;
            } else if (std::fabs(yGpu[k]) > 1e-6f) {
                caseOk = false;
            }
        }
        GLM_LOG_INFO("[slt] case " + std::to_string(h) + "x" + std::to_string(i) +
                     " match=" + std::to_string(int(caseOk)) +
                     " worstRelErr=" + std::to_string(worst));
        allOk = allOk && caseOk;

        g_expertCache->release();
        delete g_expertCache;
        g_expertCache = prev;
    }
    return allOk;
}

} // namespace glm
#else

namespace glm {

bool cudaInit() { return false; }
void cudaShutdown() {}
std::string cudaDeviceName() { return {}; }
size_t cudaDeviceMemoryMB() { return 0; }
bool gpuAvailable() { return false; }

bool gpuEmbeddingLookup(const BFloat16* emb, int tokenId, float* out, int hidden) {
    for (int i = 0; i < hidden; ++i) out[i] = emb[int64_t(tokenId) * hidden + i].toF32();
    return true;
}

bool gpuMatvecBf16(const BFloat16* W, const float* x, float* y, int M, int K) {
    for (int i = 0; i < M; ++i) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) acc += W[int64_t(i) * K + k].toF32() * x[k];
        y[i] = acc;
    }
    return true;
}

bool gpuSoftmax(float* x, int n) {
    float maxVal = x[0];
    for (int i = 1; i < n; ++i) maxVal = std::fmax(maxVal, x[i]);
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - maxVal);
        sum += x[i];
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; ++i) x[i] *= inv;
    return true;
}

int gpuArgmax(const float* x, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) {
        if (x[i] > x[best]) best = i;
    }
    return best;
}

bool gpuExpertCacheConfig(int, int) { return false; }
void gpuExpertCacheShutdown() {}
bool gpuExpertsEnabled() { return false; }

bool gpuExpertStage(int, int, const BFloat16*, int, int) { return false; }

bool gpuExpertFFN(int, int, const float*, float*, int, int) { return false; }

bool cudaExpertSelfTest() { return true; }

} // namespace glm
#endif