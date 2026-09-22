#include "runtime/runtime_config.h"

#include "utils/logger.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <unistd.h>
#else
#include <unistd.h>
#endif

#if defined(GLM_ENABLE_CUDA) && GLM_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace glm {

namespace {

constexpr uint64_t kGiB = 1024ULL * 1024 * 1024;
constexpr uint64_t kMiB = 1024ULL * 1024;

// VRAM held back for the graphics driver / display context.
uint64_t gpuReserveBytes(uint64_t total) {
    return std::min<uint64_t>(total / 8, 512ULL * kMiB);
}

// RAM the OS and the Python tokenizer subprocess need to stay responsive.
uint64_t osReserveBytes(uint64_t ram) {
    uint64_t r = ram / 12;  // ~8%
    return std::max<uint64_t>(1ULL * kGiB, std::min<uint64_t>(r, 8ULL * kGiB));
}

#if defined(_WIN32)
void winVersionInfo(int& major, int& minor, int& build) {
    major = minor = build = 0;
    typedef LONG(WINAPI* RtlGetVersionFn)(OSVERSIONINFOW*);
    if (HMODULE ntdll = GetModuleHandleA("ntdll.dll")) {
        if (auto fn = reinterpret_cast<RtlGetVersionFn>(GetProcAddress(ntdll, "RtlGetVersion"))) {
            OSVERSIONINFOW ovi{};
            ovi.dwOSVersionInfoSize = sizeof(ovi);
            if (fn(&ovi) == 0) {
                major = int(ovi.dwMajorVersion);
                minor = int(ovi.dwMinorVersion);
                build = int(ovi.dwBuildNumber);
            }
        }
    }
}
#elif defined(__APPLE__)
std::string macSysctlString(const char* name) {
    char buf[512] = {};
    size_t sz = sizeof(buf);
    if (sysctlbyname(name, buf, &sz, nullptr, 0) == 0) {
        return std::string(buf, buf + (sz ? sz - 1 : 0));
    }
    return {};
}
uint64_t macSysctlU64(const char* name) {
    uint64_t v = 0;
    size_t sz = sizeof(v);
    sysctlbyname(name, &v, &sz, nullptr, 0);
    return v;
}
int macSysctlInt(const char* name) {
    int v = 0;
    size_t sz = sizeof(v);
    sysctlbyname(name, &v, &sz, nullptr, 0);
    return v;
}
bool macIsAppleSilicon() {
    int v = 0;
    size_t sz = sizeof(v);
    return sysctlbyname("hw.optional.arm64", &v, &sz, nullptr, 0) == 0 && v == 1;
}
#endif

uint64_t totalRamBytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) return ms.ullTotalPhys;
    return 0;
#elif defined(__APPLE__)
    return macSysctlU64("hw.memsize");
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long ps = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && ps > 0) return uint64_t(pages) * uint64_t(ps);
    return 0;
#endif
}

int logicalThreads() {
#if defined(_WIN32)
    SYSTEM_INFO si{};
    GetSystemInfo(&si);
    return int(si.dwNumberOfProcessors);
#elif defined(__APPLE__)
    int n = macSysctlInt("hw.logicalcpu");
    return n > 0 ? n : int(std::thread::hardware_concurrency());
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? int(n) : 1;
#endif
}

int physicalCores() {
#if defined(_WIN32)
    DWORD bufLen = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bufLen);
    // Bail out on implausible sizes instead of looping on a partial buffer.
    if (bufLen < sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) || bufLen > (1u << 20)) return 0;
    std::vector<uint8_t> raw(bufLen);
    if (!GetLogicalProcessorInformationEx(
            RelationProcessorCore,
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(raw.data()),
            &bufLen)) {
        return 0;
    }
    int cores = 0;
    for (DWORD off = 0; off + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) <= bufLen;) {
        auto* p = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(raw.data() + off);
        if (p->Size < sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)) break;
        if (p->Relationship == RelationProcessorCore) ++cores;
        if (off + p->Size <= off) break;  // size overflow guard
        off += p->Size;
    }
    return cores;
#elif defined(__APPLE__)
    return macSysctlInt("hw.physicalcpu");
#else
    (void)0;
    return 0;  // caller falls back to logical threads
#endif
}

GpuProfile probeGpu() {
#if defined(GLM_ENABLE_CUDA) && GLM_ENABLE_CUDA
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) == cudaSuccess && deviceCount > 0) {
        cudaDeviceProp props{};
        if (cudaGetDeviceProperties(&props, 0) == cudaSuccess) {
            GpuProfile g;
            g.backend = GpuBackend::Cuda;
            g.name = props.name ? props.name : "NVIDIA CUDA device";
            g.memoryBytes = props.totalGlobalMem;
            return g;
        }
    }
#endif
#if defined(__APPLE__)
    if (macIsAppleSilicon()) {
        GpuProfile g;
        g.backend = GpuBackend::Mps;
        g.name = macSysctlString("machdep.cpu.brand_string");
        if (g.name.empty()) g.name = "Apple Silicon (MPS)";
        g.memoryBytes = totalRamBytes();  // unified memory: GPU shares RAM
        return g;
    }
#endif
    return {};
}

} // namespace

// ---- Public detection ----

SystemProfile detectSystemProfile() {
    SystemProfile p;
#if defined(_WIN32)
    p.os = PlatformOS::Windows;
    winVersionInfo(p.osMajor, p.osMinor, p.osBuild);
    p.osLabel = "Windows " + (p.osMajor >= 10 ? std::string("10/11 (build ") : std::string("(build ")) +
                 std::to_string(p.osBuild) + ")";
#elif defined(__APPLE__)
    p.os = PlatformOS::macOS;
    p.osLabel = "macOS " + macSysctlString("kern.osproductversion");
    p.osLabel += macIsAppleSilicon() ? " (Apple Silicon)" : " (Intel)";
#else
    p.os = PlatformOS::Linux;
    p.osLabel = "Linux";
#endif

    p.ramBytes = totalRamBytes();
    p.logicalThreads = std::max(1, logicalThreads());
    p.physicalCores = physicalCores();
    if (p.physicalCores <= 0) p.physicalCores = p.logicalThreads;
    p.gpu = probeGpu();
    return p;
}

// ---- Environment overrides ----

namespace {
bool parseMbEnv(const char* name, uint64_t& out) {
    const char* v = std::getenv(name);
    if (!v || !*v) return false;
    long long mb = std::atoll(v);
    if (mb <= 0) return false;
    out = uint64_t(mb) * kMiB;
    return true;
}
bool parseIntEnv(const char* name, int& out) {
    const char* v = std::getenv(name);
    if (!v || !*v) return false;
    int n = std::atoi(v);
    if (n <= 0) return false;
    out = n;
    return true;
}
} // namespace

RuntimeOverrides readEnvOverrides() {
    RuntimeOverrides o;
    if (const char* v = std::getenv("GLM_GPU")) {
        std::string s(v);
        if (s == "auto") { o.gpuSet = true; o.gpuRequest = BackendRequest::Auto; }
        else if (s == "cpu") { o.gpuSet = true; o.gpuRequest = BackendRequest::Cpu; }
        else if (s == "cuda") { o.gpuSet = true; o.gpuRequest = BackendRequest::Cuda; }
        else if (s == "mps") { o.gpuSet = true; o.gpuRequest = BackendRequest::Mps; }
    }
    o.lruSet = parseMbEnv("GLM_LRU_MB", o.lruBytes);
    o.ramSet = parseMbEnv("GLM_RAM_MB", o.ramBytes);
    o.vramSet = parseMbEnv("GLM_VRAM_MB", o.vramBytes);
    o.iocpSet = parseIntEnv("GLM_IOCP_WORKERS", o.iocpWorkers);
    o.threadsSet = parseIntEnv("GLM_THREADS", o.computeThreads);
    return o;
}

// ---- Pure derivation ----

GpuBackend resolveGpuBackend(const SystemProfile& profile, BackendRequest request) {
    switch (request) {
        case BackendRequest::Cpu: return GpuBackend::None;
        case BackendRequest::Cuda:
            return profile.gpu.backend == GpuBackend::Cuda ? GpuBackend::Cuda : GpuBackend::None;
        case BackendRequest::Mps:
            return profile.gpu.backend == GpuBackend::Mps ? GpuBackend::Mps : GpuBackend::None;
        case BackendRequest::Auto:
        default:
            return profile.gpu.backend;
    }
}

AdaptiveConfig computeAdaptiveConfig(const SystemProfile& profile,
                                     const RuntimeOverrides& overrides) {
    AdaptiveConfig c;
    c.os = profile.os;
    c.osLabel = profile.osLabel;

    uint64_t ram = profile.ramBytes;
    if (overrides.ramSet && overrides.ramBytes > 0) {
        ram = std::max<uint64_t>(overrides.ramBytes, 2ULL * kGiB);
        c.hasOverrides = true;
    } else {
        ram = std::max<uint64_t>(ram, 2ULL * kGiB);  // floor for pathological probes
    }
    c.ramBytes = ram;

    c.gpu = resolveGpuBackend(profile, overrides.gpuRequest);
    if (overrides.gpuSet && overrides.gpuRequest != BackendRequest::Auto) c.hasOverrides = true;

    uint64_t vram = 0;
    if (profile.gpu.memoryBytes > 0) vram = profile.gpu.memoryBytes - gpuReserveBytes(profile.gpu.memoryBytes);
    if (overrides.vramSet) {
        vram = overrides.vramBytes;
        c.hasOverrides = true;
    }
    const bool unified = c.gpu == GpuBackend::Mps;

    uint64_t lru = 0;
    if (overrides.lruSet) {
        lru = overrides.lruBytes;
        c.hasOverrides = true;
    } else {
        // Route-window sizing (O1, doc/design.md 17.2): one forward touches ~27-30 GB
        // of distinct routed experts + shared projections. The former RAM/4 window
        // (~16 GB on a 64 GB host) was smaller than a single forward's footprint, so
        // it thrashed *within* a forward and every token re-streamed the expert set
        // (measured 21% LRU hit, 69.5 GiB/token -- see doc/design.md 17.1). Size the
        // window from RAM so the whole per-forward footprint stays resident and
        // carries over across tokens. CPU-only: min(RAM*3/5, 40 GiB); unified-memory
        // (MPS) hosts can afford a larger share.
        uint64_t cap = unified ? 96ULL * kGiB : 40ULL * kGiB;
        uint64_t share = unified ? 3 * ram / 4 : 3 * ram / 5;
        lru = std::max<uint64_t>(256ULL * kMiB, std::min<uint64_t>(share, cap));
    }
    const uint64_t reserve = osReserveBytes(ram);
    c.lruBytes = reserve >= ram ? lru : std::min<uint64_t>(lru, ram - reserve);
    c.lruBytes = std::max<uint64_t>(256ULL * kMiB, c.lruBytes);

    // Placement RAM budget: 80% of RAM or 1.5x the LRU window, never below the LRU,
    // never above RAM minus the OS/Python reserve. Informational cap only (the LRU
    // is the actual memory consumer).
    uint64_t budget = std::max<uint64_t>(4 * ram / 5, c.lruBytes + c.lruBytes / 2);
    c.ramBudgetBytes = reserve >= ram ? c.lruBytes : std::min<uint64_t>(budget, ram - reserve);
    c.ramBudgetBytes = std::max<uint64_t>(c.ramBudgetBytes, c.lruBytes);

    // VRAM budget: dedicated VRAM minus driver reserve (CUDA), or half the
    // unified memory (MPS). 0 when running pure CPU.
    if (c.gpu == GpuBackend::Cuda) {
        c.vramBudgetBytes = vram;
    } else if (unified) {
        c.vramBudgetBytes = std::min<uint64_t>(vram / 2, 64ULL * kGiB);
    } else {
        c.vramBudgetBytes = 0;
    }
    c.vramBytes = vram;

    int threads = profile.logicalThreads;
    if (overrides.threadsSet) {
        threads = overrides.computeThreads;
        c.hasOverrides = true;
    }
    c.computeThreads = std::max<int>(1, std::min<int>(threads, 512));

    if (overrides.iocpSet) {
        c.iocpWorkers = std::max<int>(1, std::min<int>(overrides.iocpWorkers, 16));
        c.hasOverrides = true;
    } else {
        // Scale disk-read parallelism with host core count (1..8 workers).
        c.iocpWorkers = std::max<int>(1, std::min<int>(8, std::max<int>(2, c.computeThreads / 4)));
    }

    c.gpuName = c.gpu == profile.gpu.backend ? profile.gpu.name : std::string("none");
    return c;
}

void logAdaptiveConfig(const AdaptiveConfig& c) {
    GLM_LOG_INFO("Adaptive: LRU " + std::to_string(c.lruBytes / kMiB) +
                 " MB, RAM budget " + std::to_string(c.ramBudgetBytes / kMiB) +
                 " MB, VRAM budget " + std::to_string(c.vramBudgetBytes / kMiB) +
                 " MB, IOCP workers " + std::to_string(c.iocpWorkers) +
                 ", compute threads " + std::to_string(c.computeThreads) +
                 (c.hasOverrides ? " (overrides active)" : ""));
}

} // namespace glm