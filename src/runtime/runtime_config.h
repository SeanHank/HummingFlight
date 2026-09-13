#pragma once

#include <cstdint>
#include <string>

namespace glm {

// Cross-platform adaptive runtime configuration (doc/design.md §12).
//
// Responsibilities:
//  * Probe the host: OS family + version, physical RAM, CPU cores/threads, and
//    GPU class (NVIDIA CUDA / Apple Silicon MPS / none).
//  * Derive inference budgets per platform: expert LRU bytes, PlacementDirector
//    RAM budget, VRAM budget, IOCP disk-read workers, compute threads.
//  * Honor explicit overrides (CLI/env) without ever exceeding detected RAM.

enum class PlatformOS { Windows, Linux, macOS };

// GPU classes the engine can target at runtime.
enum class GpuBackend { None, Cuda, Mps };

// User-visible backend request ("auto" probes the host).
enum class BackendRequest { Auto, Cpu, Cuda, Mps };

// Detected accelerator.
struct GpuProfile {
    GpuBackend backend = GpuBackend::None;
    std::string name;              // "NVIDIA GeForce RTX 3060", "Apple M3 Pro (MPS)", ...
    uint64_t memoryBytes = 0;      // dedicated VRAM (CUDA) or unified-memory cap (MPS)
};

// Host snapshot produced by detectSystemProfile().
struct SystemProfile {
    PlatformOS os = PlatformOS::Windows;
    std::string osLabel;           // "Windows 10/11 (build 19044)", "macOS 14.5", "Linux"
    uint64_t ramBytes = 0;         // total physical RAM
    int physicalCores = 0;         // 0 when not measurable on this host
    int logicalThreads = 0;        // >= 1
    int osMajor = 0, osMinor = 0, osBuild = 0;
    GpuProfile gpu;
};

// Explicit overrides read from GLM_* environment variables (see README §runtime).
struct RuntimeOverrides {
    bool gpuSet = false;      BackendRequest gpuRequest = BackendRequest::Auto;
    bool lruSet = false;      uint64_t lruBytes = 0;    // GLM_LRU_MB
    bool ramSet = false;      uint64_t ramBytes = 0;    // GLM_RAM_MB (fake total RAM)
    bool vramSet = false;     uint64_t vramBytes = 0;   // GLM_VRAM_MB
    bool iocpSet = false;     int iocpWorkers = 0;      // GLM_IOCP_WORKERS
    bool threadsSet = false;  int computeThreads = 0;   // GLM_THREADS
};

// Fully-resolved engine budgets for one process. Exported (not built) on every
// platform so the derivation is deterministic and unit-testable.
struct AdaptiveConfig {
    PlatformOS os = PlatformOS::Windows;
    std::string osLabel;
    uint64_t ramBytes = 0;

    uint64_t lruBytes = 0;        // expert LRU bytes (MoE + shared experts)
    uint64_t ramBudgetBytes = 0;  // PlacementDirector RAM budget
    uint64_t vramBudgetBytes = 0; // PlacementDirector VRAM budget (0 = CPU-only)
    int iocpWorkers = 0;          // async disk-read workers
    int computeThreads = 0;       // kernel thread target

    GpuBackend gpu = GpuBackend::None;  // effective backend after request resolution
    std::string gpuName;
    uint64_t vramBytes = 0;             // effective VRAM visible to the engine

    bool hasOverrides = false;          // true when any GLM_* override applied
};

// ---- Detection (compiles on Windows, macOS >= 14, Linux >= 10) ----
SystemProfile detectSystemProfile();

// ---- Environment overrides ----
RuntimeOverrides readEnvOverrides();

// ---- Pure derivation (deterministic; used by tests) ----
// Resolves the effective GPU backend from a request against the detected GPU.
GpuBackend resolveGpuBackend(const SystemProfile& profile, BackendRequest request);

// Derives all budgets from the profile and overrides (pure, no I/O).
AdaptiveConfig computeAdaptiveConfig(const SystemProfile& profile,
                                     const RuntimeOverrides& overrides);

// ---- Logging ----
void logAdaptiveConfig(const AdaptiveConfig& config);

} // namespace glm