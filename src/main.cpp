// GLM-5.2 local full inference engine - entry
// Usage:
//   glm.exe --model D:\glm-5.2-bf16 --prompt "hello"
//   glm.exe --model D:\glm-5.2-bf16 --prompt "hello" --max-tokens 8 --python "C:\...\python.exe"
//   glm.exe --model D:\glm-5.2-bf16 --prompt "hello" --temperature 0.7 --top-p 0.9 --top-k 40

#include "compute/cpu_kernels.h"
#include "compute/cuda_backend.h"
#include "compute/sampler.h"
#include "engine/inference_stats.h"
#include "engine/scheduler.h"
#include "model/config.h"
#include "model/glm_forward.h"
#include "model/weight_index.h"
#include "runtime/runtime_config.h"
#include "self_test.h"
#include "tokenizer/python_tokenizer.h"
#include "tokenizer/native_tokenizer.h"
#include "utils/logger.h"
#include "utils/json_util.h"
#include "utils/timer.h"
#include "version.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>
#endif

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

#if defined(_WIN32)
static std::string wcharToUtf8(const wchar_t* wstr) {
    if (!wstr) return {};
    int len = WideCharToMultiByte(CP_UTF8, 0, wstr, -1, nullptr, 0, nullptr, nullptr);
    if (len <= 0) return {};
    std::string result(len - 1, '\0');
    WideCharToMultiByte(CP_UTF8, 0, wstr, -1, result.data(), len, nullptr, nullptr);
    return result;
}
#endif

using namespace glm;

#if defined(_WIN32)
// Global crash handler: print a diagnostic instead of silently dying
static LONG WINAPI crashHandler(EXCEPTION_POINTERS* ep) {
    if (!ep) return EXCEPTION_CONTINUE_SEARCH;
    DWORD code = ep->ExceptionRecord->ExceptionCode;
    std::fprintf(stderr, "\n[CRASH] Unhandled exception during inference! Code=0x%08lx\n",
                 static_cast<unsigned long>(code));
    if (code == EXCEPTION_ACCESS_VIOLATION && ep->ExceptionRecord->NumberParameters >= 2) {
        ULONG_PTR addr = ep->ExceptionRecord->ExceptionInformation[1];
        std::fprintf(stderr, "[CRASH] Access violation at address 0x%llx\n",
                     static_cast<unsigned long long>(addr));
    }
    std::fprintf(stderr,
        "[CRASH] This likely indicates a weight tensor location error, KV cache "
        "bound overrun, or invalid expert routing. Check EXCEPTION_DEBUG build for details.\n");
    return EXCEPTION_EXECUTE_HANDLER;
}
#endif

static void printUsage() {
    std::fprintf(stderr,
        "GLM-5.2 local full inference engine v%s\n"
        "MLA Attention + MoE (DeepSeek-V3 style, BF16 no quantization)\n\n"
        "Usage: glm.exe [--model <dir>] [options]\n\n"
        "Options:\n"
        "  --model <dir>        Model directory (contains config.json + *.safetensors)\n"
        "  --prompt <text>      Input prompt (user message)\n"
        "  --raw                Skip chat template, encode prompt directly\n"
        "  --prompt-tokens <csv> Raw comma-separated input token ids (headless/CI; no tokenizer)\n"
        "  --max-tokens <n>     Max generated tokens (default -1 = unlimited, stop on EOS)\n"
        "  --temperature <f>    Sampling temperature (default 0.0 = greedy)\n"
        "  --top-p <f>          Top-p (nucleus) sampling threshold (default 1.0 = disabled)\n"
        "  --top-k <n>          Top-k sampling limit (default 0 = disabled)\n"
        "  --min-p <f>          Min-p filter: keep tokens within f x p(max) (default 0 = disabled)\n"
        "  --typical-p <f>      Locally-typical filter mass threshold (default 0 = disabled)\n"
        "  --repetition-penalty <f>  Repetition penalty on the last 64 generated tokens (default 0)\n"
        "  --frequency-penalty <f>   Frequency penalty: subtract f per occurrence in the window (default 0)\n"
        "  --presence-penalty <f>    Presence penalty: subtract f per distinct token in the window (default 0)\n"
        "  --seed <n>           Random seed (default: random)\n"
        "  --router-prefetch <n> Router-probability extra prefetch depth (default 0 = off)\n"
        "  --prob-resident     Promote ~max-probability experts to residency pins\n"
        "  --predictor ema     EMA popularity predictor for lookahead prefetch + LRU\n"
        "                      soft-boost (item 4/14; default lookahead reuse)\n"
        "  --ema-alpha <f>     EMA smoothing (0,1]; default 0.1\n"
        "  --mtp                MTP/nextn predict head (hard gate: refuses to start with\n"
        "                      a silent base-LM-head fallback; requires config-declared\n"
        "                      MTP layers with tensors present AND verified forward math)\n"
        "  --python <path>      Python interpreter path\n"
        "  --script <path>      Tokenizer script path (default: auto-detect)\n"
        "  --tokenizer <python|native>  Tokenizer backend (default: python subprocess)\n"
        "  --dump-tokens <text> Encode text with the active tokenizer and exit\n"
        "  --gpu <auto|cpu|cuda|mps>  Compute backend (default: auto-detect)\n"
        "  --no-gpu             Force CPU compute (same as --gpu cpu)\n"
        "  --gpu-experts <n>    Offload routed expert FFN to the GPU with a VRAM-\n"
        "                       resident window of n experts (opt-in; default off)\n"
        "  --gpu-expert-depth <d>  Expert staging depth: H2D copies d experts ahead\n"
        "                       of compute (default 2; only with --gpu-experts)\n"
        "  --verbose            Enable debug logging\n"
        "  --version            Print version and exit\n"
        "  --self-test          Run functional self-tests (no model required) and exit\n"
        "  --cuda-self-test     Run the GPU expert FFN parity self-test and exit\n"
        "  --check-weights      Validate model directory structurally and exit\n"
        "  --help               Show help\n",
        GLM_VERSION_STRING);
}

struct Args {
    std::string modelDir;
    std::string prompt;
    bool rawEncode = false;
    std::vector<int> promptTokens;  // --prompt-tokens: raw input ids (headless/CI, no tokenizer)
    int maxTokens = -1;
    float temperature = 0.0f;
    float topP = 1.0f;
    int topK = 0;
    float minP = 0.0f;
    float typicalP = 0.0f;
    float repetitionPenalty = 0.0f;
    float frequencyPenalty = 0.0f;
    float presencePenalty = 0.0f;
    int routerPrefetchExtra = 0;
    bool probPriority = false;
    bool emaPredictor = false;
    float emaAlpha = 0.1f;
    bool mtp = false;
    unsigned seed = 0;
    bool useSeed = false;
    std::string pythonExe = "python";
    std::string scriptPath;  // optional: explicit tokenizer script path
    bool nativeTokenizer = false;
    std::string dumpTokens;
    bool useGpu = true;
    BackendRequest gpuRequest = BackendRequest::Auto;
    int gpuExpertCapacity = 0;
    int gpuExpertDepth = 2;
    bool verbose = false;
    bool showVersion = false;
    bool selfTest = false;
    bool checkWeights = false;
    bool cudaSelfTest = false;
};

static bool parseArgs(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") { printUsage(); return false; }
        else if (a == "--model" && i + 1 < argc) { args.modelDir = argv[++i]; }
        else if (a == "--prompt" && i + 1 < argc) { args.prompt = argv[++i]; }
        else if (a == "--raw") { args.rawEncode = true; }
        else if (a == "--prompt-tokens" && i + 1 < argc) {
            // Comma-separated raw token ids, e.g. "3,7,12". Used by the
            // headless/CI end-to-end test (no tokenizer required).
            std::string csv = argv[++i];
            std::string cur;
            for (char ch : csv) {
                if (ch == ',' || ch == ' ') {
                    if (!cur.empty()) { args.promptTokens.push_back(std::atoi(cur.c_str())); cur.clear(); }
                } else {
                    cur += ch;
                }
            }
            if (!cur.empty()) args.promptTokens.push_back(std::atoi(cur.c_str()));
        }
        else if (a == "--max-tokens" && i + 1 < argc) { args.maxTokens = std::atoi(argv[++i]); }
        else if (a == "--temperature" && i + 1 < argc) { args.temperature = static_cast<float>(std::atof(argv[++i])); }
        else if (a == "--top-p" && i + 1 < argc) { args.topP = static_cast<float>(std::atof(argv[++i])); }
        else if (a == "--top-k" && i + 1 < argc) { args.topK = std::atoi(argv[++i]); }
        else if (a == "--min-p" && i + 1 < argc) { args.minP = static_cast<float>(std::atof(argv[++i])); }
        else if (a == "--typical-p" && i + 1 < argc) { args.typicalP = static_cast<float>(std::atof(argv[++i])); }
        else if (a == "--repetition-penalty" && i + 1 < argc) { args.repetitionPenalty = static_cast<float>(std::atof(argv[++i])); }
        else if (a == "--frequency-penalty" && i + 1 < argc) { args.frequencyPenalty = static_cast<float>(std::atof(argv[++i])); }
        else if (a == "--presence-penalty" && i + 1 < argc) { args.presencePenalty = static_cast<float>(std::atof(argv[++i])); }
        else if (a == "--router-prefetch" && i + 1 < argc) { args.routerPrefetchExtra = std::atoi(argv[++i]); }
        else if (a == "--prob-resident") { args.probPriority = true; }
        else if (a == "--predictor" && i + 1 < argc) { std::string p = argv[++i]; args.emaPredictor = (p == "ema"); }
        else if (a == "--ema-alpha" && i + 1 < argc) { args.emaAlpha = static_cast<float>(std::atof(argv[++i])); }
        else if (a == "--mtp") { args.mtp = true; }
        else if (a == "--seed" && i + 1 < argc) { args.seed = std::atoi(argv[++i]); args.useSeed = true; }
        else if (a == "--python" && i + 1 < argc) { args.pythonExe = argv[++i]; }
        else if (a == "--script" && i + 1 < argc) { args.scriptPath = argv[++i]; }
        else if (a == "--tokenizer" && i + 1 < argc) {
            std::string mode = argv[++i];
            if (mode == "native") args.nativeTokenizer = true;
            else if (mode == "python") args.nativeTokenizer = false;
            else { std::fprintf(stderr, "Unknown tokenizer mode: %s\n", mode.c_str()); return false; }
        }
        else if (a == "--dump-tokens" && i + 1 < argc) { args.dumpTokens = argv[++i]; }
        else if (a == "--no-gpu") { args.useGpu = false; }
        else if (a == "--gpu-experts" && i + 1 < argc) {
            const int v = std::atoi(argv[++i]);
            args.gpuExpertCapacity = v > 0 ? v : 0;
            if (args.gpuExpertCapacity == 0) GLM_LOG_WARN("--gpu-experts 0 disables GPU expert FFN");
        }
        else if (a == "--gpu-expert-depth" && i + 1 < argc) {
            const int v = std::atoi(argv[++i]);
            args.gpuExpertDepth = v >= 1 ? v : 1;
        }
        else if (a == "--gpu" && i + 1 < argc) {
            std::string g = argv[++i];
            if (g == "auto") args.gpuRequest = BackendRequest::Auto;
            else if (g == "cpu" || g == "cpu-only") args.gpuRequest = BackendRequest::Cpu;
            else if (g == "cuda") args.gpuRequest = BackendRequest::Cuda;
            else if (g == "mps") args.gpuRequest = BackendRequest::Mps;
            else GLM_LOG_WARN("Unknown --gpu value '" + g + "' (auto|cpu|cuda|mps)");
        }
        else if (a == "--verbose") { args.verbose = true; }
        else if (a == "--version") { args.showVersion = true; }
        else if (a == "--self-test") { args.selfTest = true; }
        else if (a == "--cuda-self-test") { args.cudaSelfTest = true; }
        else if (a == "--check-weights") { args.checkWeights = true; }
        else { GLM_LOG_WARN("Unknown argument: " + a); }
    }
    // --version / --self-test / --cuda-self-test / --check-weights do not require a model
    if (args.showVersion || args.selfTest || args.cudaSelfTest || args.checkWeights) return true;
    if (args.modelDir.empty()) {
        GLM_LOG_ERROR("--model not specified");
        printUsage();
        return false;
    }
    return true;
}

static void setupConsole() {
#if defined(_WIN32)
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
}

// Directory containing the running executable (cross-platform).
static std::string exeDirPath() {
#if defined(_WIN32)
    char path[MAX_PATH] = {};
    GetModuleFileNameA(nullptr, path, MAX_PATH);
    return std::filesystem::path(path).parent_path().string();
#elif defined(__APPLE__)
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::string buf(size, '\0');
    _NSGetExecutablePath(buf.data(), &size);
    return std::filesystem::path(buf).parent_path().string();
#else
    try {
        return std::filesystem::read_symlink("/proc/self/exe").parent_path().string();
    } catch (...) {
        return ".";
    }
#endif
}

// ---------- Sampling ----------
// The sampling pipeline (temperature -> top-k -> top-p -> categorical) lives in
// compute/sampler.h and is used below when --temperature > 0. Greedy
// (temperature <= 0) is a pure argmax so goldens stay deterministic.

int main(int argc, char** argv) {
    setupConsole();

#if defined(_WIN32)
    SetUnhandledExceptionFilter(crashHandler);
#endif

#if defined(_WIN32)
    int wArgc = 0;
    LPWSTR* wArgv = CommandLineToArgvW(GetCommandLineW(), &wArgc);
    std::vector<std::string> utf8Args;
    std::vector<char*> utf8ArgPtrs;
    if (wArgv) {
        utf8Args.reserve(wArgc);
        utf8ArgPtrs.reserve(wArgc);
        for (int i = 0; i < wArgc; ++i) {
            utf8Args.push_back(wcharToUtf8(wArgv[i]));
            utf8ArgPtrs.push_back(utf8Args.back().data());
        }
        LocalFree(wArgv);
        argc = wArgc;
        argv = utf8ArgPtrs.data();
    }
#endif

    Args args;
    if (!parseArgs(argc, argv, args)) return 1;

    if (args.showVersion) {
        std::printf("%s v%s\n", GLM_PROJECT_NAME, GLM_VERSION_STRING);
        return 0;
    }

    if (args.selfTest) {
        int fails = runSelfTests(args.verbose, "tests/fixtures");
        return fails == 0 ? 0 : 1;
    }

    // GPU expert FFN device parity self-test (--cuda-self-test; no model needed).
    if (args.cudaSelfTest) {
        if (cudaInit()) {
            const bool ok = cudaExpertSelfTest();
            cudaShutdown();
            if (ok) {
                GLM_LOG_INFO("CUDA expert FFN parity self-test: PASS");
                return 0;
            }
            GLM_LOG_ERROR("CUDA expert FFN parity self-test: FAIL");
            return 1;
        }
        GLM_LOG_WARN("--cuda-self-test: no CUDA device available, skipping");
        return 0;
    }

    if (args.checkWeights) {
        if (args.modelDir.empty()) {
            GLM_LOG_ERROR("--model not specified for --check-weights");
            return 1;
        }
        std::string report;
        int fails = validateWeights(args.modelDir, report);
        std::fprintf(stderr, "%s", report.c_str());
        return fails == 0 ? 0 : 1;
    }

    if (args.verbose) Logger::setLevel(LogLevel::Debug);

    // ---- Adaptive runtime configuration (design doc §12) ----
    // Probe the host (OS, RAM, cores, GPU) and derive per-platform inference
    // budgets. CLI flags override env overrides, which override auto-detection.
    SystemProfile profile = detectSystemProfile();
    RuntimeOverrides overrides = readEnvOverrides();
    if (!args.useGpu) {
        overrides.gpuSet = true;
        overrides.gpuRequest = BackendRequest::Cpu;
    } else if (args.gpuRequest != BackendRequest::Auto) {
        overrides.gpuSet = true;
        overrides.gpuRequest = args.gpuRequest;
    }
    AdaptiveConfig ac = computeAdaptiveConfig(profile, overrides);

    GLM_LOG_INFO("=== " + std::string(GLM_PROJECT_NAME) +
                 " local full inference engine v" + GLM_VERSION_STRING + " ===");
    GLM_LOG_INFO("Platform: " + ac.osLabel + ", RAM " +
                 std::to_string(ac.ramBytes / (1024 * 1024)) +
                 " MB, " + std::to_string(profile.physicalCores) + "/" +
                 std::to_string(profile.logicalThreads) + " physical/logical cores");
    if (ac.gpu != GpuBackend::None) {
        GLM_LOG_INFO("GPU: " + ac.gpuName + " (" +
                     std::to_string(ac.vramBytes / (1024 * 1024)) + " MB)");
        if (ac.gpu == GpuBackend::Mps) {
            GLM_LOG_INFO("MPS detected: Metal offload unavailable in this build; kernels run on CPU");
        }
    } else {
        GLM_LOG_INFO("GPU: none (CPU compute)");
    }
    logAdaptiveConfig(ac);

    GLM_LOG_INFO("CPU threads: " + std::to_string(cpuThreadCount()));
    GLM_LOG_INFO("Model directory: " + args.modelDir);

    if (args.temperature > 0.0f) {
        GLM_LOG_INFO("Sampling: temperature=" + std::to_string(args.temperature) +
                     ", top-p=" + std::to_string(args.topP) +
                     ", top-k=" + std::to_string(args.topK) +
                     ", min-p=" + std::to_string(args.minP) +
                     ", typical-p=" + std::to_string(args.typicalP) +
                     ", repetition-penalty=" + std::to_string(args.repetitionPenalty) +
                     ", frequency-penalty=" + std::to_string(args.frequencyPenalty) +
                     ", presence-penalty=" + std::to_string(args.presencePenalty));
    } else {
        GLM_LOG_INFO("Sampling: greedy (temperature=0)");
    }

    // 1. Build weight index
    Timer indexTimer;
    WeightIndex weightIndex;
    if (!weightIndex.build(args.modelDir)) {
        GLM_LOG_ERROR("Weight index build failed");
        return 1;
    }
    weightIndex.config.printSummary();
    weightIndex.printStats();
    GLM_LOG_INFO("Index build time: " + std::to_string(indexTimer.elapsedSec()) + " s");

    // MTP / multi-token-prediction gating (item 8). GLM-5.2 declares MTP through
    // num_nextn_predict_layers, but the checkpoint ships the nextn_predict_layers.*
    // and nextn_predict_head weights separately. Per item 15 there is no silent
    // degradation: a --mtp request that cannot be honoured is a hard error.
    if (args.mtp) {
        if (weightIndex.config.numMtpModules <= 0) {
            GLM_LOG_ERROR("--mtp requested, but the checkpoint config declares no MTP "
                          "modules (num_nextn_predict_layers / num_mtp_modules absent or 0). "
                          "Remove --mtp and retry.");
            return 1;
        }
        if (!weightIndex.hasMtpTensors) {
            GLM_LOG_ERROR("--mtp requested, config declares " +
                          std::to_string(weightIndex.config.numMtpModules) +
                          " nextn predict layer(s), but the checkpoint ships no "
                          "nextn_predict_layers.* / mtp_layers.* tensors, so MTP cannot "
                          "run. Refusing to start (a silent base-LM-head fallback is "
                          "forbidden, doc/design.md item 15). Add the MTP weight files "
                          "or remove --mtp.");
            return 1;
        }
        GLM_LOG_ERROR("--mtp: MTP weights detected (" +
                      std::to_string(weightIndex.mtpTensorLayers) + " layers" +
                      (weightIndex.hasMtpHead ? " + head" : ", no head") +
                      "), but the MTP forward path is not yet validated against a "
                      "reference; refusing to run unverified math.");
        return 1;
    }
    if (weightIndex.config.numMtpModules > 0) {
        GLM_LOG_INFO("Checkpoint config carries MTP heads (nextn_predict_layers=" +
                     std::to_string(weightIndex.config.numMtpModules) +
                     (weightIndex.hasMtpTensors
                          ? ", tensors present: " + std::to_string(weightIndex.mtpTensorLayers) + " layer(s)"
                          : ", but no MTP tensors in the index; --mtp will not start") +
                     "; sampling runs on the base LM head.");
    }

    // 2. CUDA backend (optional; only attempted when the adaptive profile
    //    resolved to CUDA and the user did not force CPU).
    if (ac.gpu == GpuBackend::Cuda) {
#if GLM_ENABLE_CUDA
        if (cudaInit()) {
            GLM_LOG_INFO("GPU: " + cudaDeviceName() + " (" +
                         std::to_string(cudaDeviceMemoryMB()) + " MB)");
        } else {
            GLM_LOG_WARN("CUDA initialization failed, using pure CPU mode");
        }
#else
        GLM_LOG_WARN("CUDA backend not compiled (enable with -DENABLE_CUDA=ON); using CPU");
#endif
    } else if (args.useGpu && overrides.gpuSet &&
               overrides.gpuRequest == BackendRequest::Cuda) {
        GLM_LOG_WARN("Requested CUDA backend but no NVIDIA device is available");
    } else if (args.useGpu && overrides.gpuSet &&
               overrides.gpuRequest == BackendRequest::Mps) {
        GLM_LOG_WARN("Requested MPS backend but this host has no Apple Silicon GPU");
    }

    // 3. Tokenizer. Default is the Python subprocess (byte-for-byte parity with
    //    the reference transformers implementation). --tokenizer native enables
    //    the in-process BPE tokenizer (tokenizer.json, opt-in). Skipped entirely
    //    when --prompt-tokens provides raw input ids (headless/CI path).
    PythonTokenizer tokenizer;
    NativeBpeTokenizer nativeToken;
    const bool useTokenizer = args.promptTokens.empty();
    if (args.promptTokens.empty()) {
        if (const char* pyEnv = std::getenv("GLM_PYTHON")) {
            if (*pyEnv) args.pythonExe = pyEnv;
        }
    }
    if (useTokenizer && args.nativeTokenizer) {
        const std::string tj = args.modelDir + "/tokenizer.json";
        const std::string tc = args.modelDir + "/tokenizer_config.json";
        if (!nativeToken.load(tj, tc)) {
            GLM_LOG_ERROR("Native tokenizer load failed (--tokenizer native): " + tj);
            return 1;
        }
    } else if (useTokenizer) {
        // Resolve script path: --script > executable dir > CWD
        std::string scriptPath;
        if (!args.promptTokens.empty()) {
            scriptPath = "";
        } else if (!args.scriptPath.empty() && std::filesystem::exists(args.scriptPath)) {
            scriptPath = std::filesystem::canonical(args.scriptPath).string();
        } else {
            // Try next to the executable
            std::filesystem::path exeDir = exeDirPath();
            std::filesystem::path candidate1 = exeDir / ".." / ".." / "tools" / "tokenizer_server.py"; // build/Release -> project root
            std::filesystem::path candidate2 = exeDir / ".." / "tools" / "tokenizer_server.py";         // build -> project root
            std::filesystem::path candidate3 = exeDir / "tools" / "tokenizer_server.py";                // exe dir
            std::filesystem::path candidate4 = std::filesystem::current_path() / "tools" / "tokenizer_server.py";

            if (std::filesystem::exists(candidate1)) {
                scriptPath = std::filesystem::canonical(candidate1).string();
            } else if (std::filesystem::exists(candidate2)) {
                scriptPath = std::filesystem::canonical(candidate2).string();
            } else if (std::filesystem::exists(candidate3)) {
                scriptPath = std::filesystem::canonical(candidate3).string();
            } else if (std::filesystem::exists(candidate4)) {
                scriptPath = std::filesystem::canonical(candidate4).string();
            } else {
                scriptPath = (std::filesystem::current_path() / "tools" / "tokenizer_server.py").string();
            }
        }
        GLM_LOG_INFO("Starting Python tokenizer: " + args.pythonExe);
        GLM_LOG_INFO("Tokenizer script: " + scriptPath);
        if (!tokenizer.start(args.pythonExe, scriptPath, args.modelDir)) {
            GLM_LOG_ERROR("Python tokenizer start failed");
            return 1;
        }
    } else {
        GLM_LOG_INFO("Raw token ids provided (--prompt-tokens); tokenizer skipped");
    }

    // Generic encode/decode entry points (python subprocess or native BPE).
    auto encodeText = [&](const std::string& text, bool special) -> std::vector<int> {
        if (args.nativeTokenizer) return nativeToken.encode(text, special);
        return tokenizer.encode(text, special);
    };
    auto decodeIds = [&](const std::vector<int>& ids) -> std::string {
        if (args.nativeTokenizer) return nativeToken.decode(ids, true);
        return tokenizer.decode(ids, true);
    };
    auto vocabSizeOf = [&]() -> int {
        if (args.nativeTokenizer) return nativeToken.vocabSize();
        return tokenizer.vocabSize();
    };
    auto eosOf = [&]() -> int {
        if (args.nativeTokenizer) return nativeToken.eosTokenId();
        return tokenizer.eosTokenId();
    };

    // Hidden debug/CI helper: encode one text with the active tokenizer and exit.
    // Used by tests/test_native_tokenizer.py to compare the native BPE backend
    // against the reference transformers tokenizer (real model directory).
    if (!args.dumpTokens.empty()) {
        std::vector<int> dumpIds = encodeText(args.dumpTokens, false);
        std::string csv;
        for (size_t i = 0; i < dumpIds.size(); ++i) {
            if (i) csv += ",";
            csv += std::to_string(dumpIds[i]);
        }
        GLM_LOG_INFO("TOKEN: " + csv);
        GLM_LOG_INFO("TOKEN_DEC: " + jsonEscapeString(decodeIds(dumpIds)));
        return 0;
    }

    // 4. Forward engine + scheduler (LRU/IOCP expert pipeline)
    // Scheduler is declared first so it outlives the forwarder (which holds its
    // pointer). Shutdown is explicit at the end of generation.
    Scheduler scheduler;
    if (!scheduler.init(weightIndex, ac.lruBytes, ac.iocpWorkers,
                        ac.ramBudgetBytes, ac.vramBudgetBytes)) {
        GLM_LOG_WARN("Scheduler init failed, running without expert prefetch");
    }
    GLM_LOG_INFO("Scheduler ready: LRU " + std::to_string(ac.lruBytes / (1024 * 1024)) +
                 " MB, IOCP workers " + std::to_string(ac.iocpWorkers));
    scheduler.setRouterPrefetch(args.routerPrefetchExtra, args.probPriority);
    if (args.emaPredictor) scheduler.setPopularityPredictor(args.emaAlpha, true);

    GLMForward forwarder(weightIndex);
    if (!forwarder.init()) {
        GLM_LOG_ERROR("Forward engine initialization failed");
        return 1;
    }
    forwarder.attachScheduler(&scheduler);
    if (args.gpuExpertCapacity > 0) {
        forwarder.enableGpuExperts(args.gpuExpertCapacity, args.gpuExpertDepth);
        if (!forwarder.gpuExpertsEnabled()) {
            // Item 15: a requested, unavailable offload is a hard error, never a
            // silent degradation. The user can re-run with --gpu-experts 0.
            GLM_LOG_ERROR("--gpu-experts <n> requested but the CUDA expert FFN is "
                          "not available. Re-run without it (or with --gpu-experts 0).");
            return 1;
        }
    }

    // 5. Random number generator for sampling
    std::mt19937 rng;
    if (args.useSeed) {
        rng.seed(args.seed);
    } else {
        std::random_device rd;
        rng.seed(rd());
    }

    // 6. Generation
    const bool hasPromptTokens = !args.promptTokens.empty();
    if (!args.prompt.empty() || hasPromptTokens) {
        std::vector<int> inputIds;
        if (hasPromptTokens) {
            inputIds = args.promptTokens;
        } else if (args.rawEncode) {
            inputIds = encodeText(args.prompt, true);
        } else if (args.nativeTokenizer) {
            GLM_LOG_WARN("Native tokenizer has no chat template; encoding prompt directly");
            inputIds = encodeText(args.prompt, true);
        } else {
            // The prompt must be JSON-escaped so quotes/backslashes/newlines in
            // the user message cannot break the chat-template JSON payload.
            std::string messagesJson = "[{\"role\":\"user\",\"content\":\"" +
                                       jsonEscapeString(args.prompt) + "\"}]";
            inputIds = tokenizer.applyChatTemplate(messagesJson, true);
        }

        // Headless raw-token input cannot stop on EOS (no tokenizer); require an
        // explicit bound so the loop is always finite and deterministic.
        int maxTokens = args.maxTokens;
        if (hasPromptTokens && maxTokens < 0) maxTokens = 8;
        bool unlimited = !hasPromptTokens && (args.maxTokens < 0);
        GLM_LOG_INFO("Input: " + (hasPromptTokens ? "raw " + std::to_string(inputIds.size()) + " tokens"
                                                  : args.prompt));
        GLM_LOG_INFO("Encoded: " + std::to_string(inputIds.size()) + " tokens");
        if (unlimited) {
            GLM_LOG_INFO("max-tokens = unlimited (stop on EOS)");
        } else {
            GLM_LOG_INFO("max-tokens = " + std::to_string(maxTokens));
        }

        Timer genTimer;
        const clock_t cpuStart = std::clock();
        std::vector<int> generatedIds;
        std::vector<int> allIds = inputIds;
        const bool useSampling = args.temperature > 0.0f && useTokenizer;
        const int vocabSize = useTokenizer ? vocabSizeOf() : 0;
        const int eosTok = useTokenizer ? eosOf() : -1;
        // Repetition-penalty window: last kRewind generated tokens.
        constexpr int kPenaltyWindow = 64;
        std::vector<int> recentTokens(kPenaltyWindow, -1);

        for (int t = 0; unlimited || t < maxTokens; ++t) {
            // forward() fills the full logits vector (when requested) and returns
            // the greedy argmax for the sampling fallback path.
            std::vector<float> logits;
            int nextToken = forwarder.forward(allIds, useSampling ? &logits : nullptr);
            if (nextToken < 0) {
                GLM_LOG_WARN("Generation aborted (token=" + std::to_string(nextToken) + ")");
                break;
            }

            if (useSampling) {
                nextToken = sampleFromLogits(logits.data(), vocabSize,
                                             args.temperature, args.topP, args.topK,
                                             args.minP, args.repetitionPenalty,
                                             recentTokens.data(),
                                             int(recentTokens.size()),
                                             args.typicalP, args.frequencyPenalty,
                                             args.presencePenalty,
                                             rng);
            }

            generatedIds.push_back(nextToken);
            allIds.push_back(nextToken);
            std::rotate(recentTokens.begin(), recentTokens.begin() + 1, recentTokens.end());
            recentTokens.back() = nextToken;

            if (useTokenizer && nextToken == eosTok) {
                GLM_LOG_INFO("EOS encountered, stopping generation");
                break;
            }

            double elapsed = genTimer.elapsedSec();
            if (unlimited) {
                GLM_LOG_INFO("[" + std::to_string(t + 1) + "] token=" +
                             std::to_string(nextToken) +
                             " (" + std::to_string(elapsed) + " s)");
            } else {
                GLM_LOG_INFO("[" + std::to_string(t + 1) + "/" + std::to_string(maxTokens) +
                             "] token=" + std::to_string(nextToken) +
                             " (" + std::to_string(elapsed) + " s)");
            }
        }

        double elapsed = genTimer.elapsedSec();
        GLM_LOG_INFO("=== Generation complete ===");
        GLM_LOG_INFO("Generated " + std::to_string(generatedIds.size()) + " tokens, time " +
                     std::to_string(elapsed) + " s");
        if (elapsed > 0 && !generatedIds.empty()) {
            GLM_LOG_INFO("Speed: " + std::to_string(generatedIds.size() / elapsed) + " tok/s");
        }

        // Emit runtime telemetry for scripts/benchmark.py (item: benchmark
        // upgrade): a single machine-parseable JSON line.
        InferenceStats::instance().prefetchWasted += scheduler.lru().evictedPrefetchedUnused();
        const uint64_t cpuNs = uint64_t(double(std::clock() - cpuStart) / CLOCKS_PER_SEC * 1e9);
        GLM_LOG_INFO("STATS: " + InferenceStats::instance().toJson(cpuNs, elapsed, generatedIds.size()));

        if (useTokenizer) {
            std::string output = decodeIds(generatedIds);
            GLM_LOG_INFO("Output: " + output);
        } else {
            std::string idsStr;
            for (size_t i = 0; i < generatedIds.size(); ++i) {
                if (i) idsStr += ",";
                idsStr += std::to_string(generatedIds[i]);
            }
            GLM_LOG_INFO("Output ids: " + idsStr);
        }
    }

scheduler.shutdown();

#if GLM_ENABLE_CUDA
    if (ac.gpu == GpuBackend::Cuda) cudaShutdown();
#endif

    GLM_LOG_INFO("Done");
    return 0;
}
