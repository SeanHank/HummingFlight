#include "model/glm_forward.h"
#include "compute/cpu_kernels.h"
#include "compute/moe_router.h"
#include "utils/logger.h"
#include "utils/timer.h"
#include <algorithm>
#include <cmath>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(_MSC_VER)
#include <immintrin.h>
#endif

namespace glm {

// ---------- Helper: vector addition (AVX2) ----------
static void addVec(float* dst, const float* src, int n) {
#ifdef _MSC_VER
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 a = _mm256_loadu_ps(dst + i);
        __m256 b = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, _mm256_add_ps(a, b));
    }
    for (; i < n; ++i) dst[i] += src[i];
#else
    for (int i = 0; i < n; ++i) dst[i] += src[i];
#endif
}

// ---------- Helper: copy vector ----------
static void copyVec(float* dst, const float* src, int n) {
    std::memcpy(dst, src, n * sizeof(float));
}

// ---------- Helper: AVX2 F32 dot product ----------
static float dotF32(const float* a, const float* b, int n) {
#ifdef _MSC_VER
    __m256 sum = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 av = _mm256_loadu_ps(a + i);
        __m256 bv = _mm256_loadu_ps(b + i);
        sum = _mm256_fmadd_ps(av, bv, sum);
    }
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 s = _mm_add_ps(hi, lo);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float result = _mm_cvtss_f32(s);
    for (; i < n; ++i) result += a[i] * b[i];
    return result;
#else
    float result = 0.0f;
    for (int i = 0; i < n; ++i) result += a[i] * b[i];
    return result;
#endif
}

// ---------- Helper: BF16 dot product (AVX2) ----------
static float dotBf16F32(const BFloat16* a, const float* b, int n) {
#ifdef _MSC_VER
    const int n8 = n & ~7;
    __m256 sum = _mm256_setzero_ps();
    int i = 0;
    for (; i < n8; i += 8) {
        __m128i bf16_bits = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
        __m256i u32 = _mm256_cvtepu16_epi32(bf16_bits);
        __m256i f32_bits = _mm256_slli_epi32(u32, 16);
        __m256 af = _mm256_castsi256_ps(f32_bits);
        __m256 bf = _mm256_loadu_ps(b + i);
        sum = _mm256_fmadd_ps(af, bf, sum);
    }
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 s = _mm_add_ps(hi, lo);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float result = _mm_cvtss_f32(s);
    for (; i < n; ++i) result += a[i].toF32() * b[i];
    return result;
#else
    float result = 0.0f;
    for (int i = 0; i < n; ++i) result += a[i].toF32() * b[i];
    return result;
#endif
}

// ---------- RoPE (interleave mode) ----------
static void applyRoPEInterleave(float* x, int dim, float theta, int pos) {
    for (int i = 0; i < dim; i += 2) {
        float freq = 1.0f / std::pow(theta, float(i) / float(dim));
        float angle = float(pos) * freq;
        float c = std::cos(angle);
        float s = std::sin(angle);
        float x0 = x[i], x1 = x[i + 1];
        x[i]     = x0 * c - x1 * s;
        x[i + 1] = x0 * s + x1 * c;
    }
}

// ---------- GLMForward implementation ----------

bool GLMForward::init() {
    GLM_LOG_INFO("Initializing forward engine (MLA + MoE, AVX2 + OpenMP)...");
    const auto& cfg = index_->config;

    // Detect whether the DSA indexer weights exist for the layers that need them.
    dsaEnabled_ = cfg.indexHeadDim > 0 && !cfg.indexerTypes.empty();
    if (dsaEnabled_) {
        for (int layer = 0; layer < cfg.numLayers && dsaEnabled_; ++layer) {
            if (cfg.isIndexerFull(layer)) {
                const auto& lw = index_->layers[layer];
                if (lw.indexWqB.byteSize == 0 || lw.indexWk.byteSize == 0 ||
                    lw.indexWeightsProj.byteSize == 0 || lw.indexKNormW.byteSize == 0) {
                    dsaEnabled_ = false;
                }
            }
        }
    }
    GLM_LOG_INFO(dsaEnabled_ ? "DSA indexer: ENABLED" : "DSA indexer: DISABLED (weights absent)");

    // Initialize expanded MLA KV + DSA indexer key cache.
    kvCache_.init(cfg.numLayers, cfg.numAttentionHeads, cfg.qkHeadDim, cfg.vHeadDim,
                  cfg.indexHeadDim, 64); // Small initial pre-allocation, grows dynamically
    lastTopkByLayer_.assign(cfg.numLayers, {});

    // Load final norm weights
    if (index_->finalNorm.byteSize > 0) {
        const BFloat16* fn = getWeightPtr(index_->finalNorm);
        if (fn) {
            finalNorm_.resize(cfg.hiddenSize);
            for (int i = 0; i < cfg.hiddenSize; ++i) {
                finalNorm_[i] = fn[i].toF32();
            }
            GLM_LOG_INFO("Loaded model.norm.weight");
        }
    }
    if (finalNorm_.empty()) {
        finalNorm_.assign(cfg.hiddenSize, 1.0f);
        GLM_LOG_WARN("model.norm.weight not found, using default value 1.0");
    }

#ifdef _OPENMP
    GLM_LOG_INFO("OpenMP thread count: " + std::to_string(omp_get_max_threads()));
#endif

    initialized_ = true;
    GLM_LOG_INFO("Forward engine ready: " + std::to_string(cfg.numLayers) + " layers, " +
                 std::to_string(cfg.hiddenSize) + " dim");
    return true;
}

void GLMForward::reset() {
    kvCache_.clear();
    seqLen_ = 0;
}

const BFloat16* GLMForward::getWeightPtr(const TensorLocation& loc) {
    const uint8_t* view = index_->getShardView(loc.shardPath);
    if (!view) return nullptr;
    return reinterpret_cast<const BFloat16*>(view + loc.byteOffset);
}

bool GLMForward::loadWeightToF32(const TensorLocation& loc, float* buf, int numel) {
    const uint8_t* raw = getWeightRaw(loc);
    if (!raw) return false;
    if (loc.dtype == "F32") {
        const float* f = reinterpret_cast<const float*>(raw);
        for (int i = 0; i < numel; ++i) buf[i] = f[i];
    } else {
        const BFloat16* b = reinterpret_cast<const BFloat16*>(raw);
        for (int i = 0; i < numel; ++i) buf[i] = b[i].toF32();
    }
    return true;
}

const uint8_t* GLMForward::getWeightRaw(const TensorLocation& loc) {
    const uint8_t* view = index_->getShardView(loc.shardPath);
    if (!view) return nullptr;
    return view + loc.byteOffset;
}

void GLMForward::rmsNorm(const TensorLocation& w, float* x, int n) {
    if (w.byteSize == 0) return;
    float eps = index_->config.rmsNormEps;
    float sqSum = 0.0f;
    for (int i = 0; i < n; ++i) sqSum += x[i] * x[i];
    float rms = 1.0f / std::sqrt(sqSum / float(n) + eps);
    const BFloat16* wptr = getWeightPtr(w);
    if (wptr) {
        for (int i = 0; i < n; ++i) x[i] = x[i] * rms * wptr[i].toF32();
    } else {
        for (int i = 0; i < n; ++i) x[i] = x[i] * rms;
    }
}

void GLMForward::applyRoPE(float* x, int dim, int pos, int offset, bool interleave) {
    if (interleave) {
        applyRoPEInterleave(x, dim, float(index_->config.ropeTheta), pos);
    } else {
        int half = dim / 2;
        for (int i = 0; i < half; ++i) {
            float freq = 1.0f / std::pow(float(index_->config.ropeTheta), float(i) / float(half));
            float angle = float(pos) * freq;
            float c = std::cos(angle), s = std::sin(angle);
            float x0 = x[i], x1 = x[i + half];
            x[i]      = x0 * c - x1 * s;
            x[i+half] = x0 * s + x1 * c;
        }
    }
}

void GLMForward::layerNorm(const TensorLocation& w, const TensorLocation& b, float* x, int n) {
    if (n <= 0) return;
    float mean = 0.0f;
    for (int i = 0; i < n; ++i) mean += x[i];
    mean /= float(n);
    float var = 0.0f;
    for (int i = 0; i < n; ++i) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= float(n);
    float inv = 1.0f / std::sqrt(var + 1e-6f);
    const BFloat16* wp = (w.byteSize > 0) ? getWeightPtr(w) : nullptr;
    const BFloat16* bp = (b.byteSize > 0) ? getWeightPtr(b) : nullptr;
    for (int i = 0; i < n; ++i) {
        float y = (x[i] - mean) * inv;
        x[i] = wp ? y * wp[i].toF32() : y;
        if (bp) x[i] += bp[i].toF32();
    }
}

void GLMForward::dsaIndexer(int layer, const float* hiddenStates, const float* qResid,
                            std::vector<int>& selSet) {
    const auto& cfg = index_->config;
    const auto& lw = index_->layers[layer];
    int idxHeads = cfg.indexNHeads;
    int idxDim = cfg.indexHeadDim;
    int idxRope = std::min<int>(cfg.qkRopeHeadDim, idxDim);
    int hidden = cfg.hiddenSize;

    // ---- Indexer Q: wq_b(q_resid) -> [idxHeads * idxDim], rope on first idxRope ----
    int iqDim = idxHeads * idxDim;
    ensureFloat(ws_.iq, size_t(iqDim));
    const BFloat16* wqB = getWeightPtr(lw.indexWqB);
    gemv_bf16_f32(wqB, qResid, ws_.iq.data(), iqDim, cfg.qLoraRank);
    for (int h = 0; h < idxHeads; ++h) {
        applyRoPE(ws_.iq.data() + size_t(h) * size_t(idxDim), idxRope, seqLen_, 0, cfg.ropeInterleave);
    }

    // ---- Indexer K: k_norm(wk(hidden_states)), rope on first idxRope ----
    ensureFloat(ws_.iK, size_t(idxDim));
    const BFloat16* wk = getWeightPtr(lw.indexWk);
    gemv_bf16_f32(wk, hiddenStates, ws_.iK.data(), idxDim, hidden);
    layerNorm(lw.indexKNormW, lw.indexKNormB, ws_.iK.data(), idxDim);
    applyRoPE(ws_.iK.data(), idxRope, seqLen_, 0, cfg.ropeInterleave);
    kvCache_.appendIndexKey(layer, seqLen_, ws_.iK.data());

    // ---- Per-head gate weights: weights_proj(hidden_states) * n_heads^-0.5 ----
    ensureFloat(ws_.wVec, size_t(idxHeads));
    const BFloat16* wProj = getWeightPtr(lw.indexWeightsProj);
    gemv_bf16_f32(wProj, hiddenStates, ws_.wVec.data(), idxHeads, hidden);
    float wScale = 1.0f / std::sqrt(float(idxHeads));
    for (int h = 0; h < idxHeads; ++h) ws_.wVec[h] *= wScale;

    // ---- Index scores: sum_h w_h * relu(q_h . k_t * head_dim^-0.5) ----
    int seqLen = seqLen_ + 1;
    float invScale = 1.0f / std::sqrt(float(idxDim));
    int topk = std::min<int>(cfg.indexTopk, seqLen);
    auto& ranked = ws_.ranked;
    ranked.clear();
    ranked.reserve(size_t(seqLen));
    for (int t = 0; t < seqLen; ++t) {
        const float* kt = kvCache_.getIndexKey(layer, t);
        float s = 0.0f;
        for (int h = 0; h < idxHeads; ++h) {
            float score = dotF32(ws_.iq.data() + size_t(h) * size_t(idxDim), kt, idxDim) * invScale;
            if (score < 0.0f) score = 0.0f;  // ReLU
            s += ws_.wVec[h] * score;
        }
        ranked.emplace_back(s, t);
    }
    // Stable top-k: score descending, ties by ascending position.
    std::partial_sort(ranked.begin(), ranked.begin() + topk, ranked.end(),
                      [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                          if (a.first != b.first) return a.first > b.first;
                          return a.second < b.second;
                      });
    selSet.clear();
    selSet.reserve(topk);
    for (int i = 0; i < topk; ++i) selSet.push_back(ranked[i].second);
}

void GLMForward::attentionLayer(int layer, const float* hiddenStates, float* output,
                                std::vector<int>& selSet) {
    const auto& cfg = index_->config;
    const auto& lw = index_->layers[layer];
    int hidden = cfg.hiddenSize;
    int heads = cfg.numAttentionHeads;
    int qLora = cfg.qLoraRank;
    int kvLora = cfg.kvLoraRank;
    int qkNope = cfg.qkNopeHeadDim;
    int qkRope = cfg.qkRopeHeadDim;
    int vDim = cfg.vHeadDim;
    int qkTotal = cfg.qkHeadDim;
    auto& ws = ws_;

    // ---- Q: q_a_proj -> q_a_layernorm -> q_b_proj ----
    ensureFloat(ws.qResid, size_t(qLora));
    const BFloat16* qAW = getWeightPtr(lw.qAProj);
    gemv_bf16_f32(qAW, hiddenStates, ws.qResid.data(), qLora, hidden);
    rmsNorm(lw.qALayernorm, ws.qResid.data(), qLora);

    int qOutDim = heads * qkTotal;
    ensureFloat(ws.qB, size_t(qOutDim));
    const BFloat16* qBW = getWeightPtr(lw.qBProj);
    gemv_bf16_f32(qBW, ws.qResid.data(), ws.qB.data(), qOutDim, qLora);

    // Split Q: [qkNope] and [qkRope] per head
    ensureFloat(ws.qNope, size_t(heads) * size_t(qkNope));
    ensureFloat(ws.qRope, size_t(heads) * size_t(qkRope));
    for (int h = 0; h < heads; ++h) {
        const float* src = ws.qB.data() + h * qkTotal;
        std::memcpy(ws.qNope.data() + h * qkNope, src, qkNope * sizeof(float));
        std::memcpy(ws.qRope.data() + h * qkRope, src + qkNope, qkRope * sizeof(float));
        applyRoPE(ws.qRope.data() + h * qkRope, qkRope, seqLen_, 0, cfg.ropeInterleave);
    }

    // ---- Compressed KV for the current token ----
    int kvAOut = kvLora + qkRope;
    ensureFloat(ws.kvA, size_t(kvAOut));
    const BFloat16* kvAW = getWeightPtr(lw.kvAProjWithMqa);
    gemv_bf16_f32(kvAW, hiddenStates, ws.kvA.data(), kvAOut, hidden);

    ensureFloat(ws.kvLatent, size_t(kvLora));
    ensureFloat(ws.kRope, size_t(qkRope));
    std::memcpy(ws.kvLatent.data(), ws.kvA.data(), kvLora * sizeof(float));
    std::memcpy(ws.kRope.data(), ws.kvA.data() + kvLora, qkRope * sizeof(float));
    rmsNorm(lw.kvALayernorm, ws.kvLatent.data(), kvLora);
    applyRoPE(ws.kRope.data(), qkRope, seqLen_, 0, cfg.ropeInterleave);

    // ---- Expand the current token's per-head k/v and cache it ----
    int kvBOut = heads * (qkNope + vDim);
    ensureFloat(ws.expandedNew, size_t(kvBOut));
    const BFloat16* kvBW = getWeightPtr(lw.kvBProj);
    gemv_bf16_f32(kvBW, ws.kvLatent.data(), ws.expandedNew.data(), kvBOut, kvLora);

    ensureFloat(ws.kNew, size_t(heads) * size_t(qkTotal));
    ensureFloat(ws.vNew, size_t(heads) * size_t(vDim));
    for (int h = 0; h < heads; ++h) {
        const float* src = ws.expandedNew.data() + size_t(h) * size_t(qkNope + vDim);
        float* kd = ws.kNew.data() + size_t(h) * size_t(qkTotal);
        std::memcpy(kd, src, qkNope * sizeof(float));
        std::memcpy(kd + qkNope, ws.kRope.data(), qkRope * sizeof(float));
        std::memcpy(ws.vNew.data() + size_t(h) * size_t(vDim), src + qkNope, vDim * sizeof(float));
    }
    kvCache_.appendKey(layer, seqLen_, ws.kNew.data());
    kvCache_.appendValue(layer, seqLen_, ws.vNew.data());

    // ---- DSA indexer: full layers compute, shared layers reuse selSet ----
    if (dsaEnabled_ && cfg.isIndexerFull(layer)) {
        dsaIndexer(layer, hiddenStates, ws.qResid.data(), selSet);
    }
    if (selSet.empty()) {  // degenerate safety: attend to every position
        int seqLen = seqLen_ + 1;
        selSet.resize(seqLen);
        for (int i = 0; i < seqLen; ++i) selSet[i] = i;
    }

    // ---- Sparse attention over the selected positions ----
    int seqLen = seqLen_ + 1;
    float invSqrtD = 1.0f / std::sqrt(float(qkTotal));
    ensureFloat(ws.attnOut, size_t(heads) * size_t(vDim));
    std::fill(ws.attnOut.begin(), ws.attnOut.end(), 0.0f);

    for (int h = 0; h < heads; ++h) {
        ensureFloat(ws.qCur, size_t(qkTotal));
        std::memcpy(ws.qCur.data(), ws.qNope.data() + h * qkNope, qkNope * sizeof(float));
        std::memcpy(ws.qCur.data() + qkNope, ws.qRope.data() + h * qkRope, qkRope * sizeof(float));

        ensureFloat(ws.scores, selSet.size());
        float* scores = ws.scores.data();
        for (size_t i = 0; i < selSet.size(); ++i) {
            int t = selSet[i];
            if (t < 0 || t >= seqLen) { scores[i] = -1e30f; continue; }
            const float* histK = kvCache_.getKey(layer, h, t);
            scores[i] = dotF32(ws.qCur.data(), histK, qkTotal) * invSqrtD;
        }

        softmax_f32(scores, int(selSet.size()));

        for (size_t i = 0; i < selSet.size(); ++i) {
            int t = selSet[i];
            if (t < 0 || t >= seqLen) continue;
            const float* histV = kvCache_.getValue(layer, h, t);
            float weight = scores[i];
            for (int d = 0; d < vDim; ++d) {
                ws.attnOut[h * vDim + d] += weight * histV[d];
            }
        }
    }

    // ---- o_proj ----
    const BFloat16* oW = getWeightPtr(lw.oProj);
    gemv_bf16_f32(oW, ws.attnOut.data(), output, hidden, heads * vDim);
}

void GLMForward::denseMlp(const LayerWeights& lw, const float* input, float* output) {
    const auto& cfg = index_->config;
    int hidden = cfg.hiddenSize;
    int inter = cfg.intermediateSize;

    ensureFloat(ws_.gate, size_t(inter));
    ensureFloat(ws_.up, size_t(inter));
    ensureFloat(ws_.mid, size_t(inter));
    const BFloat16* gateW = getWeightPtr(lw.mlpGate);
    const BFloat16* upW = getWeightPtr(lw.mlpUp);
    const BFloat16* downW = getWeightPtr(lw.mlpDown);

    gemv_bf16_f32(gateW, input, ws_.gate.data(), inter, hidden);
    gemv_bf16_f32(upW, input, ws_.up.data(), inter, hidden);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < inter; ++i) {
        ws_.mid[i] = ws_.gate[i] / (1.0f + std::exp(-ws_.gate[i])) * ws_.up[i];
    }

    gemv_bf16_f32(downW, ws_.mid.data(), output, hidden, inter);
}

void GLMForward::expertFFNRaw(const BFloat16* gateW, const BFloat16* upW, const BFloat16* downW,
                              const float* hiddenStates, float* output,
                              int hiddenSize, int interSize) {
    ensureFloat(ws_.gate, size_t(interSize));
    ensureFloat(ws_.up, size_t(interSize));
    ensureFloat(ws_.mid, size_t(interSize));

    gemv_bf16_f32(gateW, hiddenStates, ws_.gate.data(), interSize, hiddenSize);
    gemv_bf16_f32(upW, hiddenStates, ws_.up.data(), interSize, hiddenSize);

    for (int i = 0; i < interSize; ++i) {
        ws_.mid[i] = ws_.gate[i] / (1.0f + std::exp(-ws_.gate[i])) * ws_.up[i];
    }

    gemv_bf16_f32(downW, ws_.mid.data(), output, hiddenSize, interSize);
}

void GLMForward::expertFFN(const ExpertWeights& ew, const float* hiddenStates,
                            float* output, int hiddenSize, int interSize) {
    const BFloat16* gateW = getWeightPtr(ew.gateProj);
    const BFloat16* upW = getWeightPtr(ew.upProj);
    const BFloat16* downW = getWeightPtr(ew.downProj);
    expertFFNRaw(gateW, upW, downW, hiddenStates, output, hiddenSize, interSize);
}

void GLMForward::sharedExpertFFNRaw(const BFloat16* gateW, const BFloat16* upW,
                                    const BFloat16* downW, const float* input,
                                    float* output, int gateInter, int hidden) {
    if (!gateW || !upW || !downW) {
        std::fill(output, output + hidden, 0.0f);
        return;
    }
    ensureFloat(ws_.gate, size_t(gateInter));
    ensureFloat(ws_.up, size_t(gateInter));
    ensureFloat(ws_.mid, size_t(gateInter));

    gemv_bf16_f32(gateW, input, ws_.gate.data(), gateInter, hidden);
    gemv_bf16_f32(upW, input, ws_.up.data(), gateInter, hidden);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < gateInter; ++i) {
        ws_.mid[i] = ws_.gate[i] / (1.0f + std::exp(-ws_.gate[i])) * ws_.up[i];
    }

    gemv_bf16_f32(downW, ws_.mid.data(), output, hidden, gateInter);
}

void GLMForward::sharedExpertFFN(const LayerWeights& lw, const float* input, float* output) {
    const auto& cfg = index_->config;
    int hidden = cfg.hiddenSize;
    int inter = cfg.nSharedExperts * cfg.moeIntermediateSize;
    if (lw.sharedGateProj.byteSize == 0) {
        std::fill(output, output + hidden, 0.0f);
        return;
    }
    const BFloat16* gateW = getWeightPtr(lw.sharedGateProj);
    const BFloat16* upW = getWeightPtr(lw.sharedUpProj);
    const BFloat16* downW = getWeightPtr(lw.sharedDownProj);
    sharedExpertFFNRaw(gateW, upW, downW, input, output, inter, hidden);
}

void GLMForward::enableGpuExperts(int capacity, int stagingDepth) {
    if (capacity <= 0) return;
    gpuStagingDepth_ = std::max(1, stagingDepth);
    if (gpuExpertCacheConfig(capacity, gpuStagingDepth_)) {
        gpuExperts_ = true;
        GLM_LOG_INFO("GPU expert FFN: ENABLED (VRAM-resident window of " +
                     std::to_string(capacity) + " experts, staging depth " +
                     std::to_string(gpuStagingDepth_) + ", CPU fallback active)");
    } else {
        GLM_LOG_WARN("GPU expert FFN not available (no CUDA device or CUDA build); "
                     "running expert FFNs on CPU");
    }
}

void GLMForward::moeLayer(int layer, const float* input, float* output) {
    const auto& cfg = index_->config;
    const auto& lw = index_->layers[layer];
    int hidden = cfg.hiddenSize;
    int inter = cfg.moeIntermediateSize;
    int numExperts = cfg.numLocalExperts;
    int topK = cfg.numExpertsPerTok;

    // ---- Router ----
    ensureFloat(ws_.routerScores, size_t(numExperts));
    const BFloat16* gateW = getWeightPtr(lw.mlpGate);
    gemv_bf16_f32(gateW, input, ws_.routerScores.data(), numExperts, hidden);

    for (int i = 0; i < numExperts; ++i) {
        ws_.routerScores[i] = 1.0f / (1.0f + std::exp(-ws_.routerScores[i]));
    }

    const float* biasPtr = nullptr;
    if (lw.gateBias.byteSize > 0) {
        ensureFloat(ws_.bias, size_t(numExperts));
        if (!loadWeightToF32(lw.gateBias, ws_.bias.data(), numExperts)) {
            GLM_LOG_ERROR("Cannot read router correction bias for layer " + std::to_string(layer));
            return;
        }
        biasPtr = ws_.bias.data();
    }

    auto topk = topKScores(ws_.routerScores.data(), numExperts, biasPtr, topK,
                           cfg.routedScalingFactor, cfg.normTopkProb);

    // Feed the router output (with its weights/probabilities) to the scheduler
    // for one-layer-ahead prefetch + router-probability mode.
    if (scheduler_) {
        std::vector<int> idx(topk.indices, topk.indices + topk.k);
        std::vector<float> wts(topk.weights, topk.weights + topk.k);
        scheduler_->onLayerRouterDone(layer, idx, wts);
    }

    // ---- Routed expert FFN weighted merge ----
    // With --gpu-experts the elected experts stream through the CUDA deep
    // staging pipeline: compute(N) || H2D(N+1) ... || H2D(N+stagingDepth).
    // Any device error restarts the layer's experts on CPU so output is exact.
    ensureFloat(ws_.moeOut, size_t(hidden));
    std::fill(ws_.moeOut.begin(), ws_.moeOut.end(), 0.0f);

    int k = topk.k;
    bool gpuOk = gpuExperts_ && scheduler_ != nullptr && k > 0;
    int stagedEnd = 0;  // next topk index to stage (already really staged)

    auto stageExpertAt = [&](int idx) -> bool {
        if (idx >= k) return true;
        const uint8_t* ne = scheduler_->getExpertPtr(layer, topk.indices[idx]);
        if (!ne) return false;
        return gpuExpertStage(layer, topk.indices[idx],
                              reinterpret_cast<const BFloat16*>(ne), hidden, inter);
    };

    if (gpuOk) {
        // Deep preload: stage experts 0..depth-1 so the first depth copies are
        // already in flight before any compute, exactly the #2 pattern when a
        // router looks depth layers ahead.
        const int preload = std::min(k, std::max(1, gpuStagingDepth_));
        for (int d = 0; d < preload && gpuOk; ++d) {
            gpuOk = stageExpertAt(d);
            stagedEnd = d + 1;
        }
    }

    for (int i = 0; i < k; ++i) {
        if (gpuOk && stagedEnd < k) {
            // Advance the pipeline one slot past the expert we are about to
            // compute; its async H2D overlaps this expert's FFN on the device.
            if (!stageExpertAt(stagedEnd)) {
                gpuOk = false;
            }
            stagedEnd++;
        }

        ensureFloat(ws_.expertOut, size_t(hidden));
        std::fill(ws_.expertOut.begin(), ws_.expertOut.end(), 0.0f);

        if (gpuOk) {
            if (!gpuExpertFFN(layer, topk.indices[i], input,
                              ws_.expertOut.data(), hidden, inter)) {
                gpuOk = false;
                stagedEnd = k;  // no more staging; rest stays on CPU
                std::fill(ws_.moeOut.begin(), ws_.moeOut.end(), 0.0f);
                i = -1;  // restart the whole layer on CPU for byte-exact output
                continue;
            }
        } else {
            int eid = topk.indices[i];
            float w = topk.weights[i];
            if (scheduler_) {
                const uint8_t* eptr = scheduler_->getExpertPtr(layer, eid);
                if (eptr) {
                    const BFloat16* eG = reinterpret_cast<const BFloat16*>(eptr);
                    const BFloat16* eU = eG + int64_t(inter) * hidden;
                    const BFloat16* eD = eU + int64_t(inter) * hidden;
                    expertFFNRaw(eG, eU, eD, input, ws_.expertOut.data(), hidden, inter);
                } else {
                    expertFFN(index_->routedExperts[layer][eid], input,
                              ws_.expertOut.data(), hidden, inter);
                }
            } else {
                expertFFN(index_->routedExperts[layer][eid], input,
                          ws_.expertOut.data(), hidden, inter);
            }
        }

        float w = topk.weights[i];
        for (int j = 0; j < hidden; ++j) ws_.moeOut[j] += w * ws_.expertOut[j];
    }

    // ---- Shared experts ----
    ensureFloat(ws_.sharedOut, size_t(hidden));
    if (scheduler_) {
        const uint8_t* g = scheduler_->getSharedProjectionPtr(layer, kSharedGateProjSlot);
        const uint8_t* u = scheduler_->getSharedProjectionPtr(layer, kSharedUpProjSlot);
        const uint8_t* d = scheduler_->getSharedProjectionPtr(layer, kSharedDownProjSlot);
        int sharedInter = cfg.nSharedExperts * cfg.moeIntermediateSize;
        sharedExpertFFNRaw(reinterpret_cast<const BFloat16*>(g),
                           reinterpret_cast<const BFloat16*>(u),
                           reinterpret_cast<const BFloat16*>(d),
                           input, ws_.sharedOut.data(), sharedInter, hidden);
    } else {
        sharedExpertFFN(lw, input, ws_.sharedOut.data());
    }

    for (int j = 0; j < hidden; ++j) {
        output[j] = ws_.moeOut[j] + ws_.sharedOut[j];
    }
}

int GLMForward::lmHeadAndSample(const float* hiddenStates, std::vector<float>* logitsOut) {
    const auto& cfg = index_->config;
    int hidden = cfg.hiddenSize;
    int vocab = cfg.vocabSize;

    // Final RMSNorm
    ensureFloat(ws_.normed, size_t(hidden));
    std::memcpy(ws_.normed.data(), hiddenStates, hidden * sizeof(float));
    float eps = cfg.rmsNormEps;
    float sqSum = 0.0f;
    for (int i = 0; i < hidden; ++i) sqSum += ws_.normed[i] * ws_.normed[i];
    float rms = 1.0f / std::sqrt(sqSum / float(hidden) + eps);
    for (int i = 0; i < hidden; ++i) ws_.normed[i] = ws_.normed[i] * rms * finalNorm_[i];

    // LM Head: logits = lm_head @ hidden
    const BFloat16* lmW = getWeightPtr(index_->lmHead);

    int bestId = 0;
    float bestVal = -1e30f;

    if (logitsOut) {
        logitsOut->assign(vocab, 0.0f);
    }

    #pragma omp parallel
    {
        int localBestId = 0;
        float localBestVal = -1e30f;

        #pragma omp for nowait schedule(static)
        for (int v = 0; v < vocab; ++v) {
            const BFloat16* row = lmW + int64_t(v) * hidden;
            float dot = dotBf16F32(row, ws_.normed.data(), hidden);
            if (logitsOut) (*logitsOut)[v] = dot;
            if (dot > localBestVal) {
                localBestVal = dot;
                localBestId = v;
            }
        }

        #pragma omp critical
        {
            if (localBestVal > bestVal) {
                bestVal = localBestVal;
                bestId = localBestId;
            }
        }
    }

    return bestId;
}

int GLMForward::forward(const std::vector<int>& inputIds, std::vector<float>* logitsOut) {
    if (!initialized_) {
        GLM_LOG_ERROR("Forward engine not initialized");
        return -1;
    }

    const auto& cfg = index_->config;
    int hidden = cfg.hiddenSize;

    int startTok = (seqLen_ == 0) ? 0 : seqLen_;
    int endTok = int(inputIds.size());

    // Pre-reserve the KV cache for the whole horizon so per-token appends never
    // reallocate mid-run (item: KV decode-cost reduction).
    kvCache_.reserve(endTok + 16);

    auto& ws = ws_;
    ensureFloat(ws.hiddenStates, size_t(hidden));
    ensureFloat(ws.residual, size_t(hidden));
    ensureFloat(ws.attnOutL, size_t(hidden));
    ensureFloat(ws.mlpOut, size_t(hidden));

    for (int tok = startTok; tok < endTok; ++tok) {
        int tokenId = inputIds[tok];
        Timer t;

        // ---- Embedding lookup ----
        std::fill(ws.hiddenStates.begin(), ws.hiddenStates.end(), 0.0f);
        const BFloat16* embW = getWeightPtr(index_->embedTokens);
        const BFloat16* embRow = embW + int64_t(tokenId) * hidden;
        for (int i = 0; i < hidden; ++i) ws.hiddenStates[i] = embRow[i].toF32();

        // ---- Per-layer forward ----
        // DSA shared top-k selection propagates from "full" to "shared" layers.
        std::vector<int> selSet;
        selSet.reserve(size_t(std::max(cfg.indexTopk, 1)));
        for (int layer = 0; layer < cfg.numLayers; ++layer) {
            const auto& lw = index_->layers[layer];

            // ---- Attention residual: x = x + attention(norm1(x)) ----
            copyVec(ws.residual.data(), ws.hiddenStates.data(), hidden);

            rmsNorm(lw.inputLayernorm, ws.hiddenStates.data(), hidden);

            std::fill(ws.attnOutL.begin(), ws.attnOutL.end(), 0.0f);
            attentionLayer(layer, ws.hiddenStates.data(), ws.attnOutL.data(), selSet);
            lastTopkByLayer_[layer] = selSet;

            copyVec(ws.hiddenStates.data(), ws.residual.data(), hidden);
            addVec(ws.hiddenStates.data(), ws.attnOutL.data(), hidden);

            // ---- MLP residual: x = x + mlp(norm2(x)) ----
            copyVec(ws.residual.data(), ws.hiddenStates.data(), hidden);
            rmsNorm(lw.postAttentionLayernorm, ws.hiddenStates.data(), hidden);

            std::fill(ws.mlpOut.begin(), ws.mlpOut.end(), 0.0f);
            if (lw.isMoe) {
                moeLayer(layer, ws.hiddenStates.data(), ws.mlpOut.data());
            } else {
                denseMlp(lw, ws.hiddenStates.data(), ws.mlpOut.data());
            }

            copyVec(ws.hiddenStates.data(), ws.residual.data(), hidden);
            addVec(ws.hiddenStates.data(), ws.mlpOut.data(), hidden);
        }

        // Advance KV cache to include this token (all layers appended)
        kvCache_.advance();
        seqLen_++;

        if (tok == endTok - 1) {
            int nextToken = lmHeadAndSample(ws.hiddenStates.data(), logitsOut);
            GLM_LOG_DEBUG("token " + std::to_string(tok) + " (" +
                          std::to_string(t.elapsedMs()) + " ms) -> " + std::to_string(nextToken));
            return nextToken;
        }
        GLM_LOG_DEBUG("prefill token " + std::to_string(tok) + " (" +
                      std::to_string(t.elapsedMs()) + " ms)");
    }
    return -1;
}

} // namespace glm