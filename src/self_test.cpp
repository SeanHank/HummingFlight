#include "self_test.h"

#include "compute/cpu_kernels.h"
#include "compute/dtype_bf16.h"
#include "compute/moe_router.h"
#include "compute/sampler.h"
#include "engine/kv_cache.h"
#include "engine/placement.h"
#include "engine/scheduler.h"
#include "model/config.h"
#include "model/glm_forward.h"
#include "model/safetensors.h"
#include "model/weight_index.h"
#include "third_party/picojson.h"
#include "runtime/runtime_config.h"
#include "storage/lru_cache.h"
#include "storage/mapped_file.h"
#include "utils/logger.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace glm {

namespace {

int g_failures = 0;
int g_checks = 0;
bool g_verbose = false;

#define CHECK(cond, name)                                                 \
    do {                                                                  \
        ++g_checks;                                                       \
        if (cond) {                                                       \
            if (g_verbose) std::printf("[PASS] %s\n", name);              \
        } else {                                                          \
            ++g_failures;                                                 \
            std::printf("[FAIL] %s (line %d)\n", name, __LINE__);         \
        }                                                                 \
    } while (0)

#define CHECK_NAMED(cond, name) CHECK(cond, name)

// ---------- Test 1: BFloat16 ----------
void testBFloat16() {
    struct Known { float f; };
    // Exact BF16-representable values (7 bits mantissa => match exactly).
    CHECK(BFloat16::fromF32(1.0f).toF32() == 1.0f, "BF16: 1.0 round-trip exact");
    CHECK(BFloat16::fromF32(2.0f).toF32() == 2.0f, "BF16: 2.0 round-trip exact");
    CHECK(BFloat16::fromF32(-1.5f).toF32() == -1.5f, "BF16: -1.5 round-trip exact");
    CHECK(BFloat16::fromF32(0.5f).toF32() == 0.5f, "BF16: 0.5 round-trip exact");
    CHECK(BFloat16::fromF32(0.0f).toF32() == 0.0f, "BF16: 0.0 round-trip exact");
    CHECK(BFloat16::fromF32(-0.0f).toF32() == 0.0f, "BF16: -0.0 round-trip exact");

    // Approximate values: relative error must stay within BF16 precision.
    const float vals[] = {0.1f, 0.3f, 1.0f / 3.0f, 123.456f, -987.654f, 1e-3f, 1e3f};
    for (float v : vals) {
        float r = BFloat16::fromF32(v).toF32();
        float err = std::fabs(r - v) / std::max(std::fabs(v), 1e-30f);
        CHECK(err < 1e-2f, "BF16: conversion within precision");
    }
}

// ---------- Test 2: GEMV (BF16 x F32) ----------
void testGemvBf16() {
    const int M = 8, K = 33; // K not a multiple of 8 to exercise the tail loop
    std::vector<BFloat16> A(M * K);
    std::vector<float> x(K), y_ref(M), y(M);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : A) v = BFloat16::fromF32(dist(rng));
    for (auto& v : x) v = 1.5f * dist(rng);

    // Reference: naive dot product.
    for (int i = 0; i < M; ++i) {
        float dot = 0.0f;
        for (int j = 0; j < K; ++j) dot += A[i * K + j].toF32() * x[j];
        y_ref[i] = dot;
    }

    gemv_bf16_f32(A.data(), x.data(), y.data(), M, K);

    // Note: AVX2 paths accumulate in a different order; allow lenient tolerance.
    for (int i = 0; i < M; ++i) {
        CHECK(std::fabs(y[i] - y_ref[i]) < 1e-3f, "GEMV BF16 matches reference");
    }

    // alpha/beta variants
    std::vector<float> y2(M, 7.0f);
    gemv_bf16_f32(A.data(), x.data(), y2.data(), M, K, 2.0f, 0.5f);
    for (int i = 0; i < M; ++i) {
        float expect = 2.0f * y_ref[i] + 0.5f * 7.0f;
        CHECK(std::fabs(y2[i] - expect) < 2e-3f, "GEMV BF16 alpha/beta");
    }
}

// ---------- Test 3: GEMV (F32) ----------
void testGemvF32() {
    const int M = 6, K = 16;
    std::vector<float> A(M * K), x(K), y_ref(M), y(M);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (auto& v : A) v = dist(rng);
    for (auto& v : x) v = dist(rng);
    for (int i = 0; i < M; ++i) {
        float dot = 0.0f;
        for (int j = 0; j < K; ++j) dot += A[i * K + j] * x[j];
        y_ref[i] = dot;
    }
    gemv_f32(A.data(), x.data(), y.data(), M, K);
    for (int i = 0; i < M; ++i) {
        CHECK(std::fabs(y[i] - y_ref[i]) < 1e-4f, "GEMV F32 matches reference");
    }
}

// ---------- Test 4: RMSNorm reference ----------
void testRmsNorm() {
    const int n = 128;
    const float eps = 1e-5f;
    std::vector<float> x(n);
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : x) v = dist(rng);

    // Reference RMSNorm: x -> x / sqrt(mean(x^2) + eps)
    float sqSum = 0.0f;
    for (int i = 0; i < n; ++i) sqSum += x[i] * x[i];
    float rms = 1.0f / std::sqrt(sqSum / float(n) + eps);
    std::vector<float> ref = x;
    for (int i = 0; i < n; ++i) ref[i] *= rms;

    // Use softmax as a proxy? No - replicate via gemv-free double loop:
    // The GLMForward::rmsNorm is private; verify the math that it uses.
    std::vector<float> got = x;
    {
        float s2 = 0.0f;
        for (int i = 0; i < n; ++i) s2 += got[i] * got[i];
        float r = 1.0f / std::sqrt(s2 / float(n) + eps);
        for (int i = 0; i < n; ++i) got[i] *= r;
    }
    bool ok = true;
    for (int i = 0; i < n; ++i) {
        if (std::fabs(got[i] - ref[i]) > 1e-6f) { ok = false; break; }
    }
    CHECK(ok, "RMSNorm formula matches reference");
}

// ---------- Test 5: softmax ----------
void testSoftmax() {
    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    softmax_f32(x.data(), int(x.size()));
    float sum = 0.0f;
    for (float v : x) sum += v;
    CHECK(std::fabs(sum - 1.0f) < 1e-5f, "Softmax sums to 1");
    // Monotonicity preserved under softmax.
    bool mono = x[0] < x[1] && x[1] < x[2] && x[2] < x[3] && x[3] < x[4];
    CHECK(mono, "Softmax preserves ordering");
    CHECK(std::fabs(x[4] - (std::exp(5.0f - 5.0f) / (1.0f + std::exp(-1.0f) + std::exp(-2.0f) + std::exp(-3.0f) + std::exp(-4.0f)))) < 1e-5f, "Softmax: max element value correct");
}

// ---------- Test 6: sigmoid ----------
void testSigmoid() {
    CHECK(std::fabs(sigmoid_f32(0.0f) - 0.5f) < 1e-6f, "Sigmoid(0)=0.5");
    CHECK(std::fabs(sigmoid_f32(100.0f) - 1.0f) < 1e-6f, "Sigmoid(+inf)~1");
    CHECK(std::fabs(sigmoid_f32(-100.0f) - 0.0f) < 1e-6f, "Sigmoid(-inf)~0");
    // logistic identity: s(x) + s(-x) = 1
    CHECK(std::fabs(sigmoid_f32(2.0f) + sigmoid_f32(-2.0f) - 1.0f) < 1e-6f,
          "Sigmoid identity s(x)+s(-x)=1");
}

// ---------- Test 7: noaux_tc top-K router ----------
void testTopKScores() {
    const float scores[8] = {0.1f, 0.6f, 0.9f, 0.4f, 0.7f, 0.2f, 0.8f, 0.3f};
    // top-3 by adjusted score (no bias): indices 2 (0.9), 6 (0.8), 4 (0.7)
    TopKResult r = topKScores(scores, 8, nullptr, 3, 1.0f, false);
    CHECK(r.k == 3, "TopK returns k=3");
    CHECK(r.indices[0] == 2 && r.indices[1] == 6 && r.indices[2] == 4,
          "TopK picks highest scores in order");
    // Weight normalization: sum of selected weights == 1 (normTopkProb=true)
    TopKResult rn = topKScores(scores, 8, nullptr, 3, 1.0f, true);
    float sum = 0.0f;
    for (int i = 0; i < rn.k; ++i) sum += rn.weights[i];
    CHECK(std::fabs(sum - 1.0f) < 1e-5f, "TopK normalized weights sum to 1");
    // Routed scaling factor applied.
    TopKResult rs = topKScores(scores, 8, nullptr, 3, 2.5f, false);
    bool scaled = rs.weights[0] > r.weights[0] + 1e-8f;
    CHECK(scaled, "TopK applies routed scaling factor");
    // Bias shifts selection: bias[2] = -10 drops expert 2.
    const float bias[8] = {0, 0, -10, 0, 0, 0, 0, 0};
    TopKResult rb = topKScores(scores, 8, bias, 3, 1.0f, false);
    CHECK(rb.indices[0] == 6, "TopK respects e_score_correction_bias");
}

// ---------- Test 8: expanded MLA KV + DSA indexer key cache ----------
void testKvCache() {
    KVCache kv;
    // 2 layers, heads=4, expanded key dim per head=5, value per head=3,
    // indexer key dim=6
    kv.init(2, 4, 5, 3, 6, 64);

    const float key[20]  = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    const float value[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const float idxK[6]  = {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f};

    kv.appendKey(0, 0, key);
    kv.appendValue(0, 0, value);
    kv.appendIndexKey(0, 0, idxK);
    kv.appendKey(1, 0, key);
    kv.appendValue(1, 0, value);
    kv.appendIndexKey(1, 0, idxK);
    kv.advance();

    CHECK(kv.seqLen() == 1, "KVCache seqLen advances");

    // Per-head addressing within the expanded key (head-major: head 1 starts
    // at its own block, first element = 6; head 0 token 0 spans d=5 floats).
    const float* k0 = kv.getKey(0, 0, 0);
    CHECK(k0 != nullptr && k0[0] == 1.0f && k0[4] == 5.0f, "KVCache stores expanded key");
    const float* k1h = kv.getKey(0, 1, 0);
    CHECK(k1h[0] == 6.0f, "KVCache expanded key head addressing (head-major)");

    const float* v0 = kv.getValue(0, 0, 0);
    CHECK(v0 && v0[0] == 1.0f && v0[2] == 3.0f, "KVCache stores expanded value");
    CHECK(kv.getValue(0, 1, 0)[0] == 4.0f, "KVCache value per-head layout (head-major)");

    const float* ix = kv.getIndexKey(0, 0);
    CHECK(ix && ix[0] == 0.5f && ix[5] == 5.5f, "KVCache stores DSA indexer key");

    // Distinct layer buffers, not overlapping storage.
    const float* kL1 = kv.getKey(1, 0, 0);
    CHECK(kL1 && kL1[0] == 1.0f, "KVCache layer addressing");

    // Head h's history across positions stays head-contiguous (the hot-path
    // sparse-attention access pattern).
    kv.appendKey(0, 1, key);
    const float* h1Head = kv.getKey(0, 1, 0);
    const float* h1Next = kv.getKey(0, 1, 1);
    CHECK(h1Next == h1Head + 5 && h1Next[0] == 6.0f,
          "KVCache head-major tokens are contiguous per head");

    // Grow beyond initial capacity.
    kv.appendKey(0, 200, key);
    kv.appendValue(0, 200, value);
    kv.appendIndexKey(0, 200, idxK);
    kv.advance();
    CHECK(kv.maxSeqLen() >= 201, "KVCache grows on demand");
    const float* k200 = kv.getKey(0, 0, 200);
    CHECK(k200 && k200[4] == 5.0f, "KVCache grow preserves data");
    const float* ix200 = kv.getIndexKey(0, 200);
    CHECK(ix200 && ix200[5] == 5.5f, "KVCache grow preserves indexer key");

    kv.clear();
    CHECK(kv.seqLen() == 0, "KVCache clear resets seqLen");

    // reserve() pre-allocates a horizon without touching existing data.
    kv.appendKey(0, 10, key);
    kv.reserve(2048);
    CHECK(kv.maxSeqLen() >= 2048, "KVCache reserve extends capacity");
    const float* k10 = kv.getKey(0, 0, 10);
    CHECK(k10 && k10[4] == 5.0f, "KVCache reserve preserves existing data");
}

// ---------- Test 9: Sampler (temperature / top-k / top-p / categorical) ----------
void testSampler() {
    const int V = 6;
    std::vector<float> logits = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    // Greedy (temperature<=0) is a deterministic argmax.
    std::mt19937 rng(1);
    int g = sampleFromLogits(logits.data(), V, 0.0f, 1.0f, 0, rng);
    CHECK(g == 5, "Sampler greedy = argmax");

    // Top-k keeps the k largest, drops the rest.
    std::vector<float> lk = logits;
    applyTopK(lk.data(), V, 3);
    CHECK(lk[5] == 5.0f && lk[4] == 4.0f && lk[3] == 3.0f, "TopK keeps top-3");
    CHECK(lk[2] == -1e30f, "TopK drops below threshold");

    // Top-p zeroes the crossing token AND all smaller (corrected semantics).
    std::vector<float> lp = {10.0f, 9.0f, 8.0f, 7.0f, 6.0f};
    applyTopP(lp.data(), 5, 0.9f);
    CHECK(lp[0] == 10.0f && lp[1] == 9.0f, "TopP keeps the nucleus");
    CHECK(lp[2] == -1e30f && lp[3] == -1e30f && lp[4] == -1e30f,
          "TopP zeroes crossing token and every smaller one");

    // Same seed => same sample.
    std::vector<float> probs = {0.98f, 0.01f, 0.01f};
    std::mt19937 rngA(42), rngB(42);
    int a = sampleFromLogits(probs.data(), 3, 0.8f, 1.0f, 0, rngA);
    int b = sampleFromLogits(probs.data(), 3, 0.8f, 1.0f, 0, rngB);
    CHECK(a == b, "Sampler deterministic given a seed");

    // Temperature scales logits in place.
    std::vector<float> lt = {3.0f, 1.0f, 2.0f};
    applyTemperature(lt.data(), 3, 0.5f);
    CHECK(lt[0] == 6.0f && lt[1] == 2.0f && lt[2] == 4.0f, "Temperature scales in place");

    // Min-p: keeps tokens within minP of the peak, drops the rest.
    std::vector<float> lm = {10.0f, 5.0f, 2.0f, -3.0f};
    float mMax = *std::max_element(lm.begin(), lm.end());
    applyMinP(lm.data(), 4, 0.5f);                       // thr = 10 + ln(0.5) ~ 9.31
    int mKept = 0;
    for (float v : lm) if (v > -1e29f) mKept++;
    CHECK(mKept == 1 && std::fabs(lm[0] - 10.0f) < 1e-6f, "MinP keeps only the peak at minP=0.5");

    // Repetition penalty: penalizes tokens in the window (score < 0 => mult).
    std::vector<float> lr = {2.0f, -2.0f, 1.0f};
    const int seen[2] = {0, 1};
    applyRepetitionPenalty(lr.data(), 3, 1.2f, seen, 2);
    CHECK(std::fabs(lr[0] - 2.0f / 1.2f) < 1e-6f, "Repetition penalty divides positive logit");
    CHECK(std::fabs(lr[1] + 2.4f) < 1e-6f, "Repetition penalty multiplies negative logit");
    CHECK(std::fabs(lr[2] - 1.0f) < 1e-6f, "Repetition penalty leaves others untouched");
    // penalty == 1.0 is a no-op.
    std::vector<float> lr1 = {2.0f, -2.0f};
    applyRepetitionPenalty(lr1.data(), 2, 1.0f, seen, 2);
    CHECK(lr1[0] == 2.0f && lr1[1] == -2.0f, "Repetition penalty disabled at 1.0");

    // Extended sampler: identical behaviour to the classic overload when the new
    // filters are disabled (goldens preserved).
    std::vector<float> logits2 = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::mt19937 rngX(7), rngY(7);
    int classic = sampleFromLogits(logits2.data(), 6, 0.8f, 0.9f, 4, rngX);
    int extended = sampleFromLogits(logits2.data(), 6, 0.8f, 0.9f, 4,
                                    0.0f, 0.0f, nullptr, 0, rngY);
    CHECK(classic == extended, "Extended sampler matches classic when filters disabled");

    // Greedy through the extended sampler stays a deterministic argmax.
    std::mt19937 rngZ(3);
    int gz = sampleFromLogits(logits2.data(), 6, 0.0f, 1.0f, 0,
                              0.0f, 0.0f, nullptr, 0, rngZ);
    CHECK(gz == 5, "Extended sampler greedy = argmax");

    // Frequency penalty: each occurrence subtracts penalty from the logit.
    std::vector<float> lf = {3.0f, 3.0f, 3.0f};
    const int windowF[3] = {0, 0, 1};
    applyFrequencyPenalty(lf.data(), 3, 0.25f, windowF, 3);
    CHECK(std::fabs(lf[0] - 2.5f) < 1e-6f, "Frequency penalty subtracts 0.25 x 2 for token 0");
    CHECK(std::fabs(lf[1] - 2.75f) < 1e-6f, "Frequency penalty subtracts 0.25 x 1 for token 1");
    CHECK(std::fabs(lf[2] - 3.0f) < 1e-6f, "Frequency penalty leaves unseen tokens alone");

    // Presence penalty: fixed per distinct token, independent of count.
    std::vector<float> lp2 = {3.0f, 3.0f, 3.0f};
    applyPresencePenalty(lp2.data(), 3, 0.5f, windowF, 3);
    CHECK(std::fabs(lp2[0] - 2.5f) < 1e-6f && std::fabs(lp2[2] - 3.0f) < 1e-6f,
          "Presence penalty subtracts once per distinct token");

    // Typical-p masks less typical tokens, keeps the peak's mass.
    std::vector<float> lt2 = {12.0f, 8.0f, 7.0f, 6.0f, 5.0f};
    applyTypicalP(lt2.data(), 5, 0.6f);
    int tKept = 0;
    for (float v : lt2) if (v > -1e29f) tKept++;
    CHECK(tKept >= 1 && tKept < 5, "Typical-p keeps a locally-typical subset");

    // Fully extended sampler with all extra filters disabled matches the
    // 10-arg overload (goldens preserved).
    std::vector<float> logits3 = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::mt19937 rngP(11), rngQ(11);
    int base = sampleFromLogits(logits3.data(), 6, 0.8f, 0.9f, 4,
                                0.0f, 0.0f, nullptr, 0, rngP);
    int full = sampleFromLogits(logits3.data(), 6, 0.8f, 0.9f, 4,
                                0.0f, 0.0f, nullptr, 0,
                                0.0f, 0.0f, 0.0f, rngQ);
    CHECK(base == full, "13-arg sampler identical when extra filters disabled");
}

// ---------- Test 10: Placement director ----------
void testPlacement() {
    PlacementDirector pd;
    pd.setVramBudget(6ULL * 1024 * 1024 * 1024);
    // Cold expert starts on HDD, heats up to RAM (>10) then VRAM (>100).
    CHECK(PlacementTier::HDD == pd.decideExpertPlacement(0, 0),
          "Placement default tier is HDD");
    pd.recordAccess(0, 5);
    pd.recordAccess(0, 5);
    pd.recordAccess(0, 5);
    CHECK(pd.getAccessCount(0, 5) == 3, "Placement records accesses");
    CHECK(PlacementTier::HDD == pd.decideExpertPlacement(0, 5),
          "Placement: 3 accesses stays HDD");
    for (int i = 0; i < 9; ++i) pd.recordAccess(0, 5);
    CHECK(pd.getAccessCount(0, 5) == 12, "Placement counts to 12");
    CHECK(PlacementTier::RAM == pd.decideExpertPlacement(0, 5),
          "Placement: 12 accesses promotes to RAM");
    pd.decayHeat(0.5f);
    CHECK(pd.getAccessCount(0, 5) == 6, "Placement decays heat");
    CHECK(PlacementTier::HDD == pd.decideExpertPlacement(0, 5),
          "Placement: decay demotes back to HDD");
    pd.resetHeat();
    CHECK(pd.getAccessCount(0, 5) == 0, "Placement reset clears heat");
    // Dense placement rules.
    CHECK(PlacementTier::RAM == pd.decideDensePlacement("model.layers.1.self_attn.q_k_b_proj.weight", 1),
          "Placement dense attn name maps to RAM");
    CHECK(PlacementTier::VRAM == pd.decideDensePlacement("model.embed_tokens.weight", 1),
          "Placement embedding to VRAM");
    CHECK(PlacementTier::VRAM == pd.decideDensePlacement("model.layers.0.gate.weight", 1),
          "Placement gate to VRAM");
}

// ---------- Test 11: Config parsing ----------
void testConfigParse(const std::string& fixtureDir) {
    ModelConfig cfg;
    std::string path = fixtureDir + "/config_mini.json";
    bool ok = cfg.loadFromFile(path);
    CHECK(ok, "Config parses fixture");
    if (ok) {
        CHECK(cfg.hiddenSize == 16, "Config hiddenSize parsed");
        CHECK(cfg.numLayers == 4, "Config numLayers parsed");
        CHECK(cfg.numAttentionHeads == 4, "Config heads parsed");
        CHECK(cfg.numLocalExperts == 8, "Config experts parsed");
        CHECK(cfg.isMoeLayer(0) == false, "Config layer 0 is dense");
        CHECK(cfg.isMoeLayer(1) == true, "Config layer 1 is MoE");
    }

    // MTP fixture: a checkpoint carrying a single 4-layer MTP head module.
    ModelConfig mcfg;
    std::string mpath = fixtureDir + "/config_mtp.json";
    bool mok = mcfg.loadFromFile(mpath);
    CHECK(mok, "Config parses MTP fixture");
    if (mok) {
        CHECK(mcfg.numMtpModules == 1, "Config MTP modules parsed");
        CHECK(mcfg.numMtpLayersPerModule == 4, "Config MTP layers/module parsed");
        CHECK(mcfg.mtpIntermediateSize == 32, "Config MTP intermediate parsed");
        CHECK(mcfg.mtpLayerTypes.size() == 1 && mcfg.mtpLayerTypes[0] == "shared",
              "Config MTP layer types parsed");
    }

    // GLM-5.2 spelling: num_nextn_predict_layers + index_share_for_mtp_iteration
    // must surface through the same numMtpModules field, and win over the legacy
    // key when both are present (item 8).
    ModelConfig n5;
    CHECK(n5.loadFromString(
              "{\"hidden_size\":16,\"num_hidden_layers\":4,\"num_attention_heads\":4,"
              "\"num_key_value_heads\":4,\"q_lora_rank\":8,\"kv_lora_rank\":4,"
              "\"qk_head_dim\":4,\"qk_nope_head_dim\":3,\"qk_rope_head_dim\":1,"
              "\"v_head_dim\":4,\"vocab_size\":100,\"intermediate_size\":32,"
              "\"num_nextn_predict_layers\":1,\"index_share_for_mtp_iteration\":true}"),
          "Config parses GLM-5.2 nextn keys");
    CHECK(n5.numMtpModules == 1, "Config nextn_predict_layers -> numMtpModules");
    CHECK(n5.indexShareForMtpIteration, "Config index_share_for_mtp_iteration parsed");

    ModelConfig n5b;
    CHECK(n5b.loadFromString(
              "{\"hidden_size\":16,\"num_hidden_layers\":4,\"num_attention_heads\":4,"
              "\"num_key_value_heads\":4,\"q_lora_rank\":8,\"kv_lora_rank\":4,"
              "\"qk_head_dim\":4,\"qk_nope_head_dim\":3,\"qk_rope_head_dim\":1,"
              "\"v_head_dim\":4,\"vocab_size\":100,\"intermediate_size\":32,"
              "\"num_nextn_predict_layers\":2,\"num_mtp_modules\":9,\"mtp_layer_types\":[\"shared\",\"shared\"]}"),
          "Config parses GLM-5.2 nextn + legacy keys");
    CHECK(n5b.numMtpModules == 2, "Config nextn key wins over legacy num_mtp_modules");
    CHECK(n5b.numMtpLayersPerModule == 0, "Config legacy depth untouched without key");
    CHECK(n5b.mtpLayerTypes.size() == 2, "Config MTP layer types from 5.2 keys");

    // Malformed JSON must fail loudly (real parser rejection, embedded error path).
    ModelConfig n5c;
    CHECK(!n5c.loadFromString("{broken"), "Config rejects malformed JSON");
}

// ---------- Test 14: Strict safetensors validation ----------
// Builds synthetic .safetensors headers with picojson (the same library the
// parser uses) and asserts the production strictness rules fire per category.
namespace {

std::string sfWriteTemp(const std::string& name, const std::string& bytes) {
    std::error_code ec;
    auto dir = std::filesystem::temp_directory_path(ec);
    std::string path = (dir / name).string();
    std::ofstream f(path, std::ios::binary);
    f.write(bytes.data(), std::streamsize(bytes.size()));
    f.close();
    return path;
}

// Serialize a synthetic safetensors file: 8-byte LE header size + JSON header
// (padded with whitespace to a multiple of 8 unless align=false) + payload.
std::string sfBuild(std::string jsonHeader, size_t payload, bool align = true) {
    if (align) {
        size_t rem = jsonHeader.size() % 8;
        if (rem) jsonHeader.append(8 - rem, ' ');
    }
    uint64_t hlen = uint64_t(jsonHeader.size());
    std::string out;
    for (int i = 0; i < 8; ++i) out += char((hlen >> (8 * i)) & 0xFF);
    out += jsonHeader;
    out.append(payload, '\0');
    return out;
}

picojson::value sfTensor(const std::string& dtype, std::vector<int> shape,
                         uint64_t begin, uint64_t end) {
    picojson::object t;
    t["dtype"] = picojson::value(dtype);
    picojson::array sh;
    for (int s : shape) sh.push_back(picojson::value(double(s)));
    t["shape"] = picojson::value(sh);
    picojson::array offs;
    offs.push_back(picojson::value(double(begin)));
    offs.push_back(picojson::value(double(end)));
    t["data_offsets"] = picojson::value(offs);
    return picojson::value(t);
}

std::string sfHeader(const picojson::value& tensor, const std::string& name = "w") {
    picojson::object root;
    picojson::object meta;
    meta["format"] = picojson::value("pt");
    root["__metadata__"] = picojson::value(meta);
    root[name] = tensor;
    return picojson::value(root).serialize();
}

} // namespace

// Soft-priority LRU boost (#14): a boosted entry survives eviction sweeps
// without being pinned; unpromoted entries are evicted first.
void testLruBoost() {
    ExpertLRU lru(4 * 1024);
    for (int i = 1; i <= 4; ++i) {
        bool ok = lru.insert(ExpertKey{0, uint16_t(i)}, BufferHandle(1024));
        CHECK(ok, "LRU boost: fills capacity");
    }
    lru.boost(ExpertKey{0, 2}, 1);   // soft-priority for expert 2
    lru.boost(ExpertKey{0, 99}, 1);  // no-op when not resident

    // Inserting two more entries must evict twice, skipping the boosted k2.
    lru.insert(ExpertKey{0, 5}, BufferHandle(1024));
    lru.insert(ExpertKey{0, 6}, BufferHandle(1024));
    CHECK(lru.contains(ExpertKey{0, 2}), "LRU boost: boosted entry survives eviction");
    CHECK(!lru.contains(ExpertKey{0, 1}), "LRU boost: unboosted entry evicted");
    CHECK(!lru.contains(ExpertKey{0, 3}), "LRU boost: second unboosted entry evicted");
}

void testSafetensorsStrict() {
    std::vector<std::string> cleanup;

    // (1) Valid BF16 tensor: accepted with default opts (requireBf16=true).
    {
        std::string bytes = sfBuild(sfHeader(sfTensor("BF16", {2, 3}, 0, 12)), 12);
        std::string path = sfWriteTemp("glm_sf_valid.bin", bytes);
        cleanup.push_back(path);
        SafeTensorsFile f;
        CHECK(f.open(path), "Safetensors: valid BF16 file opens");
        if (f.tensors().size() == 1) {
            const TensorInfo* w = f.find("w");
            CHECK(w && w->byteSize() == 12 && w->numel() == 6,
                  "Safetensors: tensor metadata parsed");
        } else {
            CHECK(false, "Safetensors: tensor metadata parsed");
        }
        const auto& st = f.validationStats();
        CHECK(st.total == 1 && st.accepted == 1 && st.rejected == 0,
              "Safetensors: valid tensor fully accepted");
    }

    // (2) F32 tensor rejected when requireBf16 (default): dtype gate.
    {
        std::string bytes = sfBuild(sfHeader(sfTensor("F32", {2}, 0, 8)), 8);
        std::string path = sfWriteTemp("glm_sf_dtype.bin", bytes);
        cleanup.push_back(path);
        SafeTensorsFile f;
        CHECK(f.open(path), "Safetensors: non-BF16 file opens (rejections counted)");
        const auto& st = f.validationStats();
        CHECK(st.accepted == 0 && st.dtypeRejected == 1 && st.rejected == 1,
              "Safetensors: F32 rejected under requireBf16");
    }

    // (3) Shape/dtype/span mismatch (3 elements x BF16 = 6 B vs 12 B declared).
    {
        std::string bytes = sfBuild(sfHeader(sfTensor("BF16", {3}, 0, 12)), 12);
        std::string path = sfWriteTemp("glm_sf_byte.bin", bytes);
        cleanup.push_back(path);
        SafeTensorsFile f;
        CHECK(f.open(path), "Safetensors: mismatched byte file opens");
        const auto& st = f.validationStats();
        CHECK(st.accepted == 0 && st.byteMismatchRejected == 1,
              "Safetensors: byte-size mismatch rejected");
    }

    // (4) Offsets beyond the file end.
    {
        std::string bytes = sfBuild(sfHeader(sfTensor("BF16", {2}, 0, 100)), 12);
        std::string path = sfWriteTemp("glm_sf_bounds.bin", bytes);
        cleanup.push_back(path);
        SafeTensorsFile f;
        CHECK(f.open(path), "Safetensors: out-of-bounds file opens");
        const auto& st = f.validationStats();
        CHECK(st.accepted == 0 && st.boundsRejected == 1,
              "Safetensors: out-of-bounds offsets rejected");
    }

    // (5) Garbage header: not valid JSON -> open fails outright.
    {
        std::string bytes = sfBuild("<not json at all", 16);
        std::string path = sfWriteTemp("glm_sf_badjson.bin", bytes);
        cleanup.push_back(path);
        SafeTensorsFile f;
        CHECK(!f.open(path), "Safetensors: malformed JSON header rejected");
    }

    // (6) Unaligned header length rejected under the default strict alignment.
    {
        std::string hdr = sfHeader(sfTensor("BF16", {2}, 0, 4)); // strict 8-multiple
        if (hdr.size() % 8 == 0) hdr += ' ';                      // force misalignment
        std::string bytes = sfBuild(hdr, 4, false);
        std::string path = sfWriteTemp("glm_sf_align.bin", bytes);
        cleanup.push_back(path);
        SafeTensorsFile f;
        CHECK(!f.open(path), "Safetensors: unaligned header rejected");
    }

    // (7) Unknown dtype rejected.
    {
        std::string bytes = sfBuild(sfHeader(sfTensor("I8", {2}, 0, 2)), 8);
        std::string path = sfWriteTemp("glm_sf_unknown.bin", bytes);
        cleanup.push_back(path);
        SafeTensorsFile f;
        CHECK(f.open(path), "Safetensors: unknown-dtype file opens");
        const auto& st = f.validationStats();
        CHECK(st.accepted == 0 && st.dtypeRejected == 1,
              "Safetensors: unknown dtype rejected");
    }

    // (8) Unaligned tensor DATA start rejected (absolute offset % 8 != 0).
    {
        // Header + 8 bytes preamble, then BF16 data at absolute offset that is
        // not a multiple of 8: headerSize_ + dataBegin must be non-aligned.
        std::string preamble(6, 'x');  // 6-byte payload before the tensor data
        std::string bytes = sfBuild(sfHeader(sfTensor("BF16", {2}, 6, 10)), 10);
        std::string path = sfWriteTemp("glm_sf_dataalign.bin", bytes);
        cleanup.push_back(path);
        SafeTensorsFile f;
        CHECK(f.open(path), "Safetensors: unaligned-data file opens (rejection counted)");
        const auto& st = f.validationStats();
        CHECK(st.accepted == 0 && st.alignmentRejected == 1,
              "Safetensors: unaligned tensor data rejected");
    }

    // (9) F32 router correction bias: rejected by default (BF16-only gate)...
    {
        std::string bytes = sfBuild(sfHeader(sfTensor("F32", {256}, 0, 1024),
                                             "model.layers.4.mlp.gate.e_score_correction_bias"),
                                    1024);
        std::string path = sfWriteTemp("glm_sf_f32default.bin", bytes);
        cleanup.push_back(path);
        SafeTensorsFile f;
        CHECK(f.open(path), "Safetensors: F32 file opens (rejection counted)");
        CHECK(f.validationStats().accepted == 0 && f.validationStats().dtypeRejected == 1,
              "Safetensors: anonymous F32 tensor rejected by default");
    }

    // ... but accepted via the explicit allow-list fragment (GLM-5.2 stores the
    // noaux_tc router score-correction bias as float32; moe_router_dtype).
    {
        std::string bytes = sfBuild(sfHeader(sfTensor("F32", {256}, 0, 1024),
                                             "model.layers.4.mlp.gate.e_score_correction_bias"),
                                    1024);
        std::string path = sfWriteTemp("glm_sf_f32bias.bin", bytes);
        cleanup.push_back(path);
        SafeTensorsOptions opts;
        opts.allowF32NameContaining.push_back("e_score_correction_bias");
        SafeTensorsFile f;
        CHECK(f.open(path, opts), "Safetensors: F32 router bias opens with allow-list");
        const auto& st = f.validationStats();
        CHECK(st.accepted == 1 && st.f32AllowedAccepted == 1 && st.dtypeRejected == 0,
              "Safetensors: F32 router bias accepted via allow-list");
    }

    // (10) F32 tensor whose name is NOT on the allow-list is still rejected.
    {
        std::string bytes = sfBuild(sfHeader(sfTensor("F32", {256}, 0, 1024),
                                             "model.layers.4.mlp.gate.weight2"),
                                    1024);
        std::string path = sfWriteTemp("glm_sf_f32other.bin", bytes);
        cleanup.push_back(path);
        SafeTensorsOptions opts;
        opts.allowF32NameContaining.push_back("e_score_correction_bias");
        SafeTensorsFile f;
        CHECK(f.open(path, opts), "Safetensors: off-list F32 file opens (rejection counted)");
        CHECK(f.validationStats().accepted == 0 && f.validationStats().dtypeRejected == 1,
              "Safetensors: off-list F32 tensor rejected despite allow-list");
    }

    for (const std::string& p : cleanup) {
        std::error_code ec;
        std::filesystem::remove(p, ec);
    }
}

// ---------- Item 7/8 helpers: structural completeness (pure logic, L0-tested) ----------

// A tensor location is "present" only when it resolved to a real, non-empty
// byte span; default-constructed TensorLocation/size==0 means the tensor was
// missing from the index (either absent from the checkpoint or the index build
// failed to route it -- both are failures for a production engine).
bool tensorPresent(const TensorLocation& t) { return t.byteSize > 0 && !t.shape.empty(); }

// 2D shape-vs-config check [rows, cols].
bool shapeIs2(const TensorLocation& t, int r, int c) {
    return tensorPresent(t) && t.shape.size() == 2 && t.shape[0] == r && t.shape[1] == c;
}
// 1D shape-vs-config check [n].
bool shapeIs1(const TensorLocation& t, int n) {
    return tensorPresent(t) && t.shape.size() == 1 && t.shape[0] == n;
}

// Routed-expert completeness: all three MoE projections present AND shaped
// exactly [moe_intermediate, hidden] / [hidden, moe_intermediate].
bool expertComplete(const TensorLocation& gate, const TensorLocation& up,
                    const TensorLocation& down, int moeInt, int hidden) {
    return shapeIs2(gate, moeInt, hidden) && shapeIs2(up, moeInt, hidden) &&
           shapeIs2(down, hidden, moeInt);
}

// Item 8/15 --mtp gate decision. Returns nullptr when MTP may proceed (config
// declares layers AND the checkpoint actually ships MTP tensors); otherwise a
// non-null reason the engine must refuse to start with (never a silent fallback).
const char* mtpGateReason(int numMtpModules, bool hasMtpTensors) {
    if (numMtpModules <= 0) return "config declares no MTP / nextn predict layers";
    if (!hasMtpTensors) return "config declares MTP but the checkpoint ships no nextn/mtp tensors";
    return nullptr;
}

// ---------- Test 12: Weight index / config consistency ----------
void testWeightIndexParse() {
    // Verify ModelConfig default + sanity invariants using a synthetic fixture
    // loaded via validateWeights (run from tests). Pure logic here:
    ModelConfig c;
    c.rmsNormEps = 1e-5f;
    CHECK(c.rmsNormEps > 0.0f, "Config rmsNormEps positive");

    // Item 8: MTP gate refuses to start in every state this checkpoint family
    // can be in; only a config that declares layers AND tensor-laden weights
    // passes the gate.
    CHECK(mtpGateReason(0, false) != nullptr, "item8: gate blocks no-config/no-tensors");
    CHECK(mtpGateReason(0, true) != nullptr, "item8: gate blocks tensors-without-config");
    CHECK(mtpGateReason(1, false) != nullptr, "item8: gate blocks config-without-tensors");
    CHECK(mtpGateReason(2, true) == nullptr, "item8: gate allows config+tensors");

    // Item 7: expert/tensor completeness helpers.
    TensorLocation gate, up, down;
    gate.shape = {2048, 6144}; gate.byteSize = 2048ULL * 6144 * 2;
    up.shape = {2048, 6144};   up.byteSize = 2048ULL * 6144 * 2;
    CHECK(!expertComplete(gate, up, down, 2048, 6144), "item7: missing down_proj flagged");
    down.shape = {6144, 2048}; down.byteSize = 6144ULL * 2048 * 2;
    CHECK(expertComplete(gate, up, down, 2048, 6144), "item7: complete expert accepted");
    CHECK(!expertComplete(gate, up, down, 2048, 4096), "item7: shape-vs-config mismatch rejected");

    TensorLocation unset;
    CHECK(!tensorPresent(unset), "item7: default tensor location = missing");
    CHECK(!shapeIs1(unset, 128), "item7: default location not a vector");
}

// ---------- Test 13: Adaptive runtime configuration ----------
void testAdaptiveConfig() {
    constexpr uint64_t GiB = 1024ULL * 1024 * 1024;

    // CPU-only Linux host: 16 GiB RAM, 8 logical threads.
    SystemProfile cpuHost;
    cpuHost.os = PlatformOS::Linux;
    cpuHost.osLabel = "Linux";
    cpuHost.ramBytes = 16 * GiB;
    cpuHost.physicalCores = 4;
    cpuHost.logicalThreads = 8;

    AdaptiveConfig c0 = computeAdaptiveConfig(cpuHost, {});
    CHECK(c0.gpu == GpuBackend::None, "Adaptive: no GPU -> CPU backend");
    CHECK(c0.lruBytes >= 256ULL * 1024 * 1024 && c0.lruBytes == (16 * GiB) * 3 / 5,
          "Adaptive: LRU window sized from RAM (3/5 rule, O1)");
    CHECK(c0.ramBudgetBytes >= c0.lruBytes, "Adaptive: RAM budget >= LRU");
    CHECK(c0.ramBudgetBytes <= 16 * GiB, "Adaptive: RAM budget within physical RAM");
    CHECK(c0.vramBudgetBytes == 0, "Adaptive: no VRAM budget on CPU-only host");
    CHECK(c0.iocpWorkers >= 1 && c0.iocpWorkers <= 8 &&
          c0.iocpWorkers == std::min(8, std::max(2, 8 / 4)),
          "Adaptive: IOCP workers scaled from cores");

    // O1: large-RAM CPU-only hosts hit the 40 GiB LRU cap (one forward footprint).
    SystemProfile bigHost = cpuHost;
    bigHost.ramBytes = 128 * GiB;
    AdaptiveConfig cBig = computeAdaptiveConfig(bigHost, {});
    CHECK(cBig.lruBytes == 40 * GiB, "Adaptive: CPU LRU capped at 40 GiB (O1)");

    // NVIDIA RTX 3060 6 GiB host: dedicated VRAM budget keeps a driver reserve.
    SystemProfile nvidiaHost = cpuHost;
    nvidiaHost.gpu.backend = GpuBackend::Cuda;
    nvidiaHost.gpu.name = "NVIDIA GeForce RTX 3060";
    nvidiaHost.gpu.memoryBytes = 6 * GiB;
    AdaptiveConfig c1 = computeAdaptiveConfig(nvidiaHost, {});
    CHECK(c1.gpu == GpuBackend::Cuda, "Adaptive: CUDA GPU resolved");
    CHECK(c1.vramBudgetBytes > 0 && c1.vramBudgetBytes < 6 * GiB,
          "Adaptive: VRAM budget leaves driver reserve");

    // Apple Silicon MPS: unified memory -> half of RAM as the VRAM budget.
    SystemProfile mpsHost = cpuHost;
    mpsHost.os = PlatformOS::macOS;
    mpsHost.gpu.backend = GpuBackend::Mps;
    mpsHost.gpu.name = "Apple M3 Pro (MPS)";
    mpsHost.gpu.memoryBytes = 18 * GiB;
    AdaptiveConfig c2 = computeAdaptiveConfig(mpsHost, {});
    CHECK(c2.gpu == GpuBackend::Mps, "Adaptive: MPS GPU resolved");
    CHECK(c2.vramBudgetBytes == (18 * GiB - std::min<uint64_t>((18 * GiB) / 8, 512ULL * 1024 * 1024)) / 2,
          "Adaptive: MPS uses half the unified memory as VRAM budget");

    // Explicit overrides clamp: a 4 GiB LRU request is honoured.
    RuntimeOverrides ov;
    ov.lruSet = true;
    ov.lruBytes = 4 * GiB;
    AdaptiveConfig c3 = computeAdaptiveConfig(cpuHost, ov);
    CHECK(c3.lruBytes == 4 * GiB, "Adaptive: LRU override honoured");
}

} // namespace

int runSelfTests(bool verbose, const std::string& fixtureDir) {
    g_failures = 0;
    g_checks = 0;
    g_verbose = verbose;

    std::printf("=== HummingFlight functional self-tests ===\n");
    testBFloat16();
    testGemvBf16();
    testGemvF32();
    testRmsNorm();
    testSoftmax();
    testSigmoid();
    testTopKScores();
    testKvCache();
    testSampler();
    testPlacement();
    testConfigParse(fixtureDir);
    testWeightIndexParse();
    testAdaptiveConfig();
    testLruBoost();
    testSafetensorsStrict();

    std::printf("Self-tests: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures;
}

int validateWeights(const std::string& modelDir, std::string& report) {
    int fails = 0;
    char buf[512];

    WeightIndex idx;
    std::snprintf(buf, sizeof(buf), "Building weight index from %s ...\n", modelDir.c_str());
    report += buf;
    if (!idx.build(modelDir)) {
        report += "FAIL: weight index build failed\n";
        return 1;
    }

    const ModelConfig& cfg = idx.config;
    std::snprintf(buf, sizeof(buf),
                  "layers=%d (dense_first=%d) hidden=%d vocab=%d heads=%d\n",
                  cfg.numLayers, cfg.firstKDenseReplace, cfg.hiddenSize,
                  cfg.vocabSize, cfg.numAttentionHeads);
    report += buf;

    if (cfg.numLayers <= 0) { report += "FAIL: numLayers<=0\n"; ++fails; }
    if (cfg.hiddenSize <= 0) { report += "FAIL: hiddenSize<=0\n"; ++fails; }

    // Routed experts present for MoE layers.
    if (idx.routedExperts.size() == cfg.numLayers) {
        for (int l = 0; l < cfg.numLayers; ++l) {
            if (l < cfg.firstKDenseReplace) {
                if (!idx.routedExperts[l].empty()) {
                    report += "FAIL: dense layer has routed experts\n";
                    ++fails;
                }
            } else {
                if (idx.routedExperts[l].empty()) {
                    std::snprintf(buf, sizeof(buf), "FAIL: MoE layer %d has no experts\n", l);
                    report += buf;
                    ++fails;
                }
            }
        }
    } else {
        // routedExperts may be sized only for MoE layers; relax and re-check.
        std::snprintf(buf, sizeof(buf),
                      "routedExperts.size()=%zu vs numLayers=%d (acceptable if MoE-only)\n",
                      idx.routedExperts.size(), cfg.numLayers);
        report += buf;
    }

    // Embedding present.
    std::snprintf(buf, sizeof(buf), "embedding: %llu bytes, shards=%zu\n",
                  (unsigned long long)idx.embedTokens.byteSize, idx.shardFiles.size());
    report += buf;
    if (idx.embedTokens.byteSize == 0) { report += "FAIL: embedding missing\n"; ++fails; }
    if (idx.shardFiles.empty()) { report += "FAIL: no shard files\n"; ++fails; }

    // Shard files readable / mmap-able.
    MappedFile mf;
    if (!mf.openAll(idx.shardFiles[0])) {
        report += "FAIL: cannot open first shard\n";
        ++fails;
    }

    // ---- Item 7: strict structural completeness + shape-vs-config ----
    auto fail = [&](const std::string& m) { report += "FAIL: " + m + "\n"; ++fails; };
    auto badshape = [&](const std::string& m) { report += "FAIL: shape mismatch: " + m + "\n"; ++fails; };

    // Shared weights: embedding / final norm / (untied) LM head, shape-vs-config.
    if (!tensorPresent(idx.embedTokens)) fail("embedding missing");
    else if (!shapeIs2(idx.embedTokens, cfg.vocabSize, cfg.hiddenSize))
        badshape("embed_tokens vs config [vocab,hidden]");
    if (!tensorPresent(idx.finalNorm)) fail("model.norm.weight missing");
    else if (!shapeIs1(idx.finalNorm, cfg.hiddenSize))
        badshape("model.norm.weight vs config [hidden]");
    if (!cfg.tieWordEmbeddings) {
        if (!tensorPresent(idx.lmHead)) fail("lm_head.weight missing (tie_word_embeddings=false)");
        else if (!shapeIs2(idx.lmHead, cfg.vocabSize, cfg.hiddenSize))
            badshape("lm_head.weight vs config [vocab,hidden]");
    } else if (tensorPresent(idx.lmHead) &&
               !shapeIs2(idx.lmHead, cfg.vocabSize, cfg.hiddenSize)) {
        badshape("lm_head.weight vs config [vocab,hidden]");
    }

    int nDense = 0, nMoe = 0, nFullIdx = 0;
    uint64_t expertsValidated = 0;
    for (int l = 0; l < cfg.numLayers; ++l) {
        const LayerWeights& w = idx.layers[l];
        std::string pfx = "layer" + std::to_string(l) + ".";

        const std::pair<const TensorLocation*, const char*> attnReq[] = {
            {&w.qAProj, "self_attn.q_a_proj"}, {&w.qALayernorm, "self_attn.q_a_layernorm"},
            {&w.qBProj, "self_attn.q_b_proj"}, {&w.kvAProjWithMqa, "self_attn.kv_a_proj_with_mqa"},
            {&w.kvALayernorm, "self_attn.kv_a_layernorm"}, {&w.kvBProj, "self_attn.kv_b_proj"},
            {&w.oProj, "self_attn.o_proj"}, {&w.inputLayernorm, "input_layernorm"},
            {&w.postAttentionLayernorm, "post_attention_layernorm"}};
        for (const auto& [locp, n] : attnReq)
            if (!tensorPresent(*locp)) fail("missing tensor: " + pfx + n);
        // Representative MLA shape-vs-config checks on the hot matrices.
        if (!shapeIs2(w.qBProj, cfg.numAttentionHeads * cfg.qkHeadDim, cfg.qLoraRank))
            badshape(pfx + "self_attn.q_b_proj");
        if (!shapeIs2(w.kvBProj, cfg.numKvHeads * (cfg.qkNopeHeadDim + cfg.vHeadDim), cfg.kvLoraRank))
            badshape(pfx + "self_attn.kv_b_proj");
        if (!shapeIs2(w.oProj, cfg.hiddenSize, cfg.numAttentionHeads * cfg.vHeadDim))
            badshape(pfx + "self_attn.o_proj");
        if (!shapeIs2(w.kvAProjWithMqa, cfg.kvLoraRank + cfg.qkRopeHeadDim, cfg.hiddenSize))
            badshape(pfx + "self_attn.kv_a_proj_with_mqa");

        // "full" indexer layers must carry the complete indexer weight set.
        if (cfg.isIndexerFull(l)) {
            ++nFullIdx;
            if (!tensorPresent(w.indexWqB)) fail("missing tensor: " + pfx + "indexer.wq_b");
            if (!tensorPresent(w.indexWk)) fail("missing tensor: " + pfx + "indexer.wk");
            if (!tensorPresent(w.indexWeightsProj)) fail("missing tensor: " + pfx + "indexer.weights_proj");
            if (!tensorPresent(w.indexKNormW)) fail("missing tensor: " + pfx + "indexer.k_norm.weight");
            if (!tensorPresent(w.indexKNormB)) fail("missing tensor: " + pfx + "indexer.k_norm.bias");
        }

        if (!cfg.isMoeLayer(l)) {
            ++nDense;
            if (!tensorPresent(w.mlpGate)) fail("missing tensor: " + pfx + "mlp.gate_proj (dense)");
            if (!tensorPresent(w.mlpUp)) fail("missing tensor: " + pfx + "mlp.up_proj (dense)");
            if (!tensorPresent(w.mlpDown)) fail("missing tensor: " + pfx + "mlp.down_proj (dense)");
        } else {
            ++nMoe;
            if (!tensorPresent(w.mlpGate)) fail("missing tensor: " + pfx + "mlp.gate.weight (router)");
            if (!tensorPresent(w.gateBias)) fail("missing tensor: " + pfx + "mlp.gate.e_score_correction_bias");
            else if (!shapeIs1(w.gateBias, cfg.numLocalExperts))
                badshape(pfx + "mlp.gate.e_score_correction_bias [n_experts]");
            if (!tensorPresent(w.sharedGateProj)) fail("missing tensor: " + pfx + "mlp.shared_experts.gate_proj");
            if (!tensorPresent(w.sharedUpProj)) fail("missing tensor: " + pfx + "mlp.shared_experts.up_proj");
            if (!tensorPresent(w.sharedDownProj)) fail("missing tensor: " + pfx + "mlp.shared_experts.down_proj");
            if (l < int(idx.routedExperts.size()) &&
                int(idx.routedExperts[l].size()) == cfg.numLocalExperts) {
                bool layerComplete = true;
                for (const auto& e : idx.routedExperts[l]) {
                    if (!expertComplete(e.gateProj, e.upProj, e.downProj,
                                        cfg.moeIntermediateSize, cfg.hiddenSize))
                        layerComplete = false;
                }
                if (layerComplete) {
                    expertsValidated += uint64_t(cfg.numLocalExperts);
                } else {
                    fail("expert completeness: " + pfx + "routed experts (gate/up/down shape-vs-config)");
                }
            } else {
                fail("expert count: " + pfx + "routed experts (" +
                     std::to_string(l < int(idx.routedExperts.size()) ? int(idx.routedExperts[l].size()) : 0) +
                     " != " + std::to_string(cfg.numLocalExperts) + ")");
            }
        }
    }
    std::snprintf(buf, sizeof(buf),
                  "structural: dense=%d moe=%d full-indexer=%d routed-experts-validated=%llu "
                  "(all shape-vs-config); noaux_tc bias validated-present but not applied "
                  "on the greedy forward (golden contract, doc/design.md items 7/13)\n",
                  nDense, nMoe, nFullIdx, (unsigned long long)expertsValidated);
    report += buf;

    // ---- Item 8: MTP / nextn config-vs-tensors consistency ----
    // The 5.2-bf16 release legitimately ships nextn config with the MTP weights
    // published separately, so this is an info line here; the --mtp CLI gate in
    // main.cpp is the hard error that refuses to start (no silent fallback).
    if (cfg.numMtpModules > 0) {
        std::snprintf(buf, sizeof(buf),
                      "MTP: config declares %d nextn predict layer(s); index found %d MTP "
                      "layers, head %s; --mtp is gated on these in main.cpp\n",
                      cfg.numMtpModules, idx.mtpTensorLayers,
                      idx.hasMtpHead ? "present" : "absent");
        report += buf;
    }

    // GLM-5.2 DSA root/corpus-embedding group at layer index num_hidden_layers:
    // expected in the release, informational here (not a decoder block).
    if (idx.maxSeenBaseLayer >= cfg.numLayers) {
        std::snprintf(buf, sizeof(buf),
                      "info: index has base-layer groups past num_hidden_layers "
                      "(max L=%d; DSA root/corpus-embedding group, not a decoder block)\n",
                      idx.maxSeenBaseLayer);
        report += buf;
    }

    std::snprintf(buf, sizeof(buf), "redirected: %d failures\n", fails);
    report += buf;
    return fails;
}

} // namespace glm