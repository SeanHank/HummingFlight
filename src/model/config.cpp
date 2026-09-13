#include "model/config.h"
#include "third_party/picojson.h"
#include "utils/logger.h"
#include <algorithm>
#include <fstream>
#include <iterator>
#include <string>

namespace glm {

namespace {

// PicoJSON type-safe scalar readers. Keys that are missing or have the wrong
// JSON type leave the caller's struct default untouched, so config files that
// omit optional fields keep prior behaviour. This replaces the previous
// string-search parser with a real JSON parser (item 6).
bool getInt(const picojson::object& o, const char* key, int& out) {
    auto it = o.find(key);
    if (it == o.end() || !it->second.is<double>()) return false;
    out = int(it->second.get<double>());
    return true;
}

bool getFloat(const picojson::object& o, const char* key, float& out) {
    auto it = o.find(key);
    if (it == o.end() || !it->second.is<double>()) return false;
    out = float(it->second.get<double>());
    return true;
}

bool getBool(const picojson::object& o, const char* key, bool& out) {
    auto it = o.find(key);
    if (it == o.end() || !it->second.is<bool>()) return false;
    out = it->second.get<bool>();
    return true;
}

bool getString(const picojson::object& o, const char* key, std::string& out) {
    auto it = o.find(key);
    if (it == o.end() || !it->second.is<std::string>()) return false;
    out = it->second.get<std::string>();
    return true;
}

// String array, e.g. ["dense","sparse",...]. Missing or non-array keys return false.
bool getStringArray(const picojson::object& o, const char* key, std::vector<std::string>& out) {
    auto it = o.find(key);
    if (it == o.end() || !it->second.is<picojson::array>()) return false;
    out.clear();
    for (const auto& item : it->second.get<picojson::array>()) {
        if (item.is<std::string>()) out.push_back(item.get<std::string>());
    }
    return true;
}

} // namespace

bool ModelConfig::loadFromFile(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        GLM_LOG_ERROR("Cannot open config.json: " + path);
        return false;
    }
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return loadFromString(content);
}

bool ModelConfig::loadFromString(const std::string& content) {
    picojson::value root;
    const std::string err = picojson::parse(root, content);
    if (!err.empty()) {
        GLM_LOG_ERROR("config.json is not valid JSON: " + err);
        return false;
    }
    if (!root.is<picojson::object>()) {
        GLM_LOG_ERROR("config.json root must be a JSON object");
        return false;
    }
    const picojson::object& o = root.get<picojson::object>();

    // Basic
    getInt(o, "hidden_size", hiddenSize);
    getInt(o, "num_hidden_layers", numLayers);
    getInt(o, "vocab_size", vocabSize);
    getInt(o, "intermediate_size", intermediateSize);
    getInt(o, "max_position_embeddings", maxPositionEmbeddings);

    // MLA
    getInt(o, "num_attention_heads", numAttentionHeads);
    getInt(o, "num_key_value_heads", numKvHeads);
    getInt(o, "q_lora_rank", qLoraRank);
    getInt(o, "kv_lora_rank", kvLoraRank);
    getInt(o, "qk_head_dim", qkHeadDim);
    getInt(o, "qk_nope_head_dim", qkNopeHeadDim);
    getInt(o, "qk_rope_head_dim", qkRopeHeadDim);
    getInt(o, "v_head_dim", vHeadDim);
    getBool(o, "rope_interleave", ropeInterleave);

    // Indexer (DSA)
    getInt(o, "index_head_dim", indexHeadDim);
    getInt(o, "index_n_heads", indexNHeads);
    getInt(o, "index_topk", indexTopk);
    getInt(o, "index_topk_freq", indexTopkFreq);
    if (indexTopkFreq <= 0) indexTopkFreq = 4;
    getInt(o, "index_skip_topk_offset", indexSkipTopkOffset);

    // MoE
    getInt(o, "n_routed_experts", numLocalExperts);
    getInt(o, "num_experts_per_tok", numExpertsPerTok);
    getInt(o, "n_shared_experts", nSharedExperts);
    getInt(o, "moe_intermediate_size", moeIntermediateSize);
    getInt(o, "first_k_dense_replace", firstKDenseReplace);
    getString(o, "scoring_func", scoringFunc);
    getString(o, "topk_method", topkMethod);
    getFloat(o, "routed_scaling_factor", routedScalingFactor);
    getBool(o, "norm_topk_prob", normTopkProb);
    getInt(o, "topk_group", topkGroup);

    // MTP (Multi-Token Prediction heads, GLM-4.5/GLM-5.2).
    // GLM-5.2 "GlmMoeDsaForCausalLM" uses num_nextn_predict_layers (the count of
    // next-next-token prediction layers) plus index_share_for_mtp_iteration.
    // Legacy GLM-4.5 spelled this num_mtp_modules / num_mtp_layers_per_module.
    // The 5.2 key wins when both are present.
    if (!getInt(o, "num_nextn_predict_layers", numMtpModules)) {
        getInt(o, "num_mtp_modules", numMtpModules);
    }
    getInt(o, "num_mtp_layers_per_module", numMtpLayersPerModule);
    getInt(o, "mtp_hidden_size", mtpHiddenSize);
    getInt(o, "mtp_intermediate_size", mtpIntermediateSize);
    getString(o, "mtp_activation", mtpActivation);
    getStringArray(o, "mtp_layer_types", mtpLayerTypes);
    getBool(o, "index_share_for_mtp_iteration", indexShareForMtpIteration);
    if (mtpLayerTypes.empty() && numMtpModules > 0) {
        mtpLayerTypes.assign(size_t(numMtpModules), "shared");
    }

    // Norm/Act
    getFloat(o, "rms_norm_eps", rmsNormEps);
    getString(o, "hidden_act", hiddenAct);

    // RoPE
    getInt(o, "rope_theta", ropeTheta);
    getBool(o, "tie_word_embeddings", tieWordEmbeddings);

    // Layer types
    getStringArray(o, "mlp_layer_types", mlpLayerTypes);

    // Indexer types: use the checkpoint's explicit list when present; otherwise
    // derive the deterministic schedule matching the reference glm_moe_dsa
    // __post_init__ (skip = index_skip_topk_offset, freq = index_topk_freq):
    // "full" when i < skip, or (i - (skip - 1)) % freq == 0; checkpoint pattern
    // = {0,1,2} + {6,10,...} (i.e. skip=3, freq=4). Everything else "shared".
    getStringArray(o, "indexer_types", indexerTypes);
    if (indexerTypes.empty() && numLayers > 0) {
        indexerTypes.reserve(numLayers);
        int freq = std::max(indexTopkFreq, 1);
        int skip = std::max(indexSkipTopkOffset, 1);
        for (int i = 0; i < numLayers; ++i) {
            bool full = (i < skip) || ((i - (skip - 1)) % freq == 0);
            indexerTypes.push_back(full ? "full" : "shared");
        }
    }

    if (hiddenSize == 0 || numLayers == 0) {
        GLM_LOG_ERROR("config.json parse failed: key fields are empty");
        return false;
    }
    return true;
}

void ModelConfig::printSummary() const {
    GLM_LOG_INFO("=== GLM-5.2 (GlmMoeDsaForCausalLM) config ===");
    GLM_LOG_INFO("hidden_size: " + std::to_string(hiddenSize));
    GLM_LOG_INFO("num_layers: " + std::to_string(numLayers) +
                 " (dense: " + std::to_string(firstKDenseReplace) +
                 ", moe: " + std::to_string(numLayers - firstKDenseReplace) + ")");
    GLM_LOG_INFO("vocab_size: " + std::to_string(vocabSize));
    GLM_LOG_INFO("-- MLA Attention --");
    GLM_LOG_INFO("heads: " + std::to_string(numAttentionHeads) + ", kv_heads: " + std::to_string(numKvHeads));
    GLM_LOG_INFO("q_lora_rank: " + std::to_string(qLoraRank) + ", kv_lora_rank: " + std::to_string(kvLoraRank));
    GLM_LOG_INFO("qk_nope: " + std::to_string(qkNopeHeadDim) + ", qk_rope: " + std::to_string(qkRopeHeadDim) +
                 ", v: " + std::to_string(vHeadDim));
    GLM_LOG_INFO("-- DSA Indexer --");
    int nFull = 0;
    for (const auto& t : indexerTypes) if (t == "full") nFull++;
    GLM_LOG_INFO("heads: " + std::to_string(indexNHeads) + ", head_dim: " + std::to_string(indexHeadDim) +
                 ", topk: " + std::to_string(indexTopk) + ", full_layers: " + std::to_string(nFull) +
                 "/" + std::to_string(numLayers));
    GLM_LOG_INFO("-- MoE --");
    GLM_LOG_INFO("experts: " + std::to_string(numLocalExperts) + ", top_k: " + std::to_string(numExpertsPerTok) +
                 ", shared: " + std::to_string(nSharedExperts));
    GLM_LOG_INFO("moe_intermediate: " + std::to_string(moeIntermediateSize) +
                 ", dense_intermediate: " + std::to_string(intermediateSize));
    GLM_LOG_INFO("scoring: " + scoringFunc + ", topk_method: " + topkMethod +
                 ", scale: " + std::to_string(routedScalingFactor));
    if (numMtpModules > 0) {
        GLM_LOG_INFO("-- MTP --");
        GLM_LOG_INFO("modules/nextn_layers: " + std::to_string(numMtpModules) +
                     ", layers/module: " + std::to_string(numMtpLayersPerModule) +
                     ", hidden: " + std::to_string(mtpHiddenSize));
        GLM_LOG_INFO("index_share_for_mtp_iteration: " +
                     std::string(indexShareForMtpIteration ? "yes" : "no"));
    }
}

} // namespace glm