#include "model/config.h"
#include "utils/logger.h"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace glm {

// Minimal JSON value extraction
static std::string extractJsonValue(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\"";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos + pattern.size());
    if (pos == std::string::npos) return "";
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    if (pos >= json.size()) return "";
    if (json[pos] == '"') {
        size_t end = json.find('"', pos + 1);
        return json.substr(pos + 1, end - pos - 1);
    }
    size_t end = pos;
    while (end < json.size() && json[end] != ',' && json[end] != '}' && json[end] != '\n') end++;
    return json.substr(pos, end - pos);
}

static int toInt(const std::string& s) { try { return std::stoi(s); } catch (...) { return 0; } }
static float toFloat(const std::string& s) { try { return std::stof(s); } catch (...) { return 0.0f; } }
static bool toBool(const std::string& s) { return s == "true" || s == "1"; }

// Parse string array ["dense","sparse",...]
static std::vector<std::string> toStringArray(const std::string& json, const std::string& key) {
    std::vector<std::string> result;
    std::string pat = "\"" + key + "\"";
    size_t pos = json.find(pat);
    if (pos == std::string::npos) return result;
    pos = json.find('[', pos);
    if (pos == std::string::npos) return result;
    pos++;
    while (pos < json.size() && json[pos] != ']') {
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == ',' || json[pos] == '\n')) pos++;
        if (pos >= json.size() || json[pos] == ']' || json[pos] != '"') { if (pos < json.size() && json[pos] != ']') pos++; continue; }
        size_t end = json.find('"', pos + 1);
        if (end == std::string::npos) break;
        result.push_back(json.substr(pos + 1, end - pos - 1));
        pos = end + 1;
    }
    return result;
}

bool ModelConfig::loadFromFile(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        GLM_LOG_ERROR("Cannot open config.json: " + path);
        return false;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    std::string c = ss.str();

    // Basic
    hiddenSize = toInt(extractJsonValue(c, "hidden_size"));
    numLayers = toInt(extractJsonValue(c, "num_hidden_layers"));
    vocabSize = toInt(extractJsonValue(c, "vocab_size"));
    intermediateSize = toInt(extractJsonValue(c, "intermediate_size"));
    maxPositionEmbeddings = toInt(extractJsonValue(c, "max_position_embeddings"));

    // MLA
    numAttentionHeads = toInt(extractJsonValue(c, "num_attention_heads"));
    numKvHeads = toInt(extractJsonValue(c, "num_key_value_heads"));
    qLoraRank = toInt(extractJsonValue(c, "q_lora_rank"));
    kvLoraRank = toInt(extractJsonValue(c, "kv_lora_rank"));
    qkHeadDim = toInt(extractJsonValue(c, "qk_head_dim"));
    qkNopeHeadDim = toInt(extractJsonValue(c, "qk_nope_head_dim"));
    qkRopeHeadDim = toInt(extractJsonValue(c, "qk_rope_head_dim"));
    vHeadDim = toInt(extractJsonValue(c, "v_head_dim"));
    ropeInterleave = toBool(extractJsonValue(c, "rope_interleave"));

    // Indexer
    indexHeadDim = toInt(extractJsonValue(c, "index_head_dim"));
    indexNHeads = toInt(extractJsonValue(c, "index_n_heads"));
    indexTopk = toInt(extractJsonValue(c, "index_topk"));
    indexTopkFreq = toInt(extractJsonValue(c, "index_topk_freq"));
    if (indexTopkFreq <= 0) indexTopkFreq = 4;
    indexSkipTopkOffset = toInt(extractJsonValue(c, "index_skip_topk_offset"));

    // MoE
    numLocalExperts = toInt(extractJsonValue(c, "n_routed_experts"));
    numExpertsPerTok = toInt(extractJsonValue(c, "num_experts_per_tok"));
    nSharedExperts = toInt(extractJsonValue(c, "n_shared_experts"));
    moeIntermediateSize = toInt(extractJsonValue(c, "moe_intermediate_size"));
    firstKDenseReplace = toInt(extractJsonValue(c, "first_k_dense_replace"));
    scoringFunc = extractJsonValue(c, "scoring_func");
    topkMethod = extractJsonValue(c, "topk_method");
    routedScalingFactor = toFloat(extractJsonValue(c, "routed_scaling_factor"));
    normTopkProb = toBool(extractJsonValue(c, "norm_topk_prob"));
    topkGroup = toInt(extractJsonValue(c, "topk_group"));

    // Norm/Act
    rmsNormEps = toFloat(extractJsonValue(c, "rms_norm_eps"));
    hiddenAct = extractJsonValue(c, "hidden_act");

    // RoPE (nested within rope_parameters)
    ropeTheta = toInt(extractJsonValue(c, "rope_theta"));
    tieWordEmbeddings = toBool(extractJsonValue(c, "tie_word_embeddings"));

    // Layer types
    mlpLayerTypes = toStringArray(c, "mlp_layer_types");

    // Indexer types: use the checkpoint's explicit list when present; otherwise
    // derive the deterministic schedule matching the reference glm_moe_dsa
    // __post_init__ (skip = index_skip_topk_offset, freq = index_topk_freq):
    // "full" when i < skip, or (i - (skip - 1)) % freq == 0; checkpoint pattern
    // = {0,1,2} + {6,10,...} (i.e. skip=3, freq=4). Everything else "shared".
    indexerTypes = toStringArray(c, "indexer_types");
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
}

} // namespace glm
