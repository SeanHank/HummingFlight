#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace glm {

// GLM-5.2 (GlmMoeDsaForCausalLM) model hyperparameters
// Architecture: DeepSeek-V3 style MLA (Multi-head Latent Attention) + MoE
struct ModelConfig {
    // ---- Basic structure ----
    int hiddenSize = 0;           // hidden_size = 6144
    int numLayers = 0;            // num_hidden_layers = 78
    int vocabSize = 0;            // vocab_size = 154880
    int intermediateSize = 0;     // intermediate_size = 12288 (dense FFN)
    int maxPositionEmbeddings = 0;

    // ---- MLA Attention (DeepSeek style) ----
    int numAttentionHeads = 0;    // num_attention_heads = 64
    int numKvHeads = 0;           // num_key_value_heads = 64
    int qLoraRank = 0;            // q_lora_rank = 2048 (low-rank of Q)
    int kvLoraRank = 0;           // kv_lora_rank = 512 (low-rank compression of KV)
    int qkHeadDim = 0;            // qk_head_dim = 256 (total QK dim)
    int qkNopeHeadDim = 0;        // qk_nope_head_dim = 192 (non-RoPE part)
    int qkRopeHeadDim = 0;        // qk_rope_head_dim = 64 (RoPE part)
    int vHeadDim = 0;             // v_head_dim = 256
    bool ropeInterleave = true;   // rope_interleave

    // ---- Indexer (DSA: dynamic sparse attention; reference: glm_moe_dsa) ----
    int indexHeadDim = 0;         // index_head_dim = 128
    int indexNHeads = 0;          // index_n_heads = 32
    int indexTopk = 0;            // index_topk = 2048
    int indexTopkFreq = 4;        // index_topk_freq (fallback schedule spacing)
    int indexSkipTopkOffset = 3;  // index_skip_topk_offset (fallback schedule offset)
    // Per-layer indexer mode, "full" or "shared" (indexer_types; built deterministically
    // from index_topk_freq / index_skip_topk_offset when the checkpoint omits it).
    std::vector<std::string> indexerTypes;

    // Whether layer L runs its own indexer (true) or reuses the previous full
    // layer's top-k positions (false).
    bool isIndexerFull(int layer) const {
        if (layer >= 0 && layer < int(indexerTypes.size())) return indexerTypes[layer] == "full";
        if (layer == 0) return true;  // first layer is always full
        return (std::max(layer - 1, 0) % std::max(indexTopkFreq, 1)) == 0;
    }

    // ---- MoE structure ----
    int numLocalExperts = 0;      // n_routed_experts = 256
    int numExpertsPerTok = 0;     // num_experts_per_tok = 8 (Top-K)
    int nSharedExperts = 0;       // n_shared_experts = 1
    int moeIntermediateSize = 0;  // moe_intermediate_size = 2048 (single expert FFN)
    int firstKDenseReplace = 0;   // first_k_dense_replace = 3 (first 3 layers dense)
    std::string scoringFunc = "sigmoid";   // sigmoid
    std::string topkMethod = "noaux_tc";   // noaux_tc
    float routedScalingFactor = 1.0f;      // 2.5
    bool normTopkProb = true;
    int topkGroup = 1;

    // ---- Normalization and activation ----
    float rmsNormEps = 1e-5f;
    std::string hiddenAct = "silu";

    // ---- RoPE ----
    int ropeTheta = 500000;
    bool tieWordEmbeddings = false;

    // ---- Layer types (dense / sparse) ----
    std::vector<std::string> mlpLayerTypes;  // mlp_layer_types

    // Parse from file
    bool loadFromFile(const std::string& path);

    // Determine whether layer L is a MoE layer
    bool isMoeLayer(int layer) const {
        if (layer < int(mlpLayerTypes.size())) return mlpLayerTypes[layer] == "sparse";
        return layer >= firstKDenseReplace;
    }

    // Print config summary
    void printSummary() const;
};

} // namespace glm
