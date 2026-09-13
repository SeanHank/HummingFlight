#pragma once

#include "model/config.h"
#include "storage/mapped_file.h"
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace glm {

// Physical location of a single weight tensor (which shard file, which offset)
struct TensorLocation {
    std::string shardPath;
    uint64_t byteOffset = 0;   // Absolute byte offset within the shard file (including header)
    uint64_t byteSize = 0;
    std::vector<int> shape;
    std::string dtype;         // "BF16" / "F32"
    int numel() const {
        int n = 1;
        for (int s : shape) n *= s;
        return n;
    }
};

// Complete weights of a routed expert (gate_proj, up_proj, down_proj matrices)
struct ExpertWeights {
    TensorLocation gateProj;   // [moe_intermediate, hidden]
    TensorLocation upProj;     // [moe_intermediate, hidden]
    TensorLocation downProj;   // [hidden, moe_intermediate]
    uint64_t totalBytes() const {
        return gateProj.byteSize + upProj.byteSize + downProj.byteSize;
    }
};

// Dense layer weight collection (MLA attention + dense/sparse MLP)
struct LayerWeights {
    // RMSNorm
    TensorLocation inputLayernorm;       // [hidden]
    TensorLocation postAttentionLayernorm; // [hidden]

    // MLA Attention
    TensorLocation qAProj;               // [q_lora_rank, hidden]
    TensorLocation qALayernorm;          // [q_lora_rank]
    TensorLocation qBProj;               // [heads * qk_head_dim, q_lora_rank]
    TensorLocation kvAProjWithMqa;       // [kv_lora_rank + qk_rope_head_dim, hidden]
    TensorLocation kvALayernorm;         // [kv_lora_rank]
    TensorLocation kvBProj;              // [heads * (qk_nope + v), kv_lora_rank]
    TensorLocation oProj;                // [hidden, heads * v_head_dim]

    // DSA indexer weight locations (present on "full" indexer layers only)
    TensorLocation indexWqB;           // [index_n_heads * index_head_dim, q_lora_rank]
    TensorLocation indexWk;            // [index_head_dim, hidden]
    TensorLocation indexWeightsProj;   // [index_n_heads, hidden]
    TensorLocation indexKNormW;        // [index_head_dim]
    TensorLocation indexKNormB;        // [index_head_dim]

    // Dense MLP (first 3 layers) or Shared Experts (MoE layers)
    bool isMoe = false;
    TensorLocation mlpGate;              // dense: [intermediate, hidden]; moe: gate.weight
    TensorLocation mlpUp;                // dense: [intermediate, hidden]
    TensorLocation mlpDown;              // dense: [hidden, intermediate]
    TensorLocation gateBias;             // e_score_correction_bias for noaux_tc
    TensorLocation sharedGateProj;       // shared_experts of moe layers
    TensorLocation sharedUpProj;
    TensorLocation sharedDownProj;
};

struct WeightIndex {
    ModelConfig config;
    std::string modelDir;

    // Dense weights per layer
    std::vector<LayerWeights> layers;

    // Routed expert index: [layer][expert_id] -> ExpertWeights
    // Only MoE layers have this (layer >= first_k_dense_replace)
    std::vector<std::vector<ExpertWeights>> routedExperts;

    // Embedding & LM Head
    TensorLocation embedTokens;          // [vocab, hidden] BF16
    TensorLocation lmHead;               // [vocab, hidden] (independent when tie_word_embeddings=false)
    TensorLocation finalNorm;            // [hidden] model.norm.weight

    // MTP (_model.nextn_predict_layers.<i>.*_ / _mtp_layers.<i>._ prefix scan).
    // No MTP-bearing checkpoint is available in this environment (GLM-5.2-bf16
    // config curs `num_nextn_predict_layers: 1` but ships zero MTP tensors), so
    // the index records presence/count only and main.cpp gates --mtp on them.
    bool hasMtpTensors = false;          // any nextn_predict_layers.* / mtp_layers.* tensor
    bool hasMtpHead = false;             // nextn_predict_head* / mtp_head* present
    int mtpTensorLayers = 0;             // distinct MTP layer indices discovered

    // Largest "model.layers.<i>.*" group index seen. GLM-5.2 ships a DSA
    // root/corpus-embedding group at index num_hidden_layers (eh_proj/enorm/
    // hnorm + full decoder block) that is not a decoder layer the engine runs;
    // --check-weights surfaces it as information rather than failing.
    int maxSeenBaseLayer = 0;

    // List of all shard files
    std::vector<std::string> shardFiles;

    // mmap cache of shard files (path -> MappedFile)
    // Opened on demand, reused on repeated access
    std::unordered_map<std::string, std::unique_ptr<MappedFile>> mmapCache;

    // Build full index from model directory
    bool build(const std::string& modelDir);

    // Print index statistics
    void printStats() const;

    // Get mmap view of a shard (auto-cached)
    const uint8_t* getShardView(const std::string& shardPath);
};

} // namespace glm
