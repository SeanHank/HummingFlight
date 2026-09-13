#include "model/weight_index.h"
#include "model/safetensors.h"
#include "utils/logger.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace glm {

namespace fs = std::filesystem;

// Extract tensor -> shard filename mapping from index.json
static std::unordered_map<std::string, std::string> parseIndexJson(const std::string& path) {
    std::unordered_map<std::string, std::string> result;
    std::ifstream f(path);
    if (!f.is_open()) return result;
    std::stringstream ss;
    ss << f.rdbuf();
    std::string content = ss.str();

    size_t wm = content.find("\"weight_map\"");
    if (wm == std::string::npos) return result;
    size_t pos = content.find('{', wm);
    if (pos == std::string::npos) return result;
    pos++;

    while (pos < content.size() && content[pos] != '}') {
        while (pos < content.size() && (content[pos] == ' ' || content[pos] == ',' || content[pos] == '\n' || content[pos] == '\t')) pos++;
        if (pos >= content.size() || content[pos] == '}' || content[pos] != '"') { if (pos < content.size() && content[pos] != '}') pos++; continue; }
        size_t nameEnd = content.find('"', pos + 1);
        if (nameEnd == std::string::npos) break;
        std::string tensorName = content.substr(pos + 1, nameEnd - pos - 1);
        size_t colon = content.find(':', nameEnd);
        if (colon == std::string::npos) break;
        size_t shardStart = content.find('"', colon);
        size_t shardEnd = content.find('"', shardStart + 1);
        if (shardEnd == std::string::npos) break;
        std::string shardName = content.substr(shardStart + 1, shardEnd - shardStart - 1);
        result[tensorName] = shardName;
        pos = shardEnd + 1;
    }
    return result;
}

// Extract layer number: "model.layers.10.xxx" -> 10
static int extractLayerNum(const std::string& name) {
    size_t p = name.find("layers.");
    if (p == std::string::npos) return -1;
    try { return std::stoi(name.substr(p + 7)); } catch (...) { return -1; }
}

// Extract expert number: "model.layers.10.mlp.experts.123.xxx" -> 123
static int extractExpertNum(const std::string& name) {
    size_t p = name.find("experts.");
    if (p == std::string::npos) return -1;
    try { return std::stoi(name.substr(p + 8)); } catch (...) { return -1; }
}

bool WeightIndex::build(const std::string& dir) {
    modelDir = dir;

    // 1. config.json
    if (!config.loadFromFile(dir + "/config.json")) return false;

    // 2. Enumerate shards
    for (auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() == ".safetensors") {
            shardFiles.push_back(entry.path().string());
        }
    }
    std::sort(shardFiles.begin(), shardFiles.end());
    if (shardFiles.empty()) {
        GLM_LOG_ERROR("No .safetensors files found: " + dir);
        return false;
    }
    GLM_LOG_INFO("Found " + std::to_string(shardFiles.size()) + " shard files");

    // 3. Initialize layers and routedExperts containers
    int numLayers = config.numLayers;
    layers.resize(numLayers);
    routedExperts.resize(numLayers);
    for (int i = 0; i < numLayers; ++i) {
        if (config.isMoeLayer(i)) {
            routedExperts[i].resize(config.numLocalExperts);
            layers[i].isMoe = true;
        }
    }

    // 4. Build shard name -> path mapping
    std::unordered_map<std::string, std::string> shardNameToPath;
    for (const auto& p : shardFiles) {
        fs::path fsp(p);
        shardNameToPath[fsp.filename().string()] = p;
    }

    // 5. Try index.json first (fast path: skip traversing all headers)
    std::string indexPath = dir + "/model.safetensors.index.json";
    auto indexMap = parseIndexJson(indexPath);

    if (!indexMap.empty()) {
        GLM_LOG_INFO("Using index.json for fast startup (" + std::to_string(indexMap.size()) + " tensors mapped)");

        // For each tensor in index.json, we need shape/dtype/offset from the shard header
        // Group by shard to minimize header parsing
        std::unordered_map<std::string, std::vector<std::string>> shardToTensors;
        for (const auto& [tensorName, shardName] : indexMap) {
            shardToTensors[shardName].push_back(tensorName);
        }

        int totalExperts = 0;
        for (const auto& [shardName, tensorNames] : shardToTensors) {
            auto pathIt = shardNameToPath.find(shardName);
            if (pathIt == shardNameToPath.end()) continue;

            SafeTensorsFile st;
            if (!st.open(pathIt->second)) continue;

            for (const auto& tensorName : tensorNames) {
                const TensorInfo* info = st.find(tensorName);
                if (!info) continue;

                TensorLocation loc;
                loc.shardPath = pathIt->second;
                loc.byteOffset = st.headerSize() + info->dataBegin;
                loc.byteSize = info->byteSize();
                loc.shape = info->shape;
                loc.dtype = info->dtype;

                // Dispatch to corresponding location
                int layer = extractLayerNum(tensorName);

                if (tensorName == "model.embed_tokens.weight") {
                    embedTokens = loc;
                } else if (tensorName == "lm_head.weight") {
                    lmHead = loc;
                } else if (tensorName == "model.norm.weight") {
                    finalNorm = loc;
                } else if (layer >= 0 && layer < numLayers) {
                    auto& lw = layers[layer];

                    // RMSNorm
                    if (tensorName.find("input_layernorm.weight") != std::string::npos)
                        lw.inputLayernorm = loc;
                    else if (tensorName.find("post_attention_layernorm.weight") != std::string::npos)
                        lw.postAttentionLayernorm = loc;

                    // MLA Attention
                    else if (tensorName.find("self_attn.q_a_proj.weight") != std::string::npos)
                        lw.qAProj = loc;
                    else if (tensorName.find("self_attn.q_a_layernorm.weight") != std::string::npos)
                        lw.qALayernorm = loc;
                    else if (tensorName.find("self_attn.q_b_proj.weight") != std::string::npos)
                        lw.qBProj = loc;
                    else if (tensorName.find("self_attn.kv_a_proj_with_mqa.weight") != std::string::npos)
                        lw.kvAProjWithMqa = loc;
                    else if (tensorName.find("self_attn.kv_a_layernorm.weight") != std::string::npos)
                        lw.kvALayernorm = loc;
                    else if (tensorName.find("self_attn.kv_b_proj.weight") != std::string::npos)
                        lw.kvBProj = loc;
                    else if (tensorName.find("self_attn.o_proj.weight") != std::string::npos)
                        lw.oProj = loc;

                    // Indexer
                    else if (tensorName.find("self_attn.indexer.wq_b.weight") != std::string::npos)
                        lw.indexWqB = loc;
                    else if (tensorName.find("self_attn.indexer.wk.weight") != std::string::npos)
                        lw.indexWk = loc;
                    else if (tensorName.find("self_attn.indexer.weights_proj.weight") != std::string::npos)
                        lw.indexWeightsProj = loc;
                    else if (tensorName.find("self_attn.indexer.k_norm.weight") != std::string::npos)
                        lw.indexKNormW = loc;
                    else if (tensorName.find("self_attn.indexer.k_norm.bias") != std::string::npos)
                        lw.indexKNormB = loc;

                    // Dense MLP
                    else if (!lw.isMoe && tensorName.find("mlp.gate_proj.weight") != std::string::npos)
                        lw.mlpGate = loc;
                    else if (!lw.isMoe && tensorName.find("mlp.up_proj.weight") != std::string::npos)
                        lw.mlpUp = loc;
                    else if (!lw.isMoe && tensorName.find("mlp.down_proj.weight") != std::string::npos)
                        lw.mlpDown = loc;

                    // MoE gate
                    else if (lw.isMoe && tensorName.find("mlp.gate.weight") != std::string::npos)
                        lw.mlpGate = loc;
                    else if (lw.isMoe && tensorName.find("mlp.gate.e_score_correction_bias") != std::string::npos)
                        lw.gateBias = loc;

                    // Shared experts
                    else if (lw.isMoe && tensorName.find("mlp.shared_experts.gate_proj.weight") != std::string::npos)
                        lw.sharedGateProj = loc;
                    else if (lw.isMoe && tensorName.find("mlp.shared_experts.up_proj.weight") != std::string::npos)
                        lw.sharedUpProj = loc;
                    else if (lw.isMoe && tensorName.find("mlp.shared_experts.down_proj.weight") != std::string::npos)
                        lw.sharedDownProj = loc;

                    // Routed experts
                    else if (lw.isMoe && tensorName.find("mlp.experts.") != std::string::npos) {
                        int eid = extractExpertNum(tensorName);
                        if (eid >= 0 && eid < config.numLocalExperts) {
                            if (tensorName.find(".gate_proj.weight") != std::string::npos)
                                routedExperts[layer][eid].gateProj = loc;
                            else if (tensorName.find(".up_proj.weight") != std::string::npos)
                                routedExperts[layer][eid].upProj = loc;
                            else if (tensorName.find(".down_proj.weight") != std::string::npos)
                                routedExperts[layer][eid].downProj = loc;
                        }
                    }
                }
            }
        }

        // Statistics
        for (int i = config.firstKDenseReplace; i < numLayers; ++i) {
            if (!routedExperts[i].empty() && routedExperts[i][0].totalBytes() > 0) {
                totalExperts += int(routedExperts[i].size());
            }
        }

        GLM_LOG_INFO("Index build complete: " + std::to_string(totalExperts) + " routed experts, " +
                     std::to_string(numLayers) + " layers");
        return true;
    }

    // 6. Fallback: traverse all shard headers (slow path)
    GLM_LOG_WARN("index.json not found, falling back to full shard header traversal");

    int totalExperts = 0;
    for (const auto& shardPath : shardFiles) {
        SafeTensorsFile st;
        if (!st.open(shardPath)) continue;

        for (const auto& [name, info] : st.tensors()) {
            TensorLocation loc;
            loc.shardPath = shardPath;
            loc.byteOffset = st.headerSize() + info.dataBegin;
            loc.byteSize = info.byteSize();
            loc.shape = info.shape;
            loc.dtype = info.dtype;

            int layer = extractLayerNum(name);

            if (name == "model.embed_tokens.weight") {
                embedTokens = loc;
            } else if (name == "lm_head.weight") {
                lmHead = loc;
            } else if (name == "model.norm.weight") {
                finalNorm = loc;
            } else if (layer >= 0 && layer < numLayers) {
                auto& lw = layers[layer];

                if (name.find("input_layernorm.weight") != std::string::npos)
                    lw.inputLayernorm = loc;
                else if (name.find("post_attention_layernorm.weight") != std::string::npos)
                    lw.postAttentionLayernorm = loc;

                else if (name.find("self_attn.q_a_proj.weight") != std::string::npos)
                    lw.qAProj = loc;
                else if (name.find("self_attn.q_a_layernorm.weight") != std::string::npos)
                    lw.qALayernorm = loc;
                else if (name.find("self_attn.q_b_proj.weight") != std::string::npos)
                    lw.qBProj = loc;
                else if (name.find("self_attn.kv_a_proj_with_mqa.weight") != std::string::npos)
                    lw.kvAProjWithMqa = loc;
                else if (name.find("self_attn.kv_a_layernorm.weight") != std::string::npos)
                    lw.kvALayernorm = loc;
                else if (name.find("self_attn.kv_b_proj.weight") != std::string::npos)
                    lw.kvBProj = loc;
                else if (name.find("self_attn.o_proj.weight") != std::string::npos)
                    lw.oProj = loc;

                else if (name.find("self_attn.indexer.wq_b.weight") != std::string::npos)
                    lw.indexWqB = loc;
                else if (name.find("self_attn.indexer.wk.weight") != std::string::npos)
                    lw.indexWk = loc;
                else if (name.find("self_attn.indexer.weights_proj.weight") != std::string::npos)
                    lw.indexWeightsProj = loc;
                else if (name.find("self_attn.indexer.k_norm.weight") != std::string::npos)
                    lw.indexKNormW = loc;
                else if (name.find("self_attn.indexer.k_norm.bias") != std::string::npos)
                    lw.indexKNormB = loc;

                else if (!lw.isMoe && name.find("mlp.gate_proj.weight") != std::string::npos)
                    lw.mlpGate = loc;
                else if (!lw.isMoe && name.find("mlp.up_proj.weight") != std::string::npos)
                    lw.mlpUp = loc;
                else if (!lw.isMoe && name.find("mlp.down_proj.weight") != std::string::npos)
                    lw.mlpDown = loc;

                else if (lw.isMoe && name.find("mlp.gate.weight") != std::string::npos)
                    lw.mlpGate = loc;
                else if (lw.isMoe && name.find("mlp.gate.e_score_correction_bias") != std::string::npos)
                    lw.gateBias = loc;

                else if (lw.isMoe && name.find("mlp.shared_experts.gate_proj.weight") != std::string::npos)
                    lw.sharedGateProj = loc;
                else if (lw.isMoe && name.find("mlp.shared_experts.up_proj.weight") != std::string::npos)
                    lw.sharedUpProj = loc;
                else if (lw.isMoe && name.find("mlp.shared_experts.down_proj.weight") != std::string::npos)
                    lw.sharedDownProj = loc;

                else if (lw.isMoe && name.find("mlp.experts.") != std::string::npos) {
                    int eid = extractExpertNum(name);
                    if (eid >= 0 && eid < config.numLocalExperts) {
                        if (name.find(".gate_proj.weight") != std::string::npos)
                            routedExperts[layer][eid].gateProj = loc;
                        else if (name.find(".up_proj.weight") != std::string::npos)
                            routedExperts[layer][eid].upProj = loc;
                        else if (name.find(".down_proj.weight") != std::string::npos)
                            routedExperts[layer][eid].downProj = loc;
                    }
                }
            }
        }
    }

    for (int i = config.firstKDenseReplace; i < numLayers; ++i) {
        if (!routedExperts[i].empty() && routedExperts[i][0].totalBytes() > 0) {
            totalExperts += int(routedExperts[i].size());
        }
    }

    GLM_LOG_INFO("Index build complete: " + std::to_string(totalExperts) + " routed experts, " +
                 std::to_string(numLayers) + " layers");
    return true;
}

void WeightIndex::printStats() const {
    uint64_t totalExpertBytes = 0;
    int expertCount = 0;
    for (int i = config.firstKDenseReplace; i < config.numLayers; ++i) {
        for (const auto& e : routedExperts[i]) {
            if (e.totalBytes() > 0) {
                totalExpertBytes += e.totalBytes();
                expertCount++;
            }
        }
    }
    GLM_LOG_INFO("=== Weight index statistics ===");
    GLM_LOG_INFO("Layers: " + std::to_string(layers.size()));
    GLM_LOG_INFO("Routed expert count: " + std::to_string(expertCount));
    GLM_LOG_INFO("Routed expert total: " + std::to_string(totalExpertBytes / (1024ULL * 1024 * 1024)) + " GB");
    if (!routedExperts.empty() && !routedExperts[config.firstKDenseReplace].empty()) {
        GLM_LOG_INFO("Single expert size: " + std::to_string(routedExperts[config.firstKDenseReplace][0].totalBytes() / (1024 * 1024)) + " MB");
    }
    GLM_LOG_INFO("Embedding: " + std::to_string(embedTokens.byteSize / (1024 * 1024)) + " MB");
    GLM_LOG_INFO("LM Head: " + std::to_string(lmHead.byteSize / (1024 * 1024)) + " MB");
    GLM_LOG_INFO("Shard file count: " + std::to_string(shardFiles.size()));
}

const uint8_t* WeightIndex::getShardView(const std::string& shardPath) {
    auto it = mmapCache.find(shardPath);
    if (it != mmapCache.end()) {
        return it->second->data();
    }
    auto mf = std::make_unique<MappedFile>();
    if (!mf->openAll(shardPath)) {
        GLM_LOG_ERROR("mmap failed: " + shardPath);
        return nullptr;
    }
    const uint8_t* ptr = mf->data();
    mmapCache[shardPath] = std::move(mf);
    return ptr;
}

} // namespace glm
