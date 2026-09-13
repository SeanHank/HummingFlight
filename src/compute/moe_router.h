#pragma once

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace glm {

// Result of noaux_tc Top-K routing: selected expert ids + normalized weights.
// Fixed-size arrays keep the structure trivially copyable and allocation-free.
struct TopKResult {
    int indices[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float weights[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    int k = 0;
};

// Sigmoid activation: f(x) = 1 / (1 + exp(-x))
inline float sigmoid_f32(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// noaux_tc Top-K routing (GLM-5.2 / DeepSeek-V3 style load balancing).
// scores: per-expert affinity logits [numExperts]
// bias:   optional per-expert correction bias (e_score_correction_bias), may be null
// topK:   number of experts to select (num_experts_per_tok, must be <= 8)
// routedScalingFactor: multiply final weights (routed_scaling_factor)
// normTopkProb: whether to renormalize the selected weights to sum==1
inline TopKResult topKScores(const float* scores, int numExperts,
                             const float* bias, int topK,
                             float routedScalingFactor, bool normTopkProb) {
    TopKResult result;
    result.k = 0;
    if (numExperts <= 0 || topK <= 0) return result;

    std::vector<std::pair<float, int>> ranked;
    ranked.reserve(numExperts);
    for (int i = 0; i < numExperts; ++i) {
        ranked.emplace_back(scores[i] + (bias ? bias[i] : 0.0f), i);
    }
    const int k = std::min(topK, numExperts);
    std::partial_sort(ranked.begin(), ranked.begin() + k, ranked.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

    result.k = k;
    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
        result.indices[i] = ranked[i].second;
        result.weights[i] = scores[ranked[i].second];
        sum += result.weights[i];
    }
    if (normTopkProb && sum > 0.0f) {
        for (int i = 0; i < k; ++i) result.weights[i] /= sum;
    }
    for (int i = 0; i < k; ++i) result.weights[i] *= routedScalingFactor;
    return result;
}

} // namespace glm