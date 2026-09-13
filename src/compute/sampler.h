#pragma once

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace glm {

// Sampler module (CLI Sampler in the architecture diagram).
// Applies temperature / top-k / top-p filtering to a raw logits vector and
// draws one token. Greedy (temperature <= 0) is a pure argmax so goldens stay
// byte-identical across hosts.

// Apply temperature scaling in place (identity transform for temperature <= 0 or == 1).
inline void applyTemperature(float* logits, int vocab, float temperature) {
    if (temperature <= 0.0f || temperature == 1.0f) return;
    float invT = 1.0f / temperature;
    for (int i = 0; i < vocab; ++i) logits[i] *= invT;
}

// Top-k filter: keep only the k largest logits (ties are also kept since we
// mask below the k-th order statistic, matching standard implementations).
inline void applyTopK(float* logits, int vocab, int k) {
    if (k <= 0 || k >= vocab) return;
    std::vector<float> sorted(logits, logits + vocab);
    std::nth_element(sorted.begin(), sorted.begin() + k - 1, sorted.end(),
                     std::greater<float>());
    float threshold = sorted[k - 1];
    for (int i = 0; i < vocab; ++i) {
        if (logits[i] < threshold) logits[i] = -1e30f;
    }
}

// Top-p (nucleus) filter: after converting to probabilities, drop every token
// below the smallest-prefix cut that leaves cumulative probability >= p.
inline void applyTopP(float* logits, int vocab, float p) {
    if (p >= 1.0f) return;
    if (p <= 0.0f) {
        // Degenerate: only the argmax survives.
        int best = 0;
        for (int i = 1; i < vocab; ++i) {
            if (logits[i] > logits[best]) best = i;
        }
        for (int i = 0; i < vocab; ++i) logits[i] = -1e30f;
        logits[best] = 0.0f;
        return;
    }

    float maxVal = *std::max_element(logits, logits + vocab);
    std::vector<float> probs(vocab);
    float sum = 0.0f;
    for (int i = 0; i < vocab; ++i) {
        probs[i] = std::exp(logits[i] - maxVal);
        sum += probs[i];
    }
    float invSum = sum > 0.0f ? 1.0f / sum : 1.0f;
    for (int i = 0; i < vocab; ++i) probs[i] *= invSum;

    std::vector<int> order(vocab);
    for (int i = 0; i < vocab; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return probs[a] > probs[b];
    });

    float cum = 0.0f;
    for (int idx : order) {
        cum += probs[idx];
        if (cum > p) logits[idx] = -1e30f;   // this token and every smaller one
    }
}

// Sample one token (index) from logits after temperature + top-k + top-p.
// temperature <= 0 means greedy argmax (deterministic, ignores rng).
inline int sampleFromLogits(const float* logits, int vocab,
                            float temperature, float topP, int topK,
                            std::mt19937& rng) {
    std::vector<float> work(logits, logits + vocab);
    applyTemperature(work.data(), vocab, temperature);
    applyTopK(work.data(), vocab, topK);
    applyTopP(work.data(), vocab, topP);

    float maxVal = *std::max_element(work.begin(), work.end());
    std::vector<float> probs(vocab);
    float sum = 0.0f;
    for (int i = 0; i < vocab; ++i) {
        probs[i] = std::exp(work[i] - maxVal);
        sum += probs[i];
    }
    if (sum <= 0.0f || !std::isfinite(sum)) {
        int best = 0;
        for (int i = 1; i < vocab; ++i) {
            if (work[i] > work[best]) best = i;
        }
        return best;
    }
    float invSum = 1.0f / sum;
    for (int i = 0; i < vocab; ++i) probs[i] *= invSum;

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    float cum = 0.0f;
    for (int i = 0; i < vocab; ++i) {
        cum += probs[i];
        if (cum >= r) return i;
    }
    return vocab - 1;
}

} // namespace glm