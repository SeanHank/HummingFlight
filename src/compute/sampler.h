#pragma once

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace glm {

// Fully extended sampler (item 9): adds locally-typical filtering (typicalP),
// frequency penalty and presence penalty. Each filter can be disabled with its
// zero/identity value, in which case the pipeline is exactly the 10-arg
// overload (goldens preserved). The pipeline order is:
//   temperature -> top-k -> repetition -> frequency -> presence -> min-p
//   -> typical-p -> top-p -> categorical
inline int sampleFromLogits(const float* logits, int vocab,
                            float temperature, float topP, int topK,
                            float minP, float repetitionPenalty,
                            const int* lastTokens, int lastCount,
                            float typicalP, float frequencyPenalty,
                            float presencePenalty,
                            std::mt19937& rng);

// Classic Sampler module (CLI Sampler in the architecture diagram).
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

// Min-p filter (log-space): keep every token within minP * p(max) of the peak,
// i.e. logit >= maxLogit + log(minP). minP in (0,1]; 0 or 1 disables it.
inline void applyMinP(float* logits, int vocab, float minP) {
    if (minP <= 0.0f || minP >= 1.0f) return;
    float maxVal = *std::max_element(logits, logits + vocab);
    float threshold = maxVal + std::log(minP);
    for (int i = 0; i < vocab; ++i) {
        if (logits[i] < threshold) logits[i] = -1e30f;
    }
}

// Repetition penalty: depress (or boost, penalty < 1) logits of tokens present
// in the recent generation window. penalty == 1 or <= 0 disables it.
// Applies standard semantics: score < 0 => score * penalty, else score / penalty.
inline void applyRepetitionPenalty(float* logits, int vocab, float penalty,
                                   const int* lastTokens, int lastCount) {
    if (penalty <= 0.0f || penalty == 1.0f || lastCount <= 0) return;
    for (int i = 0; i < lastCount; ++i) {
        int t = lastTokens[i];
        if (t < 0 || t >= vocab) continue;
        logits[t] = logits[t] < 0.0f ? logits[t] * penalty : logits[t] / penalty;
    }
}

// Frequency penalty (log-space, HF semantics): score(t) -= penalty * count(t)
// in the recent window. Dips the absolute value of the repeated token each time
// it appears, so repeats decay linearly. <= 0 disables it.
inline void applyFrequencyPenalty(float* logits, int vocab, float penalty,
                                  const int* lastTokens, int lastCount) {
    if (penalty <= 0.0f || lastCount <= 0) return;
    std::vector<int> counts(vocab, 0);
    for (int i = 0; i < lastCount; ++i) {
        int t = lastTokens[i];
        if (t >= 0 && t < vocab) counts[t]++;
    }
    for (int t = 0; t < vocab; ++t) {
        if (counts[t] > 0) logits[t] -= penalty * float(counts[t]);
    }
}

// Presence penalty (log-space, HF semantics): each distinct token in the window
// gets a fixed subtract regardless of how often it appears. <= 0 disables it.
inline void applyPresencePenalty(float* logits, int vocab, float penalty,
                                 const int* lastTokens, int lastCount) {
    if (penalty <= 0.0f || lastCount <= 0) return;
    std::vector<char> seen(vocab, 0);
    for (int i = 0; i < lastCount; ++i) {
        int t = lastTokens[i];
        if (t >= 0 && t < vocab) seen[t] = 1;
    }
    for (int t = 0; t < vocab; ++t) {
        if (seen[t]) logits[t] -= penalty;
    }
}

// Typical-p (locally typical) filter: after converting to probabilities, keep
// the tokens whose negative-log-probability sits closest to the distribution
// entropy (| -ln p - H |), in increasing typicality order, until their
// cumulative mass reaches p; everything smaller is masked. p in (0,1), or >= 1
// to disable. Deterministic (no RNG), so seeded runs stay reproducible.
inline void applyTypicalP(float* logits, int vocab, float p) {
    if (p <= 0.0f || p >= 1.0f) return;
    float maxVal = *std::max_element(logits, logits + vocab);
    std::vector<float> probs(vocab);
    float sum = 0.0f;
    for (int i = 0; i < vocab; ++i) {
        probs[i] = std::exp(logits[i] - maxVal);
        sum += probs[i];
    }
    if (sum <= 0.0f || !std::isfinite(sum)) return;
    float invSum = 1.0f / sum;
    float entropy = 0.0f;
    for (int i = 0; i < vocab; ++i) {
        probs[i] *= invSum;
        if (probs[i] > 0.0f) entropy -= probs[i] * std::log(probs[i]);
    }

    std::vector<std::pair<float, int>> byTypical;
    byTypical.reserve(size_t(vocab));
    for (int i = 0; i < vocab; ++i) {
        if (probs[i] <= 0.0f) continue;
        float typicality = std::fabs(-std::log(probs[i]) - entropy);
        byTypical.emplace_back(typicality, i);
    }
    std::sort(byTypical.begin(), byTypical.end(),
              [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                  return a.first < b.first;
              });

    float cum = 0.0f;
    for (size_t idx = 0; idx < byTypical.size(); ++idx) {
        cum += probs[byTypical[idx].second];
        if (cum > p) {
            // The boundary token that crossed p stays in the set; everything
            // strictly less typical from here on is masked.
            for (size_t k = idx + 1; k < byTypical.size(); ++k) logits[byTypical[k].second] = -1e30f;
            break;
        }
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

// Extended sampler: same pipeline plus optional min-p and repetition penalty.
// minP / repetitionPenalty of 0 disable their respective filter so behaviour
// is identical to the classic overload (goldens preserved). lastTokens is the
// generation window (most recent first is not required; all are penalized).
inline int sampleFromLogits(const float* logits, int vocab,
                            float temperature, float topP, int topK,
                            float minP, float repetitionPenalty,
                            const int* lastTokens, int lastCount,
                            std::mt19937& rng) {
    return sampleFromLogits(logits, vocab, temperature, topP, topK,
                            minP, repetitionPenalty, lastTokens, lastCount,
                            0.0f, 0.0f, 0.0f, rng);
}

// Fully extended sampler (item 9): adds locally-typical filtering (typicalP),
// frequency penalty and presence penalty. Each filter can be disabled with its
// zero/identity value, in which case the pipeline is exactly the 10-arg
// overload (goldens preserved). Pipeline order:
//   temperature -> top-k -> repetition -> frequency -> presence -> min-p
//   -> typical-p -> top-p -> categorical
inline int sampleFromLogits(const float* logits, int vocab,
                            float temperature, float topP, int topK,
                            float minP, float repetitionPenalty,
                            const int* lastTokens, int lastCount,
                            float typicalP, float frequencyPenalty,
                            float presencePenalty,
                            std::mt19937& rng) {
    std::vector<float> work(logits, logits + vocab);
    applyTemperature(work.data(), vocab, temperature);
    applyTopK(work.data(), vocab, topK);
    applyRepetitionPenalty(work.data(), vocab, repetitionPenalty, lastTokens, lastCount);
    applyFrequencyPenalty(work.data(), vocab, frequencyPenalty, lastTokens, lastCount);
    applyPresencePenalty(work.data(), vocab, presencePenalty, lastTokens, lastCount);
    applyMinP(work.data(), vocab, minP);
    applyTypicalP(work.data(), vocab, typicalP);
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