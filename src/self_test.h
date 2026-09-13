#pragma once

#include <string>

namespace glm {

// Functional self-test suite. Runs a battery of deterministic checks against the
// production kernels (BF16 conversion, GEMV, RMSNorm, softmax, router top-K,
// KV cache, placement policy) using synthetic data. No model weights required.
//
// fixtureDir: directory containing tests/fixtures assets (config_mini.json).
// Returns number of failed checks (0 == all passed).
int runSelfTests(bool verbose, const std::string& fixtureDir = "tests/fixtures");

// Structural weights validation against a real model directory.
// Verifies config.json parses, shard files exist, index.json is consistent,
// expected tensor names resolve and byte offsets stay inside their shards.
// Appends a human-readable report to `report`.
// Returns number of failed checks (0 == all passed).
int validateWeights(const std::string& modelDir, std::string& report);

} // namespace glm