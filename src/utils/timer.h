#pragma once

#include <chrono>
#include <string>

namespace glm {

// Simple timer for performance benchmarking
class Timer {
public:
    Timer() : start_(clock::now()) {}

    void reset() { start_ = clock::now(); }

    double elapsedMs() const {
        auto now = clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }

    double elapsedSec() const { return elapsedMs() / 1000.0; }

private:
    using clock = std::chrono::steady_clock;
    clock::time_point start_;
};

} // namespace glm
