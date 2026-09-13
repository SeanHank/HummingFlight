#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace glm {

// BF16 software emulation type (R7-5800H has no native AVX-512 BF16 instructions, store BF16, compute as F32)
// Memory layout: 2 bytes, fully compatible with IEEE 754 BF16
struct BFloat16 {
    uint16_t bits_;

    BFloat16() : bits_(0) {}
    explicit BFloat16(uint16_t b) : bits_(b) {}

    // BF16 -> F32 (zero-extend high bits)
    float toF32() const {
        uint32_t u = uint32_t(bits_) << 16;
        float f;
        std::memcpy(&f, &u, sizeof(f));
        return f;
    }

    // F32 -> BF16 (truncate low 16 bits, with round-to-nearest-even)
    static BFloat16 fromF32(float f) {
        uint32_t u;
        std::memcpy(&u, &f, sizeof(u));
        // round-to-nearest-even
        uint32_t rounding_bias = 0x7FFF + ((u >> 16) & 1);
        return BFloat16(uint16_t((u + rounding_bias) >> 16));
    }
};

static_assert(sizeof(BFloat16) == 2, "BFloat16 must be 2 bytes");

} // namespace glm
