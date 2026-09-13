#pragma once

#include "compute/dtype_bf16.h"
#include "storage/mapped_file.h"
#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace glm {

// Safetensors file header parser
// Format: first 8 bytes = header_size (u64 LE), followed by header_size bytes of JSON header
// Each tensor: { name, dtype, shape, data_offsets: [begin, end] }

struct TensorInfo {
    std::string name;
    std::string dtype;          // "BF16", "F32", "F64", etc.
    std::vector<int> shape;
    uint64_t dataBegin;         // Start offset of data within the file (relative to header end)
    uint64_t dataEnd;
    uint64_t byteSize() const { return dataEnd - dataBegin; }
    int numel() const {
        int n = 1;
        for (int s : shape) n *= s;
        return n;
    }
};

// Production-enforced validation (design: strict safetensors checking).
// When requireBf16 is set (default), every tensor must be BF16 -- GLM-5.2 bf16
// checkpoints are BF16-only, so any other dtype indicates a mixed/damaged
// checkpoint that the engine refuses to trust. The ONE architectural exception
// is the noaux_tc router score-correction bias (e_score_correction_bias), which
// GLM-5.2 stores as float32 (config moe_router_dtype=float32); a tensor may be
// F32 only when its name contains one of allowF32NameContaining.
struct SafeTensorsOptions {
    bool requireBf16 = true;          // dtype must be "BF16" (or an allowed F32)
    // Tensors whose name contains any of these fragments are permitted to be
    // float32; everything else is still hard-gated to BF16 (item 7).
    std::vector<std::string> allowF32NameContaining;
    bool requireAlignedHeader = true; // header length must be a multiple of 8 (spec)
    // Tensor data must start at an 8-byte-aligned absolute offset
    // (header_size + data_offsets[0]). The format guarantees this; a
    // misaligned tensor breaks SIMD zero-copy reads, so it is treated as
    // corruption rather than silently accepted.
    bool requireAlignedData = true;
};

struct SafeTensorsStats {
    size_t total = 0;
    size_t accepted = 0;
    size_t rejected = 0;
    size_t duplicateRejected = 0;
    size_t dtypeRejected = 0;
    size_t f32AllowedAccepted = 0;    // F32 accepted via allowF32NameContaining
    size_t shapeRejected = 0;
    size_t boundsRejected = 0;
    size_t byteMismatchRejected = 0;  // shape x dtype bytes != declared byte span
    size_t alignmentRejected = 0;     // data start not 8-byte aligned
};

class SafeTensorsFile {
public:
    bool open(const std::string& path, const SafeTensorsOptions& opts = {});

    // Find by tensor name
    const TensorInfo* find(const std::string& name) const;

    // Read tensor data into buffer (buffer must be large enough)
    bool readTensor(const std::string& name, void* buffer, size_t bufSize);

    // Get mmap pointer directly (requires mmap mode, returns nullptr if not mmap'd)
    const uint8_t* tensorData(const TensorInfo& info) const;

    const std::map<std::string, TensorInfo>& tensors() const { return tensors_; }
    uint64_t headerSize() const { return headerSize_; }

    // Get underlying mmap view (for offset-based access)
    const uint8_t* fileBase() const { return mmap_ ? mmap_->data() : nullptr; }

    // Validation counters from open() (strict-production reports).
    const SafeTensorsStats& validationStats() const { return stats_; }

private:
    std::string filePath_;
    std::ifstream fileStream_;
    std::map<std::string, TensorInfo> tensors_;
    SafeTensorsStats stats_;
    uint64_t headerSize_ = 0;
    uint64_t fileSize_ = 0;
    // mmap mode for zero-copy tensor access
    std::unique_ptr<MappedFile> mmap_;
};

} // namespace glm
