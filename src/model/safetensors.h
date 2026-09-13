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

class SafeTensorsFile {
public:
    bool open(const std::string& path);

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

private:
    std::string filePath_;
    std::ifstream fileStream_;
    std::map<std::string, TensorInfo> tensors_;
    uint64_t headerSize_ = 0;
    uint64_t fileSize_ = 0;
    // mmap mode for zero-copy tensor access
    std::unique_ptr<MappedFile> mmap_;
};

} // namespace glm
