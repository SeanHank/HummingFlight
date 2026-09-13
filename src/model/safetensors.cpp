#include "model/safetensors.h"
#include "utils/logger.h"
#include <cstring>
#include <sstream>

namespace glm {

namespace {

// Safetensors-spec dtypes (subset actually used by this engine: BF16).
bool isKnownDtype(const std::string& dtype) {
    static const char* const kKnown[] = {
        "F64", "F32", "F16", "BF16",
        "I64", "I32", "I16", "I8",
        "U64", "U32", "U8", "BOOL",
    };
    for (const char* k : kKnown) {
        if (dtype == k) return true;
    }
    return false;
}

} // namespace

// Minimal JSON array parser (extract integers from shape)
static std::vector<int> parseIntArray(const std::string& json, size_t start) {
    std::vector<int> result;
    size_t pos = json.find('[', start);
    if (pos == std::string::npos) return result;
    pos++;
    while (pos < json.size() && json[pos] != ']') {
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == ',')) pos++;
        if (pos >= json.size() || json[pos] == ']') break;
        size_t end = pos;
        while (end < json.size() && json[end] != ',' && json[end] != ']') end++;
        try {
            result.push_back(std::stoi(json.substr(pos, end - pos)));
        } catch (...) {}
        pos = end;
    }
    return result;
}

bool SafeTensorsFile::open(const std::string& path) {
    filePath_ = path;
    fileStream_.open(path, std::ios::binary);
    if (!fileStream_.is_open()) {
        GLM_LOG_ERROR("Cannot open safetensors file: " + path);
        return false;
    }

    // Full file size (for bounds validation against data_offsets).
    fileStream_.seekg(0, std::ios::end);
    fileSize_ = uint64_t(fileStream_.tellg());
    fileStream_.seekg(0, std::ios::beg);

    // Read header size (first 8 bytes, u64 LE)
    uint64_t hsize = 0;
    fileStream_.read(reinterpret_cast<char*>(&hsize), 8);
    if (!fileStream_) {
        GLM_LOG_ERROR("Failed to read header size: " + path);
        return false;
    }
    headerSize_ = hsize + 8; // Data area start = 8 + header_size
    if (headerSize_ + 8 > fileSize_) {
        GLM_LOG_ERROR("Header size exceeds file size: " + path);
        return false;
    }

    // Read header JSON
    std::string header(hsize, '\0');
    fileStream_.read(header.data(), hsize);
    if (!fileStream_) {
        GLM_LOG_ERROR("Failed to read header content: " + path);
        return false;
    }

    // Parse tensors (simple state machine, looking for "name": { "dtype":..., "shape":..., "data_offsets":[...] })
    size_t pos = 0;
    while (pos < header.size()) {
        // Find tensor name (quote-wrapped key)
        size_t nameStart = header.find('"', pos);
        if (nameStart == std::string::npos) break;
        size_t nameEnd = header.find('"', nameStart + 1);
        if (nameEnd == std::string::npos) break;
        std::string name = header.substr(nameStart + 1, nameEnd - nameStart - 1);

        // Skip __metadata__ etc.
        if (name.empty() || name[0] == '_') {
            pos = nameEnd + 1;
            continue;
        }

        // Find next '{'
        size_t objStart = header.find('{', nameEnd);
        if (objStart == std::string::npos) break;
        size_t objEnd = header.find('}', objStart);
        if (objEnd == std::string::npos) break;
        std::string obj = header.substr(objStart, objEnd - objStart + 1);

        // Extract fields
        TensorInfo info;
        info.name = name;

        // dtype
        size_t dp = obj.find("\"dtype\"");
        if (dp != std::string::npos) {
            size_t dq1 = obj.find('"', dp + 7);
            size_t dq2 = obj.find('"', dq1 + 1);
            info.dtype = obj.substr(dq1 + 1, dq2 - dq1 - 1);
        }

        // shape
        size_t sp = obj.find("\"shape\"");
        if (sp != std::string::npos) {
            info.shape = parseIntArray(obj, sp);
        }

        // data_offsets
        size_t dop = obj.find("\"data_offsets\"");
        if (dop != std::string::npos) {
            size_t arr = obj.find('[', dop);
            size_t comma = obj.find(',', arr);
            size_t close = obj.find(']', arr);
            try {
                info.dataBegin = std::stoull(obj.substr(arr + 1, comma - arr - 1));
                info.dataEnd = std::stoull(obj.substr(comma + 1, close - comma - 1));
            } catch (...) {}
        }

        if (info.dataEnd > info.dataBegin) {
            // Bounds validation: the tensor must live entirely inside the file.
            bool inFile = (headerSize_ + info.dataEnd <= fileSize_);
            // dtype validation: unknown dtypes are skipped (cannot trust byte layout).
            bool knownType = isKnownDtype(info.dtype);
            // shape validation: non-empty, non-negative dims.
            bool validShape = !info.shape.empty();
            for (int s : info.shape) {
                if (s < 0) validShape = false;
            }
            if (inFile && knownType && validShape) {
                tensors_[name] = info;
            } else {
                GLM_LOG_WARN("safetensors: skipping tensor '" + name + "' (inFile=" +
                             std::to_string(inFile) + ", dtype=" + info.dtype +
                             ", shape=" + std::to_string(info.shape.size()) + " dims)");
            }
        }

        pos = objEnd + 1;
    }

    // Close file stream (no longer needed once mmap is opened)
    fileStream_.close();

    // Open mmap for zero-copy access to tensor data
    mmap_ = std::make_unique<MappedFile>();
    if (!mmap_->openAll(path)) {
        GLM_LOG_WARN("mmap open failed for: " + path + " (will use readTensor fallback)");
        mmap_.reset();
    }

    GLM_LOG_INFO("safetensors parsed: " + path + " (" + std::to_string(tensors_.size()) + " tensors)");
    return true;
}

const TensorInfo* SafeTensorsFile::find(const std::string& name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

bool SafeTensorsFile::readTensor(const std::string& name, void* buffer, size_t bufSize) {
    auto* info = find(name);
    if (!info) return false;
    if (bufSize < info->byteSize()) return false;

    // Try mmap first (zero-copy)
    if (mmap_ && mmap_->data()) {
        if (mmap_->size() < headerSize_ + info->dataEnd) return false;
        const uint8_t* src = mmap_->data() + headerSize_ + info->dataBegin;
        std::memcpy(buffer, src, info->byteSize());
        return true;
    }

    // Fallback: file seek + read
    if (!fileStream_.is_open()) {
        fileStream_.open(filePath_, std::ios::binary);
    }
    if (!fileStream_.is_open()) return false;

    uint64_t fileOffset = headerSize_ + info->dataBegin;
    fileStream_.clear();
    fileStream_.seekg(fileOffset, std::ios::beg);
    fileStream_.read(static_cast<char*>(buffer), static_cast<std::streamsize>(info->byteSize()));
    return bool(fileStream_);
}

const uint8_t* SafeTensorsFile::tensorData(const TensorInfo& info) const {
    if (!mmap_ || !mmap_->data()) return nullptr;
    if (mmap_->size() < headerSize_ + info.dataEnd) return nullptr;
    return mmap_->data() + headerSize_ + info.dataBegin;
}

} // namespace glm
