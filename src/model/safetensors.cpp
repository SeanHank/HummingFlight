#include "model/safetensors.h"
#include "third_party/picojson.h"
#include "utils/logger.h"
#include <cstring>
#include <iterator>
#include <map>

namespace glm {

namespace {

// Safetensors-spec dtypes with their element byte size. The engine only writes
// BF16 weights, but the full known set is honoured so mixed checkpoints are
// reported with a specific reason instead of being silently mis-sized.
const std::map<std::string, size_t>& knownDtypes() {
    static const std::map<std::string, size_t> k = {
        {"F64", 8}, {"F32", 4}, {"F16", 2}, {"BF16", 2},
        {"I64", 8}, {"I32", 4}, {"I16", 2}, {"I8", 1},
        {"U64", 8}, {"U32", 4}, {"U8", 1}, {"BOOL", 1},
    };
    return k;
}

size_t dtypeByteSize(const std::string& dtype) {
    auto it = knownDtypes().find(dtype);
    return it == knownDtypes().end() ? 0 : it->second;
}

// Build the JSON header for a synthetic in-memory safetensors file.
std::string buildHeader(const std::map<std::string, TensorInfo>& tensors) {
    picojson::object root;
    for (const auto& [name, info] : tensors) {
        picojson::object t;
        t["dtype"] = picojson::value(info.dtype);
        picojson::array shape;
        for (int s : info.shape) shape.push_back(picojson::value(double(s)));
        t["shape"] = picojson::value(shape);
        picojson::array offs;
        offs.push_back(picojson::value(double(info.dataBegin)));
        offs.push_back(picojson::value(double(info.dataEnd)));
        t["data_offsets"] = picojson::value(offs);
        root[name] = picojson::value(t);
    }
    picojson::object meta;
    meta["format"] = picojson::value("pt");
    root["__metadata__"] = picojson::value(meta);
    return picojson::value(root).serialize();
}

} // namespace

bool SafeTensorsFile::open(const std::string& path, const SafeTensorsOptions& opts) {
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

    // Spec: the JSON header length must be a multiple of 8 (strict mode).
    if (opts.requireAlignedHeader && (hsize % 8) != 0) {
        GLM_LOG_ERROR("safetensors header length not a multiple of 8 (" + path + "): " +
                      std::to_string(hsize) + " bytes");
        return false;
    }
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

    // ---- Real JSON parse of the header (item 6: no handwritten JSON search) ----
    picojson::value root;
    const std::string err = picojson::parse(root, header);
    if (!err.empty()) {
        GLM_LOG_ERROR("safetensors header is not valid JSON (" + path + "): " + err);
        return false;
    }
    if (!root.is<picojson::object>()) {
        GLM_LOG_ERROR("safetensors header must be a JSON object (" + path + ")");
        return false;
    }

    const picojson::object& obj = root.get<picojson::object>();
    for (const auto& [name, v] : obj) {
        if (name.empty() || name[0] == '_') continue;  // __metadata__ etc.
        ++stats_.total;
        if (!v.is<picojson::object>()) {
            ++stats_.rejected;
            ++stats_.shapeRejected;  // malformed entry: no usable shape
            continue;
        }
        const picojson::object& to = v.get<picojson::object>();

        TensorInfo info;
        info.name = name;

        // dtype
        auto dIt = to.find("dtype");
        if (dIt != to.end() && dIt->second.is<std::string>()) {
            info.dtype = dIt->second.get<std::string>();
        }
        if (!dtypeByteSize(info.dtype)) {
            ++stats_.rejected;
            ++stats_.dtypeRejected;
            continue;
        }
        if (opts.requireBf16 && info.dtype != "BF16") {
            // Explicit F32 allow-list: GLM-5.2 legitimately stores the noaux_tc
            // router score-correction bias as float32 (moe_router_dtype). F32
            // still flows through shape/bounds/alignment/byte checks below.
            if (info.dtype != "F32") {
                ++stats_.rejected;
                ++stats_.dtypeRejected;
                continue;
            }
            bool f32Allowed = false;
            for (const auto& frag : opts.allowF32NameContaining) {
                if (name.find(frag) != std::string::npos) { f32Allowed = true; break; }
            }
            if (!f32Allowed) {
                ++stats_.rejected;
                ++stats_.dtypeRejected;
                continue;
            }
            ++stats_.f32AllowedAccepted;
        }
        const size_t dtypeBytes = dtypeByteSize(info.dtype);

        // shape
        auto sIt = to.find("shape");
        size_t numel = 1;
        bool okShape = false;
        if (sIt != to.end() && sIt->second.is<picojson::array>()) {
            okShape = true;
            for (const auto& dim : sIt->second.get<picojson::array>()) {
                if (!dim.is<double>()) { okShape = false; break; }
                double dv = dim.get<double>();
                if (dv < 0.0 || dv != std::floor(dv)) { okShape = false; break; }
                int d = int(dv);
                if (d > (1 << 30)) { okShape = false; break; }  // numel overflow guard
                info.shape.push_back(d);
                numel *= size_t(d);
            }
            if (okShape && info.shape.empty()) okShape = false;
        }
        if (!okShape) {
            ++stats_.rejected;
            ++stats_.shapeRejected;
            continue;
        }

        // data_offsets: exactly [begin, end], begin < end (non-empty tensor)
        bool okOffsets = false;
        auto oIt = to.find("data_offsets");
        if (oIt != to.end() && oIt->second.is<picojson::array>()) {
            const auto& arr = oIt->second.get<picojson::array>();
            if (arr.size() == 2 && arr[0].is<double>() && arr[1].is<double>()) {
                info.dataBegin = uint64_t(arr[0].get<double>());
                info.dataEnd = uint64_t(arr[1].get<double>());
                okOffsets = (info.dataEnd > info.dataBegin);
            }
        }
        if (!okOffsets) {
            ++stats_.rejected;
            ++stats_.boundsRejected;
            continue;
        }

        // Bounds: the tensor must live entirely inside the file.
        if (headerSize_ + info.dataEnd > fileSize_) {
            ++stats_.rejected;
            ++stats_.boundsRejected;
            continue;
        }

        // Tensor data alignment (item 7): absolute start offset must be
        // 8-byte aligned for unaligned-SIMD-safe zero-copy weight views.
        if (opts.requireAlignedData && ((headerSize_ + info.dataBegin) % 8) != 0) {
            ++stats_.rejected;
            ++stats_.alignmentRejected;
            continue;
        }

        // Shape x dtype byte consistency: catches header corruption that a pure
        // bounds check cannot see (e.g. shape/dtype/offset edited inconsistently).
        if (numel * dtypeBytes != info.byteSize()) {
            ++stats_.rejected;
            ++stats_.byteMismatchRejected;
            continue;
        }

        // Duplicate tensor names: reject the second occurrence (never silently
        // overwrite; a duplicate is corruption per the safetensors contract).
        if (tensors_.count(name) != 0) {
            ++stats_.rejected;
            ++stats_.duplicateRejected;
            continue;
        }

        tensors_[name] = info;
        ++stats_.accepted;
    }

    // Close file stream (no longer needed once mmap is opened)
    fileStream_.close();

    // Open mmap for zero-copy access to tensor data
    mmap_ = std::make_unique<MappedFile>();
    if (!mmap_->openAll(path)) {
        GLM_LOG_WARN("mmap open failed for: " + path + " (will use readTensor fallback)");
        mmap_.reset();
    }

    GLM_LOG_INFO("safetensors parsed: " + path + " (" + std::to_string(tensors_.size()) +
                 " of " + std::to_string(stats_.total) + " tensors accepted)");
    if (stats_.rejected > 0) {
        GLM_LOG_WARN("safetensors rejections (" + path + "): duplicate=" +
                     std::to_string(stats_.duplicateRejected) + " dtype=" +
                     std::to_string(stats_.dtypeRejected) + " shape=" +
                     std::to_string(stats_.shapeRejected) + " bounds=" +
                     std::to_string(stats_.boundsRejected) + " byte-mismatch=" +
                     std::to_string(stats_.byteMismatchRejected) + " unaligned-data=" +
                     std::to_string(stats_.alignmentRejected));
    }
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