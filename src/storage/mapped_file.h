#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace glm {

// Cross-platform shard file mapping.
//   Windows:  mmap_win.cpp   (CreateFileMappingA + MapViewOfFile)
//   POSIX:    mmap_posix.cpp (open + mmap with MAP_PRIVATE / PROT_READ)
// The class provides zero-copy views over the shard files; the OS page cache
// manages residency for the read-mostly weights.
class MappedFile {
public:
    MappedFile() = default;
    ~MappedFile();

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    MappedFile(MappedFile&& other) noexcept;
    MappedFile& operator=(MappedFile&& other) noexcept;

    // Open file and map the specified [offset, offset+size) region.
    bool open(std::string_view path, uint64_t offset, uint64_t size);

    // Map the entire file.
    bool openAll(std::string_view path);

    void close();

    const uint8_t* data() const { return mappedData_; }
    uint64_t size() const { return mappedSize_; }
    bool isOpen() const { return mappedData_ != nullptr; }

private:
    void* fileHandle_   = nullptr; // HANDLE (WIN32) / int fd (POSIX)
    void* mappingHandle_ = nullptr; // HANDLE (WIN32) / unused (POSIX)
    uint8_t* mappedBase_ = nullptr; // Unadjusted base address (for correct unmap)
    uint8_t* mappedData_ = nullptr; // User-facing pointer (may be offset-adjusted)
    uint64_t mappedSize_ = 0;
};

} // namespace glm