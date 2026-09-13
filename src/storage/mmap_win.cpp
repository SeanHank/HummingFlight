#include "storage/mapped_file.h"
#include "utils/logger.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#error "mmap_win.cpp is a Windows-only translation unit; use mmap_posix.cpp on POSIX"
#endif

namespace glm {

MappedFile::~MappedFile() { close(); }

MappedFile::MappedFile(MappedFile&& other) noexcept
    : fileHandle_(other.fileHandle_)
    , mappingHandle_(other.mappingHandle_)
    , mappedBase_(other.mappedBase_)
    , mappedData_(other.mappedData_)
    , mappedSize_(other.mappedSize_) {
    other.fileHandle_ = nullptr;
    other.mappingHandle_ = nullptr;
    other.mappedBase_ = nullptr;
    other.mappedData_ = nullptr;
    other.mappedSize_ = 0;
}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
    if (this != &other) {
        close();
        fileHandle_ = other.fileHandle_;
        mappingHandle_ = other.mappingHandle_;
        mappedBase_ = other.mappedBase_;
        mappedData_ = other.mappedData_;
        mappedSize_ = other.mappedSize_;
        other.fileHandle_ = nullptr;
        other.mappingHandle_ = nullptr;
        other.mappedBase_ = nullptr;
        other.mappedData_ = nullptr;
        other.mappedSize_ = 0;
    }
    return *this;
}

bool MappedFile::open(std::string_view path, uint64_t offset, uint64_t size) {
    close();

    std::string p(path);
    fileHandle_ = CreateFileA(p.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (fileHandle_ == INVALID_HANDLE_VALUE) {
        GLM_LOG_ERROR("Cannot open file: " + p);
        return false;
    }

    mappingHandle_ = CreateFileMappingA(fileHandle_, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mappingHandle_) {
        GLM_LOG_ERROR("CreateFileMapping failed: " + p);
        CloseHandle(fileHandle_);
        fileHandle_ = nullptr;
        return false;
    }

    // Windows mmap offset must be 64KB aligned
    uint64_t alignedOffset = offset & ~0xFFFFULL;
    uint64_t adjust = offset - alignedOffset;
    uint64_t mapSize = size + adjust;

    mappedBase_ = static_cast<uint8_t*>(
        MapViewOfFile(mappingHandle_, FILE_MAP_READ,
                      DWORD(alignedOffset >> 32), DWORD(alignedOffset & 0xFFFFFFFF),
                       SIZE_T(mapSize)));
    if (!mappedBase_) {
        GLM_LOG_ERROR("MapViewOfFile failed: " + p);
        CloseHandle(mappingHandle_);
        CloseHandle(fileHandle_);
        mappingHandle_ = nullptr;
        fileHandle_ = nullptr;
        return false;
    }

    mappedData_ = mappedBase_ + adjust;
    mappedSize_ = size;
    return true;
}

bool MappedFile::openAll(std::string_view path) {
    close();

    std::string p(path);
    fileHandle_ = CreateFileA(p.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (fileHandle_ == INVALID_HANDLE_VALUE) {
        GLM_LOG_ERROR("Cannot open file: " + p);
        return false;
    }

    LARGE_INTEGER fsize;
    GetFileSizeEx(fileHandle_, &fsize);
    mappedSize_ = uint64_t(fsize.QuadPart);

    mappingHandle_ = CreateFileMappingA(fileHandle_, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mappingHandle_) {
        GLM_LOG_ERROR("CreateFileMapping failed: " + p);
        CloseHandle(fileHandle_);
        fileHandle_ = nullptr;
        return false;
    }

    mappedBase_ = static_cast<uint8_t*>(
        MapViewOfFile(mappingHandle_, FILE_MAP_READ, 0, 0, 0));
    if (!mappedBase_) {
        GLM_LOG_ERROR("MapViewOfFile failed: " + p);
        CloseHandle(mappingHandle_);
        CloseHandle(fileHandle_);
        mappingHandle_ = nullptr;
        fileHandle_ = nullptr;
        return false;
    }

    mappedData_ = mappedBase_;
    return true;
}

void MappedFile::close() {
    // UnmapViewOfFile requires the ORIGINAL base address (not the offset-adjusted pointer)
    if (mappedBase_) {
        UnmapViewOfFile(mappedBase_);
        mappedBase_ = nullptr;
        mappedData_ = nullptr;
    }
    if (mappingHandle_) {
        CloseHandle(mappingHandle_);
        mappingHandle_ = nullptr;
    }
    if (fileHandle_) {
        CloseHandle(fileHandle_);
        fileHandle_ = nullptr;
    }
    mappedSize_ = 0;
}

} // namespace glm
