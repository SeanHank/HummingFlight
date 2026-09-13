#include "storage/mapped_file.h"
#include "utils/logger.h"

#ifndef _WIN32

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

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

// POSIX requires mapping offsets to be page-aligned.
static uint64_t pageSize() {
    static const uint64_t p = uint64_t(sysconf(_SC_PAGESIZE));
    return p;
}

bool MappedFile::open(std::string_view path, uint64_t offset, uint64_t size) {
    close();

    std::string p(path);
    int fd = ::open(p.c_str(), O_RDONLY);
    if (fd < 0) {
        GLM_LOG_ERROR("Cannot open file: " + p);
        return false;
    }
    fileHandle_ = reinterpret_cast<void*>(intptr_t(fd));

    uint64_t ps = pageSize();
    uint64_t alignedOffset = offset & ~(ps - 1);
    uint64_t adjust = offset - alignedOffset;

    void* base = ::mmap(nullptr, size_t(size + adjust), PROT_READ, MAP_PRIVATE, fd, off_t(alignedOffset));
    if (base == MAP_FAILED) {
        GLM_LOG_ERROR("mmap failed: " + p);
        ::close(fd);
        fileHandle_ = nullptr;
        return false;
    }

    mappedBase_ = static_cast<uint8_t*>(base);
    mappedData_ = mappedBase_ + adjust;
    mappedSize_ = size;
    // Keep the fd open for the lifetime of the mapping (equivalent to the
    // Windows behavior where the file handle stays open).
    return true;
}

bool MappedFile::openAll(std::string_view path) {
    close();

    std::string p(path);
    int fd = ::open(p.c_str(), O_RDONLY);
    if (fd < 0) {
        GLM_LOG_ERROR("Cannot open file: " + p);
        return false;
    }
    fileHandle_ = reinterpret_cast<void*>(intptr_t(fd));

    struct stat st {};
    if (fstat(fd, &st) != 0) {
        GLM_LOG_ERROR("fstat failed: " + p);
        ::close(fd);
        fileHandle_ = nullptr;
        return false;
    }
    uint64_t fsize = uint64_t(st.st_size);

    void* base = ::mmap(nullptr, size_t(fsize), PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) {
        GLM_LOG_ERROR("mmap failed: " + p);
        ::close(fd);
        fileHandle_ = nullptr;
        return false;
    }

    mappedBase_ = static_cast<uint8_t*>(base);
    mappedData_ = mappedBase_;
    mappedSize_ = fsize;
    return true;
}

void MappedFile::close() {
    if (mappedBase_) {
        ::munmap(mappedBase_, size_t(mappedSize_));
        mappedBase_ = nullptr;
        mappedData_ = nullptr;
    }
    if (fileHandle_) {
        ::close(int(intptr_t(fileHandle_)));
        fileHandle_ = nullptr;
    }
    mappingHandle_ = nullptr;
    mappedSize_ = 0;
}

} // namespace glm

#endif // !_WIN32