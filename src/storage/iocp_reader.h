#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace glm {

// IOCP async read queue (Windows)
// Used for streaming prefetch of expert weights, overlapping with CPU computation

struct ReadRequest {
    std::string filePath;
    uint64_t offset;
    uint64_t size;
    void* destBuffer;
};

using ReadCallback = std::function<void(bool success, size_t bytesRead)>;

class IOCPReader {
public:
#ifdef _WIN32
    IOCPReader() = default;
#else
    IOCPReader();
#endif
    ~IOCPReader();

    // Start worker thread pool
    bool start(int workerCount = 2);

    void stop();

    // Submit async read request, callback on completion
    void submitRead(const ReadRequest& req, ReadCallback cb);

    // Wait for all submitted requests to complete (blocking)
    void waitAll();

    // Number of pending requests
    int pendingCount() const { return pendingCount_.load(); }

    // Called by worker threads on request completion (public for friend access)
    void onComplete() {
        pendingCount_--;
        waitCv_.notify_all();
    }

private:
#ifndef _WIN32
    struct Impl;
    std::shared_ptr<Impl> impl_;
#endif
    void* iocpHandle_ = nullptr; // HANDLE
    std::vector<std::thread> workers_;
    bool running_ = false;

    // Reference counting for waitAll()
    std::atomic<int> pendingCount_{0};
    std::mutex waitMtx_;
    std::condition_variable waitCv_;
};

} // namespace glm
