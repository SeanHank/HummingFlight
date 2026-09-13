#include "storage/iocp_reader.h"
#include "utils/logger.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <thread>

namespace glm {

namespace {

struct OverlappedEx : OVERLAPPED {
    ReadRequest req;
    ReadCallback cb;
    HANDLE fileHandle;
    IOCPReader* owner;
};

DWORD WINAPI workerLoop(LPVOID param) {
    HANDLE iocp = HANDLE(param);
    DWORD bytesRead = 0;
    ULONG_PTR key = 0;
    LPOVERLAPPED ov = nullptr;

    while (GetQueuedCompletionStatus(iocp, &bytesRead, &key, &ov, INFINITE)) {
        if (ov == nullptr) break; // Stop signal
        auto* ex = static_cast<OverlappedEx*>(ov);
        bool ok = (bytesRead == ex->req.size);
        if (ex->cb) ex->cb(ok, bytesRead);
        CloseHandle(ex->fileHandle);

        // Decrement pending count and notify waiters
        ex->owner->onComplete();

        delete ex;
    }
    return 0;
}

} // namespace

IOCPReader::~IOCPReader() { stop(); }

bool IOCPReader::start(int workerCount) {
    iocpHandle_ = CreateIoCompletionPort(INVALID_HANDLE_VALUE, nullptr, 0, DWORD(workerCount));
    if (!iocpHandle_) {
        GLM_LOG_ERROR("CreateIoCompletionPort failed");
        return false;
    }
    running_ = true;
    for (int i = 0; i < workerCount; ++i) {
        workers_.emplace_back(workerLoop, iocpHandle_);
    }
    return true;
}

void IOCPReader::stop() {
    if (!running_) return;
    running_ = false;
    // Wait for all pending requests to drain first
    waitAll();
    // Send stop signal
    for (size_t i = 0; i < workers_.size(); ++i) {
        PostQueuedCompletionStatus(iocpHandle_, 0, 0, nullptr);
    }
    for (auto& t : workers_) {
        if (t.joinable()) t.join();
    }
    workers_.clear();
    if (iocpHandle_) {
        CloseHandle(iocpHandle_);
        iocpHandle_ = nullptr;
    }
}

void IOCPReader::submitRead(const ReadRequest& req, ReadCallback cb) {
    auto* ex = new OverlappedEx{};
    ZeroMemory(&ex->Internal, sizeof(OVERLAPPED));
    ex->Offset = DWORD(req.offset & 0xFFFFFFFF);
    ex->OffsetHigh = DWORD(req.offset >> 32);
    ex->req = req;
    ex->cb = std::move(cb);
    ex->owner = this;

    ex->fileHandle = CreateFileA(req.filePath.c_str(), GENERIC_READ, FILE_SHARE_READ,
                                  nullptr, OPEN_EXISTING, FILE_FLAG_OVERLAPPED, nullptr);
    if (ex->fileHandle == INVALID_HANDLE_VALUE) {
        if (cb) cb(false, 0);
        delete ex;
        return;
    }

    CreateIoCompletionPort(ex->fileHandle, iocpHandle_, 0, 0);

    pendingCount_++;

    BOOL ok = ReadFile(ex->fileHandle, req.destBuffer, DWORD(req.size), nullptr, ex);
    if (!ok && GetLastError() != ERROR_IO_PENDING) {
        if (cb) cb(false, 0);
        CloseHandle(ex->fileHandle);
        pendingCount_--;
        delete ex;
    }
}

void IOCPReader::waitAll() {
    std::unique_lock<std::mutex> lock(waitMtx_);
    waitCv_.wait(lock, [this]() { return pendingCount_.load() <= 0; });
}

} // namespace glm
