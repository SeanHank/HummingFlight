#include "storage/iocp_reader.h"
#include "utils/logger.h"

#ifndef _WIN32

#include <condition_variable>
#include <deque>
#include <fstream>
#include <mutex>
#include <thread>

namespace glm {

// POSIX counterpart of the Windows IOCP reader: same interface, same semantics
// (async submission + completion callback, overlap with CPU compute) built on a
// small pool of reader threads draining a FIFO of ReadRequests. Reads are
// submitted synchronously by the workers (the OS kernel does the async overlap).

namespace {

struct Pending {
    ReadRequest req;
    ReadCallback cb;
};

} // namespace

struct IOCPReader::Impl {
    std::mutex mtx;
    std::condition_variable cv;
    std::deque<Pending> queue;
    bool quit = false;
};

IOCPReader::IOCPReader() : impl_(std::make_shared<Impl>()) {}
IOCPReader::~IOCPReader() { stop(); }

bool IOCPReader::start(int workerCount) {
    running_ = true;
    IOCPReader* self = this;
    for (int i = 0; i < workerCount; ++i) {
        workers_.emplace_back([self, impl = impl_]() {
            while (true) {
                Pending job;
                {
                    std::unique_lock<std::mutex> lock(impl->mtx);
                    impl->cv.wait(lock, [&]() { return impl->quit || !impl->queue.empty(); });
                    if (impl->quit && impl->queue.empty()) return;
                    job = std::move(impl->queue.front());
                    impl->queue.pop_front();
                }
                // Blocking read; the OS chunks the transfer.
                std::ifstream in(job.req.filePath, std::ios::binary);
                size_t bytesRead = 0;
                if (in) {
                    in.seekg(std::streamoff(job.req.offset), std::ios::beg);
                    in.read(static_cast<char*>(job.req.destBuffer),
                            static_cast<std::streamsize>(job.req.size));
                    bytesRead = in.gcount();
                }
                bool ok = (bytesRead == size_t(job.req.size));
                if (job.cb) job.cb(ok, bytesRead);
                self->onComplete();
            }
        });
    }
    return true;
}

void IOCPReader::stop() {
    if (!running_) return;
    waitAll();
    {
        std::lock_guard<std::mutex> lock(impl_->mtx);
        impl_->quit = true;
    }
    impl_->cv.notify_all();
    for (auto& t : workers_) {
        if (t.joinable()) t.join();
    }
    workers_.clear();
    running_ = false;
}

void IOCPReader::submitRead(const ReadRequest& req, ReadCallback cb) {
    if (!running_) {
        if (cb) cb(false, 0);
        return;
    }
    pendingCount_++;
    {
        std::lock_guard<std::mutex> lock(impl_->mtx);
        impl_->queue.push_back({req, std::move(cb)});
    }
    impl_->cv.notify_one();
}

void IOCPReader::waitAll() {
    std::unique_lock<std::mutex> lock(waitMtx_);
    waitCv_.wait(lock, [this]() { return pendingCount_.load() <= 0; });
}

} // namespace glm

#endif // !_WIN32