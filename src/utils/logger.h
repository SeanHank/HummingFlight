#pragma once

#include <chrono>
#include <cstdio>
#include <mutex>
#include <string>
#include <string_view>

namespace glm {

enum class LogLevel { Debug, Info, Warn, Error };

class Logger {
public:
    static void setLevel(LogLevel level) { g_level = level; }
    static LogLevel level() { return g_level; }

    static void log(LogLevel level, std::string_view msg) {
        if (level < g_level) return;

        // Get timestamp (HH:MM:SS.mmm)
        auto now = std::chrono::system_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        auto time_t = std::chrono::system_clock::to_time_t(now);
        struct tm tmBuf;
#ifdef _WIN32
        localtime_s(&tmBuf, &time_t);
#else
        localtime_r(&time_t, &tmBuf);
#endif

        const char* tag = "";
        switch (level) {
            case LogLevel::Debug: tag = "DEBUG"; break;
            case LogLevel::Info:  tag = "INFO "; break;
            case LogLevel::Warn:  tag = "WARN "; break;
            case LogLevel::Error: tag = "ERROR"; break;
        }

        // Thread-safe write (single fprintf call to avoid interleaving)
        std::lock_guard<std::mutex> lock(g_mutex);
        std::fprintf(stderr, "[%02d:%02d:%02d.%03d] [%s] %.*s\n",
                     tmBuf.tm_hour, tmBuf.tm_min, tmBuf.tm_sec,
                     static_cast<int>(ms.count()),
                     tag,
                     static_cast<int>(msg.size()), msg.data());
    }

    static void info(std::string_view m)  { log(LogLevel::Info, m); }
    static void warn(std::string_view m)  { log(LogLevel::Warn, m); }
    static void error(std::string_view m) { log(LogLevel::Error, m); }
    static void debug(std::string_view m) { log(LogLevel::Debug, m); }

private:
    static inline LogLevel g_level = LogLevel::Info;
    static inline std::mutex g_mutex;
};

} // namespace glm

#define GLM_LOG_INFO(...)  ::glm::Logger::info(__VA_ARGS__)
#define GLM_LOG_WARN(...)  ::glm::Logger::warn(__VA_ARGS__)
#define GLM_LOG_ERROR(...) ::glm::Logger::error(__VA_ARGS__)
#define GLM_LOG_DEBUG(...) ::glm::Logger::debug(__VA_ARGS__)
