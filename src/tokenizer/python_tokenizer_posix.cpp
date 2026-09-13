#include "tokenizer/python_tokenizer.h"
#include "tokenizer/subprocess_json.h"
#include "utils/logger.h"

#ifndef _WIN32

#include <cstdlib>
#include <sstream>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

namespace glm {

namespace {
constexpr size_t kMaxPipeBytes = 64 * 1024 * 1024;  // hard guard on one line
}

PythonTokenizer::~PythonTokenizer() { stop(); }

bool PythonTokenizer::start(const std::string& pythonExe,
                            const std::string& scriptPath,
                            const std::string& modelDir) {
    int toChild[2];   // parent writes stdin
    int fromChild[2]; // parent reads stdout
    if (pipe(toChild) != 0 || pipe(fromChild) != 0) {
        GLM_LOG_ERROR("pipe() failed for tokenizer subprocess");
        return false;
    }

    pid_t pid = fork();
    if (pid < 0) {
        GLM_LOG_ERROR("fork() failed for tokenizer subprocess");
        ::close(toChild[0]); ::close(toChild[1]);
        ::close(fromChild[0]); ::close(fromChild[1]);
        return false;
    }

    if (pid == 0) { // child
        ::dup2(toChild[0], STDIN_FILENO);
        ::dup2(fromChild[1], STDOUT_FILENO);
        // Keep stderr inherited so Python warnings surface in the console.
        ::close(toChild[0]); ::close(toChild[1]);
        ::close(fromChild[0]); ::close(fromChild[1]);

        std::string py(pythonExe), sc(scriptPath), md(modelDir);
        execl(py.c_str(), py.c_str(), sc.c_str(), md.c_str(), nullptr);
        // Only reached on exec failure.
        _exit(127);
    }

    // Parent: keep the parent-side ends, close child-side ends.
    ::close(toChild[0]);
    ::close(fromChild[1]);
    stdinWrite_ = intptr_t(toChild[1]);
    stdoutRead_ = intptr_t(fromChild[0]);
    childProcess_ = intptr_t(pid);

    // Read ready signal (loop read, skipping warning lines).
    std::string readyLine;
    bool readyFound = false;
    for (int attempt = 0; attempt < 50; ++attempt) {
        if (!readLine(readyLine)) {
            GLM_LOG_ERROR("Timed out waiting for Python tokenizer ready");
            stop();
            return false;
        }
        if (readyLine.find("\"ready\"") != std::string::npos &&
            (readyLine.find("true") != std::string::npos ||
             readyLine.find("True") != std::string::npos)) {
            readyFound = true;
            break;
        }
        GLM_LOG_DEBUG("tokenizer subprocess output (skipped): " + readyLine);
    }
    if (!readyFound) {
        GLM_LOG_ERROR("Python tokenizer ready signal not found (last read: " + readyLine + ")");
        stop();
        return false;
    }
    vocabSize_ = std::atoi(extractJsonField(readyLine, "vocab_size").c_str());
    auto extractNested = [&](const std::string& key) -> int {
        std::string pat = "\"" + key + "\":";
        size_t p = readyLine.find(pat);
        if (p == std::string::npos) return -1;
        p += pat.size();
        while (p < readyLine.size() && readyLine[p] == ' ') p++;
        if (p < readyLine.size() && readyLine[p] == 'n') return -1; // null
        return std::atoi(readyLine.c_str() + p);
    };
    bosId_ = extractNested("bos");
    eosId_ = extractNested("eos");
    padId_ = extractNested("pad");

    GLM_LOG_INFO("Python tokenizer ready: vocab=" + std::to_string(vocabSize_) +
                 ", eos=" + std::to_string(eosId_));
    return true;
}

void PythonTokenizer::stop() {
    if (stdinWrite_) {
        writeLine("{\"cmd\":\"quit\"}");
        ::close(int(stdinWrite_));
        stdinWrite_ = 0;
    }
    if (stdoutRead_) {
        ::close(int(stdoutRead_));
        stdoutRead_ = 0;
    }
    if (childProcess_) {
        pid_t pid = pid_t(childProcess_);
        childProcess_ = 0;
        for (int i = 0; i < 300; ++i) {  // up to ~3s
            int status = 0;
            pid_t r = waitpid(pid, &status, WNOHANG);
            if (r == pid) break;
            struct timespec ts = {0, 10 * 1000 * 1000}; // 10 ms
            nanosleep(&ts, nullptr);
        }
    }
}

bool PythonTokenizer::readLine(std::string& line) {
    line.clear();
    char c;
    while (true) {
        ssize_t n = ::read(int(stdoutRead_), &c, 1);
        if (n <= 0) return !line.empty();
        if (c == '\n') return true;
        if (c != '\r') line += c;
        if (line.size() > kMaxPipeBytes) return false;
    }
}

bool PythonTokenizer::writeLine(const std::string& line) {
    std::string data = line + "\n";
    size_t written = 0;
    while (written < data.size()) {
        ssize_t n = ::write(int(stdinWrite_), data.data() + written, data.size() - written);
        if (n <= 0) return false;
        written += size_t(n);
    }
    return true;
}

bool PythonTokenizer::sendAndRecv(const std::string& jsonReq, std::string& jsonResp) {
    if (!writeLine(jsonReq)) return false;
    return readLine(jsonResp);
}

std::vector<int> PythonTokenizer::encode(const std::string& text, bool addSpecial) {
    std::string req = "{\"cmd\":\"encode\",\"text\":\"" + escapeJsonString(text) +
                      "\",\"add_special\":" + (addSpecial ? "true" : "false") + "}";
    std::string resp;
    if (!sendAndRecv(req, resp)) return {};
    return parseIntArray(extractJsonField(resp, "ids"));
}

std::string PythonTokenizer::decode(const std::vector<int>& ids, bool skipSpecial) {
    std::string req = "{\"cmd\":\"decode\",\"ids\":[";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i) req += ",";
        req += std::to_string(ids[i]);
    }
    req += "],\"skip_special\":";
    req += (skipSpecial ? "true" : "false");
    req += "}";
    std::string resp;
    if (!sendAndRecv(req, resp)) return "";
    return parseJsonString(extractJsonField(resp, "text"));
}

std::vector<int> PythonTokenizer::applyChatTemplate(const std::string& messagesJson, bool addGenPrompt) {
    std::string req = "{\"cmd\":\"apply_chat\",\"messages\":" + messagesJson +
                      ",\"add_generation_prompt\":";
    req += (addGenPrompt ? "true" : "false");
    req += "}";
    std::string resp;
    if (!sendAndRecv(req, resp)) return {};
    return parseIntArray(extractJsonField(resp, "ids"));
}

} // namespace glm

#endif // !_WIN32