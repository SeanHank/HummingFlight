#include "tokenizer/python_tokenizer.h"
#include "tokenizer/subprocess_json.h"
#include "utils/logger.h"
#include <sstream>

namespace glm {

PythonTokenizer::~PythonTokenizer() { stop(); }

bool PythonTokenizer::start(const std::string& pythonExe,
                            const std::string& scriptPath,
                            const std::string& modelDir) {
    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(SECURITY_ATTRIBUTES);
    sa.bInheritHandle = TRUE;

    HANDLE stdinRead = nullptr, stdoutWrite = nullptr;
    if (!CreatePipe(&stdinRead, &stdinWrite_, &sa, 0)) return false;
    if (!CreatePipe(&stdoutRead_, &stdoutWrite, &sa, 0)) {
        CloseHandle(stdinRead); CloseHandle(stdinWrite_); return false;
    }
    // Ensure the C++ side pipe ends are not inherited by the subprocess
    SetHandleInformation(stdinWrite_, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(stdoutRead_, HANDLE_FLAG_INHERIT, 0);

    // Build command line
    std::string cmd = "\"" + pythonExe + "\" \"" + scriptPath + "\" \"" + modelDir + "\"";
    GLM_LOG_DEBUG("Tokenizer command: " + cmd);
    std::vector<char> cmdBuf(cmd.begin(), cmd.end());
    cmdBuf.push_back('\0');

    STARTUPINFOA si{};
    si.cb = sizeof(STARTUPINFOA);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = stdinRead;
    si.hStdOutput = stdoutWrite;
    si.hStdError = GetStdHandle(STD_ERROR_HANDLE); // Keep stderr separate to avoid warning pollution
    PROCESS_INFORMATION pi{};

    BOOL ok = CreateProcessA(nullptr, cmdBuf.data(), nullptr, nullptr, TRUE,
                             CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
    // Close inherited handles on the subprocess side
    CloseHandle(stdinRead);
    CloseHandle(stdoutWrite);
    if (!ok) {
        GLM_LOG_ERROR("Failed to start Python tokenizer subprocess: " + cmd);
        CloseHandle(stdinWrite_); CloseHandle(stdoutRead_);
        return false;
    }
    childProcess_ = pi.hProcess;
    CloseHandle(pi.hThread);

    // Read ready signal (loop read, skipping warning lines from transformers etc.)
    std::string readyLine;
    bool readyFound = false;
    for (int attempt = 0; attempt < 50; ++attempt) {
        if (!readLine(readyLine)) {
            GLM_LOG_ERROR("Timed out waiting for Python tokenizer ready");
            stop();
            return false;
        }
        // Ready signal format: {"ready": true, "info": {...}}
        if (readyLine.find("\"ready\"") != std::string::npos &&
            (readyLine.find("true") != std::string::npos ||
             readyLine.find("True") != std::string::npos)) {
            readyFound = true;
            break;
        }
        // Skip non-ready lines (e.g. transformers warnings)
        GLM_LOG_DEBUG("tokenizer subprocess output (skipped): " + readyLine);
    }
    if (!readyFound) {
        GLM_LOG_ERROR("Python tokenizer ready signal not found (last read: " + readyLine + ")");
        stop();
        return false;
    }
    vocabSize_ = std::atoi(extractJsonField(readyLine, "vocab_size").c_str());
    // bos/eos/pad are in the nested info object, simple extraction
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
        CloseHandle(stdinWrite_);
        stdinWrite_ = nullptr;
    }
    if (stdoutRead_) { CloseHandle(stdoutRead_); stdoutRead_ = nullptr; }
    if (childProcess_) {
        WaitForSingleObject(childProcess_, 3000);
        CloseHandle(childProcess_);
        childProcess_ = nullptr;
    }
}

bool PythonTokenizer::readLine(std::string& line) {
    line.clear();
    char c;
    DWORD bytesRead = 0;
    while (ReadFile(stdoutRead_, &c, 1, &bytesRead, nullptr) && bytesRead > 0) {
        if (c == '\n') return true;
        if (c != '\r') line += c;
    }
    return !line.empty();
}

bool PythonTokenizer::writeLine(const std::string& line) {
    std::string data = line + "\n";
    DWORD written = 0;
    return WriteFile(stdinWrite_, data.data(), DWORD(data.size()), &written, nullptr) &&
           written == DWORD(data.size());
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
    std::string idsStr = extractJsonField(resp, "ids");
    return parseIntArray(idsStr);
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
