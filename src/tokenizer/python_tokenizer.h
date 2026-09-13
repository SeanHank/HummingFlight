#pragma once

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace glm {

// Python Tokenizer subprocess communication class
// Spawns tools/tokenizer_server.py via pipes, line-delimited JSON protocol
// Ensures Tokenizer semantics are 100% consistent with the official transformers implementation
//
// Windows:  python_tokenizer.cpp   (CreateProcessA + anonymous pipes)
// POSIX:    python_tokenizer_posix.cpp (fork + exec + pipe)
class PythonTokenizer {
public:
    PythonTokenizer() = default;
    ~PythonTokenizer();

    PythonTokenizer(const PythonTokenizer&) = delete;
    PythonTokenizer& operator=(const PythonTokenizer&) = delete;

    // Start subprocess: pythonExe = python path, scriptPath = tokenizer_server.py path, modelDir = model directory
    bool start(const std::string& pythonExe,
               const std::string& scriptPath,
               const std::string& modelDir);

    void stop();

    bool isRunning() const { return childProcess_ != 0; }

    // Encode: text -> ids
    std::vector<int> encode(const std::string& text, bool addSpecial = true);

    // Decode: ids -> text
    std::string decode(const std::vector<int>& ids, bool skipSpecial = true);

    // Apply chat template: messages -> ids
    std::vector<int> applyChatTemplate(const std::string& messagesJson, bool addGenPrompt = true);

    int vocabSize() const { return vocabSize_; }
    int eosTokenId() const { return eosId_; }
    int bosTokenId() const { return bosId_; }

private:
#ifdef _WIN32
    using HandleType = HANDLE;
#else
    using HandleType = intptr_t;  // raw POSIX fd / pid, NULL-represented by 0
#endif

    // Send one line of JSON and read one line of response
    bool sendAndRecv(const std::string& jsonReq, std::string& jsonResp);

    // Read one line (terminated by \n)
    bool readLine(std::string& line);

    // Write one line
    bool writeLine(const std::string& line);

    HandleType childProcess_ = 0;
    HandleType stdinWrite_   = 0;
    HandleType stdoutRead_   = 0;
    int vocabSize_ = 0;
    int bosId_ = -1;
    int eosId_ = -1;
    int padId_ = -1;
};

} // namespace glm