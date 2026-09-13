#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace glm {

// Native in-process BPE tokenizer (#10).
//
// Parses the model's tokenizer.json (BPE + GPT-2 style pre-tokenizer +
// ByteLevel) with picojson and runs the full pipeline in-process:
//
//   added-token split -> GPT-2 regex pre-tokenize -> ByteLevel byte map
//   -> BPE merges -> vocab ids
//
// Used with --tokenizer native (opt-in; the Python bridge stays the default
// so golden outputs are untouched).
class NativeBpeTokenizer {
public:
    NativeBpeTokenizer() = default;

    // Load tokenizer.json (required) and tokenizer_config.json (for bos/eos).
    bool load(const std::string& tokenizerJsonPath,
              const std::string& tokenizerConfigPath = {});

    std::vector<int> encode(const std::string& text, bool addSpecial = true);
    std::string decode(const std::vector<int>& ids, bool skipSpecial = true);

    int vocabSize() const { return vocabSize_; }
    int eosTokenId() const { return eosId_; }
    int bosTokenId() const { return bosId_; }
    bool isLoaded() const { return vocabSize_ > 0; }

private:
    struct AddedToken {
        std::string content;
        int id;
    };

    // Pre-tokenize raw UTF-8 into GPT-2 regex chunks (each a byte slice).
    std::vector<std::string> preTokenize(const std::string& text);

    // ByteLevel encode of one chunk -> sequence of byte-mapped unicode chars
    // joined into one string, then BPE-merged into vocab sub-token ids.
    std::vector<int> encodeChunk(const std::string& chunk);

    // BPE merge on a sequence of single-codepoint strings.
    std::vector<std::string> bpeMerge(const std::vector<std::string>& chars);

    int vocabSize_ = 0;
    int bosId_ = -1;
    int eosId_ = -1;

    std::unordered_map<std::string, int> vocab_;       // token -> id
    std::vector<std::string> vocabById_;               // id -> token
    std::unordered_map<std::string, int> merges_;      // "a b" -> rank
    std::vector<AddedToken> addedTokens_;              // longest-first order
};

} // namespace glm