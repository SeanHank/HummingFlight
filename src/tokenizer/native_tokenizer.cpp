#include "tokenizer/native_tokenizer.h"

#include "third_party/picojson.h"
#include "tokenizer/unicode_ranges.h"
#include "utils/logger.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace glm {

namespace {

// GPT-2 byte->unicode table (the standard ByteLevel mapping).
struct ByteMap {
    uint32_t byteToCp[256] = {};
    int extraCount = 0;
    ByteMap() {
        int n = 0;
        struct Range { int a, b; };
        const Range ranges[] = {{0x21, 0x7E}, {0xA1, 0xAC}, {0xAE, 0xFF}};
        for (int b = 0; b < 256; ++b) byteToCp[b] = 0xFFFFFFFF;
        for (const Range& r : ranges)
            for (int b = r.a; b <= r.b; ++b) byteToCp[b] = uint32_t(b);
        for (int b = 0; b < 256; ++b) {
            if (byteToCp[b] == 0xFFFFFFFF) byteToCp[b] = uint32_t(256 + (n++));
        }
        extraCount = n;
    }
};

const ByteMap& byteMap() {
    static const ByteMap m;
    return m;
}

int inRuns(uint32_t cp, const uint32_t (&runs)[][2], uint32_t count) {
    uint32_t lo = 0, hi = count;
    while (lo < hi) {
        uint32_t mid = (lo + hi) / 2;
        if (cp < runs[mid][0]) hi = mid;
        else if (cp > runs[mid][1]) lo = mid + 1;
        else return 1;
    }
    return 0;
}

using tokenizer_detail::kLetterRuns;
using tokenizer_detail::kLetterRunCount;
using tokenizer_detail::kNumberRuns;
using tokenizer_detail::kNumberRunCount;
using tokenizer_detail::kSpaceRuns;
using tokenizer_detail::kSpaceRunCount;

bool isLetter(uint32_t cp) { return inRuns(cp, kLetterRuns, kLetterRunCount) != 0; }
bool isNumber(uint32_t cp) { return inRuns(cp, kNumberRuns, kNumberRunCount) != 0; }
bool isSpace(uint32_t cp) { return inRuns(cp, kSpaceRuns, kSpaceRunCount) != 0; }
bool isNewline(uint32_t cp) { return cp == '\r' || cp == '\n'; }
bool isC(uint32_t cp) { return !isSpace(cp) && !isLetter(cp) && !isNumber(cp); }
bool isNonspace(uint32_t cp) { return !isSpace(cp); }
bool isLetterNumber(uint32_t cp) { return isLetter(cp) || isNumber(cp); }

void utf8Append(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
        out.push_back(char(cp));
    } else if (cp < 0x800) {
        out.push_back(char(0xC0 | (cp >> 6)));
        out.push_back(char(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(char(0xE0 | (cp >> 12)));
        out.push_back(char(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(char(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(char(0xF0 | (cp >> 18)));
        out.push_back(char(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(char(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(char(0x80 | (cp & 0x3F)));
    }
}

bool asciiCaseEq(uint32_t cp, char c) {
    if (cp >= 'A' && cp <= 'Z') return cp == (uint32_t)c || cp == (uint32_t)(c - 'A' + 'a');
    if (cp >= 'a' && cp <= 'z') return cp == (uint32_t)c || cp == (uint32_t)(c - 'a' + 'A');
    return cp == (uint32_t)c;
}

struct Cp {
    uint32_t v;
    size_t off;
};

std::vector<Cp> decodeCps(const std::string& s, size_t& totalBytes) {
    std::vector<Cp> out;
    totalBytes = s.size();
    size_t i = 0;
    while (i < s.size()) {
        uint8_t b0 = uint8_t(s[i]);
        uint32_t cp = b0;
        size_t len = 1;
        if (b0 >= 0xF0 && i + 3 < s.size()) { cp = ((b0 & 7) << 18) | ((uint8_t(s[i + 1]) & 63) << 12) | ((uint8_t(s[i + 2]) & 63) << 6) | (uint8_t(s[i + 3]) & 63); len = 4; }
        else if (b0 >= 0xE0 && i + 2 < s.size()) { cp = ((b0 & 15) << 12) | ((uint8_t(s[i + 1]) & 63) << 6) | (uint8_t(s[i + 2]) & 63); len = 3; }
        else if (b0 >= 0xC0 && i + 1 < s.size()) { cp = ((b0 & 31) << 6) | (uint8_t(s[i + 1]) & 63); len = 2; }
        out.push_back({cp, i});
        i += len;
    }
    return out;
}

bool matchContraction(const std::vector<Cp>& cps, size_t& i) {
    if (cps[i].v != '\'') return false;
    size_t j = i + 1;
    if (j >= cps.size()) return false;
    const char* rest = nullptr;
    if (asciiCaseEq(cps[j].v, 's')) rest = "";
    else if (asciiCaseEq(cps[j].v, 't')) rest = "";
    else if (j + 1 < cps.size() && asciiCaseEq(cps[j].v, 'r') && asciiCaseEq(cps[j + 1].v, 'e')) { rest = ""; j += 1; }
    else if (j + 1 < cps.size() && asciiCaseEq(cps[j].v, 'v') && asciiCaseEq(cps[j + 1].v, 'e')) { rest = ""; j += 1; }
    else if (asciiCaseEq(cps[j].v, 'm')) rest = "";
    else if (j + 1 < cps.size() && asciiCaseEq(cps[j].v, 'l') && asciiCaseEq(cps[j + 1].v, 'l')) { rest = ""; j += 1; }
    else if (asciiCaseEq(cps[j].v, 'd')) rest = "";
    else return false;
    (void)rest;
    i = j + 1;
    return true;
}

} // namespace

bool NativeBpeTokenizer::load(const std::string& jsonPath, const std::string& configPath) {
    std::ifstream in(jsonPath, std::ios::binary);
    if (!in) {
        GLM_LOG_ERROR("NativeBpeTokenizer: cannot open " + jsonPath);
        return false;
    }
    std::stringstream ss;
    ss << in.rdbuf();
    picojson::value root;
    std::string err = picojson::parse(root, ss.str());
    if (!err.empty()) {
        GLM_LOG_ERROR("NativeBpeTokenizer: tokenizer.json parse error: " + err);
        return false;
    }
    const picojson::object& obj = root.get<picojson::object>();
    auto find = [&obj](const std::string& k) -> const picojson::value* {
        auto it = obj.find(k);
        return it == obj.end() ? nullptr : &it->second;
    };

    const picojson::value* model = find("model");
    if (!model || !model->is<picojson::object>() ||
        model->get<picojson::object>().at("type").to_str() != "BPE") {
        GLM_LOG_ERROR("NativeBpeTokenizer: unsupported model (native path requires BPE)");
        return false;
    }
    const picojson::object& m = model->get<picojson::object>();
    const picojson::value* vocabV = m.count("vocab") ? &m.at("vocab") : nullptr;
    if (!vocabV) {
        GLM_LOG_ERROR("NativeBpeTokenizer: no vocab in tokenizer.json");
        return false;
    }
    for (const auto& kv : vocabV->get<picojson::object>()) {
        vocab_[kv.first] = int(kv.second.get<double>());
    }
    vocabById_.assign(vocab_.size(), {});
    for (const auto& kv : vocab_) {
        if (size_t(kv.second) < vocabById_.size()) vocabById_[size_t(kv.second)] = kv.first;
    }
    vocabSize_ = int(vocab_.size());

    const picojson::value* mergesV = m.count("merges") ? &m.at("merges") : nullptr;
    if (mergesV) {
        int rank = 0;
        for (const picojson::value& mv : mergesV->get<picojson::array>()) {
            const picojson::array& pair = mv.get<picojson::array>();
            merges_[pair[0].to_str() + " " + pair[1].to_str()] = rank++;
        }
    }

    const picojson::value* addedV = find("added_tokens");
    if (addedV) {
        std::vector<AddedToken> at;
        for (const picojson::value& av : addedV->get<picojson::array>()) {
            const picojson::object& ao = av.get<picojson::object>();
            at.push_back({ao.at("content").to_str(), int(ao.at("id").get<double>())});
        }
        std::stable_sort(at.begin(), at.end(), [](const AddedToken& a, const AddedToken& b) {
            if (a.content.size() != b.content.size()) return a.content.size() > b.content.size();
            return a.id < b.id;
        });
        addedTokens_ = std::move(at);
    }

    int eosId = -1, bosId = -1;
    if (!configPath.empty()) {
        std::ifstream cin(configPath, std::ios::binary);
        if (cin) {
            std::stringstream css;
            css << cin.rdbuf();
            picojson::value croot;
            picojson::parse(croot, css.str());
            const picojson::object& cobj = croot.get<picojson::object>();
            auto resolve = [&](const char* key) {
                auto it = cobj.find(key);
                if (it == cobj.end()) return -1;
                std::string content;
                if (it->second.is<std::string>()) content = it->second.to_str();
                else if (it->second.is<picojson::object>()) {
                    const picojson::object& o = it->second.get<picojson::object>();
                    if (o.find("id") != o.end()) return int(o.at("id").get<double>());
                    if (o.find("content") != o.end()) content = o.at("content").to_str();
                }
                if (content.empty()) return -1;
                auto vit = vocab_.find(content);
                if (vit != vocab_.end()) return vit->second;
                for (const AddedToken& t : addedTokens_)
                    if (t.content == content) return t.id;
                return -1;
            };
            eosId = resolve("eos_token");
            bosId = resolve("bos_token");
        }
    }
    if (eosId < 0 || bosId < 0) {
        for (const AddedToken& t : addedTokens_) {
            if (eosId < 0 && t.content == "<|endoftext|>") eosId = t.id;
            if (bosId < 0 && t.content == "<s>") bosId = t.id;
        }
    }
    eosId_ = eosId;
    bosId_ = bosId;
    GLM_LOG_INFO("NativeBpeTokenizer loaded: vocab=" + std::to_string(vocabSize_) +
                 " merges=" + std::to_string(merges_.size()) + " added=" +
                 std::to_string(addedTokens_.size()));
    return vocabSize_ > 0;
}

std::vector<std::string> NativeBpeTokenizer::preTokenize(const std::string& text) {
    size_t totalBytes = 0;
    std::vector<Cp> cps = decodeCps(text, totalBytes);
    std::vector<std::string> chunks;
    size_t i = 0;
    while (i < cps.size()) {
        size_t startOff = cps[i].off;
        size_t endOff = 0;
        size_t start = i;

        size_t j = i;
        bool matched = false;
        if (matchContraction(cps, j)) { matched = true; }
        if (!matched) {
            const Cp& c0 = cps[i];
            if (isLetter(c0.v)) {
                size_t k = i + 1;
                while (k < cps.size() && isLetter(cps[k].v)) ++k;
                j = k;
                matched = true;
            } else if (!isNewline(c0.v) && !isLetterNumber(c0.v)) {
                size_t k = i + 1;
                if (k < cps.size() && isLetter(cps[k].v)) {
                    while (k < cps.size() && isLetter(cps[k].v)) ++k;
                    j = k;
                    matched = true;
                }
            }
        }
        if (!matched) {
            const Cp& c0 = cps[i];
            if (isNumber(c0.v)) {
                size_t k = i + 1;
                int n = 1;
                while (k < cps.size() && isNumber(cps[k].v) && n < 3) { ++k; ++n; }
                j = k;
                matched = true;
            }
        }
        if (!matched) {
            size_t k = i;
            if (isSpace(cps[k].v)) ++k;
            if (k < cps.size() && isC(cps[k].v)) {
                while (k < cps.size() && isC(cps[k].v)) ++k;
                while (k < cps.size() && isNewline(cps[k].v)) ++k;
                j = k;
                matched = true;
            }
        }
        if (!matched) {
            size_t k = i;
            size_t lastNl = i;
            bool foundNl = false;
            while (k < cps.size() && isSpace(cps[k].v)) {
                if (isNewline(cps[k].v)) { lastNl = k; foundNl = true; }
                ++k;
            }
            if (foundNl) {
                j = lastNl + 1;
                matched = true;
            }
        }
        // \s+(?!\S): whitespace run whose suffix is followed by something that
        // is not a non-space. Greedy with backtracking: prefer the whole run,
        // else drop trailing whitespace one char at a time.
        if (!matched) {
            size_t k = i;
            while (k < cps.size() && isSpace(cps[k].v)) ++k;
            size_t run = k - i;
            if (run >= 1) {
                bool afterNotS = (k == cps.size()) || !isNonspace(cps[k].v);
                size_t take = afterNotS ? run : (run >= 2 ? run - 1 : 0);
                if (take > 0 && run >= 1) {
                    j = i + take;
                    matched = true;
                }
            }
        }
        // \s+: any remaining whitespace run.
        if (!matched) {
            size_t k = i;
            while (k < cps.size() && isSpace(cps[k].v)) ++k;
            if (k > i) { j = k; matched = true; }
        }
        if (!matched) {
            j = start + 1;
        }
        endOff = (j < cps.size()) ? cps[j].off : totalBytes;
        chunks.push_back(text.substr(startOff, endOff - startOff));
        i = j;
    }
    return chunks;
}

std::vector<std::string> NativeBpeTokenizer::bpeMerge(const std::vector<std::string>& chars) {
    std::vector<std::string> word = chars;
    if (word.size() < 2) return word;
    auto bestPair = [&]() -> std::pair<int, int> {
        int bestRank = 0x7FFFFFFF;
        int bestIdx = -1;
        for (size_t k = 0; k + 1 < word.size(); ++k) {
            auto it = merges_.find(word[k] + " " + word[k + 1]);
            int r = it == merges_.end() ? 0x7FFFFFFF : it->second;
            if (r < bestRank) { bestRank = r; bestIdx = int(k); }
        }
        return {bestIdx, bestRank};
    };
    while (word.size() >= 2) {
        auto [idx, rank] = bestPair();
        if (idx < 0) break;
        const std::string first = std::move(word[size_t(idx)]);
        const std::string second = std::move(word[size_t(idx) + 1]);
        word[size_t(idx)] = first + second;
        word.erase(word.begin() + idx + 1);
        if (word.size() == 1) break;
    }
    return word;
}

std::vector<int> NativeBpeTokenizer::encodeChunk(const std::string& chunk) {
    std::vector<std::string> chars;
    chars.reserve(chunk.size());
    for (unsigned char b : chunk) {
        std::string s;
        utf8Append(s, byteMap().byteToCp[b]);
        chars.push_back(std::move(s));
    }
    std::vector<std::string> merged = bpeMerge(chars);
    std::vector<int> out;
    for (const std::string& piece : merged) {
        auto it = vocab_.find(piece);
        if (it != vocab_.end()) {
            out.push_back(it->second);
            continue;
        }
        for (size_t c = 0; c + 1 < piece.size();) {
            size_t n = 1;
            if (uint8_t(piece[c]) >= 0xF0) n = 4;
            else if (uint8_t(piece[c]) >= 0xE0) n = 3;
            else if (uint8_t(piece[c]) >= 0xC0) n = 2;
            auto vit = vocab_.find(piece.substr(c, n));
            if (vit != vocab_.end()) out.push_back(vit->second);
            c += n;
        }
    }
    return out;
}

std::vector<int> NativeBpeTokenizer::encode(const std::string& text, bool addSpecial) {
    std::vector<int> ids;
    struct Span {
        bool added;
        int id;
        size_t begin, end;
    };
    std::vector<Span> spans;
    size_t pos = 0;
    while (pos < text.size()) {
        const AddedToken* best = nullptr;
        for (const AddedToken& t : addedTokens_) {
            if (t.content.size() <= text.size() - pos &&
                text.compare(pos, t.content.size(), t.content) == 0) {
                best = &t;
                break;
            }
        }
        if (best) {
            spans.push_back({true, best->id, pos, pos + best->content.size()});
            pos += best->content.size();
        } else {
            size_t x = pos;
            while (x < text.size()) {
                bool startAtX = false;
                for (const AddedToken& t : addedTokens_) {
                    if (t.content.size() <= text.size() - x &&
                        text.compare(x, t.content.size(), t.content) == 0) {
                        startAtX = true;
                        break;
                    }
                }
                if (startAtX) break;
                ++x;
            }
            spans.push_back({false, -1, pos, x});
            pos = x;
        }
    }
    for (const Span& sp : spans) {
        if (sp.added) {
            ids.push_back(sp.id);
        } else {
            if (sp.begin == sp.end) continue;
            std::vector<std::string> chunks = preTokenize(text.substr(sp.begin, sp.end - sp.begin));
            for (const std::string& c : chunks) {
                std::vector<int> cids = encodeChunk(c);
                ids.insert(ids.end(), cids.begin(), cids.end());
            }
        }
    }
    if (addSpecial && eosId_ >= 0) {
        if (ids.empty() || ids.back() != eosId_) ids.push_back(eosId_);
    }
    return ids;
}

std::string NativeBpeTokenizer::decode(const std::vector<int>& ids, bool skipSpecial) {
    std::unordered_map<int, const std::string*> specials;
    for (const AddedToken& t : addedTokens_) specials[t.id] = &t.content;
    std::vector<uint8_t> bytes;
    for (int id : ids) {
        auto sit = specials.find(id);
        if (sit != specials.end()) {
            if (!skipSpecial) {
                const std::string& c = *sit->second;
                bytes.insert(bytes.end(), c.begin(), c.end());
            }
            continue;
        }
        if (id < 0 || size_t(id) >= vocabById_.size()) continue;
        const std::string& token = vocabById_[size_t(id)];
        for (size_t i = 0; i < token.size();) {
            size_t n = 1;
            uint32_t cp = uint8_t(token[i]);
            if (cp >= 0xF0 && i + 3 < token.size()) { cp = ((cp & 7) << 18) | ((uint8_t(token[i + 1]) & 63) << 12) | ((uint8_t(token[i + 2]) & 63) << 6) | (uint8_t(token[i + 3]) & 63); n = 4; }
            else if (cp >= 0xE0 && i + 2 < token.size()) { cp = ((cp & 15) << 12) | ((uint8_t(token[i + 1]) & 63) << 6) | (uint8_t(token[i + 2]) & 63); n = 3; }
            else if (cp >= 0xC0 && i + 1 < token.size()) { cp = ((cp & 31) << 6) | (uint8_t(token[i + 1]) & 63); n = 2; }
            const uint32_t* lk = byteMap().byteToCp;
            for (int b = 0; b < 256; ++b) {
                if (lk[b] == cp) { bytes.push_back(uint8_t(b)); break; }
            }
            i += n;
        }
    }
    std::string out;
    out.reserve(bytes.size());
    size_t i = 0;
    while (i < bytes.size()) {
        uint8_t b = bytes[i];
        if (b < 0x80) { out.push_back(char(b)); ++i; }
        else if ((b & 0xE0) == 0xC0 && i + 1 < bytes.size()) {
            uint8_t b1 = bytes[i + 1];
            if ((b1 & 0xC0) == 0x80) {
                uint32_t cp = ((b & 0x1F) << 6) | (b1 & 0x3F);
                if (cp >= 0x80) { utf8Append(out, cp); i += 2; }
                else { utf8Append(out, 0xFFFD); ++i; }
            } else { utf8Append(out, 0xFFFD); ++i; }
        }
        else if ((b & 0xF0) == 0xE0 && i + 2 < bytes.size()) {
            uint8_t b1 = bytes[i + 1], b2 = bytes[i + 2];
            if ((b1 & 0xC0) == 0x80 && (b2 & 0xC0) == 0x80) {
                uint32_t cp = ((b & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F);
                if (cp >= 0x800 && !(cp >= 0xD800 && cp <= 0xDFFF)) { utf8Append(out, cp); i += 3; }
                else { utf8Append(out, 0xFFFD); ++i; }
            } else { utf8Append(out, 0xFFFD); ++i; }
        }
        else if ((b & 0xF8) == 0xF0 && i + 3 < bytes.size()) {
            uint8_t b1 = bytes[i + 1], b2 = bytes[i + 2], b3 = bytes[i + 3];
            if ((b1 & 0xC0) == 0x80 && (b2 & 0xC0) == 0x80 && (b3 & 0xC0) == 0x80) {
                uint32_t cp = ((b & 0x07) << 18) | ((b1 & 0x3F) << 12) | ((b2 & 0x3F) << 6) | (b3 & 0x3F);
                if (cp >= 0x10000 && cp <= 0x10FFFF) { utf8Append(out, cp); i += 4; }
                else { utf8Append(out, 0xFFFD); ++i; }
            } else { utf8Append(out, 0xFFFD); ++i; }
        }
        else { utf8Append(out, 0xFFFD); ++i; }
    }
    return out;
}

} // namespace glm