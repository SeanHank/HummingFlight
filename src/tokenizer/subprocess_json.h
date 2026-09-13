#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace glm {

// Minimal JSON helpers shared by the Windows and POSIX Python-tokenizer
// subprocess bridges. The protocol is line-delimited JSON, so only simple
// scalar/array parsing is required.

inline std::string escapeJsonString(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (unsigned char(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

// Extract scalar value from JSON like {"key": value}.
inline std::string extractJsonField(const std::string& json, const std::string& key) {
    std::string pat = "\"" + key + "\"";
    size_t pos = json.find(pat);
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos + pat.size());
    if (pos == std::string::npos) return "";
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    if (pos >= json.size()) return "";
    if (json[pos] == '"') {
        size_t end = pos + 1;
        while (end < json.size() && json[end] != '"') {
            if (json[end] == '\\') end++;
            end++;
        }
        return json.substr(pos + 1, end - pos - 1);
    }
    size_t end = pos;
    int depth = 0;
    while (end < json.size()) {
        if (json[end] == '[' || json[end] == '{') depth++;
        else if (json[end] == ']' || json[end] == '}') { if (depth == 0) break; depth--; }
        else if (json[end] == ',' && depth == 0) break;
        end++;
    }
    return json.substr(pos, end - pos);
}

// Parse JSON array [1,2,3] -> vector<int>.
inline std::vector<int> parseIntArray(const std::string& s) {
    std::vector<int> result;
    size_t pos = s.find('[');
    if (pos == std::string::npos) return result;
    pos++;
    while (pos < s.size() && s[pos] != ']') {
        while (pos < s.size() && (s[pos] == ' ' || s[pos] == ',')) pos++;
        if (pos >= s.size() || s[pos] == ']') break;
        size_t end = pos;
        while (end < s.size() && s[end] != ',' && s[end] != ']') end++;
        try {
            result.push_back(std::stoi(s.substr(pos, end - pos)));
        } catch (...) {}
        pos = end;
    }
    return result;
}

// Parse JSON string value (strip quotes + unescape).
inline std::string parseJsonString(const std::string& s) {
    if (s.empty() || s[0] != '"') return s;
    std::string out;
    for (size_t i = 1; i < s.size() && s[i] != '"'; ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            char next = s[++i];
            switch (next) {
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case 'u': {
                    if (i + 4 < s.size()) {
                        std::string hex = s.substr(i + 1, 4);
                        unsigned int cp = std::stoul(hex, nullptr, 16);
                        i += 4;
                        if (cp < 0x80) out += char(cp);
                        else if (cp < 0x800) {
                            out += char(0xC0 | (cp >> 6));
                            out += char(0x80 | (cp & 0x3F));
                        } else {
                            out += char(0xE0 | (cp >> 12));
                            out += char(0x80 | ((cp >> 6) & 0x3F));
                            out += char(0x80 | (cp & 0x3F));
                        }
                    }
                    break;
                }
                default: out += next;
            }
        } else {
            out += s[i];
        }
    }
    return out;
}

} // namespace glm