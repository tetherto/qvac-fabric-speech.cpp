#include "parakeet_coreml_path.h"

#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>

namespace parakeet {
namespace {

constexpr std::array<const char *, 11> kQuantTags = {
    "f32", "f16", "bf16", "q8_0", "q5_0", "q4_0", "q6_k", "q5_k", "q4_k", "mxfp4", "nvfp4",
};
constexpr const char * kDiarizerInfix = "-diarizer-";
constexpr const char * kSidecarExtension = ".mlmodelc";
constexpr int kDecimalBase = 10;

bool is_quant_tag(const std::string & token) {
    for (const char * tag : kQuantTags) {
        if (token == tag) {
            return true;
        }
    }
    return false;
}

std::size_t basename_separator(const std::string & path, std::size_t slash) {
    const std::size_t sep = path.find_last_of(".-");
    if (sep == std::string::npos) {
        return std::string::npos;
    }
    if (slash != std::string::npos && sep <= slash) {
        return std::string::npos;
    }
    return sep;
}

}  // namespace

static std::string coreml_sidecar_base(const std::string & gguf_path) {
    std::string path  = gguf_path;
    const std::size_t slash = path.find_last_of("/\\");

    const std::size_t dot = path.rfind('.');
    if (dot != std::string::npos && (slash == std::string::npos || dot > slash)) {
        path.erase(dot);
    }

    const std::size_t sep = basename_separator(path, slash);
    if (sep != std::string::npos && is_quant_tag(path.substr(sep + 1))) {
        path.erase(sep);
    }

    return path;
}

std::string coreml_encoder_sidecar_path(const std::string & gguf_path) {
    return coreml_sidecar_base(gguf_path) + "-encoder.mlmodelc";
}

std::string coreml_bypass_encoder_sidecar_path(const std::string & gguf_path) {
    return coreml_sidecar_base(gguf_path) + "-encoder-bypass-pre-encode.mlmodelc";
}

std::string coreml_decoder_sidecar_path(const std::string & gguf_path) {
    return coreml_sidecar_base(gguf_path) + "-decoder.mlmodelc";
}

std::string coreml_diarizer_sidecar_path(const std::string & gguf_path, int capacity) {
    return coreml_sidecar_base(gguf_path) + kDiarizerInfix + std::to_string(capacity) +
        kSidecarExtension;
}

static bool ends_with(const std::string & value, const std::string & suffix) {
    return value.size() >= suffix.size() &&
        value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static bool all_digits(const std::string & value) {
    if (value.empty()) return false;
    for (unsigned char character : value) {
        if (!std::isdigit(character)) return false;
    }
    return true;
}

int coreml_diarizer_sidecar_capacity(const std::string & gguf_path,
                                     const std::string & candidate_path) {
    const std::string prefix = coreml_sidecar_base(gguf_path) + kDiarizerInfix;
    const std::string extension = kSidecarExtension;
    if (candidate_path.size() <= prefix.size() + extension.size() ||
        candidate_path.compare(0, prefix.size(), prefix) != 0 ||
        !ends_with(candidate_path, extension)) {
        return 0;
    }
    const std::string digits = candidate_path.substr(
        prefix.size(), candidate_path.size() - prefix.size() - extension.size());
    if (!all_digits(digits) || digits.size() >= std::to_string(INT32_MAX).size()) return 0;
    return static_cast<int>(std::stol(digits, nullptr, kDecimalBase));
}

}  // namespace parakeet
