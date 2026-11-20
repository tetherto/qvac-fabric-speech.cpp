#pragma once

// music-cli's Core ML cache directory. In order: the explicit
// --coreml-cache-dir, then QVAC_COREML_CACHE_DIR (left to the library, which
// resolves it), then, on macOS, the CLI default
// ~/Library/Application Support/qvac-speech/coreml, so a bare matching GGUF
// generates its VAE decoder sidecar with no flag. Not under Library/Caches:
// a purge there means a new path and a cold compile. nullopt leaves the
// library's resolution as it is.

#include <optional>
#include <string>

namespace tts_cpp::acestep::music_cli {

inline constexpr const char * COREML_CLI_CACHE_SUBDIR = "/Library/Application Support/qvac-speech/coreml";

inline std::optional<std::string> coreml_cache_dir(const char * flag, const char * env, const char * home,
                                                   bool macos) {
    if (flag != nullptr) return std::string(flag);
    if (env != nullptr && *env != '\0') return std::nullopt;
    if (macos && home != nullptr && *home != '\0') return std::string(home) + COREML_CLI_CACHE_SUBDIR;
    return std::nullopt;
}

}  // namespace tts_cpp::acestep::music_cli
