#pragma once

// The supertonic CLI's Core ML cache directory. In order: the explicit
// --coreml-cache-dir; then QVAC_COREML_CACHE_DIR, which the library resolves
// (left unset here); then, on macOS, the CLI default
// ~/Library/Application Support/qvac-speech/coreml, so a bare GGUF that a
// vocoder template matches generates its sidecar with no flag. Not under
// Library/Caches: a purge there means a new path and a cold compile.

#include "tts-cpp/coreml_options.h"

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

#include <string>

namespace tts_cpp::supertonic::detail {

inline constexpr const char * CLI_COREML_CACHE_SUBDIR = "/Library/Application Support/qvac-speech/coreml";

inline constexpr bool cli_on_macos() {
#if defined(__APPLE__) && TARGET_OS_OSX
    return true;
#else
    return false;
#endif
}

// `env`: QVAC_COREML_CACHE_DIR; `home`: HOME (either may be null).
inline void apply_cli_coreml_cache_default(CoremlOptions & o, const char * env, const char * home, bool macos) {
    if (o.cache_dir) return;
    if (env != nullptr && *env != '\0') return;
    if (macos && home != nullptr && *home != '\0') o.cache_dir = std::string(home) + CLI_COREML_CACHE_SUBDIR;
}

}  // namespace tts_cpp::supertonic::detail
