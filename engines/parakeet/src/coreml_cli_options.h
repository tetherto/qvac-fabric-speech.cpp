#pragma once

// Core ML command-line options shared by the CLIs of this library
// (parakeet-cli and moss-transcribe): the --coreml-* values and the CLI's
// default cache directory.

#include "parakeet/coreml_options.h"
#include "parakeet_log.h"

#include <cstdlib>
#include <string>

namespace parakeet {

inline bool parse_coreml_use(const std::string & v, CoremlOptions::Use & out) {
    using U = CoremlOptions::Use;
    if (v == "auto") out = U::automatic;
    else if (v == "off") out = U::off;
    else if (v == "require") out = U::require;
    else {
        PARAKEET_LOG_ERROR("error: --coreml-use must be auto, off or require (got '%s')\n", v.c_str());
        return false;
    }
    return true;
}

inline bool parse_coreml_generate(const std::string & v, CoremlOptions::Generate & out) {
    using G = CoremlOptions::Generate;
    if (v == "background") out = G::background;
    else if (v == "never") out = G::never;
    else {
        PARAKEET_LOG_ERROR("error: --coreml-generate must be background or never (got '%s')\n", v.c_str());
        return false;
    }
    return true;
}

inline bool parse_coreml_wait(const std::string & v, CoremlOptions::Wait & out) {
    using W = CoremlOptions::Wait;
    if (v == "none") out = W::none;
    else if (v == "construct") out = W::construct;
    else if (v == "call") out = W::call;
    else {
        PARAKEET_LOG_ERROR("error: --coreml-wait must be none, construct or call (got '%s')\n", v.c_str());
        return false;
    }
    return true;
}

inline bool parse_coreml_prewarm(const std::string & v, CoremlOptions::Prewarm & out) {
    using P = CoremlOptions::Prewarm;
    if (v == "none") out = P::none;
    else if (v == "minimal") out = P::minimal;
    else if (v == "predicted") out = P::predicted;
    else if (v == "all") out = P::all;
    else {
        PARAKEET_LOG_ERROR("error: --coreml-prewarm must be none, minimal, predicted or all (got '%s')\n",
                           v.c_str());
        return false;
    }
    return true;
}

// CLIs keep warm hints under Application Support (Library/Caches can
// be purged, and a purge means a new path and a cold ANE compile).
inline void apply_cli_coreml_defaults(CoremlOptions & coreml) {
#if defined(__APPLE__)
    const char * shared = std::getenv("QVAC_COREML_CACHE_DIR");
    const char * home = std::getenv("HOME");
    if (!coreml.cache_dir && (shared == nullptr || *shared == '\0') && home != nullptr && *home != '\0') {
        coreml.cache_dir = std::string(home) + "/Library/Application Support/qvac-speech/coreml";
    }
#else
    (void) coreml;
#endif
}

}  // namespace parakeet
