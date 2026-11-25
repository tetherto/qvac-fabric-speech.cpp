#pragma once

// Core ML options of cosyvoice-cli and cosyvoice-bench (header-only, shared
// by the two tools):
//   --coreml-cache-dir DIR      generate the front-end, F0 and HiFT sidecars
//                               from the GGUFs into DIR when none is prebuilt
//                               (macOS 15+); default QVAC_COREML_CACHE_DIR,
//                               else ~/Library/Application Support/qvac-speech/coreml
//   --coreml-wait none|construct|call
//   --coreml-prewarm none|minimal|predicted|all
//   --coreml-units all|cpu_and_ane|cpu_and_gpu|cpu_only
//   --coreml-ignore-prebuilt    (development) plan as if no sidecar sat next
//                               to the GGUFs
//   --coreml-status-json FILE   write the Core ML status (with the
//                               construction group's result) after the runs

#include "tts-cpp/cosyvoice/engine.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace cosyvoice_cli {

constexpr const char * COREML_USAGE =
    "          [--coreml-cache-dir DIR] [--coreml-wait none|construct|call]\n"
    "          [--coreml-prewarm none|minimal|predicted|all]\n"
    "          [--coreml-units all|cpu_and_ane|cpu_and_gpu|cpu_only]\n"
    "          [--coreml-ignore-prebuilt] [--coreml-status-json FILE]\n";

struct coreml_args {
    tts_cpp::CoremlOptions coreml;
    bool ignore_prebuilt = false;
    std::string status_json;
};

// Consumes argv[i] (and its value) when it is a Core ML option; throws
// std::invalid_argument for a bad value.
inline bool parse_coreml_arg(int & i, int argc, char ** argv, coreml_args & out) {
    using O = tts_cpp::CoremlOptions;
    const std::string a = argv[i];
    auto value = [&]() -> std::string {
        if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
        return argv[++i];
    };
    if (a == "--coreml-cache-dir") {
        out.coreml.cache_dir = value();
    } else if (a == "--coreml-wait") {
        const std::string v = value();
        if (v == "none") out.coreml.wait = O::Wait::none;
        else if (v == "construct") out.coreml.wait = O::Wait::construct;
        else if (v == "call") out.coreml.wait = O::Wait::call;
        else throw std::invalid_argument("--coreml-wait must be none, construct or call");
    } else if (a == "--coreml-prewarm") {
        const std::string v = value();
        if (v == "none") out.coreml.prewarm = O::Prewarm::none;
        else if (v == "minimal") out.coreml.prewarm = O::Prewarm::minimal;
        else if (v == "predicted") out.coreml.prewarm = O::Prewarm::predicted;
        else if (v == "all") out.coreml.prewarm = O::Prewarm::all;
        else throw std::invalid_argument("--coreml-prewarm must be none, minimal, predicted or all");
    } else if (a == "--coreml-units") {
        out.coreml.compute_units = value();
    } else if (a == "--coreml-ignore-prebuilt") {
        out.ignore_prebuilt = true;
    } else if (a == "--coreml-status-json") {
        out.status_json = value();
    } else {
        return false;
    }
    return true;
}

// The CLIs keep generated sidecars and warm-load hints under Application
// Support when neither --coreml-cache-dir nor QVAC_COREML_CACHE_DIR names a
// directory (Library/Caches can be purged, and a purge means a new path and a
// cold compile). The library's own default stays "no cache".
inline void apply_coreml_args(const coreml_args & args, tts_cpp::cosyvoice::EngineOptions & opts) {
    opts.coreml = args.coreml;
    opts.coreml_ignore_prebuilt = args.ignore_prebuilt;
#if defined(__APPLE__)
    const char * shared = std::getenv("QVAC_COREML_CACHE_DIR");
    const char * home = std::getenv("HOME");
    if (!opts.coreml.cache_dir && (shared == nullptr || *shared == '\0') && home != nullptr && *home != '\0')
        opts.coreml.cache_dir = std::string(home) + "/Library/Application Support/qvac-speech/coreml";
#endif
}

// {"status": ..., "construction": ...}; false when the file cannot be written.
inline bool write_coreml_status(const coreml_args & args, const tts_cpp::cosyvoice::Engine & engine) {
    if (args.status_json.empty()) return true;
    FILE * f = std::fopen(args.status_json.c_str(), "w");
    if (!f) return false;
    const std::string s = "{\"status\":" + engine.coreml_status() + ",\"construction\":" +
                          engine.coreml_construction() + "}\n";
    const bool ok = std::fwrite(s.data(), 1, s.size(), f) == s.size();
    return std::fclose(f) == 0 && ok;
}

}  // namespace cosyvoice_cli
