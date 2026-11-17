#pragma once

// Cache fixtures for the tests that check a stage's Core ML route and fit do
// not depend on the cache (test-*-fit-stable): what a generation leaves in a
// cache directory, and warm hints written exactly where the runtime looks for
// them (the hint id of an artifact's model, its compute units, this host and
// OS build), so a route or fit that read them would see them.
//
// The hint helpers need the engine's runtime namespace: define
// QVAC_COREML_NS (and TTS_CPP_USE_COREML, or the engine's equivalent
// QVAC_COREML_FIT_STABLE_HINTS) and add engines/coreml/{include,src} to the
// include path.

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

#if defined(TTS_CPP_USE_COREML) || defined(QVAC_COREML_FIT_STABLE_HINTS)
#include "hints.h"
#include "qvac_coreml/manager.h"
#endif

namespace qvac_coreml_fit_stable {

namespace fs = std::filesystem;

inline void write_file(const fs::path & p, const std::string & text) {
    fs::create_directories(p.parent_path());
    std::ofstream(p) << text;
}

inline std::vector<fs::path> artifact_dirs(const fs::path & cache) {
    std::vector<fs::path> out;
    std::error_code ec;
    for (const auto & e : fs::directory_iterator(cache / "qvac-coreml-v1" / "artifacts", ec)) {
        if (e.is_directory()) out.push_back(e.path());
    }
    return out;
}

#if defined(TTS_CPP_USE_COREML) || defined(QVAC_COREML_FIT_STABLE_HINTS)
namespace pcm = QVAC_COREML_NS;

// The hint record the runtime reads for the model of `artifact_dir`, in this
// process or (`host_id`) in another program: the runtime's host id is the
// program's name.
inline std::string runtime_hint_path(const fs::path & cache, const fs::path & artifact_dir, const std::string & units,
                              const std::string & host_id = {}) {
    const auto m = pcm::Manager::instance();
    return pcm::detail::hint_path_for(cache.string(), host_id.empty() ? m->host_id() : host_id,
                                      (artifact_dir / "model.mlmodelc").string(), "main", units, m->os_build());
}

inline std::optional<int64_t> stale_until(bool stale) {
    if (!stale) return std::nullopt;
    return pcm::detail::unix_now_s() + 365LL * 24 * 3600;
}

// Rewrites every hint record under `cache` with `stale` and writes one at the
// runtime's id for every artifact; returns how many read back at a runtime id.
inline int write_hints(const fs::path & cache, const std::string & units, bool stale) {
    std::error_code ec;
    for (const auto & e : fs::directory_iterator(cache / "qvac-coreml-v1" / "hints", ec)) {
        if (e.path().extension() != ".json") continue;
        std::optional<pcm::detail::HintRecord> rec = pcm::detail::read_hint(e.path().string());
        if (!rec) continue;
        rec->stale_until_s = stale_until(stale);
        pcm::detail::write_hint(e.path().string(), *rec);
    }
    int at_runtime_id = 0;
    for (const fs::path & art : artifact_dirs(cache)) {
        const std::string path = runtime_hint_path(cache, art, units);
        pcm::detail::HintRecord rec;
        rec.warm_load_ms = 120;
        rec.cold_load_ms = 8000;
        rec.last_ok = pcm::detail::iso8601_utc(pcm::detail::unix_now_s());
        rec.stale_until_s = stale_until(stale);
        rec.artifact = (art / "model.mlmodelc").string();
        fs::create_directories(fs::path(path).parent_path());
        const std::optional<pcm::detail::HintRecord> back =
            pcm::detail::write_hint(path, rec) ? pcm::detail::read_hint(path) : std::nullopt;
        if (back && back->stale_until_s.has_value() == stale) ++at_runtime_id;
    }
    return at_runtime_id;
}
#endif

// What a generation leaves in `cache`: the artifact of `template_id` under
// `key` (32 hex digits), its memo and its used marker.
inline void simulate_generation(const fs::path & cache, const std::string & template_id,
                                const std::string & key = "0123456789abcdef0123456789abcdef") {
    const fs::path root = cache / "qvac-coreml-v1";
    const fs::path art = root / "artifacts" / (template_id + "." + key);
    write_file(art / "model.mlmodelc" / "model.mil", "program(1.0)\n");
    write_file(art / "artifact.json", "{\"schema\":1,\"complete\":true,\"artifact_key\":\"" + key +
                                          "\",\"inventory\":[{\"path\":\"model.mlmodelc/model.mil\",\"bytes\":13}]}\n");
    write_file(root / "sources" / ("fedcba9876543210fedcba9876543210." + key + ".json"), "{\"schema\":1}\n");
    write_file(root / "used" / key, "");
}

}  // namespace qvac_coreml_fit_stable
