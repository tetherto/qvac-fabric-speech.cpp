// The Parler DAC's Core ML route and fit envelope do not depend on the cache.
//
//   test-parler-fit-stable --gguf MODEL.gguf [--fit] [--cache-dir DIR]
//
// The route (prebuilt | generate | none) is decided from the options, OS,
// build, template match and a prebuilt sidecar's presence, never from the
// cache, so fit is the same before and after the first generation. This
// checks parler_dac_plan_route (and, with --fit, parler::fit_params' Core ML
// part) on an empty cache directory against a cache holding what a
// generation leaves behind, then the same cache with a non-stale warm hint
// and with a stale one. The hints are written where the runtime looks for
// them (the hint id of each artifact's model, its compute units, this host
// and OS build), so a route or fit that read them would see them.
//
// Without --cache-dir the generation is simulated in a scratch cache: an
// artifact with its memo and used marker. With --cache-dir DIR (a cache a
// real run populated) DIR itself is compared as it is and left untouched; it
// is then copied to scratch and the hint cases run on the copy, with every
// existing hint rewritten non-stale and then stale, plus a hint at the
// runtime's id for every artifact of the copy. If DIR holds hints, at least
// one must sit at the id this test computes for one of its artifacts and a
// program that populates caches (the runtime's host id is the program name),
// which checks the computed id against what the runtime wrote.
//
// --gguf may be a twin of the template's reference manifest (only the route
// is checked) or a full Parler GGUF (--fit).

#include "tts-cpp/parler/fit.h"
#include "parler/internal.h"
#include "../test_env_portable.h"

#ifdef TTS_CPP_USE_COREML
#include "hints.h"
#include "qvac_coreml/manager.h"
#endif

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

using namespace tts_cpp::parler::detail;
namespace fs = std::filesystem;

namespace {

int g_failures = 0;

void check(bool ok, const std::string & what) {
    fprintf(stderr, "  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_failures;
}

std::string describe(const parler_dac_route & r) {
    return r.route + "|" + r.state + "|" + r.reason + "|" + r.compute_units + "|" + std::to_string(r.envelope_bytes) +
           "|" + (r.validated ? "validated" : "unvalidated") + "|" + (r.strict ? "strict" : "");
}

std::string describe(const tts_cpp::FitResult & f) {
    return f.reason + "|" + f.coreml.route + "|" + std::to_string(f.coreml.bytes) + "|" +
           (f.coreml.validated ? "validated" : "unvalidated") + "|" + f.coreml.disabled_reason;
}

void write_file(const fs::path & p, const std::string & text) {
    fs::create_directories(p.parent_path());
    std::ofstream(p) << text;
}

std::vector<fs::path> artifact_dirs(const fs::path & cache) {
    std::vector<fs::path> out;
    std::error_code ec;
    for (const auto & e : fs::directory_iterator(cache / "qvac-coreml-v1" / "artifacts", ec)) {
        if (e.is_directory()) out.push_back(e.path());
    }
    return out;
}

#ifdef TTS_CPP_USE_COREML
namespace pcm = QVAC_COREML_NS;

// The hint record the runtime reads for the model of `artifact_dir`, in this
// process or (`host_id`) in another program: the runtime's host id is the
// program's name.
std::string runtime_hint_path(const fs::path & cache, const fs::path & artifact_dir, const std::string & units,
                              const std::string & host_id = {}) {
    const auto m = pcm::Manager::instance();
    return pcm::detail::hint_path_for(cache.string(), host_id.empty() ? m->host_id() : host_id,
                                      (artifact_dir / "model.mlmodelc").string(), "main", units, m->os_build());
}

std::optional<int64_t> stale_until(bool stale) {
    if (!stale) return std::nullopt;
    return pcm::detail::unix_now_s() + 365LL * 24 * 3600;
}

// Rewrites every hint record under `cache` with `stale` and writes one at the
// runtime's id for every artifact; returns how many read back at a runtime id.
int write_hints(const fs::path & cache, const std::string & units, bool stale) {
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

} // namespace

int main(int argc, char ** argv) {
    std::string gguf, given_cache;
    bool fit = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--gguf" && i + 1 < argc) gguf = argv[++i];
        else if (a == "--cache-dir" && i + 1 < argc) given_cache = argv[++i];
        else if (a == "--fit") fit = true;
    }
    if (gguf.empty() || !std::ifstream(gguf).good()) {
        fprintf(stderr, "SKIP: no --gguf model (%s)\n", gguf.c_str());
        return 77;
    }
    unsetenv("PARLER_COREML_DISABLE");
    unsetenv("PARLER_COREML_STRICT");
    unsetenv("QVAC_COREML_CACHE_DIR");

    const fs::path scratch = fs::temp_directory_path() / ("parler-fit-stable-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const fs::path empty = scratch / "empty";
    fs::create_directories(empty);

    auto options_at = [](const fs::path & cache) {
        tts_cpp::CoremlOptions o;
        o.cache_dir = cache.string();
        return o;
    };
    auto route_at = [&](const fs::path & cache) {
        return describe(parler_dac_plan_route(gguf, options_at(cache), 0));
    };
    auto fit_at = [&](const fs::path & cache) {
        tts_cpp::parler::FitOptions f;
        f.model_gguf_path = gguf;
        f.coreml = options_at(cache);
        return describe(tts_cpp::parler::fit_params(f));
    };

    const std::string route0 = route_at(empty);
    const std::string fit0 = fit ? fit_at(empty) : std::string();
    fprintf(stderr, "route on an empty cache: %s\n", route0.c_str());
    if (fit) fprintf(stderr, "fit on an empty cache:   %s\n", fit0.c_str());

    auto same = [&](const fs::path & cache, const std::string & what) {
        const std::string r = route_at(cache);
        check(r == route0, what + ": route " + r);
        if (fit) {
            const std::string f = fit_at(cache);
            check(f == fit0, what + ": fit " + f);
        }
    };

#ifdef TTS_CPP_USE_COREML
    const std::string units = parler_dac_plan_route(gguf, options_at(empty), 0).compute_units;
#endif
    fs::path cache;
    if (given_cache.empty()) {
        // What a generation leaves: the artifact, its memo, the used marker.
        cache = scratch / "cache";
        const std::string key = "0123456789abcdef0123456789abcdef";
        const fs::path root = cache / "qvac-coreml-v1";
        const fs::path art = root / "artifacts" / ("parler-dac-w128." + key);
        write_file(art / "model.mlmodelc" / "model.mil", "program(1.0)\n");
        write_file(art / "artifact.json",
                   "{\"schema\":1,\"complete\":true,\"artifact_key\":\"" + key +
                       "\",\"inventory\":[{\"path\":\"model.mlmodelc/model.mil\",\"bytes\":13}]}\n");
        write_file(root / "sources" / ("fedcba9876543210fedcba9876543210." + key + ".json"), "{\"schema\":1}\n");
        write_file(root / "used" / key, "");
        same(cache, "after a generation (artifact, memo)");
    } else {
        // The populated cache as it is, then a copy for the hint cases.
        same(given_cache, "populated cache " + given_cache);
#ifdef TTS_CPP_USE_COREML
        std::error_code ec;
        int hints = 0, found = 0;
        for (const auto & e : fs::directory_iterator(fs::path(given_cache) / "qvac-coreml-v1" / "hints", ec)) {
            if (e.path().extension() == ".json") ++hints;
        }
        // The programs that populate a cache in CI and locally.
        for (const fs::path & art : artifact_dirs(given_cache)) {
            for (const char * host : {"", "test-parler-dac-coreml", "parler-bench", "parler-cli", "tts-cli"}) {
                if (fs::exists(runtime_hint_path(given_cache, art, units, host))) ++found;
            }
        }
        check(hints == 0 || found > 0, "the runtime's hint id: " + std::to_string(found) + " of " +
                                           std::to_string(hints) + " hint(s) in " + given_cache +
                                           " sit at the id computed for an artifact and a known program");
#endif
        cache = scratch / "copy";
        std::error_code cec;
        fs::copy(given_cache, cache, fs::copy_options::recursive | fs::copy_options::copy_symlinks, cec);
        check(!cec, "copied " + given_cache + " to " + cache.string() + (cec ? ": " + cec.message() : ""));
        same(cache, "a copy of the populated cache");
    }

#ifdef TTS_CPP_USE_COREML
    const size_t n_artifacts = artifact_dirs(cache).size();
    int written = write_hints(cache, units, /*stale=*/false);
    check(n_artifacts > 0 && written == (int) n_artifacts,
          "non-stale hints at the runtime's id for " + std::to_string(written) + " of " +
              std::to_string(n_artifacts) + " artifact(s)");
    same(cache, "with non-stale hints");
    written = write_hints(cache, units, /*stale=*/true);
    check(n_artifacts > 0 && written == (int) n_artifacts,
          "stale hints at the runtime's id for " + std::to_string(written) + " of " + std::to_string(n_artifacts) +
              " artifact(s)");
    same(cache, "with stale hints");
#else
    fprintf(stderr, "built without TTS_CPP_COREML: hint cases skipped\n");
#endif

    std::error_code ec;
    fs::remove_all(scratch, ec);
    fprintf(stderr, g_failures ? "FAILED (%d)\n" : "OK\n", g_failures);
    return g_failures ? 1 : 0;
}
