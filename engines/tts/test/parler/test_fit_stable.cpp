// The Parler DAC's Core ML route and fit envelope do not depend on the cache.
//
//   test-parler-fit-stable --gguf MODEL.gguf [--fit] [--cache-dir DIR]
//
// The route (prebuilt | generate | none) is decided from the options, OS,
// build, template match and a prebuilt sidecar's presence, never from the
// cache, so fit is the same before and after the first generation. This
// checks parler_dac_plan_route (and, with --fit, parler::fit_params' Core ML
// part) against an empty cache directory, then after the cache holds what a
// generation leaves behind: an artifact with its memo and used marker, then
// a non-stale warm hint, then a stale one. --cache-dir checks a cache a real
// run populated instead of a scratch one (left untouched). --gguf may be a
// twin of the template's reference manifest (only the route is checked) or a
// full Parler GGUF (--fit).

#include "tts-cpp/parler/fit.h"
#include "parler/internal.h"
#include "../test_env_portable.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

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
    const fs::path cache = given_cache.empty() ? scratch / "cache" : fs::path(given_cache);
    fs::create_directories(cache);
    tts_cpp::CoremlOptions opts;
    opts.cache_dir = cache.string();

    auto route_now = [&] { return describe(parler_dac_plan_route(gguf, opts, 0)); };
    auto fit_now = [&] {
        tts_cpp::parler::FitOptions f;
        f.model_gguf_path = gguf;
        f.coreml = opts;
        return describe(tts_cpp::parler::fit_params(f));
    };

    const std::string route0 = route_now();
    const std::string fit0 = fit ? fit_now() : std::string();
    fprintf(stderr, "route: %s\n", route0.c_str());
    if (fit) fprintf(stderr, "fit:   %s\n", fit0.c_str());

    auto same = [&](const std::string & what) {
        const std::string r = route_now();
        check(r == route0, what + ": route " + r);
        if (fit) {
            const std::string f = fit_now();
            check(f == fit0, what + ": fit " + f);
        }
    };

    if (given_cache.empty()) {
        // What a generation leaves: the artifact, its memo, the used marker.
        const std::string key = "0123456789abcdef0123456789abcdef";
        const fs::path root = cache / "qvac-coreml-v1";
        const fs::path art = root / "artifacts" / ("parler-dac-w128." + key);
        write_file(art / "model.mlmodelc" / "model.mil", "program(1.0)\n");
        write_file(art / "artifact.json",
                   "{\"schema\":1,\"complete\":true,\"artifact_key\":\"" + key +
                       "\",\"inventory\":[{\"path\":\"model.mlmodelc/model.mil\",\"bytes\":13}]}\n");
        write_file(root / "sources" / ("fedcba9876543210fedcba9876543210." + key + ".json"), "{\"schema\":1}\n");
        write_file(root / "used" / key, "");
        same("after a generation (artifact, memo)");
        // A non-stale warm hint, then a stale one.
        const fs::path hint = root / "hints" / "00112233445566778899aabbccddeeff.json";
        write_file(hint, "{\"schema\":1,\"warm_load_ms\":120,\"cold_load_ms\":8000,\"last_ok\":"
                         "\"2026-11-13T00:00:00Z\",\"stale_until\":null}\n");
        same("with a non-stale hint");
        write_file(hint, "{\"schema\":1,\"warm_load_ms\":120,\"cold_load_ms\":8000,\"last_ok\":"
                         "\"2026-11-13T00:00:00Z\",\"stale_until\":\"2099-01-01T00:00:00Z\"}\n");
        same("with a stale hint");
    } else {
        // A populated cache against the same options on an empty one.
        const fs::path empty = scratch / "empty";
        fs::create_directories(empty);
        tts_cpp::CoremlOptions e = opts;
        e.cache_dir = empty.string();
        const std::string r = describe(parler_dac_plan_route(gguf, e, 0));
        check(r == route0, "populated cache " + given_cache + " vs an empty one: " + r);
    }

    std::error_code ec;
    fs::remove_all(scratch, ec);
    fprintf(stderr, g_failures ? "FAILED (%d)\n" : "OK\n", g_failures);
    return g_failures ? 1 : 0;
}
