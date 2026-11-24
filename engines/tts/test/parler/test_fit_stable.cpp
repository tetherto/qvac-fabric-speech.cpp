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
#include "fit_stable_cache.h"

#ifdef TTS_CPP_USE_COREML
#include "hints.h"
#include "qvac_coreml/generation.h"
#include "qvac_coreml/manager.h"
#endif
#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

#include <chrono>
#include <cstdint>
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

// Cache fixtures shared with the other engines' fit-stable tests.
using qvac_coreml_fit_stable::artifact_dirs;
using qvac_coreml_fit_stable::write_file;
#ifdef TTS_CPP_USE_COREML
namespace pcm = QVAC_COREML_NS;
using qvac_coreml_fit_stable::runtime_hint_path;
using qvac_coreml_fit_stable::write_hints;
#endif

// The parler-dac-w128 template's memory charges under `all` (the default),
// before the margin, and the envelope fit charges without the host quantizer
// copy. Each charge is 1.25 x the largest value measured over every
// overlapped generated-schedule run on the validated hosts, rounded up to
// 1 MiB, so the envelope's own margin is 1.0 (template.json memory.measured).
// The M4 Max CI maxima were 443.0 MiB for prepare, 328.3 MiB for load and
// warm-up, 328.5 MiB for the resident part and 1116.4 MiB for the compiler's
// absolute size during loading. The compiler reserve gates admission
// separately from the fit envelope.
constexpr uint64_t MIB = 1ull << 20;
constexpr uint64_t DAC_WEIGHTS = 108316032;                    // weight.bin
constexpr uint64_t DAC_IO = 2 * 393472;                        // latent, valid and pcm, twice
constexpr uint64_t DAC_PREPARE = 554 * MIB;                    // generation_peak 216 + 8 + compile_peak 330
constexpr uint64_t DAC_LOAD = 411 * MIB;                       // load_peak_bytes.all
constexpr uint64_t DAC_RESIDENT = 411 * MIB;                   // runtime weights + 2 x I/O + workspace
constexpr uint64_t DAC_WORKSPACE = DAC_RESIDENT - DAC_WEIGHTS - DAC_IO;
constexpr uint64_t DAC_ANE_RESERVE = 1396 * MIB;               // ane_compile_reserve_bytes
constexpr uint64_t DAC_ENVELOPE = DAC_RESIDENT + DAC_PREPARE;  // margin 1.0: 965 MiB

void template_charges(const parler_dac_route & r) {
#if defined(TTS_CPP_USE_COREML) && defined(__APPLE__) && TARGET_OS_OSX
    // A prebuilt-only Core ML build (TTS_CPP_COREML_GENERATE off) or macOS
    // before 15 generates nothing: the route is none for that reason and the
    // DAC stays on ggml, so there is no charge to check.
    const bool generator = pcm::generation::built();
    const bool os_supported = pcm::os_at_least(pcm::Manager::instance()->os_version(), 15, 0);
    if (!generator || !os_supported) {
        const bool fallback = r.route == "none" && (!generator ? r.state == "absent" && r.reason == "no sidecar"
                                                               : r.state == "unsupported" && r.reason == "os");
        check(fallback, std::string("template charges: ") + (!generator ? "no generator in this build" : "macOS before 15") +
                            ", so route none (" + r.route + " " + r.state + ": " + r.reason + ")");
        return;
    }
#endif
    if (r.route != "generate" || r.compute_units != "all") {
#if defined(TTS_CPP_USE_COREML) && defined(__APPLE__) && TARGET_OS_OSX
        // A macOS 15+ build with the generator generates this GGUF's DAC under
        // `all`, so a charge it cannot check is a failure, not a skip.
        check(false, "template charges: route " + r.route +
                         (r.reason.empty() ? std::string() : " (" + r.state + ": " + r.reason + ")") + ", units " +
                         r.compute_units + "; expected route generate under all");
#else
        fprintf(stderr, "template charges: not checked (route %s, units %s; no generated route in this build)\n",
                r.route.c_str(), r.compute_units.c_str());
#endif
        return;
    }
    check(r.prepare_peak_bytes == DAC_PREPARE && r.load_peak_bytes == DAC_LOAD && r.workspace_bytes == DAC_WORKSPACE &&
              r.resident_bytes == DAC_RESIDENT && r.ane_compile_reserve_bytes == DAC_ANE_RESERVE,
          "template charges: prepare " + std::to_string(r.prepare_peak_bytes) + ", load " +
              std::to_string(r.load_peak_bytes) + ", workspace " + std::to_string(r.workspace_bytes) + ", resident " +
              std::to_string(r.resident_bytes) + ", compiler reserve " + std::to_string(r.ane_compile_reserve_bytes));
    check(r.envelope_bytes == DAC_ENVELOPE && r.validated,
          "template envelope " + std::to_string(r.envelope_bytes) + " bytes (expected " + std::to_string(DAC_ENVELOPE) +
              ", margin 1.0), validated for all");
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

    template_charges(parler_dac_plan_route(gguf, options_at(empty), 0));
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
        qvac_coreml_fit_stable::simulate_generation(cache, "parler-dac-w128");
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
