// The CosyVoice3 front-end, F0 and HiFT Core ML routes and their fit envelope
// do not depend on the cache.
//
//   test-cosyvoice-fit-stable --flow FLOW.gguf --hift HIFT.gguf
//                             [--model-dir DIR --fit] [--cache-dir DIR]
//
// The routes (prebuilt | generate | none) are decided from the options, OS,
// build, template match and a prebuilt sidecar's presence, never from the
// cache, so fit is the same before and after the first generation. This
// checks cosyvoice_plan_coreml (and, with --fit, cosyvoice::fit_params on the
// model directory) on an empty cache directory against a cache holding what a
// generation of the three stages leaves behind, then the same cache with
// non-stale warm hints and with stale ones, written where the runtime looks
// for them (fit_stable_cache.h).
//
// With --cache-dir DIR (a cache a real run populated) DIR itself is compared
// as it is and left untouched; it is then copied to scratch and the hint
// cases run on the copy.
//
// --flow and --hift may be twins of the templates' reference manifests (only
// the routes are checked) or the real GGUFs.

#include "cosyvoice_coreml_stages.h"
#include "tts-cpp/cosyvoice/fit.h"
#include "test_env_portable.h"
#include "fit_stable_cache.h"

#ifdef TTS_CPP_USE_COREML
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
#include <string>
#include <system_error>

namespace fs = std::filesystem;
namespace fsc = qvac_coreml_fit_stable;

namespace {

int g_failures = 0;

void check(bool ok, const std::string & what) {
    fprintf(stderr, "  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_failures;
}

std::string describe(const cosyvoice_coreml_plan & p) {
    std::string s = std::to_string(p.envelope_bytes) + "|" + (p.validated ? "validated" : "unvalidated");
    for (const cosyvoice_stage_route & r : p.stages) {
        s += "|" + r.route + ":" + r.state + ":" + r.reason + ":" + r.compute_units + ":" +
             std::to_string(r.resident_bytes) + ":" + std::to_string(r.prepare_peak_bytes) + ":" +
             std::to_string(r.load_peak_bytes) + ":" + std::to_string(r.ane_compile_reserve_bytes);
    }
    return s;
}

std::string describe(const tts_cpp::FitResult & f) {
    return f.reason + "|" + f.coreml.route + "|" + std::to_string(f.coreml.bytes) + "|" +
           (f.coreml.validated ? "validated" : "unvalidated") + "|" + f.coreml.disabled_reason;
}

constexpr const char * TEMPLATE_IDS[COSYVOICE_STAGE_COUNT] = {
    "cosyvoice3-frontend-f1024", "cosyvoice3-hift-f0-m512", "cosyvoice3-hift-m512"};

// The templates' charges under cpu_and_ane (the default) and the envelope fit
// charges. Each constant is 1.25 x the largest value measured over every
// overlapped generated-schedule run on the validated hosts, rounded up to
// 1 MiB, so the envelope's own margin is 1.0 (template.json memory.measured):
//   resident  weights + 2 x I/O, and for HiFT the workspace that makes the
//             three cover 1.25 x the steady state over the warm ggml baseline
//             (203.8 MiB, M2);
//   prepare   generation_peak_bytes + 8 MiB + compile_peak_bytes (largest
//             prepares 11.5, 7.8 and 45.8 MiB, M2);
//   load      load_peak_bytes.cpu_and_ane (2.1 and 6.8 MiB on M2, 124.5 MiB on M4 Max);
//   reserve   ane_compile_reserve_bytes, ANECompilerService's absolute size
//             while the stage loads (773.6, 546.5 and 546.5 MiB, M4 Max).
// The envelope is the resident parts summed plus the largest one-off charge
// (HiFT's load). A change of a template's memory changes these on purpose.
struct stage_charges {
    uint64_t resident, prepare, load, reserve;
};
[[maybe_unused]] constexpr stage_charges CHARGES[COSYVOICE_STAGE_COUNT] = {
    {2896032, 15728640, 3145728, 968 * (1ull << 20)},
    {7287552, 13107200, 9437184, 684 * (1ull << 20)},
    {257203296, 60817408, 163577856, 684 * (1ull << 20)},
};
[[maybe_unused]] constexpr uint64_t ENVELOPE = 2896032 + 7287552 + 257203296 + 163577856;  // 411 MiB

void template_charges(const cosyvoice_coreml_plan & p) {
#if defined(TTS_CPP_USE_COREML) && defined(__APPLE__) && TARGET_OS_OSX
    namespace pcm = QVAC_COREML_NS;
    const bool generator = pcm::generation::built();
    const bool os_supported = pcm::os_at_least(pcm::Manager::instance()->os_version(), 15, 0);
    if (!generator || !os_supported) {
        bool fallback = true;
        for (const cosyvoice_stage_route & r : p.stages) {
            fallback = fallback && r.route == "none" &&
                       (!generator ? r.state == "absent" && r.reason == "no sidecar"
                                   : r.state == "unsupported" && r.reason == "os");
        }
        check(fallback, std::string("template charges: ") + (!generator ? "no generator in this build" : "macOS before 15") +
                            ", so every route is none");
        return;
    }
    bool generate = true;
    for (const cosyvoice_stage_route & r : p.stages) generate = generate && r.route == "generate" && r.compute_units == "cpu_and_ane";
    // A macOS 15+ build with the generator generates every stage under
    // cpu_and_ane, so a charge it cannot check is a failure, not a skip.
    check(generate, "template charges: every route generate under cpu_and_ane (" + describe(p) + ")");
    if (!generate) return;
    check(p.validated, "every generated stage validated for cpu_and_ane");
    for (int i = 0; i < COSYVOICE_STAGE_COUNT; ++i) {
        const cosyvoice_stage_route & r = p.stages[i];
        const stage_charges & c = CHARGES[i];
        check(r.margin == 1.0 && r.resident_bytes == c.resident && r.prepare_peak_bytes == c.prepare &&
                  r.load_peak_bytes == c.load && r.ane_compile_reserve_bytes == c.reserve,
              std::string("template charges of ") + TEMPLATE_IDS[i] + ": resident " + std::to_string(r.resident_bytes) +
                  ", prepare " + std::to_string(r.prepare_peak_bytes) + ", load " + std::to_string(r.load_peak_bytes) +
                  ", compiler reserve " + std::to_string(r.ane_compile_reserve_bytes) + ", margin " +
                  std::to_string(r.margin));
    }
    check(p.envelope_bytes == ENVELOPE,
          "template envelope " + std::to_string(p.envelope_bytes) + " bytes (expected " + std::to_string(ENVELOPE) + ")");
#else
    fprintf(stderr, "template charges: not checked (no generated route in this build): %s\n", describe(p).c_str());
#endif
}

}  // namespace

int main(int argc, char ** argv) {
    std::string flow, hift, model_dir, given_cache;
    bool fit = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--flow" && i + 1 < argc) flow = argv[++i];
        else if (a == "--hift" && i + 1 < argc) hift = argv[++i];
        else if (a == "--model-dir" && i + 1 < argc) model_dir = argv[++i];
        else if (a == "--cache-dir" && i + 1 < argc) given_cache = argv[++i];
        else if (a == "--fit") fit = true;
    }
    if (!model_dir.empty()) {
        if (flow.empty()) flow = model_dir + "/cosyvoice3-flow-f32.gguf";
        if (hift.empty()) hift = model_dir + "/cosyvoice3-hift-f32.gguf";
    }
    if (flow.empty() || hift.empty() || !std::ifstream(flow).good() || !std::ifstream(hift).good()) {
        fprintf(stderr, "SKIP: no --flow / --hift GGUF (%s, %s)\n", flow.c_str(), hift.c_str());
        return 77;
    }
    if (fit && model_dir.empty()) {
        fprintf(stderr, "--fit needs --model-dir\n");
        return 2;
    }
    for (const char * v : {"COSYVOICE_COREML_DISABLE", "COSYVOICE_COREML_STRICT", "COSYVOICE_COREML_COMPUTE_UNITS",
                           "QVAC_COREML_CACHE_DIR"}) {
        unsetenv(v);
    }

    const fs::path scratch = fs::temp_directory_path() /
        ("cosyvoice-fit-stable-" + test_process_tag() + "-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const fs::path empty = scratch / "empty";
    fs::create_directories(empty);

    auto options_at = [](const fs::path & cache) {
        tts_cpp::CoremlOptions o;
        o.cache_dir = cache.string();
        return o;
    };
    auto route_at = [&](const fs::path & cache) { return describe(cosyvoice_plan_coreml(flow, hift, options_at(cache))); };
    auto fit_at = [&](const fs::path & cache) {
        tts_cpp::cosyvoice::FitOptions f;
        f.llm_gguf_path = model_dir + "/cosyvoice3-llm-q8_0.gguf";
        f.flow_gguf_path = flow;
        f.hift_gguf_path = hift;
        f.voice_gguf_path = model_dir + "/voice.gguf";
        f.coreml = options_at(cache);
        return describe(tts_cpp::cosyvoice::fit_params(f));
    };

    template_charges(cosyvoice_plan_coreml(flow, hift, options_at(empty)));
    const std::string route0 = route_at(empty);
    const std::string fit0 = fit ? fit_at(empty) : std::string();
    fprintf(stderr, "routes on an empty cache: %s\n", route0.c_str());
    if (fit) fprintf(stderr, "fit on an empty cache:    %s\n", fit0.c_str());

    auto same = [&](const fs::path & cache, const std::string & what) {
        check(route_at(cache) == route0, what + ": routes " + route_at(cache));
        if (fit) {
            const std::string f = fit_at(cache);
            check(f == fit0, what + ": fit " + f);
        }
    };

    fs::path cache;
    if (given_cache.empty()) {
        // What a generation of the three stages leaves: artifacts, memos, used markers.
        cache = scratch / "cache";
        const char * keys[COSYVOICE_STAGE_COUNT] = {"0123456789abcdef0123456789abcde0", "0123456789abcdef0123456789abcde1",
                                                    "0123456789abcdef0123456789abcde2"};
        for (int i = 0; i < COSYVOICE_STAGE_COUNT; ++i) fsc::simulate_generation(cache, TEMPLATE_IDS[i], keys[i]);
        same(cache, "after a generation (artifacts, memos)");
    } else {
        same(given_cache, "populated cache " + given_cache);
        cache = scratch / "copy";
        std::error_code cec;
        fs::copy(given_cache, cache, fs::copy_options::recursive | fs::copy_options::copy_symlinks, cec);
        check(!cec, "copied " + given_cache + " to " + cache.string() + (cec ? ": " + cec.message() : ""));
        same(cache, "a copy of the populated cache");
    }

#ifdef TTS_CPP_USE_COREML
    const std::string units = cosyvoice_plan_coreml(flow, hift, options_at(empty)).stages[0].compute_units;
    const size_t n_artifacts = fsc::artifact_dirs(cache).size();
    int written = fsc::write_hints(cache, units, /*stale=*/false);
    check(n_artifacts > 0 && written == (int) n_artifacts,
          "non-stale hints at the runtime's id for " + std::to_string(written) + " of " + std::to_string(n_artifacts) +
              " artifact(s)");
    same(cache, "with non-stale hints");
    written = fsc::write_hints(cache, units, /*stale=*/true);
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
