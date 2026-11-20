// The Supertonic vocoder's Core ML route: device class, template choice, the
// prebuilt sidecar's weight floor, and route and fit stability across the
// cache.
//
//   test-supertonic-coreml-route --self-test
//   test-supertonic-coreml-route --gguf TWIN.gguf --template ID [--gguf TWIN2.gguf --template ID2 ...]
//   test-supertonic-coreml-route --gguf MODEL.gguf --template ID --fit [--cache-dir DIR]
//
// --self-test: the device class of brand strings and the device preference
// (no model). With --gguf (twins of the templates' reference manifests, or a
// full Supertonic GGUF) each GGUF must choose template ID; a Max or Ultra brand
// string keeps a generated sidecar off unless use = require; an adjacent
// prebuilt sidecar wins unless the GGUF's weights are below its floor; no
// cache directory, generate = never and use = off give no route; the
// synthesis CLI's macOS cache default makes a bare GGUF plan generation; the
// template's memory charges are pinned. The route (and, with --fit,
// supertonic::fit_params' Core ML part) must then be the same on an empty
// cache, after a generation, and with non-stale and stale warm hints at the
// runtime's own hint ids; wherever the build can generate, that route is the
// generated one (with use = require on a device class that prefers ggml) and
// fit charges its envelope. --fit needs a whole Supertonic GGUF. With --cache-dir DIR (a cache a real run populated)
// DIR is compared as it is, then a copy of it gets the hint cases.

#include "supertonic_cli_coreml.h"
#include "supertonic_coreml_path.h"
#include "supertonic_vocoder_coreml.h"
#include "tts-cpp/supertonic/fit.h"

#include "../test/test_env_portable.h"
#include "fit_stable_cache.h"

#ifdef TTS_CPP_USE_COREML
#include "qvac_coreml/generation.h"
#include "qvac_coreml/manager.h"
#include "qvac_coreml/options.h"
namespace pcm = QVAC_COREML_NS;
#endif

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

using namespace tts_cpp::supertonic::detail;
namespace fs = std::filesystem;

namespace {

int g_failures = 0;

void check(bool ok, const std::string & what) {
    std::fprintf(stderr, "  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_failures;
}

// Whether this build can choose the generated route at all: a macOS Core ML
// build with the generator, on macOS 15 or later (the runtime's own checks).
bool generation_expected() {
#if defined(TTS_CPP_USE_COREML) && defined(__APPLE__) && TARGET_OS_OSX
    const auto m = pcm::Manager::instance();
    return pcm::generation::built() && m->platform() == pcm::Platform::macos && pcm::os_at_least(m->os_version(), 15, 0);
#else
    return false;
#endif
}

// Where the build cannot generate, the state and reason of the `none` route a
// twin plans on an M2 brand with a cache directory, per configuration: no
// Core ML runtime; the runtime without the generator (a prebuilt-only build:
// no adjacent sidecar, so absent); the generator on an OS it does not run on.
std::pair<std::string, std::string> no_generation_state() {
#if defined(TTS_CPP_USE_COREML)
    const auto m = pcm::Manager::instance();
    if (m->platform() == pcm::Platform::other) return {"unsupported", "platform"};
    if (!pcm::generation::built()) return {"absent", "no sidecar"};
    return {"unsupported", "os"};
#else
    return {"unsupported", "build"};
#endif
}

void device_class_gates() {
    std::fprintf(stderr, "device class\n");
    const std::map<std::string, std::string> classes = {
        {"Apple M1", "base"},           {"Apple M2", "base"},         {"Apple M4", "base"},
        {"Apple M10", "base"},          {"Apple M2 Pro", "pro"},      {"Apple M4 Max", "max"},
        {"Apple M3 Ultra", "ultra"},    {"Apple M4 Max (Virtual)", "max"}, {"Apple M2 (Virtual)", "base"},
        {"Apple M4 Pro (Virtual)", "pro"},
        {"", ""},                       {"Apple", ""},                {"Apple M", ""},
        {"Apple MX", ""},               {"Apple M2 Turbo", ""},       {"Intel(R) Core(TM) i9", ""},
        {"apple m2 max", ""},           {"Apple M2Max", ""},          {"Apple A17 Pro", ""},
    };
    for (const auto & [brand, cls] : classes) {
        const std::string got = supertonic_device_class(brand);
        check(got == cls, "'" + brand + "' -> '" + got + "' (expected '" + cls + "')");
    }
    const std::vector<std::string> prefer = {"max", "ultra"};
    check(supertonic_prefers_ggml(prefer, "Apple M3 Ultra") && supertonic_prefers_ggml(prefer, "Apple M4 Max") &&
              supertonic_prefers_ggml(prefer, "Apple M4 Max (Virtual)"),
          "Max and Ultra prefer ggml");
    check(!supertonic_prefers_ggml(prefer, "Apple M2") && !supertonic_prefers_ggml(prefer, "Apple M4 Pro") &&
              !supertonic_prefers_ggml(prefer, "") && !supertonic_prefers_ggml(prefer, "Intel(R) Xeon(R)") &&
              !supertonic_prefers_ggml({}, "Apple M3 Ultra"),
          "every other class, an unknown brand and an empty list do not");
    check(supertonic_merge_vocoder_backend("", "coreml-all") == "coreml-all" &&
              supertonic_merge_vocoder_backend("coreml-all", "coreml-all") == "coreml-all" &&
              supertonic_merge_vocoder_backend("coreml-all", "ggml") == "mixed",
          "backend label merge");
}

std::string describe(const supertonic_vocoder_route & r) {
    return r.route + "|" + r.state + "|" + r.reason + "|" + r.template_id + "|" + r.compute_units + "|" +
           std::to_string(r.envelope_bytes) + "|" + (r.validated ? "validated" : "unvalidated");
}

std::string describe(const tts_cpp::FitResult & f) {
    return f.reason + "|" + f.coreml.route + "|" + std::to_string(f.coreml.bytes) + "|" +
           (f.coreml.validated ? "validated" : "unvalidated") + "|" + f.coreml.disabled_reason;
}

tts_cpp::CoremlOptions at(const fs::path & cache) {
    tts_cpp::CoremlOptions o;
    o.cache_dir = cache.string();
    return o;
}

// The templates' memory charges under `all` (the default), before the
// margin. Each is 1.25 x the largest value measured over every overlapped
// generated-schedule run on the validated hosts (an M2 and the M4 Max CI
// runner), rounded up to 1 MiB, so the envelope's own margin is 1.0
// (template.json memory.measured). Both templates compile the same program
// with the same weights, so the load, resident and compiler-reserve charges
// come from both templates' runs; the prepare is each template's own. A change
// of a template's memory changes these on purpose.
constexpr uint64_t MIB = 1ull << 20;
constexpr uint64_t VOCODER_WEIGHTS = 50670336;                  // weight.bin
constexpr uint64_t VOCODER_IO = 2 * (2 * (144 * 64 + 196608));  // latent and wav (fp16), twice
constexpr uint64_t GENERATION_PEAK = 18 * MIB;                  // generation_peak_bytes
constexpr uint64_t VOCODER_LOAD = 62 * MIB;      // 1.25 x 49.4 MiB (an M4 Max run with the weights in the process)
constexpr uint64_t VOCODER_RESIDENT = 63 * MIB;  // 1.25 x 49.8 MiB of steady state, the same run
// 1.25 x 659.1 MiB, ANECompilerService's largest absolute size while loading
// on any validated host (the M4 Max)
constexpr uint64_t VOCODER_ANE_RESERVE = 824 * MIB;
const std::map<std::string, uint64_t> TEMPLATE_PREPARE = {
    {SUPERTONIC_VOCODER_TEMPLATE_W64, 34 * MIB},       // 1.25 x 27.0 MiB (an M4 Max run)
    {SUPERTONIC_VOCODER_TEMPLATE_PW2D_W64, 52 * MIB},  // 1.25 x 41.5 MiB (dequantising the pointwise weights)
};

void template_charges(const supertonic_vocoder_route & r) {
    auto it = TEMPLATE_PREPARE.find(r.template_id);
    if (it == TEMPLATE_PREPARE.end()) {
        check(false, "template charges: no pin for template '" + r.template_id + "'");
        return;
    }
    const uint64_t prepare = it->second;
    check(r.prepare_peak_bytes == prepare && r.load_peak_bytes == VOCODER_LOAD &&
              r.workspace_bytes == VOCODER_RESIDENT - VOCODER_WEIGHTS - VOCODER_IO &&
              r.resident_bytes == VOCODER_RESIDENT && r.ane_compile_reserve_bytes == VOCODER_ANE_RESERVE,
          r.template_id + " charges: prepare " + std::to_string(r.prepare_peak_bytes) + ", load " +
              std::to_string(r.load_peak_bytes) + ", workspace " + std::to_string(r.workspace_bytes) + ", resident " +
              std::to_string(r.resident_bytes) + ", compiler reserve " + std::to_string(r.ane_compile_reserve_bytes));
    check(r.prepare_peak_bytes >= GENERATION_PEAK + 8 * MIB, r.template_id + ": prepare covers generation + 8 MiB");
    const uint64_t envelope = VOCODER_RESIDENT + std::max(prepare, VOCODER_LOAD);
    check(r.envelope_bytes == envelope && r.validated,
          r.template_id + " envelope " + std::to_string(r.envelope_bytes) + " bytes (expected " +
              std::to_string(envelope) + ", margin 1.0), validated for all");
}

// Route rules on one GGUF that a template `want` matches.
void route_gates(const std::string & gguf, const std::string & want, const fs::path & scratch) {
    std::fprintf(stderr, "route gates (%s, template %s)\n", gguf.c_str(), want.c_str());
    const fs::path cache = scratch / "cache-route";
    fs::create_directories(cache);
    const supertonic_vocoder_route r = supertonic_vocoder_plan_route(gguf, at(cache), "Apple M2");
    if (!generation_expected()) {
        const auto [state, reason] = no_generation_state();
        check(r.route == "none" && r.state == state && r.reason == reason,
              "this build has no generated route (expected none|" + state + "|" + reason + "): " + describe(r));
        return;
    }
    check(r.route == "generate" && r.template_id == want && r.compute_units == "all",
          "Apple M2: " + describe(r));
    if (r.route == "generate") template_charges(r);

    for (const char * brand : {"Apple M3 Ultra", "Apple M4 Max", "Apple M4 Max (Virtual)", "Apple M2 Max"}) {
        const supertonic_vocoder_route g = supertonic_vocoder_plan_route(gguf, at(cache), brand);
        check(g.route == "none" && g.state == "unsupported" && g.reason == "device", std::string(brand) + ": " + describe(g));
    }
    for (const char * brand : {"Apple M4", "Apple M2 Pro", "Apple M4 Pro (Virtual)", "Intel(R) Xeon(R)"}) {
        check(supertonic_vocoder_plan_route(gguf, at(cache), brand).route == "generate",
              std::string(brand) + " generates");
    }
    tts_cpp::CoremlOptions req = at(cache);
    req.use = tts_cpp::CoremlOptions::Use::require;
    check(supertonic_vocoder_plan_route(gguf, req, "Apple M3 Ultra").route == "generate",
          "use = require generates on an Ultra");

    tts_cpp::CoremlOptions none;
    const supertonic_vocoder_route nc = supertonic_vocoder_plan_route(gguf, none, "Apple M2");
    check(nc.route == "none" && nc.reason == "no_cache_dir", "no cache directory: " + describe(nc));
    tts_cpp::CoremlOptions never = at(cache);
    never.generate = tts_cpp::CoremlOptions::Generate::never;
    check(supertonic_vocoder_plan_route(gguf, never, "Apple M2").state == "disabled", "generate = never");
    tts_cpp::CoremlOptions off = at(cache);
    off.use = tts_cpp::CoremlOptions::Use::off;
    check(supertonic_vocoder_plan_route(gguf, off, "Apple M2").state == "disabled", "use = off");
    tts_cpp::CoremlOptions bad = at(cache);
    bad.compute_units = "gpu";
    bool threw = false;
    try {
        supertonic_vocoder_plan_route(gguf, bad);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    check(threw, "invalid compute units throw");

    // An adjacent prebuilt sidecar wins over generation on every device
    // class, unless the GGUF's weights are below its floor.
    const fs::path dir = scratch / "prebuilt";
    fs::create_directories(dir);
    const fs::path linked = dir / fs::path(gguf).filename();
    std::error_code ec;
    fs::remove(linked, ec);
    fs::create_symlink(fs::absolute(gguf), linked, ec);
    fs::create_directories(coreml_vocoder_sidecar_path(linked.string()), ec);
    check(supertonic_vocoder_plan_route(linked.string(), at(cache), "Apple M3 Ultra").route == "prebuilt",
          "an adjacent prebuilt wins, on an Ultra too");
    check(supertonic_vocoder_plan_route(linked.string(), at(cache), "Apple M2", true).route == "generate",
          "a prebuilt below its weight floor does not count: generate");
    // The benchmark's development override plans generation beside the
    // exporter's sidecar, on every device class it would generate on.
    check(supertonic_vocoder_plan_route(linked.string(), at(cache), "Apple M2", false, true).route == "generate",
          "ignore_prebuilt: generate beside an adjacent prebuilt");
    check(supertonic_vocoder_plan_route(linked.string(), at(cache), "Apple M2").route == "prebuilt",
          "without ignore_prebuilt the adjacent prebuilt wins");
    fs::remove_all(dir, ec);
}

// The synthesis CLI's cache default with an isolated home and no adjacent
// sidecar: on macOS a bare GGUF gets Application Support's cache directory and
// plans the generated route (where the build can generate) instead of
// unsupported(no_cache_dir); an explicit directory and QVAC_COREML_CACHE_DIR
// win; elsewhere nothing is defaulted.
void cli_cache_default_gates(const std::string & gguf, const fs::path & scratch) {
    std::fprintf(stderr, "CLI cache default (%s)\n", gguf.c_str());
    const fs::path home = scratch / "home";
    fs::create_directories(home);
    // HOME as the CLI reads it (a narrow string, also on Windows).
    const std::string home_env = home.string();
    const std::string expected = home_env + CLI_COREML_CACHE_SUBDIR;
    tts_cpp::CoremlOptions o;
    apply_cli_coreml_cache_default(o, nullptr, home_env.c_str(), cli_on_macos());
    if (cli_on_macos()) {
        check(o.cache_dir && *o.cache_dir == expected, "macOS: Application Support default " + o.cache_dir.value_or("(unset)"));
    } else {
        check(!o.cache_dir, "not macOS: no CLI default");
    }
    const supertonic_vocoder_route r = supertonic_vocoder_plan_route(gguf, o, "Apple M2");
    if (generation_expected()) {
        check(r.route == "generate", "a bare GGUF with the CLI default generates: " + describe(r));
    } else if (cli_on_macos()) {
        const auto [state, reason] = no_generation_state();
        check(r.route == "none" && r.state == state && r.reason == reason,
              "a bare GGUF with the CLI default, no generation in this build: " + describe(r));
    }
    tts_cpp::CoremlOptions flag;
    flag.cache_dir = (scratch / "explicit").string();
    apply_cli_coreml_cache_default(flag, "/env/dir", home_env.c_str(), true);
    check(flag.cache_dir && *flag.cache_dir == (scratch / "explicit").string(), "an explicit directory wins");
    tts_cpp::CoremlOptions env;
    apply_cli_coreml_cache_default(env, "/env/dir", home_env.c_str(), true);
    check(!env.cache_dir, "QVAC_COREML_CACHE_DIR is left to the library");
    tts_cpp::CoremlOptions other;
    apply_cli_coreml_cache_default(other, nullptr, home_env.c_str(), false);
    apply_cli_coreml_cache_default(other, nullptr, "", true);
    check(!other.cache_dir, "no default off macOS or without a home");
    std::error_code ec;
    fs::remove_all(home, ec);
}

// Route and fit before a generation, after it, and with warm hints. On a
// device class the template prefers ggml on (this host's brand), the
// comparisons run with use = require, so they hold the generated route and
// its envelope there too instead of comparing two `none` routes.
void stability_gates(const std::string & gguf, const fs::path & scratch, bool fit, const std::string & given_cache) {
    std::fprintf(stderr, "route and fit stability (%s)\n", gguf.c_str());
    const fs::path empty = scratch / "empty";
    fs::create_directories(empty);
    const bool gated = supertonic_vocoder_plan_route(gguf, at(empty)).reason == "device";
    auto options_at = [&](const fs::path & cache) {
        tts_cpp::CoremlOptions o = at(cache);
        if (gated) o.use = tts_cpp::CoremlOptions::Use::require;
        return o;
    };
    if (gated) std::fprintf(stderr, "  this device class prefers ggml: the comparisons run with use = require\n");
    auto route_at = [&](const fs::path & cache) {
        return describe(supertonic_vocoder_plan_route(gguf, options_at(cache)));
    };
    auto fit_at = [&](const fs::path & cache) {
        tts_cpp::supertonic::FitOptions f;
        f.model_gguf_path = gguf;
        f.n_gpu_layers = 99;
        f.coreml = options_at(cache);
        return tts_cpp::supertonic::fit_params(f);
    };
    const supertonic_vocoder_route r0 = supertonic_vocoder_plan_route(gguf, options_at(empty));
    tts_cpp::FitResult f0;
    if (fit) f0 = fit_at(empty);
    if (generation_expected()) {
        check(r0.route == "generate" && r0.envelope_bytes > 0,
              "the comparisons hold a generated route: " + describe(r0));
        if (fit) {
            check(f0.coreml.route == "generate" && f0.coreml.bytes == r0.envelope_bytes,
                  "fit charges the generated route's envelope " + std::to_string(r0.envelope_bytes) + ": " +
                      describe(f0));
        }
    }
    const std::string route0 = describe(r0);
    const std::string fit0 = fit ? describe(f0) : std::string();
    std::fprintf(stderr, "  route on an empty cache: %s\n", route0.c_str());
    if (fit) std::fprintf(stderr, "  fit on an empty cache:   %s\n", fit0.c_str());
    auto same = [&](const fs::path & cache, const std::string & what) {
        const std::string r = route_at(cache);
        check(r == route0, what + ": route " + r);
        if (fit) {
            const std::string f = describe(fit_at(cache));
            check(f == fit0, what + ": fit " + f);
        }
    };
    fs::path cache;
    if (given_cache.empty()) {
        cache = scratch / "cache";
        qvac_coreml_fit_stable::simulate_generation(cache, r0.template_id.empty() ? "supertonic-vocoder-w64"
                                                                                  : r0.template_id);
        same(cache, "after a generation (artifact, memo)");
    } else {
        same(given_cache, "populated cache " + given_cache);
        cache = scratch / "copy";
        std::error_code cec;
        fs::copy(given_cache, cache, fs::copy_options::recursive | fs::copy_options::copy_symlinks, cec);
        check(!cec, "copied " + given_cache + (cec ? ": " + cec.message() : ""));
        same(cache, "a copy of the populated cache");
    }
#ifdef TTS_CPP_USE_COREML
    const size_t n = qvac_coreml_fit_stable::artifact_dirs(cache).size();
    int written = qvac_coreml_fit_stable::write_hints(cache, r0.compute_units, /*stale=*/false);
    check(n > 0 && written == (int) n, "non-stale hints at the runtime's id for " + std::to_string(written) + " of " +
                                           std::to_string(n) + " artifact(s)");
    same(cache, "with non-stale hints");
    written = qvac_coreml_fit_stable::write_hints(cache, r0.compute_units, /*stale=*/true);
    check(n > 0 && written == (int) n,
          "stale hints at the runtime's id for " + std::to_string(written) + " of " + std::to_string(n));
    same(cache, "with stale hints");
#else
    std::fprintf(stderr, "  built without TTS_CPP_COREML: hint cases skipped\n");
#endif
}

}  // namespace

int main(int argc, char ** argv) {
    std::vector<std::pair<std::string, std::string>> ggufs;  // (gguf, template)
    std::string given_cache;
    bool fit = false, self_test = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--self-test") self_test = true;
        else if (a == "--gguf") ggufs.emplace_back(next(), "");
        else if (a == "--template" && !ggufs.empty()) ggufs.back().second = next();
        else if (a == "--cache-dir") given_cache = next();
        else if (a == "--fit") fit = true;
        else {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    unsetenv("SUPERTONIC_COREML_DISABLE");
    unsetenv("SUPERTONIC_COREML_STRICT");
    unsetenv("SUPERTONIC_COREML_COMPUTE_UNITS");
    unsetenv("SUPERTONIC_COREML_ALLOW_LOW_BIT");
    unsetenv("QVAC_COREML_CACHE_DIR");

    device_class_gates();
    if (!self_test) {
        if (ggufs.empty()) {
            std::fprintf(stderr, "no --gguf\n");
            return 2;
        }
        const fs::path scratch = fs::temp_directory_path() /
                                 ("supertonic-coreml-route-" +
                                  std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        for (const auto & [gguf, tmpl] : ggufs) {
            if (!std::ifstream(gguf).good()) {
                check(false, "no GGUF at " + gguf);
                continue;
            }
            route_gates(gguf, tmpl, scratch);
            cli_cache_default_gates(gguf, scratch);
            stability_gates(gguf, scratch, fit, given_cache);
        }
        std::error_code ec;
        fs::remove_all(scratch, ec);
    }
    std::fprintf(stderr, g_failures ? "FAILED (%d)\n" : "OK\n", g_failures);
    return g_failures ? 1 : 0;
}
