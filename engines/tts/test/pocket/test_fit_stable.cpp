// Pocket's Core ML routes and fit envelope do not depend on the cache.
//
//   test-pocket-fit-stable --step-gguf FLOW_LM.gguf --mimi-gguf MIMI.gguf
//                          [--cache-dir DIR]
//   test-pocket-fit-stable --bundle DIR [--cache-dir DIR]
//
// Each stage's route (prebuilt | generate | none) is decided from the
// options, OS, build, template match and a prebuilt sidecar's presence, never
// from the cache, so fit is the same before and after the first generation.
// This checks coreml_plan_route for the FlowLM step and the Mimi decoder and
// coreml_envelope_bytes (and, with --bundle, pocket::fit_params) on an empty
// cache directory against a cache holding what a generation leaves behind,
// then the same cache with a non-stale warm hint and with a stale one. The
// hints are written where the runtime looks for them (the hint id of each
// artifact's model, its stage's compute units, this host and OS build), so a
// route or fit that read them would see them.
//
// Without --cache-dir the generation is simulated in a scratch cache: one
// artifact per template with its memo and used marker. With --cache-dir DIR
// (a cache a real run populated) DIR itself is compared as it is and left
// untouched; it is then copied to scratch and the hint cases run on the
// copy, with every existing hint rewritten non-stale and then stale, plus a
// hint at the runtime's id for every artifact of the copy. If DIR holds
// hints, at least one must sit at the id this test computes for one of its
// artifacts and a program that populates caches (the runtime's host id is
// the program name), which checks the computed id against what the runtime
// wrote.
//
// --step-gguf and --mimi-gguf may be twins of the templates' reference
// manifests (routes and envelope only); --bundle is a complete Pocket bundle
// (flow-lm.gguf, mimi.gguf, frontend.json, voice.gguf), which also checks
// the whole fit result.

#include "pocket/coreml.h"
#include "tts-cpp/pocket/fit.h"

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

namespace pd = tts_cpp::pocket::detail;
namespace fs = std::filesystem;

namespace {

int g_failures = 0;

void check(bool ok, const std::string & what) {
    fprintf(stderr, "  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_failures;
}

constexpr const char * STEP_TEMPLATE = "pocket-step-c1024";
constexpr const char * DECODER_TEMPLATE = "pocket-mimi";

std::string describe(const pd::CoremlRoute & r) {
    return r.route + "|" + r.state + "|" + r.reason + "|" + r.compute_units + "|" +
           std::to_string(r.resident_bytes) + "|" + std::to_string(r.transient_bytes) + "|" +
           std::to_string(r.margin) + "|" + std::to_string(r.ane_compile_reserve_bytes) + "|" +
           (r.validated ? "validated" : "unvalidated") + "|" + (r.strict ? "strict" : "");
}

std::string describe(const tts_cpp::FitResult & f) {
    return f.reason + "|" + f.coreml.route + "|" + std::to_string(f.coreml.bytes) + "|" +
           (f.coreml.validated ? "validated" : "unvalidated") + "|" + f.coreml.disabled_reason + "|" +
           std::to_string(f.device.total_bytes) + "|" + std::to_string(f.host_bytes);
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

bool starts_with(const std::string & s, const std::string & prefix) {
    return s.compare(0, prefix.size(), prefix) == 0;
}

#ifdef TTS_CPP_USE_COREML
namespace pcm = QVAC_COREML_NS;

// The compute units the runtime loads an artifact with: its stage's, from
// the template id the artifact directory starts with.
std::string units_of(const fs::path & artifact_dir, const std::string & step_units, const std::string & decoder_units) {
    return starts_with(artifact_dir.filename().string(), std::string(STEP_TEMPLATE) + ".") ? step_units
                                                                                             : decoder_units;
}

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
int write_hints(const fs::path & cache, const std::string & step_units, const std::string & decoder_units,
                bool stale) {
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
        const std::string path = runtime_hint_path(cache, art, units_of(art, step_units, decoder_units));
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

// The envelope fit charges for both generated stages, from the two
// templates' memory (each charge is 1.25 x the largest value measured over
// every generated-schedule run on the validated hosts, so the margin is
// 1.0): the step's and the decoder's resident parts summed plus the larger
// transient. A change of either template's memory changes this on purpose.
constexpr uint64_t POCKET_ENVELOPE = 349023360;

void template_charges(const pd::CoremlRoute & step, const pd::CoremlRoute & decoder) {
#if defined(TTS_CPP_USE_COREML) && defined(__APPLE__) && TARGET_OS_OSX
    // A prebuilt-only Core ML build (TTS_CPP_COREML_GENERATE off) or macOS
    // before 15 generates nothing: both routes are none for that reason and
    // both stages stay on ggml, so there is no charge to check.
    const bool generator = pcm::generation::built();
    const bool os_supported = pcm::os_at_least(pcm::Manager::instance()->os_version(), 15, 0);
    if (!generator || !os_supported) {
        for (const pd::CoremlRoute * r : {&step, &decoder}) {
            const bool fallback = r->route == "none" && (!generator ? r->state == "absent" && r->reason == "no sidecar"
                                                                    : r->state == "unsupported" && r->reason == "os");
            check(fallback, std::string("template charges: ") +
                                (!generator ? "no generator in this build" : "macOS before 15") + ", so route none (" +
                                r->route + " " + r->state + ": " + r->reason + ")");
        }
        return;
    }
#endif
    if (step.route != "generate" || decoder.route != "generate") {
#if defined(TTS_CPP_USE_COREML) && defined(__APPLE__) && TARGET_OS_OSX
        // A macOS 15+ build with the generator generates both stages of these
        // GGUFs, so a charge it cannot check is a failure, not a skip.
        check(false, "template charges: step route " + step.route + " (" + step.state + ": " + step.reason +
                         "), decoder route " + decoder.route + " (" + decoder.state + ": " + decoder.reason +
                         "); expected both generate");
#else
        fprintf(stderr, "template charges: not checked (step %s, decoder %s; no generated route in this build)\n",
                step.route.c_str(), decoder.route.c_str());
#endif
        return;
    }
    check(step.compute_units == pd::COREML_STEP_DEFAULT_UNITS &&
              decoder.compute_units == pd::COREML_DECODER_DEFAULT_UNITS,
          "default units: step " + step.compute_units + ", decoder " + decoder.compute_units);
    const uint64_t envelope = pd::coreml_envelope_bytes(step, decoder);
    check(envelope == POCKET_ENVELOPE && step.validated && decoder.validated,
          "template envelope " + std::to_string(envelope) + " bytes (expected " + std::to_string(POCKET_ENVELOPE) +
              "), validated for both stages' units");
}

} // namespace

int main(int argc, char ** argv) {
    std::string step_gguf, mimi_gguf, bundle, given_cache;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--step-gguf" && i + 1 < argc) step_gguf = argv[++i];
        else if (a == "--mimi-gguf" && i + 1 < argc) mimi_gguf = argv[++i];
        else if (a == "--bundle" && i + 1 < argc) bundle = argv[++i];
        else if (a == "--cache-dir" && i + 1 < argc) given_cache = argv[++i];
    }
    if (!bundle.empty()) {
        step_gguf = (fs::path(bundle) / "flow-lm.gguf").string();
        mimi_gguf = (fs::path(bundle) / "mimi.gguf").string();
    }
    if (step_gguf.empty() || mimi_gguf.empty() || !std::ifstream(step_gguf).good() ||
        !std::ifstream(mimi_gguf).good()) {
        fprintf(stderr, "SKIP: no step or Mimi GGUF (%s, %s)\n", step_gguf.c_str(), mimi_gguf.c_str());
        return 77;
    }
    unsetenv(pd::COREML_DISABLE_ENV);
    unsetenv(pd::COREML_STRICT_ENV);
    unsetenv(pd::COREML_UNITS_ENV);
    unsetenv("QVAC_COREML_CACHE_DIR");

    const fs::path scratch =
        fs::temp_directory_path() /
        ("pocket-fit-stable-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const fs::path empty = scratch / "empty";
    fs::create_directories(empty);

    auto engine_at = [&](const fs::path & cache) {
        tts_cpp::pocket::EngineOptions e;
        e.flow_lm_path = step_gguf;
        e.mimi_path = mimi_gguf;
        if (!bundle.empty()) {
            e.frontend_path = (fs::path(bundle) / "frontend.json").string();
            e.voice_path = (fs::path(bundle) / "voice.gguf").string();
        }
        e.coreml.cache_dir = cache.string();
        return e;
    };
    auto routes_at = [&](const fs::path & cache) {
        const pd::CoremlEngineOptions o = pd::coreml_engine_options(engine_at(cache));
        const pd::CoremlRoute step = pd::coreml_plan_route(pd::CoremlStage::step, o);
        const pd::CoremlRoute decoder = pd::coreml_plan_route(pd::CoremlStage::decoder, o);
        return std::make_pair(step, decoder);
    };
    auto route_at = [&](const fs::path & cache) {
        const auto [step, decoder] = routes_at(cache);
        return "step " + describe(step) + "; decoder " + describe(decoder) + "; envelope " +
               std::to_string(pd::coreml_envelope_bytes(step, decoder));
    };
    auto fit_at = [&](const fs::path & cache) {
        tts_cpp::pocket::FitOptions f;
        f.text = "Hello.";
        f.engine = engine_at(cache);
        return describe(tts_cpp::pocket::fit_params(f));
    };

    const auto routes0 = routes_at(empty);
    template_charges(routes0.first, routes0.second);
    const std::string route0 = route_at(empty);
    const std::string fit0 = bundle.empty() ? std::string() : fit_at(empty);
    fprintf(stderr, "routes on an empty cache: %s\n", route0.c_str());
    if (!bundle.empty()) fprintf(stderr, "fit on an empty cache:    %s\n", fit0.c_str());

    auto same = [&](const fs::path & cache, const std::string & what) {
        const std::string r = route_at(cache);
        check(r == route0, what + ": routes " + r);
        if (!bundle.empty()) {
            const std::string f = fit_at(cache);
            check(f == fit0, what + ": fit " + f);
        }
    };

#ifdef TTS_CPP_USE_COREML
    const std::string step_units = routes0.first.compute_units;
    const std::string decoder_units = routes0.second.compute_units;
#endif
    fs::path cache;
    if (given_cache.empty()) {
        // What a generation of both stages leaves: the artifacts, their
        // memos, the used markers.
        cache = scratch / "cache";
        const fs::path root = cache / "qvac-coreml-v1";
        const std::pair<const char *, const char *> artifacts[] = {
            {STEP_TEMPLATE, "0123456789abcdef0123456789abcdef"},
            {DECODER_TEMPLATE, "fedcba9876543210fedcba9876543210"},
        };
        for (const auto & [id, key] : artifacts) {
            const fs::path art = root / "artifacts" / (std::string(id) + "." + key);
            write_file(art / "model.mlmodelc" / "model.mil", "program(1.0)\n");
            write_file(art / "artifact.json", std::string("{\"schema\":1,\"complete\":true,\"artifact_key\":\"") +
                                                  key +
                                                  "\",\"inventory\":[{\"path\":\"model.mlmodelc/model.mil\","
                                                  "\"bytes\":13}]}\n");
            write_file(root / "sources" / ("00112233445566778899aabbccddeeff." + std::string(key) + ".json"),
                       "{\"schema\":1}\n");
            write_file(root / "used" / key, "");
        }
        same(cache, "after a generation (artifacts, memos)");
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
            const std::string units = units_of(art, step_units, decoder_units);
            for (const char * host : {"", "test-pocket-coreml", "pocket-cli", "tts-cli"}) {
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
    int written = write_hints(cache, step_units, decoder_units, /*stale=*/false);
    check(n_artifacts > 0 && written == (int) n_artifacts,
          "non-stale hints at the runtime's id for " + std::to_string(written) + " of " +
              std::to_string(n_artifacts) + " artifact(s)");
    same(cache, "with non-stale hints");
    written = write_hints(cache, step_units, decoder_units, /*stale=*/true);
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
