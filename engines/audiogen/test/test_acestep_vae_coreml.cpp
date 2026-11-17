// The ACE-Step VAE decoder on the shared Core ML runtime: low-memory
// semantics, construction, eviction, fit and the memory schedule.
//
//   test-acestep-vae-coreml --self-test
//       Model-free checks (every build): the decode deadline, label merging,
//       option validation, the per-phase Core ML charge and the Core ML fit
//       verdict with memory-degrade.
//   test-acestep-vae-coreml --fit-stable --vae GGUF [--models DIR]
//       The decoder's route and envelope, and with --models the whole fit,
//       are the same on an empty cache directory, on one holding what a
//       generation leaves (artifact, memo, used marker in the runtime's
//       layout) and with a non-stale and then a stale warm hint written where
//       the runtime reads it: the route never reads the cache. With the
//       device cases, the cache the lowmem case populated (its compiled
//       artifact and the hint its load wrote) is compared the same way.
//   test-acestep-vae-coreml --models DIR --cache-root DIR [--cases LIST]
//       [--duration S] [--require-coreml]
//       Device cases (AUDIOGEN_COREML, macOS 15), each on a fresh cache under
//       --cache-root (LIST: encode,lowmem,cached,prewarm,evict,degrade; default all):
//       encode   an encoding (ensure_vae(true)) starts no decoder job;
//       lowmem   three generate() calls in low-memory mode: the first decodes
//                on ggml while the cold load completes after its eviction (no
//                repopulation, hint written), the second and third reload warm
//                within their deadlines and decode on Core ML;
//       cached   (after lowmem) a second process on the primed cache: its
//                first decode, with the default wait settings, adopts the
//                cached artifact and decodes on Core ML;
//       prewarm  {prewarm all, wait construct} succeeds with no decoder
//                retained, and the next generate() decodes on Core ML;
//       evict    an eviction while a job is unstarted cancels it;
//       degrade  memory-degrade near the budget, in fit and in the Engine.
//   test-acestep-vae-coreml --vae GGUF --memory-schedule DIR [--late]
//       The overlapped memory schedule: ggml decodes run on this thread while
//       the decoder is generated, compiled, loaded and warmed into DIR; each
//       phase is held to the template constant that charges it. --late evicts
//       the decoder while it loads (the late-completion schedule) and holds
//       what the load costs after the eviction to the load charge.
//
// Without --require-coreml a route other than `generate` skips the device
// cases (exit 77); with it, that fails.

#include "audiogen-cpp/acestep/engine.h"
#include "audiogen-cpp/acestep/fit.h"
#include "audiogen-cpp/acestep/vae.h"

#include "fit_coreml.h"
#include "vae_coreml.h"
#include "vae_coreml_path.h"
#include "memory_probe.h"

#ifdef AUDIOGEN_USE_COREML
#include "hints.h"
#include "qvac_coreml/manager.h"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#ifdef AUDIOGEN_USE_COREML
#include <spawn.h>
#include <sys/wait.h>

extern char ** environ;
#endif

namespace fs = std::filesystem;
using namespace tts_cpp::acestep;
using Clock = std::chrono::steady_clock;

namespace {

constexpr int SKIP_EXIT_CODE  = 77;
constexpr int LATENT_CHANNELS = 64;

int g_failures = 0;

void check(bool ok, const std::string & what) {
    std::fprintf(stderr, "  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_failures;
}

bool contains(const std::string & s, const std::string & needle) { return s.find(needle) != std::string::npos; }

std::string head(const std::string & s, size_t n = 240) { return s.substr(0, std::min(s.size(), n)); }

// The first "<key>":"<value>" string of a JSON text ("" when absent).
std::string json_string(const std::string & json, const std::string & key) {
    const std::string k = "\"" + key + "\":\"";
    const size_t at = json.find(k);
    if (at == std::string::npos) return {};
    const size_t begin = at + k.size();
    return json.substr(begin, json.find('"', begin) - begin);
}

std::string state_of(const std::string & status) { return json_string(status, "state"); }
bool has_hint(const std::string & status) { return contains(status, "\"hint\":{"); }

double mib(uint64_t bytes) { return bytes / 1048576.0; }

std::string host_chip() {
#if defined(__APPLE__)
    char buf[256] = {};
    size_t n = sizeof(buf);
    if (sysctlbyname("machdep.cpu.brand_string", buf, &n, nullptr, 0) == 0) return buf;
#endif
    return "unknown";
}

std::string fresh_dir(const std::string & root, const std::string & name) {
    const std::string d = root + "/" + name + "-" + std::to_string(Clock::now().time_since_epoch().count());
    fs::create_directories(d);
    return d;
}

// Every compiled bundle or package under `dir`.
int bundles_under(const std::string & dir) {
    int n = 0;
    std::error_code ec;
    for (auto it = fs::recursive_directory_iterator(dir, ec); it != fs::recursive_directory_iterator(); ++it) {
        const std::string ext = it->path().extension().string();
        if (ext == ".mlmodelc" || ext == ".mlpackage") ++n;
    }
    return n;
}

// Polls `status` until `done` holds or `timeout` passes. True when it held.
bool poll_until(const std::function<std::string()> & status, const std::function<bool(const std::string &)> & done,
                std::chrono::seconds timeout, std::string * last = nullptr) {
    const auto t0 = Clock::now();
    for (;;) {
        const std::string s = status();
        if (last) *last = s;
        if (done(s)) return true;
        if (Clock::now() - t0 > timeout) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

std::vector<float> make_latent(int T_latent, uint32_t seed) {
    std::vector<float> latent((size_t) T_latent * LATENT_CHANNELS);
    uint32_t state = seed;
    for (float & v : latent) {
        state = state * 1664525u + 1013904223u;
        v = ((float) (state >> 8) / (float) (1u << 24)) * 4.0f - 2.0f;
    }
    return latent;
}

// ---- model-free checks -------------------------------------------------------

void self_test() {
    std::fprintf(stderr, "self-test\n");
    using detail::vae_ggml_decode_estimate_ms;
    check(vae_ggml_decode_estimate_ms(0) == detail::VAE_GGML_DECODE_MS_FLOOR &&
              vae_ggml_decode_estimate_ms(-5) == detail::VAE_GGML_DECODE_MS_FLOOR &&
              vae_ggml_decode_estimate_ms(500) == 1500 && vae_ggml_decode_estimate_ms(1500) == 4500,
          "decode deadline: 3 ms per latent frame, 100 ms floor");
    check(detail::vae_merge_backend("", "coreml-all") == "coreml-all" &&
              detail::vae_merge_backend("coreml-all", "coreml-all") == "coreml-all" &&
              detail::vae_merge_backend("ggml", "coreml-all") == "mixed" &&
              detail::vae_merge_backend("mixed", "ggml") == "mixed",
          "label merge: equal labels stay, anything else is mixed");

    auto throws = [](const CoremlOptions & o) {
        try {
            detail::vae_coreml_validate_options(o);
        } catch (const std::invalid_argument &) {
            return true;
        }
        return false;
    };
    CoremlOptions bad_units, bad_wait, bad_budget, good;
    bad_units.compute_units = "ane";
    bad_wait.strict_wait_ms = -1;
    bad_budget.memory_budget_bytes = -1;
    good.compute_units = "cpu_and_ane";
    good.strict_wait_ms = 0;
    good.memory_budget_bytes = 0;
    check(throws(bad_units) && throws(bad_wait) && throws(bad_budget) && !throws(good) && !throws(CoremlOptions{}),
          "option validation: invalid units and negative waits or budgets throw");

    // Per-phase charge: the transient everywhere, the resident in the VAE
    // decode phase only unless it can be held in any phase.
    const CoremlPhaseCharge none = acestep_coreml_phase_charge(false, 100, 40, true);
    const CoremlPhaseCharge vae_only = acestep_coreml_phase_charge(true, 100, 40, false);
    const CoremlPhaseCharge every = acestep_coreml_phase_charge(true, 100, 40, true);
    check(none.every_phase == 0 && none.vae_decode == 0 && vae_only.every_phase == 40 && vae_only.vae_decode == 100 &&
              every.every_phase == 140 && every.vae_decode == 0,
          "phase charge: transient in every phase, resident in the VAE phase (or every phase)");

    // The verdict. The first check is the shared pool, the second a discrete
    // device that a budget never caps.
    auto checks = [](uint64_t shared_required, uint64_t shared_free) {
        return std::vector<FitBudgetCheck>{{shared_free, shared_required}, {1000, 10}};
    };
    const CoremlFitVerdict no_gen = acestep_decide_coreml_fit(checks(500, 1000), checks(900, 1000), false, 600);
    check(no_gen.fits && no_gen.ggml_only_fits && no_gen.coreml_fits && !no_gen.degraded,
          "no generated route: the ggml verdict, no memory-degrade");
    const CoremlFitVerdict both = acestep_decide_coreml_fit(checks(500, 1000), checks(900, 1000), true, std::nullopt);
    check(both.fits && both.coreml_fits && !both.degraded, "generated, fits, no budget: Core ML-inclusive verdict");
    const CoremlFitVerdict over = acestep_decide_coreml_fit(checks(500, 1000), checks(1100, 1000), true, std::nullopt);
    check(!over.fits && !over.coreml_fits && over.ggml_only_fits && !over.degraded,
          "generated, over the free memory, no budget: does not fit (no memory-degrade without a budget)");
    const CoremlFitVerdict near = acestep_decide_coreml_fit(checks(500, 1000), checks(900, 1000), true, 700);
    check(near.degraded && near.fits && near.ggml_only_fits && !near.coreml_fits,
          "near the budget (ggml 500 fits 700, ggml + Core ML 900 does not): memory-degrade, the ggml verdict");
    const CoremlFitVerdict room = acestep_decide_coreml_fit(checks(500, 1000), checks(900, 1000), true, 950);
    check(!room.degraded && room.fits, "a budget with room: Core ML kept");
    const CoremlFitVerdict tight = acestep_decide_coreml_fit(checks(500, 1000), checks(1100, 1000), true, 1LL << 40);
    check(tight.degraded && tight.fits && !tight.coreml_fits,
          "near the free memory with a huge budget: memory-degrade on the free memory, the fit stands");
    const CoremlFitVerdict neither = acestep_decide_coreml_fit(checks(800, 1000), checks(1200, 1000), true, 700);
    check(!neither.degraded && !neither.fits, "a budget neither fits: no memory-degrade, does not fit");
    // Both projections fit the free memory but neither fits the budget: the
    // budget caps both verdicts, so the fit is refused, not reported as fitting.
    const CoremlFitVerdict under = acestep_decide_coreml_fit(checks(800, 1000), checks(900, 1000), true, 700);
    check(!under.degraded && !under.fits && !under.ggml_only_fits && !under.coreml_fits,
          "free 1000, ggml 800, Core ML 900, budget 700: neither fits the budget, no memory-degrade, does not fit");
    const CoremlFitVerdict no_gen_budget =
        acestep_decide_coreml_fit(checks(800, 1000), checks(800, 1000), false, 700);
    check(no_gen_budget.fits && !no_gen_budget.degraded,
          "no generated route: a budget leaves today's ggml verdict as it is");
}

// ---- fit stability ------------------------------------------------------------

std::string route_line(const detail::vae_coreml_route & r) {
    return r.route + "|" + r.state + "|" + r.reason + "|" + r.compute_units + "|" + std::to_string(r.envelope_bytes) +
           "|" + std::to_string(r.resident_bytes) + "|" + std::to_string(r.transient_bytes) + "|" +
           std::to_string(r.ane_compile_reserve_bytes) + "|" + (r.validated ? "validated" : "unvalidated") + "|" +
           (r.strict ? "strict" : "");
}

std::string fit_line(const FitResult & f) {
    return f.reason + "|" + f.coreml.route + "|" + std::to_string(f.coreml.bytes) + "|" +
           (f.coreml.validated ? "v" : "u") + "|" + (f.coreml.fits ? "fits" : "no") + "|" +
           f.coreml.disabled_reason + "|" + std::to_string(f.peak_device_bytes) + "|" +
           std::to_string(f.peak_host_bytes);
}

void write_file(const fs::path & p, const std::string & text) {
    fs::create_directories(p.parent_path());
    std::ofstream(p) << text;
}

// Every entry (file or directory) under `dir`.
int entries_under(const fs::path & dir) {
    int n = 0;
    std::error_code ec;
    for (auto it = fs::recursive_directory_iterator(dir, ec); it != fs::recursive_directory_iterator(); ++it) ++n;
    return n;
}

// The artifact directories of the runtime's cache layout under `cache`.
std::vector<fs::path> artifact_dirs(const fs::path & cache) {
    std::vector<fs::path> out;
    std::error_code ec;
    for (const auto & e : fs::directory_iterator(cache / "qvac-coreml-v1" / "artifacts", ec)) {
        if (e.is_directory()) out.push_back(e.path());
    }
    return out;
}

// What a generation of the decoder leaves in the runtime's cache layout: the
// artifact with its compiled model and record, the source memo naming it and
// the used marker.
void simulate_generation(const fs::path & cache) {
    const std::string key = "0123456789abcdef0123456789abcdef";
    const fs::path root = cache / "qvac-coreml-v1";
    const fs::path art = root / "artifacts" / ("acestep-vae-t64." + key);
    write_file(art / "model.mlmodelc" / "model.mil", "program(1.0)\n");
    write_file(art / "artifact.json", "{\"schema\":1,\"complete\":true,\"artifact_key\":\"" + key +
                                          "\",\"inventory\":[{\"path\":\"model.mlmodelc/model.mil\",\"bytes\":13}]}\n");
    write_file(root / "sources" / ("fedcba9876543210fedcba9876543210." + key + ".json"), "{\"schema\":1}\n");
    write_file(root / "used" / key, "");
}

#ifdef AUDIOGEN_USE_COREML
namespace rt = QVAC_COREML_NS;

// The hint record this process's runtime reads for the model of
// `artifact_dir` (function main, `units`, this host and OS build).
std::string runtime_hint_path(const fs::path & cache, const fs::path & artifact_dir, const std::string & units) {
    const auto m = rt::Manager::instance();
    return rt::detail::hint_path_for(cache.string(), m->host_id(), (artifact_dir / "model.mlmodelc").string(), "main",
                                     units, m->os_build());
}

// Artifacts of `cache` with a hint record at the runtime's id.
int hints_at_runtime_id(const fs::path & cache, const std::string & units) {
    int n = 0;
    for (const fs::path & art : artifact_dirs(cache)) {
        if (rt::detail::read_hint(runtime_hint_path(cache, art, units))) ++n;
    }
    return n;
}

// Rewrites every hint record under `cache` with `stale` and writes one at the
// runtime's id for every artifact; returns how many read back at a runtime id
// with that staleness.
int write_hints(const fs::path & cache, const std::string & units, bool stale) {
    const std::optional<int64_t> until =
        stale ? std::optional<int64_t>(rt::detail::unix_now_s() + 365LL * 24 * 3600) : std::nullopt;
    std::error_code ec;
    for (const auto & e : fs::directory_iterator(cache / "qvac-coreml-v1" / "hints", ec)) {
        if (e.path().extension() != ".json") continue;
        std::optional<rt::detail::HintRecord> rec = rt::detail::read_hint(e.path().string());
        if (!rec) continue;
        rec->stale_until_s = until;
        rt::detail::write_hint(e.path().string(), *rec);
    }
    int n = 0;
    for (const fs::path & art : artifact_dirs(cache)) {
        const std::string path = runtime_hint_path(cache, art, units);
        std::optional<rt::detail::HintRecord> rec = rt::detail::read_hint(path);
        if (!rec) {
            rec.emplace();
            rec->warm_load_ms = 120;
            rec->cold_load_ms = 21000;
            rec->last_ok = rt::detail::iso8601_utc(rt::detail::unix_now_s());
            rec->artifact = (art / "model.mlmodelc").string();
        }
        rec->stale_until_s = until;
        fs::create_directories(fs::path(path).parent_path());
        const std::optional<rt::detail::HintRecord> back =
            rt::detail::write_hint(path, *rec) ? rt::detail::read_hint(path) : std::nullopt;
        if (back && back->stale_until_s.has_value() == stale) ++n;
    }
    return n;
}
#endif

// The route (and with `models` the whole fit) planned on an empty cache, to
// compare caches against.
struct cache_comparison {
    std::string vae, models, route0, fit0;

    std::string route_at(const fs::path & cache) const {
        CoremlOptions o;
        o.cache_dir = cache.string();
        return route_line(detail::vae_coreml_plan_route(vae, o));
    }
    std::string fit_at(const fs::path & cache) const {
        FitOptions f;
        f.models_dir = models;
        f.vae_model_path = vae;
        f.coreml.cache_dir = cache.string();
        const FitResult r = fit_params(f);
        return (r.status == FitStatus::Error ? "error|" : "") + fit_line(r);
    }
    cache_comparison(const std::string & v, const std::string & m, const fs::path & empty) : vae(v), models(m) {
        route0 = route_at(empty);
        std::fprintf(stderr, "  route on an empty cache: %s\n", route0.c_str());
        if (!models.empty()) {
            fit0 = fit_at(empty);
            std::fprintf(stderr, "  fit on an empty cache:   %s\n", fit0.c_str());
            check(fit0.rfind("error|", 0) != 0, "fit on an empty cache");
        }
    }
    void same(const fs::path & cache, const std::string & what) const {
        const std::string r = route_at(cache);
        check(r == route0, what + ": route " + r);
        if (!models.empty()) {
            const std::string f = fit_at(cache);
            check(f == fit0, what + ": fit " + f);
        }
    }
};

// The hint cases on `cache`: a non-stale hint, then a stale one, at the id the
// runtime reads for every artifact of it.
void hint_cases(const cache_comparison & cmp, const fs::path & cache, const std::string & units) {
#ifdef AUDIOGEN_USE_COREML
    const size_t n = artifact_dirs(cache).size();
    int written = write_hints(cache, units, /*stale=*/false);
    check(n > 0 && written == (int) n, "non-stale hints at the runtime's id for " + std::to_string(written) + " of " +
                                           std::to_string(n) + " artifact(s)");
    cmp.same(cache, "with non-stale hints");
    written = write_hints(cache, units, /*stale=*/true);
    check(n > 0 && written == (int) n, "stale hints at the runtime's id for " + std::to_string(written) + " of " +
                                           std::to_string(n) + " artifact(s)");
    cmp.same(cache, "with stale hints");
#else
    (void) cmp;
    (void) cache;
    (void) units;
    std::fprintf(stderr, "  built without AUDIOGEN_COREML: hint cases skipped\n");
#endif
}

// The decoder's route and envelope (and with `models` the whole fit) are the
// same on an empty cache, after a generation, and with a non-stale and a stale
// warm hint: the route never reads the cache.
void fit_stable(const std::string & vae, const std::string & models, const std::string & root, bool require) {
    std::fprintf(stderr, "fit stability (%s)\n", vae.c_str());
    const fs::path empty = fresh_dir(root, "fit-empty");
    const fs::path cache = fresh_dir(root, "fit-generated");
    CoremlOptions a;
    a.cache_dir = empty.string();
    const detail::vae_coreml_route ra = detail::vae_coreml_plan_route(vae, a);
#ifdef AUDIOGEN_USE_COREML
    if (require) check(ra.route == "generate", "route generate (" + ra.route + " " + ra.state + " " + ra.reason + ")");
#else
    (void) require;
#endif
    if (ra.route == "generate") {
        // The template's measured charges (template.json memory), pinned so a
        // change to them is deliberate.
        constexpr uint64_t RESIDENT = 1119879168, TRANSIENT = 1120927744, RESERVE = 430964736;
        check(ra.validated && ra.resident_bytes == RESIDENT && ra.transient_bytes == TRANSIENT &&
                  ra.envelope_bytes == RESIDENT + TRANSIENT && ra.ane_compile_reserve_bytes == RESERVE,
              "template charges: resident " + std::to_string(ra.resident_bytes) + ", transient " +
                  std::to_string(ra.transient_bytes) + ", envelope " + std::to_string(ra.envelope_bytes) +
                  ", compiler reserve " + std::to_string(ra.ane_compile_reserve_bytes));
    }
    const cache_comparison cmp(vae, models, empty);
    simulate_generation(cache);
    cmp.same(cache, "after a generation (artifact, memo, used marker)");
    hint_cases(cmp, cache, ra.compute_units);
    check(entries_under(empty) == 0, "planning and fit wrote nothing to the empty cache (" +
                                         std::to_string(entries_under(empty)) + " entries)");
    std::error_code ec;
    fs::remove_all(empty, ec);
    fs::remove_all(cache, ec);
}

// The same comparison on a cache a real run populated (a compiled artifact and
// the hint its load wrote) against a fresh empty directory: the cache as it
// is, then with its hints stale.
void fit_stable_on(const std::string & vae, const std::string & models, const std::string & root,
                   const fs::path & cache) {
    std::fprintf(stderr, "fit stability on the populated cache %s\n", cache.string().c_str());
    const fs::path empty = fresh_dir(root, "fit-empty");
    CoremlOptions a;
    a.cache_dir = empty.string();
    const std::string units = detail::vae_coreml_plan_route(vae, a).compute_units;
    const cache_comparison cmp(vae, models, empty);
    const size_t n = artifact_dirs(cache).size();
#ifdef AUDIOGEN_USE_COREML
    const int hinted = hints_at_runtime_id(cache, units);
    check(n > 0 && hinted > 0, "the populated cache holds " + std::to_string(n) + " artifact(s), " +
                                   std::to_string(hinted) + " with the hint its load wrote at the runtime's id");
#else
    check(n > 0, "the populated cache holds " + std::to_string(n) + " artifact(s)");
#endif
    cmp.same(cache, "the populated cache as it is");
    hint_cases(cmp, cache, units);
    std::error_code ec;
    fs::remove_all(empty, ec);
}

// ---- device cases --------------------------------------------------------------

EngineOptions engine_options(const std::string & models, const std::string & cache) {
    EngineOptions o;
    o.models_dir = models;
    o.n_gpu_layers = 99;
    o.coreml.cache_dir = cache;
    return o;
}

GenerateParams short_params(float duration) {
    GenerateParams p;
    p.caption = "Calm ambient synth pad";
    p.lyrics = AUDIO_EDIT_DEFAULT_LYRICS;
    p.duration = duration;
    p.seed = 42;
    p.lm_phase1 = false;
    p.bpm = 90;
    p.keyscale = "C major";
    p.timesignature = "4/4";
    return p;
}

// An encoding (ensure_vae(true)) requests no decoder. understand() loads the
// VAE with its encoder and reports the "source" stage before and after the
// encode, both before free_vae() evicts the decoder slot; an eviction reports
// the slot idle even while a started job runs, so the slot is read there.
// After the eviction nothing may reach the cache for longer than a cold
// generation, compilation and load take.
void case_encode(const std::string & models, const std::string & root) {
    std::fprintf(stderr, "encode: an encoding starts no decoder job\n");
    const std::string cache = fresh_dir(root, "encode");
    std::unique_ptr<Engine> engine = Engine::create(engine_options(models, cache));
    UnderstandParams up;
    const int frames = 48000 * 4;
    up.audio.resize((size_t) frames * 2);
    for (int i = 0; i < frames; ++i) {
        const float v = 0.3f * std::sin(2.0f * 3.14159265f * 220.0f * (float) i / 48000.0f);
        up.audio[(size_t) 2 * i] = v;
        up.audio[(size_t) 2 * i + 1] = v;
    }
    std::vector<std::string> at_source;
    bool stopped = false;
    engine->understand(up, [&](const std::string & stage, int, int) {
        if (stage == "source") at_source.push_back(engine->coreml_status());
        if (stage == "tok") {
            stopped = true;
            return false;  // free_vae() has run; the encoding is what this case checks
        }
        return true;
    });
    bool idle = at_source.size() == 2;
    for (const std::string & st : at_source) {
        std::fprintf(stderr, "  decoder slot during the encoding: %s%s\n", state_of(st).c_str(),
                     has_hint(st) ? ", hint" : "");
        idle = idle && state_of(st) == "idle" && !has_hint(st);
    }
    check(stopped && idle, "with the VAE encoder loaded, before and after the encode (before free_vae): decoder slot "
                           "idle, no hint (" + std::to_string(at_source.size()) + " samples)");
    // A started job keeps running after the eviction; it would leave a bundle
    // or a hint well within this window (a cold generation, compilation and
    // load take about 12 s on an M2).
    std::string last;
    const auto touched = [&](const std::string & st) {
        return has_hint(st) || bundles_under(cache) > 0 || !artifact_dirs(cache).empty() ||
               entries_under(fs::path(cache) / "qvac-coreml-v1" / "hints") > 0;
    };
    const bool written = poll_until([&] { return engine->coreml_status(); }, touched, std::chrono::seconds(30), &last);
    check(!written && state_of(last) == "idle",
          "nothing reached the cache within 30 s of the eviction (" + std::to_string(bundles_under(cache)) +
              " bundle(s), " + std::to_string(artifact_dirs(cache).size()) + " artifact(s), slot " + state_of(last) +
              ")");
}

// Three generate() calls in low-memory mode on a fresh cache. Returns the
// cache they populated.
std::string case_lowmem(const std::string & models, const std::string & root, float duration) {
    std::fprintf(stderr, "lowmem: three generations, %.0f s each\n", duration);
    const std::string cache = fresh_dir(root, "lowmem");
    std::unique_ptr<Engine> engine = Engine::create(engine_options(models, cache));
    const std::string at_create = engine->coreml_status();
    check(state_of(at_create) == "idle", "create() requests nothing (" + state_of(at_create) + ")");

    // Gen 1: hold the ggml decode until the cold load has started, so its
    // completion falls after the eviction at the end of the VAE stage.
    std::string at_vae;
    bool load_started = false;
    // The engine reports ("vae", 0, 1) before the decode; the decode's own
    // progress (the ggml graph's node counts) comes after its acquisition.
    const GenerateResult g1 = engine->generate(short_params(duration), [&](const std::string & stage, int step, int) {
        if (stage == "vae" && step > 0 && at_vae.empty()) {
            load_started = poll_until([&] { return engine->coreml_status(); },
                                      [](const std::string & s) {
                                          const std::string st = state_of(s);
                                          return st == "loading" || st == "warming";
                                      },
                                      std::chrono::seconds(300), &at_vae);
        }
        return true;
    });
    const std::string after1 = engine->coreml_status();
    std::fprintf(stderr, "  gen 1: %s pending=%d, %zu samples; slot after: %s\n", g1.vae_backend.c_str(),
                 (int) g1.coreml_pending, g1.pcm.size(), head(after1).c_str());
    check(load_started, "gen 1: the cold load started during the ggml decode (" + state_of(at_vae) + ")");
    check(!g1.pcm.empty() && g1.vae_backend == "ggml" && g1.coreml_pending,
          "gen 1 decodes on ggml, coreml_pending (" + g1.vae_backend + ")");
    check(state_of(after1) == "idle", "gen 1: free_vae() evicted the slot (" + state_of(after1) + ")");
    // The started load completes after the eviction: it writes the hint and
    // never repopulates the slot.
    bool repopulated = false;
    std::string last;
    const bool hinted = poll_until([&] { return engine->coreml_status(); },
                                   [&](const std::string & s) {
                                       if (state_of(s) != "idle") repopulated = true;
                                       return has_hint(s);
                                   },
                                   std::chrono::seconds(300), &last);
    std::this_thread::sleep_for(std::chrono::seconds(1));
    const std::string settled = engine->coreml_status();
    check(hinted, "the late cold load wrote its hint: " + head(last));
    check(!repopulated && state_of(settled) == "idle", "the late load never repopulated the evicted slot (" +
                                                           state_of(settled) + ")");

    for (int g = 2; g <= 3; ++g) {
        const auto t0 = Clock::now();
        const GenerateResult r = engine->generate(short_params(duration));
        const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
        const std::string st = engine->coreml_status();
        std::fprintf(stderr, "  gen %d: %s pending=%d, %.1f s; slot after: %s\n", g, r.vae_backend.c_str(),
                     (int) r.coreml_pending, secs, st.c_str());
        check(!r.pcm.empty() && r.vae_backend.rfind("coreml-", 0) == 0 && !r.coreml_pending,
              "gen " + std::to_string(g) + " reloads warm within its deadline and decodes on " + r.vae_backend);
        check(state_of(st) == "idle", "gen " + std::to_string(g) + ": evicted again (" + state_of(st) + ")");
    }
    return cache;
}

// A fresh process on a cache another process primed (the lowmem case left a
// compiled artifact and the warm hint its load wrote): the first decode of
// this process adopts the cached artifact and waits for its warm load within
// the decode's deadline, with the default wait settings, and decodes on Core
// ML. Runs in a child process (--first-decode) so nothing of the priming
// process's runtime is shared.
void first_decode_on(const std::string & models, const std::string & cache, float duration) {
    std::fprintf(stderr, "first decode of a fresh process on the primed cache %s\n", cache.c_str());
    std::unique_ptr<Engine> engine = Engine::create(engine_options(models, cache));
    const std::string at_create = engine->coreml_status();
    check(state_of(at_create) == "idle", "create() requests nothing (" + state_of(at_create) + ")");
    const GenerateResult r = engine->generate(short_params(duration));
    const std::string st = engine->coreml_status();
    std::fprintf(stderr, "  first generate: %s pending=%d; slot after: %s\n", r.vae_backend.c_str(),
                 (int) r.coreml_pending, head(st, 400).c_str());
    check(!r.pcm.empty() && r.vae_backend.rfind("coreml-", 0) == 0 && !r.coreml_pending,
          "the first decode of a fresh process uses the cached artifact (" + r.vae_backend + ")");
    check(contains(st, "\"route\":\"cached\""), "status route cached");
}

int run_child(const std::vector<std::string> & args) {
#ifndef AUDIOGEN_USE_COREML
    (void) args;
    return -1;
#else
    std::vector<char *> argv;
    for (const std::string & a : args) argv.push_back(const_cast<char *>(a.c_str()));
    argv.push_back(nullptr);
    pid_t pid = 0;
    if (posix_spawn(&pid, argv[0], nullptr, nullptr, argv.data(), environ) != 0) return -1;
    int status = 0;
    if (waitpid(pid, &status, 0) != pid) return -1;
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
}

void case_cached(const std::string & self, const std::string & models, const std::string & cache, float duration) {
    std::fprintf(stderr, "cached: a second process's first decode on a primed cache\n");
    const int rc = run_child({self, "--models", models, "--first-decode", cache, "--duration",
                              std::to_string(duration)});
    check(rc == 0, "the child process's first decode ran on Core ML (exit " + std::to_string(rc) + ")");
}

// Install-time preparation: {prewarm all, wait construct} in low-memory mode.
void case_prewarm(const std::string & models, const std::string & root, float duration) {
    std::fprintf(stderr, "prewarm: {prewarm all, wait construct}\n");
    const std::string cache = fresh_dir(root, "prewarm");
    EngineOptions o = engine_options(models, cache);
    o.coreml.prewarm = CoremlOptions::Prewarm::all;
    o.coreml.wait = CoremlOptions::Wait::construct;
    const auto t0 = Clock::now();
    std::unique_ptr<Engine> engine = Engine::create(o);
    const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
    const std::string group = engine->coreml_construction();
    const std::string st = engine->coreml_status();
    std::fprintf(stderr, "  construction %.1f s: %s\n  status: %s\n", secs, head(group, 400).c_str(),
                 head(st, 400).c_str());
    check(contains(group, "\"success\":true") && contains(group, "\"nothing_to_prepare\":false"),
          "the construction group succeeds");
    check(state_of(st) == "idle" && has_hint(st) && !engine->coreml_wait(0),
          "no decoder retained after construction (" + state_of(st) + "), hint written");
    const GenerateResult r = engine->generate(short_params(duration));
    check(!r.pcm.empty() && r.vae_backend.rfind("coreml-", 0) == 0 && !r.coreml_pending,
          "the next generate() decodes on " + r.vae_backend);
}

// An eviction while a job is unstarted cancels it: two stages on two caches;
// the first one's preparation holds the queue while the second's waits; the
// second is evicted and never prepares anything.
void case_evict(const std::string & vae, const std::string & root) {
    std::fprintf(stderr, "evict: an unstarted job is cancelled\n");
    const std::string ca = fresh_dir(root, "evict-a");
    const std::string cb = fresh_dir(root, "evict-b");
    CoremlOptions oa, ob;
    oa.cache_dir = ca;
    ob.cache_dir = cb;
    auto a = detail::vae_coreml_stage::create(vae, oa, false, true);
    auto b = detail::vae_coreml_stage::create(vae, ob, false, true);
    if (!a || !b || !a->active() || !b->active()) {
        check(false, "two active stages");
        return;
    }
    a->acquire(64);
    poll_until([&] { return a->status_json(); },
               [](const std::string & s) { return state_of(s) != "queued" && state_of(s) != "idle"; },
               std::chrono::seconds(30));
    b->acquire(64);
    const std::string queued = b->status_json();
    b->evict();
    const std::string evicted = b->status_json();
    std::fprintf(stderr, "  a: %s\n  b before eviction: %s\n", state_of(a->status_json()).c_str(),
                 state_of(queued).c_str());
    check(state_of(queued) == "queued" && state_of(evicted) == "idle",
          "b was queued behind a (" + state_of(queued) + ") and is idle after its eviction (" + state_of(evicted) +
              ")");
    std::string last;
    const bool a_ready = poll_until([&] { return a->status_json(); },
                                    [](const std::string & s) { return state_of(s) == "ready"; },
                                    std::chrono::seconds(300), &last);
    std::this_thread::sleep_for(std::chrono::seconds(1));
    check(a_ready, "a was prepared and loaded: " + state_of(last));
    check(state_of(b->status_json()) == "idle" && bundles_under(cb) == 0 && !has_hint(b->status_json()),
          "b's cancelled job never ran (" + std::to_string(bundles_under(cb)) + " bundle(s) in b's cache)");
}

// Memory-degrade near the budget: a budget that fits the ggml projection
// but not the Core ML-inclusive one resolves use = off (fit and Engine); a
// budget with room keeps Core ML.
void case_degrade(const std::string & models, const std::string & root) {
    std::fprintf(stderr, "degrade: memory-degrade near the budget\n");
    const std::string cache = fresh_dir(root, "degrade");
    FitOptions f;
    f.models_dir = models;
    f.n_gpu_layers = 99;
    f.duration_seconds = 20.0f;
    f.coreml.cache_dir = cache;
    FitOptions off = f;
    off.coreml.use = CoremlOptions::Use::off;
    const FitResult with = fit_params(f);
    const FitResult without = fit_params(off);
    const uint64_t need_with = with.peak_device_bytes + with.peak_host_bytes;
    const uint64_t need_without = without.peak_device_bytes + without.peak_host_bytes;
    std::fprintf(stderr, "  ggml %.1f MiB, with Core ML %.1f MiB (envelope %.1f MiB)\n", mib(need_without),
                 mib(need_with), mib(with.coreml.bytes));
    check(with.coreml.route == "generate" && without.coreml.route == "none" && need_with > need_without,
          "the Core ML-inclusive projection charges more (" + with.coreml.route + ")");
    const uint64_t delta = need_with - need_without;
    FitOptions near = f;
    near.coreml.memory_budget_bytes = (int64_t) (need_without + f.margin_bytes + delta / 2);
    FitOptions roomy = f;
    roomy.coreml.memory_budget_bytes = (int64_t) (need_with + f.margin_bytes + delta);
    const FitResult degraded = fit_params(near);
    const FitResult kept = fit_params(roomy);
    check(degraded.coreml.disabled_reason == "memory" && degraded.fits && kept.coreml.disabled_reason.empty() &&
              kept.fits,
          "fit: '" + degraded.coreml.disabled_reason + "' near the budget, '" + kept.coreml.disabled_reason +
              "' with room");
    EngineOptions o = engine_options(models, cache);
    o.coreml.memory_budget_bytes = near.coreml.memory_budget_bytes;
    std::unique_ptr<Engine> low = Engine::create(o);
    const std::string st = low->coreml_status();
    check(state_of(st) == "disabled", "Engine near the budget: decoder " + state_of(st) + " (" + head(st, 200) + ")");
    o.coreml.memory_budget_bytes = roomy.coreml.memory_budget_bytes;
    std::unique_ptr<Engine> room = Engine::create(o);
    check(state_of(room->coreml_status()) == "idle", "Engine with room: decoder " + state_of(room->coreml_status()));
}

// ---- memory schedule ----------------------------------------------------------

enum class mem_phase { ggml, generate, compile, load, warm, ready, late };

mem_phase mem_phase_of(const std::string & state) {
    if (state == "digesting" || state == "generating") return mem_phase::generate;
    if (state == "compiling") return mem_phase::compile;
    if (state == "loading") return mem_phase::load;
    if (state == "warming") return mem_phase::warm;
    if (state == "ready") return mem_phase::ready;
    return mem_phase::ggml;
}

struct mem_sample {
    uint64_t footprint = 0;
    mem_phase phase = mem_phase::ggml;
};

// Increments in bytes, each over the footprint just before its group began:
// prepare (generate, compile), load and warm-up, the steady state (over the
// footprint just before the load), the late tail (from the eviction until the
// late load released its model, over the footprint just before the load) and
// the whole run over the baseline.
struct mem_breakdown {
    uint64_t prepare = 0, load_warm = 0, steady = 0, late = 0, total = 0;
    bool prepared = false, loaded = false, ready = false, late_seen = false;
};

mem_breakdown mem_breakdown_of(const std::vector<mem_sample> & samples, uint64_t base) {
    const auto group = [](mem_phase p) {
        if (p == mem_phase::generate || p == mem_phase::compile) return 1;
        if (p == mem_phase::load || p == mem_phase::warm) return 2;
        if (p == mem_phase::late) return 3;
        return 0;
    };
    const auto over = [](uint64_t v, uint64_t from) { return v > from ? v - from : (uint64_t) 0; };
    mem_breakdown b;
    uint64_t prev = base, group_start = base;
    std::optional<uint64_t> load_start;
    mem_phase last = mem_phase::ggml;
    for (const mem_sample & s : samples) {
        if (group(s.phase) != group(last)) group_start = prev;
        if (group(s.phase) == 2 && !load_start) load_start = group_start;
        switch (group(s.phase)) {
            case 1: b.prepared = true; b.prepare = std::max(b.prepare, over(s.footprint, group_start)); break;
            case 2: b.loaded = true; b.load_warm = std::max(b.load_warm, over(s.footprint, group_start)); break;
            case 3: b.late_seen = true; b.late = std::max(b.late, over(s.footprint, load_start.value_or(base))); break;
            default: break;
        }
        if (s.phase == mem_phase::ready) {
            b.ready = true;
            b.steady = std::max(b.steady, over(s.footprint, load_start.value_or(base)));
        }
        b.total = std::max(b.total, over(s.footprint, base));
        prev = s.footprint;
        last = s.phase;
    }
    return b;
}

void memory_schedule(const std::string & vae, const std::string & cache, bool late, bool require) {
    std::fprintf(stderr, "memory schedule%s (cache %s, host %s)\n", late ? ", late completion" : "", cache.c_str(),
                 host_chip().c_str());
    CoremlOptions opts;
    opts.cache_dir = cache;
    const detail::vae_coreml_route route = detail::vae_coreml_plan_route(vae, opts);
    if (route.route != "generate") {
        check(!require, "route generate (" + route.route + " " + route.state + " " + route.reason + ")");
        return;
    }
    // The ggml work: a decode-only ggml VAE on the GPU, like the engine's.
    VaeOptions vo;
    vo.with_encoder = false;
    vo.n_gpu_layers = 99;
    vo.coreml.use = CoremlOptions::Use::off;
    std::unique_ptr<Vae> ggml_vae = Vae::load(vae, vo);
    const int T = 250;  // 10 s of audio: four Core ML windows
    const std::vector<float> latent = make_latent(T, 0x2468ace1u);
    for (int i = 0; i < 2; ++i) ggml_vae->decode(latent, T);  // warm the ggml arena
    const uint64_t base = qvac_coreml_memory::phys_footprint();

    std::atomic<bool> stop{false};
    std::atomic<int> phase{(int) mem_phase::ggml};
    std::atomic<bool> in_late{false};
    std::shared_ptr<detail::vae_coreml_stage> stage = detail::vae_coreml_stage::create(vae, opts, false, true);
    std::vector<qvac_coreml_memory::AneSample> ane_samples;
    std::vector<mem_sample> samples;
    samples.reserve(1 << 17);
    std::thread sampler([&] {
        for (int tick = 0; !stop.load(); ++tick) {
            if (tick % 5 == 0) {
                const mem_phase p = in_late.load() ? mem_phase::late : mem_phase_of(state_of(stage->status_json()));
                phase.store((int) p);
            }
            samples.push_back({qvac_coreml_memory::phys_footprint(), (mem_phase) phase.load()});
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    });
    std::thread ane_sampler([&] {
        while (!stop.load()) {
            const uint64_t a = qvac_coreml_memory::ane_compiler_rss();
            const auto p = (mem_phase) phase.load();
            ane_samples.push_back({a, p == mem_phase::load || p == mem_phase::warm || p == mem_phase::late});
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    });

    const auto t0 = Clock::now();
    int ggml_calls = 0, coreml_calls = 0;
    bool evicted = false, hint_seen = false;
    while (Clock::now() - t0 < std::chrono::minutes(6)) {
        if (late && evicted) {
            // After the eviction: ggml keeps working until the late load
            // wrote its hint, then two more seconds for the release.
            ggml_vae->decode(latent, T);
            ++ggml_calls;
            if (!hint_seen && has_hint(stage->status_json())) {
                hint_seen = true;
                const auto t_hint = Clock::now();
                while (Clock::now() - t_hint < std::chrono::seconds(2)) {
                    ggml_vae->decode(latent, T);
                    ++ggml_calls;
                }
                break;
            }
            continue;
        }
        detail::vae_coreml_call call = stage->acquire(T);
        if (late && !evicted && (phase.load() == (int) mem_phase::load || phase.load() == (int) mem_phase::warm)) {
            call = {};
            in_late.store(true);
            stage->evict();  // a VAE stage boundary while the cold load runs
            evicted = true;
            continue;
        }
        if (!call.decoder) {
            ggml_vae->decode(latent, T);
            ++ggml_calls;
            continue;
        }
        std::vector<float> pcm(latent.size() / LATENT_CHANNELS * 1920 * 2);
        bool ok = true;
        const int w = call.decoder->window();
        std::vector<float> win((size_t) w * 1920 * 2);
        for (int a = 0; a + w <= T && ok; a += w - 16) ok = call.decoder->decode(latent.data() + (size_t) a * 64, win.data());
        if (!ok) {
            check(false, "a Core ML decode failed during the schedule");
            break;
        }
        if (++coreml_calls >= 5) break;
    }
    const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
    if (!late) stage->evict();  // the early-completion schedule ends at a stage boundary
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    stop.store(true);
    sampler.join();
    ane_sampler.join();
    std::fprintf(stderr, "  status: %s\n", head(stage->status_json(), 600).c_str());

    const mem_breakdown b = mem_breakdown_of(samples, base);
    const qvac_coreml_memory::AneGrowth ane = qvac_coreml_memory::ane_compiler_growth(ane_samples);
    const char * baseline = qvac_coreml_memory::ane_baseline_name(qvac_coreml_memory::ane_compiler_baseline(ane));
    uint64_t ane_peak = 0;
    for (const auto & a : ane_samples) ane_peak = std::max(ane_peak, a.rss);
    std::fprintf(stderr,
                 "  memory (MiB; base %.1f): prepare %.1f (charged %.1f), load+warm %.1f (charged %.1f), steady %.1f "
                 "(resident charged %.1f), late tail %.1f (load charged %.1f), whole run %.1f (envelope %.1f); "
                 "ANECompilerService growth %.1f from %.1f (%s), peak %.1f, reserve %.1f\n",
                 mib(base), mib(b.prepare), mib(route.prepare_peak_bytes), mib(b.load_warm), mib(route.load_peak_bytes),
                 mib(b.steady), mib(route.resident_bytes), mib(b.late), mib(route.load_peak_bytes), mib(b.total),
                 mib(route.envelope_bytes), mib(ane.growth), mib(ane.before), baseline, mib(ane_peak),
                 mib(route.ane_compile_reserve_bytes));
    std::fprintf(stderr,
                 "  memory report: {\"chip\":\"%s\",\"units\":\"%s\",\"schedule\":\"%s\",\"base\":%llu,\"prepare\":%llu,"
                 "\"load_warm\":%llu,\"steady\":%llu,\"late\":%llu,\"total\":%llu,\"ane_compiler_before_load\":%llu,"
                 "\"ane_compiler_growth\":%llu,\"ane_compiler_load_peak\":%llu,\"ane_compiler_peak\":%llu,"
                 "\"ane_compiler_baseline\":\"%s\",\"ggml_calls\":%d,\"coreml_calls\":%d,\"seconds\":%.1f}\n",
                 host_chip().c_str(), route.compute_units.c_str(), late ? "late" : "early", (unsigned long long) base,
                 (unsigned long long) b.prepare, (unsigned long long) b.load_warm, (unsigned long long) b.steady,
                 (unsigned long long) b.late, (unsigned long long) b.total, (unsigned long long) ane.before,
                 (unsigned long long) ane.growth, (unsigned long long) ane.peak, (unsigned long long) ane_peak, baseline,
                 ggml_calls, coreml_calls, secs);

    check(b.prepared && b.loaded, "the schedule sampled the preparation and the load");
    check(b.prepare <= route.prepare_peak_bytes, "prepare within its charge");
    check(b.load_warm <= route.load_peak_bytes, "load and warm-up within the load charge");
    check(b.total <= route.envelope_bytes, "whole run within the envelope");
    if (route.ane_compile_reserve_bytes > 0) {
        check(ane.growth <= route.ane_compile_reserve_bytes, "ANECompilerService growth within the reserve");
    }
    if (late) {
        check(evicted && hint_seen && b.late_seen, "late completion: evicted while loading, the load wrote its hint");
        check(b.late <= route.load_peak_bytes, "late tail within the load charge (the transient fit charges in "
                                               "every phase)");
        check(coreml_calls == 0, "late completion: no Core ML decode ran (the slot was never repopulated)");
    } else {
        check(b.ready && coreml_calls >= 5 && ggml_calls >= 1, "early completion: ggml then 5 Core ML calls (" +
                                                                    std::to_string(ggml_calls) + " / " +
                                                                    std::to_string(coreml_calls) + ")");
        check(b.steady <= route.resident_bytes, "steady state within the resident charge");
    }
}

}  // namespace

int main(int argc, char ** argv) {
    bool self = false, fit = false, require = false, late = false;
    std::string models, vae, cache_root, memory_dir, first_decode, cases = "encode,lowmem,cached,prewarm,evict,degrade";
    float duration = 20.0f;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
            return argv[++i];
        };
        if (a == "--self-test") self = true;
        else if (a == "--fit-stable") fit = true;
        else if (a == "--require-coreml") require = true;
        else if (a == "--late") late = true;
        else if (a == "--models") models = next();
        else if (a == "--vae") vae = next();
        else if (a == "--cache-root") cache_root = next();
        else if (a == "--cases") cases = next();
        else if (a == "--duration") duration = std::stof(next());
        else if (a == "--memory-schedule") memory_dir = next();
        else if (a == "--first-decode") first_decode = next();
        else {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    if (vae.empty() && !models.empty()) vae = models + "/vae-BF16.gguf";

    if (self) self_test();
    if (fit) {
        if (vae.empty()) {
            std::fprintf(stderr, "--fit-stable needs --vae or --models\n");
            return 2;
        }
        fit_stable(vae, models, cache_root.empty() ? fs::temp_directory_path().string() : cache_root, require);
    }
#ifdef AUDIOGEN_USE_COREML
    if (!first_decode.empty()) {
        first_decode_on(models, first_decode, duration);
        std::fprintf(stderr, "%s (%d failure(s))\n", g_failures ? "FAIL" : "OK", g_failures);
        return g_failures ? 1 : 0;
    }
    if (!memory_dir.empty()) memory_schedule(vae, memory_dir, late, require);
    if (!models.empty() && !cache_root.empty()) {
        CoremlOptions probe;
        probe.cache_dir = cache_root;
        const detail::vae_coreml_route r = detail::vae_coreml_plan_route(vae, probe);
        if (fs::exists(coreml_vae_sidecar_path(vae))) {
            check(false, "no prebuilt sidecar next to the VAE GGUF (the generated route needs none)");
        } else if (r.route != "generate") {
            std::fprintf(stderr, "route %s (%s: %s)\n", r.route.c_str(), r.state.c_str(), r.reason.c_str());
            if (!require) return SKIP_EXIT_CODE;
            check(false, "route generate");
        } else {
            std::fprintf(stderr, "device cases on %s (host %s)\n", models.c_str(), host_chip().c_str());
            if (contains(cases, "encode")) case_encode(models, cache_root);
            std::string lowmem_cache;
            if (contains(cases, "lowmem")) lowmem_cache = case_lowmem(models, cache_root, duration);
            if (contains(cases, "cached") && !lowmem_cache.empty()) case_cached(argv[0], models, lowmem_cache, duration);
            if (contains(cases, "prewarm")) case_prewarm(models, cache_root, duration);
            if (contains(cases, "evict")) case_evict(vae, cache_root);
            if (contains(cases, "degrade")) case_degrade(models, cache_root);
            // The cache the low-memory generations left (a compiled artifact
            // and the hint its load wrote) against an empty one.
            if (fit && !lowmem_cache.empty()) fit_stable_on(vae, models, cache_root, lowmem_cache);
        }
    }
#else
    if (!memory_dir.empty() || (!models.empty() && !cache_root.empty())) {
        std::fprintf(stderr, "built without AUDIOGEN_COREML: device cases skipped\n");
        if (require) check(false, "--require-coreml in a build without AUDIOGEN_COREML");
        else if (!self && !fit) return SKIP_EXIT_CODE;
    }
#endif
    std::fprintf(stderr, "%s (%d failure(s))\n", g_failures ? "FAIL" : "OK", g_failures);
    return g_failures ? 1 : 0;
}
