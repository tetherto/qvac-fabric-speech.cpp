// The ACE-Step VAE decoder on the shared Core ML runtime: low-memory
// semantics, construction, eviction, fit and the memory schedule.
//
//   test-acestep-vae-coreml --self-test
//       Model-free checks (every build): the decode deadline, label merging,
//       option validation, the per-phase Core ML charge and the Core ML fit
//       verdict with memory-degrade.
//   test-acestep-vae-coreml --fit-stable --vae GGUF [--models DIR]
//       The decoder's route and envelope, and with --models the whole fit,
//       are the same with an empty cache directory and with one holding
//       artifacts and warm hints (fresh, stale): the route never reads the
//       cache.
//   test-acestep-vae-coreml --models DIR --cache-root DIR [--cases LIST]
//       [--duration S] [--require-coreml]
//       Device cases (AUDIOGEN_COREML, macOS 15), each on a fresh cache under
//       --cache-root (LIST: encode,lowmem,prewarm,evict,degrade; default all):
//       encode   an encoding (ensure_vae(true)) starts no decoder job;
//       lowmem   three generate() calls in low-memory mode: the first decodes
//                on ggml while the cold load completes after its eviction (no
//                repopulation, hint written), the second and third reload warm
//                within their deadlines and decode on Core ML;
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
    check(near.degraded && near.fits && near.coreml_fits,
          "near the budget (ggml 500 fits 700, ggml + Core ML 900 does not): memory-degrade, the ggml verdict");
    const CoremlFitVerdict room = acestep_decide_coreml_fit(checks(500, 1000), checks(900, 1000), true, 950);
    check(!room.degraded && room.fits, "a budget with room: Core ML kept");
    const CoremlFitVerdict tight = acestep_decide_coreml_fit(checks(500, 1000), checks(1100, 1000), true, 1LL << 40);
    check(tight.degraded && tight.fits && !tight.coreml_fits,
          "near the free memory with a huge budget: memory-degrade on the free memory, the fit stands");
    const CoremlFitVerdict neither = acestep_decide_coreml_fit(checks(800, 1000), checks(1200, 1000), true, 700);
    check(!neither.degraded && !neither.fits, "a budget neither fits: no memory-degrade, does not fit");
}

// ---- fit stability ------------------------------------------------------------

std::string route_line(const detail::vae_coreml_route & r) {
    return r.route + "|" + r.state + "|" + r.compute_units + "|" + std::to_string(r.envelope_bytes) + "|" +
           std::to_string(r.resident_bytes) + "|" + std::to_string(r.transient_bytes) + "|" +
           (r.validated ? "validated" : "unvalidated");
}

std::string fit_line(const FitResult & f) {
    return f.reason + "|" + f.coreml.route + "|" + std::to_string(f.coreml.bytes) + "|" +
           (f.coreml.validated ? "v" : "u") + "|" + (f.coreml.fits ? "fits" : "no") + "|" +
           f.coreml.disabled_reason + "|" + std::to_string(f.peak_device_bytes) + "|" +
           std::to_string(f.peak_host_bytes);
}

void populate_cache(const std::string & dir) {
    fs::create_directories(dir + "/hints");
    fs::create_directories(dir + "/artifacts/fake.mlmodelc/weights");
    std::ofstream(dir + "/hints/fresh.json") << "{\"warm_load_ms\":120,\"cold_load_ms\":21000,\"last_ok\":1,\"stale_until\":0}";
    std::ofstream(dir + "/hints/stale.json") << "{\"warm_load_ms\":120,\"cold_load_ms\":21000,\"last_ok\":1,"
                                                "\"stale_until\":99999999999999}";
    std::ofstream(dir + "/artifacts/fake.mlmodelc/weights/weight.bin") << "x";
}

void fit_stable(const std::string & vae, const std::string & models, const std::string & root, bool require) {
    std::fprintf(stderr, "fit stability (%s)\n", vae.c_str());
    const std::string empty = fresh_dir(root, "fit-empty");
    const std::string full = fresh_dir(root, "fit-populated");
    populate_cache(full);
    CoremlOptions a, b;
    a.cache_dir = empty;
    b.cache_dir = full;
    const detail::vae_coreml_route ra = detail::vae_coreml_plan_route(vae, a);
    const detail::vae_coreml_route rb = detail::vae_coreml_plan_route(vae, b);
    std::fprintf(stderr, "  route %s\n", route_line(ra).c_str());
#ifdef AUDIOGEN_USE_COREML
    if (require) check(ra.route == "generate", "route generate (" + ra.route + " " + ra.state + " " + ra.reason + ")");
#else
    (void) require;
#endif
    check(route_line(ra) == route_line(rb), "route and envelope: empty cache == populated cache with hints");
    if (ra.route == "generate") {
        // The template's measured charges (template.json memory), pinned so a
        // change to them is deliberate.
        constexpr uint64_t RESIDENT = 1119879168, TRANSIENT = 1120927744, RESERVE = 245366784;
        check(ra.validated && ra.resident_bytes == RESIDENT && ra.transient_bytes == TRANSIENT &&
                  ra.envelope_bytes == RESIDENT + TRANSIENT && ra.ane_compile_reserve_bytes == RESERVE,
              "template charges: resident " + std::to_string(ra.resident_bytes) + ", transient " +
                  std::to_string(ra.transient_bytes) + ", envelope " + std::to_string(ra.envelope_bytes) +
                  ", compiler reserve " + std::to_string(ra.ane_compile_reserve_bytes));
    }
    // A second plan of the empty cache after the first one: nothing was written.
    check(route_line(detail::vae_coreml_plan_route(vae, a)) == route_line(ra) && bundles_under(empty) == 0,
          "planning writes nothing to the cache");
    if (!models.empty()) {
        FitOptions fa;
        fa.models_dir = models;
        fa.vae_model_path = vae;
        fa.coreml = a;
        FitOptions fb = fa;
        fb.coreml = b;
        const FitResult x = fit_params(fa);
        const FitResult y = fit_params(fb);
        std::fprintf(stderr, "  fit %s\n", fit_line(x).c_str());
        check(x.status != FitStatus::Error && fit_line(x) == fit_line(y), "fit: empty cache == populated cache");
        if (ra.route == "generate") {
            check(x.coreml.route == "generate" && x.coreml.bytes == ra.envelope_bytes,
                  "fit charges the generated envelope (" + std::to_string(x.coreml.bytes) + " bytes)");
        }
    }
    std::error_code ec;
    fs::remove_all(empty, ec);
    fs::remove_all(full, ec);
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

// An encoding (ensure_vae(true)) requests no decoder: understand() VAE-encodes
// its audio and tokenizes it; when the LM stage starts, the decoder slot is
// still idle and nothing was generated into the cache.
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
    std::string at_lm;
    bool encoded = false;
    engine->understand(up, [&](const std::string & stage, int, int) {
        if (stage == "source") encoded = true;
        if (stage == "understand" && at_lm.empty()) {
            at_lm = engine->coreml_status();
            return false;  // the encoding is what this case checks
        }
        return true;
    });
    std::fprintf(stderr, "  status after the encoding: %s\n", head(at_lm).c_str());
    check(encoded && state_of(at_lm) == "idle" && !has_hint(at_lm),
          "after ensure_vae(true) + encode + free_vae: decoder slot idle, no hint (" + state_of(at_lm) + ")");
    std::this_thread::sleep_for(std::chrono::seconds(1));
    const std::string after = engine->coreml_status();
    check(state_of(after) == "idle" && bundles_under(cache) == 0,
          "nothing was generated into the cache (" + std::to_string(bundles_under(cache)) + " bundle(s))");
}

// Three generate() calls in low-memory mode on a fresh cache.
void case_lowmem(const std::string & models, const std::string & root, float duration) {
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
    std::string models, vae, cache_root, memory_dir, cases = "encode,lowmem,prewarm,evict,degrade";
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
            if (contains(cases, "lowmem")) case_lowmem(models, cache_root, duration);
            if (contains(cases, "prewarm")) case_prewarm(models, cache_root, duration);
            if (contains(cases, "evict")) case_evict(vae, cache_root);
            if (contains(cases, "degrade")) case_degrade(models, cache_root);
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
