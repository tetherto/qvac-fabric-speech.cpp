// The ACE-Step VAE decoder on the shared Core ML runtime: low-memory
// semantics, construction, eviction, fit and the memory schedule.
//
//   test-acestep-vae-coreml --self-test
//       Model-free checks (every build): the decode deadline, label merging,
//       option validation, the per-phase Core ML charge and the Core ML fit
//       verdict with memory-degrade.
//   test-acestep-vae-coreml --fit-stable --vae GGUF [--models DIR]
//       [--require-coreml | --host-route]
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
//       --cache-root (LIST: encode,lowmem,cached,cancel,repaint,prewarm,
//       prewarm-async,prewarm-call,evict,short,degrade; default all):
//       encode   an encoding (ensure_vae(true)) starts no decoder job;
//       lowmem   three generate() calls in low-memory mode: the first decodes
//                on ggml while the cold load completes after its eviction (no
//                repopulation, hint written), the second and third reload warm
//                within their deadlines and decode on Core ML;
//       cached   a second process on the primed cache (the lowmem case's, or
//                one primed by install-time preparation when lowmem is not
//                selected): its first decode, with the default wait
//                settings, adopts the cached artifact and decodes on Core ML;
//       cancel   cancelled VAE phases (ggml decode with the decoder pending,
//                Core ML decode, an acquisition waiting for the loading
//                decoder, also strict) end at the stage boundary with the slot
//                idle and never start a ggml decode they were not asked for;
//       repaint  (after lowmem) a warm repaint that materialises its source
//                evicts the decoder before the materialisation's encode;
//       prewarm  {prewarm all, wait construct} succeeds with no decoder
//                retained, and the next generate() decodes on Core ML;
//       prewarm-async  {prewarm all, wait none} succeeds in the background
//                and the decoder is released without any generate(); an
//                Engine destroyed while that release is held returns at once
//                and the release completes after it;
//       prewarm-call  Engine::coreml_prewarm on an idle Engine: scope all
//                succeeds and releases the decoder (waited and not), scope
//                none is an empty group;
//       evict    an eviction while a job is unstarted cancels it;
//       short    a 32-frame latent on a ready slot decodes on ggml, not
//                pending, without acquiring the decoder (strict: fails);
//       degrade  memory-degrade near the budget, in fit and in the Engine.
//   test-acestep-vae-coreml --vae GGUF_A --source-swap GGUF_B
//       The VAE GGUF replaced in place after the decoder's slot was made: a
//       ggml VAE loaded from the other file decodes on ggml without
//       requesting the decoder (strict: fails); one loaded after the original
//       file is renamed back serves it again. Needs no device.
//   test-acestep-vae-coreml --vae GGUF --short-latent
//       A 32-frame latent (shorter than the decoder's window) on a cold slot
//       decodes on ggml, not pending, without requesting the decoder
//       (strict: fails). Needs no device.
//   test-acestep-vae-coreml --vae GGUF --memory-schedule DIR [--late]
//       The overlapped memory schedule: ggml decodes run on this thread while
//       the decoder is generated, compiled, loaded and warmed into DIR; each
//       phase is held to the template constant that charges it. --late
//       holds the load after its warm-up, evicts the decoder, then lets the
//       load publish (the late-completion schedule); every load, warm-up and
//       post-eviction sample over the warmed ggml-only baseline is held to
//       the transient charge fit applies outside the VAE decode phase. A
//       failed footprint or compiler query, no compiler reading during the
//       load, or a ggml decode without the whole PCM fails the run.
//
// Without --require-coreml a route other than `generate` skips the device
// cases (exit 77); with it, that fails. --host-route (fit-stable) expects the
// route this build and host support: `generate` with the generator built on
// macOS 15 or later, `none` otherwise.

#include "audiogen-cpp/acestep/engine.h"
#include "audiogen-cpp/acestep/fit.h"
#include "audiogen-cpp/acestep/vae.h"

#include "fit_coreml.h"
#include "music_cli_coreml.h"
#include "vae_access.h"
#include "vae_coreml.h"
#include "vae_coreml_path.h"
#include "vae_gguf.h"
#include "memory_probe.h"

#ifdef AUDIOGEN_USE_COREML
#include "hints.h"
#include "qvac_coreml/generation.h"
#include "qvac_coreml/manager.h"
#endif

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
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

// Whether the comma-separated `list` names `name` exactly.
bool has_case(const std::string & list, const std::string & name) {
    size_t at = 0;
    while (at <= list.size()) {
        const size_t end = std::min(list.find(',', at), list.size());
        if (list.compare(at, end - at, name) == 0) return true;
        at = end + 1;
    }
    return false;
}

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

// Holds the decoder load's completion (after its warm-up, before the runtime
// publishes it) until release(), so a late-completion case orders the
// completion after its eviction instead of hoping a cold load is slow enough.
struct completion_barrier {
    std::mutex mu;
    std::condition_variable cv;
    bool entered = false, released = false;

    static completion_barrier & get() {
        static completion_barrier b;
        return b;
    }
    static void hold() {
        completion_barrier & b = get();
        std::unique_lock<std::mutex> lk(b.mu);
        b.entered = true;
        b.cv.notify_all();
        b.cv.wait(lk, [&] { return b.released; });
    }
    static void arm() {
        completion_barrier & b = get();
        {
            std::lock_guard<std::mutex> lk(b.mu);
            b.entered = b.released = false;
        }
        detail::vae_coreml_set_completion_hook(&completion_barrier::hold);
    }
    static bool is_entered() {
        completion_barrier & b = get();
        std::lock_guard<std::mutex> lk(b.mu);
        return b.entered;
    }
    static bool wait_entered(std::chrono::seconds timeout) {
        completion_barrier & b = get();
        std::unique_lock<std::mutex> lk(b.mu);
        return b.cv.wait_for(lk, timeout, [&] { return b.entered; });
    }
    static void release() {
        detail::vae_coreml_set_completion_hook(nullptr);
        completion_barrier & b = get();
        std::lock_guard<std::mutex> lk(b.mu);
        b.released = true;
        b.cv.notify_all();
    }
};

// Holds the release of an un-waited group's decoder (vae_coreml_set_release_hook)
// until the test releases it, and records that it ran.
struct release_barrier {
    std::mutex mu;
    std::condition_variable cv;
    bool entered = false, released = false, exited = false;

    static release_barrier & get() {
        static release_barrier b;
        return b;
    }
    static void hold() {
        release_barrier & b = get();
        std::unique_lock<std::mutex> lk(b.mu);
        b.entered = true;
        b.cv.notify_all();
        b.cv.wait(lk, [&] { return b.released; });
        b.exited = true;
        b.cv.notify_all();
    }
    static void arm() {
        release_barrier & b = get();
        {
            std::lock_guard<std::mutex> lk(b.mu);
            b.entered = b.released = b.exited = false;
        }
        detail::vae_coreml_set_release_hook(&release_barrier::hold);
    }
    static bool wait_entered(std::chrono::seconds timeout) {
        release_barrier & b = get();
        std::unique_lock<std::mutex> lk(b.mu);
        return b.cv.wait_for(lk, timeout, [&] { return b.entered; });
    }
    static bool is_exited() {
        release_barrier & b = get();
        std::lock_guard<std::mutex> lk(b.mu);
        return b.exited;
    }
    static bool wait_exited(std::chrono::seconds timeout) {
        release_barrier & b = get();
        std::unique_lock<std::mutex> lk(b.mu);
        return b.cv.wait_for(lk, timeout, [&] { return b.exited; });
    }
    static void release() {
        detail::vae_coreml_set_release_hook(nullptr);
        release_barrier & b = get();
        std::lock_guard<std::mutex> lk(b.mu);
        b.released = true;
        b.cv.notify_all();
    }
};

// ---- model-free checks -------------------------------------------------------

float eligible_duration(int64_t warm_load_ms, float at_least);

// The ggml VAE reads its weights as BF16 while the Core ML decoder's recipe
// converts by the GGUF type, so a VAE file with a tensor of any other type is
// refused at load: otherwise a decode before the decoder is ready (ggml) and
// one after (Core ML) would run different weights.
void bf16_only_self_test() {
    for (const ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_BF16}) {
        const fs::path path = fs::temp_directory_path() / ("acestep-vae-type-" + std::to_string((int) type) + ".gguf");
        ggml_init_params ip{16 * 1024 * 1024, nullptr, /*no_alloc=*/false};
        ggml_context * ctx = ggml_init(ip);
        gguf_context * g = gguf_init_empty();
        ggml_tensor * bias = ggml_new_tensor_1d(ctx, type, 2048);
        ggml_set_name(bias, "decoder.conv1.bias");
        std::memset(bias->data, 0, ggml_nbytes(bias));
        ggml_tensor * g_w = ggml_new_tensor_3d(ctx, GGML_TYPE_BF16, 1, 1, 2048);
        ggml_set_name(g_w, "decoder.conv1.weight_g");
        std::memset(g_w->data, 0, ggml_nbytes(g_w));
        gguf_add_tensor(g, g_w);
        gguf_add_tensor(g, bias);
        const bool written = gguf_write_to_file(g, path.string().c_str(), /*only_meta=*/false);
        gguf_free(g);
        ggml_free(ctx);
        check(written, std::string("write a VAE GGUF with a ") + ggml_type_name(type) + " bias");
        if (type == GGML_TYPE_BF16) {
            // Not loaded: the file holds two tensors of the decoder, so only
            // the type check is asked here.
            VaeGGUF vg;
            const bool opened = vae_gguf_open(vg, path.string());
            check(opened && vae_gguf_type_refusal(vg, path.string()).empty(), "an all-BF16 file passes the type check");
            vae_gguf_close(vg);
        } else {
            std::string err;
            VaeOptions vo;
            vo.backend = "cpu";
            try {
                Vae::load(path.string(), vo);
            } catch (const std::exception & e) {
                err = e.what();
            }
            const std::string want = std::string("decoder.conv1.bias is ") + ggml_type_name(type) +
                                     "; the VAE loader reads BF16 weights only";
            check(contains(err, want), std::string("a ") + ggml_type_name(type) + " tensor refuses the load: " + err);
        }
        std::error_code ec;
        fs::remove(path, ec);
    }
}

// With no usable route, install-time preparation still observes a normal
// result: an empty, successful construction group with nothing to prepare and
// the decoder under skipped_route_none. Without the runtime (Core ML off) the
// route is unsupported(build); with it, use = off gives disabled. No file is
// read either way.
void route_none_group_self_test() {
    CoremlOptions o;
    o.prewarm = CoremlOptions::Prewarm::all;
    o.wait = CoremlOptions::Wait::construct;
#ifdef AUDIOGEN_USE_COREML
    o.use = CoremlOptions::Use::off;
    const std::string want_state = "disabled";
#else
    const std::string want_state = "unsupported(build)";
#endif
    std::shared_ptr<detail::vae_coreml_stage> stage =
        detail::vae_coreml_stage::create("/nonexistent/vae-BF16.gguf", o, false, true);
    const std::string group = stage ? stage->construction_json() : std::string("(no stage)");
    const std::string st = stage ? stage->status_json() : std::string("(no stage)");
    std::fprintf(stderr, "  route none: %s\n  status: %s\n", head(group, 300).c_str(), head(st, 300).c_str());
    check(stage && !stage->active() && contains(group, "\"success\":true") &&
              contains(group, "\"nothing_to_prepare\":true") && contains(group, "\"skipped_route_none\":[{") &&
              contains(group, "\"stage\":\"vae\"") && state_of(group) == want_state,
          "route none: an empty successful construction group with the decoder's status (" + want_state + ")");
    check(state_of(st) == want_state && contains(st, "\"engine\":\"acestep\""),
          "route none: the status reports the decoder " + want_state);
    const std::string again = stage ? stage->prewarm(CoremlOptions::Prewarm::all, true, true) : std::string();
    check(contains(again, "\"group\":\"prewarm\"") && contains(again, "\"success\":true") &&
              contains(again, "\"nothing_to_prepare\":true") && state_of(again) == want_state,
          "route none: a prewarm group is empty and successful too (" + head(again, 120) + ")");
}

void self_test() {
    std::fprintf(stderr, "self-test\n");
    route_none_group_self_test();
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
    {
        using music_cli::coreml_cache_dir;
        const std::string def = std::string("/Users/u") + music_cli::COREML_CLI_CACHE_SUBDIR;
        check(coreml_cache_dir("/x", "/env", "/Users/u", true) == std::optional<std::string>("/x") &&
                  !coreml_cache_dir(nullptr, "/env", "/Users/u", true) &&
                  coreml_cache_dir(nullptr, nullptr, "/Users/u", true) == std::optional<std::string>(def) &&
                  coreml_cache_dir(nullptr, "", "/Users/u", true) == std::optional<std::string>(def) &&
                  !coreml_cache_dir(nullptr, nullptr, "/Users/u", false) && !coreml_cache_dir(nullptr, nullptr, "", true),
              "music-cli cache directory: the flag, then QVAC_COREML_CACHE_DIR, then the macOS default " + def);
    }
    check(eligible_duration(1000, 20.0f) == 40.0f && eligible_duration(100, 20.0f) == 20.0f,
          "a fresh process's eligible decode: 3 ms per frame covers twice the 1.5 x warm-load budget");
    check(has_case("lowmem,prewarm-async", "prewarm-async") && !has_case("lowmem,prewarm-async", "prewarm") &&
              has_case("prewarm", "prewarm") && !has_case("", "prewarm"),
          "device case names match exactly");

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
void fit_stable(const std::string & vae, const std::string & models, const std::string & root, bool require,
                bool host_route) {
    std::fprintf(stderr, "fit stability (%s)\n", vae.c_str());
    const fs::path empty = fresh_dir(root, "fit-empty");
    const fs::path cache = fresh_dir(root, "fit-generated");
    CoremlOptions a;
    a.cache_dir = empty.string();
    const detail::vae_coreml_route ra = detail::vae_coreml_plan_route(vae, a);
    const std::string got = ra.route + " " + ra.state + " " + ra.reason;
#ifdef AUDIOGEN_USE_COREML
    if (require) {
        check(ra.route == "generate", "route generate (" + got + ")");
    } else if (host_route) {
        // The route this build and host support: generation needs the
        // generator built in and macOS 15 or later; otherwise (a prebuilt-only
        // build, an older macOS) the decoder plans no generated route.
        const auto m = rt::Manager::instance();
        const bool capable = rt::generation::built() && m->platform() == rt::Platform::macos &&
                             rt::os_at_least(m->os_version(), 15, 0);
        if (capable) check(ra.route == "generate", "generation built, macOS 15 or later: route generate (" + got + ")");
        else check(ra.route == "none", "no generation on this build or OS: route none (" + got + ")");
    }
#else
    (void) require;
    (void) host_route;
    (void) got;
#endif
    if (ra.route == "generate") {
        // The template's measured charges (template.json memory), pinned so a
        // change to them is deliberate.
        constexpr uint64_t RESIDENT = 1119879168, TRANSIENT = 1438646272, RESERVE = 959447040;
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

// Route none (generation off, nothing prebuilt; with prewarm all and wait
// construct, which must not block) still issues the construction group: an
// empty one that succeeds with nothing to prepare and lists the decoder under
// skipped_route_none.
void route_none_construction(const std::string & vae, const std::string & root) {
#ifdef AUDIOGEN_USE_COREML
    std::fprintf(stderr, "route none: the construction group\n");
    const fs::path cache = fresh_dir(root, "route-none");
    for (const bool generate_off : {true, false}) {
        CoremlOptions o;
        o.cache_dir = cache.string();
        o.prewarm = CoremlOptions::Prewarm::all;
        o.wait = CoremlOptions::Wait::construct;
        if (generate_off) o.generate = CoremlOptions::Generate::never;
        else o.use = CoremlOptions::Use::off;
        auto stage = detail::vae_coreml_stage::create(vae, o, false, true);
        const std::string group = stage ? stage->construction_json() : std::string("(no stage)");
        const std::string what = generate_off ? "generate never" : "use off";
        std::fprintf(stderr, "  %s: %s\n", what.c_str(), head(group, 300).c_str());
        check(stage && !stage->active() && contains(group, "\"success\":true") &&
                  contains(group, "\"nothing_to_prepare\":true") && contains(group, "\"skipped_route_none\":[{") &&
                  contains(group, "\"stage\":\"vae\""),
              what + ": route none issues an empty construction group with the decoder's status");
    }
    std::error_code ec;
    fs::remove_all(cache, ec);
#else
    (void) vae;
    (void) root;
#endif
}

// Links (or copies) `from` to `to` through a temporary name and renames it
// over `to`, so `to` names the other file in one step, as a replacement in
// place would.
void replace_with(const fs::path & from, const fs::path & to) {
    const fs::path tmp = to.string() + ".swap";
    std::error_code ec;
    fs::remove(tmp, ec);
    fs::create_hard_link(from, tmp, ec);
    if (ec) fs::copy_file(from, tmp, ec);
    if (!ec) fs::rename(tmp, to, ec);
    check(!ec, "replace " + to.string() + " with " + from.string() + (ec ? ": " + ec.message() : std::string()));
}

// The VAE GGUF replaced under a live decoder slot: the slot was made for file
// A; a ggml VAE (re)loaded after `path` names twin B must decode on ggml
// without requesting the decoder (strict: fail), and one loaded after A is
// renamed back serves it again. Requests nothing from Core ML, so it runs
// without a device; any request it made would show as a non-idle slot.
void source_swap(const std::string & vae_a, const std::string & vae_b, const std::string & root) {
    std::fprintf(stderr, "source identity: the VAE GGUF replaced after the slot was made\n");
    const fs::path dir = fresh_dir(root, "source-swap");
    const fs::path a = dir / "a.gguf", b = dir / "b.gguf";
    std::error_code ec;
    fs::create_hard_link(vae_a, a, ec);
    if (ec) fs::copy_file(vae_a, a, ec);
    fs::create_hard_link(vae_b, b, ec);
    if (ec) fs::copy_file(vae_b, b, ec);
    VaeOptions vo;
    vo.n_gpu_layers = 0;  // CPU: nothing here needs a device
    const int T = 64;
    const std::vector<float> latent((size_t) T * 64, 0.0f);
    const size_t expected = (size_t) T * 1920 * 2;
    for (const bool strict : {false, true}) {
        const fs::path path = dir / (strict ? "strict.gguf" : "vae.gguf");
        replace_with(a, path);
        CoremlOptions o;
        o.cache_dir = (dir / "cache").string();
        if (strict) o.use = CoremlOptions::Use::require;
        std::shared_ptr<detail::vae_coreml_stage> stage;
        try {
            stage = detail::vae_coreml_stage::create(path.string(), o, false, true);
        } catch (const std::exception & e) {
            std::fprintf(stderr, "  stage: %s\n", e.what());
        }
        if (!stage || stage->route().route != "generate") {
            std::fprintf(stderr, "  route %s: case skipped\n", stage ? stage->route().route.c_str() : "none");
            break;
        }
        const std::string mode = strict ? "strict: " : "";
        auto loaded_a = detail::VaeAccess::load(path.string(), vo, stage);
        check(detail::VaeAccess::coreml_source_ok(*loaded_a), mode + "a VAE loaded from the slot's file serves it");
        replace_with(b, path);
        auto loaded_b = detail::VaeAccess::load(path.string(), vo, stage);
        check(!detail::VaeAccess::coreml_source_ok(*loaded_b), mode + "a VAE loaded after the replacement does not");
        const std::vector<float> pcm = loaded_b->decode(latent, T);
        const std::string st = state_of(stage->status_json());
        std::fprintf(stderr, "  %sdecode after the replacement: %zu samples, backend %s, pending %d, slot %s\n",
                     mode.c_str(), pcm.size(), loaded_b->last_decode_backend().c_str(),
                     (int) loaded_b->last_decode_coreml_pending(), st.c_str());
        if (strict) {
            check(pcm.empty(), "strict: the replaced VAE's decode fails");
        } else {
            check(pcm.size() == expected && loaded_b->last_decode_backend() == "ggml" &&
                      !loaded_b->last_decode_coreml_pending(),
                  "the replaced VAE decodes on ggml, not pending");
        }
        check(st == "idle", mode + "the decoder was never requested (slot idle)");
        // A renamed back (the same file): its identity again.
        replace_with(a, path);
        auto reloaded = detail::VaeAccess::load(path.string(), vo, stage);
        check(detail::VaeAccess::coreml_source_ok(*reloaded), mode + "a VAE loaded after A is restored serves it");
    }
    fs::remove_all(dir, ec);
}

// A latent shorter than the decoder's 64-frame window decodes on ggml and is
// not pending: on a cold (idle) slot without requesting the decoder, and, with
// `ready`, on a slot whose decoder is loaded (a device run); strict fails it.
void short_latent(const std::string & vae, const std::string & root, bool ready) {
    std::fprintf(stderr, "short latent: 32 frames on a %s slot\n", ready ? "ready" : "cold");
    const fs::path cache = fresh_dir(root, "short-latent");
    VaeOptions vo;
    vo.n_gpu_layers = ready ? 99 : 0;
    const int T = 32;
    const std::vector<float> latent = make_latent(T, 0x51u);
    for (const bool strict : {false, true}) {
        CoremlOptions o;
        o.cache_dir = cache.string();
        if (strict) o.use = CoremlOptions::Use::require;
        if (ready) {
            o.prewarm = CoremlOptions::Prewarm::all;
            o.wait = CoremlOptions::Wait::construct;
        }
        std::shared_ptr<detail::vae_coreml_stage> stage;
        try {
            stage = detail::vae_coreml_stage::create(vae, o, false, /*release_after_construction=*/false);
        } catch (const std::exception & e) {
            std::fprintf(stderr, "  stage: %s\n", e.what());
        }
        if (!stage || stage->route().route != "generate") {
            std::fprintf(stderr, "  route %s: case skipped\n", stage ? stage->route().route.c_str() : "none");
            break;
        }
        const std::string mode = strict ? "strict: " : "";
        const std::string before = state_of(stage->status_json());
        check(before == (ready ? "ready" : "idle"), mode + "the slot is " + (ready ? "ready" : "idle") + " (" + before + ")");
        auto v = detail::VaeAccess::load(vae, vo, stage);
        const std::vector<float> pcm = v->decode(latent, T);
        const std::string after = state_of(stage->status_json());
        std::fprintf(stderr, "  %s%zu samples, backend %s, pending %d, slot %s\n", mode.c_str(), pcm.size(),
                     v->last_decode_backend().c_str(), (int) v->last_decode_coreml_pending(), after.c_str());
        if (strict) {
            check(pcm.empty(), "strict: a latent shorter than the window fails");
        } else {
            check(pcm.size() == (size_t) T * 1920 * 2 && v->last_decode_backend() == "ggml",
                  "a latent shorter than the window decodes on ggml");
        }
        check(!v->last_decode_coreml_pending(), mode + "and is not pending");
        check(after == before, mode + "without requesting the decoder (slot " + after + ")");
    }
    std::error_code ec;
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

    // Gen 1: the decode runs on ggml while the cold load proceeds; the load's
    // completion is held at its end (after the warm-up, before it publishes)
    // until free_vae() has evicted the slot, then released: it must write its
    // hint and never repopulate the slot.
    completion_barrier::arm();
    std::string at_vae;
    bool held = false, asked = false;
    // The engine reports ("vae", 0, 1) before the decode; the decode's own
    // progress (the ggml graph's node counts) comes after its acquisition.
    const GenerateResult g1 = engine->generate(short_params(duration), [&](const std::string & stage, int step, int) {
        if (stage == "vae" && step > 0 && !asked) {
            asked = true;
            held = completion_barrier::wait_entered(std::chrono::seconds(300));
            at_vae = engine->coreml_status();
        }
        return true;
    });
    const std::string after1 = engine->coreml_status();
    std::fprintf(stderr, "  gen 1: %s pending=%d, %zu samples; slot while held %s; after: %s\n", g1.vae_backend.c_str(),
                 (int) g1.coreml_pending, g1.pcm.size(), state_of(at_vae).c_str(), head(after1).c_str());
    check(held && state_of(at_vae) == "warming",
          "gen 1: the cold load reached its completion during the ggml decode and is held there (" +
              state_of(at_vae) + ")");
    check(!g1.pcm.empty() && g1.vae_backend == "ggml" && g1.coreml_pending,
          "gen 1 decodes on ggml, coreml_pending (" + g1.vae_backend + ")");
    check(state_of(after1) == "idle" && !has_hint(after1),
          "gen 1: free_vae() evicted the slot before the load completed (" + state_of(after1) +
              (has_hint(after1) ? ", hint already written" : ", no hint yet") + ")");
    // Released after the eviction: the load completes, writes the hint and
    // never repopulates the slot.
    completion_barrier::release();
    bool repopulated = false;
    std::string last;
    const bool hinted = poll_until([&] { return engine->coreml_status(); },
                                   [&](const std::string & s) {
                                       if (state_of(s) != "idle") repopulated = true;
                                       return has_hint(s);
                                   },
                                   std::chrono::seconds(300), &last);
    for (int i = 0; i < 20; ++i) {
        if (state_of(engine->coreml_status()) != "idle") repopulated = true;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
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

// A cache primed for the cases that need one when the lowmem case did not run:
// install-time preparation ({prewarm all, wait construct}) leaves the
// generated artifact and its warm hint. "" (and a failed check) when it did
// not succeed.
std::string prime_cache(const std::string & models, const std::string & root) {
    std::fprintf(stderr, "priming a cache (the lowmem case did not run)\n");
    const std::string cache = fresh_dir(root, "primed");
    EngineOptions o = engine_options(models, cache);
    o.coreml.prewarm = CoremlOptions::Prewarm::all;
    o.coreml.wait = CoremlOptions::Wait::construct;
    std::unique_ptr<Engine> engine = Engine::create(o);
    const std::string group = engine->coreml_construction();
    const std::string st = engine->coreml_status();
    const bool ok = contains(group, "\"success\":true") && contains(group, "\"nothing_to_prepare\":false") &&
                    has_hint(st);
    check(ok, "the cache is primed (artifact and warm hint): " + head(group, 200));
    return ok ? cache : std::string();
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
                 (int) r.coreml_pending, st.c_str());
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

// The first decode the bounded wait serves: its deadline (3 ms per latent
// frame, 25 frames per second) must cover the wait's budget (1.5 x the hinted
// warm load) twice over, because a new process's first load also starts Core
// ML and its Neural Engine client (on the M2 about 1.7 s against 0.9-1.0 s
// for the in-process reloads that trained the hint).
float eligible_duration(int64_t warm_load_ms, float at_least) {
    const double frames = 2.0 * 1.5 * (double) warm_load_ms / detail::VAE_GGML_DECODE_MS_PER_FRAME;
    return std::max(at_least, (float) std::ceil(frames / 25.0));
}

void case_cached(const std::string & self, const std::string & models, const std::string & cache, float duration) {
    std::fprintf(stderr, "cached: a second process's first decode on a primed cache\n");
    int64_t w = 0;
#ifdef AUDIOGEN_USE_COREML
    CoremlOptions o;
    o.cache_dir = cache;
    const std::string units = detail::vae_coreml_plan_route(models + "/vae-BF16.gguf", o).compute_units;
    for (const fs::path & art : artifact_dirs(cache)) {
        if (auto h = rt::detail::read_hint(runtime_hint_path(cache, art, units)); h && h->warm_load_ms) {
            w = std::max(w, *h->warm_load_ms);
        }
    }
#endif
    check(w > 0, "the primed cache holds a warm hint (" + std::to_string(w) + " ms)");
    const float d = eligible_duration(w, duration);
    std::fprintf(stderr, "  hinted warm load %lld ms: the child generates %.0f s\n", (long long) w, d);
    const int rc = run_child({self, "--models", models, "--first-decode", cache, "--duration", std::to_string(d)});
    check(rc == 0, "the child process's first decode ran on Core ML (exit " + std::to_string(rc) + ")");
}

// Cancellation never bypasses the low-memory stage boundary and never starts a
// ggml decode it was not asked for:
//  - a ggml decode (the decoder pending) cancelled from its progress: the
//    slot is evicted when generate() returns;
//  - a Core ML decode (the decoder ready, on the primed cache) cancelled after
//    its first window: the decoder is released when generate() returns;
//  - an acquisition that waits for a loading decoder (wait call), cancelled
//    with Engine::cancel(): generate() returns without any ggml decode, also
//    under use = require, and the slot is idle afterwards.
void case_cancel(const std::string & models, const std::string & root, const std::string & primed, float duration) {
    std::fprintf(stderr, "cancel: cancelled VAE phases end at the stage boundary\n");
    {
        const std::string cache = fresh_dir(root, "cancel-pending");
        std::unique_ptr<Engine> engine = Engine::create(engine_options(models, cache));
        std::string during;
        const GenerateResult r = engine->generate(short_params(duration), [&](const std::string & stage, int step, int) {
            if (stage == "vae" && step > 0) {
                during = engine->coreml_status();
                return false;
            }
            return true;
        });
        const std::string after = engine->coreml_status();
        std::fprintf(stderr, "  pending decode: slot during %s, after %s\n", state_of(during).c_str(),
                     state_of(after).c_str());
        check(r.pcm.empty() && !during.empty() && state_of(during) != "idle" && state_of(after) == "idle",
              "a cancelled ggml decode with the decoder pending: the slot is evicted when generate() returns (" +
                  state_of(during) + " -> " + state_of(after) + ")");
    }
    if (!primed.empty()) {
        EngineOptions o = engine_options(models, primed);
        o.coreml.wait = CoremlOptions::Wait::call;
        std::unique_ptr<Engine> engine = Engine::create(o);
        std::string during;
        const GenerateResult r = engine->generate(short_params(duration), [&](const std::string & stage, int step, int) {
            if (stage == "vae" && step > 0) {
                during = engine->coreml_status();
                return false;
            }
            return true;
        });
        const std::string after = engine->coreml_status();
        std::fprintf(stderr, "  ready decode: slot during %s, after %s\n", state_of(during).c_str(),
                     state_of(after).c_str());
        check(r.pcm.empty() && state_of(during) == "ready" && state_of(after) == "idle",
              "a cancelled Core ML decode: the decoder is released when generate() returns (" + state_of(during) +
                  " -> " + state_of(after) + ")");
    } else {
        std::fprintf(stderr, "  ready decode: needs the lowmem case's primed cache, not run\n");
    }
    const std::string cache = fresh_dir(root, "cancel-loading");
    for (const bool strict : {false, true}) {
        EngineOptions o = engine_options(models, cache);
        o.coreml.wait = CoremlOptions::Wait::call;
        if (strict) o.coreml.use = CoremlOptions::Use::require;
        std::unique_ptr<Engine> engine = Engine::create(o);
        std::atomic<bool> done{false};
        std::atomic<int> vae_steps{0};
        std::string at_cancel;
        std::thread canceller([&] {
            poll_until([&] { return engine->coreml_status(); },
                       [&](const std::string & st) {
                           const std::string s = state_of(st);
                           return done.load() || (s != "idle" && s != "ready");
                       },
                       std::chrono::seconds(600), &at_cancel);
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            engine->cancel();
        });
        GenerateResult r;
        bool threw = false;
        Clock::time_point returned;
        try {
            r = engine->generate(short_params(duration), [&](const std::string & stage, int step, int) {
                if (stage == "vae" && step > 0) ++vae_steps;
                return true;
            });
        } catch (const std::exception & e) {
            threw = true;
            std::fprintf(stderr, "  generate threw: %s\n", e.what());
        }
        returned = Clock::now();
        done.store(true);
        canceller.join();
        const std::string after = engine->coreml_status();
        std::fprintf(stderr, "  %s acquisition: slot at cancel %s, after %s, %d VAE progress report(s)\n",
                     strict ? "strict" : "waiting", state_of(at_cancel).c_str(), state_of(after).c_str(),
                     vae_steps.load());
        (void) returned;
        check(!threw && r.pcm.empty() && vae_steps.load() == 0 && state_of(after) == "idle",
              std::string(strict ? "use = require: " : "wait call: ") +
                  "cancelling an acquisition that waits for the loading decoder runs no ggml decode and leaves "
                  "the slot idle (" + state_of(at_cancel) + ")");
    }
}

// A warm repaint that materialises its source (a flow edit, then a repaint)
// decodes on the ready Core ML decoder and then encodes the result with the
// ggml VAE in the same phase. Fit charges that encode without the resident
// decoder, so the decoder is evicted at the decode -> encode boundary: the
// first progress report of the encode finds the slot idle.
void case_repaint(const std::string & models, const std::string & primed) {
    std::fprintf(stderr, "repaint: the decoder is evicted before the materialisation's encode\n");
    EngineOptions o = engine_options(models, primed);
    o.coreml.wait = CoremlOptions::Wait::call;
    std::unique_ptr<Engine> engine = Engine::create(o);
    GenerateParams p;
    p.caption = "Replace the middle with bright synth guitar";
    p.lyrics = AUDIO_EDIT_DEFAULT_LYRICS;
    const int seconds = 10, frames = 48000 * seconds;
    p.source_audio.resize((size_t) frames * 2);
    for (int i = 0; i < frames; ++i) {
        const float v = 0.3f * std::sin(2.0f * 3.14159265f * 220.0f * (float) i / 48000.0f);
        p.source_audio[(size_t) 2 * i] = v;
        p.source_audio[(size_t) 2 * i + 1] = v;
    }
    p.inference_steps = 2;
    p.shift = 3.0f;
    p.seed = 22886;
    p.lm_phase1 = false;
    FlowEditParams flow;
    flow.source_caption = "A plain sine tone";
    flow.source_lyrics = AUDIO_EDIT_DEFAULT_LYRICS;
    flow.target_caption = "A warm evolving synth pad";
    flow.target_lyrics = AUDIO_EDIT_DEFAULT_LYRICS;
    flow.n_max = 0.5f;
    RepaintParams repaint;
    repaint.start_seconds = 3.0f;
    repaint.end_seconds = 6.0f;
    repaint.mode = RepaintMode::Balanced;
    p.edit_plan.emplace_back(flow);
    p.edit_plan.emplace_back(repaint);
    struct report { int done, total; std::string state; };
    std::vector<report> vae;
    const GenerateResult r = engine->generate(p, [&](const std::string & stage, int done, int total) {
        if (stage == "vae") vae.push_back({done, total, state_of(engine->coreml_status())});
        return true;
    });
    // The materialisation's decode: the reports up to the first done == total
    // (> 0); the next report is the encode's first.
    size_t end = 0;
    while (end < vae.size() && !(vae[end].done > 0 && vae[end].done == vae[end].total)) ++end;
    const bool decoded_ready = end < vae.size() && vae[end].state == "ready";
    const std::string at_encode = end + 1 < vae.size() ? vae[end + 1].state : std::string("none");
    std::fprintf(stderr, "  %zu vae reports; materialisation decode ends at %zu (%s); encode starts with the slot %s; "
                         "result %s, %zu samples\n",
                 vae.size(), end, end < vae.size() ? vae[end].state.c_str() : "-", at_encode.c_str(),
                 r.vae_backend.c_str(), r.pcm.size());
    check(!r.pcm.empty() && decoded_ready, "the materialisation decoded on the ready Core ML decoder");
    check(at_encode == "idle", "the decoder was evicted before the materialisation's encode (" + at_encode + ")");
    check(state_of(engine->coreml_status()) == "idle", "idle after the repaint");
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

// Asynchronous install-time preparation: {prewarm all, wait none} in
// low-memory mode on a fresh cache. create() returns at once; the group
// generates, loads and warms the decoder in the background, reports success,
// and the decoder is released without any generate().
void case_prewarm_async(const std::string & models, const std::string & root) {
    std::fprintf(stderr, "prewarm-async: {prewarm all, wait none}\n");
    const std::string cache = fresh_dir(root, "prewarm-async");
    EngineOptions o = engine_options(models, cache);
    o.coreml.prewarm = CoremlOptions::Prewarm::all;
    o.coreml.wait = CoremlOptions::Wait::none;
    std::unique_ptr<Engine> engine = Engine::create(o);
    std::string group;
    const bool finished = poll_until([&] { return engine->coreml_construction(); },
                                     [](const std::string & g) { return contains(g, "\"success\":true"); },
                                     std::chrono::seconds(600), &group);
    std::string last;
    const bool idle = poll_until([&] { return engine->coreml_status(); },
                                 [](const std::string & st) { return state_of(st) == "idle" && has_hint(st); },
                                 std::chrono::seconds(30), &last);
    std::fprintf(stderr, "  group: %s\n  status: %s\n", head(group, 300).c_str(), head(last, 300).c_str());
    check(finished && contains(group, "\"nothing_to_prepare\":false"), "the asynchronous construction group succeeds");
    check(idle, "the decoder is released after the group finished, without generate() (" + state_of(last) + ")");
    engine.reset();

    // The owner disappears while its release is held: the release of an
    // un-waited group runs off the Engine's threads and holds the stage only
    // while it runs, so destroying the Engine returns at once and the
    // release completes on its own afterwards.
    std::fprintf(stderr, "prewarm-async: the Engine destroyed while its release is held\n");
    release_barrier::arm();
    engine = Engine::create(o);  // the cache is primed: a warm load
    const bool entered = release_barrier::wait_entered(std::chrono::seconds(120));
    check(entered, "the release of the finished group started");
    std::atomic<bool> destroyed{false};
    const auto t_destroy = Clock::now();
    std::thread destroyer([&] {
        engine.reset();
        destroyed.store(true);
    });
    while (!destroyed.load() && Clock::now() - t_destroy < std::chrono::seconds(10)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    const double destroy_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_destroy).count();
    const bool returned_while_held = destroyed.load() && !release_barrier::is_exited();
    release_barrier::release();
    destroyer.join();
    const bool exited = release_barrier::wait_exited(std::chrono::seconds(10));
    std::fprintf(stderr, "  destroyed in %.1f ms while the release was held: %d; the release finished after: %d\n",
                 destroy_ms, (int) returned_while_held, (int) exited);
    check(entered && returned_while_held && destroy_ms < 2000.0,
          "destroying the Engine never waits for its release (returned while it was held)");
    check(exited, "the held release completed after its owner was destroyed");
}

// Engine::coreml_prewarm on an existing, idle low-memory Engine: scope all
// prepares, loads and warms the decoder and reports success; the decoder is
// released again (hint kept). Scope none is an empty group. Without wait the
// decoder is released in the background once the group finished.
void case_prewarm_call(const std::string & models, const std::string & root) {
    std::fprintf(stderr, "prewarm-call: Engine::coreml_prewarm on an idle Engine\n");
    const std::string cache = fresh_dir(root, "prewarm-call");
    std::unique_ptr<Engine> engine = Engine::create(engine_options(models, cache));
    const std::string before = engine->coreml_status();
    check(state_of(before) == "idle" && !has_hint(before), "the default Engine requested nothing (" +
                                                               state_of(before) + ")");
    const std::string all = engine->coreml_prewarm(CoremlOptions::Prewarm::all, true);
    const std::string after = engine->coreml_status();
    std::fprintf(stderr, "  all: %s\n  status: %s\n", head(all, 300).c_str(), head(after, 300).c_str());
    check(contains(all, "\"group\":\"prewarm\"") && contains(all, "\"success\":true") &&
              contains(all, "\"nothing_to_prepare\":false"),
          "scope all succeeds with the decoder in the group");
    check(state_of(after) == "idle" && has_hint(after) && !engine->coreml_wait(0),
          "the decoder is released after the waited group (" + state_of(after) + "), hint written");
    const std::string none = engine->coreml_prewarm(CoremlOptions::Prewarm::none, true);
    std::fprintf(stderr, "  none: %s\n", head(none, 300).c_str());
    check(contains(none, "\"success\":true") && contains(none, "\"slots\":[]"), "scope none is an empty group");
    const std::string snapshot = engine->coreml_prewarm(CoremlOptions::Prewarm::all, false);
    std::string last;
    const bool released = poll_until([&] { return engine->coreml_status(); },
                                     [](const std::string & st) { return state_of(st) == "idle"; },
                                     std::chrono::seconds(60), &last);
    std::fprintf(stderr, "  all without wait: %s\n  status: %s\n", head(snapshot, 200).c_str(),
                 head(last, 200).c_str());
    check(contains(snapshot, "\"group\":\"prewarm\"") && released,
          "scope all without wait: the decoder is released once the group finished");
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

    // The caller's decision is the one construction applies. Memory-degrade
    // holds for a budget B with ggml + margin <= B < ggml + Core ML + margin;
    // a workload whose interval differs from the default workload's (60 s,
    // no source audio) has a budget where the two choose different routes.
    struct span { uint64_t ggml, coreml; };
    auto span_of = [&](FitOptions w) {
        w.coreml.memory_budget_bytes.reset();
        FitOptions w_off = w;
        w_off.coreml.use = CoremlOptions::Use::off;
        const FitResult rw = fit_params(w), ro = fit_params(w_off);
        return span{ro.peak_device_bytes + ro.peak_host_bytes + w.margin_bytes,
                     rw.peak_device_bytes + rw.peak_host_bytes + w.margin_bytes};
    };
    FitOptions dflt = f;
    dflt.duration_seconds = FitOptions{}.duration_seconds;
    dflt.with_source_audio = FitOptions{}.with_source_audio;
    const span sd = span_of(dflt);
    FitOptions assessed = f;
    std::optional<int64_t> budget;
    for (const auto & [seconds, source] : std::vector<std::pair<float, bool>>{{240.0f, false}, {240.0f, true},
                                                                            {20.0f, true}, {20.0f, false}}) {
        FitOptions w = f;
        w.duration_seconds = seconds;
        w.with_source_audio = source;
        const span sw = span_of(w);
        uint64_t b = 0;
        if (sw.ggml < sd.ggml) b = sw.ggml;            // degrades for w, ggml-only does not fit for the default
        else if (sw.ggml > sd.ggml) b = sd.ggml;       // the other way round
        else if (sw.coreml != sd.coreml) b = std::min(sw.coreml, sd.coreml);
        if (b == 0 || !(sw.ggml <= b && b < sw.coreml) && !(sd.ggml <= b && b < sd.coreml)) continue;
        assessed = w;
        budget = (int64_t) b;
        break;
    }
    check(budget.has_value(), "a workload whose memory-degrade interval differs from the default workload's");
    if (!budget) return;
    assessed.coreml.memory_budget_bytes = budget;
    dflt.coreml.memory_budget_bytes = budget;
    const FitResult a = fit_params(assessed);
    const FitResult d = fit_params(dflt);
    std::fprintf(stderr, "  budget %.1f MiB: assessed (%.0f s%s) '%s', default workload (%.0f s) '%s'\n",
                 mib((uint64_t) *budget), assessed.duration_seconds, assessed.with_source_audio ? ", source audio" : "",
                 a.coreml.disabled_reason.c_str(), dflt.duration_seconds, d.coreml.disabled_reason.c_str());
    check((a.coreml.disabled_reason == "memory") != (d.coreml.disabled_reason == "memory"),
          "the assessed and the default workload choose different routes at this budget");
    EngineOptions decided = engine_options(models, cache);
    decided.coreml.memory_budget_bytes = budget;
    decided.coreml.memory_degrade = a.coreml.disabled_reason == "memory";
    std::unique_ptr<Engine> e = Engine::create(decided);
    const std::string want = *decided.coreml.memory_degrade ? "disabled" : "idle";
    check(state_of(e->coreml_status()) == want, "Engine with the caller's decision (" +
                                                    std::string(*decided.coreml.memory_degrade ? "degrade" : "keep") +
                                                    "): decoder " + state_of(e->coreml_status()));
    // Without the decision construction assesses the default workload, which
    // chooses the other route here.
    EngineOptions undecided = engine_options(models, cache);
    undecided.coreml.memory_budget_bytes = budget;
    std::unique_ptr<Engine> u = Engine::create(undecided);
    const std::string want_u = d.coreml.disabled_reason == "memory" ? "disabled" : "idle";
    check(state_of(u->coreml_status()) == want_u && want_u != want,
          "Engine without a decision (default workload): decoder " + state_of(u->coreml_status()));
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
// footprint just before the load) and the whole run over the baseline. The
// late tail (every sample from the eviction until the late load released its
// model) is taken over the baseline itself: the warmed ggml-only footprint,
// so it holds everything this process kept above what the ggml work needs
// outside the VAE decode phase, preparation leftovers included, which is what
// fit's transient charge covers there.
// load_over_base is every load and warm-up sample over the baseline: in the
// late schedule the load runs outside the VAE decode phase, so each of them is
// held to the transient charge too (late_window).
struct mem_breakdown {
    uint64_t prepare = 0, load_warm = 0, load_over_base = 0, steady = 0, late = 0, total = 0;
    bool prepared = false, compiled = false, loaded = false, ready = false, late_seen = false;
};

// What the late schedule holds to the transient charge: every sample of the
// load, its warm-up and the tail after the eviction, over the baseline.
uint64_t late_window(const mem_breakdown & b) { return std::max(b.load_over_base, b.late); }

bool loading_phase(mem_phase p) { return p == mem_phase::load || p == mem_phase::warm || p == mem_phase::late; }

// The label of an ANECompilerService query bracketed by two reads of the
// phase: loading when both are, not loading when neither is, and none (the
// sample is dropped) when the query crossed the load's start or end.
std::optional<bool> ane_query_loading(mem_phase before, mem_phase after) {
    if (loading_phase(before) != loading_phase(after)) return std::nullopt;
    return loading_phase(before);
}

// What the probes failed to measure: the baseline, footprint samples or (on
// the Neural Engine) the compiler service while loading. A run with any of
// them fails instead of reading the missing values as 0.
std::vector<std::string> schedule_probe_failures(bool base_ok, size_t footprint_failures, bool ane_units,
                                                 const qvac_coreml_memory::AneGrowth & ane) {
    std::vector<std::string> out;
    if (!base_ok) out.push_back("no footprint baseline (the probe failed)");
    if (footprint_failures > 0) out.push_back(std::to_string(footprint_failures) + " footprint probe(s) failed");
    if (ane_units && !qvac_coreml_memory::ane_compiler_measured(ane)) {
        out.push_back("no successful ANECompilerService query while loading or warming");
    } else if (ane_units && ane.failed_samples > 0) {
        out.push_back(std::to_string(ane.failed_samples) + " ANECompilerService query(ies) failed");
    }
    return out;
}

// Whether one ggml decode of the schedule did its work: the whole PCM, on ggml.
bool ggml_decode_ok(const std::vector<float> & pcm, int T, const std::string & backend) {
    return pcm.size() == (size_t) T * 1920 * 2 && backend == "ggml";
}

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
            case 1:
                b.prepared = true;
                b.compiled = b.compiled || s.phase == mem_phase::compile;
                b.prepare = std::max(b.prepare, over(s.footprint, group_start));
                break;
            case 2:
                b.loaded = true;
                b.load_warm = std::max(b.load_warm, over(s.footprint, group_start));
                b.load_over_base = std::max(b.load_over_base, over(s.footprint, base));
                break;
            case 3: b.late_seen = true; b.late = std::max(b.late, over(s.footprint, base)); break;
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

// The breakdown on a synthetic late schedule: preparation leaves 200 above
// the baseline, the load adds 50, and after the eviction the footprint stays
// at 280 above the baseline. The late tail is 280 (over the baseline), not
// the 30 it is over the footprint just before the load.
void breakdown_self_test() {
    const uint64_t base = 1000;
    const std::vector<mem_sample> samples = {
        {1000, mem_phase::ggml},    {1150, mem_phase::generate}, {1200, mem_phase::compile},
        {1200, mem_phase::ggml},    {1230, mem_phase::load},     {1250, mem_phase::warm},
        {1280, mem_phase::late},    {1240, mem_phase::late},     {1210, mem_phase::ggml},
    };
    const mem_breakdown b = mem_breakdown_of(samples, base);
    check(b.prepared && b.compiled && b.loaded && b.late_seen && b.prepare == 200 && b.load_warm == 50 &&
              b.late == 280 && b.total == 280,
          "memory breakdown: the late tail is every post-eviction sample over the ggml-only baseline (" +
              std::to_string(b.late) + ")");
    // A load peak that is gone before the completion hook: preparation keeps
    // 240 above the baseline, the load peaks 1500 above it, the tail after the
    // eviction is 240 again. The late window is 1500, not the tail's 240.
    const std::vector<mem_sample> peak = {
        {1000, mem_phase::ggml}, {1300, mem_phase::generate}, {1240, mem_phase::compile},
        {1240, mem_phase::ggml}, {2500, mem_phase::load},     {1300, mem_phase::warm},
        {1240, mem_phase::late}, {1240, mem_phase::late},
    };
    const mem_breakdown p = mem_breakdown_of(peak, base);
    check(p.late == 240 && p.load_over_base == 1500 && late_window(p) == 1500,
          "memory breakdown: the late window holds the load's own samples over the baseline (" +
              std::to_string(late_window(p)) + ")");
    // Measurements that failed are not readings.
    qvac_coreml_memory::AneGrowth measured;
    measured.loading_samples = 3;
    check(schedule_probe_failures(true, 0, true, measured).empty() &&
              schedule_probe_failures(true, 0, false, qvac_coreml_memory::AneGrowth{}).empty(),
          "memory schedule: complete measurements pass");
    qvac_coreml_memory::AneGrowth failed_query = measured;
    failed_query.failed_samples = 1;
    check(schedule_probe_failures(false, 0, false, measured).size() == 1 &&
              schedule_probe_failures(true, 2, false, measured).size() == 1 &&
              schedule_probe_failures(true, 0, true, qvac_coreml_memory::AneGrowth{}).size() == 1 &&
              schedule_probe_failures(true, 0, true, failed_query).size() == 1,
          "memory schedule: a failed baseline, failed footprint samples, no compiler reading during the load or a "
          "failed compiler query each fail the run");
    check(ane_query_loading(mem_phase::load, mem_phase::warm) == std::optional<bool>(true) &&
              ane_query_loading(mem_phase::compile, mem_phase::ggml) == std::optional<bool>(false) &&
              !ane_query_loading(mem_phase::compile, mem_phase::load).has_value() &&
              !ane_query_loading(mem_phase::warm, mem_phase::ready).has_value(),
          "memory schedule: a compiler query that crossed the load's start or end has no label");
    const int T = 64;
    const std::vector<float> full((size_t) T * 1920 * 2, 0.0f);
    check(ggml_decode_ok(full, T, "ggml") && !ggml_decode_ok({}, T, "ggml") && !ggml_decode_ok(full, T, "coreml-all") &&
              !ggml_decode_ok(std::vector<float>(full.size() - 2), T, "ggml"),
          "memory schedule: a ggml call counts only with the full PCM on ggml");
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
    // One ggml decode: true when it produced the whole PCM on ggml. A failed
    // one fails the run instead of counting.
    bool bad_ggml = false;
    const auto ggml_decode = [&]() {
        const std::vector<float> pcm = ggml_vae->decode(latent, T);
        if (ggml_decode_ok(pcm, T, ggml_vae->last_decode_backend())) return true;
        if (!bad_ggml) {
            check(false, "every ggml decode of the schedule returns the whole PCM on ggml (" +
                             std::to_string(pcm.size()) + " samples on " + ggml_vae->last_decode_backend() + ")");
        }
        bad_ggml = true;
        return false;
    };
    for (int i = 0; i < 2; ++i) {  // warm the ggml arena
        if (!ggml_decode()) return;
    }
    uint64_t base = 0;
    const bool base_ok = qvac_coreml_memory::phys_footprint(base);

    std::atomic<bool> stop{false};
    std::atomic<int> phase{(int) mem_phase::ggml};
    std::atomic<bool> in_late{false};
    std::shared_ptr<detail::vae_coreml_stage> stage = detail::vae_coreml_stage::create(vae, opts, false, true);
    const auto slot_phase = [&] {
        return in_late.load() ? mem_phase::late : mem_phase_of(state_of(stage->status_json()));
    };
    std::vector<qvac_coreml_memory::AneSample> ane_samples;
    std::vector<mem_sample> samples;
    samples.reserve(1 << 17);
    size_t footprint_failures = 0, ane_crossing = 0;
    std::thread sampler([&] {
        for (int tick = 0; !stop.load(); ++tick) {
            if (tick % 5 == 0) phase.store((int) slot_phase());
            uint64_t fp = 0;
            if (qvac_coreml_memory::phys_footprint(fp)) {
                samples.push_back({fp, (mem_phase) phase.load()});
            } else {
                ++footprint_failures;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    });
    // Each compiler query is bracketed by its own reads of the phase; one that
    // crossed the load's start or end is dropped (counted).
    std::thread ane_sampler([&] {
        while (!stop.load()) {
            const mem_phase before = slot_phase();
            uint64_t a = 0;
            const bool ok = qvac_coreml_memory::ane_compiler_rss(a);
            const std::optional<bool> loading = ane_query_loading(before, slot_phase());
            if (loading) {
                ane_samples.push_back({a, *loading, ok});
            } else {
                ++ane_crossing;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    });

    const auto t0 = Clock::now();
    int ggml_calls = 0, coreml_calls = 0;
    bool evicted = false, hint_seen = false, repopulated = false;
    if (late) completion_barrier::arm();
    while (Clock::now() - t0 < std::chrono::minutes(6)) {
        if (late && evicted) {
            // After the eviction: ggml keeps working until the late load
            // wrote its hint, then two more seconds for the release. The slot
            // must stay idle throughout.
            if (!ggml_decode()) break;
            ++ggml_calls;
            const std::string st = stage->status_json();
            if (state_of(st) != "idle") repopulated = true;
            if (!hint_seen && has_hint(st)) {
                hint_seen = true;
                const auto t_hint = Clock::now();
                while (Clock::now() - t_hint < std::chrono::seconds(2) && ggml_decode()) {
                    ++ggml_calls;
                    if (state_of(stage->status_json()) != "idle") repopulated = true;
                }
                break;
            }
            continue;
        }
        detail::vae_coreml_call call = stage->acquire(T);
        if (late && !evicted && completion_barrier::is_entered()) {
            // The cold load finished its warm-up and is held before it
            // publishes: a VAE stage boundary now, then the completion.
            call = {};
            in_late.store(true);
            stage->evict();
            evicted = true;
            const std::string st = stage->status_json();
            if (state_of(st) != "idle" || has_hint(st)) repopulated = true;
            completion_barrier::release();
            continue;
        }
        if (!call.decoder) {
            if (!ggml_decode()) break;
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
    if (late) completion_barrier::release();  // never leave a load held (the loop may have timed out)
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
                 "(resident charged %.1f), late window over the baseline %.1f (transient charged %.1f), whole run %.1f (envelope %.1f); "
                 "ANECompilerService growth %.1f from %.1f (%s), peak %.1f, reserve %.1f\n",
                 mib(base), mib(b.prepare), mib(route.prepare_peak_bytes), mib(b.load_warm), mib(route.load_peak_bytes),
                 mib(b.steady), mib(route.resident_bytes), mib(late ? late_window(b) : b.late), mib(route.transient_bytes), mib(b.total),
                 mib(route.envelope_bytes), mib(ane.growth), mib(ane.before), baseline, mib(ane_peak),
                 mib(route.ane_compile_reserve_bytes));
    std::fprintf(stderr,
                 "  memory report: {\"chip\":\"%s\",\"units\":\"%s\",\"schedule\":\"%s\",\"base\":%llu,\"prepare\":%llu,"
                 "\"load_warm\":%llu,\"steady\":%llu,\"late\":%llu,\"total\":%llu,\"ane_compiler_before_load\":%llu,"
                 "\"ane_compiler_growth\":%llu,\"ane_compiler_load_peak\":%llu,\"ane_compiler_peak\":%llu,"
                 "\"ane_compiler_baseline\":\"%s\",\"ane_loading_samples\":%zu,\"ane_crossing\":%zu,"
                 "\"ggml_calls\":%d,\"coreml_calls\":%d,\"seconds\":%.1f}\n",
                 host_chip().c_str(), route.compute_units.c_str(), late ? "late" : "early", (unsigned long long) base,
                 (unsigned long long) b.prepare, (unsigned long long) b.load_warm, (unsigned long long) b.steady,
                 (unsigned long long) b.late, (unsigned long long) b.total, (unsigned long long) ane.before,
                 (unsigned long long) ane.growth, (unsigned long long) ane.peak, (unsigned long long) ane_peak, baseline,
                 ane.loading_samples, ane_crossing, ggml_calls, coreml_calls, secs);

    const bool ane_units = route.compute_units == "all" || route.compute_units == "cpu_and_ane";
    for (const std::string & f : schedule_probe_failures(base_ok, footprint_failures, ane_units, ane)) {
        check(false, "memory schedule measured: " + f);
    }
    check(!bad_ggml, "every ggml decode of the schedule succeeded on ggml");
    check(b.compiled && b.loaded, "the schedule sampled the compile and the load");
    check(b.prepare <= route.prepare_peak_bytes, "prepare within its charge");
    check(b.load_warm <= route.load_peak_bytes, "load and warm-up within the load charge");
    check(b.total <= route.envelope_bytes, "whole run within the envelope");
    if (route.ane_compile_reserve_bytes > 0) {
        // The reserve is set from the service's absolute size while loading
        // (a growth over a service that already holds memory understates a
        // cold compile), so the absolute size is what it must cover.
        check(ane.peak <= route.ane_compile_reserve_bytes,
              "ANECompilerService absolute size while loading within the reserve (" + std::string(baseline) +
                  " baseline)");
    }
    if (late) {
        check(evicted && hint_seen && b.late_seen, "late completion: evicted while loading, the load wrote its hint");
        // The load ran outside the VAE decode phase from the eviction on, but
        // it was held only after its warm-up: so each load and warm-up sample
        // is held to the transient charge over the baseline as well as the
        // tail after the eviction.
        check(late_window(b) <= route.transient_bytes,
              "load, warm-up and late tail over the warmed ggml-only baseline within the transient charge (what "
              "fit charges outside the VAE decode phase): " + std::to_string(mib(late_window(b))) + " MiB");
        check(coreml_calls == 0 && !repopulated && state_of(stage->status_json()) == "idle",
              "late completion: the load completed after the eviction and never repopulated the slot");
    } else {
        check(b.ready && coreml_calls >= 5 && ggml_calls >= 1, "early completion: ggml then 5 Core ML calls (" +
                                                                    std::to_string(ggml_calls) + " / " +
                                                                    std::to_string(coreml_calls) + ")");
        check(b.steady <= route.resident_bytes, "steady state within the resident charge");
    }
}

}  // namespace

int main(int argc, char ** argv) {
    bool self = false, fit = false, require = false, late = false, host_route = false, short_cold = false;
    std::string models, vae, vae_b, cache_root, memory_dir, first_decode, cases = "encode,lowmem,cached,cancel,repaint,prewarm,prewarm-async,prewarm-call,evict,short,degrade";
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
        else if (a == "--host-route") host_route = true;
        else if (a == "--late") late = true;
        else if (a == "--models") models = next();
        else if (a == "--vae") vae = next();
        else if (a == "--source-swap") vae_b = next();
        else if (a == "--short-latent") short_cold = true;
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
    {
        // An unknown case name would select nothing and pass without a check.
        const std::string known = "encode,lowmem,cached,cancel,repaint,prewarm,prewarm-async,prewarm-call,evict,short,degrade";
        size_t at = 0;
        while (at <= cases.size()) {
            const size_t end = std::min(cases.find(',', at), cases.size());
            const std::string name = cases.substr(at, end - at);
            if (!has_case(known, name)) {
                std::fprintf(stderr, "unknown case '%s' (known: %s)\n", name.c_str(), known.c_str());
                return 2;
            }
            at = end + 1;
        }
    }

    if (self) {
        self_test();
        bf16_only_self_test();
        breakdown_self_test();
    }
    if (fit) {
        if (vae.empty()) {
            std::fprintf(stderr, "--fit-stable needs --vae or --models\n");
            return 2;
        }
        fit_stable(vae, models, cache_root.empty() ? fs::temp_directory_path().string() : cache_root, require,
                   host_route);
        route_none_construction(vae, cache_root.empty() ? fs::temp_directory_path().string() : cache_root);
    }
    if (!vae_b.empty()) {
        if (vae.empty()) {
            std::fprintf(stderr, "--source-swap needs --vae\n");
            return 2;
        }
        source_swap(vae, vae_b, cache_root.empty() ? fs::temp_directory_path().string() : cache_root);
    }
    if (short_cold) {
        if (vae.empty()) {
            std::fprintf(stderr, "--short-latent needs --vae\n");
            return 2;
        }
        short_latent(vae, cache_root.empty() ? fs::temp_directory_path().string() : cache_root, false);
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
            if (has_case(cases, "encode")) case_encode(models, cache_root);
            std::string lowmem_cache;
            if (has_case(cases, "lowmem")) lowmem_cache = case_lowmem(models, cache_root, duration);
            // The cases that need a primed cache prime one themselves when the
            // lowmem case did not run; they never pass without assertions.
            std::string primed = lowmem_cache;
            if (primed.empty() && (has_case(cases, "cached") || has_case(cases, "cancel") || has_case(cases, "repaint"))) {
                primed = prime_cache(models, cache_root);
            }
            if (has_case(cases, "cached")) {
                if (primed.empty()) check(false, "the cached case needs a primed cache");
                else case_cached(argv[0], models, primed, duration);
            }
            if (has_case(cases, "cancel")) {
                if (primed.empty()) check(false, "the cancel case needs a primed cache");
                else case_cancel(models, cache_root, primed, duration);
            }
            if (has_case(cases, "repaint")) {
                if (primed.empty()) check(false, "the repaint case needs a primed cache");
                else case_repaint(models, primed);
            }
            if (has_case(cases, "prewarm")) case_prewarm(models, cache_root, duration);
            if (has_case(cases, "prewarm-async")) case_prewarm_async(models, cache_root);
            if (has_case(cases, "prewarm-call")) case_prewarm_call(models, cache_root);
            if (has_case(cases, "evict")) case_evict(vae, cache_root);
            if (has_case(cases, "short")) short_latent(vae, cache_root, true);
            if (has_case(cases, "degrade")) case_degrade(models, cache_root);
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
