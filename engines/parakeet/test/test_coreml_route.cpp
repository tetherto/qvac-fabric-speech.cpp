// Parakeet's Core ML routing on the shared runtime. Hermetic: fake loaders and a fake encoder, no Core ML, no model.
// Built against the runtime's test copy (QVAC_COREML_NS=qvac_coreml_test) on
// Linux and macOS; it has no skip path.
//
//   null-route-plan       a null route gives exactly the ggml long-form plan,
//                         whatever capacity a sidecar has; a held route gives
//                         the fixed-shape plan (so the check can fail)
//   window-k-failure      a failure at window k retires the sidecar, windows
//                         k.. of the call run on ggml with the plan already
//                         chosen, and the call reports "mixed"
//   next-call-reverts     the next call acquires a null route: ggml plan,
//                         label "ggml"
//   cannot-serve          a capacity/shape miss (encode rc 1-4) runs that
//                         dispatch on ggml, leaves the slot ready and the
//                         next dispatch on Core ML; only rc 5-7 retire
//   owner-route           the synchronous owner route returns the error and
//                         never retires
//   strict-route          use = require: an input over the fixed capacity
//                         (long-form windowing disabled) throws CoremlError
//                         instead of running on ggml and leaves the slot
//                         ready; a cannot-serve code throws without retiring;
//                         a failed prediction retires and throws
//   pending-route         a slot still loading gives the ggml route marked
//                         pending; with wait = call the acquisition waits
//   sidecar-groups        construction group = the minimal set; prewarm all
//                         adds the rest; an absent sidecar is skipped
//   options               public CoremlOptions -> runtime options, resolution
//   fit-stable            the fit route and envelope never read the cache:
//                         identical with no hint, a warm hint and a stale hint
//   generated-envelope    the generated-diarizer fit charge: capacities a
//                         range loads (prebuilt ones shape the selection),
//                         resident sum + one transient, largest margin;
//                         every generated capacity under prewarm = all
//   verdicts-validated    coreml.validated only when every charged byte is
//                         a generated envelope
//   generated-sidecar     a stage with no adjacent bundle and a matching
//                         template is generated into the cache (route
//                         generate, fake compile), loads under the artifact's
//                         lease and reports "generated", then "cached" in the
//                         next instance; a GGUF the template does not match,
//                         unvalidated units or no cache directory give no
//                         route, and the first two a status-only slot that
//                         status and the construction group report
//                         (no_template, unsupported(unvalidated_units),
//                         generation.reason unvalidated_units); the
//                         generated envelope and route are the same before
//                         and after generation (fit-stable); a
//                         memory-degraded instance has no generate route
//   diarizer-capacities   fit and discovery choose the same generated
//                         capacities; a capacity that is not generated keeps
//                         a status-only slot that never serves a pass, is
//                         listed by the construction group (every capacity
//                         when generation is off) and, under use = require,
//                         refuses a pass it covers; a pass longer than every
//                         capacity still runs on ggml; adjacent encoder and
//                         bypass-encoder sidecars stay in the construction
//                         group unless a diarizer capacity serves

#include "coreml/parakeet_coreml_sidecars.h"
#include "coreml_fit.h"
#include "encoder_route.h"
#include "long_form.h"
#include "qvac_coreml/generation.h"

#if defined(QVAC_COREML_HAVE_GGML)
#include "ggml.h"
#include "gguf.h"
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

namespace fs = std::filesystem;
using namespace parakeet;

namespace {

int g_failures = 0;
int g_checks = 0;

#define CHECK(cond, what)                                                          \
    do {                                                                           \
        ++g_checks;                                                                \
        if (!(cond)) {                                                             \
            ++g_failures;                                                          \
            std::fprintf(stderr, "FAIL %s:%d: %s (%s)\n", __FILE__, __LINE__, what, #cond); \
        }                                                                          \
    } while (0)

uint64_t big_memory() { return uint64_t(1) << 40; }
bool no_low_power() { return false; }

pcm::Config fast_config() {
    pcm::Config c;
    c.available_memory = &big_memory;
    c.low_power = &no_low_power;
    c.host_id = "parakeet-route-test";
    c.os_build = "TEST1";
    c.memory_recheck = std::chrono::milliseconds(20);
    c.backoff_initial = std::chrono::milliseconds(10);
    c.backoff_max = std::chrono::milliseconds(80);
    c.wait_poll = std::chrono::milliseconds(5);
    c.pin_image = false;
    return c;
}

fs::path make_temp_dir(const char * tag) {
    const fs::path base = fs::temp_directory_path() /
                          ("parakeet-coreml-route-" + std::string(tag) + "-" + std::to_string(::getpid()));
    fs::remove_all(base);
    fs::create_directories(base);
    return base;
}

// ---- fake encoder ---------------------------------------------------------

constexpr int kFixedMelFrames = 1504;  // 188 encoder frames at 8x subsampling
constexpr int kSeamContext = 16;
constexpr const char * kFakeLabel = "coreml-ane";

struct FakeEncoder {
    std::atomic<int> encodes{0};
    std::atomic<int> fail_at{-1};  // encode index that fails, -1 never
    std::atomic<int> fail_rc{7};   // the code it fails with (7: a failed prediction)
};

std::atomic<int> g_fake_loads{0};

pcm::LoadResult fake_load(const pcm::LoadRequest &) {
    ++g_fake_loads;
    pcm::LoadResult r;
    r.model = std::make_shared<FakeEncoder>();
    return r;
}

// A load that blocks until released (a slow cold ANE compile).
std::mutex g_gate_mu;
std::condition_variable g_gate_cv;
bool g_gate_open = false;

pcm::LoadResult gated_load(const pcm::LoadRequest & req) {
    std::unique_lock<std::mutex> lock(g_gate_mu);
    g_gate_cv.wait(lock, [] { return g_gate_open; });
    return fake_load(req);
}

void open_gate() {
    std::lock_guard<std::mutex> lock(g_gate_mu);
    g_gate_open = true;
    g_gate_cv.notify_all();
}

parakeet_coreml_context * fake_context(void * model) {
    return static_cast<parakeet_coreml_context *>(model);
}

EncoderRouteGeometry fake_describe(parakeet_coreml_context *, int n_mels) {
    EncoderRouteGeometry g;
    g.fixed_mel_frames = n_mels > 0 ? kFixedMelFrames : 0;
    g.seam_context_frames = kSeamContext;
    return g;
}

const char * fake_label(parakeet_coreml_context *) { return kFakeLabel; }

int fake_encode(void *, parakeet_coreml_context * ctx, int, int, int, const float *, int, int, float *) {
    auto * enc = reinterpret_cast<FakeEncoder *>(ctx);
    const int index = enc->encodes++;
    return index == enc->fail_at.load() ? enc->fail_rc.load() : 0;
}

const EncoderModelOps & fake_ops() {
    static const EncoderModelOps ops{&fake_context, &fake_describe, &fake_label, &fake_encode};
    return ops;
}

std::shared_ptr<pcm::Slot> make_encoder_slot(pcm::Manager & m, pcm::LoadFn load, pcm::Wait wait,
                                             const fs::path & dir) {
    pcm::SlotSpec spec;
    spec.engine = "parakeet";
    spec.stage = "encoder";
    spec.units = pcm::ComputeUnits::cpu_and_ane;
    spec.route = pcm::Route::prebuilt;
    spec.artifact_path = (dir / "model-encoder.mlmodelc").string();
    spec.load_fn = load;
    pcm::SlotPolicy policy;
    policy.wait = wait;
    return m.make_slot(spec, policy);
}

// A TDT-shaped plan input: 300 s of audio at 10 ms hops, 8x subsampling.
constexpr long long kLongMel = 30000;
constexpr int kPosEmbMaxLen = 5000;
constexpr int kSub = 8;

LongFormPlan ggml_plan(long long n_mel) {
    return resolve_long_form_plan_frames(0, 0, kPosEmbMaxLen, kSub, n_mel);
}

LongFormPlan routed_plan(const EncoderRoute & route, long long n_mel,
                         CoremlShapeKind kind = CoremlShapeKind::fixed, int opts_window = 0) {
    return choose_long_form_plan(ggml_plan(n_mel), opts_window, 0, kind, route.holds(),
                                 route.geometry().fixed_mel_frames,
                                 route.geometry().seam_context_frames, kSub, n_mel);
}

bool same_plan(const LongFormPlan & a, const LongFormPlan & b) {
    return a.enabled == b.enabled && a.window_frames == b.window_frames &&
           a.context_frames == b.context_frames && a.center_frames == b.center_frames &&
           a.sub == b.sub && a.left_context_frames == b.left_context_frames &&
           a.right_context_frames == b.right_context_frames &&
           a.exact_mel_frames == b.exact_mel_frames;
}

// One call over `windows` dispatches, as run_encoder_windowed drives them.
struct CallOutcome {
    std::vector<bool> on_coreml;
    int ggml_dispatches = 0;
    std::string label;
};

CallOutcome run_call(const EncoderRoute & route, int windows) {
    CallOutcome out;
    EncoderBackendTally tally;
    tally.add_route(route);
    for (int w = 0; w < windows; ++w) {
        bool used = false;
        const int rc = dispatch_on_route(
            route, EncoderEligibility::coreml,
            [&] { return route.encode(kFixedMelFrames, kFixedMelFrames, 128, nullptr, 188, 1024, nullptr); },
            [&] { ++out.ggml_dispatches; return 0; }, used);
        CHECK(rc == 0, "every window succeeds (Core ML or the ggml fallback)");
        out.on_coreml.push_back(used);
        tally.add(used, route.coreml_label());
    }
    out.label = tally.label();
    return out;
}

// ---- cases -----------------------------------------------------------------

void case_null_route_plan() {
    const EncoderRoute null_route = EncoderRoute::ggml(/*pending=*/true);
    CHECK(!null_route.holds(), "the ggml route holds no handle");
    CHECK(same_plan(routed_plan(null_route, kLongMel), ggml_plan(kLongMel)),
          "null route, long input: exactly the ggml plan");
    CHECK(same_plan(routed_plan(null_route, 1000), ggml_plan(1000)),
          "null route, short input: exactly the ggml plan");
    CHECK(same_plan(routed_plan(null_route, kLongMel, CoremlShapeKind::exact), ggml_plan(kLongMel)),
          "null route, EOU: the ggml plan");

    // A held route with the same capacity changes the plan: the check above
    // can fail.
    FakeEncoder enc;
    const EncoderRoute held = EncoderRoute::coreml(reinterpret_cast<parakeet_coreml_context *>(&enc),
                                                   fake_describe(nullptr, 128), kFakeLabel,
                                                   nullptr, EncoderRoute::Ops{}, false);
    const LongFormPlan coreml_plan = routed_plan(held, kLongMel);
    CHECK(coreml_plan.enabled && coreml_plan.window_frames == kFixedMelFrames / kSub,
          "a held route tiles the input at the sidecar capacity");
    CHECK(coreml_plan.left_context_frames == kSeamContext,
          "the sidecar's seam context applies when the caller left it automatic");
    CHECK(!same_plan(coreml_plan, ggml_plan(kLongMel)), "the Core ML plan differs from the ggml plan");
    CHECK(same_plan(routed_plan(held, kLongMel, CoremlShapeKind::fixed, -1), ggml_plan(kLongMel)),
          "a negative window request keeps the ggml plan even with a handle");
    CHECK(same_plan(routed_plan(held, kLongMel, CoremlShapeKind::exact), ggml_plan(kLongMel)),
          "oversized EOU input stays on the ggml plan");
}

// TDT tiles an oversized input into Core ML windows only when the handle
// reports a validated seam context. A standard TDT sidecar keeps the ggml
// plan, so its oversized dispatch is a capacity miss (ggml, or a refusal
// under use = require), never a short Core ML window.
void case_tdt_plan_needs_validated_seams() {
    FakeEncoder enc;
    const auto held = [&](int seam_context_frames) {
        EncoderRouteGeometry g = fake_describe(nullptr, 128);
        g.seam_context_frames = seam_context_frames;
        return EncoderRoute::coreml(reinterpret_cast<parakeet_coreml_context *>(&enc), g, kFakeLabel,
                                    nullptr, EncoderRoute::Ops{}, false);
    };
    const EncoderRoute standard = held(0);
    const EncoderRoute validated = held(kSeamContext);

    // The engine plans with the kind its model family maps to, so a change of
    // that mapping is caught here, not only by the fixture tests.
    const CoremlShapeKind kTdt = coreml_shape_kind_for(ParakeetModelType::TDT);
    CHECK(kTdt == CoremlShapeKind::fixed_validated_seams,
          "TDT plans fixed windows only with validated seams");
    CHECK(coreml_shape_kind_for(ParakeetModelType::CTC) == CoremlShapeKind::fixed &&
              coreml_shape_kind_for(ParakeetModelType::RNNT) == CoremlShapeKind::fixed,
          "CTC and RNN-T plan fixed windows");
    CHECK(coreml_shape_kind_for(ParakeetModelType::EOU) == CoremlShapeKind::exact,
          "EOU plans exact-shape calls only");
    CHECK(coreml_shape_kind_for(ParakeetModelType::NEMOTRON) == CoremlShapeKind::none &&
              coreml_shape_kind_for(ParakeetModelType::SORTFORMER) == CoremlShapeKind::none &&
              coreml_shape_kind_for(ParakeetModelType::NEMOTRON_DIARIZATION) ==
                  CoremlShapeKind::none,
          "Nemotron and the diarizers have no long-form Core ML plan");

    CHECK(same_plan(routed_plan(standard, kLongMel, kTdt), ggml_plan(kLongMel)),
          "standard TDT sidecar: an oversized input keeps the ggml plan");
    CHECK(routed_plan(standard, kLongMel, CoremlShapeKind::fixed).enabled &&
              !same_plan(routed_plan(standard, kLongMel, CoremlShapeKind::fixed), ggml_plan(kLongMel)),
          "the same handle tiles a CTC or RNN-T input at its capacity");
    CHECK(same_plan(routed_plan(standard, 1000, kTdt), ggml_plan(1000)),
          "standard TDT sidecar: an input within capacity keeps the single pass");
    CHECK(!route_fits_capacity(standard, (int) kLongMel),
          "a whole oversized TDT input does not fit the standard sidecar");

    const LongFormPlan tdt = routed_plan(validated, kLongMel, kTdt);
    CHECK(tdt.enabled && tdt.window_frames == kFixedMelFrames / kSub &&
              tdt.left_context_frames == kSeamContext,
          "validated TDT sidecar: tiles at its capacity with its seam context");
    CHECK(same_plan(tdt, routed_plan(validated, kLongMel, CoremlShapeKind::fixed)),
          "validated TDT sidecar: the same windows as CTC and RNN-T");
    CHECK(same_plan(routed_plan(validated, kLongMel, kTdt, -1), ggml_plan(kLongMel)),
          "a negative window request keeps the ggml plan for TDT too");
}

void case_window_k_failure_and_next_call() {
    const fs::path dir = make_temp_dir("windowk");
    auto m = pcm::Manager::create(fast_config());
    auto slot = make_encoder_slot(*m, &fake_load, pcm::Wait::call, dir);

    // Call 1: the slot is ready (wait = call), window 2 of 5 fails.
    EncoderRoute route = route_from_acquired(slot->acquire(pcm::Demand::now), fake_ops(), 128);
    CHECK(route.holds() && route.usable() && !route.pending(), "call 1 holds the encoder");
    CHECK(route.coreml_label() == kFakeLabel, "the route carries the handle's label");
    CHECK(route.geometry().fixed_mel_frames == kFixedMelFrames, "capacity comes from the handle");
    const LongFormPlan call1_plan = routed_plan(route, kLongMel);
    reinterpret_cast<FakeEncoder *>(route.context())->fail_at = 2;

    const CallOutcome call1 = run_call(route, 5);
    CHECK(call1.on_coreml.size() == 5, "five windows");
    CHECK(call1.on_coreml[0] && call1.on_coreml[1], "windows before k ran on Core ML");
    CHECK(!call1.on_coreml[2] && !call1.on_coreml[3] && !call1.on_coreml[4],
          "window k and the rest of the call ran on ggml");
    CHECK(call1.ggml_dispatches == 3, "three ggml dispatches");
    CHECK(call1.label == "mixed", "the call reports mixed");
    CHECK(route.failed() && route.holds(), "the route is spent but the plan stays the Core ML plan");
    CHECK(same_plan(routed_plan(route, kLongMel), call1_plan), "the plan chosen at the start is kept");
    CHECK(slot->state() == pcm::SlotState::retired, "the failure retired the slot");

    // Call 2: reverts to the ggml plan.
    const EncoderRoute next = route_from_acquired(slot->acquire(pcm::Demand::now), fake_ops(), 128);
    CHECK(!next.holds(), "the next call acquires a null route");
    CHECK(!next.pending(), "a retired slot is not pending");
    CHECK(same_plan(routed_plan(next, kLongMel), ggml_plan(kLongMel)), "the next call uses the ggml plan");
    CHECK(run_call(next, 3).label == "ggml", "the next call reports ggml");

    // A second, independent owner of the same artifact sees the retirement
    // too (process-wide per (artifact, function, units)).
    const int loads_before = g_fake_loads.load();
    auto other = make_encoder_slot(*m, &fake_load, pcm::Wait::call, dir);
    const EncoderRoute other_route = route_from_acquired(other->acquire(pcm::Demand::now), fake_ops(), 128);
    CHECK(!other_route.holds(), "another owner of the retired artifact gets the ggml route");
    CHECK(other->state() == pcm::SlotState::retired, "and its slot reports retired");
    CHECK(g_fake_loads.load() == loads_before, "a retired artifact is never loaded again");
    fs::remove_all(dir);
}

void case_cannot_serve_keeps_route() {
    const fs::path dir = make_temp_dir("cannotserve");
    auto m = pcm::Manager::create(fast_config());
    auto slot = make_encoder_slot(*m, &fake_load, pcm::Wait::call, dir);

    // Window 1 of 3 exceeds the handle's capacity: the sidecar returns 2
    // (parakeet_coreml_encode_valid's capacity / ANE-window code).
    EncoderRoute route = route_from_acquired(slot->acquire(pcm::Demand::now), fake_ops(), 128);
    CHECK(route.usable(), "the call holds the encoder");
    auto * enc = reinterpret_cast<FakeEncoder *>(route.context());
    enc->fail_rc = 2;
    enc->fail_at = 1;
    const CallOutcome call = run_call(route, 3);
    CHECK(call.on_coreml.size() == 3 && call.on_coreml[0] && !call.on_coreml[1] && call.on_coreml[2],
          "only the window the handle cannot serve ran on ggml; the next one ran on Core ML");
    CHECK(call.ggml_dispatches == 1, "one ggml dispatch");
    CHECK(route.usable() && !route.failed(), "a capacity miss leaves the route usable");
    CHECK(slot->state() == pcm::SlotState::ready, "a capacity miss does not retire the slot");

    const EncoderRoute next = route_from_acquired(slot->acquire(pcm::Demand::now), fake_ops(), 128);
    CHECK(next.holds(), "the next call still acquires the encoder");
    CHECK(run_call(next, 2).label == kFakeLabel, "the next call runs on Core ML");

    // Every cannot-serve code keeps the route; every prediction code retires.
    for (int rc = 1; rc <= 7; ++rc) {
        CHECK(encode_prediction_failed(rc) == (rc >= 5), "rc 1-4 cannot serve, rc 5-7 failed prediction");
    }
    CHECK(!encode_prediction_failed(0) && !encode_prediction_failed(101) &&
              !encode_prediction_failed(-10),
          "success, non-finite features and the engine's CTC-head codes are not sidecar failures");

    // A real prediction failure on the same slot still retires it.
    const EncoderRoute last = route_from_acquired(slot->acquire(pcm::Demand::now), fake_ops(), 128);
    auto * enc2 = reinterpret_cast<FakeEncoder *>(last.context());
    enc2->fail_rc = 5;
    enc2->fail_at = enc2->encodes.load() + 1;  // the second window
    CHECK(run_call(last, 2).label == "mixed", "a failed prediction makes the call mixed");
    CHECK(slot->state() == pcm::SlotState::retired, "a failed prediction retires the slot");
    fs::remove_all(dir);
}

// A call whose every dispatch ran on Core ML always names a program: the one
// the model picked for its dispatches (TranscribePsy EU picks per dispatch),
// else the held route's label, else "coreml".
void case_call_label() {
    FakeEncoder enc;
    const auto held = [&](const char * label) {
        return EncoderRoute::coreml(reinterpret_cast<parakeet_coreml_context *>(&enc),
                                    fake_describe(nullptr, 128), label, nullptr,
                                    EncoderRoute::Ops{}, false);
    };
    CHECK(coreml_call_label(held(kFakeLabel)) == kFakeLabel, "the held route's label");
    CHECK(coreml_call_label(held(kFakeLabel), "coreml-cpu") == "coreml-cpu",
          "the program that ran wins over the route's label");
    CHECK(coreml_call_label(held("")) == "coreml", "an unlabeled route reports coreml");
    CHECK(coreml_call_label(EncoderRoute::ggml(), "coreml-cpu-gpu") == "coreml-cpu-gpu",
          "a per-dispatch program without a held route is still reported");
    CHECK(coreml_call_label(EncoderRoute::ggml()) == "coreml", "the label is never empty");
}

void case_owner_route() {
    FakeEncoder enc;
    enc.fail_at = 0;
    static std::atomic<int> retires{0};
    EncoderRoute::Ops ops;
    ops.encode = &fake_encode;
    ops.retire = [](void *, const char *) { ++retires; };
    const EncoderRoute owner = EncoderRoute::coreml(reinterpret_cast<parakeet_coreml_context *>(&enc),
                                                    fake_describe(nullptr, 128), kFakeLabel,
                                                    nullptr, ops, /*owner=*/true);
    bool used = true;
    int ggml_runs = 0;
    const int rc = dispatch_on_route(
        owner, EncoderEligibility::coreml, [&] { return owner.encode(1, 1, 128, nullptr, 1, 1, nullptr); },
        [&] { ++ggml_runs; return 0; }, used);
    CHECK(rc == 7, "an owner route returns the Core ML error");
    CHECK(ggml_runs == 0, "an owner route never falls back to ggml (no weights)");
    CHECK(!used, "the failed dispatch did not run on Core ML");
    CHECK(retires.load() == 0, "an owner route never retires");

    // Ineligible call shapes (per-stage captures, EOU mismatches) run on ggml
    // without touching the route.
    FakeEncoder ok;
    EncoderRoute::Ops ok_ops;
    ok_ops.encode = &fake_encode;
    const EncoderRoute held = EncoderRoute::coreml(reinterpret_cast<parakeet_coreml_context *>(&ok),
                                                   fake_describe(nullptr, 128), kFakeLabel, nullptr,
                                                   ok_ops, false);
    ggml_runs = 0;
    dispatch_on_route(held, EncoderEligibility::ggml, [&] { return 0; }, [&] { ++ggml_runs; return 0; }, used);
    CHECK(ggml_runs == 1 && !used && held.usable(), "an ineligible dispatch runs on ggml, route intact");
}

// One dispatch of `n_mel` frames as run_encoder drives it: the capacity check
// first (coreml_encoder_eligibility), then the route.
struct StrictOutcome {
    bool threw = false;
    pcm::SlotState error_state = pcm::SlotState::idle;
    pcm::ErrorClass error_class = pcm::ErrorClass::none;
    int ggml_runs = 0;
    bool used = false;
};

StrictOutcome dispatch_strict(const EncoderRoute & route, int n_mel) {
    StrictOutcome out;
    const EncoderEligibility e = route_fits_capacity(route, n_mel) ? EncoderEligibility::coreml
                                                                   : EncoderEligibility::cannot_serve;
    try {
        dispatch_on_route(
            route, e, [&] { return route.encode(n_mel, n_mel, 128, nullptr, n_mel / kSub, 1024, nullptr); },
            [&] { ++out.ggml_runs; return 0; }, out.used);
    } catch (const pcm::CoremlError & err) {
        out.threw = true;
        out.error_state = err.state;
        out.error_class = err.cls;
    }
    return out;
}

void case_strict_route() {
    const fs::path dir = make_temp_dir("strict");
    auto m = pcm::Manager::create(fast_config());
    auto slot = make_encoder_slot(*m, &fake_load, pcm::Wait::call, dir);

    // Long-form windowing disabled (a negative window request): the plan is
    // one pass over the whole input, even with a held fixed-capacity route.
    EncoderRoute route = route_from_acquired(slot->acquire(pcm::Demand::now), fake_ops(), 128,
                                             /*strict=*/true);
    CHECK(route.usable() && route.strict(), "a strict acquisition gives a strict route");
    const LongFormPlan single = routed_plan(route, kLongMel, CoremlShapeKind::fixed, -1);
    CHECK(!single.enabled, "windowing disabled: one pass over the whole input");
    CHECK(!route_fits_capacity(route, (int) kLongMel), "the single pass exceeds the fixed capacity");

    const StrictOutcome over = dispatch_strict(route, (int) kLongMel);
    CHECK(over.threw, "strict: an over-capacity input throws instead of falling back");
    CHECK(over.ggml_runs == 0 && !over.used, "strict: nothing ran on ggml");
    CHECK(over.error_state == pcm::SlotState::ready && over.error_class == pcm::ErrorClass::deterministic,
          "strict: a capacity miss is a deterministic error on a ready slot");
    CHECK(route.usable() && slot->state() == pcm::SlotState::ready,
          "strict: a capacity miss does not retire the slot");

    // The same input on a non-strict route of the same slot runs on ggml.
    const EncoderRoute relaxed = route_from_acquired(slot->acquire(pcm::Demand::now), fake_ops(), 128);
    CHECK(!relaxed.strict(), "the default route is not strict");
    const StrictOutcome fallback = dispatch_strict(relaxed, (int) kLongMel);
    CHECK(!fallback.threw && fallback.ggml_runs == 1 && !fallback.used,
          "automatic: the over-capacity input runs on ggml");

    // An input that fits runs on Core ML.
    const StrictOutcome fits = dispatch_strict(route, kFixedMelFrames);
    CHECK(!fits.threw && fits.used && fits.ggml_runs == 0, "strict: an input that fits runs on Core ML");

    // The handle refuses the shape (rc 1-4): throws, the slot stays ready.
    auto * enc = reinterpret_cast<FakeEncoder *>(route.context());
    enc->fail_rc = 2;
    enc->fail_at = enc->encodes.load();
    const StrictOutcome refused = dispatch_strict(route, kFixedMelFrames);
    CHECK(refused.threw && refused.ggml_runs == 0, "strict: a cannot-serve code throws");
    CHECK(slot->state() == pcm::SlotState::ready, "strict: a cannot-serve code does not retire");

    // A failed prediction (rc 5-7): retires, then throws.
    enc->fail_rc = 6;
    enc->fail_at = enc->encodes.load();
    const StrictOutcome failed = dispatch_strict(route, kFixedMelFrames);
    CHECK(failed.threw && failed.ggml_runs == 0, "strict: a failed prediction throws");
    CHECK(failed.error_state == pcm::SlotState::retired && failed.error_class == pcm::ErrorClass::transient,
          "strict: the error reports the retirement");
    CHECK(slot->state() == pcm::SlotState::retired, "strict: a failed prediction retires the slot");

    // The owner route keeps its own contract: it returns the error.
    FakeEncoder owned;
    EncoderRoute::Ops ops;
    ops.encode = &fake_encode;
    const EncoderRoute owner = EncoderRoute::coreml(reinterpret_cast<parakeet_coreml_context *>(&owned),
                                                    fake_describe(nullptr, 128), kFakeLabel, nullptr, ops,
                                                    /*owner=*/true, /*strict=*/true);
    const StrictOutcome owner_over = dispatch_strict(owner, (int) kLongMel);
    CHECK(!owner_over.threw && owner_over.ggml_runs == 1,
          "an owner route hands an over-capacity input to the ggml path, which reports the error");
    fs::remove_all(dir);
}

void case_pending_route() {
    const fs::path dir = make_temp_dir("pending");
    auto m = pcm::Manager::create(fast_config());
    auto slot = make_encoder_slot(*m, &gated_load, pcm::Wait::none, dir);
    slot->request(pcm::Priority::minimal);

    const EncoderRoute first = route_from_acquired(slot->acquire(pcm::Demand::now), fake_ops(), 128);
    CHECK(!first.holds(), "a slot still loading gives no handle (wait = none never blocks)");
    CHECK(first.pending(), "the call is marked coreml_pending");
    CHECK(same_plan(routed_plan(first, kLongMel), ggml_plan(kLongMel)), "pending -> the ggml plan");
    CHECK(run_call(first, 2).label == "ggml", "pending call reports ggml");

    open_gate();
    // A unit that asks to wait (wait = call) gets the handle once ready.
    auto waiting = make_encoder_slot(*m, &fake_load, pcm::Wait::call, dir);
    const EncoderRoute later = route_from_acquired(waiting->acquire(pcm::Demand::now), fake_ops(), 128);
    CHECK(later.holds() && !later.pending(), "wait = call returns the ready handle");
    fs::remove_all(dir);
}

void case_sidecar_groups() {
    const fs::path dir = make_temp_dir("groups");
    fs::create_directories(dir / "m-encoder.mlmodelc" / "weights");
    {
        std::ofstream(dir / "m-encoder.mlmodelc" / "weights" / "weight.bin") << std::string(1000, 'x');
    }
    auto m = pcm::Manager::create(fast_config());
    const bool routable = m->platform() != pcm::Platform::other;

    auto request = [&](const char * stage, const char * name, bool present, bool minimal) {
        SidecarRequest r;
        r.stage = stage;
        r.path = (dir / name).string();
        r.present = present;
        r.stage_default = pcm::ComputeUnits::cpu_and_ane;
        r.load_fn = &fake_load;
        r.minimal = minimal;
        return r;
    };

    CoremlOptions opts;
    opts.wait = CoremlOptions::Wait::construct;
    opts.prewarm = CoremlOptions::Prewarm::predicted;
    {
        CoremlSidecars sidecars(opts, m);
        auto enc = sidecars.add(request("encoder", "m-encoder.mlmodelc", true, true));
        auto d745 = sidecars.add(request("diarizer-745", "m-diarizer-745.mlmodelc", true, false));
        auto gone = sidecars.add(request("bypass-encoder", "m-missing.mlmodelc", false, true));
        CHECK(gone->spec().route == pcm::Route::none, "an absent sidecar has no route");
        CHECK(enc->spec().route == (routable ? pcm::Route::prebuilt : pcm::Route::none),
              "a present sidecar is prebuilt on Apple platforms");
        CHECK(enc->spec().ane_compile_reserve_bytes == (routable ? 1600u : 0u),
              "cold ANE loads reserve 1.6x the weights");
        sidecars.start();  // wait = construct: returns after the group finished
        CHECK(enc->state() == (routable ? pcm::SlotState::ready : pcm::SlotState::unsupported),
              "the minimal encoder is ready after construction");
        CHECK(d745->state() == (routable ? pcm::SlotState::idle : pcm::SlotState::unsupported),
              "a non-minimal capacity is not requested by the predicted scope");
        const std::string construction = sidecars.construction_json();
        CHECK(construction.find("\"skipped_route_none\"") != std::string::npos, "group JSON");
        CHECK(construction.find("bypass-encoder") != std::string::npos,
              "the absent sidecar is listed as skipped");
        // Off Apple platforms every route is none: an empty group, which counts
        // as success (nothing_to_prepare).
        CHECK(sidecars.wait_construction(1000), "the construction group succeeded");

        const pcm::PrewarmResult all = sidecars.prewarm(CoremlOptions::Prewarm::all, /*wait=*/true);
        CHECK(all.success && all.nothing_to_prepare == !routable, "prewarm all loads every routable sidecar");
        CHECK(d745->state() == (routable ? pcm::SlotState::ready : pcm::SlotState::unsupported),
              "prewarm all reaches the larger capacity");
        const std::string status = sidecars.status_json();
        CHECK(status.find("\"engine\":\"parakeet\"") != std::string::npos ||
                  status.find("\"engine\": \"parakeet\"") != std::string::npos,
              "status names the engine");
        CHECK(status.find("diarizer-745") != std::string::npos, "status lists every slot");
    }
    {
        CoremlOptions none = opts;
        none.prewarm = CoremlOptions::Prewarm::none;
        CoremlSidecars sidecars(none, m);
        auto enc = sidecars.add(request("encoder", "m-encoder.mlmodelc", true, true));
        sidecars.start();
        CHECK(sidecars.construction_json().find("\"nothing_to_prepare\":true") != std::string::npos ||
                  sidecars.construction_json().find("\"nothing_to_prepare\": true") != std::string::npos,
              "prewarm none with construct returns at once with an empty group");
    }
    {
        CoremlOptions off = opts;
        off.use = CoremlOptions::Use::off;
        CoremlSidecars sidecars(off, m);
        auto enc = sidecars.add(request("encoder", "m-encoder.mlmodelc", true, true));
        CHECK(enc->spec().route == pcm::Route::none, "use = off gives no route");
        CHECK(enc->state() == pcm::SlotState::disabled, "use = off reports disabled");
    }
    {
        CoremlOptions budget = opts;
        budget.memory_budget_bytes = 1;
        CoremlSidecars sidecars(budget, m, /*memory_degraded=*/true);
        CHECK(!sidecars.resolved().enabled() && sidecars.resolved().disabled_reason == "memory",
              "memory-degrade resolves use = off with reason memory");
    }
    fs::remove_all(dir);
}

void case_options() {
    CoremlOptions o;
    o.use = CoremlOptions::Use::require;
    o.generate = CoremlOptions::Generate::never;
    o.wait = CoremlOptions::Wait::call;
    o.prewarm = CoremlOptions::Prewarm::all;
    o.cache_dir = "/tmp/x";
    o.compute_units = "cpu_only";
    o.strict_wait_ms = 5;
    o.memory_budget_bytes = 9;
    const pcm::Options r = to_runtime_options(o);
    CHECK(r.use == pcm::Use::require && r.generate == pcm::Generate::never && r.wait == pcm::Wait::call &&
              r.prewarm == pcm::Prewarm::all,
          "enums map member for member");
    CHECK(r.cache_dir == o.cache_dir && r.compute_units == o.compute_units &&
              r.strict_wait_ms == o.strict_wait_ms && r.memory_budget_bytes == o.memory_budget_bytes,
          "optional fields copy through");
    const pcm::Resolved resolved = resolve_parakeet_coreml(o);
    CHECK(resolved.strict() && resolved.compute_units == pcm::ComputeUnits::cpu_only,
          "explicit fields resolve");
    CHECK(!resolve_parakeet_coreml(CoremlOptions{}).strict() || std::getenv("PARAKEET_COREML_STRICT"),
          "defaults resolve to automatic");

    // The synchronous loaders (TDT decoder, encoder owner) take the resolved
    // units as an argument: the explicit option over the environment, the
    // environment when nothing is set, null (the loader's stage default)
    // otherwise.
    const char * saved = std::getenv("PARAKEET_COREML_COMPUTE_UNITS");
    const std::string saved_value = saved ? saved : "";
    setenv("PARAKEET_COREML_COMPUTE_UNITS", "all", 1);
    CoremlOptions explicit_units;
    explicit_units.compute_units = "cpu_only";
    const char * arg = loader_units(resolve_parakeet_coreml(explicit_units));
    CHECK(arg != nullptr && std::string(arg) == "cpu_only", "the explicit option reaches the loader");
    arg = loader_units(resolve_parakeet_coreml(CoremlOptions{}));
    CHECK(arg != nullptr && std::string(arg) == "all", "the environment reaches the loader through resolution");
    unsetenv("PARAKEET_COREML_COMPUTE_UNITS");
    CHECK(loader_units(resolve_parakeet_coreml(CoremlOptions{})) == nullptr,
          "unset units leave the loader's own default");
    if (saved) setenv("PARAKEET_COREML_COMPUTE_UNITS", saved_value.c_str(), 1);
}

void write_text(const fs::path & p, const std::string & text) {
    fs::create_directories(p.parent_path());
    std::ofstream(p) << text;
}

void case_fit_stable() {
    const fs::path dir = make_temp_dir("fit");
    const fs::path sidecar = dir / "m-encoder.mlmodelc";
    write_text(sidecar / "weights" / "weight.bin", std::string(4096, 'w'));
    const fs::path cache = dir / "cache";

    auto project = [&](const CoremlOptions & o) {
        const pcm::Resolved resolved = resolve_parakeet_coreml(o);
        const CoremlEncoderFitRoute route = coreml_encoder_fit_route(sidecar.string(), resolved.enabled());
        CoremlEncoderEnvelopeInputs in;
        in.weight_file_bytes = route.weight_file_bytes;
        in.encoder_parameters = 3000;
        in.io_bytes = 100;
        in.units = CoremlFitUnits::cpu_and_ane;
        FitResult r;
        r.device_free_bytes = 15000;  // ggml-only (10000) fits; + Core ML (9080) does not
        r.device_shares_host_memory = true;
        r.coreml_bytes = 50;
        apply_coreml_verdicts(r, /*required=*/10050, resolved.enabled(), resolved.disabled_reason,
                              route.prebuilt, route.prebuilt ? coreml_encoder_envelope_bytes(in) : 0);
        return r;
    };
    auto same = [](const FitResult & a, const FitResult & b) {
        return a.ggml_required_bytes == b.ggml_required_bytes && a.ggml_only_fits == b.ggml_only_fits &&
               a.coreml.fits == b.coreml.fits && a.coreml.bytes == b.coreml.bytes &&
               a.coreml.encoder_route == b.coreml.encoder_route &&
               a.coreml.disabled_reason == b.coreml.disabled_reason;
    };

    CoremlOptions o;
    o.cache_dir = cache.string();
    const FitResult no_hint = project(o);
    CHECK(no_hint.coreml.encoder_route == "prebuilt", "an adjacent sidecar gives the prebuilt route");
    // runtime weights max(4096, 2 x 3000) = 6000; I/O 2 x 100; transient 4096/4;
    // margin x1.25 -> (6000 + 200 + 1024) * 1.25 = 9030; plus the diarizers' 50.
    CHECK(no_hint.coreml.bytes == 9030 + 50, "envelope arithmetic (default factors)");
    CHECK(no_hint.ggml_required_bytes == 10000, "the ggml-only charge excludes the diarizer sidecars");
    CHECK(no_hint.ggml_only_fits && !no_hint.coreml.fits, "two verdicts");

    // Non-ANE units charge a 1.0x load transient (measured on TDT 0.6B fp16):
    // (6000 + 200 + 4096) * 1.25 = 12870 for all, cpu_and_gpu and cpu_only.
    for (CoremlFitUnits units : {CoremlFitUnits::all, CoremlFitUnits::cpu_and_gpu, CoremlFitUnits::cpu_only}) {
        CoremlEncoderEnvelopeInputs in;
        in.weight_file_bytes = 4096;
        in.encoder_parameters = 3000;
        in.io_bytes = 100;
        in.units = units;
        CHECK(coreml_encoder_envelope_bytes(in) == 12870, "non-ANE envelope charges 1.0x weight.bin");
    }
    // The fp16 TDT 0.6B case the update measured: weight.bin 1.2 GB, about
    // 1.2 GB resident + 1.08 GiB transient. The all-units envelope covers it.
    {
        CoremlEncoderEnvelopeInputs in;
        in.weight_file_bytes = 1200000000ull;
        in.units = CoremlFitUnits::all;
        const uint64_t measured = 1200000000ull + 1080ull * 1024 * 1024 * 1024 / 1000;
        CHECK(coreml_encoder_envelope_bytes(in) >= measured,
              "the all-units envelope is not below the measured fp16 peak");
    }

    // A real warm hint for this artifact, written by a completed load.
    {
        auto m = pcm::Manager::create(fast_config());
        pcm::SlotSpec spec;
        spec.engine = "parakeet";
        spec.stage = "encoder";
        spec.units = pcm::ComputeUnits::cpu_and_ane;
        spec.route = pcm::Route::prebuilt;
        spec.artifact_path = sidecar.string();
        spec.load_fn = &fake_load;
        pcm::SlotPolicy policy;
        policy.wait = pcm::Wait::call;
        policy.cache_dir = cache.string();
        auto slot = m->make_slot(spec, policy);
        CHECK(static_cast<bool>(slot->acquire(pcm::Demand::now).model), "the hinting load completes");
        CHECK(fs::exists(slot->hint_path()), "the load wrote its hint");
        const FitResult warm = project(o);
        CHECK(same(warm, no_hint), "fit with a non-stale hint equals fit with none");

        // Stale it: stale_until a day ahead.
        std::ifstream in(slot->hint_path());
        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        const auto pos = text.find("\"stale_until\"");
        CHECK(pos != std::string::npos, "hint record has stale_until");
        if (pos != std::string::npos) {
            const auto colon = text.find(':', pos);
            const auto end = text.find_first_of(",}", colon);
            text.replace(colon + 1, end - colon - 1, "\"2999-01-01T00:00:00Z\"");
            std::ofstream(slot->hint_path()) << text;
        }
        CHECK(slot->hint().has_value() && slot->hint()->stale, "the hint now reads as stale");
        CHECK(same(project(o), no_hint), "fit with a stale hint equals fit with none");
    }

    CoremlOptions off = o;
    off.use = CoremlOptions::Use::off;
    const FitResult disabled = project(off);
    CHECK(disabled.coreml.encoder_route == "none" && disabled.coreml.bytes == 0 &&
              disabled.coreml.disabled_reason == "use=off" && disabled.coreml.fits == disabled.ggml_only_fits,
          "use = off: no Core ML charge, the ggml verdict");
    fs::remove_all(dir);
}

// The generated-diarizer charge: which capacities a range loads (prebuilt
// ones included in the selection, never charged), resident sum, one
// transient, the largest margin among the loaded ones.
void case_generated_diarizer_envelope() {
    auto gen = [](int frames, uint64_t resident, uint64_t transient, double margin) {
        return CoremlDiarizerCharge{frames, true, resident, transient, margin};
    };
    const std::vector<CoremlDiarizerCharge> caps = {
        gen(1125, 300, 40, 1.5), CoremlDiarizerCharge{200}, gen(358, 100, 30, 1.25), gen(745, 200, 50, 1.25)};
    auto env = [&](std::vector<CoremlFrameRange> ranges) { return coreml_generated_diarizer_envelope(caps, ranges); };

    CHECK(env({}) == 0, "no ranges, no charge");
    CHECK(env({{1, 150}}) == 0, "a range only the prebuilt capacity covers charges nothing");
    // 200 (prebuilt) and 358: (100 + 30) x 1.25 = 162.5 -> 163.
    CHECK(env({{1, 358}}) == 163, "one generated capacity: resident + transient, margin, rounded up");
    // 358, 745, 1125: (100 + 200 + 300 + max(30, 50, 40)) x 1.5 = 975.
    CHECK(env({{300, 900}}) == 975, "resident sum, one transient, the largest margin");
    // Above every capacity: from the first covering capacity to the largest.
    CHECK(env({{400, 5000}}) == 825, "a range over the largest capacity loads every larger one");
    // Two ranges: 200 (prebuilt) and 745 -> (200 + 50) x 1.25 = 312.5 -> 313.
    CHECK(env({{1, 150}, {700, 745}}) == 313, "ranges are unioned; capacities between them are not loaded");
    // The prebuilt capacity shifts the selection: without it a 1..150 range loads 358.
    const std::vector<CoremlDiarizerCharge> generated_only = {caps[0], caps[2], caps[3]};
    CHECK(coreml_generated_diarizer_envelope(generated_only, {{1, 150}}) == 163,
          "without the prebuilt capacity the smallest generated one covers the range");
    // Saturates instead of wrapping.
    const std::vector<CoremlDiarizerCharge> huge = {gen(358, UINT64_MAX - 1, UINT64_MAX - 1, 1.25)};
    CHECK(coreml_generated_diarizer_envelope(huge, {{1, 358}}) == UINT64_MAX, "saturating arithmetic");

    // prewarm = all loads every capacity at construction, whatever the calls
    // need: every generated capacity is charged, prebuilt ones never are.
    // (100 + 200 + 300 + max(30, 50, 40)) x 1.5 = 975.
    auto all = [&](std::vector<CoremlFrameRange> ranges) {
        return coreml_generated_diarizer_envelope(caps, ranges, /*every_capacity=*/true);
    };
    CHECK(all({{1, 358}}) == 975, "prewarm all: a short range still charges every generated capacity");
    CHECK(all({}) == 975, "prewarm all: no ranges, every generated capacity");
    CHECK(all({{1, 150}}) == 975, "prewarm all: a range only the prebuilt capacity covers");
    CHECK(coreml_generated_diarizer_envelope({CoremlDiarizerCharge{200}}, {}, true) == 0,
          "prewarm all: prebuilt capacities alone charge nothing");
}

// coreml.validated: true only when the charged bytes are all generated
// envelopes. A prebuilt diarizer or encoder charge in the bytes is today's
// unvalidated estimate.
void case_coreml_verdicts_validated() {
    auto verdict = [](uint64_t prebuilt_diarizers, bool encoder_prebuilt, uint64_t generated) {
        FitResult r;
        r.device_shares_host_memory = true;
        r.device_free_bytes = uint64_t(1) << 40;
        r.coreml_bytes = prebuilt_diarizers;
        apply_coreml_verdicts(r, 1000 + prebuilt_diarizers, true, "", encoder_prebuilt, encoder_prebuilt ? 77 : 0,
                              generated);
        return r;
    };
    const FitResult generated_only = verdict(0, false, 500);
    CHECK(generated_only.coreml.validated && generated_only.coreml.bytes == 500,
          "validated: only generated capacities are charged");
    const FitResult with_prebuilt = verdict(300, false, 500);
    CHECK(!with_prebuilt.coreml.validated && with_prebuilt.coreml.bytes == 800,
          "not validated: a prebuilt diarizer charge is in the bytes");
    const FitResult with_encoder = verdict(0, true, 500);
    CHECK(!with_encoder.coreml.validated && with_encoder.coreml.bytes == 577,
          "not validated: a prebuilt encoder envelope is in the bytes");
    CHECK(!verdict(0, false, 0).coreml.validated, "not validated: nothing generated is charged");
}

}  // namespace


#if defined(QVAC_COREML_HAVE_GGML)
// ---- generated sidecar ----------------------------------------------------------------

// The fake on-device compile: a "bundle" holding the package's weights.
std::string fake_compile(const std::string & pkg, const std::string & out) {
    std::error_code ec;
    fs::create_directories(fs::path(out) / "weights", ec);
    fs::copy_file(fs::path(pkg) / "Data/com.apple.CoreML/weights/weight.bin", fs::path(out) / "weights/weight.bin", ec);
    if (ec) return ec.message();
    write_text(fs::path(out) / "model.mil", "fake program");
    write_text(fs::path(out) / "coremldata.bin", "fake");
    return {};
}

std::atomic<int> g_generated_loads{0};
std::atomic<int> g_generated_leases{0};

// Loads like an engine's LoadFn: under the artifact's lease, and the
// "bundle" must hold the generated weights.
pcm::LoadResult generated_load(const pcm::LoadRequest & req) {
    pcm::LoadResult r;
    if (!pcm::generation::begin_load(req, r)) return r;
    if (r.lease_fd >= 0) ++g_generated_leases;
    if (fs::file_size(fs::path(req.artifact_path) / "weights/weight.bin") == 4064) {
        ++g_generated_loads;
        r.model = std::make_shared<FakeEncoder>();
    } else {
        r.error = pcm::Error::deterministic("not the generated bundle");
    }
    pcm::generation::end_load(req, r);
    return r;
}

// The toy-stateless template's GGUF: architecture qvac-toy, kind stateless.
void write_toy_gguf(const fs::path & path, const char * kind, float scale) {
    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "qvac-toy");
    gguf_set_val_str(g, "qvac_toy.kind", kind);
    ggml_init_params p{ggml_tensor_overhead() * 8 + 64 * 1024, nullptr, false};
    ggml_context * ctx = ggml_init(p);
    auto add = [&](const char * name, int64_t ne0, int64_t ne1) {
        ggml_tensor * t = ne1 > 1 ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1)
                                  : ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0);
        ggml_set_name(t, name);
        float * d = static_cast<float *>(t->data);
        for (int64_t i = 0; i < ggml_nelements(t); ++i) d[i] = scale * static_cast<float>((i % 17) - 8) / 16.0f;
        gguf_add_tensor(g, t);
    };
    add("toy.w1", 8, 32);
    add("toy.b1", 32, 1);
    add("toy.w2", 48, 32);
    add("toy.b2", 48, 1);
    gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
}

void case_generated_sidecar() {
    const fs::path dir = make_temp_dir("generated");
    // The toy template, validated for cpu_and_ane, under a development template directory.
    const fs::path templates = dir / "templates";
    fs::create_directories(templates);
    fs::copy(fs::path(QVAC_COREML_TOY_TEMPLATES) / "toy-stateless", templates / "toy-stateless");
    {
        std::ifstream in(templates / "toy-stateless" / "template.json");
        std::string j((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        const std::string from = "\"validated_for\": []";
        const size_t at = j.find(from);
        CHECK(at != std::string::npos, "the toy template lists validated_for");
        if (at == std::string::npos) return;
        j.replace(at, from.size(), "\"validated_for\": [{\"units\": \"cpu_and_ane\", \"os\": \"test\", \"chip\": \"test\"}]");
        write_text(templates / "toy-stateless" / "template.json", j);
    }
    ::setenv("QVAC_COREML_TEMPLATE_DIR", templates.c_str(), 1);
    const fs::path gguf = dir / "m.gguf";
    write_toy_gguf(gguf, "stateless", 1.0f);
    const fs::path other = dir / "other.gguf";
    write_toy_gguf(other, "multi", 1.0f);
    pcm::generation::set_compile_override(&fake_compile);

    pcm::Config cfg = fast_config();
    cfg.platform = pcm::Platform::macos;
    cfg.os = pcm::OsVersion{15, 7, 0};
    auto m = pcm::Manager::create(cfg);

    CoremlOptions opts;
    opts.cache_dir = (dir / "cache").string();
    opts.wait = CoremlOptions::Wait::construct;
    opts.prewarm = CoremlOptions::Prewarm::all;

    auto request_for = [&](const pcm::generation::Candidate * c, const fs::path & model) {
        SidecarRequest r;
        r.stage = "diarizer-358";
        r.path = (dir / "m-diarizer-358.mlmodelc").string();  // absent: no prebuilt
        r.present = false;
        r.stage_default = pcm::ComputeUnits::cpu_and_ane;
        r.load_fn = &generated_load;
        r.geometry = {358};
        r.minimal = true;
        r.generated = c;
        pcm::generation::source_stat(model.string(), r.source);
        return r;
    };

    std::vector<pcm::generation::Candidate> probed;
    uint64_t envelope_before = 0;
    {
        CoremlSidecars sidecars(opts, m);
        CHECK(sidecars.generation_status().supported, "generation is supported with a cache directory on macOS 15");
        probed = pcm::generation::probe({"toy-stateless", "no-such-template"}, gguf.string(), sidecars.resolved());
        CHECK(probed.size() == 2 && probed[0].available && probed[0].matched, "the template matches its GGUF");
        CHECK(probed.size() == 2 && !probed[1].available, "an unknown template is unavailable");
        if (probed.size() != 2 || !probed[0].matched) return;
        CHECK(probed[0].memory.weight_bin_size == 4064 && probed[0].memory.validated_units.size() == 1,
              "the template's memory section is read");
        envelope_before = pcm::generation::envelope_bytes(probed[0].memory, pcm::ComputeUnits::cpu_and_ane);
        // resident 4064 + 2 x (8 + 48) x 4 = 4512; transient max(generation peak + 8 MiB + 2032, 1016); x1.25.
        const double transient = static_cast<double>(probed[0].memory.generation_peak_bytes) + 8388608.0 + 2032.0;
        CHECK(envelope_before == static_cast<uint64_t>(std::ceil((4512.0 + transient) * 1.25)),
              "envelope arithmetic (default factors)");
        auto slot = sidecars.add(request_for(&probed[0], gguf));
        CHECK(slot->spec().route == pcm::Route::generate, "no prebuilt and a matching template: route generate");
        CHECK(slot->spec().artifact_path.empty() && slot->spec().prepare_fn != nullptr, "the prepare job supplies the path");
        CHECK(slot->spec().ane_compile_reserve_bytes == static_cast<uint64_t>(std::ceil(4064 * 1.6)),
              "the cold-compile reserve comes from the template");
        sidecars.start();
        CHECK(slot->state() == pcm::SlotState::ready, "generated, compiled and loaded during construction");
        const pcm::SlotStatus st = slot->status();
        CHECK(st.route == pcm::StatusRoute::generated, "status route generated");
        CHECK(st.path.find("qvac-coreml-v1/artifacts/toy-stateless.") != std::string::npos, "the artifact is in the cache");
        CHECK(g_generated_loads == 1 && g_generated_leases == 1, "the load held the artifact's lease");
        CHECK(sidecars.status_json().find("\"generated\"") != std::string::npos, "status JSON");

        CHECK(sidecars.generation_status().supported, "a registered generated route keeps generation supported");

        // Unvalidated units and a mismatching GGUF give no route, and a
        // status-only slot that status and the construction group list.
        CoremlOptions gpu = opts;
        gpu.compute_units = "cpu_and_gpu";
        CoremlSidecars gpu_sidecars(gpu, m);
        CHECK(gpu_sidecars.plan(request_for(&probed[0], gguf)).route == pcm::Route::none &&
                  gpu_sidecars.plan(request_for(&probed[0], gguf)).reason == "unvalidated_units",
              "units the template was not validated for suppress generation");
        gpu_sidecars.add(request_for(&probed[0], gguf));
        CHECK(gpu_sidecars.slots().size() == 1 && gpu_sidecars.slots()[0]->spec().route == pcm::Route::none,
              "a suppressed route keeps a status-only slot");
        CHECK(!gpu_sidecars.generation_status().supported &&
                  gpu_sidecars.generation_status().reason == "unvalidated_units",
              "every generated route suppressed for the units: generation reason unvalidated_units");
        const std::string gpu_status = gpu_sidecars.status_json();
        CHECK(gpu_status.find("\"generation\":{\"reason\":\"unvalidated_units\",\"supported\":false}") !=
                  std::string::npos,
              "status JSON: generation.reason unvalidated_units");
        CHECK(gpu_status.find("\"stage\":\"diarizer-358\"") != std::string::npos &&
                  gpu_status.find("\"state\":\"unsupported(unvalidated_units)\"") != std::string::npos,
              "status JSON: the slot reports unsupported(unvalidated_units)");
        gpu_sidecars.start();
        const std::string gpu_group = gpu_sidecars.construction_json();
        CHECK(gpu_group.find("\"slots\":[]") != std::string::npos &&
                  gpu_group.find("\"skipped_route_none\":[{") != std::string::npos &&
                  gpu_group.find("unsupported(unvalidated_units)") != std::string::npos,
              "construction JSON: the suppressed slot is skipped, nothing is loaded");

        const std::vector<pcm::generation::Candidate> mismatch =
            pcm::generation::probe({"toy-stateless"}, other.string(), sidecars.resolved());
        CHECK(!mismatch.empty() && mismatch[0].available && !mismatch[0].matched, "another GGUF does not match");
        if (mismatch.empty()) return;
        CHECK(sidecars.plan(request_for(&mismatch[0], other)).none_state == pcm::SlotState::no_template,
              "a mismatching GGUF reports no_template");
        CoremlSidecars other_sidecars(opts, m);
        CHECK(other_sidecars.add(request_for(&mismatch[0], other))->spec().route == pcm::Route::none,
              "a mismatching GGUF gets a status-only slot");
        CHECK(other_sidecars.generation_status().supported && other_sidecars.generation_status().reason.empty(),
              "no_template is not a units suppression: generation stays supported");
        const std::string other_status = other_sidecars.status_json();
        CHECK(other_status.find("\"generation\":{\"reason\":null,\"supported\":true}") != std::string::npos &&
                  other_status.find("\"state\":\"no_template\"") != std::string::npos,
              "status JSON: the slot reports no_template");
        other_sidecars.start();
        const std::string other_group = other_sidecars.construction_json();
        CHECK(other_group.find("\"slots\":[]") != std::string::npos &&
                  other_group.find("\"skipped_route_none\":[{") != std::string::npos &&
                  other_group.find("no_template") != std::string::npos,
              "construction JSON: the no_template slot is skipped");
        CoremlOptions no_cache = opts;
        no_cache.cache_dir = "";
        CoremlSidecars no_cache_sidecars(no_cache, m);
        CHECK(!no_cache_sidecars.generation_status().supported &&
                  no_cache_sidecars.generation_status().reason == "no_cache_dir",
              "no cache directory, no generation");
        CHECK(no_cache_sidecars.plan(request_for(&probed[0], gguf)).route == pcm::Route::none, "and no route");

        // Near the budget: the ggml-only fit passes and the Core ML-inclusive
        // one does not, so the instance is memory-degraded (use = off).
        CoremlOptions budgeted = opts;
        budgeted.memory_budget_bytes = int64_t(1) << 30;
        CHECK(CoremlSidecars(budgeted, m).generation_status().supported, "a budget alone leaves generation on");
        CoremlSidecars degraded(budgeted, m, /*memory_degraded=*/true);
        CHECK(!degraded.generation_status().supported && degraded.generation_status().reason == "memory",
              "memory-degraded: generation is off, reason memory");
        CHECK(degraded.plan(request_for(&probed[0], gguf)).route == pcm::Route::none &&
                  degraded.plan(request_for(&probed[0], gguf)).none_state == pcm::SlotState::disabled,
              "memory-degraded: no generate route");

        // The fit's per-capacity charge equals the template's envelope when one capacity loads.
        const pcm::generation::TemplateMemory & mem = probed[0].memory;
        const std::vector<CoremlDiarizerCharge> one = {
            {358, true, pcm::generation::resident_bytes(mem),
             pcm::generation::transient_bytes(mem, pcm::ComputeUnits::cpu_and_ane), mem.margin}};
        CHECK(coreml_generated_diarizer_envelope(one, {{1, 358}}) == envelope_before,
              "the fit charge of one generated capacity is the template's envelope");
    }
    {
        // The next instance finds the artifact: route generate again (plan()
        // never reads the cache), status route cached, nothing compiled.
        CoremlSidecars sidecars(opts, m);
        const std::vector<pcm::generation::Candidate> again =
            pcm::generation::probe({"toy-stateless"}, gguf.string(), sidecars.resolved());
        CHECK(!again.empty() && again[0].matched, "the template still matches");
        if (again.empty()) return;
        CHECK(pcm::generation::envelope_bytes(again[0].memory, pcm::ComputeUnits::cpu_and_ane) == envelope_before,
              "fit-stable: the generated envelope is the same once the artifact exists");
        auto slot = sidecars.add(request_for(&again[0], gguf));
        CHECK(slot->spec().route == pcm::Route::generate, "fit-stable: the route is the same once the artifact exists");
        sidecars.start();
        CHECK(slot->state() == pcm::SlotState::ready && slot->status().route == pcm::StatusRoute::cached,
              "the second instance adopts the cached artifact");
        CHECK(g_generated_loads == 2, "loaded again from the cache");
    }
    pcm::generation::set_compile_override(nullptr);
    ::unsetenv("QVAC_COREML_TEMPLATE_DIR");
}

// The diarizer capacities of a model: fit and discovery choose the same ones
// (diarizer_capacities + plan); a template capacity that is not generated
// keeps a status-only slot, which the construction group lists (every one of
// them when generation is off), which never serves a pass, and which under
// use = require refuses a pass it covers instead of letting it run on ggml.
void case_diarizer_capacities() {
    const fs::path dir = make_temp_dir("diarizers");
    const fs::path templates = dir / "templates";
    fs::create_directories(templates);
    fs::copy(fs::path(QVAC_COREML_TOY_TEMPLATES) / "toy-stateless", templates / "toy-stateless");
    {
        std::ifstream in(templates / "toy-stateless" / "template.json");
        std::string j((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        const std::string from = "\"validated_for\": []";
        const size_t at = j.find(from);
        CHECK(at != std::string::npos, "the toy template lists validated_for");
        if (at == std::string::npos) return;
        j.replace(at, from.size(), "\"validated_for\": [{\"units\": \"cpu_and_ane\", \"os\": \"test\", \"chip\": \"test\"}]");
        write_text(templates / "toy-stateless" / "template.json", j);
    }
    ::setenv("QVAC_COREML_TEMPLATE_DIR", templates.c_str(), 1);
    const fs::path gguf = dir / "m.gguf";
    write_toy_gguf(gguf, "stateless", 1.0f);
    const fs::path other = dir / "other.gguf";
    write_toy_gguf(other, "multi", 1.0f);
    pcm::generation::set_compile_override(&fake_compile);

    pcm::Config cfg = fast_config();
    cfg.platform = pcm::Platform::macos;
    cfg.os = pcm::OsVersion{15, 7, 0};
    auto m = pcm::Manager::create(cfg);

    CoremlOptions base;
    base.cache_dir = (dir / "cache").string();
    const std::vector<int> template_frames = {358, 745, 1125};
    const std::vector<std::string> ids(3, "toy-stateless");
    const std::vector<int> prebuilt = {745};

    struct Config {
        const char * name;
        CoremlOptions opts;
        fs::path model;
        std::vector<int> generated;  // capacities with route generate
    };
    std::vector<Config> configs;
    configs.push_back({"validated units", base, gguf, {358, 1125}});
    {
        CoremlOptions o = base;
        o.compute_units = "cpu_and_gpu";
        configs.push_back({"unvalidated units", o, gguf, {}});
    }
    {
        CoremlOptions o = base;
        o.cache_dir = "";
        configs.push_back({"no cache directory", o, gguf, {}});
    }
    {
        CoremlOptions o = base;
        o.generate = CoremlOptions::Generate::never;
        configs.push_back({"generate never", o, gguf, {}});
    }
    configs.push_back({"a GGUF the template does not match", base, other, {}});

    for (const Config & c : configs) {
        const std::string what = std::string(c.name) + ": ";
        CoremlSidecars fit_sidecars(c.opts, m);
        const std::vector<pcm::generation::Candidate> probed =
            probe_diarizer_templates(fit_sidecars, c.model.string(), ids);
        const std::vector<DiarizerCapacity> capacities = diarizer_capacities(prebuilt, template_frames, probed);
        CHECK(capacities.size() == 3 && capacities[1].prebuilt && !capacities[0].prebuilt && !capacities[2].prebuilt,
              (what + "one capacity per frame count, the prebuilt bundle wins").c_str());
        std::vector<int> charged;
        for (const CoremlDiarizerCharge & charge :
             diarizer_fit_charges(fit_sidecars, c.model.string(), capacities)) {
            if (charge.generated) charged.push_back(charge.frames);
        }
        CoremlSidecars sidecars(c.opts, m);
        const DiarizerSlots slots(c.model.string(), capacities, sidecars, &fake_load);
        std::vector<int> registered;
        for (const auto & slot : sidecars.slots()) {
            if (slot->spec().route == pcm::Route::generate) registered.push_back(slot->spec().geometry.at(0));
        }
        CHECK(charged == c.generated, (what + "fit charges the expected generated capacities").c_str());
        CHECK(registered == charged, (what + "discovery registers the capacities fit charges").c_str());
        std::vector<int> serving = c.generated;
        serving.push_back(745);
        std::sort(serving.begin(), serving.end());
        CHECK(slots.capacities() == serving, (what + "the serving capacities: prebuilt and generated").c_str());
        CHECK(sidecars.slots().size() == 3 && slots.status_only().size() == 3 - serving.size(),
              (what + "every other capacity keeps a status-only slot").c_str());
    }

    const auto predict_ok = [](int * seen) {
        return [seen](pcm::Handle &, int capacity) {
            *seen = capacity;
            return 0;
        };
    };
    const auto throws = [](const DiarizerSlots & slots, int frames, std::string * what) {
        try {
            slots.run(frames, [](pcm::Handle &, int) { return 0; });
        } catch (const pcm::CoremlError & e) {
            if (what != nullptr) *what = e.what();
            return true;
        }
        return false;
    };

    // Unvalidated units: 358 and 1125 are status-only, 745 is prebuilt.
    {
        CoremlOptions o = base;
        o.compute_units = "cpu_and_gpu";
        CoremlSidecars probe_sidecars(o, m);
        const std::vector<pcm::generation::Candidate> probed =
            probe_diarizer_templates(probe_sidecars, gguf.string(), ids);
        const std::vector<DiarizerCapacity> capacities = diarizer_capacities(prebuilt, template_frames, probed);

        CoremlOptions require = o;
        require.use = CoremlOptions::Use::require;
        CoremlSidecars strict_sidecars(require, m);
        const DiarizerSlots strict_slots(gguf.string(), capacities, strict_sidecars, &fake_load);
        int seen = 0;
        DiarizerPass pass;
        bool threw = false;
        try {
            pass = strict_slots.run(300, predict_ok(&seen));
        } catch (const pcm::CoremlError &) {
            threw = true;
        }
        CHECK(!threw && pass.status == 0 && seen == 745,
              "require: a status-only capacity gives way to the next covering one that serves");
        std::string what;
        CHECK(strict_slots.covers(800) && throws(strict_slots, 800, &what) &&
                  what.find("unsupported(unvalidated_units)") != std::string::npos,
              "require: a pass only a status-only capacity covers throws (unsupported(unvalidated_units))");
        seen = 0;
        threw = false;
        try {
            pass = strict_slots.run(2000, predict_ok(&seen));
        } catch (const pcm::CoremlError &) {
            threw = true;
        }
        CHECK(!threw && pass.status == kNoDiarizerSidecar && seen == 0,
              "require: a pass longer than every capacity runs on ggml");

        CoremlSidecars auto_sidecars(o, m);
        const DiarizerSlots auto_slots(gguf.string(), capacities, auto_sidecars, &fake_load);
        CHECK(!auto_slots.covers(800), "auto: a status-only capacity covers nothing");
        threw = false;
        seen = 0;
        try {
            pass = auto_slots.run(800, predict_ok(&seen));
        } catch (const pcm::CoremlError &) {
            threw = true;
        }
        CHECK(!threw && pass.status == kNoDiarizerSidecar && !pass.not_ready && seen == 0,
              "auto: a pass only a status-only capacity covers runs on ggml, not pending");
    }

    // A GGUF the template does not match, no prebuilt: require refuses every
    // pass a capacity covers (no_template).
    {
        CoremlOptions require = base;
        require.use = CoremlOptions::Use::require;
        CoremlSidecars sidecars(require, m);
        const std::vector<pcm::generation::Candidate> probed =
            probe_diarizer_templates(sidecars, other.string(), ids);
        const DiarizerSlots slots(other.string(), diarizer_capacities({}, template_frames, probed), sidecars,
                                  &fake_load);
        std::string what;
        CHECK(slots.capacities().empty() && throws(slots, 300, &what) &&
                  what.find("no_template") != std::string::npos,
              "require: no_template capacities refuse the pass");
    }

    // No adjacent bundle and no cache directory (the library's default):
    // under require every pass a template capacity covers throws, up to the
    // largest capacity; a longer pass runs on ggml.
    {
        CoremlOptions require = base;
        require.use = CoremlOptions::Use::require;
        require.cache_dir = "";
        CoremlSidecars sidecars(require, m);
        const std::vector<pcm::generation::Candidate> probed =
            probe_diarizer_templates(sidecars, gguf.string(), ids);
        const DiarizerSlots slots(gguf.string(), diarizer_capacities({}, template_frames, probed), sidecars,
                                  &fake_load);
        std::string what;
        CHECK(throws(slots, 1, &what) && what.find("unsupported(no_cache_dir)") != std::string::npos &&
                  throws(slots, 1125, nullptr),
              "require, no cache directory: every pass up to the largest capacity throws (no_cache_dir)");
        int seen = 0;
        bool threw = false;
        DiarizerPass pass;
        try {
            pass = slots.run(1126, predict_ok(&seen));
        } catch (const pcm::CoremlError &) {
            threw = true;
        }
        CHECK(!threw && pass.status == kNoDiarizerSidecar && seen == 0,
              "require, no cache directory: a pass longer than every capacity runs on ggml");
    }

    // Generation off for the whole instance (and Core ML off altogether):
    // every template capacity keeps a status-only slot, and the construction
    // group lists each one as skipped.
    for (const char * reason : {"no_cache_dir", "generate=never", "use=off"}) {
        CoremlOptions o = base;
        o.wait = CoremlOptions::Wait::construct;
        if (std::string(reason) == "no_cache_dir") {
            o.cache_dir = "";
        } else if (std::string(reason) == "generate=never") {
            o.generate = CoremlOptions::Generate::never;
        } else {
            o.use = CoremlOptions::Use::off;
        }
        CoremlSidecars sidecars(o, m);
        CHECK(!sidecars.generation_status().supported && sidecars.generation_status().reason == reason,
              "generation is off for the instance");
        const std::vector<pcm::generation::Candidate> probed =
            probe_diarizer_templates(sidecars, gguf.string(), ids);
        const DiarizerSlots slots(gguf.string(), diarizer_capacities({}, template_frames, probed), sidecars,
                                  &fake_load);
        sidecars.start();
        const std::string group = sidecars.construction_json();
        int listed = 0;
        for (const char * stage : {"diarizer-358", "diarizer-745", "diarizer-1125"}) {
            if (group.find(std::string("\"stage\":\"") + stage + "\"") != std::string::npos) ++listed;
        }
        CHECK(group.find("\"slots\":[]") != std::string::npos && listed == 3,
              (std::string(reason) + ": the construction group lists every capacity under skipped_route_none").c_str());
        CHECK(slots.status_only().size() == 3 && slots.status_only()[0]->status().reason == reason,
              (std::string(reason) + ": the status-only slots say why").c_str());
    }

    // The fit bytes of the generated capacities, by hand from the toy
    // template's memory: weight.bin 4064 bytes (runtime weights the same),
    // I/O (1 x 8 + 1 x 48) fp32 = 224 bytes, generation peak 12288 bytes,
    // the compile and load peaks at their defaults, margin 1.25:
    //   resident  = 4064 + 2 x 224 = 4512
    //   transient = max(12288 + 8 MiB + 4064 / 2, 4064 / 4) = 8402928
    // 358 and 1125 are generated; 745 is the prebuilt bundle, which this
    // envelope does not charge.
    {
        CoremlSidecars sidecars(base, m);
        const std::vector<pcm::generation::Candidate> probed =
            probe_diarizer_templates(sidecars, gguf.string(), ids);
        const std::vector<CoremlDiarizerCharge> charges =
            diarizer_fit_charges(sidecars, gguf.string(), diarizer_capacities(prebuilt, template_frames, probed));
        bool parts = charges.size() == 3;
        for (const CoremlDiarizerCharge & c : charges) {
            parts = parts && (c.frames == 745 ? !c.generated && c.resident_bytes == 0 && c.transient_bytes == 0
                                              : c.generated && c.resident_bytes == 4512 &&
                                                    c.transient_bytes == 8402928 && c.margin == 1.25);
        }
        CHECK(parts, "fit: each generated capacity carries its template's resident part, transient and margin");
        CHECK(coreml_generated_diarizer_envelope(charges, {}, /*every_capacity=*/true) == 10514940,
              "fit, prewarm all: ceil((2 x 4512 + 8402928) x 1.25) = 10514940 bytes");
        CHECK(coreml_generated_diarizer_envelope(charges, {{1, 358}}) == 10509300,
              "fit, calls up to 358 frames: ceil((4512 + 8402928) x 1.25) = 10509300 bytes");
    }

    // The smallest serving capacity is minimal even above a status-only one.
    {
        CoremlOptions o = base;
        o.compute_units = "cpu_and_gpu";
        o.wait = CoremlOptions::Wait::construct;
        CoremlSidecars sidecars(o, m);
        const std::vector<pcm::generation::Candidate> probed =
            probe_diarizer_templates(sidecars, gguf.string(), ids);
        const DiarizerSlots slots(gguf.string(), diarizer_capacities(prebuilt, template_frames, probed), sidecars,
                                  &fake_load);
        const int loads = g_fake_loads.load();
        sidecars.start();
        const std::string group = sidecars.construction_json();
        CHECK(g_fake_loads.load() == loads + 1 && group.find("\"stage\":\"diarizer-745\"") != std::string::npos,
              "the construction group loads the smallest prebuilt capacity above the status-only ones");
    }

    // A diarization model with one template capacity and adjacent prebuilt
    // encoder and bypass-encoder bundles. With the capacity status-only (no
    // cache directory, the library's default) the encoders stay in the
    // construction group as requests, so they load there as with no diarizer
    // at all; with a serving capacity the diarizer takes their place.
    {
        fs::create_directories(dir / "m-encoder.mlmodelc");
        fs::create_directories(dir / "m-bypass-encoder.mlmodelc");
        auto encoder_request = [&](const char * stage, const char * name, bool minimal) {
            SidecarRequest r;
            r.stage = stage;
            r.path = (dir / name).string();
            r.present = true;
            r.stage_default = pcm::ComputeUnits::cpu_and_ane;
            r.load_fn = &fake_load;
            r.minimal = minimal;
            return r;
        };
        // The group JSON lists `skipped_route_none` before `slots` (sorted
        // keys): a stage after `"slots":[` is a request of the group.
        auto requested = [](const std::string & group, const char * stage) {
            const size_t slots = group.find("\"slots\":[");
            const size_t at = group.find(std::string("\"stage\":\"") + stage + "\"");
            return slots != std::string::npos && at != std::string::npos && at > slots;
        };
        for (const bool serving : {false, true}) {
            const std::string what = serving ? "serving diarizer capacity: " : "status-only diarizer capacity: ";
            CoremlOptions o = base;
            o.wait = CoremlOptions::Wait::construct;
            o.prewarm = CoremlOptions::Prewarm::predicted;
            if (!serving) o.cache_dir = "";
            CoremlSidecars sidecars(o, m);
            const std::vector<pcm::generation::Candidate> probed =
                probe_diarizer_templates(sidecars, gguf.string(), {"toy-stateless"});
            const DiarizerSlots slots(gguf.string(), diarizer_capacities({}, {410}, probed), sidecars, &fake_load);
            CHECK(slots.capacities().size() == (serving ? 1u : 0u),
                  (what + "the template capacity serves only with a cache directory").c_str());
            const bool minimal = encoder_sidecars_minimal(slots.capacities());
            CHECK(minimal == !serving, (what + "the encoders are minimal only without a serving capacity").c_str());
            auto encoder = sidecars.add(encoder_request("encoder", "m-encoder.mlmodelc", minimal));
            auto bypass = sidecars.add(encoder_request("bypass-encoder", "m-bypass-encoder.mlmodelc", minimal));
            sidecars.start();
            const std::string group = sidecars.construction_json();
            CHECK(requested(group, "encoder") == !serving && requested(group, "bypass-encoder") == !serving,
                  (what + "the construction group requests the encoder and bypass encoder only "
                          "without a serving capacity").c_str());
            CHECK(requested(group, "diarizer-410") == serving,
                  (what + "the construction group requests the diarizer only when it serves").c_str());
            const pcm::SlotState encoders = serving ? pcm::SlotState::idle : pcm::SlotState::ready;
            CHECK(encoder->state() == encoders && bypass->state() == encoders,
                  (what + "wait = construct returns with the encoders loaded only when they are minimal").c_str());
        }
    }

    pcm::generation::set_compile_override(nullptr);
    ::unsetenv("QVAC_COREML_TEMPLATE_DIR");
}
#endif

int main() {
    case_null_route_plan();
    case_tdt_plan_needs_validated_seams();
    case_window_k_failure_and_next_call();
    case_cannot_serve_keeps_route();
    case_call_label();
    case_owner_route();
    case_strict_route();
    case_pending_route();
    case_sidecar_groups();
    case_options();
    case_fit_stable();
    case_generated_diarizer_envelope();
    case_coreml_verdicts_validated();
#if defined(QVAC_COREML_HAVE_GGML)
    case_generated_sidecar();
    case_diarizer_capacities();
#endif
    std::printf("test-parakeet-coreml-route: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
