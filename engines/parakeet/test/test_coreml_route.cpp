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

#include "coreml/parakeet_coreml_sidecars.h"
#include "coreml_fit.h"
#include "encoder_route.h"
#include "long_form.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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
    CHECK(!encode_prediction_failed(0) && !encode_prediction_failed(101),
          "success and the engine's CTC-head codes are not sidecar failures");

    // A real prediction failure on the same slot still retires it.
    const EncoderRoute last = route_from_acquired(slot->acquire(pcm::Demand::now), fake_ops(), 128);
    auto * enc2 = reinterpret_cast<FakeEncoder *>(last.context());
    enc2->fail_rc = 5;
    enc2->fail_at = enc2->encodes.load() + 1;  // the second window
    CHECK(run_call(last, 2).label == "mixed", "a failed prediction makes the call mixed");
    CHECK(slot->state() == pcm::SlotState::retired, "a failed prediction retires the slot");
    fs::remove_all(dir);
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

}  // namespace

int main() {
    case_null_route_plan();
    case_window_k_failure_and_next_call();
    case_cannot_serve_keeps_route();
    case_owner_route();
    case_strict_route();
    case_pending_route();
    case_sidecar_groups();
    case_options();
    case_fit_stable();
    std::printf("test-parakeet-coreml-route: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
