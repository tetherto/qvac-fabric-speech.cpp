#include "coreml/parakeet_coreml_sidecars.h"

#include <sys/stat.h>

#include <chrono>
#include <utility>

namespace parakeet {

namespace {

constexpr const char * kEngine = "parakeet";
constexpr const char * kEnvPrefix = "PARAKEET";
constexpr const char * kWeightsFile = "/weights/weight.bin";
// B.1 default ane_compile_reserve_bytes: ANECompilerService peaked at about
// 1.6x the weights compiling a 199 MB model (platform-feasibility §4).
constexpr double kAneCompileReserveFactor = 1.6;

uint64_t file_bytes(const std::string & path) {
    struct stat st {};
    return ::stat(path.c_str(), &st) == 0 ? static_cast<uint64_t>(st.st_size) : 0;
}

template <class E, class F>
F map_enum(E v) {
    return static_cast<F>(static_cast<int>(v));
}

void retire_handle(void * hold, const char * reason) {
    auto * handle = static_cast<pcm::Handle *>(hold);
    if (handle != nullptr && *handle) (*handle)->retire(reason != nullptr ? reason : "prediction failed");
}

std::unique_lock<std::mutex> lock_handle(void * hold) {
    auto * handle = static_cast<pcm::Handle *>(hold);
    if (handle == nullptr || !*handle) return {};
    return (*handle)->lock_for_prediction();
}

}  // namespace

pcm::Options to_runtime_options(const CoremlOptions & o) {
    // The public enums mirror the runtime's member for member (B.6).
    static_assert(static_cast<int>(CoremlOptions::Use::require) == static_cast<int>(pcm::Use::require), "Use");
    static_assert(static_cast<int>(CoremlOptions::Generate::background) ==
                      static_cast<int>(pcm::Generate::background), "Generate");
    static_assert(static_cast<int>(CoremlOptions::Wait::call) == static_cast<int>(pcm::Wait::call), "Wait");
    static_assert(static_cast<int>(CoremlOptions::Prewarm::all) == static_cast<int>(pcm::Prewarm::all), "Prewarm");
    pcm::Options r;
    r.use = map_enum<CoremlOptions::Use, pcm::Use>(o.use);
    r.generate = map_enum<CoremlOptions::Generate, pcm::Generate>(o.generate);
    r.wait = map_enum<CoremlOptions::Wait, pcm::Wait>(o.wait);
    r.prewarm = map_enum<CoremlOptions::Prewarm, pcm::Prewarm>(o.prewarm);
    r.cache_dir = o.cache_dir;
    r.compute_units = o.compute_units;
    r.strict_wait_ms = o.strict_wait_ms;
    r.memory_budget_bytes = o.memory_budget_bytes;
    return r;
}

pcm::Resolved resolve_parakeet_coreml(const CoremlOptions & opts) {
    return pcm::resolve_coreml(to_runtime_options(opts), kEnvPrefix);
}

pcm::Prewarm to_runtime_prewarm(CoremlOptions::Prewarm scope) {
    return map_enum<CoremlOptions::Prewarm, pcm::Prewarm>(scope);
}

EncoderRoute route_from_acquired(pcm::Acquired && acquired, const EncoderModelOps & ops, int n_mels) {
    if (!acquired.model || ops.context == nullptr) return EncoderRoute::ggml(acquired.pending);
    parakeet_coreml_context * ctx = ops.context(acquired.model->get());
    if (ctx == nullptr) return EncoderRoute::ggml(acquired.pending);
    const EncoderRouteGeometry geometry = ops.describe != nullptr ? ops.describe(ctx, n_mels) : EncoderRouteGeometry{};
    const char * label = ops.label != nullptr ? ops.label(ctx) : nullptr;
    auto hold = std::make_shared<pcm::Handle>(std::move(acquired.model));
    EncoderRoute::Ops route_ops;
    route_ops.retire = &retire_handle;
    route_ops.lock = &lock_handle;
    route_ops.encode = ops.encode;
    return EncoderRoute::coreml(ctx, geometry, label != nullptr ? label : "coreml", std::move(hold),
                                route_ops, /*owner=*/false);
}

uint64_t ane_compile_reserve_bytes(const std::string & mlmodelc, pcm::ComputeUnits units) {
    if (!pcm::units_include_ane(units)) return 0;
    return static_cast<uint64_t>(kAneCompileReserveFactor *
                                 static_cast<double>(file_bytes(mlmodelc + kWeightsFile)));
}

CoremlSidecars::CoremlSidecars(const CoremlOptions & opts, std::shared_ptr<pcm::Manager> manager,
                               bool memory_degraded)
    : resolved_(resolve_parakeet_coreml(opts)),
      manager_(manager ? std::move(manager) : pcm::Manager::instance()) {
    if (memory_degraded) {
        pcm::FitVerdict verdict;
        verdict.ggml_only_fits = true;
        verdict.coreml_fits = false;
        pcm::apply_memory_degrade(resolved_, verdict);
    }
}

CoremlSidecars::~CoremlSidecars() = default;

std::shared_ptr<pcm::Slot> CoremlSidecars::add(const SidecarRequest & request) {
    pcm::PlanInput in;
    in.coreml_built = true;
    in.generation_built = false;  // P1: prebuilt sidecars only; templates arrive with P2-P4
    in.prebuilt_present = request.present;
    in.units = resolved_.units_or(request.stage_default);
    in.platform = manager_->platform();
    in.os = manager_->os_version();
    const pcm::Plan plan = pcm::plan(in, resolved_);

    pcm::SlotSpec spec;
    spec.engine = kEngine;
    spec.stage = request.stage;
    spec.units = in.units;
    spec.route = plan.route;
    spec.none_state = plan.none_state;
    spec.none_reason = plan.reason;
    spec.artifact_path = request.path;
    spec.load_fn = request.load_fn;
    spec.geometry = request.geometry;
    spec.ane_compile_reserve_bytes =
        plan.route == pcm::Route::prebuilt ? ane_compile_reserve_bytes(request.path, in.units) : 0;
    auto slot = manager_->make_slot(std::move(spec), pcm::SlotPolicy::from(resolved_));
    entries_.push_back(Entry{slot, request.minimal});
    return slot;
}

std::vector<pcm::GroupRequest> CoremlSidecars::requests(pcm::Prewarm scope, pcm::GroupKind kind,
                                                        std::vector<pcm::SlotStatus> & skipped) const {
    std::vector<pcm::GroupRequest> out;
    if (scope == pcm::Prewarm::none || scope == pcm::Prewarm::unset) return out;
    for (const Entry & e : entries_) {
        if (scope != pcm::Prewarm::all && !e.minimal) continue;
        if (e.slot->spec().route == pcm::Route::none) {
            skipped.push_back(e.slot->status());
            continue;
        }
        // The construction-time minimal set is `minimal`; `all` extras and
        // later prewarm groups are `prewarm` (§2.9 priorities).
        const bool minimal = kind == pcm::GroupKind::construction && e.minimal;
        out.push_back(pcm::GroupRequest{e.slot, minimal ? pcm::Priority::minimal : pcm::Priority::prewarm});
    }
    return out;
}

void CoremlSidecars::start() {
    std::vector<pcm::SlotStatus> skipped;
    auto reqs = requests(resolved_.prewarm, pcm::GroupKind::construction, skipped);
    const bool waited = resolved_.wait == pcm::Wait::construct;
    construction_ = manager_->issue_group(pcm::GroupKind::construction, reqs, waited, std::move(skipped));
    if (waited) construction_->wait();
}

pcm::PrewarmResult CoremlSidecars::prewarm(CoremlOptions::Prewarm scope, bool wait,
                                           const std::atomic<bool> * cancel) {
    std::vector<pcm::SlotStatus> skipped;
    const auto reqs = requests(to_runtime_prewarm(scope), pcm::GroupKind::prewarm, skipped);
    auto group = manager_->issue_group(pcm::GroupKind::prewarm, reqs, wait, std::move(skipped));
    return wait ? group->wait(std::nullopt, cancel) : group->result();
}

bool CoremlSidecars::wait_construction(int64_t timeout_ms, const std::atomic<bool> * cancel) {
    if (!construction_) return false;
    std::optional<pcm::Clock::duration> timeout;
    if (timeout_ms >= 0) timeout = std::chrono::milliseconds(timeout_ms);
    return construction_->wait(timeout, cancel).success;
}

std::string CoremlSidecars::construction_json() const {
    return construction_ ? pcm::to_json(construction_->result()) : std::string("{}");
}

std::string CoremlSidecars::status_json() const {
    pcm::GenerationStatus generation;
    generation.supported = false;
    generation.reason = "no_template";
    return pcm::to_json(manager_->status(kEngine, resolved_.cache_dir, slots(), generation));
}

std::vector<std::shared_ptr<pcm::Slot>> CoremlSidecars::slots() const {
    std::vector<std::shared_ptr<pcm::Slot>> out;
    out.reserve(entries_.size());
    for (const Entry & e : entries_) out.push_back(e.slot);
    return out;
}

}  // namespace parakeet
