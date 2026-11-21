#include "audio8/codec_coreml.h"

#include "fit_util.h"

#ifdef TTS_CPP_USE_COREML
#include "audio8/coreml/codec-synth.h"
#include "audio8/coreml_path.h"
#include "audio8/internal.h"
#include "coreml_sidecar_path.h"

#include "qvac_coreml/generation.h"
#include "qvac_coreml/manager.h"

#include <sys/stat.h>

#include <chrono>
#include <cstdio>
#include <thread>
#endif

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace tts_cpp::audio8::detail {

void validate_codec_coreml_options(const CoremlOptions & options) {
    if (options.compute_units) {
        const auto & units = *options.compute_units;
        if (units != "all" && units != "cpu_and_ane" &&
            units != "cpu_and_gpu" && units != "cpu_only") {
            throw std::invalid_argument(
                "audio8: invalid Core ML compute_units '" + units + "'");
        }
    }
    if (options.strict_wait_ms && *options.strict_wait_ms < 0) {
        throw std::invalid_argument(
            "audio8: Core ML strict_wait_ms must be nonnegative");
    }
    if (options.memory_budget_bytes && *options.memory_budget_bytes < 0) {
        throw std::invalid_argument(
            "audio8: Core ML memory_budget_bytes must be nonnegative");
    }
}

uint64_t codec_fit_capacity(const CoremlOptions & options, uint64_t free_bytes) {
    if (!options.memory_budget_bytes)
        return free_bytes;
    return std::min(free_bytes,
                    static_cast<uint64_t>(*options.memory_budget_bytes));
}

#ifdef TTS_CPP_USE_COREML
namespace pcm = QVAC_COREML_NS;

namespace {

constexpr const char * TEMPLATE_ID = "audio8-codec-w64";
constexpr const char * ENGINE = "audio8";
constexpr const char * STAGE = "codec_synthesis";
constexpr int GENERATED_WINDOW = 64;
constexpr double PREBUILT_ANE_RESERVE_FACTOR = 1.6;
constexpr uint64_t PREBUILT_UNKNOWN_WEIGHT_RESERVE = 512ULL << 20;

pcm::Resolved resolve(const CoremlOptions & options) {
    static_assert(static_cast<int>(CoremlOptions::Use::require) ==
                  static_cast<int>(pcm::Use::require));
    static_assert(static_cast<int>(CoremlOptions::Generate::background) ==
                  static_cast<int>(pcm::Generate::background));
    static_assert(static_cast<int>(CoremlOptions::Wait::call) ==
                  static_cast<int>(pcm::Wait::call));
    static_assert(static_cast<int>(CoremlOptions::Prewarm::all) ==
                  static_cast<int>(pcm::Prewarm::all));
    pcm::Options o;
    o.use = static_cast<pcm::Use>(options.use);
    o.generate = static_cast<pcm::Generate>(options.generate);
    o.wait = static_cast<pcm::Wait>(options.wait);
    o.prewarm = static_cast<pcm::Prewarm>(options.prewarm);
    o.cache_dir = options.cache_dir;
    o.compute_units = options.compute_units;
    o.strict_wait_ms = options.strict_wait_ms;
    o.memory_budget_bytes = options.memory_budget_bytes;
    return pcm::resolve_coreml(o, "AUDIO8");
}

struct planned_codec {
    pcm::Plan plan;
    pcm::ComputeUnits units = pcm::ComputeUnits::all;
    std::string prebuilt;
    std::optional<pcm::generation::Candidate> candidate;
};

planned_codec plan_codec(const std::string & path,
                         const pcm::Resolved & options,
                         const pcm::Manager & manager,
                         bool source_changed = false) {
    planned_codec codec;
    codec.prebuilt = coreml_codec_sidecar_path(path);
    pcm::PlanInput input;
    input.coreml_built = true;
    input.generation_built = pcm::generation::built();
    input.platform = manager.platform();
    input.os = manager.os_version();
    input.prebuilt_present =
        ::tts_cpp::detail::coreml_sidecar_exists(codec.prebuilt);
    if (options.enabled() && input.generation_built && !input.prebuilt_present) {
        auto candidates = pcm::generation::probe({TEMPLATE_ID}, path, options);
        if (!candidates.empty())
            codec.candidate = std::move(candidates.front());
    }
    const auto & candidate = codec.candidate;
    input.template_available = candidate && candidate->available;
    // A changed source fails in prepare even if its new header would not
    // match. Keep the policy gates so unavailable units still stay disabled.
    input.template_matched =
        input.template_available && (candidate->matched || source_changed);
    if (input.template_available)
        input.validated_units = candidate->memory.validated_units;
    codec.units = options.units_or(
        input.template_available
            ? candidate->default_units.value_or(pcm::ComputeUnits::all)
            : pcm::ComputeUnits::all);
    input.units = codec.units;
    codec.plan = pcm::plan(input, options);
    if (codec.plan.none_state == pcm::SlotState::no_template && candidate &&
        !candidate->reason.empty()) {
        codec.plan.reason += ": " + candidate->reason;
    }
    return codec;
}

using codec_context = std::unique_ptr<audio8_coreml_codec_context,
                                      decltype(&audio8_coreml_codec_free)>;

pcm::LoadResult load_codec_sidecar(const pcm::LoadRequest & req) {
    pcm::LoadResult result;
    if (!pcm::generation::begin_load(req, result))
        return result;
    if (req.geometry.size() != 3) {
        result.error =
            pcm::Error::deterministic("audio8: missing codec geometry");
        pcm::generation::end_load(req, result);
        return result;
    }
    auto * model =
        audio8_coreml_codec_init(req.artifact_path.c_str(), req.geometry[0],
                                 req.geometry[1], pcm::to_string(req.units));
    codec_context context(model, audio8_coreml_codec_free);
    const auto window = audio8_coreml_codec_window_frames(model);
    if (!model) {
        result.error = pcm::Error::transient(
            "audio8: could not load codec sidecar " + req.artifact_path);
    } else if (req.tmpl != nullptr &&
               (window != GENERATED_WINDOW || window <= req.geometry[2])) {
        result.error = pcm::Error::deterministic(
            "audio8: unexpected generated codec window");
    } else {
        result.model = std::move(context);
    }
    pcm::generation::end_load(req, result);
    return result;
}

pcm::Error warm_codec(void * model, const pcm::LoadRequest & req) {
    auto * context = static_cast<audio8_coreml_codec_context *>(model);
    const auto window = audio8_coreml_codec_window_frames(context);
    // Prebuilts with too little context keep their first-use fallback.
    if (window <= req.geometry[2])
        return {};
    std::vector<float> post(window * req.geometry[0]);
    std::vector<float> pcm_out(window * req.geometry[1]);
    if (audio8_coreml_codec_synthesize(context, post.data(), pcm_out.data()) !=
        0) {
        return pcm::Error::transient(
            "audio8: codec sidecar warm-up prediction failed");
    }
    return {};
}

class acquired_codec final : public codec_coreml_model {
  public:
    explicit acquired_codec(pcm::Handle model) : model_(std::move(model)) {}
    int window() const override {
        return static_cast<int>(audio8_coreml_codec_window_frames(
            model_->as<audio8_coreml_codec_context>()));
    }
    const char * label() const override { return model_->label(); }
    bool synthesize(const float * post, float * pcm_out) const override {
        auto lock = model_->lock_for_prediction();
        return audio8_coreml_codec_synthesize(
                   model_->as<audio8_coreml_codec_context>(), post, pcm_out) ==
               0;
    }
    void retire(const char * reason) const override { model_->retire(reason); }

  private:
    pcm::Handle model_;
};

} // namespace

uint64_t prebuilt_codec_ane_reserve(const std::string & path) {
    struct stat weights{};
    if (stat((path + "/weights/weight.bin").c_str(), &weights) != 0 ||
        !S_ISREG(weights.st_mode) || weights.st_size <= 0) {
        // Compiled bundle layouts can change. Keep cold ANE admission in
        // effect when there is no usable weight size.
        return PREBUILT_UNKNOWN_WEIGHT_RESERVE;
    }
    return fitutil::sat_u64_from_double(PREBUILT_ANE_RESERVE_FACTOR *
                                        weights.st_size);
}

struct codec_coreml::Impl {
    pcm::Resolved options;
    std::shared_ptr<pcm::Manager> manager;
    std::shared_ptr<pcm::Slot> slot;
    pcm::SlotStatus unavailable;
    std::shared_ptr<pcm::Group> construction;
    std::shared_ptr<pcm::Group> group(pcm::GroupKind kind, pcm::Prewarm scope,
                                      bool wait) const {
        std::vector<pcm::GroupRequest> requests;
        std::vector<pcm::SlotStatus> skipped;
        if (scope != pcm::Prewarm::none) {
            if (slot) {
                requests.push_back({slot, kind == pcm::GroupKind::construction
                                              ? pcm::Priority::minimal
                                              : pcm::Priority::prewarm});
            } else {
                skipped.push_back(unavailable);
            }
        }
        return manager->issue_group(kind, requests, wait, std::move(skipped));
    }
};

std::shared_ptr<codec_coreml>
codec_coreml::create(const codec_model & codec, const std::string & path,
                     const codec_source & source, const CoremlOptions & options,
                     codec_budget budget) {
    auto impl = std::make_unique<Impl>();
    impl->options = resolve(options);
    for (const auto & warning : impl->options.warnings) {
        std::fprintf(stderr, "[audio8] %s\n", warning.c_str());
    }
    impl->manager = pcm::Manager::instance();
    auto codec_plan =
        plan_codec(path, impl->options, *impl->manager, source.stat.changed);
    if (codec_plan.plan.route == pcm::Route::generate &&
        impl->options.memory_budget_bytes) {
        if (budget == codec_budget::degrade) {
            impl->options.use = pcm::Use::off;
            impl->options.disabled_reason = "memory";
            codec_plan.plan = {pcm::Route::none, pcm::SlotState::disabled,
                               "memory"};
        } else if (budget == codec_budget::unprojected) {
            // The final route can change while the ggml models are uploaded.
            // A prebuilt or unavailable route has no generated envelope.
            codec_plan.plan = {pcm::Route::none, pcm::SlotState::unsupported,
                               "unprojected_budget"};
        }
    }
    if (codec_plan.plan.route == pcm::Route::none) {
        impl->unavailable.stage = STAGE;
        impl->unavailable.function = "main";
        impl->unavailable.state = codec_plan.plan.none_state;
        impl->unavailable.reason = codec_plan.plan.reason;
        impl->unavailable.compute_units = pcm::to_string(codec_plan.units);
        impl->unavailable.label = "ggml";
    } else {
        pcm::SlotSpec spec;
        spec.engine = ENGINE;
        spec.stage = STAGE;
        spec.units = codec_plan.units;
        spec.route = codec_plan.plan.route;
        spec.load_fn = load_codec_sidecar;
        spec.warm_fn = warm_codec;
        spec.geometry = {codec.hp.latent_dim, codec.hp.frame_size,
                         synthesis_context_frames(codec)};
        if (spec.route == pcm::Route::prebuilt) {
            spec.artifact_path = codec_plan.prebuilt;
            if (pcm::units_include_ane(spec.units))
                spec.ane_compile_reserve_bytes =
                    prebuilt_codec_ane_reserve(spec.artifact_path);
        } else {
            const auto & candidate = *codec_plan.candidate;
            spec.tmpl = candidate.tmpl;
            spec.prepare = pcm::generation::prepare_request(
                candidate, source.stat, impl->options);
            spec.prepare_fn = pcm::generation::prepare;
            spec.ane_compile_reserve_bytes =
                pcm::generation::ane_compile_reserve_bytes(candidate.memory,
                                                           spec.units);
        }
        impl->slot = impl->manager->make_slot(
            std::move(spec), pcm::SlotPolicy::from(impl->options));
    }
    const bool waited = impl->options.wait == pcm::Wait::construct;
    impl->construction = impl->group(pcm::GroupKind::construction,
                                     impl->options.prewarm, waited);
    if (waited) {
        std::optional<pcm::Clock::duration> timeout;
        if (impl->options.strict() && impl->options.strict_wait_ms) {
            timeout = std::chrono::milliseconds(*impl->options.strict_wait_ms);
        }
        if (!impl->construction->wait(timeout).success &&
            impl->options.strict() && impl->slot) {
            const auto status = impl->slot->status();
            throw std::runtime_error(
                "audio8: required codec sidecar unavailable after construction "
                "(" +
                pcm::state_label(status.state, status.reason) +
                "): " + status.error);
        }
    }
    return std::shared_ptr<codec_coreml>(new codec_coreml(std::move(impl)));
}

codec_coreml_call
codec_coreml::acquire(const std::atomic<bool> * cancel) const {
    codec_coreml_call call;
    call.strict = impl_->options.strict();
    if (cancel && cancel->load(std::memory_order_relaxed))
        return call;
    if (!impl_->slot) {
        if (call.strict) {
            throw std::runtime_error(
                "audio8: required codec sidecar unavailable (" +
                pcm::state_label(impl_->unavailable.state,
                                 impl_->unavailable.reason) +
                ")");
        }
        return call;
    }
    // A cold load never blocks the default acquisition. A warm hint may wait
    // within this short deadline; explicit call / require use the slot policy.
    constexpr auto WARM_WAIT_BUDGET = std::chrono::milliseconds(20);
    auto acquired = impl_->slot->acquire(
        pcm::Demand::now, pcm::Clock::now() + WARM_WAIT_BUDGET, cancel);
    if (acquired.model) {
        call.model =
            std::make_shared<acquired_codec>(std::move(acquired.model));
    }
    call.pending = acquired.pending;
    return call;
}

std::string codec_coreml::status() const {
    pcm::GenerationStatus generation;
    generation.supported =
        pcm::generation::built() &&
        impl_->manager->platform() == pcm::Platform::macos &&
        pcm::os_at_least(impl_->manager->os_version(), 15, 0);
    if (!generation.supported) {
        generation.reason = pcm::generation::built() ? "os" : "build";
    } else if (impl_->unavailable.reason == "unvalidated_units") {
        generation.reason = impl_->unavailable.reason;
    }
    if (impl_->slot) {
        return pcm::to_json(impl_->manager->status(
            ENGINE, impl_->options.cache_dir, {impl_->slot}, generation));
    }
    pcm::EngineStatus status;
    status.engine = ENGINE;
    status.cache_dir = impl_->options.cache_dir;
    status.generation = generation;
    status.stages.push_back(impl_->unavailable);
    return pcm::to_json(status);
}

std::string codec_coreml::construction_json() const {
    return pcm::to_json(impl_->construction->result());
}

std::string codec_coreml::prewarm(CoremlOptions::Prewarm scope,
                                  bool wait,
                                  const std::atomic<uint64_t> & cancels) const {
    const auto since = cancels.load(std::memory_order_relaxed);
    const auto group = impl_->group(pcm::GroupKind::prewarm,
                                    static_cast<pcm::Prewarm>(scope), wait);
    if (!wait)
        return pcm::to_json(group->result());
    return pcm::to_json(group->wait(std::nullopt, cancels, since));
}

bool codec_coreml::wait(int64_t timeout_ms,
                        const std::atomic<uint64_t> * cancels) const {
    if (!impl_->slot)
        return false;
    const auto start = pcm::Clock::now();
    const auto since = cancels ? cancels->load(std::memory_order_relaxed) : 0;
    for (;;) {
        const auto state = impl_->slot->state();
        if (state == pcm::SlotState::ready)
            return true;
        if (pcm::is_terminal(state) || state == pcm::SlotState::idle ||
            state == pcm::SlotState::deferred_memory)
            return false;
        if (cancels && cancels->load(std::memory_order_relaxed) != since)
            return false;
        if (timeout_ms >= 0 &&
            pcm::Clock::now() - start >= std::chrono::milliseconds(timeout_ms))
            return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

bool codec_needs_budget_projection(const std::string & path,
                                   const CoremlOptions & options) {
    return options.memory_budget_bytes &&
           plan_codec(path, resolve(options), *pcm::Manager::instance())
                   .plan.route == pcm::Route::generate;
}

FitCoreml fit_codec_coreml(const std::string & path,
                           const CoremlOptions & options, uint64_t ggml_bytes,
                           uint64_t free_bytes, int frames,
                           const codec_model & model) {
    using fitutil::sat_add;
    using fitutil::sat_mul;
    FitCoreml result;
    auto resolved = resolve(options);
    const auto codec = plan_codec(path, resolved, *pcm::Manager::instance());
    result.route = pcm::to_string(codec.plan.route);
    if (resolved.use == pcm::Use::off)
        result.disabled_reason = resolved.disabled_reason;
    if (codec.plan.route == pcm::Route::generate) {
        auto memory = codec.candidate->memory;
        result.validated = true;
        // Bound queued float32 input, worker I/O and retained PCM cores. The
        // ggml projection already includes one waveform and the base post slab.
        const int context = synthesis_context_frames(model);
        const uint64_t step =
            static_cast<uint64_t>(std::max(1, GENERATED_WINDOW - context));
        const uint64_t windows =
            (static_cast<uint64_t>(frames) + step - 1) / step;
        const uint64_t input =
            sat_mul(GENERATED_WINDOW * sizeof(float), model.hp.latent_dim);
        const uint64_t output =
            sat_mul(GENERATED_WINDOW * sizeof(float), model.hp.frame_size);
        const uint64_t stream_buffers =
            sat_add(sat_mul(windows + 2, input), sat_mul(windows + 1, output));
        memory.workspace_bytes = sat_add(memory.workspace_bytes, stream_buffers);
        result.bytes = pcm::generation::envelope_bytes(memory, codec.units);
    }
    const uint64_t inclusive = sat_add(ggml_bytes, result.bytes);
    const auto capacity = codec_fit_capacity(options, free_bytes);
    result.ggml_only_fits = ggml_bytes <= capacity;
    result.fits = inclusive <= capacity;
    if (resolved.memory_budget_bytes && result.validated) {
        const pcm::FitVerdict verdict{result.ggml_only_fits, result.fits,
                                      result.bytes, result.validated};
        if (pcm::apply_memory_degrade(resolved, verdict)) {
            result.disabled_reason = resolved.disabled_reason;
        }
    }
    return result;
}

#else

struct codec_coreml::Impl {};

codec_coreml_call codec_coreml::acquire(const std::atomic<bool> *) const {
    return {};
}
std::string codec_coreml::status() const { return "{}"; }
std::string codec_coreml::construction_json() const { return "{}"; }
std::string codec_coreml::prewarm(CoremlOptions::Prewarm, bool,
                                  const std::atomic<uint64_t> &) const {
    return "{}";
}
bool codec_coreml::wait(int64_t, const std::atomic<uint64_t> *) const {
    return false;
}

FitCoreml fit_codec_coreml(const std::string &, const CoremlOptions & options,
                           uint64_t ggml_bytes, uint64_t free_bytes, int,
                           const codec_model &) {
    FitCoreml result;
    result.ggml_only_fits = result.fits =
        ggml_bytes <= codec_fit_capacity(options, free_bytes);
    return result;
}

#endif

codec_coreml::codec_coreml(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
codec_coreml::~codec_coreml() = default;

} // namespace tts_cpp::audio8::detail
