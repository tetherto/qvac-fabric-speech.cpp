#include "moss/coreml_generation.h"

#include <algorithm>
#include <stdexcept>

#ifdef TTS_CPP_USE_COREML
#include "coreml_sidecar_path.h"

#include "qvac_coreml/generation.h"
#include "qvac_coreml/manager.h"
#include "qvac_coreml/status.h"
#endif

namespace tts_cpp::moss::detail {
namespace {

const char * stage_name(GeneratedStage stage) {
    return stage == GeneratedStage::speech_tokenizer ? "speech-tokenizer"
                                                     : "sfx-vae";
}

CoremlPolicy legacy_policy(const CoremlOptions & options) {
    CoremlPolicy policy = read_coreml_policy();
    if (options.use != CoremlOptions::Use::unset) {
        policy.strict = options.use == CoremlOptions::Use::require;
        policy.disabled =
            policy.disabled || options.use == CoremlOptions::Use::off;
    }
    if (options.compute_units) {
        const std::string & units = *options.compute_units;
        if (units != "all" && units != "cpu_and_ane" &&
            units != "cpu_and_gpu" && units != "cpu_only") {
            throw std::invalid_argument(
                "moss: invalid Core ML compute_units: " + units);
        }
    }
    policy.units = options.compute_units.value_or(requested_coreml_units());
    return policy;
}

#ifdef TTS_CPP_USE_COREML
namespace cm = QVAC_COREML_NS;

constexpr int GEOMETRY_SIZE = 6;
constexpr int RANK = 3;

std::string prebuilt_path(GeneratedStage stage, const std::string & gguf) {
    return stage == GeneratedStage::speech_tokenizer
               ? speech_tokenizer_sidecar_path(gguf)
               : sfx_vae_sidecar_path(gguf);
}

cm::Resolved resolve(const CoremlOptions & options) {
    cm::Options runtime;
    runtime.use = static_cast<cm::Use>(options.use);
    runtime.generate = static_cast<cm::Generate>(options.generate);
    runtime.prewarm = static_cast<cm::Prewarm>(options.prewarm);
    runtime.wait = static_cast<cm::Wait>(options.wait);
    runtime.compute_units = options.compute_units;
    runtime.cache_dir = options.cache_dir;
    runtime.strict_wait_ms = options.strict_wait_ms;
    runtime.memory_budget_bytes = options.memory_budget_bytes;
    return cm::resolve_coreml(runtime, "MOSS");
}

const char * template_id(GeneratedStage stage) {
    return stage == GeneratedStage::speech_tokenizer
               ? "moss-speech-tokenizer-30s"
               : "moss-sfx-vae-w320";
}

struct StagePlan {
    cm::Plan plan;
    cm::generation::Candidate candidate;
    cm::ComputeUnits units = cm::ComputeUnits::cpu_and_gpu;

    uint64_t envelope() const {
        return plan.route == cm::Route::generate
                   ? cm::generation::envelope_bytes(candidate.memory, units)
                   : 0;
    }
};

StagePlan plan_stage(GeneratedStage stage, const std::string & gguf,
                     const cm::Resolved & options) {
    StagePlan out;
    cm::PlanInput in;
    const auto manager = cm::Manager::instance();
    in.coreml_built = true;
    in.generation_built = cm::generation::built();
    in.prebuilt_present =
        ::tts_cpp::detail::coreml_sidecar_exists(prebuilt_path(stage, gguf));
    in.platform = manager->platform();
    in.os = manager->os_version();
    if (in.generation_built && !in.prebuilt_present && options.enabled() &&
        options.generate != cm::Generate::never && !options.cache_dir.empty()) {
        out.candidate = std::move(
            cm::generation::probe({template_id(stage)}, gguf, options).front());
    }
    in.template_available = out.candidate.available;
    in.template_matched = out.candidate.matched;
    in.validated_units = out.candidate.memory.validated_units;
    out.units = options.units_or(cm::ComputeUnits::cpu_and_gpu);
    in.units = out.units;
    out.plan = cm::plan(in, options);
    return out;
}

struct Model {
    std::unique_ptr<CoremlModel> coreml;
    CoremlTensor input;
    CoremlTensor output;

    bool predict(const std::vector<float> & data, std::vector<float> & result) {
        if (data.size() != (size_t) tensor_elements(input)) return false;
        result.resize((size_t) tensor_elements(output));
        return coreml->predict({input}, {data.data()}, {output},
                               {result.data()});
    }
};

cm::LoadResult load_model(const cm::LoadRequest & request) {
    cm::LoadResult result;
    if (request.geometry.size() != GEOMETRY_SIZE) {
        result.error =
            cm::Error::deterministic("moss: invalid generated model geometry");
        return result;
    }

    auto model = std::make_shared<Model>();
    const bool tokenizer =
        request.stage == stage_name(GeneratedStage::speech_tokenizer);
    const auto & dims = request.geometry;
    model->input = {tokenizer ? "mel" : "latents",
                    {dims.begin(), dims.begin() + RANK}};
    model->output = {tokenizer ? "states" : "pcm",
                     {dims.begin() + RANK, dims.end()}};
    if (!cm::generation::begin_load(request, result)) return result;
    try {
        model->coreml =
            CoremlModel::open(request.artifact_path,
                              {false, false, cm::to_string(request.units)});
        if (!model->coreml) {
            result.error = cm::Error::transient(
                "moss: could not load generated Core ML model");
        } else if (model->coreml->stateful() ||
                   !model->coreml->declares(model->input) ||
                   !model->coreml->declares(model->output)) {
            result.error = cm::Error::deterministic(
                "moss: generated Core ML model has unexpected features");
        } else {
            result.model = model;
        }
    } catch (...) {
        model.reset();
        cm::generation::end_load(request, result);
        throw;
    }
    if (!result.model) model.reset();
    cm::generation::end_load(request, result);
    return result;
}

cm::Error warm_model(void * loaded, const cm::LoadRequest &) {
    auto & model = *static_cast<Model *>(loaded);
    if (!model.coreml->warm()) {
        return cm::Error::transient(
            "moss: generated Core ML warm-up prediction failed");
    }
    return {};
}

class HeldModel final : public GeneratedModel {
public:
    explicit HeldModel(cm::Handle handle) : handle_(std::move(handle)) {}
    bool predict(const std::vector<float> & input,
                 std::vector<float> & output) override {
        auto lock = handle_->lock_for_prediction();
        return handle_->as<Model>()->predict(input, output);
    }
    const char * label() const override { return handle_->label(); }
    void retire() override {
        handle_->retire("moss: generated Core ML prediction failed");
    }

private:
    cm::Handle handle_;
};

bool fits_with_envelope(uint64_t ggml_bytes, uint64_t budget,
                        uint64_t envelope) {
    return ggml_bytes <= budget && envelope <= budget - ggml_bytes;
}
#endif

} // namespace

struct MossCoremlGeneration::Impl {
    GeneratedStage stage;
    CoremlPolicy legacy;
#ifdef TTS_CPP_USE_COREML
    std::string gguf;
    cm::Resolved options;
    StagePlan route;
    cm::SourceStat before_load;
    bool have_source = false;
    std::shared_ptr<cm::Slot> slot;
#endif
};

MossCoremlGeneration::MossCoremlGeneration(GeneratedStage stage,
                                           const std::string & gguf,
                                           const CoremlOptions & options)
    : impl_(new Impl) {
    impl_->stage = stage;
    impl_->legacy = legacy_policy(options);
#ifdef TTS_CPP_USE_COREML
    impl_->gguf = gguf;
    impl_->options = resolve(options);
    impl_->route = plan_stage(stage, gguf, impl_->options);
    if (can_generate())
        impl_->have_source =
            cm::generation::source_stat(gguf, impl_->before_load);
#else
    (void) gguf;
#endif
}

MossCoremlGeneration::~MossCoremlGeneration() = default;

CoremlPolicy MossCoremlGeneration::prebuilt_policy() const {
    return impl_->legacy;
}

bool MossCoremlGeneration::can_generate() const {
#ifdef TTS_CPP_USE_COREML
    return impl_->route.plan.route == cm::Route::generate;
#else
    return false;
#endif
}

void MossCoremlGeneration::start(const std::vector<int64_t> & input,
                                 const std::vector<int64_t> & output,
                                 std::optional<uint64_t> ggml_bytes) {
#ifdef TTS_CPP_USE_COREML
    if (!can_generate()) return;
    auto & p = *impl_;
    if (::tts_cpp::detail::coreml_sidecar_exists(
            prebuilt_path(p.stage, p.gguf))) {
        p.route.plan = {cm::Route::prebuilt, cm::SlotState::unsupported, {}};
        return;
    }
    const auto budget = p.options.memory_budget_bytes;
    if (budget &&
        (!ggml_bytes || !fits_with_envelope(*ggml_bytes, (uint64_t) *budget,
                                            p.route.envelope()))) {
        // Memory-degrade resolves use=off for this stage, so a strict policy
        // never turns the ggml route into a refusal.
        p.route.plan = {cm::Route::none, cm::SlotState::disabled, "memory"};
        p.legacy.strict = false;
        return;
    }
    cm::SourceStat after_load;
    const bool have_after = cm::generation::source_stat(p.gguf, after_load);
    const auto source = cm::generation::loaded_source(
        p.before_load, p.have_source, after_load, have_after, p.gguf);
    cm::SlotSpec spec;
    spec.engine = "moss";
    spec.stage = stage_name(p.stage);
    spec.units = p.route.units;
    spec.route = cm::Route::generate;
    spec.tmpl = p.route.candidate.tmpl;
    spec.prepare =
        cm::generation::prepare_request(p.route.candidate, source, p.options);
    spec.prepare_fn = cm::generation::prepare;
    spec.load_fn = load_model;
    if (spec.units == cm::ComputeUnits::cpu_and_gpu) spec.warm_fn = warm_model;
    spec.geometry = input;
    spec.geometry.insert(spec.geometry.end(), output.begin(), output.end());
    spec.ane_compile_reserve_bytes = cm::generation::ane_compile_reserve_bytes(
        p.route.candidate.memory, spec.units);
    const auto manager = cm::Manager::instance();
    p.slot =
        manager->make_slot(std::move(spec), cm::SlotPolicy::from(p.options));
    if (p.options.prewarm != cm::Prewarm::none) {
        const bool wait = p.options.wait == cm::Wait::construct;
        auto group =
            manager->issue_group(cm::GroupKind::construction,
                                 {{p.slot, cm::Priority::minimal}}, wait);
        if (wait) group->wait();
    }
#else
    (void) input;
    (void) output;
    (void) ggml_bytes;
#endif
}

std::unique_ptr<GeneratedModel>
MossCoremlGeneration::acquire(const std::atomic<bool> * cancel) {
#ifdef TTS_CPP_USE_COREML
    if (impl_->slot) {
        try {
            auto got =
                impl_->slot->acquire(cm::Demand::now, std::nullopt, cancel);
            if (got.model)
                return std::make_unique<HeldModel>(std::move(got.model));
        } catch (const cm::CoremlError &) {
            // The caller keeps its existing stage-specific strict error.
        }
    }
#else
    (void) cancel;
#endif
    return nullptr;
}

bool MossCoremlGeneration::ready() const {
#ifdef TTS_CPP_USE_COREML
    return impl_->slot && impl_->slot->state() == cm::SlotState::ready;
#else
    return false;
#endif
}

std::string MossCoremlGeneration::state() const {
#ifdef TTS_CPP_USE_COREML
    return impl_->slot ? cm::to_string(impl_->slot->state()) : "idle";
#else
    return "idle";
#endif
}

std::string MossCoremlGeneration::status_json(bool prebuilt_ready) const {
#ifdef TTS_CPP_USE_COREML
    const auto & p = *impl_;
    if (p.slot) return cm::to_json(p.slot->status());
    cm::SlotStatus status;
    status.stage = stage_name(p.stage);
    status.function = "main";
    status.compute_units = p.legacy.units;
    status.state = p.route.plan.none_state;
    status.reason = p.route.plan.reason;
    status.label = "ggml";
    if (p.route.plan.route == cm::Route::prebuilt) {
        status.route = cm::StatusRoute::prebuilt;
        status.state =
            prebuilt_ready ? cm::SlotState::ready : cm::SlotState::failed;
        status.reason = prebuilt_ready ? "" : "no usable prebuilt sidecar";
        if (prebuilt_ready) status.label = cm::label_for(p.route.units);
    }
    return cm::to_json(status);
#else
    (void) prebuilt_ready;
    return std::string("{\"stage\":\"") + stage_name(impl_->stage) +
           "\",\"state\":\"unsupported\",\"reason\":\"build\",\"route\":"
           "\"none\",\"label\":\"ggml\"}";
#endif
}

GeneratedRoute plan_generated_route(GeneratedStage stage,
                                    const std::string & gguf,
                                    const CoremlOptions & options) {
    GeneratedRoute out;
#ifdef TTS_CPP_USE_COREML
    const auto resolved = resolve(options);
    const auto planned = plan_stage(stage, gguf, resolved);
    out.template_id = template_id(stage);
    out.route = cm::to_string(planned.plan.route);
    out.compute_units = cm::to_string(planned.units);
    const auto & validated = planned.candidate.memory.validated_units;
    out.validated = std::find(validated.begin(), validated.end(),
                              planned.units) != validated.end();
    if (planned.plan.route == cm::Route::none) {
        out.state =
            cm::state_label(planned.plan.none_state, planned.plan.reason);
    }
    if (planned.plan.route == cm::Route::generate) {
        const auto & m = planned.candidate.memory;
        out.envelope_bytes = planned.envelope();
        out.resident_bytes = cm::generation::resident_bytes(m);
        out.workspace_bytes = m.workspace_bytes;
        out.prepare_peak_bytes = cm::generation::prepare_peak_bytes(m);
        out.load_peak_bytes =
            cm::generation::load_peak_bytes(m, planned.units);
        out.ane_compile_reserve_bytes =
            cm::generation::ane_compile_reserve_bytes(m, planned.units);
    }
#else
    (void) stage;
    (void) gguf;
    (void) options;
    out.route = "none";
    out.state = "unsupported(build)";
#endif
    return out;
}

FitCoreml sfx_vae_coreml_fit(const std::string & gguf,
                             const CoremlOptions & options, uint64_t ggml_bytes,
                             uint64_t budget, bool ggml_fits) {
    FitCoreml result;
    result.ggml_only_fits = ggml_fits;
    result.fits = ggml_fits;
#ifdef TTS_CPP_USE_COREML
    const auto resolved = resolve(options);
    const auto planned = plan_stage(GeneratedStage::sfx_vae, gguf, resolved);
    result.route = cm::to_string(planned.plan.route);
    if (planned.plan.route == cm::Route::generate) {
        result.bytes = planned.envelope();
        result.validated = true;
        result.fits =
            ggml_fits && fits_with_envelope(ggml_bytes, budget, result.bytes);
        if (resolved.memory_budget_bytes) {
            const uint64_t cap =
                std::min(budget, (uint64_t) *resolved.memory_budget_bytes);
            if (ggml_bytes <= cap &&
                !fits_with_envelope(ggml_bytes, cap, result.bytes)) {
                result.disabled_reason = "memory";
            }
        }
    }
#else
    (void) gguf;
    (void) options;
    (void) ggml_bytes;
    (void) budget;
#endif
    return result;
}

} // namespace tts_cpp::moss::detail
