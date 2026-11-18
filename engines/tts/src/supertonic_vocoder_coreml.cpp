// The Supertonic vocoder on the shared Core ML runtime (supertonic_vocoder_coreml.h).

#include "supertonic_vocoder_coreml.h"

#include "coreml_sidecar_path.h"
#include "supertonic_coreml_path.h"
#include "supertonic_internal.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <stdexcept>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#ifdef TTS_CPP_USE_COREML
#include "supertonic_coreml_vocoder.h"

#include "qvac_coreml/generation.h"
#include "qvac_coreml/manager.h"
#include "qvac_coreml/options.h"
#include "qvac_coreml/slot.h"
#include "qvac_coreml/status.h"

#include <sys/stat.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <thread>
#endif

namespace tts_cpp {
namespace supertonic {
namespace detail {

// ---- device class ---------------------------------------------------------------

std::string supertonic_device_class(const std::string & cpu_brand) {
    // "Apple M<generation>[ <tier>][ (<qualifier>)]"
    static const std::string prefix = "Apple M";
    if (cpu_brand.compare(0, prefix.size(), prefix) != 0) return {};
    size_t i = prefix.size();
    if (i >= cpu_brand.size() || !std::isdigit(static_cast<unsigned char>(cpu_brand[i]))) return {};
    while (i < cpu_brand.size() && std::isdigit(static_cast<unsigned char>(cpu_brand[i]))) ++i;
    if (i == cpu_brand.size() || (cpu_brand[i] == ' ' && i + 1 < cpu_brand.size() && cpu_brand[i + 1] == '(')) {
        return "base";
    }
    if (cpu_brand[i] != ' ') return {};
    const size_t end = cpu_brand.find(' ', i + 1);
    std::string tier = cpu_brand.substr(i + 1, end == std::string::npos ? std::string::npos : end - i - 1);
    std::transform(tier.begin(), tier.end(), tier.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (tier == "pro" || tier == "max" || tier == "ultra") return tier;
    return {};
}

std::string supertonic_cpu_brand() {
#if defined(__APPLE__)
    char buf[256] = {};
    size_t n = sizeof(buf);
    if (sysctlbyname("machdep.cpu.brand_string", buf, &n, nullptr, 0) == 0) return std::string(buf);
#endif
    return {};
}

bool supertonic_prefers_ggml(const std::vector<std::string> & prefer_ggml_on, const std::string & cpu_brand) {
    const std::string cls = supertonic_device_class(cpu_brand);
    if (cls.empty()) return false;
    return std::find(prefer_ggml_on.begin(), prefer_ggml_on.end(), cls) != prefer_ggml_on.end();
}

bool supertonic_vocoder_prebuilt_below_floor(const supertonic_model & model) {
    return std::getenv("SUPERTONIC_COREML_ALLOW_LOW_BIT") == nullptr &&
           supertonic_first_low_bit_vocoder_weight(model.vocoder, model.low_bit_source_tensors) != nullptr;
}

std::string supertonic_merge_vocoder_backend(const std::string & acc, const std::string & next) {
    if (acc.empty() || acc == next) return next;
    return SUPERTONIC_VOCODER_MIXED_BACKEND;
}

namespace {

constexpr const char * ENGINE_NAME = "supertonic";
constexpr const char * STAGE_NAME = "vocoder";
constexpr const char * ENV_PREFIX = "SUPERTONIC";
constexpr const char * STRICT_ENV = "SUPERTONIC_COREML_STRICT";

bool valid_units(const std::string & u) {
    return u == "all" || u == "cpu_and_ane" || u == "cpu_and_gpu" || u == "cpu_only";
}

void validate_options(const CoremlOptions & o) {
    if (o.compute_units && !valid_units(*o.compute_units)) {
        throw std::invalid_argument("supertonic: invalid coreml.compute_units='" + *o.compute_units +
                                    "' (expected all, cpu_and_ane, cpu_and_gpu or cpu_only)");
    }
    if (o.strict_wait_ms && *o.strict_wait_ms < 0) {
        throw std::invalid_argument("supertonic: coreml.strict_wait_ms must not be negative");
    }
    if (o.memory_budget_bytes && *o.memory_budget_bytes < 0) {
        throw std::invalid_argument("supertonic: coreml.memory_budget_bytes must not be negative");
    }
}

}  // namespace

#ifdef TTS_CPP_USE_COREML

namespace pcm = QVAC_COREML_NS;

namespace {

template <class E, class F> F map_enum(E v) {
    return static_cast<F>(static_cast<int>(v));
}

pcm::Options to_runtime_options(const CoremlOptions & o) {
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

uint64_t file_bytes(const std::string & path) {
    struct stat st {};
    return ::stat(path.c_str(), &st) == 0 ? (uint64_t) st.st_size : 0;
}

// The ggml vocoder costs about 0.2 ms per latent frame on an Apple GPU
// (40 ms for a 22.9 s utterance on an M4, docs/supertonic.md) and more on the
// CPU. The bounded wait for a warm load never exceeds what the call would cost
// on ggml.
constexpr double GGML_VOCODER_MS_PER_FRAME = 0.5;
constexpr int64_t GGML_VOCODER_MS_FLOOR = 10;
constexpr double ANE_COMPILE_RESERVE_FACTOR = 1.6;
constexpr size_t GEOMETRY_SIZE = 3;  // {latent_channels, samples_per_frame, context_frames}

// The loaded model a slot holds.
struct vocoder_model {
    supertonic_coreml_vocoder_context * ctx = nullptr;
    ~vocoder_model() {
        if (ctx) supertonic_coreml_vocoder_free(ctx);
    }
};

// LoadFn. Takes the shared lease of a generated artifact first, then loads
// the model on the request's compute units and validates the geometry the
// constructing thread passed by value: {latent_channels, samples_per_frame,
// context_frames}.
pcm::LoadResult load_vocoder(const pcm::LoadRequest & req) {
    pcm::LoadResult r;
    if (!pcm::generation::begin_load(req, r)) return r;
    auto fail = [&](const pcm::Error & e) {
        r.model.reset();
        r.error = e;
        pcm::generation::end_load(req, r);
        return r;
    };
    if (req.geometry.size() != GEOMETRY_SIZE) {
        return fail(pcm::Error::deterministic("supertonic: vocoder load without its geometry"));
    }
    const int64_t channels = req.geometry[0], samples_per_frame = req.geometry[1], context = req.geometry[2];
    int why = SUPERTONIC_COREML_VOCODER_OK;
    supertonic_coreml_vocoder_context * ctx = supertonic_coreml_vocoder_init(
        req.artifact_path.c_str(), channels, samples_per_frame, pcm::to_string(req.units), &why);
    if (ctx == nullptr) {
        if (why == SUPERTONIC_COREML_VOCODER_LOAD_FAILED) {
            return fail(pcm::Error::transient("supertonic: Core ML could not load " + req.artifact_path));
        }
        return fail(pcm::Error::deterministic("supertonic: " + req.artifact_path + " does not declare latent [1, " +
                                              std::to_string(channels) + ", W] and wav [1, 1, W * " +
                                              std::to_string(samples_per_frame) + "]"));
    }
    const int64_t window = supertonic_coreml_vocoder_window_frames(ctx);
    if (window <= context) {
        supertonic_coreml_vocoder_free(ctx);
        return fail(pcm::Error::deterministic("supertonic: " + req.artifact_path + ": window " +
                                              std::to_string(window) + " does not clear " +
                                              std::to_string(context) + " frames of causal context"));
    }
    auto * vm = new vocoder_model;
    vm->ctx = ctx;
    r.model = std::shared_ptr<void>(vm, [](void * p) { delete static_cast<vocoder_model *>(p); });
    pcm::generation::end_load(req, r);
    return r;
}

// WarmFn: one prediction on a silent window, so a call's first prediction does
// not pay the first-run cost.
pcm::Error warm_vocoder(void * model, const pcm::LoadRequest & req) {
    auto & vm = *static_cast<vocoder_model *>(model);
    const int64_t channels = req.geometry.size() == GEOMETRY_SIZE ? req.geometry[0] : 0;
    const int64_t samples_per_frame = req.geometry.size() == GEOMETRY_SIZE ? req.geometry[1] : 0;
    const int64_t window = supertonic_coreml_vocoder_window_frames(vm.ctx);
    std::vector<float> latent((size_t) (channels * window), 0.0f);
    std::vector<float> wav((size_t) (window * samples_per_frame));
    if (supertonic_coreml_vocoder_synthesize(vm.ctx, latent.data(), window, 0, window, 0, window, wav.data()) != 0) {
        return pcm::Error::transient("supertonic: the Core ML vocoder warm-up prediction failed");
    }
    return {};
}

// One call's sidecar: the handle acquired at the call's entry.
class handle_sidecar final : public supertonic_vocoder_sidecar {
public:
    explicit handle_sidecar(pcm::Handle h) : h_(std::move(h)), vm_(h_->as<vocoder_model>()) {}
    int window() const override { return (int) supertonic_coreml_vocoder_window_frames(vm_->ctx); }
    const char * label() const override { return h_->label(); }
    bool synthesize(const float * latent, int64_t latent_len, int64_t begin, int64_t filled, int64_t skip,
                    int64_t keep, float * wav) override {
        auto lock = h_->lock_for_prediction();
        return supertonic_coreml_vocoder_synthesize(vm_->ctx, latent, latent_len, begin, filled, skip, keep, wav) ==
               0;
    }
    void retire(const std::string & reason) { h_->retire(reason); }

private:
    pcm::Handle h_;
    vocoder_model * vm_;
};

struct planned {
    supertonic_vocoder_route route;
    pcm::Plan plan;
    pcm::ComputeUnits units = pcm::ComputeUnits::all;
    std::optional<pcm::generation::Candidate> candidate;
    std::string prebuilt_path;
};

// `prebuilt_below_floor`: the GGUF's vocoder weights are stored below the
// prebuilt sidecar's weight floor, so an adjacent prebuilt does not count.
planned plan_route(const std::string & gguf_path, const pcm::Resolved & r, bool prebuilt_below_floor,
                   const std::string & cpu_brand) {
    planned out;
    out.route.strict = r.strict();
    out.prebuilt_path = coreml_vocoder_sidecar_path(gguf_path);
    const std::shared_ptr<pcm::Manager> mgr = pcm::Manager::instance();
    pcm::PlanInput in;
    in.coreml_built = true;
    in.generation_built = pcm::generation::built();
    in.prebuilt_present = ::tts_cpp::detail::coreml_sidecar_exists(out.prebuilt_path) && !prebuilt_below_floor;
    if (in.generation_built && !in.prebuilt_present && r.enabled()) {
        std::vector<pcm::generation::Candidate> cands = pcm::generation::probe(
            {SUPERTONIC_VOCODER_TEMPLATE_W64, SUPERTONIC_VOCODER_TEMPLATE_PW2D_W64}, gguf_path, r);
        // The first template that matches; otherwise the first available one,
        // whose reason says why it does not match.
        for (pcm::generation::Candidate & c : cands) {
            if (c.available && c.matched) {
                out.candidate = std::move(c);
                break;
            }
        }
        if (!out.candidate) {
            for (pcm::generation::Candidate & c : cands) {
                if (c.available) {
                    out.candidate = std::move(c);
                    break;
                }
            }
        }
    }
    const bool available = out.candidate && out.candidate->available;
    out.units = r.units_or(available && out.candidate->default_units ? *out.candidate->default_units
                                                                     : pcm::ComputeUnits::all);
    out.route.compute_units = pcm::to_string(out.units);
    in.template_available = available;
    in.template_matched = available && out.candidate->matched;
    if (available) {
        in.validated_units = out.candidate->memory.validated_units;
        // The template's device preference holds for use = automatic; use =
        // require generates anyway.
        in.device_excluded =
            !r.strict() && supertonic_prefers_ggml(out.candidate->prefer_ggml_on,
                                                   cpu_brand.empty() ? supertonic_cpu_brand() : cpu_brand);
    }
    in.units = out.units;
    in.platform = mgr->platform();
    in.os = mgr->os_version();
    out.plan = pcm::plan(in, r);
    out.route.route = pcm::to_string(out.plan.route);
    if (out.plan.route == pcm::Route::none) {
        out.route.state = pcm::to_string(out.plan.none_state);
        out.route.reason = out.plan.reason;
        if (out.plan.none_state == pcm::SlotState::no_template && out.candidate && !out.candidate->reason.empty()) {
            out.route.reason += ": " + out.candidate->reason;
        }
    }
    if (out.plan.route == pcm::Route::generate) {
        const pcm::generation::TemplateMemory & m = out.candidate->memory;
        out.route.template_id = out.candidate->template_id;
        out.route.envelope_bytes = pcm::generation::envelope_bytes(m, out.units);
        out.route.resident_bytes = pcm::generation::resident_bytes(m);
        out.route.workspace_bytes = m.workspace_bytes;
        out.route.prepare_peak_bytes = pcm::generation::prepare_peak_bytes(m);
        out.route.load_peak_bytes = pcm::generation::load_peak_bytes(m, out.units);
        out.route.ane_compile_reserve_bytes = pcm::generation::ane_compile_reserve_bytes(m, out.units);
        out.route.validated = std::find(m.validated_units.begin(), m.validated_units.end(), out.units) !=
                              m.validated_units.end();
    }
    return out;
}

}  // namespace

struct supertonic_vocoder_coreml::impl {
    pcm::Resolved resolved;
    planned plan;
    std::shared_ptr<pcm::Manager> manager;
    std::shared_ptr<pcm::Slot> slot;
    pcm::SlotStatus none_status;
    std::shared_ptr<pcm::Group> construction;
};

supertonic_vocoder_route supertonic_vocoder_plan_route(const std::string & gguf_path, const CoremlOptions & options,
                                                       const std::string & cpu_brand, bool prebuilt_below_floor) {
    validate_options(options);
    const pcm::Resolved r = pcm::resolve_coreml(to_runtime_options(options), ENV_PREFIX);
    return plan_route(gguf_path, r, prebuilt_below_floor, cpu_brand).route;
}

struct supertonic_vocoder_source {
    pcm::SourceStat stat;  // fstat of the descriptor the load read through
    bool ok = false;       // false: no descriptor or no fstat
};

void supertonic_vocoder_source_record(supertonic_model & model, int fd, const std::string & path) {
    auto s = std::make_shared<supertonic_vocoder_source>();
    s->ok = fd >= 0 && pcm::generation::source_stat_fd(fd, path, s->stat);
    model.vocoder_source = std::move(s);
}

std::shared_ptr<supertonic_vocoder_coreml> supertonic_vocoder_coreml::create(supertonic_model & model,
                                                                             const std::string & gguf_path,
                                                                             const CoremlOptions & options,
                                                                             bool memory_degrade,
                                                                             const std::string & cpu_brand) {
    validate_options(options);
    auto p = std::make_unique<impl>();
    p->resolved = pcm::resolve_coreml(to_runtime_options(options), ENV_PREFIX);
    for (const std::string & w : p->resolved.warnings) std::fprintf(stderr, "supertonic: %s\n", w.c_str());
    if (memory_degrade && p->resolved.enabled()) {
        p->resolved.use = pcm::Use::off;
        p->resolved.disabled_reason = "memory";
    }
    p->manager = pcm::Manager::instance();
    p->plan = plan_route(gguf_path, p->resolved, supertonic_vocoder_prebuilt_below_floor(model), cpu_brand);
    const supertonic_hparams & hp = model.hparams;
    const int context = supertonic_coreml_vocoder_context_frames(model);

    if (p->plan.plan.route == pcm::Route::none) {
        p->none_status.stage = STAGE_NAME;
        p->none_status.function = "main";
        p->none_status.state = p->plan.plan.none_state;
        p->none_status.reason = p->plan.plan.reason;
        p->none_status.compute_units = p->plan.route.compute_units;
        p->none_status.label = SUPERTONIC_VOCODER_GGML_BACKEND;
        if (p->resolved.strict()) {
            throw std::runtime_error("supertonic: no usable Core ML vocoder sidecar (" + p->plan.route.state + ": " +
                                     p->plan.route.reason + "); Core ML strict policy is enabled (use = require / " +
                                     STRICT_ENV + ")");
        }
        // An empty construction group: the prewarm scope's vocoder is listed
        // as skipped (route none), and the group reports nothing_to_prepare.
        std::vector<pcm::SlotStatus> skipped;
        if (p->resolved.prewarm != pcm::Prewarm::none) skipped.push_back(p->none_status);
        p->construction = p->manager->issue_group(pcm::GroupKind::construction, {},
                                                  p->resolved.wait == pcm::Wait::construct, std::move(skipped));
        // Installed on the none route too: every call then takes the
        // resolved policy from it (no slot: ggml, never strict) instead of
        // reading the environment.
        auto out = std::shared_ptr<supertonic_vocoder_coreml>(new supertonic_vocoder_coreml(std::move(p)));
        model.vocoder_pending = out;
        return out;
    }

    pcm::SlotSpec spec;
    spec.engine = ENGINE_NAME;
    spec.stage = STAGE_NAME;
    spec.function = "main";
    spec.units = p->plan.units;
    spec.route = p->plan.plan.route;
    spec.load_fn = &load_vocoder;
    spec.warm_fn = &warm_vocoder;
    spec.geometry = {hp.latent_channels, (int64_t) hp.base_chunk_size * hp.ttl_chunk_compress_factor, context};
    spec.shareable = true;
    if (spec.route == pcm::Route::prebuilt) {
        spec.artifact_path = p->plan.prebuilt_path;
        spec.ane_compile_reserve_bytes =
            pcm::units_include_ane(spec.units)
                ? (uint64_t) (ANE_COMPILE_RESERVE_FACTOR * (double) file_bytes(spec.artifact_path +
                                                                               "/weights/weight.bin"))
                : 0;
    } else {
        const pcm::generation::Candidate & c = *p->plan.candidate;
        // The file the weights were read from (fstat of the load's one
        // descriptor), not whatever `gguf_path` names now: prepare opens the
        // path and fails with source_changed unless it is that file, and
        // asking for another path than the load's is refused here; the
        // vocoder then stays on ggml.
        pcm::SourceStat source;
        pcm::SourceStat asked;
        const bool loaded = model.vocoder_source && model.vocoder_source->ok;
        if (loaded) source = model.vocoder_source->stat;
        source.changed = !loaded || !pcm::generation::source_stat(gguf_path, asked) || asked.path != source.path;
        spec.tmpl = c.tmpl;
        spec.prepare = pcm::generation::prepare_request(c, source, p->resolved);
        spec.prepare_fn = &pcm::generation::prepare;
        spec.ane_compile_reserve_bytes = pcm::generation::ane_compile_reserve_bytes(c.memory, spec.units);
    }
    p->slot = p->manager->make_slot(std::move(spec), pcm::SlotPolicy::from(p->resolved));

    std::vector<pcm::GroupRequest> requests;
    if (p->resolved.prewarm != pcm::Prewarm::none) requests.push_back({p->slot, pcm::Priority::minimal});
    const bool waited = p->resolved.wait == pcm::Wait::construct;
    p->construction = p->manager->issue_group(pcm::GroupKind::construction, requests, waited);
    auto out = std::shared_ptr<supertonic_vocoder_coreml>(new supertonic_vocoder_coreml(std::move(p)));
    if (waited) {
        // With prewarm none the group is empty and nothing waits; strict
        // throws only for a slot of the group that ended non-ready.
        const pcm::PrewarmResult result = out->p_->construction->wait();
        if (out->p_->resolved.strict() && !requests.empty() && !result.success) {
            const pcm::SlotStatus st = out->p_->slot->status();
            throw std::runtime_error("supertonic: the Core ML vocoder sidecar is not ready after construction (" +
                                     pcm::state_label(st.state, st.reason) + (st.error.empty() ? "" : ": " + st.error) +
                                     "); Core ML strict policy is enabled (use = require / " + STRICT_ENV + ")");
        }
    }
    model.vocoder_pending = out;
    return out;
}

supertonic_vocoder_coreml::supertonic_vocoder_coreml(std::unique_ptr<impl> p) : p_(std::move(p)) {}
supertonic_vocoder_coreml::~supertonic_vocoder_coreml() = default;

supertonic_vocoder_call supertonic_vocoder_coreml::acquire(int latent_len, const std::atomic<bool> * cancel) {
    supertonic_vocoder_call call;
    if (!p_->slot) return call;
    const int64_t budget_ms = std::max<int64_t>(
        GGML_VOCODER_MS_FLOOR, (int64_t) (GGML_VOCODER_MS_PER_FRAME * (double) std::max(latent_len, 0)));
    const auto deadline = pcm::Clock::now() + std::chrono::milliseconds(budget_ms);
    pcm::Acquired got;
    try {
        got = p_->slot->acquire(pcm::Demand::now, deadline, cancel);
    } catch (const pcm::CoremlError & e) {
        throw std::runtime_error(std::string("supertonic: the Core ML vocoder sidecar cannot serve this call (") +
                                 e.what() + "); Core ML strict policy is enabled (use = require / " + STRICT_ENV +
                                 ")");
    }
    call.pending = got.pending;
    if (!got.model) return call;
    auto sidecar = std::make_shared<handle_sidecar>(std::move(got.model));
    call.retire = [sidecar](const std::string & reason) { sidecar->retire(reason); };
    call.sidecar = std::move(sidecar);
    return call;
}

bool supertonic_vocoder_coreml::strict() const { return p_->resolved.strict(); }
const supertonic_vocoder_route & supertonic_vocoder_coreml::route() const { return p_->plan.route; }
std::string supertonic_vocoder_coreml::state() const { return p_->slot ? pcm::to_string(p_->slot->state()) : ""; }

std::string supertonic_vocoder_coreml::status_json() const {
    pcm::GenerationStatus gen;
    gen.supported = pcm::generation::built() && p_->manager->platform() == pcm::Platform::macos &&
                    pcm::os_at_least(p_->manager->os_version(), 15, 0);
    if (!gen.supported) {
        gen.reason = !pcm::generation::built() ? "build" : "os";
    } else if (p_->plan.plan.route == pcm::Route::none &&
               (p_->plan.plan.reason == "unvalidated_units" || p_->plan.plan.reason == "device")) {
        gen.reason = p_->plan.plan.reason;
    }
    if (p_->slot) return pcm::to_json(p_->manager->status(ENGINE_NAME, p_->resolved.cache_dir, {p_->slot}, gen));
    pcm::EngineStatus st;
    st.engine = ENGINE_NAME;
    st.cache_dir = p_->resolved.cache_dir;
    st.generation = gen;
    st.stages.push_back(p_->none_status);
    return pcm::to_json(st);
}

std::string supertonic_vocoder_coreml::construction_json() const {
    return p_->construction ? pcm::to_json(p_->construction->result()) : std::string("{}");
}

std::string supertonic_vocoder_coreml::prewarm(CoremlOptions::Prewarm scope, bool wait,
                                               const std::atomic<uint64_t> * cancels, uint64_t since) const {
    const bool wanted = scope != CoremlOptions::Prewarm::none;
    std::vector<pcm::GroupRequest> requests;
    std::vector<pcm::SlotStatus> skipped;
    if (wanted && p_->slot) requests.push_back({p_->slot, pcm::Priority::minimal});
    if (wanted && !p_->slot) skipped.push_back(p_->none_status);
    const std::shared_ptr<pcm::Group> group =
        p_->manager->issue_group(pcm::GroupKind::prewarm, requests, wait, std::move(skipped));
    if (!wait || requests.empty()) return pcm::to_json(group->result());
    for (;;) {
        pcm::PrewarmResult r = group->wait(std::chrono::milliseconds(50));
        if (group->finished() || (cancels && cancels->load() != since)) return pcm::to_json(r);
    }
}

bool supertonic_vocoder_coreml::wait(int64_t timeout_ms, const std::atomic<uint64_t> * cancels, uint64_t since) const {
    if (!p_->slot) return false;
    return supertonic_wait_until_ready(
        [this] {
            const pcm::SlotState st = p_->slot->state();
            if (st == pcm::SlotState::ready) return 1;
            // Terminal, or nothing will move it without a request: idle
            // (prewarm none before the first call, or evicted) and
            // deferred(memory).
            if (pcm::is_terminal(st) || st == pcm::SlotState::idle || st == pcm::SlotState::deferred_memory) {
                return -1;
            }
            return 0;
        },
        timeout_ms, cancels, since);
}

#else  // !TTS_CPP_USE_COREML

void supertonic_vocoder_source_record(supertonic_model &, int, const std::string &) {}

struct supertonic_vocoder_coreml::impl {
    supertonic_vocoder_route route;
};

supertonic_vocoder_route supertonic_vocoder_plan_route(const std::string &, const CoremlOptions & options,
                                                       const std::string &, bool) {
    validate_options(options);
    supertonic_vocoder_route r;
    r.state = "unsupported";
    r.reason = "build";
    return r;
}

std::shared_ptr<supertonic_vocoder_coreml> supertonic_vocoder_coreml::create(supertonic_model &, const std::string &,
                                                                             const CoremlOptions & options, bool,
                                                                             const std::string &) {
    validate_options(options);
    return nullptr;
}

supertonic_vocoder_coreml::supertonic_vocoder_coreml(std::unique_ptr<impl> p) : p_(std::move(p)) {}
supertonic_vocoder_coreml::~supertonic_vocoder_coreml() = default;
supertonic_vocoder_call supertonic_vocoder_coreml::acquire(int, const std::atomic<bool> *) { return {}; }
bool supertonic_vocoder_coreml::strict() const { return false; }
const supertonic_vocoder_route & supertonic_vocoder_coreml::route() const { return p_->route; }
std::string supertonic_vocoder_coreml::state() const { return {}; }
std::string supertonic_vocoder_coreml::status_json() const { return "{}"; }
bool supertonic_vocoder_coreml::wait(int64_t, const std::atomic<uint64_t> *, uint64_t) const { return false; }
std::string supertonic_vocoder_coreml::construction_json() const { return "{}"; }
std::string supertonic_vocoder_coreml::prewarm(CoremlOptions::Prewarm, bool, const std::atomic<uint64_t> *,
                                               uint64_t) const {
    return "{}";
}

#endif

bool supertonic_wait_until_ready(const std::function<int()> & poll, int64_t timeout_ms,
                                 const std::atomic<uint64_t> * cancels, uint64_t since) {
    const auto start = std::chrono::steady_clock::now();
    for (;;) {
        const int r = poll();
        if (r != 0) return r > 0;
        if (cancels && cancels->load() != since) return false;
        if (timeout_ms >= 0 && std::chrono::steady_clock::now() - start >= std::chrono::milliseconds(timeout_ms)) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

}  // namespace detail
}  // namespace supertonic
}  // namespace tts_cpp
