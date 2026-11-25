#include "cosyvoice_coreml_stages.h"

#include "coreml_sidecar_path.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>

#ifdef TTS_CPP_USE_COREML
#include "moss/coreml/runner.h"

#include "qvac_coreml/generation.h"
#include "qvac_coreml/manager.h"
#include "qvac_coreml/options.h"
#include "qvac_coreml/slot.h"
#include "qvac_coreml/status.h"

#include <sys/stat.h>

#include <optional>
#endif

namespace {
[[maybe_unused]] constexpr int max_feature_dimensions = 8;
constexpr const char * UNITS_ENV = "COSYVOICE_COREML_COMPUTE_UNITS";
// The legacy value of UNITS_ENV that turns Core ML off.
constexpr const char * UNITS_DISABLED = "disabled";

[[maybe_unused]] bool units_env_disabled() {
    const char * units = std::getenv(UNITS_ENV);
    return units && std::string(units) == UNITS_DISABLED;
}
}  // namespace

const char * cosyvoice_stage_name(cosyvoice_stage s) {
    switch (s) {
        case cosyvoice_stage::frontend: return "frontend";
        case cosyvoice_stage::f0: return "f0";
        case cosyvoice_stage::hift: return "hift";
    }
    return "?";
}

const char * cosyvoice_stage_suffix(cosyvoice_stage s) {
    switch (s) {
        case cosyvoice_stage::frontend: return "-frontend";
        case cosyvoice_stage::f0: return "-f0";
        case cosyvoice_stage::hift: return "";
    }
    return "";
}

std::string cosyvoice_merge_label(const std::string & acc, const std::string & next) {
    if (acc.empty() || acc == next) return next;
    return COSYVOICE_MIXED_LABEL;
}

namespace {

bool valid_units(const std::string & u) {
    return u == "all" || u == "cpu_and_ane" || u == "cpu_and_gpu" || u == "cpu_only";
}

void validate_options(const tts_cpp::CoremlOptions & o) {
    if (o.compute_units && !valid_units(*o.compute_units)) {
        throw std::invalid_argument("cosyvoice: invalid coreml.compute_units='" + *o.compute_units +
                                    "' (expected all, cpu_and_ane, cpu_and_gpu or cpu_only)");
    }
    if (o.strict_wait_ms && *o.strict_wait_ms < 0) {
        throw std::invalid_argument("cosyvoice: coreml.strict_wait_ms must not be negative");
    }
    if (o.memory_budget_bytes && *o.memory_budget_bytes < 0) {
        throw std::invalid_argument("cosyvoice: coreml.memory_budget_bytes must not be negative");
    }
}

}  // namespace

#ifdef TTS_CPP_USE_COREML

namespace pcm = QVAC_COREML_NS;

namespace {

constexpr const char * ENGINE_NAME = "cosyvoice";
constexpr const char * ENV_PREFIX = "COSYVOICE";
// CosyVoice's Core ML stages default to the CPU and the Neural Engine.
constexpr const char * DEFAULT_UNITS = "cpu_and_ane";

constexpr const char * TEMPLATE_IDS[COSYVOICE_STAGE_COUNT] = {
    "cosyvoice3-frontend-f1024", "cosyvoice3-hift-f0-m512", "cosyvoice3-hift-m512"};

// The legacy COSYVOICE_COREML_COMPUTE_UNITS=disabled turns Core ML off, as
// COSYVOICE_COREML_DISABLE does, unless the options set `use` explicitly.
tts_cpp::CoremlOptions with_legacy_env(tts_cpp::CoremlOptions o) {
    if (o.use == tts_cpp::CoremlOptions::Use::unset && units_env_disabled()) o.use = tts_cpp::CoremlOptions::Use::off;
    return o;
}

// The bounded wait of an acquisition never exceeds what the unit costs on
// ggml (Metal on an M2; the CPU is slower): per token for the front-end
// (10 ms for 480 tokens), per mel frame for the F0 predictor (13 ms for 410
// frames) and the HiFT decode (1.7 s for 410 frames).
constexpr int64_t GGML_FLOOR_MS = 20;
constexpr double GGML_MS_PER_FRAME[COSYVOICE_STAGE_COUNT] = {0.02, 0.03, 4.0};


template <class E, class F> F map_enum(E v) {
    return static_cast<F>(static_cast<int>(v));
}

pcm::Options to_runtime_options(const tts_cpp::CoremlOptions & o) {
    using O = tts_cpp::CoremlOptions;
    static_assert(static_cast<int>(O::Use::require) == static_cast<int>(pcm::Use::require), "Use");
    static_assert(static_cast<int>(O::Generate::background) == static_cast<int>(pcm::Generate::background), "Generate");
    static_assert(static_cast<int>(O::Wait::call) == static_cast<int>(pcm::Wait::call), "Wait");
    static_assert(static_cast<int>(O::Prewarm::all) == static_cast<int>(pcm::Prewarm::all), "Prewarm");
    pcm::Options r;
    r.use = map_enum<O::Use, pcm::Use>(o.use);
    r.generate = map_enum<O::Generate, pcm::Generate>(o.generate);
    r.wait = map_enum<O::Wait, pcm::Wait>(o.wait);
    r.prewarm = map_enum<O::Prewarm, pcm::Prewarm>(o.prewarm);
    r.cache_dir = o.cache_dir;
    r.compute_units = o.compute_units;
    r.strict_wait_ms = o.strict_wait_ms;
    r.memory_budget_bytes = o.memory_budget_bytes;
    return r;
}

// The environment the resolution reads, minus the legacy "disabled" units
// value (with_legacy_env handles it), so it is not reported as invalid.
const char * resolve_getenv(const char * name) {
    if (std::strcmp(name, UNITS_ENV) == 0 && units_env_disabled()) return nullptr;
    return std::getenv(name);
}

pcm::Resolved resolve(const tts_cpp::CoremlOptions & options) {
    validate_options(options);
    return pcm::resolve_coreml(to_runtime_options(with_legacy_env(options)), ENV_PREFIX, &resolve_getenv);
}

// The loaded model a slot holds.
struct stage_model {
    moss_coreml_model * model = nullptr;
    ~stage_model() {
        if (model) moss_coreml_free(model);
    }
};

std::vector<int64_t> dims_of(moss_coreml_model * m, const char * name) {
    int64_t d[max_feature_dimensions] = {};
    const int rank = moss_coreml_feature_dims(m, name, d, max_feature_dimensions);
    return std::vector<int64_t>(d, d + std::max(rank, 0));
}

constexpr int64_t MEL = 80;
constexpr int64_t SPECTRUM = 18;
constexpr int64_t STFT_FRAMES_PER_MEL = 120;
constexpr int64_t SPEAKER = 192;

// The interface the stage's engine code feeds (cosyvoice_coreml_pipeline.cpp)
// at the model's own capacity; empty when it does not have one.
std::string interface_error(cosyvoice_stage s, moss_coreml_model * m) {
    using V = std::vector<int64_t>;
    switch (s) {
        case cosyvoice_stage::frontend: {
            const V ids = dims_of(m, "token_ids");
            const int64_t c = ids.size() == 2 ? ids[1] : 0;
            if (c > 0 && dims_of(m, "token_mask") == V{1, c, 1} && dims_of(m, "speaker") == V{1, SPEAKER} &&
                dims_of(m, "mu") == V{1, 2 * c, MEL} && dims_of(m, "spks") == V{1, MEL}) {
                return {};
            }
            return "does not declare token_ids [1, T], token_mask [1, T, 1], speaker [1, 192], mu [1, 2T, 80] "
                   "and spks [1, 80]";
        }
        case cosyvoice_stage::f0: {
            const V mel = dims_of(m, "mel");
            const int64_t c = mel.size() == 3 ? mel[2] : 0;
            if (c > 0 && mel[1] == MEL && dims_of(m, "f0") == V{1, c}) return {};
            return "does not declare mel [1, 80, T] and f0 [1, T]";
        }
        case cosyvoice_stage::hift: {
            const V mel = dims_of(m, "mel");
            const int64_t c = mel.size() == 3 ? mel[2] : 0;
            const V src{1, SPECTRUM, c * STFT_FRAMES_PER_MEL + 1};
            if (c > 0 && mel[1] == MEL && dims_of(m, "source_stft") == src && dims_of(m, "spec") == src) return {};
            return "does not declare mel [1, 80, T], source_stft and spec [1, 18, 120 T + 1]";
        }
    }
    return "unknown stage";
}

// LoadFn. Takes the shared lease of a generated artifact first, then loads
// the model on the request's compute units and checks the interface of the
// stage named by the geometry the constructing thread passed by value.
pcm::LoadResult load_stage(const pcm::LoadRequest & req) {
    pcm::LoadResult r;
    if (!pcm::generation::begin_load(req, r)) return r;
    auto fail = [&](const pcm::Error & e) {
        r.model.reset();
        r.error = e;
        pcm::generation::end_load(req, r);
        return r;
    };
    if (req.geometry.size() != 1 || req.geometry[0] < 0 || req.geometry[0] >= COSYVOICE_STAGE_COUNT) {
        return fail(pcm::Error::deterministic("cosyvoice: stage load without its geometry"));
    }
    const auto stage = static_cast<cosyvoice_stage>(req.geometry[0]);
    moss_coreml_model * m = moss_coreml_load(req.artifact_path.c_str(), pcm::to_string(req.units));
    if (m == nullptr) return fail(pcm::Error::transient("cosyvoice: Core ML could not load " + req.artifact_path));
    const std::string why = interface_error(stage, m);
    if (!why.empty()) {
        moss_coreml_free(m);
        return fail(pcm::Error::deterministic("cosyvoice: " + req.artifact_path + " " + why));
    }
    auto * sm = new stage_model;
    sm->model = m;
    r.model = std::shared_ptr<void>(sm, [](void * p) { delete static_cast<stage_model *>(p); });
    pcm::generation::end_load(req, r);
    return r;
}

// WarmFn: one prediction on silent inputs at the model's capacity, so a
// unit's first prediction does not pay the first-run cost.
pcm::Error warm_stage(void * model, const pcm::LoadRequest & req) {
    moss_coreml_model * m = static_cast<stage_model *>(model)->model;
    const auto stage = static_cast<cosyvoice_stage>(req.geometry.empty() ? 0 : req.geometry[0]);
    auto count = [](const std::vector<int64_t> & d) {
        size_t n = 1;
        for (int64_t x : d) n *= static_cast<size_t>(std::max<int64_t>(x, 1));
        return n;
    };
    std::vector<const char *> in_names, out_names;
    std::vector<std::vector<float>> in_bufs, out_bufs;
    auto in = [&](const char * name, float value) {
        in_names.push_back(name);
        in_bufs.emplace_back(count(dims_of(m, name)), value);
    };
    auto out = [&](const char * name) {
        out_names.push_back(name);
        out_bufs.emplace_back(count(dims_of(m, name)), 0.0f);
    };
    switch (stage) {
        case cosyvoice_stage::frontend:
            in("token_ids", 0.0f);
            in("token_mask", 1.0f);
            in("speaker", 1.0f);
            out("mu");
            out("spks");
            break;
        case cosyvoice_stage::f0:
            in("mel", 0.0f);
            out("f0");
            break;
        case cosyvoice_stage::hift:
            in("mel", 0.0f);
            in("source_stft", 0.0f);
            out("spec");
            break;
    }
    std::vector<const float *> in_data;
    std::vector<float *> out_data;
    for (auto & b : in_bufs) in_data.push_back(b.data());
    for (auto & b : out_bufs) out_data.push_back(b.data());
    if (moss_coreml_predict(m, in_names.data(), in_data.data(), (int) in_names.size(), out_names.data(),
                            out_data.data(), (int) out_names.size()) != 0) {
        return pcm::Error::transient(std::string("cosyvoice: the Core ML ") + cosyvoice_stage_name(stage) +
                                     " warm-up prediction failed");
    }
    return {};
}

uint64_t prebuilt_weight_bytes(const std::string & artifact) {
    struct stat st {};
    return ::stat((artifact + "/weights/weight.bin").c_str(), &st) == 0 ? (uint64_t) st.st_size : 0;
}

constexpr double ANE_COMPILE_RESERVE_FACTOR = 1.6;

struct planned_stage {
    cosyvoice_stage_route route;
    pcm::Plan plan;
    pcm::ComputeUnits units = pcm::ComputeUnits::cpu_and_ane;
    std::optional<pcm::generation::Candidate> candidate;
    std::string gguf;
    std::string prebuilt_path;
};

planned_stage plan_stage(cosyvoice_stage s, const std::string & gguf, const pcm::Resolved & r, bool ignore_prebuilt) {
    planned_stage out;
    out.gguf = gguf;
    out.prebuilt_path = tts_cpp::detail::coreml_sidecar_path(gguf, cosyvoice_stage_suffix(s));
    const std::shared_ptr<pcm::Manager> mgr = pcm::Manager::instance();
    pcm::PlanInput in;
    in.coreml_built = true;
    in.generation_built = pcm::generation::built();
    in.prebuilt_present = !ignore_prebuilt && tts_cpp::detail::coreml_sidecar_exists(out.prebuilt_path);
    if (in.generation_built && !in.prebuilt_present && r.enabled()) {
        std::vector<pcm::generation::Candidate> c =
            pcm::generation::probe({TEMPLATE_IDS[static_cast<int>(s)]}, gguf, r);
        if (!c.empty()) out.candidate = std::move(c.front());
    }
    const bool available = out.candidate && out.candidate->available;
    const pcm::ComputeUnits stage_default =
        available && out.candidate->default_units ? *out.candidate->default_units
                                                   : *pcm::parse_compute_units(DEFAULT_UNITS);
    out.units = r.units_or(stage_default);
    out.route.compute_units = pcm::to_string(out.units);
    in.template_available = available;
    in.template_matched = available && out.candidate->matched;
    if (available) in.validated_units = out.candidate->memory.validated_units;
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
        out.route.margin = m.margin > 0 ? m.margin : 1.25;
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

const std::string & stage_gguf(cosyvoice_stage s, const std::string & flow, const std::string & hift) {
    return s == cosyvoice_stage::frontend ? flow : hift;
}

cosyvoice_coreml_plan summarize(const planned_stage (&stages)[COSYVOICE_STAGE_COUNT], const pcm::Resolved & r) {
    cosyvoice_coreml_plan p;
    p.strict = r.strict();
    p.validated = true;
    uint64_t resident = 0, transient = 0;
    for (int i = 0; i < COSYVOICE_STAGE_COUNT; ++i) {
        const cosyvoice_stage_route & sr = stages[i].route;
        p.stages[i] = sr;
        if (sr.route == "prebuilt") p.any_prebuilt = true;
        if (sr.route != "generate") continue;
        p.any_generate = true;
        p.validated = p.validated && sr.validated;
        resident += (uint64_t) std::ceil((double) sr.resident_bytes * sr.margin);
        transient = std::max(transient, (uint64_t) std::ceil(
                                            (double) std::max(sr.prepare_peak_bytes, sr.load_peak_bytes) * sr.margin));
    }
    if (!p.any_generate) p.validated = false;
    p.envelope_bytes = p.any_generate ? resident + transient : 0;
    return p;
}

}  // namespace

cosyvoice_coreml_plan cosyvoice_plan_coreml(const std::string & flow_gguf, const std::string & hift_gguf,
                                            const tts_cpp::CoremlOptions & options, bool ignore_prebuilt) {
    const pcm::Resolved r = resolve(options);
    planned_stage stages[COSYVOICE_STAGE_COUNT];
    for (int i = 0; i < COSYVOICE_STAGE_COUNT; ++i) {
        const auto s = static_cast<cosyvoice_stage>(i);
        stages[i] = plan_stage(s, stage_gguf(s, flow_gguf, hift_gguf), r, ignore_prebuilt);
    }
    return summarize(stages, r);
}

bool cosyvoice_coreml_enabled(const tts_cpp::CoremlOptions & options) {
    return resolve(options).enabled();
}

struct cosyvoice_coreml_stages::impl {
    pcm::Resolved resolved;
    planned_stage stages[COSYVOICE_STAGE_COUNT];
    cosyvoice_coreml_plan plan;
    std::shared_ptr<pcm::Manager> manager;
    std::shared_ptr<pcm::Slot> slots[COSYVOICE_STAGE_COUNT];
    pcm::SlotStatus none_status[COSYVOICE_STAGE_COUNT];
    std::shared_ptr<pcm::Group> construction;
    std::mutex mu;  // guards `prewarms`
    std::vector<std::shared_ptr<pcm::Group>> prewarms;

    std::vector<pcm::GroupRequest> requests(pcm::Priority p) const {
        std::vector<pcm::GroupRequest> out;
        for (const auto & slot : slots) {
            if (slot) out.push_back({slot, p});
        }
        return out;
    }
    std::vector<pcm::SlotStatus> skipped() const {
        std::vector<pcm::SlotStatus> out;
        for (int i = 0; i < COSYVOICE_STAGE_COUNT; ++i) {
            if (!slots[i]) out.push_back(none_status[i]);
        }
        return out;
    }
};

std::shared_ptr<cosyvoice_coreml_stages> cosyvoice_coreml_stages::create(const std::string & flow_gguf,
                                                                         const std::string & hift_gguf,
                                                                         const tts_cpp::CoremlOptions & options,
                                                                         bool ignore_prebuilt, bool memory_degrade) {
    auto p = std::make_unique<impl>();
    p->resolved = resolve(options);
    for (const std::string & w : p->resolved.warnings) std::fprintf(stderr, "cosyvoice: %s\n", w.c_str());
    if (memory_degrade && p->resolved.enabled()) {
        p->resolved.use = pcm::Use::off;
        p->resolved.disabled_reason = "memory";
    }
    p->manager = pcm::Manager::instance();
    for (int i = 0; i < COSYVOICE_STAGE_COUNT; ++i) {
        const auto s = static_cast<cosyvoice_stage>(i);
        p->stages[i] = plan_stage(s, stage_gguf(s, flow_gguf, hift_gguf), p->resolved, ignore_prebuilt);
    }
    p->plan = summarize(p->stages, p->resolved);

    for (int i = 0; i < COSYVOICE_STAGE_COUNT; ++i) {
        const auto s = static_cast<cosyvoice_stage>(i);
        planned_stage & ps = p->stages[i];
        if (ps.plan.route == pcm::Route::none) {
            pcm::SlotStatus & st = p->none_status[i];
            st.stage = cosyvoice_stage_name(s);
            st.function = "main";
            st.state = ps.plan.none_state;
            st.reason = ps.plan.reason;
            st.compute_units = ps.route.compute_units;
            st.label = COSYVOICE_GGML_LABEL;
            if (p->resolved.strict()) {
                throw std::runtime_error(std::string("cosyvoice: no usable Core ML ") + cosyvoice_stage_name(s) +
                                         " sidecar (" + ps.route.state + ": " + ps.route.reason +
                                         "); Core ML strict policy is enabled (use = require / COSYVOICE_COREML_STRICT)");
            }
            continue;
        }
        pcm::SlotSpec spec;
        spec.engine = ENGINE_NAME;
        spec.stage = cosyvoice_stage_name(s);
        spec.function = "main";
        spec.units = ps.units;
        spec.route = ps.plan.route;
        spec.load_fn = &load_stage;
        spec.warm_fn = &warm_stage;
        spec.geometry = {static_cast<int64_t>(i)};
        spec.shareable = true;
        if (spec.route == pcm::Route::prebuilt) {
            spec.artifact_path = ps.prebuilt_path;
            spec.ane_compile_reserve_bytes =
                pcm::units_include_ane(spec.units)
                    ? (uint64_t) (ANE_COMPILE_RESERVE_FACTOR * (double) prebuilt_weight_bytes(spec.artifact_path))
                    : 0;
        } else {
            const pcm::generation::Candidate & c = *ps.candidate;
            // The engine reads the GGUF again for every synthesis (each stage's
            // weights load per call), so the identity is the file at the path
            // when the engine is made: prepare fails with source_changed when
            // the path names another file by then, and the stage stays on ggml.
            pcm::SourceStat source;
            if (!pcm::generation::source_stat(ps.gguf, source)) source.changed = true;
            spec.tmpl = c.tmpl;
            spec.prepare = pcm::generation::prepare_request(c, source, p->resolved);
            spec.prepare_fn = &pcm::generation::prepare;
            spec.ane_compile_reserve_bytes = pcm::generation::ane_compile_reserve_bytes(c.memory, spec.units);
        }
        p->slots[i] = p->manager->make_slot(std::move(spec), pcm::SlotPolicy::from(p->resolved));
    }

    // Construction group: every stage with a slot, in pipeline order (equal
    // priorities run first in, first out). prewarm none: nothing until a unit
    // demands it.
    std::vector<pcm::GroupRequest> requests;
    if (p->resolved.prewarm != pcm::Prewarm::none) requests = p->requests(pcm::Priority::minimal);
    const bool waited = p->resolved.wait == pcm::Wait::construct;
    p->construction = p->manager->issue_group(pcm::GroupKind::construction, requests, waited, p->skipped());
    auto out = std::shared_ptr<cosyvoice_coreml_stages>(new cosyvoice_coreml_stages(std::move(p)));
    if (waited) {
        const pcm::PrewarmResult result = out->p_->construction->wait();
        if (out->p_->resolved.strict() && !requests.empty() && !result.success) {
            for (const pcm::SlotStatus & st : result.slots) {
                if (st.state == pcm::SlotState::ready) continue;
                throw std::runtime_error("cosyvoice: the Core ML " + st.stage +
                                         " sidecar is not ready after construction (" +
                                         pcm::state_label(st.state, st.reason) + (st.error.empty() ? "" : ": " + st.error) +
                                         "); Core ML strict policy is enabled (use = require / COSYVOICE_COREML_STRICT)");
            }
        }
    }
    return out;
}

cosyvoice_coreml_stages::cosyvoice_coreml_stages(std::unique_ptr<impl> p) : p_(std::move(p)) {}
cosyvoice_coreml_stages::~cosyvoice_coreml_stages() = default;

cosyvoice_stage_call cosyvoice_coreml_stages::acquire(cosyvoice_stage s, int frames, const std::atomic<bool> * cancel) {
    cosyvoice_stage_call call;
    const int i = static_cast<int>(s);
    const std::shared_ptr<pcm::Slot> & slot = p_->slots[i];
    if (!slot) return call;
    const int64_t budget_ms = std::max<int64_t>(
        GGML_FLOOR_MS, (int64_t) (GGML_MS_PER_FRAME[i] * (double) std::max(frames, 0)));
    const auto deadline = pcm::Clock::now() + std::chrono::milliseconds(budget_ms);
    pcm::Acquired got;
    try {
        got = slot->acquire(pcm::Demand::now, deadline, cancel);
    } catch (const pcm::CoremlError & e) {
        throw std::runtime_error(std::string("cosyvoice: the Core ML ") + cosyvoice_stage_name(s) +
                                 " sidecar cannot serve this call (" + e.what() +
                                 "); Core ML strict policy is enabled (use = require / COSYVOICE_COREML_STRICT)");
    }
    call.pending = got.pending;
    if (!got.model) return call;
    auto handle = std::make_shared<pcm::Handle>(std::move(got.model));
    moss_coreml_model * m = (*handle)->as<stage_model>()->model;
    cosyvoice_coreml::binding b;
    b.lock = [handle] { return (*handle)->lock_for_prediction(); };
    b.retire = [handle](const std::string & reason) { (*handle)->retire(reason); };
    b.strict = p_->resolved.strict();
    const std::string label = (*handle)->label();
    call.model = cosyvoice_coreml::borrow(handle, m, label, std::move(b));
    return call;
}

bool cosyvoice_coreml_stages::strict() const { return p_->resolved.strict(); }
const cosyvoice_coreml_plan & cosyvoice_coreml_stages::plan() const { return p_->plan; }
bool cosyvoice_coreml_stages::active(cosyvoice_stage s) const { return p_->slots[static_cast<int>(s)] != nullptr; }

std::string cosyvoice_coreml_stages::status_json() const {
    pcm::GenerationStatus gen;
    gen.supported = pcm::generation::built() && p_->manager->platform() == pcm::Platform::macos &&
                    pcm::os_at_least(p_->manager->os_version(), 15, 0);
    if (!gen.supported) {
        gen.reason = !pcm::generation::built() ? "build" : "os";
    } else {
        for (const planned_stage & ps : p_->stages) {
            if (ps.plan.route == pcm::Route::none && ps.plan.reason == "unvalidated_units") gen.reason = "unvalidated_units";
        }
    }
    std::vector<std::shared_ptr<pcm::Slot>> slots;
    for (const auto & s : p_->slots) {
        if (s) slots.push_back(s);
    }
    pcm::EngineStatus st = p_->manager->status(ENGINE_NAME, p_->resolved.cache_dir, slots, gen);
    // Stages without a slot, in pipeline order with the others.
    std::vector<pcm::SlotStatus> ordered;
    size_t next = 0;
    for (int i = 0; i < COSYVOICE_STAGE_COUNT; ++i) {
        if (p_->slots[i]) {
            if (next < st.stages.size()) ordered.push_back(st.stages[next++]);
        } else {
            ordered.push_back(p_->none_status[i]);
        }
    }
    st.stages = std::move(ordered);
    return pcm::to_json(st);
}

std::string cosyvoice_coreml_stages::construction_json() const {
    return p_->construction ? pcm::to_json(p_->construction->result()) : std::string("{}");
}

namespace {
pcm::PrewarmResult wait_group(pcm::Group & g, const std::atomic<uint64_t> * cancels, uint64_t since) {
    while (!g.finished()) {
        if (cancels && cancels->load() != since) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return g.result();
}
}  // namespace

std::string cosyvoice_coreml_stages::prewarm_json(tts_cpp::CoremlOptions::Prewarm scope, bool wait,
                                                  const std::atomic<uint64_t> * cancels, uint64_t since) {
    std::vector<pcm::GroupRequest> requests;
    if (scope != tts_cpp::CoremlOptions::Prewarm::none) requests = p_->requests(pcm::Priority::minimal);
    std::shared_ptr<pcm::Group> g = p_->manager->issue_group(pcm::GroupKind::prewarm, requests, wait, p_->skipped());
    {
        std::lock_guard<std::mutex> lock(p_->mu);
        p_->prewarms.push_back(g);
    }
    return pcm::to_json(wait ? wait_group(*g, cancels, since) : g->result());
}

bool cosyvoice_coreml_stages::wait(int64_t timeout_ms, const std::atomic<uint64_t> * cancels, uint64_t since) const {
    bool any = false;
    for (const auto & slot : p_->slots) any = any || slot != nullptr;
    if (!any) return false;
    const auto start = pcm::Clock::now();
    for (;;) {
        bool all_ready = true;
        for (const auto & slot : p_->slots) {
            if (!slot) continue;
            const pcm::SlotState st = slot->state();
            if (st == pcm::SlotState::ready) continue;
            // Terminal, or nothing will move it without a request: idle
            // (prewarm none before the first call, or evicted) and
            // deferred(memory).
            if (pcm::is_terminal(st) || st == pcm::SlotState::idle || st == pcm::SlotState::deferred_memory) return false;
            all_ready = false;
        }
        if (all_ready) return true;
        if (cancels && cancels->load() != since) return false;
        if (timeout_ms >= 0 && pcm::Clock::now() - start >= std::chrono::milliseconds(timeout_ms)) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

#else  // !TTS_CPP_USE_COREML

cosyvoice_coreml_plan cosyvoice_plan_coreml(const std::string &, const std::string &,
                                            const tts_cpp::CoremlOptions & options, bool) {
    validate_options(options);
    cosyvoice_coreml_plan p;
    for (auto & s : p.stages) {
        s.state = "unsupported";
        s.reason = "build";
    }
    return p;
}

bool cosyvoice_coreml_enabled(const tts_cpp::CoremlOptions & options) {
    validate_options(options);
    return false;
}

struct cosyvoice_coreml_stages::impl {
    cosyvoice_coreml_plan plan;
};

std::shared_ptr<cosyvoice_coreml_stages> cosyvoice_coreml_stages::create(const std::string &, const std::string &,
                                                                         const tts_cpp::CoremlOptions & options, bool,
                                                                         bool) {
    validate_options(options);
    return nullptr;
}

cosyvoice_coreml_stages::cosyvoice_coreml_stages(std::unique_ptr<impl> p) : p_(std::move(p)) {}
cosyvoice_coreml_stages::~cosyvoice_coreml_stages() = default;
cosyvoice_stage_call cosyvoice_coreml_stages::acquire(cosyvoice_stage, int, const std::atomic<bool> *) { return {}; }
bool cosyvoice_coreml_stages::strict() const { return false; }
const cosyvoice_coreml_plan & cosyvoice_coreml_stages::plan() const { return p_->plan; }
bool cosyvoice_coreml_stages::active(cosyvoice_stage) const { return false; }
std::string cosyvoice_coreml_stages::status_json() const { return "{}"; }
std::string cosyvoice_coreml_stages::construction_json() const { return "{}"; }
std::string cosyvoice_coreml_stages::prewarm_json(tts_cpp::CoremlOptions::Prewarm, bool, const std::atomic<uint64_t> *,
                                                  uint64_t) {
    return "{}";
}
bool cosyvoice_coreml_stages::wait(int64_t, const std::atomic<uint64_t> *, uint64_t) const { return false; }

#endif
