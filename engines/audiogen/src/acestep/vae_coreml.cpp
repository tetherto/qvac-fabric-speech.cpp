#include "vae_coreml.h"

#include "vae_coreml_path.h"
#include "vae_coreml_windows.h"
#include "vae_encode_windows.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <stdexcept>

#ifdef AUDIOGEN_USE_COREML
#include "coreml/vae-decoder.h"

#include "qvac_coreml/generation.h"
#include "qvac_coreml/manager.h"
#include "qvac_coreml/options.h"
#include "qvac_coreml/slot.h"
#include "qvac_coreml/status.h"

#include <sys/stat.h>

#include <chrono>
#include <cmath>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>
#endif

namespace tts_cpp::acestep::detail {

int64_t vae_ggml_decode_estimate_ms(int T_latent) {
    const double ms = VAE_GGML_DECODE_MS_PER_FRAME * (double) std::max(T_latent, 0);
    return std::max<int64_t>(VAE_GGML_DECODE_MS_FLOOR, (int64_t) ms);
}

std::string vae_merge_backend(const std::string & acc, const std::string & next) {
    if (acc.empty() || acc == next) return next;
    return VAE_MIXED_BACKEND;
}

namespace {

bool valid_units(const std::string & u) {
    return u == "all" || u == "cpu_and_ane" || u == "cpu_and_gpu" || u == "cpu_only";
}

}  // namespace

namespace {
std::atomic<vae_coreml_completion_hook> g_completion_hook{nullptr};
}  // namespace

void vae_coreml_set_completion_hook(vae_coreml_completion_hook hook) { g_completion_hook.store(hook); }

void vae_coreml_validate_options(const CoremlOptions & o) {
    if (o.compute_units && !valid_units(*o.compute_units)) {
        throw std::invalid_argument("acestep: invalid coreml.compute_units='" + *o.compute_units +
                                    "' (expected all, cpu_and_ane, cpu_and_gpu or cpu_only)");
    }
    if (o.strict_wait_ms && *o.strict_wait_ms < 0) {
        throw std::invalid_argument("acestep: coreml.strict_wait_ms must not be negative");
    }
    if (o.memory_budget_bytes && *o.memory_budget_bytes < 0) {
        throw std::invalid_argument("acestep: coreml.memory_budget_bytes must not be negative");
    }
}

#ifdef AUDIOGEN_USE_COREML

namespace rt = QVAC_COREML_NS;

namespace {

constexpr const char * ENGINE_NAME = "acestep";
constexpr const char * STAGE_NAME  = "vae";
constexpr double ANE_COMPILE_RESERVE_FACTOR = 1.6;
constexpr int    GEOMETRY_SIZE = 3;  // {latent channels, upsample, pcm channels}

template <class E, class F> F map_enum(E v) { return static_cast<F>(static_cast<int>(v)); }

rt::Options to_runtime_options(const CoremlOptions & o) {
    static_assert(static_cast<int>(CoremlOptions::Use::require) == static_cast<int>(rt::Use::require), "Use");
    static_assert(static_cast<int>(CoremlOptions::Generate::background) == static_cast<int>(rt::Generate::background),
                  "Generate");
    static_assert(static_cast<int>(CoremlOptions::Wait::call) == static_cast<int>(rt::Wait::call), "Wait");
    static_assert(static_cast<int>(CoremlOptions::Prewarm::all) == static_cast<int>(rt::Prewarm::all), "Prewarm");
    rt::Options r;
    r.use                 = map_enum<CoremlOptions::Use, rt::Use>(o.use);
    r.generate            = map_enum<CoremlOptions::Generate, rt::Generate>(o.generate);
    r.wait                = map_enum<CoremlOptions::Wait, rt::Wait>(o.wait);
    r.prewarm             = map_enum<CoremlOptions::Prewarm, rt::Prewarm>(o.prewarm);
    r.cache_dir           = o.cache_dir;
    r.compute_units       = o.compute_units;
    r.strict_wait_ms      = o.strict_wait_ms;
    r.memory_budget_bytes = o.memory_budget_bytes;
    return r;
}

bool dir_exists(const std::string & path) {
    struct stat st {};
    return ::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

uint64_t file_bytes(const std::string & path) {
    struct stat st {};
    return ::stat(path.c_str(), &st) == 0 ? (uint64_t) st.st_size : 0;
}

uint64_t margined(uint64_t bytes, double margin) {
    return (uint64_t) std::ceil((double) bytes * (margin > 0 ? margin : 1.25));
}

// The loaded model a slot holds.
struct vae_model {
    acestep_coreml_vae_context * ctx = nullptr;
    int window = 0;
    ~vae_model() {
        if (ctx) acestep_coreml_vae_free(ctx);
    }
};

// LoadFn. Takes the shared lease of a generated artifact first, then loads the
// model on the request's compute units and validates the geometry the
// constructing thread passed by value: {latent channels, upsample, pcm
// channels}, plus a window long enough for the overlap of the stitched decode.
rt::LoadResult load_vae(const rt::LoadRequest & req) {
    rt::LoadResult r;
    if (!rt::generation::begin_load(req, r)) return r;
    auto fail = [&](const rt::Error & e) {
        r.model.reset();
        r.error = e;
        rt::generation::end_load(req, r);
        return r;
    };
    if ((int) req.geometry.size() != GEOMETRY_SIZE || req.geometry[0] != VAE_LATENT_CHANNELS ||
        req.geometry[1] != VAE_ENCODER_UPSAMPLE || req.geometry[2] != VAE_PCM_CHANNELS) {
        return fail(rt::Error::deterministic("acestep: VAE decoder load without its geometry"));
    }
    acestep_coreml_vae_context * ctx = acestep_coreml_vae_init(req.artifact_path.c_str(), rt::to_string(req.units));
    if (ctx == nullptr) {
        // A model that loads but declares other shapes is rejected inside
        // init as well; a reload cannot fix either, but a load failure can be
        // transient (e5rt cache, memory), so both stay transient here.
        return fail(rt::Error::transient("acestep: Core ML could not load " + req.artifact_path +
                                         " as a VAE decoder (latent [1, 64, W] -> pcm [1, 2, W * 1920])"));
    }
    const int64_t window = acestep_coreml_vae_window_frames(ctx);
    if (window <= 2 * VAE_COREML_OVERLAP) {
        acestep_coreml_vae_free(ctx);
        return fail(rt::Error::deterministic("acestep: " + req.artifact_path + ": window " + std::to_string(window) +
                                             " leaves no frames after " + std::to_string(VAE_COREML_OVERLAP) +
                                             " overlap frames per side"));
    }
    auto * m   = new vae_model;
    m->ctx     = ctx;
    m->window  = (int) window;
    r.model    = std::shared_ptr<void>(m, [](void * p) { delete static_cast<vae_model *>(p); });
    rt::generation::end_load(req, r);
    return r;
}

// WarmFn: one prediction on a silent window, so a decode's first window does
// not pay the first-run cost.
rt::Error warm_vae(void * model, const rt::LoadRequest &) {
    auto & m = *static_cast<vae_model *>(model);
    std::vector<float> latent((size_t) m.window * VAE_LATENT_CHANNELS, 0.0f);
    std::vector<float> pcm((size_t) m.window * VAE_ENCODER_UPSAMPLE * VAE_PCM_CHANNELS);
    if (acestep_coreml_vae_decode(m.ctx, latent.data(), pcm.data()) != 0) {
        return rt::Error::transient("acestep: the Core ML VAE warm-up prediction failed");
    }
    if (const vae_coreml_completion_hook hook = g_completion_hook.load()) hook();
    return {};
}

// One decode's decoder: the handle acquired at the decode's start.
class handle_decoder final : public vae_coreml_decoder {
public:
    explicit handle_decoder(rt::Handle h) : h_(std::move(h)), m_(h_->as<vae_model>()) {}
    int          window() const override { return m_->window; }
    const char * label() const override { return h_->label(); }
    bool decode(const float * latent, float * pcm) override {
        auto lock = h_->lock_for_prediction();
        return acestep_coreml_vae_decode(m_->ctx, latent, pcm) == 0;
    }
    void retire(const std::string & reason) { h_->retire(reason); }

private:
    rt::Handle  h_;
    vae_model * m_;
};

struct planned {
    vae_coreml_route route;
    rt::Plan plan;
    rt::ComputeUnits units = rt::ComputeUnits::all;
    std::optional<rt::generation::Candidate> candidate;
    std::string prebuilt_path;
};

planned plan_route(const std::string & gguf_path, const rt::Resolved & r) {
    planned out;
    out.route.strict = r.strict();
    out.prebuilt_path = coreml_vae_sidecar_path(gguf_path);
    const std::shared_ptr<rt::Manager> mgr = rt::Manager::instance();
    rt::PlanInput in;
    in.coreml_built     = true;
    in.generation_built = rt::generation::built();
    in.prebuilt_present = dir_exists(out.prebuilt_path);
    if (in.generation_built && !in.prebuilt_present && r.enabled()) {
        std::vector<rt::generation::Candidate> c = rt::generation::probe({VAE_COREML_TEMPLATE_ID}, gguf_path, r);
        if (!c.empty()) out.candidate = std::move(c.front());
    }
    const bool available = out.candidate && out.candidate->available;
    out.units = r.units_or(available && out.candidate->default_units ? *out.candidate->default_units
                                                                     : rt::ComputeUnits::all);
    out.route.compute_units = rt::to_string(out.units);
    in.template_available   = available;
    in.template_matched     = available && out.candidate->matched;
    if (available) in.validated_units = out.candidate->memory.validated_units;
    in.units    = out.units;
    in.platform = mgr->platform();
    in.os       = mgr->os_version();
    out.plan    = rt::plan(in, r);
    out.route.route = rt::to_string(out.plan.route);
    if (out.plan.route == rt::Route::none) {
        out.route.state  = rt::to_string(out.plan.none_state);
        out.route.reason = out.plan.reason;
        if (out.plan.none_state == rt::SlotState::no_template && out.candidate && !out.candidate->reason.empty()) {
            out.route.reason += ": " + out.candidate->reason;
        }
    }
    if (out.plan.route == rt::Route::generate) {
        const rt::generation::TemplateMemory & m = out.candidate->memory;
        out.route.resident_bytes     = margined(rt::generation::resident_bytes(m), m.margin);
        out.route.transient_bytes    = margined(rt::generation::transient_bytes(m, out.units), m.margin);
        out.route.envelope_bytes     = rt::generation::envelope_bytes(m, out.units);
        out.route.prepare_peak_bytes = rt::generation::prepare_peak_bytes(m);
        out.route.load_peak_bytes    = rt::generation::load_peak_bytes(m, out.units);
        out.route.workspace_bytes    = m.workspace_bytes;
        out.route.ane_compile_reserve_bytes = rt::generation::ane_compile_reserve_bytes(m, out.units);
        out.route.validated = std::find(m.validated_units.begin(), m.validated_units.end(), out.units) !=
                              m.validated_units.end();
    }
    return out;
}

[[noreturn]] void throw_strict(const std::string & what) {
    throw std::runtime_error("acestep: " + what +
                             "; Core ML strict policy is enabled (use = require / ACESTEP_COREML_STRICT)");
}

}  // namespace

struct vae_coreml_stage::impl {
    rt::Resolved resolved;
    planned plan;
    std::shared_ptr<rt::Manager> manager;
    std::shared_ptr<rt::Slot> slot;
    rt::SlotStatus none_status;
    std::shared_ptr<rt::Group> construction;
    const std::atomic<bool> * cancel = nullptr;
    // Decode demand: bumped by every acquire() under demand_mu, so the
    // release of an asynchronous construction load never evicts a decoder a
    // decode has asked for since the group was issued.
    std::mutex demand_mu;
    uint64_t demand = 0;
    // Low-memory mode, prewarm all without wait construct: waits for the
    // construction group off the constructing thread and releases the model
    // once the group recorded its result.
    std::thread releaser;
    std::atomic<bool> stop{false};

    ~impl() {
        stop.store(true);
        if (releaser.joinable()) releaser.join();
    }
};

vae_coreml_route vae_coreml_plan_route(const std::string & gguf_path, const CoremlOptions & options) {
    vae_coreml_validate_options(options);
    const rt::Resolved r = rt::resolve_coreml(to_runtime_options(options), VAE_COREML_ENV_PREFIX);
    return plan_route(gguf_path, r).route;
}

std::shared_ptr<vae_coreml_stage> vae_coreml_stage::create(const std::string & gguf_path,
                                                           const CoremlOptions & options, bool memory_degrade,
                                                           bool release_after_construction,
                                                           const std::atomic<bool> * cancel) {
    vae_coreml_validate_options(options);
    auto p      = std::make_unique<impl>();
    p->cancel   = cancel;
    p->resolved = rt::resolve_coreml(to_runtime_options(options), VAE_COREML_ENV_PREFIX);
    for (const std::string & w : p->resolved.warnings) fprintf(stderr, "[acestep-vae] %s\n", w.c_str());
    if (memory_degrade && p->resolved.enabled()) {
        p->resolved.use             = rt::Use::off;
        p->resolved.disabled_reason = "memory";
    }
    p->manager = rt::Manager::instance();
    p->plan    = plan_route(gguf_path, p->resolved);

    if (p->plan.plan.route == rt::Route::none) {
        p->none_status.stage         = STAGE_NAME;
        p->none_status.function      = "main";
        p->none_status.state         = p->plan.plan.none_state;
        p->none_status.reason        = p->plan.plan.reason;
        p->none_status.compute_units = p->plan.route.compute_units;
        p->none_status.label         = VAE_GGML_BACKEND;
        if (p->resolved.strict()) {
            throw_strict("no usable Core ML VAE decoder (" + p->plan.route.state + ": " + p->plan.route.reason + ")");
        }
        return std::shared_ptr<vae_coreml_stage>(new vae_coreml_stage(std::move(p)));
    }

    rt::SlotSpec spec;
    spec.engine   = ENGINE_NAME;
    spec.stage    = STAGE_NAME;
    spec.function = "main";
    spec.units    = p->plan.units;
    spec.route    = p->plan.plan.route;
    spec.load_fn  = &load_vae;
    spec.warm_fn  = &warm_vae;
    spec.geometry = {VAE_LATENT_CHANNELS, VAE_ENCODER_UPSAMPLE, VAE_PCM_CHANNELS};
    spec.shareable = true;
    if (spec.route == rt::Route::prebuilt) {
        spec.artifact_path = p->plan.prebuilt_path;
        spec.ane_compile_reserve_bytes =
            rt::units_include_ane(spec.units)
                ? (uint64_t) (ANE_COMPILE_RESERVE_FACTOR *
                              (double) file_bytes(spec.artifact_path + "/weights/weight.bin"))
                : 0;
    } else {
        const rt::generation::Candidate & c = *p->plan.candidate;
        rt::SourceStat source;
        if (!rt::generation::source_stat(gguf_path, source)) {
            throw std::runtime_error("acestep: cannot stat " + gguf_path);
        }
        spec.tmpl    = c.tmpl;
        spec.prepare = rt::generation::prepare_request(c, source, p->resolved);
        spec.prepare_fn = &rt::generation::prepare;
        // A decode adopts an artifact a previous process left in the cache
        // (and its warm hint) before deciding the bounded wait.
        spec.lookup_fn = &rt::generation::lookup;
        spec.ane_compile_reserve_bytes = rt::generation::ane_compile_reserve_bytes(c.memory, spec.units);
        if (auto w = c.expected_warm_load_ms.find("main"); w != c.expected_warm_load_ms.end()) {
            spec.expected_warm_load_ms = w->second;
        }
    }
    p->slot = p->manager->make_slot(std::move(spec), rt::SlotPolicy::from(p->resolved));

    // ACE-Step's minimal set is empty: the decoder is requested by decode
    // demand. Only prewarm all loads it at construction.
    std::vector<rt::GroupRequest> requests;
    if (p->resolved.prewarm == rt::Prewarm::all) requests.push_back({p->slot, rt::Priority::prewarm});
    const bool waited = p->resolved.wait == rt::Wait::construct;
    p->construction   = p->manager->issue_group(rt::GroupKind::construction, requests, waited);
    auto out = std::shared_ptr<vae_coreml_stage>(new vae_coreml_stage(std::move(p)));
    if (!waited && !requests.empty() && release_after_construction) {
        // Asynchronous install-time preparation in low-memory mode: once the
        // group has recorded the decoder's ready (or terminal) state, the
        // model is released, unless a decode has asked for it meanwhile (the
        // decode's VAE stage boundary releases it then).
        impl * ip = out->p_.get();
        uint64_t issued_demand = 0;
        {
            std::lock_guard<std::mutex> lk(ip->demand_mu);
            issued_demand = ip->demand;
        }
        ip->releaser = std::thread([ip, issued_demand] {
            ip->construction->wait(std::nullopt, &ip->stop);
            if (ip->stop.load()) return;
            std::lock_guard<std::mutex> lk(ip->demand_mu);
            if (ip->demand == issued_demand) ip->slot->evict();
        });
    }
    if (waited && !requests.empty()) {
        const rt::PrewarmResult result = out->p_->construction->wait(std::nullopt, cancel);
        if (out->p_->resolved.strict() && !result.success) {
            const rt::SlotStatus st = out->p_->slot->status();
            throw_strict("the Core ML VAE decoder is not ready after construction (" +
                         rt::state_label(st.state, st.reason) + (st.error.empty() ? "" : ": " + st.error) + ")");
        }
        // Install-time preparation in low-memory mode: the load warmed the
        // Neural Engine plan and wrote its hint; nothing stays resident.
        if (release_after_construction) out->p_->slot->evict();
    }
    return out;
}

vae_coreml_stage::vae_coreml_stage(std::unique_ptr<impl> p) : p_(std::move(p)) {}
vae_coreml_stage::~vae_coreml_stage() = default;

vae_coreml_call vae_coreml_stage::acquire(int T_latent) {
    vae_coreml_call call;
    if (!p_->slot) return call;
    {
        std::lock_guard<std::mutex> lk(p_->demand_mu);
        ++p_->demand;
    }
    const auto deadline = rt::Clock::now() + std::chrono::milliseconds(vae_ggml_decode_estimate_ms(T_latent));
    rt::Acquired got;
    try {
        got = p_->slot->acquire(rt::Demand::now, deadline, p_->cancel);
    } catch (const rt::CoremlError & e) {
        throw_strict(std::string("the Core ML VAE decoder cannot serve this decode (") + e.what() + ")");
    }
    call.pending = got.pending;
    if (!got.model) {
        if (p_->cancel && p_->cancel->load()) {
            call.cancelled = true;
            call.pending   = false;
            return call;
        }
        // A strict acquisition the runtime returned unserved without a
        // CoremlError (evicted meanwhile, shutting down) still cannot run on
        // ggml.
        if (p_->resolved.strict()) {
            throw_strict("the Core ML VAE decoder cannot serve this decode (" +
                         rt::state_label(got.state, std::string()) + ")");
        }
        return call;
    }
    auto decoder = std::make_shared<handle_decoder>(std::move(got.model));
    call.retire  = [decoder](const std::string & reason) { decoder->retire(reason); };
    call.decoder = std::move(decoder);
    return call;
}

void vae_coreml_stage::evict() {
    if (p_->slot) p_->slot->evict();
}

bool vae_coreml_stage::strict() const { return p_->resolved.strict(); }
bool vae_coreml_stage::active() const { return p_->slot != nullptr; }
const vae_coreml_route & vae_coreml_stage::route() const { return p_->plan.route; }

std::string vae_coreml_stage::status_json() const {
    rt::GenerationStatus gen;
    gen.supported = rt::generation::built() && p_->manager->platform() == rt::Platform::macos &&
                    rt::os_at_least(p_->manager->os_version(), 15, 0);
    if (!gen.supported) {
        gen.reason = !rt::generation::built() ? "build" : "os";
    } else if (p_->plan.plan.route == rt::Route::none && p_->plan.plan.reason == "unvalidated_units") {
        gen.reason = "unvalidated_units";
    }
    if (p_->slot) return rt::to_json(p_->manager->status(ENGINE_NAME, p_->resolved.cache_dir, {p_->slot}, gen));
    rt::EngineStatus st;
    st.engine     = ENGINE_NAME;
    st.cache_dir  = p_->resolved.cache_dir;
    st.generation = gen;
    st.stages.push_back(p_->none_status);
    return rt::to_json(st);
}

std::string vae_coreml_stage::construction_json() const {
    return p_->construction ? rt::to_json(p_->construction->result()) : std::string("{}");
}

bool vae_coreml_stage::wait(int64_t timeout_ms, const std::atomic<bool> * cancel) const {
    if (!p_->slot) return false;
    const auto start = rt::Clock::now();
    for (;;) {
        const rt::SlotState st = p_->slot->state();
        if (st == rt::SlotState::ready) return true;
        // Terminal, or nothing will move it without a request: idle (before
        // the first decode, or evicted) and deferred(memory).
        if (rt::is_terminal(st) || st == rt::SlotState::idle || st == rt::SlotState::deferred_memory) return false;
        if (cancel && cancel->load()) return false;
        if (timeout_ms >= 0 && rt::Clock::now() - start >= std::chrono::milliseconds(timeout_ms)) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

#else  // !AUDIOGEN_USE_COREML

struct vae_coreml_stage::impl {
    vae_coreml_route route;
};

vae_coreml_route vae_coreml_plan_route(const std::string &, const CoremlOptions & options) {
    vae_coreml_validate_options(options);
    vae_coreml_route r;
    r.state  = "unsupported";
    r.reason = "build";
    return r;
}

std::shared_ptr<vae_coreml_stage> vae_coreml_stage::create(const std::string &, const CoremlOptions & options, bool,
                                                           bool, const std::atomic<bool> *) {
    vae_coreml_validate_options(options);
    return nullptr;
}

vae_coreml_stage::vae_coreml_stage(std::unique_ptr<impl> p) : p_(std::move(p)) {}
vae_coreml_stage::~vae_coreml_stage() = default;
vae_coreml_call vae_coreml_stage::acquire(int) { return {}; }
void vae_coreml_stage::evict() {}
bool vae_coreml_stage::strict() const { return false; }
bool vae_coreml_stage::active() const { return false; }
const vae_coreml_route & vae_coreml_stage::route() const { return p_->route; }
std::string vae_coreml_stage::status_json() const { return "{}"; }
std::string vae_coreml_stage::construction_json() const { return "{}"; }
bool vae_coreml_stage::wait(int64_t, const std::atomic<bool> *) const { return false; }

#endif

}  // namespace tts_cpp::acestep::detail
