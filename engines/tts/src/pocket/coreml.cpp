#include "pocket/coreml.h"

#include "coreml_sidecar_path.h"

#ifdef TTS_CPP_USE_COREML
#include "moss/coreml/runner.h"

#include "qvac_coreml/generation.h"
#include "qvac_coreml/manager.h"
#include "qvac_coreml/options.h"
#include "qvac_coreml/slot.h"
#include "qvac_coreml/status.h"

#include <sys/stat.h>
#ifdef QVAC_COREML_GENERATE
#include "../../coreml-provided/pocket.h"
#endif
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace tts_cpp::pocket::detail {
namespace {

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("Pocket TTS Core ML step: " + message);
}

#ifdef TTS_CPP_USE_COREML
constexpr int MAX_RANK = 8;
constexpr int METADATA_BYTES = 256;

std::string dims_text(const std::vector<int64_t> & dims) {
    std::ostringstream text;
    text << '[';
    for (size_t i = 0; i < dims.size(); ++i) text << (i ? ", " : "") << dims[i];
    text << ']';
    return text.str();
}

std::vector<int64_t> feature_dims(moss_coreml_model * model, const char * name, bool state) {
    int64_t dims[MAX_RANK] = {};
    const int rank = state ? moss_coreml_state_dims(model, name, dims, MAX_RANK)
                           : moss_coreml_feature_dims(model, name, dims, MAX_RANK);
    return std::vector<int64_t>(dims, dims + std::max(rank, 0));
}

std::string metadata(moss_coreml_model * model, const char * key) {
    char value[METADATA_BYTES] = {};
    return moss_coreml_metadata(model, key, value, METADATA_BYTES) < 0 ? std::string() : std::string(value);
}

void require_finite(const std::vector<float> & values, const char * name) {
    if (!std::all_of(values.begin(), values.end(), [](float v) { return std::isfinite(v); }))
        fail(std::string("non-finite ") + name);
}

struct expected_feature {
    std::string name;
    bool state;
    std::vector<int64_t> dims;
};

void require_features(moss_coreml_model * model, const std::string & path, const std::vector<expected_feature> & expected) {
    for (const auto & feature : expected) {
        const auto declared = feature_dims(model, feature.name.c_str(), feature.state);
        if (declared != feature.dims)
            fail(path + " declares " + feature.name + " " + dims_text(declared) + " where the runtime needs " +
                 dims_text(feature.dims));
    }
}

void require_stamp(moss_coreml_model * model, const std::string & path, const std::string & source_hash) {
    if (metadata(model, "pocket.coreml.schema") != COREML_SCHEMA)
        fail(path + " has an unsupported schema; re-export it with scripts/export-pocket-coreml.py");
    if (metadata(model, "pocket.source_sha256") != source_hash)
        fail(path + " was exported from another checkpoint than its GGUF");
}

using Geometry = CoremlDecoderGeometry;

// (name, channels, length) of each Mimi causal-conv history, as
// histories() in scripts/pocket_coreml_mimi.py orders them.
struct history_spec {
    const char * name;
    const char * output;
    int channels, length;
};
constexpr std::array<history_spec, 9> HISTORIES = {{
    {"h_upsample", "new_h_upsample", 512, 1}, {"h_conv0", "new_h_conv0", 512, 6},
    {"h_convtr0", "new_h_convtr0", 512, 1}, {"h_residual0", "new_h_residual0", 256, 2},
    {"h_convtr1", "new_h_convtr1", 256, 1}, {"h_residual1", "new_h_residual1", 128, 2},
    {"h_convtr2", "new_h_convtr2", 128, 1}, {"h_residual2", "new_h_residual2", 64, 2},
    {"h_final", "new_h_final", 64, 2},
}};
constexpr int DECODER_ROWS = Geometry::frames * Geometry::upsample;
constexpr size_t KV_SIZE = size_t(Geometry::layers) * Geometry::heads * Geometry::keep * Geometry::head_dim;
#endif

} // namespace

bool coreml_valid_units(const std::string & units) {
    return units == "all" || units == "cpu_and_ane" || units == "cpu_and_gpu" || units == "cpu_only";
}

std::string coreml_units(const CoremlPolicy & policy, const char * sidecar_default) {
    return policy.compute_units.empty() ? sidecar_default : policy.compute_units;
}

CoremlPolicy resolve_coreml_policy(const std::string & compute_units, std::optional<bool> strict) {
    CoremlPolicy policy;
    policy.disabled = std::getenv(COREML_DISABLE_ENV) != nullptr;
    policy.strict = strict.has_value() ? *strict : std::getenv(COREML_STRICT_ENV) != nullptr;
    if (!compute_units.empty()) {
        if (!coreml_valid_units(compute_units))
            throw std::runtime_error("Pocket TTS: invalid coreml_compute_units='" + compute_units +
                                     "' (expected all, cpu_and_ane, cpu_and_gpu or cpu_only)");
        policy.compute_units = compute_units;
    } else if (const char * env = std::getenv(COREML_UNITS_ENV); env && *env) {
        if (coreml_valid_units(env)) policy.compute_units = env;
        else std::fprintf(stderr, "Pocket TTS: ignoring %s=%s (expected all, cpu_and_ane, cpu_and_gpu or cpu_only)\n",
                          COREML_UNITS_ENV, env);
    }
    return policy;
}

std::string coreml_step_path(const std::string & flow_lm_gguf) {
    return ::tts_cpp::detail::coreml_sidecar_path(flow_lm_gguf, COREML_STEP_SUFFIX);
}

std::string coreml_decoder_path(const std::string & mimi_gguf) {
    return ::tts_cpp::detail::coreml_sidecar_path(mimi_gguf, COREML_DECODER_SUFFIX);
}

namespace {
bool sidecar_enabled(const std::string & gguf, const std::string & sidecar, const CoremlPolicy & policy) {
#ifdef TTS_CPP_USE_COREML
    return !policy.disabled && !gguf.empty() && ::tts_cpp::detail::coreml_sidecar_exists(sidecar);
#else
    (void) gguf;
    (void) sidecar;
    (void) policy;
    return false;
#endif
}
} // namespace

bool coreml_step_enabled(const std::string & flow_lm_gguf, const CoremlPolicy & policy) {
    return sidecar_enabled(flow_lm_gguf, coreml_step_path(flow_lm_gguf), policy);
}

bool coreml_decoder_enabled(const std::string & mimi_gguf, const CoremlPolicy & policy) {
    return sidecar_enabled(mimi_gguf, coreml_decoder_path(mimi_gguf), policy);
}

void coreml_rope(int position, int head_dim, float base, std::vector<float> & cos, std::vector<float> & sin) {
    cos.resize(size_t(head_dim / 2));
    sin.resize(cos.size());
    for (int i = 0; i < head_dim / 2; ++i) {
        const double theta = double(position) * std::pow(double(base), -2.0 * i / head_dim);
        cos[size_t(i)] = float(std::cos(theta));
        sin[size_t(i)] = float(std::sin(theta));
    }
}

#ifdef TTS_CPP_USE_COREML
CoremlStep::CoremlStep(const std::string & path, const std::string & compute_units, const FlowConfig & config,
                       const std::string & source_hash, bool warm) : config_(config) {
    model_ = moss_coreml_load(path.c_str(), compute_units.c_str());
    if (model_ == nullptr) {
        fail("cannot load " + path + " on " + compute_units +
             (compute_units == "cpu_and_ane" ? " (its KV cache is Core ML state, which the Neural Engine cannot hold; "
                                               "use cpu_and_gpu, all or cpu_only)" : ""));
    }
    try {
        if (!moss_coreml_has_state(model_)) fail(path + " declares no KV state; re-export it");
        require_stamp(model_, path, source_hash);
        const int64_t heads = config.heads, head_dim = config.dim / config.heads;
        const std::vector<int64_t> mask = feature_dims(model_, "mask", false);
        context_ = mask.size() == 4 ? int(mask[3]) : 0;
        if (context_ < 2) fail(path + " declares mask " + dims_text(mask) + "; expected [1, 1, 1, context]");
        const int64_t c = context_, latent = config.latent_dim;
        require_features(model_, path, {
            {"latent", false, {1, latent}}, {"noise", false, {1, latent}},
            {"cos", false, {1, head_dim / 2}}, {"sin", false, {1, head_dim / 2}},
            {"mask", false, {1, 1, 1, c}}, {"write_mask", false, {1, 1, c, 1}},
            {"hidden", false, {1, config.dim}}, {"next_latent", false, {1, latent}},
            {"keys", true, {config.layers, heads, c, head_dim}}, {"values", true, {config.layers, heads, c, head_dim}},
        });
    } catch (...) {
        moss_coreml_free(model_);
        model_ = nullptr;
        throw;
    }
    label_ = moss_coreml_label(model_);
    mask_.assign(size_t(context_), COREML_MASKED);
    write_mask_.assign(size_t(context_), 0.0f);
    if (!warm) return;
    try {
        warm_up();
    } catch (...) {
        moss_coreml_free(model_);
        model_ = nullptr;
        throw;
    }
}

// One prediction at position 0 (the caller then writes the voice rows over
// it): a sidecar declaring inputs or outputs beyond the checked ones fails
// here rather than mid-synthesis, and the first frame skips Core ML's
// first-prediction setup.
void CoremlStep::warm_up() {
    std::vector<float> zeros(size_t(config_.latent_dim)), hidden, next;
    step(0, zeros, zeros, hidden, next);
}

CoremlStep::~CoremlStep() {
    if (model_ != nullptr) moss_coreml_free(model_);
}

void CoremlStep::write_rows(const CacheRows & rows) {
    const int heads = config_.heads, head_dim = config_.dim / config_.heads, n = rows.count;
    if (rows.start < 0 || n < 1 || rows.start + n > context_ || int(rows.keys.size()) != config_.layers ||
        rows.values.size() != rows.keys.size())
        fail("cache rows outside the sidecar's context");
    // ggml rows are [position][head][head_dim]; the state is [layer][head][position][head_dim].
    staging_.resize(size_t(n) * config_.dim);
    for (int layer = 0; layer < config_.layers; ++layer) {
        for (int kv = 0; kv < 2; ++kv) {
            const auto & source = kv ? rows.values[size_t(layer)] : rows.keys[size_t(layer)];
            if (source.size() != staging_.size()) fail("cache rows of the wrong width");
            for (int p = 0; p < n; ++p)
                for (int h = 0; h < heads; ++h)
                    std::copy_n(source.data() + (size_t(p) * heads + h) * head_dim, head_dim,
                                staging_.data() + (size_t(h) * n + p) * head_dim);
            const int64_t start[4] = {layer, 0, rows.start, 0};
            const int64_t count[4] = {1, heads, n, head_dim};
            if (moss_coreml_write_state(model_, kv ? "values" : "keys", staging_.data(), start, count, 4) != 0)
                fail("cannot write the KV state");
        }
    }
}

void CoremlStep::reset(const CacheRows & rows) {
    if (moss_coreml_reset_state(model_) != 0) fail("cannot replace the KV state");
    write_rows(rows);
}

void CoremlStep::step(int position, const std::vector<float> & latent, const std::vector<float> & noise,
                      std::vector<float> & hidden, std::vector<float> & next_latent) {
    if (position < 0 || position >= context_) fail("position outside the sidecar's context");
    if (latent.size() != size_t(config_.latent_dim) || noise.size() != latent.size())
        fail("invalid latent or noise dimensions");
    coreml_rope(position, config_.dim / config_.heads, config_.rope_base, cos_, sin_);
    // Attend to positions 0..position and write this frame's row at position.
    if (masked_from_ != position + 1) {
        std::fill(mask_.begin(), mask_.end(), COREML_MASKED);
        std::fill_n(mask_.begin(), position + 1, 0.0f);
        masked_from_ = position + 1;
    }
    if (written_ != position) {
        if (written_ >= 0) write_mask_[size_t(written_)] = 0.0f;
        write_mask_[size_t(position)] = 1.0f;
        written_ = position;
    }
    hidden.resize(size_t(config_.dim));
    next_latent.resize(size_t(config_.latent_dim));
    const char * in_names[] = {"latent", "noise", "cos", "sin", "mask", "write_mask"};
    const float * in_data[] = {latent.data(), noise.data(), cos_.data(), sin_.data(), mask_.data(), write_mask_.data()};
    const char * out_names[] = {"hidden", "next_latent"};
    float * out_data[] = {hidden.data(), next_latent.data()};
    if (moss_coreml_predict(model_, in_names, in_data, 6, out_names, out_data, 2) != 0) fail("prediction failed");
    require_finite(hidden, "hidden output");
    require_finite(next_latent, "latent output");
}
moss_coreml_model * load_decoder_model(const std::string & path, const std::string & compute_units,
                                       const std::string & source_hash) {
    moss_coreml_model * model = moss_coreml_load(path.c_str(), compute_units.c_str());
    if (model == nullptr) fail("cannot load " + path + " on " + compute_units);
    try {
        if (moss_coreml_has_state(model)) fail(path + " declares Core ML state; re-export it");
        require_stamp(model, path, source_hash);
        const std::vector<int64_t> kv = {Geometry::layers, Geometry::heads, Geometry::keep, Geometry::head_dim};
        std::vector<expected_feature> expected = {
            {"latents", false, {Geometry::frames, Geometry::latent}},
            {"cos", false, {DECODER_ROWS, Geometry::head_dim / 2}}, {"sin", false, {DECODER_ROWS, Geometry::head_dim / 2}},
            {"mask", false, {DECODER_ROWS, Geometry::keep + DECODER_ROWS}},
            {"keys", false, kv}, {"values", false, kv}, {"new_keys", false, kv}, {"new_values", false, kv},
            {"pcm", false, {int64_t(Geometry::frames) * Geometry::frame_samples}},
        };
        for (const auto & h : HISTORIES) {
            expected.push_back({h.name, false, {h.channels, h.length}});
            expected.push_back({h.output, false, {h.channels, h.length}});
        }
        require_features(model, path, expected);
    } catch (...) {
        moss_coreml_free(model);
        throw;
    }
    return model;
}

CoremlDecoder::CoremlDecoder(const std::string & path, const std::string & compute_units,
                             const std::string & source_hash) {
    model_ = load_decoder_model(path, compute_units, source_hash);
    try {
        label_ = moss_coreml_label(model_);
        reset();
        decode(std::vector<float>(size_t(Geometry::frames) * Geometry::latent));
        reset();
    } catch (...) {
        moss_coreml_free(model_);
        model_ = nullptr;
        throw;
    }
}

CoremlDecoder::CoremlDecoder(moss_coreml_model * borrowed, std::string label)
    : model_(borrowed), owned_(false), label_(std::move(label)) {
    reset();
}

CoremlDecoder::~CoremlDecoder() {
    if (model_ != nullptr && owned_) moss_coreml_free(model_);
}

void CoremlDecoder::reset() {
    position_ = 0;
    keys_.assign(KV_SIZE, 0.0f);
    values_.assign(KV_SIZE, 0.0f);
    histories_.resize(HISTORIES.size());
    for (size_t i = 0; i < HISTORIES.size(); ++i)
        histories_[i].assign(size_t(HISTORIES[i].channels) * HISTORIES[i].length, 0.0f);
}

// One prediction: RoPE tables for this call's transformer rows, and the mask
// letting row i see the WINDOW positions up to its own among the KEEP
// retained rows (unfilled before position KEEP) and this call's rows. The
// new state comes back into the same buffers: the runner copies every input
// before predicting and writes outputs after.
void CoremlDecoder::predict(const float * latent, float * pcm) {
    constexpr int keep = Geometry::keep, half = Geometry::head_dim / 2, columns = keep + DECODER_ROWS;
    cos_.resize(size_t(DECODER_ROWS) * half);
    sin_.resize(cos_.size());
    for (int row = 0; row < DECODER_ROWS; ++row) {
        coreml_rope(int(position_) + row, Geometry::head_dim, 10000.0f, row_cos_, row_sin_);
        std::copy(row_cos_.begin(), row_cos_.end(), cos_.begin() + size_t(row) * half);
        std::copy(row_sin_.begin(), row_sin_.end(), sin_.begin() + size_t(row) * half);
    }
    mask_.assign(size_t(DECODER_ROWS) * columns, COREML_MASKED);
    const int first = keep - int(std::min<int64_t>(position_, keep));
    for (int row = 0; row < DECODER_ROWS; ++row)
        std::fill(mask_.begin() + size_t(row) * columns + std::max(row, first),
                  mask_.begin() + size_t(row) * columns + keep + row + 1, 0.0f);
    std::vector<const char *> in_names = {"latents", "cos", "sin", "mask", "keys", "values"};
    std::vector<const float *> in_data = {latent, cos_.data(), sin_.data(), mask_.data(), keys_.data(), values_.data()};
    std::vector<const char *> out_names = {"pcm", "new_keys", "new_values"};
    std::vector<float *> out_data = {pcm, keys_.data(), values_.data()};
    for (size_t i = 0; i < HISTORIES.size(); ++i) {
        in_names.push_back(HISTORIES[i].name);
        in_data.push_back(histories_[i].data());
        out_names.push_back(HISTORIES[i].output);
        out_data.push_back(histories_[i].data());
    }
    if (moss_coreml_predict(model_, in_names.data(), in_data.data(), int(in_names.size()), out_names.data(),
                            out_data.data(), int(out_names.size())) != 0)
        fail("Mimi decoder prediction failed");
    position_ += DECODER_ROWS;
}

std::vector<float> CoremlDecoder::decode(const std::vector<float> & latents) {
    constexpr size_t frame = size_t(Geometry::frames) * Geometry::latent;
    if (latents.empty() || latents.size() % frame) fail("invalid Mimi decoder latents");
    std::vector<float> pcm(latents.size() / frame * Geometry::frames * Geometry::frame_samples);
    for (size_t at = 0, out = 0; at < latents.size(); at += frame, out += size_t(Geometry::frames) * Geometry::frame_samples)
        predict(latents.data() + at, pcm.data() + out);
    require_finite(pcm, "Mimi decoder output");
    return pcm;
}
#else
CoremlStep::CoremlStep(const std::string &, const std::string &, const FlowConfig & config, const std::string &, bool)
    : config_(config) {
    fail("this build has no Core ML (TTS_CPP_COREML=OFF)");
}
CoremlStep::~CoremlStep() = default;
void CoremlStep::warm_up() { fail("this build has no Core ML"); }
void CoremlStep::write_rows(const CacheRows &) { fail("this build has no Core ML"); }
void CoremlStep::reset(const CacheRows &) { fail("this build has no Core ML"); }
void CoremlStep::step(int, const std::vector<float> &, const std::vector<float> &, std::vector<float> &,
                      std::vector<float> &) {
    fail("this build has no Core ML");
}
CoremlDecoder::CoremlDecoder(const std::string &, const std::string &, const std::string &) {
    fail("this build has no Core ML (TTS_CPP_COREML=OFF)");
}
CoremlDecoder::CoremlDecoder(moss_coreml_model *, std::string) { fail("this build has no Core ML"); }
CoremlDecoder::~CoremlDecoder() = default;
moss_coreml_model * load_decoder_model(const std::string &, const std::string &, const std::string &) {
    fail("this build has no Core ML (TTS_CPP_COREML=OFF)");
}
void CoremlDecoder::reset() {}
void CoremlDecoder::predict(const float *, float *) { fail("this build has no Core ML"); }
std::vector<float> CoremlDecoder::decode(const std::vector<float> &) { fail("this build has no Core ML"); }
#endif

// ---- the sidecars on the shared Core ML runtime ----------------------------------

namespace {

constexpr const char * ENGINE_NAME = "pocket";
constexpr const char * ENV_PREFIX = "POCKET";
constexpr const char * STEP_TEMPLATE_ID = "pocket-step-c1024";
// The positions the generated step's state holds (the template's capacity).
constexpr int STEP_TEMPLATE_CONTEXT = 1024;
constexpr const char * DECODER_TEMPLATE_ID = "pocket-mimi";
constexpr const char * STEP_STAGE = "step";
constexpr const char * DECODER_STAGE = "mimi";

const char * stage_name(CoremlStage s) { return s == CoremlStage::step ? STEP_STAGE : DECODER_STAGE; }

void validate_units(const std::string & field, const std::string & value) {
    if (!value.empty() && !coreml_valid_units(value)) {
        throw std::invalid_argument("Pocket TTS: invalid " + field + "='" + value +
                                    "' (expected all, cpu_and_ane, cpu_and_gpu or cpu_only)");
    }
}

void validate_fields(const CoremlEngineOptions & opts) {
    validate_units("coreml.compute_units", opts.coreml.compute_units.value_or(""));
    if (opts.coreml.compute_units && opts.coreml.compute_units->empty()) {
        throw std::invalid_argument("Pocket TTS: coreml.compute_units is empty");
    }
    validate_units("coreml_step_units", opts.step_units);
    validate_units("coreml_decoder_units", opts.decoder_units);
    validate_units("coreml_compute_units", opts.legacy_units);
}

void validate_options(const CoremlOptions & o, CoremlStage stage, bool units_explicit) {
    if (stage == CoremlStage::step && units_explicit && o.compute_units && *o.compute_units == "cpu_and_ane") {
        throw std::invalid_argument("Pocket TTS: the Core ML step cannot run on cpu_and_ane (its KV cache is Core ML "
                                    "state, which the Neural Engine cannot hold); use cpu_and_gpu, all or cpu_only");
    }
    if (o.strict_wait_ms && *o.strict_wait_ms < 0) {
        throw std::invalid_argument("Pocket TTS: coreml.strict_wait_ms must not be negative");
    }
    if (o.memory_budget_bytes && *o.memory_budget_bytes < 0) {
        throw std::invalid_argument("Pocket TTS: coreml.memory_budget_bytes must not be negative");
    }
}

} // namespace

CoremlOptions coreml_stage_options(const CoremlEngineOptions & opts, CoremlStage stage) {
    CoremlOptions o = opts.coreml;
    const std::string & own = stage == CoremlStage::step ? opts.step_units : opts.decoder_units;
    if (!own.empty()) o.compute_units = own;
    if (!o.compute_units && !opts.legacy_units.empty()) o.compute_units = opts.legacy_units;
    if (o.use == CoremlOptions::Use::unset && opts.legacy_strict.has_value()) {
        o.use = *opts.legacy_strict ? CoremlOptions::Use::require : CoremlOptions::Use::automatic;
    }
    return o;
}

CoremlEngineOptions coreml_engine_options(const EngineOptions & o) {
    CoremlEngineOptions c;
    c.flow_lm_path = o.flow_lm_path;
    c.mimi_path = o.mimi_path;
    c.coreml = o.coreml;
    c.step_units = o.coreml_step_units;
    c.decoder_units = o.coreml_decoder_units;
    c.legacy_units = o.coreml_compute_units;
    c.legacy_strict = o.coreml_strict;
    return c;
}

uint64_t coreml_envelope_bytes(const CoremlRoute & step, const CoremlRoute & decoder) {
    uint64_t resident = 0, transient = 0;
    double margin = 0;
    for (const CoremlRoute * r : {&step, &decoder}) {
        if (r->route != "generate") continue;
        resident += r->resident_bytes;
        transient = std::max(transient, r->transient_bytes);
        margin = std::max(margin, r->margin);
    }
    if (margin <= 0) return 0;
    return uint64_t(std::ceil(double(resident + transient) * margin));
}

#ifdef TTS_CPP_USE_COREML

namespace pcm = QVAC_COREML_NS;

struct CoremlLoadedSource {
    pcm::SourceStat stat;
    bool ok = false;
};

std::shared_ptr<const CoremlLoadedSource> coreml_loaded_source(int fd, const std::string & path) {
    auto s = std::make_shared<CoremlLoadedSource>();
    s->ok = pcm::generation::source_stat_fd(fd, path, s->stat);
    return s;
}

namespace {

template <class E, class F> F map_enum(E v) { return static_cast<F>(static_cast<int>(v)); }

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

// Resolves one stage's options. A step resolved to cpu_and_ane from the
// environment or the legacy all-stage units keeps its default units instead
// (an explicit request throws in validate_options).
pcm::Resolved resolve_stage(const CoremlEngineOptions & opts, CoremlStage stage) {
    validate_fields(opts);
    const CoremlOptions o = coreml_stage_options(opts, stage);
    const bool units_explicit = opts.coreml.compute_units.has_value() ||
                                !(stage == CoremlStage::step ? opts.step_units : opts.decoder_units).empty();
    validate_options(o, stage, units_explicit);
    CoremlOptions runtime = o;
    if (stage == CoremlStage::step && !units_explicit && runtime.compute_units &&
        *runtime.compute_units == "cpu_and_ane") {
        runtime.compute_units.reset();
    }
    pcm::Resolved r = pcm::resolve_coreml(to_runtime_options(runtime), ENV_PREFIX);
    if (stage == CoremlStage::step && r.compute_units && *r.compute_units == pcm::ComputeUnits::cpu_and_ane) {
        r.warnings.push_back("the Core ML step cannot run on cpu_and_ane (Core ML state); it keeps " +
                             std::string(COREML_STEP_DEFAULT_UNITS));
        r.compute_units.reset();
    }
    return r;
}

uint64_t file_bytes(const std::string & path) {
    struct stat st {};
    return ::stat(path.c_str(), &st) == 0 ? uint64_t(st.st_size) : 0;
}

constexpr double ANE_COMPILE_RESERVE_FACTOR = 1.6;
// The ggml cost of one chunk bounds the wait for a warm load (M2: FlowLM about
// 10 ms and Mimi about 3 ms per frame on Metal; the CPU is slower).
constexpr double GGML_STEP_MS_PER_FRAME = 10.0;
constexpr double GGML_DECODER_MS_PER_FRAME = 3.0;
constexpr int64_t GGML_MS_FLOOR = 20;
constexpr size_t HASH_WORDS = 8;  // a 64-character source hash, 8 bytes per geometry value
constexpr size_t STEP_GEOMETRY = 6;  // dim, heads, layers, latent_dim, rope_base bits, min context

void put_hash(std::vector<int64_t> & g, const std::string & hash) {
    std::string h = hash;
    h.resize(HASH_WORDS * 8, '\0');
    for (size_t i = 0; i < HASH_WORDS; ++i) {
        int64_t v = 0;
        std::memcpy(&v, h.data() + 8 * i, 8);
        g.push_back(v);
    }
}

std::string get_hash(const std::vector<int64_t> & g, size_t at) {
    std::string h(HASH_WORDS * 8, '\0');
    for (size_t i = 0; i < HASH_WORDS && at + i < g.size(); ++i) std::memcpy(&h[8 * i], &g[at + i], 8);
    h.resize(std::strlen(h.c_str()));
    return h;
}

// A CoremlStep or load_decoder_model failure: Core ML refusing to load the
// bundle is retried by a later process; anything the checks reject is not.
pcm::Error classify(const std::exception & e) {
    const std::string what = e.what();
    if (what.find("cannot load ") != std::string::npos) return pcm::Error::transient(what);
    return pcm::Error::deterministic(what);
}

// LoadFn of the step: the model, checked against the geometry the constructing
// thread passed by value ({dim, heads, layers, latent_dim, rope_base bits,
// minimum context} and the source hash), without its warm-up (warm_step).
pcm::LoadResult load_step(const pcm::LoadRequest & req) {
    pcm::LoadResult r;
    if (!pcm::generation::begin_load(req, r)) return r;
    auto fail_with = [&](const pcm::Error & e) {
        r.model.reset();
        r.error = e;
        pcm::generation::end_load(req, r);
        return r;
    };
    if (req.geometry.size() != STEP_GEOMETRY + HASH_WORDS) {
        return fail_with(pcm::Error::deterministic("Pocket TTS: step load without its geometry"));
    }
    FlowConfig config;
    config.dim = int(req.geometry[0]);
    config.heads = int(req.geometry[1]);
    config.layers = int(req.geometry[2]);
    config.latent_dim = int(req.geometry[3]);
    const uint32_t rope_bits = uint32_t(req.geometry[4]);
    std::memcpy(&config.rope_base, &rope_bits, 4);
    const int min_context = int(req.geometry[5]);
    try {
        auto step = std::make_shared<CoremlStep>(req.artifact_path, pcm::to_string(req.units), config,
                                                 get_hash(req.geometry, STEP_GEOMETRY), false);
        if (step->context() < min_context) {
            return fail_with(pcm::Error::deterministic(
                "Pocket TTS: the Core ML step holds " + std::to_string(step->context()) + " positions; the voice needs " +
                std::to_string(min_context) + " (export a longer --context)"));
        }
        r.model = std::static_pointer_cast<void>(step);
    } catch (const std::exception & e) {
        return fail_with(classify(e));
    }
    pcm::generation::end_load(req, r);
    return r;
}

pcm::Error warm_step(void * model, const pcm::LoadRequest &) {
    try {
        static_cast<CoremlStep *>(model)->warm_up();
    } catch (const std::exception & e) {
        return pcm::Error::transient(std::string("Pocket TTS: the Core ML step warm-up failed: ") + e.what());
    }
    return {};
}

// The decoder model a slot holds: shared by every Engine on the same artifact;
// each chunk streams on it with its own CoremlDecoder state.
struct decoder_model {
    moss_coreml_model * model = nullptr;
    std::string label;
    ~decoder_model() {
        if (model) moss_coreml_free(model);
    }
};

pcm::LoadResult load_decoder(const pcm::LoadRequest & req) {
    pcm::LoadResult r;
    if (!pcm::generation::begin_load(req, r)) return r;
    auto fail_with = [&](const pcm::Error & e) {
        r.model.reset();
        r.error = e;
        pcm::generation::end_load(req, r);
        return r;
    };
    if (req.geometry.size() != HASH_WORDS) {
        return fail_with(pcm::Error::deterministic("Pocket TTS: Mimi decoder load without its source hash"));
    }
    try {
        auto dm = std::make_shared<decoder_model>();
        dm->model = load_decoder_model(req.artifact_path, pcm::to_string(req.units), get_hash(req.geometry, 0));
        dm->label = moss_coreml_label(dm->model);
        r.model = std::static_pointer_cast<void>(dm);
    } catch (const std::exception & e) {
        return fail_with(classify(e));
    }
    pcm::generation::end_load(req, r);
    return r;
}

// One frame on a silent latent, so a chunk's first frame skips Core ML's
// first-prediction setup.
pcm::Error warm_decoder(void * model, const pcm::LoadRequest &) {
    auto & dm = *static_cast<decoder_model *>(model);
    try {
        CoremlDecoder stream(dm.model, dm.label);
        stream.decode(std::vector<float>(size_t(CoremlDecoderGeometry::frames) * CoremlDecoderGeometry::latent));
    } catch (const std::exception & e) {
        return pcm::Error::transient(std::string("Pocket TTS: the Core ML Mimi decoder warm-up failed: ") + e.what());
    }
    return {};
}

struct planned {
    CoremlRoute route;
    pcm::Resolved resolved;
    pcm::Plan plan;
    pcm::ComputeUnits units = pcm::ComputeUnits::all;
    std::optional<pcm::generation::Candidate> candidate;
    std::string prebuilt_path;
};

planned plan_stage(CoremlStage stage, const CoremlEngineOptions & opts) {
    planned out;
    out.resolved = resolve_stage(opts, stage);
    const pcm::Resolved & r = out.resolved;
    out.route.strict = r.strict();
    const bool step = stage == CoremlStage::step;
    const std::string & gguf = step ? opts.flow_lm_path : opts.mimi_path;
    out.prebuilt_path = step ? coreml_step_path(gguf) : coreml_decoder_path(gguf);
    const std::shared_ptr<pcm::Manager> mgr = pcm::Manager::instance();
    pcm::PlanInput in;
    in.coreml_built = true;
    in.generation_built = pcm::generation::built();
    in.prebuilt_present = !gguf.empty() && ::tts_cpp::detail::coreml_sidecar_exists(out.prebuilt_path);
    if (in.generation_built && !in.prebuilt_present && r.enabled() && !gguf.empty()) {
#ifdef QVAC_COREML_GENERATE
        pcm::pocket_provided::register_all();
#endif
        std::vector<pcm::generation::Candidate> c =
            pcm::generation::probe({step ? STEP_TEMPLATE_ID : DECODER_TEMPLATE_ID}, gguf, r);
        if (!c.empty()) out.candidate = std::move(c.front());
    }
    const bool available = out.candidate && out.candidate->available;
    const pcm::ComputeUnits stage_default =
        available && out.candidate->default_units
            ? *out.candidate->default_units
            : *pcm::parse_compute_units(step ? COREML_STEP_DEFAULT_UNITS : COREML_DECODER_DEFAULT_UNITS);
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
        out.route.resident_bytes = pcm::generation::resident_bytes(m);
        out.route.workspace_bytes = m.workspace_bytes;
        out.route.prepare_peak_bytes = pcm::generation::prepare_peak_bytes(m);
        out.route.load_peak_bytes = pcm::generation::load_peak_bytes(m, out.units);
        out.route.transient_bytes = pcm::generation::transient_bytes(m, out.units);
        out.route.margin = m.margin > 0 ? m.margin : 1.25;
        out.route.ane_compile_reserve_bytes = pcm::generation::ane_compile_reserve_bytes(m, out.units);
        out.route.validated = std::find(m.validated_units.begin(), m.validated_units.end(), out.units) !=
                              m.validated_units.end();
    }
    return out;
}

struct stage_slot {
    planned plan;
    std::shared_ptr<pcm::Slot> slot;
    pcm::SlotStatus none_status;
};

} // namespace

CoremlRoute coreml_plan_route(CoremlStage stage, const CoremlEngineOptions & opts) {
    return plan_stage(stage, opts).route;
}

struct CoremlStages::impl {
    std::shared_ptr<pcm::Manager> manager;
    stage_slot step, decoder;
    std::shared_ptr<pcm::Group> construction;
};

namespace {

void make_stage_slot(stage_slot & s, CoremlStage stage, const CoremlEngineOptions & opts,
                     const std::shared_ptr<pcm::Manager> & mgr, std::vector<int64_t> geometry) {
    const bool step = stage == CoremlStage::step;
    if (s.plan.plan.route == pcm::Route::none) {
        s.none_status.stage = stage_name(stage);
        s.none_status.function = "main";
        s.none_status.state = s.plan.plan.none_state;
        s.none_status.reason = s.plan.plan.reason;
        s.none_status.compute_units = s.plan.route.compute_units;
        s.none_status.label = GGML_STAGE_BACKEND;
        return;
    }
    pcm::SlotSpec spec;
    spec.engine = ENGINE_NAME;
    spec.stage = stage_name(stage);
    spec.function = "main";
    spec.units = s.plan.units;
    spec.route = s.plan.plan.route;
    spec.load_fn = step ? &load_step : &load_decoder;
    spec.warm_fn = step ? &warm_step : &warm_decoder;
    spec.geometry = std::move(geometry);
    // The step holds the KV cache in Core ML state: one model per Engine.
    spec.shareable = !step;
    if (spec.route == pcm::Route::prebuilt) {
        spec.artifact_path = s.plan.prebuilt_path;
        spec.ane_compile_reserve_bytes =
            pcm::units_include_ane(spec.units)
                ? uint64_t(ANE_COMPILE_RESERVE_FACTOR * double(file_bytes(spec.artifact_path + "/weights/weight.bin")))
                : 0;
    } else {
        const pcm::generation::Candidate & c = *s.plan.candidate;
        // The file the stage's weights were read from (fstat of the load's
        // one descriptor), not whatever the path names now: prepare opens the
        // path and fails with source_changed unless it is that file. A
        // missing identity or a path that no longer resolves fails the same
        // way, and the stage stays on ggml.
        const std::string & gguf = step ? opts.flow_lm_path : opts.mimi_path;
        const std::shared_ptr<const CoremlLoadedSource> & loaded = step ? opts.step_source : opts.decoder_source;
        pcm::SourceStat source;
        pcm::SourceStat asked;
        const bool ok = loaded && loaded->ok;
        if (ok) source = loaded->stat;
        source.changed = !ok || !pcm::generation::source_stat(gguf, asked) || asked.path != source.path;
        spec.tmpl = c.tmpl;
        spec.prepare = pcm::generation::prepare_request(c, source, s.plan.resolved);
        spec.prepare_fn = &pcm::generation::prepare;
        spec.ane_compile_reserve_bytes = pcm::generation::ane_compile_reserve_bytes(c.memory, spec.units);
    }
    s.slot = mgr->make_slot(std::move(spec), pcm::SlotPolicy::from(s.plan.resolved));
}

[[noreturn]] void fail_strict(CoremlStage stage, const std::string & why) {
    throw std::runtime_error(std::string("Pocket TTS: the Core ML ") + (stage == CoremlStage::step ? "step" : "Mimi decoder") +
                             " cannot serve this call (" + why + "); Core ML strict policy is enabled (use = require / " +
                             COREML_STRICT_ENV + ")");
}

pcm::Acquired acquire_stage(const stage_slot & s, CoremlStage stage, int frames, double ms_per_frame,
                            const std::atomic<bool> * cancel) {
    if (!s.slot) return {};
    const int64_t budget_ms = std::max<int64_t>(GGML_MS_FLOOR, int64_t(ms_per_frame * double(std::max(frames, 0))));
    const auto deadline = pcm::Clock::now() + std::chrono::milliseconds(budget_ms);
    try {
        return s.slot->acquire(pcm::Demand::now, deadline, cancel);
    } catch (const pcm::CoremlError & e) {
        fail_strict(stage, e.what());
    }
}

} // namespace

std::unique_ptr<CoremlStages> CoremlStages::create(const CoremlEngineOptions & opts, const FlowConfig & config,
                                                   const std::string & source_hash, int min_step_context,
                                                   bool degrade) {
    auto p = std::make_unique<impl>();
    p->manager = pcm::Manager::instance();
    p->step.plan = plan_stage(CoremlStage::step, opts);
    p->decoder.plan = plan_stage(CoremlStage::decoder, opts);
    for (stage_slot * s : {&p->step, &p->decoder}) {
        for (const std::string & w : s->plan.resolved.warnings) std::fprintf(stderr, "Pocket TTS: %s\n", w.c_str());
        if (degrade && s->plan.resolved.enabled()) {
            // Fit's memory-degrade: only the ggml projection fits the budget.
            s->plan.resolved.use = pcm::Use::off;
            s->plan.resolved.disabled_reason = "memory";
            s->plan.plan.route = pcm::Route::none;
            s->plan.plan.none_state = pcm::SlotState::disabled;
            s->plan.plan.reason = "memory";
            s->plan.route.route = "none";
            s->plan.route.state = pcm::to_string(pcm::SlotState::disabled);
            s->plan.route.reason = "memory";
            s->plan.route.strict = false;
        }
    }
    for (stage_slot * s : {&p->step, &p->decoder}) {
        if (s->plan.plan.route == pcm::Route::none && s->plan.resolved.strict()) {
            const CoremlStage st = s == &p->step ? CoremlStage::step : CoremlStage::decoder;
            fail_strict(st, "no usable sidecar: " + s->plan.route.state + ": " + s->plan.route.reason);
        }
    }
    std::vector<int64_t> step_geometry = {config.dim, config.heads, config.layers, config.latent_dim};
    uint32_t rope_bits = 0;
    std::memcpy(&rope_bits, &config.rope_base, 4);
    step_geometry.push_back(int64_t(rope_bits));
    step_geometry.push_back(min_step_context);
    put_hash(step_geometry, source_hash);
    std::vector<int64_t> decoder_geometry;
    put_hash(decoder_geometry, source_hash);
    make_stage_slot(p->step, CoremlStage::step, opts, p->manager, std::move(step_geometry));
    make_stage_slot(p->decoder, CoremlStage::decoder, opts, p->manager, std::move(decoder_geometry));

    std::vector<pcm::GroupRequest> requests;
    std::vector<pcm::SlotStatus> skipped;
    bool waited = false;
    for (stage_slot * s : {&p->step, &p->decoder}) {
        if (!s->slot) {
            skipped.push_back(s->none_status);
            continue;
        }
        if (s->plan.resolved.prewarm != pcm::Prewarm::none) requests.push_back({s->slot, pcm::Priority::minimal});
        waited = waited || s->plan.resolved.wait == pcm::Wait::construct;
    }
    p->construction = p->manager->issue_group(pcm::GroupKind::construction, requests, waited, skipped);
    std::unique_ptr<CoremlStages> out(new CoremlStages(std::move(p)));
    if (waited && !requests.empty()) {
        const pcm::PrewarmResult result = out->p_->construction->wait();
        for (stage_slot * s : {&out->p_->step, &out->p_->decoder}) {
            if (!s->slot || !s->plan.resolved.strict() || result.success) continue;
            const pcm::SlotStatus st = s->slot->status();
            if (st.state == pcm::SlotState::ready) continue;
            fail_strict(s == &out->p_->step ? CoremlStage::step : CoremlStage::decoder,
                        "not ready after construction: " + pcm::state_label(st.state, st.reason) +
                            (st.error.empty() ? "" : ": " + st.error));
        }
    }
    return out;
}

CoremlStages::CoremlStages(std::unique_ptr<impl> p) : p_(std::move(p)) {}
CoremlStages::~CoremlStages() = default;

CoremlStepUse CoremlStages::acquire_step(int frames, const std::atomic<bool> * cancel) {
    CoremlStepUse use;
    pcm::Acquired got = acquire_stage(p_->step, CoremlStage::step, frames, GGML_STEP_MS_PER_FRAME, cancel);
    use.pending = got.pending;
    if (!got.model) return use;
    auto handle = std::make_shared<pcm::Handle>(std::move(got.model));
    pcm::SharedModel * m = handle->get();
    use.step = m->as<CoremlStep>();
    use.identity = m;
    use.label = m->label();
    use.lock = [m] { return m->lock_for_prediction(); };
    use.retire = [handle](const std::string & reason) { (*handle)->retire(reason); };
    use.hold = handle;
    return use;
}

CoremlDecoderUse CoremlStages::acquire_decoder(int frames, const std::atomic<bool> * cancel) {
    CoremlDecoderUse use;
    pcm::Acquired got = acquire_stage(p_->decoder, CoremlStage::decoder, frames, GGML_DECODER_MS_PER_FRAME, cancel);
    use.pending = got.pending;
    if (!got.model) return use;
    auto handle = std::make_shared<pcm::Handle>(std::move(got.model));
    pcm::SharedModel * m = handle->get();
    use.model = m->as<decoder_model>()->model;
    use.label = m->label();
    use.lock = [m] { return m->lock_for_prediction(); };
    use.retire = [handle](const std::string & reason) { (*handle)->retire(reason); };
    use.hold = handle;
    return use;
}

bool CoremlStages::step_strict() const { return p_->step.plan.resolved.strict(); }
bool CoremlStages::decoder_strict() const { return p_->decoder.plan.resolved.strict(); }
const CoremlRoute & CoremlStages::step_route() const { return p_->step.plan.route; }
const CoremlRoute & CoremlStages::decoder_route() const { return p_->decoder.plan.route; }
bool CoremlStages::step_ready() const { return p_->step.slot && p_->step.slot->state() == pcm::SlotState::ready; }
int CoremlStages::step_capacity() const {
    return p_->step.slot && p_->step.plan.plan.route == pcm::Route::generate ? STEP_TEMPLATE_CONTEXT : 0;
}
bool CoremlStages::decoder_ready() const {
    return p_->decoder.slot && p_->decoder.slot->state() == pcm::SlotState::ready;
}

std::string CoremlStages::status_json() const {
    pcm::GenerationStatus gen;
    gen.supported = pcm::generation::built() && p_->manager->platform() == pcm::Platform::macos &&
                    pcm::os_at_least(p_->manager->os_version(), 15, 0);
    if (!gen.supported) {
        gen.reason = !pcm::generation::built() ? "build" : "os";
    } else {
        bool all_unvalidated = true, any_generate = false;
        for (const stage_slot * s : {&p_->step, &p_->decoder}) {
            const bool unvalidated = s->plan.plan.route == pcm::Route::none && s->plan.plan.reason == "unvalidated_units";
            all_unvalidated = all_unvalidated && unvalidated;
            any_generate = any_generate || unvalidated || s->plan.plan.route == pcm::Route::generate;
        }
        if (any_generate && all_unvalidated) gen.reason = "unvalidated_units";
    }
    std::vector<std::shared_ptr<pcm::Slot>> slots;
    for (const stage_slot * s : {&p_->step, &p_->decoder}) {
        if (s->slot) slots.push_back(s->slot);
    }
    pcm::EngineStatus st = p_->manager->status(ENGINE_NAME, p_->step.plan.resolved.cache_dir, slots, gen);
    // Stages without a slot report their route-none state, in stage order.
    std::vector<pcm::SlotStatus> ordered;
    size_t next = 0;
    for (const stage_slot * s : {&p_->step, &p_->decoder}) {
        if (s->slot) {
            if (next < st.stages.size()) ordered.push_back(st.stages[next++]);
        } else {
            ordered.push_back(s->none_status);
        }
    }
    st.stages = std::move(ordered);
    return pcm::to_json(st);
}

bool CoremlStages::wait(int64_t timeout_ms, const std::atomic<uint64_t> * cancels, uint64_t since) const {
    const auto start = pcm::Clock::now();
    bool any = false;
    for (const stage_slot * s : {&p_->step, &p_->decoder}) any = any || s->slot != nullptr;
    if (!any) return false;
    for (;;) {
        bool all_ready = true;
        for (const stage_slot * s : {&p_->step, &p_->decoder}) {
            if (!s->slot) continue;
            const pcm::SlotState st = s->slot->state();
            if (st == pcm::SlotState::ready) continue;
            // Terminal, or nothing will move it without a request: idle
            // (prewarm none before the first chunk) and deferred(memory).
            if (pcm::is_terminal(st) || st == pcm::SlotState::idle || st == pcm::SlotState::deferred_memory) {
                return false;
            }
            all_ready = false;
        }
        if (all_ready) return true;
        if (cancels && cancels->load() != since) return false;
        if (timeout_ms >= 0 && pcm::Clock::now() - start >= std::chrono::milliseconds(timeout_ms)) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

std::string CoremlStages::construction_json() const { return pcm::to_json(p_->construction->result()); }

std::string CoremlStages::prewarm(CoremlOptions::Prewarm scope, bool wait, const std::atomic<uint64_t> * cancels,
                                  uint64_t since) {
    std::vector<pcm::GroupRequest> requests;
    std::vector<pcm::SlotStatus> skipped;
    for (stage_slot * s : {&p_->step, &p_->decoder}) {
        if (!s->slot) skipped.push_back(s->none_status);
        else if (scope != CoremlOptions::Prewarm::none) requests.push_back({s->slot, pcm::Priority::minimal});
    }
    const std::shared_ptr<pcm::Group> group =
        p_->manager->issue_group(pcm::GroupKind::prewarm, requests, wait, std::move(skipped));
    while (wait && !group->finished() && !(cancels && cancels->load() != since))
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    return pcm::to_json(group->result());
}

#else  // !TTS_CPP_USE_COREML

struct CoremlLoadedSource {};

std::shared_ptr<const CoremlLoadedSource> coreml_loaded_source(int, const std::string &) { return nullptr; }

CoremlRoute coreml_plan_route(CoremlStage stage, const CoremlEngineOptions & opts) {
    validate_fields(opts);
    const CoremlOptions o = coreml_stage_options(opts, stage);
    const bool units_explicit = opts.coreml.compute_units.has_value() ||
                                !(stage == CoremlStage::step ? opts.step_units : opts.decoder_units).empty();
    validate_options(o, stage, units_explicit);
    CoremlRoute r;
    r.state = "unsupported";
    r.reason = "build";
    return r;
}

struct CoremlStages::impl {};
std::unique_ptr<CoremlStages> CoremlStages::create(const CoremlEngineOptions & opts, const FlowConfig &,
                                                   const std::string &, int, bool) {
    coreml_plan_route(CoremlStage::step, opts);
    coreml_plan_route(CoremlStage::decoder, opts);
    return nullptr;
}
CoremlStages::CoremlStages(std::unique_ptr<impl> p) : p_(std::move(p)) {}
CoremlStages::~CoremlStages() = default;
CoremlStepUse CoremlStages::acquire_step(int, const std::atomic<bool> *) { return {}; }
CoremlDecoderUse CoremlStages::acquire_decoder(int, const std::atomic<bool> *) { return {}; }
bool CoremlStages::step_strict() const { return false; }
bool CoremlStages::decoder_strict() const { return false; }
const CoremlRoute & CoremlStages::step_route() const { static const CoremlRoute r; return r; }
const CoremlRoute & CoremlStages::decoder_route() const { static const CoremlRoute r; return r; }
bool CoremlStages::step_ready() const { return false; }
int CoremlStages::step_capacity() const { return 0; }
bool CoremlStages::decoder_ready() const { return false; }
std::string CoremlStages::status_json() const { return "{}"; }
bool CoremlStages::wait(int64_t, const std::atomic<uint64_t> *, uint64_t) const { return false; }
std::string CoremlStages::construction_json() const { return coreml_empty_group_json("construction"); }
std::string CoremlStages::prewarm(CoremlOptions::Prewarm, bool, const std::atomic<uint64_t> *, uint64_t) {
    return coreml_empty_group_json("prewarm");
}

#endif

std::string coreml_empty_group_json(const char * group) {
    return std::string("{\"group\":\"") + group +
           "\",\"nothing_to_prepare\":true,\"skipped_route_none\":[],\"slots\":[],\"success\":true}";
}

} // namespace tts_cpp::pocket::detail
