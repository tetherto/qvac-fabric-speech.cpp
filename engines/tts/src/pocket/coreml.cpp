#include "pocket/coreml.h"

#include "coreml_sidecar_path.h"

#ifdef TTS_CPP_USE_COREML
#include "moss/coreml/runner.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <stdexcept>

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
#endif

} // namespace

bool coreml_valid_units(const std::string & units) {
    return units == "all" || units == "cpu_and_ane" || units == "cpu_and_gpu" || units == "cpu_only";
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

bool coreml_step_enabled(const std::string & flow_lm_gguf, const CoremlPolicy & policy) {
#ifdef TTS_CPP_USE_COREML
    return !policy.disabled && !flow_lm_gguf.empty() &&
        ::tts_cpp::detail::coreml_sidecar_exists(coreml_step_path(flow_lm_gguf));
#else
    (void) flow_lm_gguf;
    (void) policy;
    return false;
#endif
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
                       const std::string & source_hash) : config_(config) {
    model_ = moss_coreml_load(path.c_str(), compute_units.c_str());
    if (model_ == nullptr) {
        fail("cannot load " + path + " on " + compute_units +
             (compute_units == "cpu_and_ane" ? " (its KV cache is Core ML state, which the Neural Engine cannot hold; "
                                               "use cpu_and_gpu, all or cpu_only)" : ""));
    }
    try {
        if (!moss_coreml_has_state(model_)) fail(path + " declares no KV state; re-export it");
        if (metadata(model_, "pocket.coreml.schema") != COREML_SCHEMA)
            fail(path + " has an unsupported schema; re-export it with scripts/export-pocket-coreml.py");
        if (metadata(model_, "pocket.source_sha256") != source_hash)
            fail(path + " was exported from another checkpoint than the FlowLM GGUF");
        const int64_t heads = config.heads, head_dim = config.dim / config.heads;
        const std::vector<int64_t> mask = feature_dims(model_, "mask", false);
        context_ = mask.size() == 4 ? int(mask[3]) : 0;
        if (context_ < 2) fail(path + " declares mask " + dims_text(mask) + "; expected [1, 1, 1, context]");
        const int64_t c = context_, latent = config.latent_dim;
        const struct { const char * name; bool state; std::vector<int64_t> dims; } expected[] = {
            {"latent", false, {1, latent}}, {"noise", false, {1, latent}},
            {"cos", false, {1, head_dim / 2}}, {"sin", false, {1, head_dim / 2}},
            {"mask", false, {1, 1, 1, c}}, {"write_mask", false, {1, 1, c, 1}},
            {"hidden", false, {1, config.dim}}, {"next_latent", false, {1, latent}},
            {"keys", true, {config.layers, heads, c, head_dim}}, {"values", true, {config.layers, heads, c, head_dim}},
        };
        for (const auto & feature : expected) {
            const auto declared = feature_dims(model_, feature.name, feature.state);
            if (declared != feature.dims)
                fail(path + " declares " + feature.name + " " + dims_text(declared) + " where this FlowLM needs " +
                     dims_text(feature.dims));
        }
    } catch (...) {
        moss_coreml_free(model_);
        model_ = nullptr;
        throw;
    }
    label_ = moss_coreml_label(model_);
    mask_.assign(size_t(context_), COREML_MASKED);
    write_mask_.assign(size_t(context_), 0.0f);
    // One prediction at position 0 (the caller then writes the voice rows over
    // it): a sidecar declaring inputs or outputs beyond the checked ones fails
    // here rather than mid-synthesis, and the first frame skips Core ML's
    // first-prediction setup.
    try {
        std::vector<float> zeros(size_t(config.latent_dim)), hidden, next;
        step(0, zeros, zeros, hidden, next);
    } catch (...) {
        moss_coreml_free(model_);
        model_ = nullptr;
        throw;
    }
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
#else
CoremlStep::CoremlStep(const std::string &, const std::string &, const FlowConfig & config, const std::string &)
    : config_(config) {
    fail("this build has no Core ML (TTS_CPP_COREML=OFF)");
}
CoremlStep::~CoremlStep() = default;
void CoremlStep::write_rows(const CacheRows &) { fail("this build has no Core ML"); }
void CoremlStep::step(int, const std::vector<float> &, const std::vector<float> &, std::vector<float> &,
                      std::vector<float> &) {
    fail("this build has no Core ML");
}
#endif

} // namespace tts_cpp::pocket::detail
