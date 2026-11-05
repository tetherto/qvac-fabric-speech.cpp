#pragma once
// Pocket FlowLM Core ML step sidecar (TTS_CPP_COREML builds, Apple only).
//
// scripts/export-pocket-coreml.py exports one generated frame of the FlowLM
// loop (input_linear, the transformer, out_norm and a one-step LSD flow) with
// the KV cache in Core ML state, next to the FlowLM GGUF under the shared
// naming rule (coreml_sidecar_path.h):
//
//   flow-lm.gguf -> flow-lm-step.mlmodelc
//
// The voice prefix and text prefill stay on ggml; their KV rows are copied
// into the state before a chunk's first frame, then every frame is one
// prediction. Mimi, a chunk whose positions exceed the sidecar's context, and
// everything once a missing or retired sidecar leaves it, run on ggml.

#include "pocket/flow_lm.h"

#include <optional>
#include <string>
#include <vector>

struct moss_coreml_model;

namespace tts_cpp::pocket::detail {

constexpr const char * COREML_DISABLE_ENV = "POCKET_COREML_DISABLE";
constexpr const char * COREML_STRICT_ENV = "POCKET_COREML_STRICT";
constexpr const char * COREML_UNITS_ENV = "POCKET_COREML_COMPUTE_UNITS";
// Core ML cannot place model state on the Neural Engine (a cpu_and_ane load
// fails); CPU + GPU measured fastest with the predictable latency on an M2.
constexpr const char * COREML_DEFAULT_UNITS = "cpu_and_gpu";
constexpr const char * COREML_STEP_SUFFIX = "-step";
constexpr const char * COREML_SCHEMA = "1";
constexpr const char * GGML_FLOW_LM_BACKEND = "ggml";
constexpr const char * MIXED_FLOW_LM_BACKEND = "mixed";
// Must equal MASKED in scripts/pocket_coreml_models.py.
constexpr float COREML_MASKED = -1e4f;

// Resolved once per Engine. Explicit values never write the environment.
struct CoremlPolicy {
    std::string compute_units = COREML_DEFAULT_UNITS;
    bool strict = false;
    bool disabled = false;
};

bool coreml_valid_units(const std::string & units);

// Empty `compute_units` inherits POCKET_COREML_COMPUTE_UNITS (an invalid
// inherited value warns and keeps the default); an explicit invalid value
// throws. Unset `strict` inherits POCKET_COREML_STRICT (presence means true).
// POCKET_COREML_DISABLE keeps FlowLM on ggml.
CoremlPolicy resolve_coreml_policy(const std::string & compute_units = {},
                                   std::optional<bool> strict = std::nullopt);

std::string coreml_step_path(const std::string & flow_lm_gguf);

// True when this build has Core ML, the policy leaves it enabled and the step
// sidecar sits next to the GGUF (presence only; nothing is loaded).
bool coreml_step_enabled(const std::string & flow_lm_gguf, const CoremlPolicy & policy);

// cos/sin [head_dim / 2] of ggml_rope_ext mode 0 (adjacent pairs) at `position`.
void coreml_rope(int position, int head_dim, float base, std::vector<float> & cos, std::vector<float> & sin);

class CoremlStep {
public:
    // Loads the sidecar and checks, before any prediction, the shape of every
    // input, output and state the step uses against `config` (they are fixed
    // for the model's life, and the shared runner copies by the declared
    // shapes), its schema and that it was exported from the checkpoint
    // `source_hash` names; then runs one warm-up prediction at position 0,
    // which also fails for a sidecar with any further feature. Throws
    // std::runtime_error.
    CoremlStep(const std::string & path, const std::string & compute_units, const FlowConfig & config,
               const std::string & source_hash);
    ~CoremlStep();
    CoremlStep(const CoremlStep &) = delete;
    CoremlStep & operator=(const CoremlStep &) = delete;

    // Positions the state holds (voice + text + generated frames).
    int context() const { return context_; }
    const std::string & label() const { return label_; }

    // Copies cache rows (FlowLM::cache_rows) into the state at their positions.
    void write_rows(const CacheRows & rows);

    // One frame at `position` (the cache holds every earlier one): latent
    // (bos_latent() for a chunk's first frame) and noise [latent_dim] ->
    // hidden [dim] (out_norm) and next_latent [latent_dim] (one flow step).
    // Throws on a failed prediction or a non-finite output.
    void step(int position, const std::vector<float> & latent, const std::vector<float> & noise,
              std::vector<float> & hidden, std::vector<float> & next_latent);

private:
    moss_coreml_model * model_ = nullptr;
    FlowConfig config_;
    int context_ = 0;
    std::string label_ = "coreml";
    std::vector<float> cos_, sin_, mask_, write_mask_, staging_;
    int masked_from_ = 0;  // mask_ is 0 before this position, MASKED from it
    int written_ = -1;     // position write_mask_ marks
};

} // namespace tts_cpp::pocket::detail
