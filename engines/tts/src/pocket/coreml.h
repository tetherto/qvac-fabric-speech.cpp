#pragma once
// Pocket Core ML sidecars (TTS_CPP_COREML builds, Apple only).
//
// scripts/export-pocket-coreml.py exports them next to their GGUFs under the
// shared naming rule (coreml_sidecar_path.h):
//
//   flow-lm.gguf -> flow-lm-step.mlmodelc
//   mimi.gguf    -> mimi-decoder.mlmodelc
//
// The step is one generated frame of the FlowLM loop (input_linear, the
// transformer, out_norm and a one-step LSD flow) with the KV cache in Core ML
// state. The voice prefix and text prefill stay on ggml; their KV rows are
// copied into the state before a chunk's first frame, then every frame is one
// prediction. A chunk whose positions exceed the step's context runs FlowLM
// on ggml.
//
// The decoder is one Mimi decode frame with its streaming state (conv
// histories, the attention window) as explicit inputs and outputs kept here,
// so it runs on the Neural Engine. Either stage runs on ggml once a missing
// or retired sidecar leaves it.
//
// The Engine loads both through the shared Core ML runtime (CoremlStages
// below): a prebuilt sidecar next to its GGUF loads in the background; without
// one, and with a cache directory, the sidecar is generated from the GGUF
// (templates pocket-step-c1024 and pocket-mimi) on macOS 15 or later. Until a
// sidecar is ready its stage runs on ggml. Each chunk takes each stage's
// backend once, at its start (the step on the producer thread, the decoder on
// the calling thread), so a chunk never switches backend except after a
// failed prediction.

#include "pocket/flow_lm.h"
#include "tts-cpp/coreml_options.h"
#include "tts-cpp/pocket/engine.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
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
constexpr const char * COREML_STEP_DEFAULT_UNITS = "cpu_and_gpu";
// The stateless decoder runs fastest on the Neural Engine, off the GPU the
// step uses (M2: 1.5 ms per frame vs 5.1 ms on CPU + GPU).
constexpr const char * COREML_DECODER_DEFAULT_UNITS = "cpu_and_ane";
constexpr const char * COREML_STEP_SUFFIX = "-step";
constexpr const char * COREML_DECODER_SUFFIX = "-decoder";
constexpr const char * COREML_SCHEMA = "1";
constexpr const char * GGML_STAGE_BACKEND = "ggml";
constexpr const char * MIXED_STAGE_BACKEND = "mixed";
// Must equal MASKED in scripts/pocket_coreml_models.py.
constexpr float COREML_MASKED = -1e4f;

// Resolved once per Engine. Explicit values never write the environment.
// Empty compute_units keeps each sidecar's default (coreml_units()).
struct CoremlPolicy {
    std::string compute_units;
    bool strict = false;
    bool disabled = false;
};

bool coreml_valid_units(const std::string & units);

// Empty `compute_units` inherits POCKET_COREML_COMPUTE_UNITS (an invalid
// inherited value warns and keeps the defaults); an explicit invalid value
// throws. A value applies to both sidecars. Unset `strict` inherits
// POCKET_COREML_STRICT (presence means true). POCKET_COREML_DISABLE keeps
// FlowLM and Mimi on ggml.
CoremlPolicy resolve_coreml_policy(const std::string & compute_units = {},
                                   std::optional<bool> strict = std::nullopt);

// The policy's compute units, or `sidecar_default` when it sets none.
std::string coreml_units(const CoremlPolicy & policy, const char * sidecar_default);

std::string coreml_step_path(const std::string & flow_lm_gguf);
std::string coreml_decoder_path(const std::string & mimi_gguf);

// True when this build has Core ML, the policy leaves it enabled and the
// sidecar sits next to the GGUF (presence only; nothing is loaded).
bool coreml_step_enabled(const std::string & flow_lm_gguf, const CoremlPolicy & policy);
bool coreml_decoder_enabled(const std::string & mimi_gguf, const CoremlPolicy & policy);

// cos/sin [head_dim / 2] of ggml_rope_ext mode 0 (adjacent pairs) at `position`.
void coreml_rope(int position, int head_dim, float base, std::vector<float> & cos, std::vector<float> & sin);

class CoremlStep {
public:
    // Loads the sidecar and checks, before any prediction, the shape of every
    // input, output and state the step uses against `config` (they are fixed
    // for the model's life, and the shared runner copies by the declared
    // shapes), its schema and that it was exported from the checkpoint
    // `source_hash` names; then runs one warm-up prediction at position 0,
    // which also fails for a sidecar with any further feature (unless
    // `warm` is false; warm_up() then runs it). Throws std::runtime_error.
    CoremlStep(const std::string & path, const std::string & compute_units, const FlowConfig & config,
               const std::string & source_hash, bool warm = true);
    ~CoremlStep();
    CoremlStep(const CoremlStep &) = delete;
    CoremlStep & operator=(const CoremlStep &) = delete;

    // The warm-up prediction at position 0 (the constructor's, by default).
    void warm_up();
    // Positions the state holds (voice + text + generated frames).
    int context() const { return context_; }
    const std::string & label() const { return label_; }

    // Copies cache rows (FlowLM::cache_rows) into the state at their positions.
    void write_rows(const CacheRows & rows);
    // Replaces the whole state with a zeroed one, then writes `rows`. Needed
    // after a failed prediction: its blended write can leave non-finite values
    // in any row, and neither a later blend (0 * NaN) nor the attention mask
    // clears them.
    void reset(const CacheRows & rows);
    // reset(voice) for the first chunk of the Engine that owns this model (the
    // voice rows never change and no frame writes below them).
    void prime(const CacheRows & voice) { reset(voice); primed_ = true; }
    bool primed() const { return primed_; }

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
    bool primed_ = false;
};

// Mimi decoder geometry; must match scripts/pocket_coreml_mimi.py.
struct CoremlDecoderGeometry {
    static constexpr int dim = 512, latent = 32, heads = 8, head_dim = 64, layers = 2;
    static constexpr int keep = 249;            // retained attention rows (window - 1)
    static constexpr int upsample = 16;         // transformer rows per latent frame
    static constexpr int frame_samples = 1920;  // PCM samples per latent frame
    static constexpr int frames = 1;            // latent frames per prediction
};

class CoremlDecoder {
public:
    // Loads the sidecar and checks the shape of every input and output it
    // uses, its schema and that it was exported from the checkpoint
    // `source_hash` names, then runs one warm-up prediction (a sidecar with
    // any further feature fails it) and resets. Throws std::runtime_error.
    CoremlDecoder(const std::string & path, const std::string & compute_units, const std::string & source_hash);
    // A stream on a decoder model loaded and checked elsewhere (load_decoder_model;
    // not owned). The caller serializes predictions on a shared model.
    CoremlDecoder(moss_coreml_model * borrowed, std::string label);
    ~CoremlDecoder();
    CoremlDecoder(const CoremlDecoder &) = delete;
    CoremlDecoder & operator=(const CoremlDecoder &) = delete;

    const std::string & label() const { return label_; }
    // Starts an independent stream (Mimi::reset_decoder()).
    void reset();
    // latents [frames, 32], any frame count -> frames * 1920 PCM samples,
    // continuing the stream. Throws on a failed prediction or a non-finite
    // output; the stream is then unusable until reset().
    std::vector<float> decode(const std::vector<float> & latents);

private:
    void predict(const float * latent, float * pcm);

    moss_coreml_model * model_ = nullptr;
    bool owned_ = true;
    std::string label_ = "coreml";
    int64_t position_ = 0;
    std::vector<float> keys_, values_, cos_, sin_, mask_, row_cos_, row_sin_;
    std::vector<std::vector<float>> histories_;
};

// Loads a decoder model and checks it as the CoremlDecoder constructor does
// (shapes, schema, source hash), without a warm-up. Throws std::runtime_error.
// The caller frees it with moss_coreml_free.
moss_coreml_model * load_decoder_model(const std::string & path, const std::string & compute_units,
                                       const std::string & source_hash);

// ---- the sidecars on the shared Core ML runtime ----------------------------------

enum class CoremlStage { step, decoder };

// One stage's route, decided from options, OS, build, template match and the
// presence of a prebuilt sidecar, never from the cache, so it is the same
// before and after the first generation. For the generated route it carries
// the Core ML memory envelope fit charges.
struct CoremlRoute {
    std::string route = "none";  // prebuilt | generate | none
    std::string state;           // when none: unsupported | disabled | absent | no_template
    std::string reason;
    std::string compute_units;   // resolved
    bool strict = false;
    bool validated = false;      // generate: the envelope is validated for compute_units
    // generate: the envelope's parts before the margin, from the template.
    uint64_t resident_bytes = 0;      // weights, 2 x I/O, state and workspace
    uint64_t workspace_bytes = 0;     // the prediction workspace (included in resident_bytes)
    uint64_t prepare_peak_bytes = 0;  // generation and compile transient
    uint64_t load_peak_bytes = 0;     // load and warm-up transient for compute_units
    uint64_t transient_bytes = 0;     // max(prepare peak, load peak)
    double margin = 1.25;
    // generate: the free memory a cold Neural Engine load waits for (runtime
    // admission; fit does not charge it). 0 when compute_units has no ANE.
    uint64_t ane_compile_reserve_bytes = 0;
};

// The Core ML options of one stage: EngineOptions::coreml, with the stage's
// own compute units (coreml_step_units / coreml_decoder_units) and the legacy
// coreml_compute_units / coreml_strict as fallbacks for its unset fields.
CoremlOptions coreml_stage_options(const struct CoremlEngineOptions & opts, CoremlStage stage);

// What the Engine passes: the options and paths of both stages.
struct CoremlEngineOptions {
    std::string flow_lm_path, mimi_path;
    CoremlOptions coreml;
    std::string step_units, decoder_units, legacy_units;
    std::optional<bool> legacy_strict;
};

// From the Engine's options (EngineOptions::coreml, coreml_step_units,
// coreml_decoder_units, coreml_compute_units, coreml_strict).
CoremlEngineOptions coreml_engine_options(const EngineOptions & opts);

// Plans one stage. Throws std::invalid_argument for invalid explicit options,
// including an explicit cpu_and_ane for the step (Core ML state cannot be
// placed on the Neural Engine, so that load would fail after seconds).
CoremlRoute coreml_plan_route(CoremlStage stage, const CoremlEngineOptions & opts);

// The Core ML envelope of the generated stages of `routes`: their resident
// parts summed plus the largest transient (one Core ML job runs at a time per
// library), with the larger margin. 0 when no route is generate.
uint64_t coreml_envelope_bytes(const CoremlRoute & step, const CoremlRoute & decoder);

// Fit's two verdicts for a projection `required` against `free` bytes: the
// ggml-only one and the Core ML-inclusive one (plus `coreml_bytes` when a stage
// is generated). With a budget (CoremlOptions::memory_budget_bytes), a
// Core ML-inclusive projection over min(free, budget) that the ggml one fits
// degrades to ggml (use = off).
struct CoremlFitVerdict {
    bool ggml_only_fits = true;
    bool coreml_fits = true;
    bool degraded = false;
    bool fits = true;
    uint64_t charged = 0;
};
CoremlFitVerdict decide_coreml_fit(uint64_t required, uint64_t coreml_bytes, bool generate, uint64_t free,
                                   const std::optional<int64_t> & budget);

// One chunk's step, acquired once at the chunk's start on the producer thread.
struct CoremlStepUse {
    std::shared_ptr<void> hold;  // the runtime handle: keeps the model for the chunk
    CoremlStep * step = nullptr; // null: this chunk's FlowLM runs on ggml
    const void * identity = nullptr;  // the model instance; a new one needs the voice rows
    bool pending = false;        // ggml only because the step was not ready yet
    std::string label;
    std::function<std::unique_lock<std::mutex>()> lock;   // held across each prediction
    std::function<void(const std::string &)> retire;      // after a failed prediction
};

// One chunk's Mimi decoder, acquired once at the chunk's start on the calling thread.
struct CoremlDecoderUse {
    std::shared_ptr<void> hold;
    moss_coreml_model * model = nullptr;  // null: this chunk's Mimi runs on ggml
    bool pending = false;
    std::string label;
    std::function<std::unique_lock<std::mutex>()> lock;
    std::function<void(const std::string &)> retire;
};

// Both stages of one Engine on the shared Core ML runtime.
class CoremlStages {
public:
    // Resolves each stage's options (with the POCKET_COREML_* environment),
    // plans its route, creates its slot and issues the construction group
    // (prewarm none: nothing until the first chunk), waiting for it with wait
    // = construct. `degrade` resolves use = off for both stages (fit's
    // memory-degrade). `min_step_context` is the step context the voice needs
    // (its frames plus one token and one frame); a smaller step fails to load.
    // Returns null in builds without TTS_CPP_COREML. Throws
    // std::invalid_argument for invalid options and std::runtime_error when
    // use = require cannot be met at construction.
    static std::unique_ptr<CoremlStages> create(const CoremlEngineOptions & opts, const FlowConfig & config,
                                                const std::string & source_hash, int min_step_context, bool degrade);
    ~CoremlStages();

    CoremlStepUse acquire_step(int frames, const std::atomic<bool> * cancel);
    CoremlDecoderUse acquire_decoder(int frames, const std::atomic<bool> * cancel);
    bool step_strict() const;
    bool decoder_strict() const;
    const CoremlRoute & step_route() const;
    const CoremlRoute & decoder_route() const;
    // Whether the stage's sidecar is loaded and not retired.
    bool step_ready() const;
    bool decoder_ready() const;
    // Status JSON (schema 1: engine, cache_dir, generation, stages).
    std::string status_json() const;
    // Blocks until both requested stages are ready or terminal, `timeout_ms`
    // passes (< 0: no limit) or `cancel` is set. True when every stage with a
    // route is ready. Returns false at once while a stage is idle (nothing
    // requested it yet) or deferred(memory).
    bool wait(int64_t timeout_ms, const std::atomic<bool> * cancel) const;

    struct impl;
private:
    explicit CoremlStages(std::unique_ptr<impl> p);
    std::unique_ptr<impl> p_;
};

} // namespace tts_cpp::pocket::detail
