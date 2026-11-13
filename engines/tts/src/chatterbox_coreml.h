#pragma once
// Chatterbox S3Gen Core ML sidecars (TTS_CPP_COREML builds, Apple only).
//
// scripts/export-chatterbox-coreml.py exports three stages of an S3Gen GGUF
// next to it, named by the shared rule (coreml_sidecar_path.h):
//
//   chatterbox-s3gen-q8_0.gguf -> chatterbox-s3gen-encoder.mlmodelc
//                                 chatterbox-s3gen-cfm.mlmodelc
//                                 chatterbox-s3gen-hift.mlmodelc
//
// The encoder and CFM estimator are multi-function models, one function per
// token ("tokens_<T>") or frame ("frames_<F>") width; a call takes the
// narrowest function that holds it and masks the padding. HiFT is one fixed
// window walked over the utterance with HIFT_CONTEXT_FRAMES of context on
// each side. Tokenization, the time embedding, the Euler update, the F0
// predictor, SineGen, STFT and inverse STFT always run on ggml.
//
// Every function is a slot of the shared Core ML runtime (engines/coreml): it
// loads in the background, in priority order, one load at a time, and a stage
// runs on ggml until its function is ready. Each S3Gen call acquires its
// models once, at its start (an S3genCoremlPlan), and keeps them for the
// whole call, so readiness never changes a stage's backend mid-call. This
// header exposes no runtime types: tests drive it without the runtime's
// headers.

#include "tts-cpp/coreml_options.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

struct S3genCoremlPlan;
struct s3gen_synthesize_opts;

namespace tts_cpp::chatterbox::detail {

constexpr const char * COREML_ENV_PREFIX = "CHATTERBOX";
constexpr const char * COREML_DISABLE_ENV = "CHATTERBOX_COREML_DISABLE";
constexpr const char * COREML_STRICT_ENV = "CHATTERBOX_COREML_STRICT";
constexpr const char * COREML_UNITS_ENV = "CHATTERBOX_COREML_COMPUTE_UNITS";
// CPU + Neural Engine: measured fastest for every stage on an M2, and it keeps
// the GPU free for T3.
constexpr const char * COREML_DEFAULT_UNITS = "cpu_and_ane";
constexpr const char * COREML_ENCODER_SUFFIX = "-encoder";
constexpr const char * COREML_CFM_SUFFIX = "-cfm";
constexpr const char * COREML_HIFT_SUFFIX = "-hift";
constexpr const char * COREML_ENCODER_PREFIX = "tokens_";
constexpr const char * COREML_CFM_PREFIX = "frames_";
constexpr const char * GGML_STAGE_BACKEND = "ggml";
constexpr const char * MIXED_STAGE_BACKEND = "mixed";

// Support radius of a HiFT output frame in mel frames; must equal
// hift_receptive_frames() in scripts/chatterbox_coreml_models.py.
constexpr int HIFT_CONTEXT_FRAMES = 15;
constexpr int HIFT_STFT_PER_FRAME = 120;
constexpr int HIFT_STFT_BINS = 18;

// At most this many encoder and this many CFM functions are live at once
// outside prewarm = all (the resident cap). The least recently used one that
// no call holds and the current call does not predict is released first.
constexpr size_t COREML_RESIDENT_CAP = 3;

// s3gen_synthesize_to_wav's return code when a Core ML stage failed on a
// plan marked with defer_coreml_failures(): nothing was produced, the failing
// function is retired, and the caller renders the call again on its own
// thread (an Engine worker never falls back to ggml while T3 has the GPU).
constexpr int S3GEN_RC_COREML_REDO = 3;

// ---------------------------------------------------------------- options --

bool coreml_valid_units(const std::string & units);

// The legacy s3gen_synthesize_opts / EngineOptions fields as aliases of
// CoremlOptions: each fills the matching field only when it is unset.
// `units` non-empty -> compute_units; `strict` > 0 -> use = require, == 0 ->
// automatic, < 0 nothing; `disabled` -> use = off (overrides use).
CoremlOptions merge_legacy_coreml(CoremlOptions options, const std::string & units, int strict, bool disabled);

// The options of one s3gen_synthesize_to_wav call (opts.coreml plus the
// legacy aliases).
CoremlOptions call_coreml_options(const s3gen_synthesize_opts & opts);

// Throws std::invalid_argument for an explicit compute_units outside the four
// spellings or a negative strict_wait_ms / memory budget, in every build.
void validate_coreml_options(const CoremlOptions & options);

// ----------------------------------------------------------------- naming --

std::string coreml_stage_path(const std::string & s3gen_gguf, const char * suffix);

// True when this build has Core ML, the options leave it enabled and any
// stage sidecar sits next to the GGUF (presence only; nothing is loaded).
// This is the prebuilt route: fit decides from it alone, never from what is
// loaded or cached.
bool coreml_prebuilt_route(const std::string & s3gen_gguf, const CoremlOptions & options);

// "frames_512,frames_640" + "frames_" -> {512, 640} (sorted, unique).
std::vector<int> coreml_buckets(const std::string & function_names, const std::string & prefix);

// The narrowest bucket >= needed, or -1.
int coreml_pick_bucket(const std::vector<int> & buckets, int needed);

// One HiFT sidecar call: frames [start, start + window) go in, the output of
// frames [keep_begin, keep_end) is kept. Kept spans tile [0, frames).
struct hift_window {
    int start;
    int keep_begin;
    int keep_end;
};

std::vector<hift_window> coreml_hift_windows(int frames, int window, int context);

// Expected speech tokens T3 produces for `text_bytes` of text (~1.15 per
// byte: Turbo 1.17-1.21, Multilingual 0.95-1.05 on English), for predicting
// which widths a call needs.
int coreml_expected_speech_tokens(size_t text_bytes);

// Speech-token windows of the S3Gen calls a streamed sentence of
// `speech_tokens` makes (Engine::synthesize_streaming_segment's chunking:
// cumulative, or capped by `left_context` when > 0).
std::vector<int> coreml_stream_windows(int speech_tokens, int first_chunk, int chunk, int left_context);

// --------------------------------------------------------- shapes and cost --

// The encoder tokens and CFM frames of one S3Gen call (voice prompt included).
struct s3gen_call_shape {
    int tokens = 0;  // encoder input tokens: prompt + speech + lookahead silence
    int frames = 0;  // CFM frames: tokens * token_mel_ratio, minus the streaming trim
};

// Lookahead silence tokens a finalizing call appends (pre_lookahead_len).
constexpr int S3GEN_LOOKAHEAD_TOKENS = 3;

// What a call's shape and cost depend on, from the S3Gen GGUF's metadata only
// (no weights are read; cached per file).
struct s3gen_gguf_facts {
    int token_mel_ratio = 2;
    int speech_vocab = 6561;
    int builtin_prompt_tokens = 0;  // the built-in voice's prompt_token length
    bool meanflow = true;           // Turbo; false: Multilingual
};

s3gen_gguf_facts s3gen_read_facts(const std::string & s3gen_gguf);

// The shape of a call on `speech_tokens` new tokens (already filtered to the
// speech vocabulary) after `prompt_tokens` of voice prompt: the arithmetic
// s3gen_synthesize_to_wav itself uses.
s3gen_call_shape s3gen_shape(int prompt_tokens, int speech_tokens, int token_mel_ratio, bool append_lookahead,
                             bool finalize);

// The shape of s3gen_synthesize_to_wav(speech_tokens, opts): the voice prompt
// by the call's own precedence (prompt_token view, override, ref_dir, else
// the built-in voice when no conditioning is given), the speech tokens inside
// the vocabulary, the lookahead silence and the streaming trim.
s3gen_call_shape s3gen_call_shape_for(const s3gen_synthesize_opts & opts, const std::vector<int32_t> & speech_tokens,
                                      const s3gen_gguf_facts & facts);

// Estimated ggml (Metal) wall time of one S3Gen call of `frames` CFM frames:
// a per-variant linear fit over measured calls (an M2, default CFM steps),
// clamped at zero. `meanflow` is Turbo; otherwise Multilingual.
double s3gen_estimate_metal_ms_frames(bool meanflow, int frames);

// ----------------------------------------------------------- the sidecars --

enum class s3gen_stage { encoder = 0, cfm = 1, hift = 2 };
constexpr int S3GEN_STAGES = 3;
const char * to_string(s3gen_stage stage);

// The sidecar slots of one S3Gen GGUF under one set of resolved options.
// Shared through a process-wide table (one set per GGUF and options), dropped
// with the S3Gen cache (s3gen_unload). Dropping a set never waits: unstarted
// loads are cancelled, a started one finishes in the background and its
// model is released.
class s3gen_sidecars {
public:
    // The shared set, or null when Core ML is not built, the options disable
    // it, or no sidecar is present.
    static std::shared_ptr<s3gen_sidecars> get(const std::string & s3gen_gguf, const CoremlOptions & options);
    // An unshared set (tests), or null as for get().
    static std::shared_ptr<s3gen_sidecars> create(const std::string & s3gen_gguf, const CoremlOptions & options);
    // Drops every shared set (the S3Gen cache release).
    static void release_all();

    ~s3gen_sidecars();
    s3gen_sidecars(const s3gen_sidecars &) = delete;
    s3gen_sidecars & operator=(const s3gen_sidecars &) = delete;

    const s3gen_gguf_facts & facts() const;
    bool strict() const;
    // Widths of a multi-function stage (empty for HiFT or an absent stage).
    std::vector<int> buckets(s3gen_stage stage) const;
    // A slot exists for the stage (its sidecar is present and listed).
    bool has(s3gen_stage stage) const;

    // Issues the construction group for the resolved prewarm scope: nothing
    // (none), the minimal set (minimal, predicted: HiFT plus the encoder and
    // CFM widths with a warm hint on this host, at most the three most recent
    // per stage) or every function (all). With wait = construct, blocks until
    // the group finished.
    void start_construction(const std::atomic<bool> * cancel = nullptr);
    std::string construction_json() const;

    // A call is starting: request the widths `calls` need at `current`
    // priority (ahead of unstarted construction loads) and mark them, plus
    // HiFT, predicted, so the resident cap never releases them.
    void predict(const std::vector<s3gen_call_shape> & calls);
    // The call ended: nothing is predicted any more.
    void end_call();

    // Acquires the models one call of `shape` runs on (see
    // s3gen_acquire_coreml_plan).
    std::shared_ptr<S3genCoremlPlan> acquire(const s3gen_call_shape & shape,
                                             std::chrono::steady_clock::time_point deadline,
                                             const std::atomic<bool> * cancel = nullptr);

    // Another prewarm group: scope none/minimal/predicted (the minimal set)
    // or all. Returns the group result as JSON (after it finished when `wait`).
    std::string prewarm(CoremlOptions::Prewarm scope, bool wait, const std::atomic<bool> * cancel = nullptr);
    // Waits for the construction group; true when every slot reached ready.
    // timeout_ms < 0: no limit.
    bool wait(int64_t timeout_ms, const std::atomic<bool> * cancel = nullptr);
    std::string status_json() const;
    // Live model instances of a stage's resident pool (0 for HiFT or
    // without a cap).
    size_t live(s3gen_stage stage) const;
    // The state label of one function's slot ("ready", "loading", ...; bucket
    // 0 for HiFT), or "" when the stage has no such function.
    std::string state(s3gen_stage stage, int bucket) const;

    struct impl;

private:
    explicit s3gen_sidecars(std::unique_ptr<impl> p);
    std::unique_ptr<impl> p_;
};

// ---------------------------------------------------------------- the plan --

// What a plan holds for one stage.
struct plan_stage_view {
    bool present = false; // the stage has a sidecar (it may still be unusable)
    bool model = false;   // a Core ML model was acquired
    int bucket = -1;      // its width (0 for HiFT), else -1
    bool pending = false; // ggml only because the function was not ready yet
    std::string label;    // coreml-* with a model, else "ggml"
};

plan_stage_view plan_view(const S3genCoremlPlan & plan, s3gen_stage stage);
// Any stage runs on ggml only because its function was not ready.
bool plan_pending(const S3genCoremlPlan & plan);
// A failing prediction of the plan's models makes s3gen_synthesize_to_wav
// return S3GEN_RC_COREML_REDO instead of falling back to ggml in place. Set
// before the plan is handed to another thread.
void defer_coreml_failures(const S3genCoremlPlan & plan);
bool coreml_failures_deferred(const S3genCoremlPlan & plan);
bool plan_strict(const S3genCoremlPlan & plan);

// Thrown by the plan_* prediction helpers when a Core ML prediction fails;
// the caller retires the function (plan_retire) and decides what runs next.
struct coreml_stage_failure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Retires the stage's model process-wide after a failed prediction.
void plan_retire(const S3genCoremlPlan & plan, s3gen_stage stage, const std::string & reason);

// Runs one stage on the plan's model under its prediction lock. Each returns
// false when the plan holds no model for the stage or it does not cover the
// shape (ggml runs it); a shape the sidecar does not declare or a failed
// prediction throws coreml_stage_failure.
//   encode:   embedding [tokens, 512] row-major -> mu [tokens * ratio, 80]
//   velocity: `batch` planes of [80, frames] (x, mu, cond), spks batch x 80,
//             t_emb batch x 1024 -> velocity planes [80, frames]; `step` is
//             the Euler step (test hooks)
//   vocode:   mel [80, frames], source_stft [18, frames * 120 + 1] ->
//             conv_post rows [18, frames * 120 + 1]
bool plan_encode(const S3genCoremlPlan & plan, const std::vector<float> & embedding, int tokens, int ratio,
                 std::vector<float> & mu);
bool plan_velocity(const S3genCoremlPlan & plan, size_t step, int batch, int frames,
                   const std::vector<const float *> & x, const std::vector<const float *> & mu,
                   const std::vector<const float *> & cond, const std::vector<const float *> & spks,
                   const std::vector<const float *> & t_emb, std::vector<std::vector<float>> & out);
bool plan_vocode(const S3genCoremlPlan & plan, const std::vector<float> & mel, int frames,
                 const std::vector<float> & source_stft, std::vector<float> & spectrum);

// --------------------------------------------------------- CFM integration --

// One Euler step of the CFM integration: advances `z` in place at step `s`.
// `on_coreml` selects the backend; a Core ML failure throws
// coreml_stage_failure. A non-zero return (cancelled, an error) stops the
// integration and is returned.
using cfm_step_fn = std::function<int(size_t s, std::vector<float> & z, bool on_coreml)>;

// Integrates `z` over `steps` Euler steps. The backend is latched at the
// start: with `on_coreml` every step runs on Core ML until one fails; then
// `on_failure` runs (it retires the function, or throws to stop), `z` is
// restored to its initial value and every step runs again on ggml with the
// same schedule and conditioning, before any mel is produced. No step draws
// random numbers, so the result equals a whole ggml integration from the
// same initial noise. Returns 0 or the first non-zero step result.
int cfm_integrate(std::vector<float> & z, size_t steps, bool on_coreml, const cfm_step_fn & step,
                  const std::function<void(const coreml_stage_failure &, size_t s)> & on_failure);

// ------------------------------------------------------------ test hooks --

// Test-only controls of the sidecar loader and predictions. Production code
// leaves them at their defaults.
struct coreml_test_hooks {
    // Every sidecar function load sleeps this long first (a slow, cold load).
    std::atomic<int> load_delay_ms{0};
    // While true, every sidecar function load waits before loading.
    std::atomic<bool> hold_loads{false};
    // The Core ML CFM prediction of this Euler step (0-based) fails once.
    std::atomic<int> fail_cfm_step{-1};
    // The next prediction of this stage fails once (-1: none; else an
    // s3gen_stage value).
    std::atomic<int> fail_stage{-1};
    // Loads started, as "<stage>/<function>", in order.
    std::vector<std::string> loads() const;
    void clear_loads();
    void record_load(const std::string & what);

private:
    mutable std::mutex mu_;
    std::vector<std::string> loads_;
};

coreml_test_hooks & coreml_test();

} // namespace tts_cpp::chatterbox::detail
