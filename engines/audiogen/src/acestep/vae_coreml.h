#pragma once

// The ACE-Step VAE decoder on the shared Core ML runtime (engines/coreml).
//
// One slot per Engine (or standalone Vae), created at construction and never
// requested there unless prewarm is `all`: the decoder is requested by decode
// demand only (Vae::decode), never by an encoding. A prebuilt
// `<gguf stem minus quant tag>-decoder.mlmodelc` next to the VAE GGUF loads in
// the background; without one, and with a cache directory, the sidecar is
// generated from the GGUF's own decoder tensors (template acestep-vae-t64) on
// macOS 15 or later. Each decode acquires the slot once, with a deadline equal
// to the ggml decode time of its latent on this host, and runs entirely on what it
// acquired: Core ML windows, or the ggml decode when nothing was ready.
//
// Low-memory mode (the default) evicts the slot at every VAE stage boundary
// (Engine free_vae()): the loaded model is released, unstarted jobs are
// cancelled, and a cold load already running finishes, writes its warm hint and
// releases its model without repopulating the slot, so the next decode reloads
// warm.

#include "audiogen-cpp/coreml_options.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace tts_cpp::acestep::detail {

constexpr const char * VAE_COREML_ENV_PREFIX   = "ACESTEP";
constexpr const char * VAE_COREML_TEMPLATE_ID  = "acestep-vae-t64";
// The generated decoder's fixed window in latent frames (the template's
// input [1, 64, 64]): a shorter latent cannot run on it.
constexpr int          VAE_COREML_TEMPLATE_WINDOW = 64;
constexpr const char * VAE_GGML_BACKEND        = "ggml";
constexpr const char * VAE_MIXED_BACKEND       = "mixed";

// The bounded wait for a warm load follows the rule every engine uses: a
// decode waits for Core ML only while the wait fits the time the decode would
// take on ggml. That time is this host's own: once a stage has decoded a
// latent of at least one window on ggml, its deadline uses the fastest cost
// per latent frame it measured. Before that, it uses this low estimate of the
// GPU decode (ggml with Metal measures 27.6 ms per latent frame on an M2,
// 20.6-21.0 s for 750 frames, and 7.9-8.3 ms on an M4 Max), so a decode never
// waits longer for Core ML than the ggml decode it replaces would take.
constexpr double  VAE_GGML_DECODE_MS_PER_FRAME = 7.0;
constexpr int64_t VAE_GGML_DECODE_MS_FLOOR     = 100;
int64_t vae_ggml_decode_estimate_ms(int T_latent, double ms_per_frame = VAE_GGML_DECODE_MS_PER_FRAME);

// Tests: the ggml cost per latent frame every stage's deadline uses instead of
// its measurement or the estimate above (<= 0: none, the default).
void vae_coreml_set_ggml_ms_per_frame_override(double ms_per_frame);

// Label merge across the decodes of one call: equal labels stay, anything else
// is "mixed" (an empty accumulator takes the next label).
std::string vae_merge_backend(const std::string & acc, const std::string & next);

// Throws std::invalid_argument for an invalid explicit compute_units, a
// negative strict_wait_ms or a negative memory budget.
void vae_coreml_validate_options(const CoremlOptions & options);

// The decoder stage's route, decided from options, OS, build, template match
// and the presence of a prebuilt sidecar, never from the cache, so it is the
// same before and after the first generation. For the generated route it
// carries the template's Core ML memory envelope, which fit charges.
struct vae_coreml_route {
    std::string route = "none";  // prebuilt | generate | none
    std::string state;           // when none: unsupported | disabled | absent | no_template
    std::string reason;
    std::string compute_units;   // resolved
    bool        strict    = false;
    bool        validated = false;  // generate: the envelope is validated for compute_units
    // generate: the decoder's window in latent frames, known before it loads;
    // 0 when unknown until a load (prebuilt).
    int         window_frames = 0;
    // generate: the parts of the envelope with the template's margin applied.
    uint64_t resident_bytes  = 0;  // weights, 2 x I/O, workspace
    uint64_t transient_bytes = 0;  // max(prepare, load) transient for compute_units
    uint64_t envelope_bytes  = 0;  // resident + transient
    // generate: unmargined parts, for memory validation.
    uint64_t prepare_peak_bytes = 0;
    uint64_t load_peak_bytes    = 0;
    uint64_t workspace_bytes    = 0;
    // generate: the free memory a cold Neural Engine load waits for (runtime
    // admission; fit does not charge it). 0 when compute_units has no ANE.
    uint64_t ane_compile_reserve_bytes = 0;
};

vae_coreml_route vae_coreml_plan_route(const std::string & gguf_path, const CoremlOptions & options);

// One loaded decoder, held for one decode. decode() takes one time-major
// latent window (latent[t * 64 + c], window() frames) and writes interleaved
// stereo PCM (window() * 1920 * 2 floats). False on a prediction failure.
class vae_coreml_decoder {
public:
    virtual ~vae_coreml_decoder() = default;
    virtual int          window() const = 0;
    virtual const char * label() const  = 0;
    virtual bool         decode(const float * latent, float * pcm) = 0;
};

// What one decode runs on, acquired once at its start.
struct vae_coreml_call {
    std::shared_ptr<vae_coreml_decoder> decoder;  // null: this decode runs on ggml (unless cancelled)
    bool pending = false;  // on ggml only because the sidecar was not ready yet
    // The stage's cancel flag interrupted the acquisition: the decode returns
    // at once, on neither backend.
    bool cancelled = false;
    // A prediction failed: retire the sidecar (for every owner of it). The
    // decode is then redone on ggml, or fails under strict.
    std::function<void(const std::string & reason)> retire;
};

// Tests: called by the decoder's warm-up after the load and its warm
// prediction, before the runtime publishes the model (null: nothing, the
// default). It may block, which holds the load's completion until the test
// releases it, so a test can order the completion after an eviction.
using vae_coreml_completion_hook = void (*)();
void vae_coreml_set_completion_hook(vae_coreml_completion_hook hook);

// Tests: called by the release of an un-waited group's decoder (low-memory
// mode) with the stage still alive, just before it is released (null:
// nothing, the default). It may block, which holds that release, so a test can
// destroy the owner meanwhile.
using vae_coreml_release_hook = void (*)();
void vae_coreml_set_release_hook(vae_coreml_release_hook hook);

class vae_coreml_stage : public std::enable_shared_from_this<vae_coreml_stage> {
public:
    // Resolves `options` (with the ACESTEP_* environment) and plans the
    // route. Unless the route is none it creates the slot, requesting
    // nothing, and issues the construction group: empty unless prewarm is
    // all. With wait = construct it waits for that group and then, when
    // `release_after_construction` (the Engine's low-memory mode), evicts the
    // decoder again: the group recorded `ready` first, so it still reports
    // success. Without wait = construct, in that mode, a background thread
    // the decoder is released once the group has finished, unless a decode
    // has asked for it since; that release runs on one process-wide thread
    // that holds the stage only weakly, so destroying the stage (or its
    // Engine) never waits for it. `memory_degrade` resolves use = off (fit's
    // memory-degrade).
    // `cancel` (optional, must outlive the stage) interrupts waits.
    // In builds without AUDIOGEN_COREML the stage has route none
    // (unsupported(build)), no slot, and reports that decoder status and an
    // empty successful construction group. Throws std::invalid_argument for
    // invalid options and (with the runtime) std::runtime_error when use =
    // require cannot be met.
    static std::shared_ptr<vae_coreml_stage> create(const std::string & gguf_path, const CoremlOptions & options,
                                                    bool memory_degrade, bool release_after_construction,
                                                    const std::atomic<bool> * cancel = nullptr);
    ~vae_coreml_stage();
    vae_coreml_stage(const vae_coreml_stage &)             = delete;
    vae_coreml_stage & operator=(const vae_coreml_stage &) = delete;

    // Once per decode of `T_latent` frames: requests the decoder and acquires
    // it with a deadline of the ggml decode time of `T_latent` frames
    // (ggml_ms_per_frame()). A set
    // cancel flag returns `cancelled`. Throws std::runtime_error when use =
    // require and the sidecar cannot serve the decode (failed, retired, timed
    // out, deferred for memory, evicted or shut down while waiting).
    vae_coreml_call acquire(int T_latent);
    // A decode of `T_latent` frames that ran entirely on ggml in `ms`: one of
    // at least the decoder's window trains the stage's cost per frame.
    void record_ggml_decode(int T_latent, int64_t ms);
    // The ggml cost per latent frame the deadline uses: the test override,
    // else the fastest one measured, else VAE_GGML_DECODE_MS_PER_FRAME.
    double ggml_ms_per_frame() const;

    // Releases the loaded decoder and cancels unstarted jobs; a started load
    // runs on, writes its hint and never repopulates the slot.
    void evict();

    // Whether a ggml VAE whose weights were read through `fd` (-1: unknown)
    // loaded the file the decoder is generated from. The generated route's
    // slot takes the identity (fstat tuple) of the GGUF when it is made; a
    // load that read another file (the path was replaced since) must not
    // decode on that decoder, or the two backends would run different
    // weights. True for the prebuilt route and without a slot.
    bool serves_loaded_source(int fd) const;

    // Whether a decode of `T_latent` frames can run on the decoder at all:
    // false when it is shorter than the decoder's known window (such a decode
    // runs on ggml without requesting the decoder, and is not pending).
    bool eligible(int T_latent) const;

    bool strict() const;
    // Whether a slot exists (route != none).
    bool active() const;
    const vae_coreml_route & route() const;
    // The status JSON (schema 1: engine, cache_dir, generation, stages).
    std::string status_json() const;
    // The construction group's result JSON; with route none an empty group
    // (success, nothing_to_prepare, the decoder under skipped_route_none).
    std::string construction_json() const;
    // Issues another request group (kind prewarm) the way construction does:
    // scope all requests the decoder; every other scope gives an empty group
    // (ACE-Step's minimal set is empty). With `wait` it blocks until the group
    // finishes or `cancel` (optional) is set, then, when `release_after` (the
    // Engine's low-memory mode), releases the decoder; the group recorded
    // `ready` first, so it still reports success. Without `wait`, in that
    // mode, the decoder is released once the group has finished, as for the
    // construction group. Returns the group's result JSON (when not waited,
    // the snapshot at issue).
    std::string prewarm(CoremlOptions::Prewarm scope, bool wait, bool release_after,
                        const std::atomic<bool> * cancel = nullptr);
    // The process-wide release of an un-waited group: releases the decoder
    // unless a decode has asked for it since `demand` was taken.
    void release_unless_demanded(uint64_t demand);
    // Blocks until the decoder is ready or terminal, `timeout_ms` passes (< 0:
    // no limit) or `cancel` (optional) is set. True when it is ready. Returns
    // false at once while nothing has requested it (idle: before the first
    // decode, or evicted) or it is deferred(memory).
    bool wait(int64_t timeout_ms, const std::atomic<bool> * cancel = nullptr) const;

    struct impl;

private:
    explicit vae_coreml_stage(std::unique_ptr<impl> p);
    std::unique_ptr<impl> p_;
};

}  // namespace tts_cpp::acestep::detail
