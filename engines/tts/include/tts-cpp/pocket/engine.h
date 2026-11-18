#pragma once
#include "tts-cpp/coreml_options.h"
#include "tts-cpp/export.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace tts_cpp::pocket {
struct EngineOptions {
    std::string flow_lm_path, mimi_path, frontend_path;
    // Supply exactly one voice source. Prepared voices are checkpoint-specific.
    std::string voice_path;
    std::string reference_audio_path;
    int n_threads = 1;  // per worker; FlowLM and Mimi have separate workers
    int output_sample_rate = 24000; // 8000..192000 Hz, streaming sinc resampling
    int context = 2048;
    int max_tokens = 50;
    int steps = 1;
    float temperature = 0.3f;
    float noise_clamp = 0;  // 0 disables truncated-normal sampling
    float eos_threshold = -4;
    int frames_after_eos = -1;  // -1: checkpoint/text default; 0..100: explicit tail
    // Context validation reserves the complete EOS tail before emitting PCM.
    uint32_t seed = 1234;
    // CPU is the default. GPU requests fail if no supported GPU is available.
    bool use_gpu = false;
    // Empty/auto follows use_gpu; cpu/opencl/webgpu/hexagon (HTP0) explicitly selects the backend.
    std::string backend;
    std::string backends_dir;
    // Core ML sidecar policy (Apple, TTS_CPP_COREML builds), resolved at
    // construction without modifying the environment. Empty inherits
    // POCKET_COREML_COMPUTE_UNITS, else each sidecar's default: "cpu_and_gpu"
    // for the FlowLM step, "cpu_and_ane" for the Mimi decoder. A value (all,
    // cpu_and_gpu, cpu_only or cpu_and_ane) applies to both; the step's state
    // cannot be held by cpu_and_ane, so the step then falls back. Invalid values
    // throw even without Core ML. Unset coreml_strict inherits
    // POCKET_COREML_STRICT (presence means true): a present sidecar that cannot
    // load or serve a chunk is then an error instead of a ggml fallback.
    // POCKET_COREML_DISABLE keeps FlowLM and Mimi on ggml.
    std::string coreml_compute_units;
    std::optional<bool> coreml_strict;

    // Both sidecars on the shared Core ML runtime (tts-cpp/coreml_options.h):
    // cache_dir enables generation from the GGUFs, wait / prewarm choose when
    // they load, use = require makes a sidecar that cannot serve a chunk an
    // error. Unset fields fall back to the per-stage units below, then to
    // coreml_compute_units / coreml_strict above, then to the POCKET_COREML_*
    // environment. compute_units applies to both stages; an explicit
    // cpu_and_ane for the step throws (Core ML state cannot be placed on the
    // Neural Engine). Pocket's minimal set is both sidecars.
    CoremlOptions coreml;
    // Per-stage compute units; override coreml.compute_units for their stage.
    std::string coreml_step_units;
    std::string coreml_decoder_units;

    // The workload and headroom of the fit_params call that assessed this
    // engine (FitOptions::text and FitOptions::margin_bytes; the defaults are
    // one short sentence and FitOptions' margin). With
    // coreml.memory_budget_bytes set, the constructor repeats that fit to
    // decide memory-degrade (Core ML off when only the ggml projection fits
    // the budget), so these must match the assessment, or the engine can keep
    // Core ML where that fit dropped it.
    std::string fit_text = "Hello.";
    uint64_t fit_margin_bytes = 256ull * 1024 * 1024;
};
struct SynthesisResult {
    std::vector<float> pcm; // populated by synthesize(); streaming uses the callback
    int sample_rate = 24000;
    bool cancelled = false;
    int generated_frames = 0;
    double first_audio_ms = 0, generation_ms = 0;
    // Where this call's FlowLM frames ran: "ggml" (on flow_lm_backend_name()),
    // the Core ML step sidecar's label ("coreml-gpu", ...) when every frame's
    // transformer and flow step ran on it, or "mixed" (some chunks on ggml, a
    // sidecar retired mid-call, or steps > 1, whose flow runs on ggml).
    std::string flow_lm_backend = "ggml";
    // Where this call's Mimi decoding ran: "ggml" (on mimi_backend_name()), the
    // Core ML decoder's label ("coreml-ane", ...) for every frame, or "mixed"
    // (a decoder retired mid-call).
    std::string mimi_backend = "ggml";
    // True when a chunk of this synthesis ran a stage on ggml only because its
    // Core ML sidecar was not ready yet (still generating, compiling or loading).
    bool coreml_pending = false;
};
using AudioCallback = std::function<bool(const float *, size_t, int)>;

// Core ML (Apple, TTS_CPP_COREML builds), whatever the backend option: the
// FlowLM transformer and first flow step of each generated frame run on the
// step sidecar for every chunk that fits its context (text prefill and extra
// flow steps, steps > 1, stay on ggml), and Mimi decoding runs on the decoder
// sidecar. Each sidecar is either prebuilt next to its GGUF (flow-lm.gguf ->
// flow-lm-step.mlmodelc, mimi.gguf -> mimi-decoder.mlmodelc, the quant tag
// dropped) or, with EngineOptions::coreml.cache_dir set and macOS 15 or later,
// generated from the GGUF into that cache. Both load in the background: until
// a sidecar is ready its stage runs on ggml (SynthesisResult::coreml_pending),
// and coreml.wait = construct makes the constructor wait instead. Each chunk
// takes its backends once, at its start. See docs/pocket-tts.md "Core ML
// sidecars".
// Persistent native CPU, GPU or Hexagon engine. One synthesis at a time per instance;
// overlapping/reentrant calls throw. cancel() is safe from another thread.
// Streaming callbacks run on the calling thread, with valid PCM until return.
// Returning false cancels the request. Each new request resets the voice,
// codec and RNG. The seed is portable within Fabric, not PyTorch-equivalent.
class TTS_CPP_API Engine {
public:
    explicit Engine(const EngineOptions & options);
    ~Engine();
    Engine(const Engine &) = delete;
    Engine & operator=(const Engine &) = delete;
    SynthesisResult synthesize(const std::string & text);
    SynthesisResult synthesize_stream(const std::string & text, const AudioCallback & callback);
    void cancel() noexcept;
    int sample_rate() const noexcept;
    const char * backend_name() const noexcept;
    const char * flow_lm_backend_name() const noexcept;
    const char * mimi_backend_name() const noexcept;
    // True while a Core ML step sidecar is loaded and not retired by a failure.
    bool flow_lm_on_coreml() const noexcept;
    // True while a Core ML Mimi decoder is loaded and not retired by a failure.
    bool mimi_on_coreml() const noexcept;
    // Core ML status of both stages as JSON (schema 1: engine, cache_dir,
    // generation, stages with state, route, compute units, label, artifact
    // key, timings and error); "{}" in builds without TTS_CPP_COREML.
    std::string coreml_status() const;
    // Waits until every Core ML stage with a sidecar is ready or one has
    // failed, `timeout_ms` passes (< 0: no limit) or cancel() is called. True
    // when all are ready. Returns false at once while a stage has not been
    // requested yet (coreml.prewarm none before the first synthesis) or waits
    // for memory (deferred). coreml_wait(0) only polls; a blocking wait never
    // clears a cancel() meant for a synthesis on another thread.
    bool coreml_wait(int64_t timeout_ms = -1) const;
    const char * device_name() const noexcept;
    std::string model_source_sha256() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
