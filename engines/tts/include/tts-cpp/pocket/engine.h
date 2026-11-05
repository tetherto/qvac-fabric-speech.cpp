#pragma once
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
    // Core ML step sidecar policy (Apple, TTS_CPP_COREML builds), resolved at
    // construction without modifying the environment. Empty inherits
    // POCKET_COREML_COMPUTE_UNITS (default "cpu_and_gpu"); explicit values must
    // be all, cpu_and_gpu, cpu_only or cpu_and_ane (which cannot hold the
    // sidecar's state, so it falls back) and invalid ones throw even without
    // Core ML. Unset coreml_strict inherits POCKET_COREML_STRICT (presence means
    // true): a present sidecar that cannot load or serve a chunk is then an
    // error instead of a ggml fallback. POCKET_COREML_DISABLE keeps FlowLM on ggml.
    std::string coreml_compute_units;
    std::optional<bool> coreml_strict;
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
};
using AudioCallback = std::function<bool(const float *, size_t, int)>;

// Core ML (Apple, TTS_CPP_COREML builds): when the step sidecar sits next to
// the FlowLM GGUF (flow-lm.gguf -> flow-lm-step.mlmodelc, the quant tag
// dropped), the FlowLM transformer and first flow step of each generated frame
// run on it whatever the backend option, for every chunk that fits its
// context; text prefill, extra flow steps (steps > 1) and Mimi stay on ggml.
// See docs/pocket-tts.md "Core ML step sidecar".
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
    const char * device_name() const noexcept;
    std::string model_source_sha256() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
