#pragma once
#include "tts-cpp/export.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tts_cpp::moss {

struct EngineOptions {
    std::string backbone_path;
    std::string decoder_path;
    // Required only for voice cloning; leave empty for direct generation.
    std::string encoder_path;
    // Mono or multi-channel WAV at the codec sample rate; channels are
    // averaged. There is no resampling: a mismatched rate is rejected.
    std::string reference_audio_path;
    std::string language = "zh";
    int n_threads = 4;
    int context = 4096;
    int max_new_tokens = 2048;
    bool use_gpu = false;
    float text_temperature  = 1.5f;
    float text_top_p        = 1.0f;
    int   text_top_k        = 50;
    float audio_temperature = 1.7f;
    float audio_top_p       = 0.8f;
    int   audio_top_k       = 25;
    float audio_repetition_penalty = 1.0f;
    uint32_t seed = 1234;
};

struct SynthesisResult {
    std::vector<float> pcm;
    int sample_rate = 0;
    bool cancelled = false;
    int generated_frames = 0;
    double generation_ms = 0;
    double decode_ms = 0;
};

// Persistent MOSS Delay TTS engine (MOSS-TTS-v1.5 / MOSS-TTSD checkpoints).
// One synthesis at a time per instance; overlapping calls throw. cancel() is
// safe from another thread and makes the active synthesis return with
// cancelled = true. Each request reseeds the RNG from the options, so equal
// requests on one instance produce equal audio.
class TTS_CPP_API Engine {
public:
    explicit Engine(const EngineOptions & options);
    ~Engine();
    Engine(const Engine &) = delete;
    Engine & operator=(const Engine &) = delete;
    SynthesisResult synthesize(const std::string & text);
    void cancel() noexcept;
    int sample_rate() const noexcept;
    const char * backend_name() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tts_cpp::moss
