#pragma once
#include "tts-cpp/export.h"
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tts_cpp::moss {

struct TranscribeOptions {
    std::string model_path;
    int n_threads = 4;
    bool use_gpu = false;
    std::string backends_dir;
};

struct TranscribeRequest {
    std::string prompt;
    int max_new_tokens = 0;
};

struct TranscriptSegment {
    double start_s = 0;
    double end_s = 0;
    std::string speaker;
    std::string text;
};

struct TranscribeResult {
    std::string text;
    std::vector<TranscriptSegment> segments;
    bool cancelled = false;
    int audio_tokens = 0;
    int prompt_tokens = 0;
    int generated_tokens = 0;
    double encode_ms = 0;
    double prefill_ms = 0;
    double decode_ms = 0;
};

using TranscribeProgress = std::function<bool(int generated_tokens, int max_new_tokens)>;

class TTS_CPP_API TranscribeEngine {
public:
    explicit TranscribeEngine(const TranscribeOptions & options);
    ~TranscribeEngine();
    TranscribeEngine(const TranscribeEngine &) = delete;
    TranscribeEngine & operator=(const TranscribeEngine &) = delete;
    TranscribeResult transcribe(const float * pcm, size_t samples, int sample_rate,
                                const TranscribeRequest & request = {}, const TranscribeProgress & progress = {});
    void cancel() noexcept;
    int sample_rate() const noexcept;
    const char * backend_name() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tts_cpp::moss
