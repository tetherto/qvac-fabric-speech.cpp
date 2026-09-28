#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tts_cpp::moss::detail {

struct SpeechVqConfig {
    int sample_rate = 0;
    int n_fft = 0;
    int hop_length = 0;
    int n_mels = 0;
    int chunk_samples = 0;
    int n_layers = 0;
    int n_embd = 0;
    int n_ff = 0;
    int n_heads = 0;
    int n_ctx = 0;
    int pooling_kernel = 0;
    int codebook_size = 0;
    int samples_per_token() const;
};

size_t speech_segment_tokens(const SpeechVqConfig & config, size_t samples);
size_t speech_padded_samples(const SpeechVqConfig & config, size_t samples);
void normalize_whisper_log_mel(std::vector<float> & mel);

class SpeechTokenizer {
public:
    SpeechTokenizer(const std::string & codec_path, bool use_gpu, int n_threads);
    ~SpeechTokenizer();
    SpeechTokenizer(const SpeechTokenizer &) = delete;
    SpeechTokenizer & operator=(const SpeechTokenizer &) = delete;

    const SpeechVqConfig & config() const;
    std::vector<float> log_mel(const float * samples, size_t count) const;
    std::vector<int32_t> encode_segment(const float * samples, size_t count);
    std::vector<int32_t> encode(const std::vector<float> & pcm_16k);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tts_cpp::moss::detail
