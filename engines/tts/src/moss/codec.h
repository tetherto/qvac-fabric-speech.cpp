#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tts_cpp::moss::detail {

class Codec {
public:
    Codec(const std::string & path, bool use_gpu, int n_threads);
    ~Codec();
    Codec(const Codec &) = delete;
    Codec & operator=(const Codec &) = delete;

    bool is_encoder() const;
    int sample_rate() const;
    int num_quantizers() const;
    int samples_per_frame() const;
    const char * backend_name() const;

    std::vector<int32_t> encode(const std::vector<float> & pcm);
    std::vector<float> decode(const std::vector<int32_t> & codes);

    void begin_decode_stream();
    std::vector<float> decode_stream(const std::vector<int32_t> & codes);
    int64_t frames_decoded() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::vector<float> decode_in_windows(Codec & codec, const std::vector<int32_t> & codes,
                                     int64_t window_frames);
std::vector<float> decode_segments(Codec & codec, const std::vector<std::vector<int32_t>> & segments,
                                   int64_t max_single_frames);

} // namespace tts_cpp::moss::detail
