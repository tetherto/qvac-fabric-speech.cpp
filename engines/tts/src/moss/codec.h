#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tts_cpp::moss::detail {

// One instance owns either the encoder or the decoder GGUF of the MOSS audio
// codec (RVQ + patched transformer stack). Graphs are built per call because
// the frame count varies with the clip; calls must be serialized by the
// caller. The encoder consumes a whole mono clip and returns row-major
// [frames, num_quantizers] codes; the decoder consumes those codes and
// returns mono PCM at sample_rate().
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
    std::vector<float> decode(const std::vector<int32_t> & codes, int n_channels = 0);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tts_cpp::moss::detail
