#pragma once
#include "pocket/memory.h"
#include <memory>
#include <string>
#include <vector>

namespace tts_cpp::pocket::detail {
class Mimi {
public:
    explicit Mimi(const std::string & path, int threads = 2);
    ~Mimi();
    Mimi(const Mimi &) = delete;
    Mimi & operator=(const Mimi &) = delete;
    static MemoryMeasure measure(const std::string & path, int threads, bool include_encoder);
    // Independent decoder/encoder histories. Decoder accepts [T,latent_dim]
    // and emits mono PCM, retaining causal state between chunks.
    void reset_decoder();
    std::vector<float> decode(const std::vector<float> & latents);
    // A whole mono reference at the native sample rate; zero-padded to the
    // next frame. Each call starts an independent encoding session.
    std::vector<float> encode(const std::vector<float> & pcm);
    int sample_rate() const;
    int frame_samples() const;
    int latent_dim() const;
    const std::string & source_hash() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace tts_cpp::pocket::detail
