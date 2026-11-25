#pragma once

// Test and benchmark helper: the Core ML encoder sidecar of a model through
// the engine's runtime path (TranscribeCoremlEncoder), loaded before the
// first window, and one window encoded on the route it gives.

#include "moss/transcribe_audio.h"
#include "moss/transcribe_coreml.h"
#include "moss/transcribe_model.h"
#include "moss/transcribe_networks.h"

#include "mel_preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace moss_transcribe_coreml {

using parakeet::moss::detail::TranscribeConfig;
using parakeet::moss::detail::TranscribeCoremlEncoder;

// Waits for the sidecar (wait = construct). `cache_dir` empty: the adjacent
// bundle only (generation off); otherwise generated or cached there when no
// bundle sits beside the GGUF. Core ML is required: a sidecar that cannot
// serve the first window throws there.
inline std::unique_ptr<TranscribeCoremlEncoder> ready_encoder(const std::string & model_path,
                                                             const TranscribeConfig & config,
                                                             const std::string & cache_dir = {}) {
    parakeet::CoremlOptions options;
    options.use = parakeet::CoremlOptions::Use::require;
    options.wait = parakeet::CoremlOptions::Wait::construct;
    if (cache_dir.empty()) {
        options.generate = parakeet::CoremlOptions::Generate::never;
    } else {
        options.cache_dir = cache_dir;
    }
    auto encoder = std::make_unique<TranscribeCoremlEncoder>(model_path, options, false);
    encoder->attach(config, nullptr);
    return encoder;
}

// The window's adaptor rows (every token of the window) on the route, or
// false when the prediction failed.
inline bool encode_window(const parakeet::EncoderRoute & route, const TranscribeConfig & config,
                          const std::vector<float> & mel, std::vector<float> & rows) {
    const int tokens = config.encoder.n_ctx / config.merge_size;
    const std::vector<float> frames =
        parakeet::moss::detail::frame_major_mel(mel, config.audio.n_mels, config.audio.chunk_frames);
    rows.assign((size_t) tokens * config.text.n_embd, 0.0f);
    return route.encode(config.audio.chunk_frames, config.audio.chunk_frames, config.audio.n_mels, frames.data(),
                        tokens, config.text.n_embd, rows.data()) == 0;
}

inline double cosine(const float * a, const float * b, size_t n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; ++i) {
        dot += (double) a[i] * b[i];
        na += (double) a[i] * a[i];
        nb += (double) b[i] * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
}

inline double worst_token_cosine(const std::vector<float> & a, const std::vector<float> & b, int width) {
    double worst = 1.0;
    for (size_t row = 0; (row + 1) * width <= std::min(a.size(), b.size()); ++row) {
        worst = std::min(worst, cosine(a.data() + row * width, b.data() + row * width, (size_t) width));
    }
    return worst;
}

// The shared greedy prefix of two token sequences, over the longer one.
inline double token_agreement(const std::vector<int32_t> & a, const std::vector<int32_t> & b) {
    const size_t count = std::min(a.size(), b.size());
    size_t same = 0;
    while (same < count && a[same] == b[same]) {
        same++;
    }
    return (double) same / (double) std::max<size_t>(std::max(a.size(), b.size()), 1);
}

inline std::vector<float> read_wav(const std::string & path) {
    std::vector<float> pcm;
    int rate = 0;
    if (parakeet::load_wav_mono_f32(path, pcm, rate) != 0) {
        throw std::runtime_error("cannot read " + path);
    }
    return pcm;
}

// One window's embeddings on the route against the ggml encoder's.
struct WindowParity {
    bool predicted = false;
    int tokens = 0;
    double whole = 0.0;  // cosine over the window's tokens
    double worst = 0.0;  // the worst token's cosine
};

inline WindowParity compare_window(parakeet::moss::detail::TranscribeModel & model,
                                   const parakeet::EncoderRoute & route, const std::vector<float> & audio,
                                   size_t samples) {
    using namespace parakeet::moss::detail;
    const TranscribeConfig & config = model.config();
    const TranscribeMel mel(config.audio, read_mel_filters(model));
    const size_t count = std::min(samples, audio.size());
    const std::vector<float> window = mel.chunk(audio.data(), count);
    WindowParity parity;
    parity.tokens = transcribe_chunk_tokens(config, count);
    const std::vector<float> ggml = encode_audio_chunk(model, window, parity.tokens, false).embeddings;
    std::vector<float> rows;
    parity.predicted = encode_window(route, config, window, rows);
    rows.resize(ggml.size());
    parity.whole = cosine(rows.data(), ggml.data(), ggml.size());
    parity.worst = worst_token_cosine(rows, ggml, config.text.n_embd);
    return parity;
}

} // namespace moss_transcribe_coreml
