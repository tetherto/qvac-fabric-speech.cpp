#pragma once

#include "moss/transcribe_model.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace tts_cpp::moss::detail {

void validate_transcribe_encoder(const TranscribeModel & model);
void validate_transcribe_decoder(const TranscribeModel & model);

std::vector<float> read_mel_filters(const TranscribeModel & model);

struct TranscribeChunkEncoding {
    std::vector<float> encoder_states;
    std::vector<float> embeddings;
};

TranscribeChunkEncoding encode_audio_chunk(TranscribeModel & model, const std::vector<float> & mel, int tokens,
                                           bool keep_encoder_states);

class TranscribeDecoder {
public:
    TranscribeDecoder(TranscribeModel & model, int n_ctx);
    ~TranscribeDecoder();
    TranscribeDecoder(const TranscribeDecoder &) = delete;
    TranscribeDecoder & operator=(const TranscribeDecoder &) = delete;

    std::vector<float> prefill(const std::vector<int32_t> & ids, const std::vector<float> & audio_embeddings,
                               int batch_tokens);
    std::vector<float> step(int32_t id);
    int position() const;
    int context() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tts_cpp::moss::detail
