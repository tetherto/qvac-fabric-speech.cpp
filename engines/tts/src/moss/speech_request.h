#pragma once

#include "tts-cpp/moss/speech.h"

#include <cstddef>
#include <cstdint>

namespace tts_cpp::moss::detail {

void validate_speech_request(const SpeechRequest & request);
void check_speech_context(size_t prompt_rows, int max_new_tokens, int n_ctx_train);
int speech_generation_budget(size_t prompt_rows, int requested_tokens, int n_ctx_train);
// Whether the device's free memory, with the LM loaded, holds the decoder and
// still leaves room for a turn. An unknown free figure (0) never fits.
bool speech_stages_fit(uint64_t device_free_bytes, uint64_t decoder_bytes);

} // namespace tts_cpp::moss::detail
