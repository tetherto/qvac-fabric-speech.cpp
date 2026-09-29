#pragma once

#include "tts-cpp/moss/speech.h"

#include <cstddef>

namespace tts_cpp::moss::detail {

void validate_speech_request(const SpeechRequest & request);
void check_speech_context(size_t prompt_rows, int max_new_tokens, int n_ctx_train);

} // namespace tts_cpp::moss::detail
