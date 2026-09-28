#pragma once

#include "tts-cpp/moss/speech.h"

namespace tts_cpp::moss::detail {

void validate_speech_request(const SpeechRequest & request);

} // namespace tts_cpp::moss::detail
