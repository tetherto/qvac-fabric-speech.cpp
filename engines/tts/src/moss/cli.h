#pragma once

#include "tts-cpp/moss/engine.h"
#include "tts-cpp/moss/sound_effect.h"
#include "tts-cpp/moss/transcribe.h"

#include <string>

namespace tts_cpp::moss::cli {

enum class Mode { Speech, SoundEffect, Transcribe };

struct CliArgs {
    Mode mode = Mode::Speech;
    EngineOptions options;
    SoundEffectOptions sound_options;
    SoundEffectRequest sound_request;
    TranscribeOptions transcribe_options;
    TranscribeRequest transcribe_request;
    std::string audio_path;
    std::string text;
    std::string out_path;
    bool stream = false;
};

void print_usage();
bool parse_args(int argc, const char * const * argv, CliArgs & args);
int run(const CliArgs & args);
std::string transcript_json(const TranscribeResult & result);

} // namespace tts_cpp::moss::cli
