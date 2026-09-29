#pragma once

#include "tts-cpp/moss/engine.h"
#include "tts-cpp/moss/sound_effect.h"

#include <string>

namespace tts_cpp::moss::cli {

enum class Mode { Speech, SoundEffect };

struct CliArgs {
    Mode mode = Mode::Speech;
    EngineOptions options;
    SoundEffectOptions sound_options;
    SoundEffectRequest sound_request;
    std::string text;
    std::string out_path = "moss-out.wav";
    bool stream = false;
};

void print_usage();
bool parse_args(int argc, const char * const * argv, CliArgs & args);
int run(const CliArgs & args);

} // namespace tts_cpp::moss::cli
