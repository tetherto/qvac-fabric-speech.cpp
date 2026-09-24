#pragma once

#include "tts-cpp/moss/engine.h"

#include <string>

namespace tts_cpp::moss::cli {

struct CliArgs {
    EngineOptions options;
    std::string text;
    std::string out_path = "moss-out.wav";
    bool stream = false;
};

void print_usage();
bool parse_args(int argc, const char * const * argv, CliArgs & args);
int run(const CliArgs & args);

} // namespace tts_cpp::moss::cli
