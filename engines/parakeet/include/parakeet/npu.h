#pragma once

#include <map>
#include <string>

namespace parakeet {

struct NpuEncoderOptions {
    std::string model_path;
    std::map<std::string, std::string> provider_options;
};

} // namespace parakeet
