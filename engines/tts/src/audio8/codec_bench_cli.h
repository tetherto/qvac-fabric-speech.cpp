#pragma once

#include <climits>
#include <stdexcept>
#include <string>

namespace audio8_codec_bench {
struct options {
    std::string encoder, decoder, input, output;
    int threads = 4;
    int gpu_layers = 0;
    bool help = false;
};

inline int integer(const std::string & value, const std::string & flag) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error(flag + " requires a nonnegative integer");
    size_t used = 0;
    const unsigned long n = std::stoul(value, &used);
    if (used != value.size() || n > INT_MAX)
        throw std::runtime_error(flag + " is out of range");
    return static_cast<int>(n);
}

inline options parse(int argc, char ** argv) {
    options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string flag(argv[i]);
        if (flag == "--help" || flag == "-h") { opts.help = true; continue; }
        if (flag != "--codec-encoder" && flag != "--codec-decoder" &&
            flag != "--in" && flag != "--out" && flag != "--threads" &&
            flag != "--n-gpu-layers") throw std::runtime_error("unknown argument: " + flag);
        if (++i == argc || std::string(argv[i]).empty() || std::string(argv[i]).rfind("--", 0) == 0)
            throw std::runtime_error("missing value for " + flag);
        const std::string value(argv[i]);
        if (flag == "--codec-encoder") opts.encoder = value;
        else if (flag == "--codec-decoder") opts.decoder = value;
        else if (flag == "--in") opts.input = value;
        else if (flag == "--out") opts.output = value;
        else if (flag == "--threads") opts.threads = integer(value, flag);
        else opts.gpu_layers = integer(value, flag);
    }
    if (opts.help) return opts;
    if (opts.encoder.empty() || opts.decoder.empty() || opts.input.empty() || opts.output.empty())
        throw std::runtime_error("--codec-encoder, --codec-decoder, --in and --out are required");
    if (opts.threads == 0) throw std::runtime_error("--threads must be positive");
    if (opts.input == opts.output || opts.encoder == opts.output || opts.decoder == opts.output)
        throw std::runtime_error("--out must differ from input and model paths");
    return opts;
}
} // namespace audio8_codec_bench
