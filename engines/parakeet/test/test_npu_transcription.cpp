#include "parakeet/engine.h"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace {

parakeet::EngineResult transcribe_cpu(const parakeet::EngineOptions & options, const char * wav) {
    parakeet::Engine engine(options);
    return engine.transcribe(wav);
}

void compare(const parakeet::EngineResult & expected, const parakeet::EngineResult & actual) {
    if (expected.text != actual.text || expected.token_ids != actual.token_ids) {
        std::fprintf(stderr, "CPU: %s\nNPU: %s\n", expected.text.c_str(), actual.text.c_str());
        throw std::runtime_error("NPU transcription differs from the native CPU reference");
    }
}

void transcribe_npu(const parakeet::EngineOptions & options, const char * wav,
                    const parakeet::EngineResult & reference) {
    parakeet::Engine engine(options);
    if (engine.encoder_backend() != "VitisAI" || engine.backend_name() != "CPU") {
        throw std::runtime_error("expected a VitisAI encoder and native CPU decoder");
    }
    const auto accelerated = engine.transcribe(wav);
    compare(reference, accelerated);
    compare(reference, engine.transcribe(wav));
    std::printf("CPU encoder %.3f ms; NPU encoder %.3f ms; NPU-path decoder %.3f ms\n", reference.encoder_ms,
                accelerated.encoder_ms, accelerated.decode_ms);
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 6) {
        std::fprintf(stderr, "usage: %s model.gguf audio.wav encoder.onnx config.json cache-dir\n", argv[0]);
        return 2;
    }
    try {
        parakeet::EngineOptions options;
        options.model_gguf_path = argv[1];
        options.backend = "cpu";
        options.n_threads = 4;
        const auto reference = transcribe_cpu(options, argv[2]);
        options.npu_encoder.model_path = argv[3];
        options.npu_encoder.provider_options = {
            {"config_file", argv[4]}, {"cache_dir", argv[5]}, {"enable_cache_file_io_in_mem", "0"}};
        transcribe_npu(options, argv[2], reference);
        transcribe_npu(options, argv[2], reference);
        std::puts(
            "Matched native CPU tokens and transcript across repeated calls and recreated NPU engines.");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
