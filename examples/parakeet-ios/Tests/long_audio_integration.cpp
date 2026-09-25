#include "LongAudioRunner.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s MODEL.gguf AUDIO.wav\n", argv[0]);
        return 2;
    }

    setenv("GGML_METAL_NO_RESIDENCY", "1", 1);
    unsetenv("PARAKEET_COREML_DISABLE");

    try {
        parakeet::EngineOptions options;
        options.model_gguf_path = argv[1];
        options.n_gpu_layers = 999;
        options.n_threads = 0;
        options.verbose = true;
        options.long_form_window_frames = 180;
        options.long_form_context_frames = 32;

        parakeet::Engine engine(options);
        if (!engine.encoder_on_coreml()) {
            throw std::runtime_error("Core ML encoder did not load");
        }

        std::atomic<bool> cancelled{false};
        int callbacks = 0;
        const auto started = std::chrono::steady_clock::now();
        const auto result = run_long_audio_wav(
            engine,
            argv[2],
            cancelled,
            true,
            [&](const parakeet::StreamingSegment & segment, double start, double end) {
                ++callbacks;
                std::printf("SEGMENT %.2f %.2f %s\n", start, end, segment.text.c_str());
                std::fflush(stdout);
            },
            11LL * 60LL * 16000LL);

        const double wall_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        const int64_t expected_samples = 11LL * 60LL * 16000LL;
        if (result.audio_samples != expected_samples) {
            throw std::runtime_error("integration fixture was shorter than 11 minutes");
        }
        if (result.batch_count != 22) {
            throw std::runtime_error("expected exactly 22 fixed-shape batches");
        }
        if (callbacks == 0 || result.segment_count == 0 || result.text.empty()) {
            throw std::runtime_error("no live transcription was emitted");
        }
        if (!result.encoder_used_coreml) {
            throw std::runtime_error("Core ML was not used for every batch");
        }

        std::printf(
            "PASS batches=%d callbacks=%d samples=%lld inference_s=%.3f wall_s=%.3f\n",
            result.batch_count,
            callbacks,
            static_cast<long long>(result.audio_samples),
            result.inference_ms / 1000.0,
            wall_seconds);
        return 0;
    } catch (const std::exception & exception) {
        std::fprintf(stderr, "FAIL %s\n", exception.what());
        return 1;
    }
}
