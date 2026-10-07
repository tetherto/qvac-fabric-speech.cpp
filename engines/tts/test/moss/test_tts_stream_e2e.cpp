// Full-checkpoint streaming: batch agreement, chunk delivery, cancellation and
// reuse. Run on CI workers; the local Vulkan streaming test uses tiny fixtures.
#include "tts-cpp/moss/engine.h"
#include "dr_wav.h"
#include "../test_env_portable.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace tts_cpp::moss;

namespace {
void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

void valid_audio(const std::vector<float> & pcm) {
    require(!pcm.empty() && std::all_of(pcm.begin(), pcm.end(),
            [](float x) { return std::isfinite(x); }), "empty or non-finite audio");
    require(std::any_of(pcm.begin(), pcm.end(), [](float x) { return std::fabs(x) > 1e-5f; }),
            "silent audio");
}

void write_wav(const fs::path & path, const std::vector<float> & pcm, int rate) {
    drwav_data_format format{drwav_container_riff, DR_WAVE_FORMAT_PCM, 1, (drwav_uint32) rate, 16};
    drwav wav{};
    require(drwav_init_file_write(&wav, path.string().c_str(), &format, nullptr), "cannot write WAV");
    std::vector<drwav_int16> samples(pcm.size());
    drwav_f32_to_s16(samples.data(), pcm.data(), pcm.size());
    const auto written = drwav_write_pcm_frames(&wav, samples.size(), samples.data());
    drwav_uninit(&wav);
    require(written == samples.size(), "short WAV write");
}

SynthesisResult batch(Engine & engine, const std::string & text, const fs::path & path, int limit) {
    auto result = engine.synthesize(text);
    require(!result.cancelled && result.generated_frames < limit, "batch cancelled or hit token limit");
    valid_audio(result.pcm);
    write_wav(path, result.pcm, result.sample_rate);
    return result;
}

void stream(Engine & engine, const std::string & text, const SynthesisResult & reference,
            const fs::path & prefix) {
    std::vector<float> pcm;
    size_t chunks = 0;
    std::ofstream timings(prefix.string() + ".csv");
    require(timings.good(), "cannot write callback timings");
    timings << "chunk,elapsed_ms,samples\n";
    const auto start = std::chrono::steady_clock::now();
    const auto result = engine.synthesize_stream(text, [&](const float * data, size_t count, int rate) {
        require(rate == reference.sample_rate && count > 0, "invalid audio callback");
        pcm.insert(pcm.end(), data, data + count);
        const auto elapsed = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
        timings << ++chunks << ',' << elapsed << ',' << count << '\n';
        return true;
    });
    require(!result.cancelled && result.pcm.empty(), "streaming result/cancellation mismatch");
    require(chunks >= 2 && result.first_audio_ms > 0, "stream did not deliver incremental audio");
    require(result.generated_frames == reference.generated_frames, "stream changed generation trajectory");
    require(pcm.size() == reference.pcm.size(), "stream lost or duplicated samples");
    valid_audio(pcm);
    double error = 0, norm = 0;
    for (size_t i = 0; i < pcm.size(); ++i) {
        const double delta = (double) pcm[i] - reference.pcm[i];
        error += delta * delta;
        norm += (double) reference.pcm[i] * reference.pcm[i];
    }
    const double relative = std::sqrt(error / std::max(norm, 1e-20));
    std::fprintf(stderr, "%s: chunks=%zu samples=%zu first_audio_ms=%.1f relative_l2=%g\n",
            prefix.filename().string().c_str(), chunks, pcm.size(), result.first_audio_ms, relative);
    // Batch/chunk shapes select different F16 kernels. The existing Linux
    // full-checkpoint sample measures 0.47% L2; guard against larger drift.
    require(relative < 0.02, "stream waveform diverged from batch");
    write_wav(prefix.string() + ".wav", pcm, result.sample_rate);
}

void cancel(Engine & engine, const std::string & text, bool explicit_cancel) {
    size_t callbacks = 0;
    const auto result = engine.synthesize_stream(text, [&](const float *, size_t, int) {
        ++callbacks;
        if (explicit_cancel) engine.cancel();
        return explicit_cancel;
    });
    require(result.cancelled && callbacks == 1, "cancellation did not stop after the first chunk");
}
} // namespace

int main(int argc, char ** argv) {
    if (argc != 6) {
        std::fprintf(stderr, "usage: %s LM.gguf DECODER.gguf ENCODER.gguf REFERENCE.wav OUTPUT_DIR\n", argv[0]);
        return 77;
    }
    try {
        setenv("TTS_CPP_GPU_BACKEND", "vulkan", 1);
        unsetenv("GGML_VK_DISABLE_F16");
        const fs::path output(argv[5]);
        fs::create_directories(output);
        const std::string short_text = "Hello, this is a test of speech synthesis.";
        const std::string long_text = "Hello, this is a test of speech synthesis. "
                "We are checking that every audio chunk arrives in order, "
                "including the final part of this sentence.";
        for (int chunk : {7, 25}) {
            EngineOptions options;
            options.backbone_path = argv[1];
            options.decoder_path = argv[2];
            options.encoder_path = argv[3];
            options.reference_audio_path = argv[4];
            options.language = "en";
            options.use_gpu = true;
            options.n_threads = 4;
            options.seed = 1234;
            options.max_new_tokens = 384;
            options.stream_chunk_frames = chunk;
            Engine engine(options);
            const std::string backend = engine.backend_name();
            require(backend.find("Vulkan") == 0, "Vulkan backend required");
            std::fprintf(stderr, "[moss-tts-stream-e2e] backend: %s\n", backend.c_str());
            const std::string label = "tts-c" + std::to_string(chunk);
            const auto reference = batch(engine, short_text, output / (label + "-batch.wav"), options.max_new_tokens);
            stream(engine, short_text, reference, output / (label + "-stream"));
            cancel(engine, short_text, false);
            stream(engine, short_text, reference, output / (label + "-after-callback-cancel"));
            cancel(engine, short_text, true);
            stream(engine, short_text, reference, output / (label + "-after-explicit-cancel"));
            if (chunk == 7) {
                const auto longer = batch(engine, long_text, output / "tts-long-batch.wav", options.max_new_tokens);
                stream(engine, long_text, longer, output / "tts-long-stream");
            }
        }
        std::puts("MOSS TTS streaming/reuse: OK");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "moss tts stream e2e: %s\n", error.what());
        return 1;
    }
}
