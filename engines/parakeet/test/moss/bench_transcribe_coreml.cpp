#include "moss/transcribe_audio.h"
#include "moss/transcribe_coreml.h"
#include "moss/transcribe_model.h"
#include "moss/transcribe_networks.h"
#include "transcribe_coreml_window.h"

#include "mel_preprocess.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

using namespace parakeet::moss::detail;

namespace {

constexpr int SKIP = 77;
constexpr int BENCH_THREADS = 8;
constexpr int RUNS = 5;
constexpr double PARITY_COSINE = 0.999;
constexpr const char * MODEL_ENV = "MOSS_TRANSCRIBE_COREML_MODEL";
// Optional: a Core ML cache directory, so a GGUF with no bundle beside it is
// benchmarked on its generated encoder.
constexpr const char * CACHE_ENV = "MOSS_TRANSCRIBE_COREML_CACHE_DIR";

double median_ms(const std::function<void()> & run) {
    run();
    std::vector<double> times;
    for (int i = 0; i < RUNS; ++i) {
        const auto start = std::chrono::steady_clock::now();
        run();
        times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

double cosine(const std::vector<float> & a, const std::vector<float> & b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        dot += (double) a[i] * b[i];
        na += (double) a[i] * a[i];
        nb += (double) b[i] * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
}

std::vector<float> read_wav(const std::string & path) {
    std::vector<float> pcm;
    int rate = 0;
    parakeet::load_wav_mono_f32(path, pcm, rate);
    return pcm;
}

} // namespace

int main(int argc, char ** argv) {
#ifndef PARAKEET_USE_COREML
    std::printf("SKIP: built without PARAKEET_COREML\n");
    return SKIP;
#endif
    const char * model_path = std::getenv(MODEL_ENV);
    if (model_path == nullptr || argc < 2) {
        std::printf("SKIP: set %s and pass a 16 kHz WAV\n", MODEL_ENV);
        return SKIP;
    }
    TranscribeModel model(model_path, true, BENCH_THREADS);
    const TranscribeConfig & config = model.config();
    const char * cache_dir = std::getenv(CACHE_ENV);
    const auto sidecar = moss_transcribe_coreml::ready_encoder(model_path, config, cache_dir ? cache_dir : "");
    const parakeet::EncoderRoute route = sidecar->acquire();
    const std::vector<float> audio = read_wav(argv[1]);
    const size_t count = std::min(audio.size(), (size_t) config.audio.chunk_samples);
    const TranscribeMel mel(config.audio, read_mel_filters(model));
    const std::vector<float> window = mel.chunk(audio.data(), count);
    const int tokens = transcribe_chunk_tokens(config, count);
    std::vector<float> reference;
    std::vector<float> rows;
    const double ggml_ms = median_ms([&] { reference = encode_audio_chunk(model, window, tokens, false).embeddings; });
    bool predicted = true;
    const double coreml_ms = median_ms([&] {
        predicted = moss_transcribe_coreml::encode_window(route, config, window, rows) && predicted;
    });
    rows.resize(reference.size());
    const double agreement = predicted ? cosine(rows, reference) : 0.0;
    std::printf("| stage | workload | ggml GPU | Core ML | speedup | agreement | placement |\n");
    std::printf("|---|---|---:|---:|---:|---:|---|\n");
    std::printf("| MOSS-Transcribe encoder | one 30 s window | %.1f ms | %.1f ms | %.2fx | %.6f | %s |\n", ggml_ms,
                coreml_ms, ggml_ms / coreml_ms, agreement, route.coreml_label().c_str());
    return agreement >= PARITY_COSINE ? 0 : 1;
}
