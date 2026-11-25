// The prebuilt Core ML encoder sidecar on the device: a GGUF with its
// -encoder.mlmodelc beside it (MOSS_TRANSCRIBE_COREML_MODEL), loaded through
// the engine's runtime path. Window parity against the ggml encoder, a
// transcription on the sidecar against the ggml one, use = require without a
// sidecar, and an invalid sidecar.

#include "moss/transcribe_coreml.h"
#include "moss/transcribe_model.h"
#include "moss/transcribe_text.h"
#include "transcribe_coreml_window.h"

#include "parakeet/moss_transcribe.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using namespace parakeet::moss::detail;
using namespace moss_transcribe_coreml;

namespace {

constexpr int SKIP = 77;
constexpr double WINDOW_COSINE = 0.999;
constexpr double TOKEN_COSINE = 0.99;
constexpr double TRANSCRIPT_AGREEMENT = 0.95;
constexpr int PARITY_THREADS = 8;
constexpr double SHORT_WINDOW_SECONDS = 3.0;
constexpr const char * MODEL_ENV = "MOSS_TRANSCRIBE_COREML_MODEL";
constexpr const char * GPU_ENV = "MOSS_TRANSCRIBE_GPU";

int failures = 0;

void check(bool condition, const std::string & label) {
    std::printf("%s: %s\n", condition ? "ok  " : "FAIL", label.c_str());
    if (!condition) {
        failures++;
    }
}

void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

void clear_env(const char * name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

std::filesystem::path scratch_dir(const char * tag) {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto dir = std::filesystem::temp_directory_path() /
        ("moss-coreml-" + std::string(tag) + "-" + std::to_string(stamp));
    std::filesystem::create_directories(dir);
    return dir;
}

std::filesystem::path linked_model(const std::filesystem::path & dir, const std::string & model_path) {
    const auto link = dir / std::filesystem::path(model_path).filename();
    std::filesystem::create_symlink(std::filesystem::absolute(model_path), link);
    return link;
}

// Core ML waits for the sidecar at construction; nothing is generated.
parakeet::moss::TranscribeOptions engine_options(const std::string & model_path) {
    parakeet::moss::TranscribeOptions options;
    options.model_path = model_path;
    options.n_threads = PARITY_THREADS;
    options.use_gpu = std::getenv(GPU_ENV) != nullptr;
    options.coreml.wait = parakeet::CoremlOptions::Wait::construct;
    options.coreml.generate = parakeet::CoremlOptions::Generate::never;
    return options;
}

void check_window(TranscribeModel & model, const parakeet::EncoderRoute & route, const std::vector<float> & audio,
                  size_t samples, const std::string & label) {
    const WindowParity parity = compare_window(model, route, audio, samples);
    check(parity.predicted, label + ": the sidecar predicts");
    std::printf("%s: %d tokens, cosine %.6f, worst token %.6f\n", label.c_str(), parity.tokens, parity.whole,
                parity.worst);
    check(parity.whole >= WINDOW_COSINE, label + ": embeddings match the ggml encoder");
    check(parity.worst >= TOKEN_COSINE, label + ": every token matches the ggml encoder");
}

void test_windows(const std::string & model_path, const std::vector<float> & audio) {
    TranscribeModel model(model_path, std::getenv(GPU_ENV) != nullptr, PARITY_THREADS);
    const auto sidecar = ready_encoder(model_path, model.config());
    const parakeet::EncoderRoute route = sidecar->acquire();
    check(route.holds() && sidecar->status_json().find("\"route\":\"prebuilt\"") != std::string::npos,
          "the staged sidecar loads on the runtime and matches the model window");
    if (!route.holds()) {
        return;
    }
    const auto & audio_config = model.config().audio;
    check_window(model, route, audio, (size_t) audio_config.chunk_samples, "first window");
    check_window(model, route, audio, (size_t) (SHORT_WINDOW_SECONDS * audio_config.sample_rate),
                 "zero-padded window");
}

parakeet::moss::TranscribeResult transcribe(parakeet::moss::TranscribeEngine & engine,
                                            const std::vector<float> & audio) {
    return engine.transcribe(audio.data(), audio.size(), engine.sample_rate());
}

void test_transcripts(const std::string & model_path, const std::vector<float> & audio) {
    auto options = engine_options(model_path);
    options.coreml.use = parakeet::CoremlOptions::Use::require;
    parakeet::moss::TranscribeEngine coreml(options);
    const auto accelerated = transcribe(coreml, audio);
    set_env(COREML_DISABLE_ENV, "1");
    parakeet::moss::TranscribeEngine ggml(engine_options(model_path));
    const auto reference = transcribe(ggml, audio);
    clear_env(COREML_DISABLE_ENV);
    check(coreml.encoder_on_coreml() && accelerated.encoder_backend.rfind("coreml", 0) == 0 &&
              !accelerated.coreml_pending,
          "the engine encodes on the sidecar and reports its label (" + accelerated.encoder_backend + ")");
    check(!ggml.encoder_on_coreml() && reference.encoder_backend == GGML_ENCODER_BACKEND,
          "MOSS_COREML_DISABLE keeps the ggml encoder");
    std::printf("encode: Core ML %.0f ms, ggml %.0f ms\n", accelerated.encode_ms, reference.encode_ms);
    std::printf("transcript match: %s\n", accelerated.text == reference.text ? "exact" : "differs");
    TranscribeModel model(model_path, false, 1, "auto", true);
    const TranscribeTokenizer tokenizer(model);
    const double agreement = token_agreement(tokenizer.encode(accelerated.text), tokenizer.encode(reference.text));
    std::printf("greedy token prefix agreement: %.4f\n", agreement);
    check(agreement >= TRANSCRIPT_AGREEMENT, "the Core ML transcript follows the ggml one");
}

void test_strict_without_sidecar(const std::string & model_path, const std::vector<float> & audio) {
    const auto dir = scratch_dir("strict");
    const auto bare = linked_model(dir, model_path);
    set_env(COREML_STRICT_ENV, "1");
    bool failed = false;
    try {
        parakeet::moss::TranscribeEngine engine(engine_options(bare.string()));
        transcribe(engine, audio);
    } catch (const std::runtime_error & e) {
        std::printf("refused: %s\n", e.what());
        failed = true;
    }
    clear_env(COREML_STRICT_ENV);
    check(failed, "MOSS_COREML_STRICT (use = require) fails a transcription without a sidecar");
    std::filesystem::remove_all(dir);
}

void test_invalid_sidecar(const std::string & model_path, const std::vector<float> & audio) {
    const auto dir = scratch_dir("invalid");
    const auto staged = linked_model(dir, model_path);
    std::filesystem::create_directories(transcribe_encoder_sidecar_path(staged.string()));
    parakeet::moss::TranscribeEngine engine(engine_options(staged.string()));
    check(!engine.encoder_on_coreml(), "a sidecar directory that is not a model is not attached");
    check(engine.coreml_status_json().find("\"state\":\"failed") != std::string::npos,
          "the status reports the sidecar failed");
    check(transcribe(engine, audio).encoder_backend == GGML_ENCODER_BACKEND,
          "an engine with an invalid sidecar transcribes on ggml");
    std::filesystem::remove_all(dir);
}

} // namespace

int main(int argc, char ** argv) {
#ifndef PARAKEET_USE_COREML
    std::printf("SKIP: built without PARAKEET_COREML\n");
    return SKIP;
#endif
    const char * model_path = std::getenv(MODEL_ENV);
    if (!model_path || argc < 2) {
        std::printf("SKIP: set %s to a GGUF with its -encoder.mlmodelc beside it, and pass a 16 kHz WAV\n",
                    MODEL_ENV);
        return SKIP;
    }
    try {
        const std::vector<float> audio = read_wav(argv[1]);
        test_windows(model_path, audio);
        test_transcripts(model_path, audio);
        test_strict_without_sidecar(model_path, audio);
        test_invalid_sidecar(model_path, audio);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    if (failures == 0) {
        std::printf("moss transcribe Core ML parity: OK\n");
        return 0;
    }
    std::fprintf(stderr, "moss transcribe Core ML parity: %d failures\n", failures);
    return 1;
}
