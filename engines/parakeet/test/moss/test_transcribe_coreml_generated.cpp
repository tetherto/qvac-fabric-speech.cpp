// The generated Core ML encoder on the device. The GGUF
// (MOSS_TRANSCRIBE_GENERATED_MODEL) has no bundle beside it, so the engine
// generates the encoder from it into an empty cache directory; the test fails
// (never skips) unless it did. Then:
//  - window parity of the generated encoder against the ggml encoder, and a
//    transcription on it against the ggml one;
//  - a second engine adopts the cached encoder;
//  - fit charges the same Core ML envelope with an empty and a full cache;
//  - compute units the template was not validated for turn generation off;
//  - memory-degrade one byte below the Core ML-inclusive verdict.

#include "moss/transcribe_coreml.h"
#include "moss/transcribe_model.h"
#include "moss/transcribe_text.h"
#include "transcribe_coreml_window.h"

#include "parakeet/moss_transcribe.h"
#include "parakeet/moss_transcribe_fit.h"

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
constexpr int THREADS = 8;
constexpr double SHORT_WINDOW_SECONDS = 3.0;
constexpr double FIT_SECONDS = 30.0;
constexpr const char * MODEL_ENV = "MOSS_TRANSCRIBE_GENERATED_MODEL";
// Units the template is not validated for (its validated_for lists `all`).
constexpr const char * UNVALIDATED_UNITS = "cpu_only";

int failures = 0;

void check(bool condition, const std::string & label) {
    std::printf("%s: %s\n", condition ? "ok  " : "FAIL", label.c_str());
    if (!condition) {
        failures++;
    }
}

bool contains(const std::string & text, const std::string & needle) {
    return text.find(needle) != std::string::npos;
}

std::filesystem::path scratch_dir(const char * tag) {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto dir = std::filesystem::temp_directory_path() /
        ("moss-coreml-" + std::string(tag) + "-" + std::to_string(stamp));
    std::filesystem::create_directories(dir);
    return dir;
}

parakeet::moss::TranscribeOptions engine_options(const std::string & model_path, const std::string & cache_dir) {
    parakeet::moss::TranscribeOptions options;
    options.model_path = model_path;
    options.n_threads = THREADS;
    options.use_gpu = true;
    options.coreml.cache_dir = cache_dir;
    options.coreml.wait = parakeet::CoremlOptions::Wait::construct;
    return options;
}

parakeet::FitResult fit(const parakeet::moss::TranscribeOptions & options) {
    return parakeet::moss::fit_params(options, {}, FIT_SECONDS, 0);
}

void report_fit(const char * when, const parakeet::FitResult & r) {
    std::printf("fit (%s): route %s, Core ML %llu bytes, validated %d, ggml-only %llu bytes\n", when,
                r.coreml.encoder_route.c_str(), (unsigned long long) r.coreml.bytes, (int) r.coreml.validated,
                (unsigned long long) r.ggml_required_bytes);
}

parakeet::moss::TranscribeResult transcribe(parakeet::moss::TranscribeEngine & engine,
                                            const std::vector<float> & audio) {
    return engine.transcribe(audio.data(), audio.size(), engine.sample_rate());
}

void check_window(TranscribeModel & model, const parakeet::EncoderRoute & route, const std::vector<float> & audio,
                  size_t samples, const std::string & label) {
    const WindowParity parity = compare_window(model, route, audio, samples);
    check(parity.predicted, label + ": the generated encoder predicts");
    std::printf("%s: %d tokens, cosine %.6f, worst token %.6f\n", label.c_str(), parity.tokens, parity.whole,
                parity.worst);
    check(parity.whole >= WINDOW_COSINE, label + ": embeddings match the ggml encoder");
    check(parity.worst >= TOKEN_COSINE, label + ": every token matches the ggml encoder");
}

void test_generated(const std::string & model_path, const std::vector<float> & audio) {
    const auto cache = scratch_dir("generated");
    const auto options = engine_options(model_path, cache.string());
    const parakeet::FitResult before = fit(options);
    report_fit("empty cache", before);
    check(before.status != parakeet::FitStatus::Error && before.coreml.encoder_route == "generate" &&
              before.coreml.validated && before.coreml.bytes > 0,
          "fit charges the generated encoder's validated envelope");

    auto required = options;
    required.coreml.use = parakeet::CoremlOptions::Use::require;
    parakeet::moss::TranscribeEngine engine(required);
    const std::string status = engine.coreml_status_json();
    std::printf("status: %s\n", status.c_str());
    check(contains(status, "\"route\":\"generated\"") && engine.encoder_on_coreml(),
          "the engine generated the encoder from the GGUF and it is ready");
    const auto accelerated = transcribe(engine, audio);
    check(accelerated.encoder_backend.rfind("coreml", 0) == 0 && !accelerated.coreml_pending,
          "every window ran on the generated encoder (" + accelerated.encoder_backend + ")");

    {
        TranscribeModel model(model_path, true, THREADS);
        const auto sidecar = ready_encoder(model_path, model.config(), cache.string());
        const parakeet::EncoderRoute route = sidecar->acquire();
        check(route.holds() && contains(sidecar->status_json(), "\"route\":\"cached\""),
              "a second instance adopts the cached encoder");
        if (route.holds()) {
            const auto & audio_config = model.config().audio;
            check_window(model, route, audio, (size_t) audio_config.chunk_samples, "first window");
            check_window(model, route, audio, (size_t) (SHORT_WINDOW_SECONDS * audio_config.sample_rate),
                         "zero-padded window");
        }
    }

    auto off = options;
    off.coreml.use = parakeet::CoremlOptions::Use::off;
    parakeet::moss::TranscribeEngine ggml(off);
    const auto reference = transcribe(ggml, audio);
    TranscribeModel metadata(model_path, false, 1, "auto", true);
    const TranscribeTokenizer tokenizer(metadata);
    const double agreement = token_agreement(tokenizer.encode(accelerated.text), tokenizer.encode(reference.text));
    std::printf("encode: Core ML %.0f ms, ggml %.0f ms; greedy token prefix agreement %.4f\n",
                accelerated.encode_ms, reference.encode_ms, agreement);
    check(agreement >= TRANSCRIPT_AGREEMENT, "the generated encoder's transcript follows the ggml one");

    const parakeet::FitResult after = fit(options);
    report_fit("full cache", after);
    check(after.coreml.bytes == before.coreml.bytes && after.coreml.encoder_route == before.coreml.encoder_route &&
              after.ggml_required_bytes == before.ggml_required_bytes,
          "fit is the same with an empty and a full cache");

    auto unvalidated = options;
    unvalidated.coreml.compute_units = UNVALIDATED_UNITS;
    const parakeet::FitResult unvalidated_fit = fit(unvalidated);
    check(unvalidated_fit.coreml.encoder_route == "none" && unvalidated_fit.coreml.bytes == 0,
          "fit charges nothing for units the template was not validated for");
    parakeet::moss::TranscribeEngine suppressed(unvalidated);
    const std::string suppressed_status = suppressed.coreml_status_json();
    check(contains(suppressed_status, "unvalidated_units") && !suppressed.encoder_on_coreml(),
          "units the template was not validated for report unsupported(unvalidated_units)");

    auto degraded = options;
    degraded.coreml_fit = parakeet::CoremlFitTotals{before.ggml_required_bytes, before.coreml.bytes};
    degraded.coreml.memory_budget_bytes = (int64_t) (before.ggml_required_bytes + before.coreml.bytes - 1);
    parakeet::moss::TranscribeEngine below(degraded);
    check(!below.encoder_on_coreml() && contains(below.coreml_status_json(), "memory"),
          "one byte below the Core ML-inclusive verdict resolves use = off (memory)");
    degraded.coreml.memory_budget_bytes = (int64_t) (before.ggml_required_bytes + before.coreml.bytes);
    parakeet::moss::TranscribeEngine at(degraded);
    check(at.encoder_on_coreml(), "at the Core ML-inclusive verdict the encoder stays on Core ML");

    std::filesystem::remove_all(cache);
}

} // namespace

int main(int argc, char ** argv) {
#ifndef PARAKEET_USE_COREML
    std::printf("SKIP: built without PARAKEET_COREML\n");
    return SKIP;
#endif
    const char * model_path = std::getenv(MODEL_ENV);
    if (!model_path || argc < 2) {
        std::printf("SKIP: set %s to a GGUF with no -encoder.mlmodelc beside it, and pass a 16 kHz WAV\n",
                    MODEL_ENV);
        return SKIP;
    }
    if (std::filesystem::exists(transcribe_encoder_sidecar_path(model_path))) {
        std::fprintf(stderr, "FAIL: a Core ML encoder sits beside %s; the test needs the GGUF alone\n", model_path);
        return 1;
    }
    try {
        test_generated(model_path, read_wav(argv[1]));
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    if (failures == 0) {
        std::printf("moss transcribe generated Core ML encoder: OK\n");
        return 0;
    }
    std::fprintf(stderr, "moss transcribe generated Core ML encoder: %d failures\n", failures);
    return 1;
}
