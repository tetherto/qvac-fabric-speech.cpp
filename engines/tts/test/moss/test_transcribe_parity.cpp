#include "moss/transcribe_audio.h"
#include "moss/transcribe_model.h"
#include "moss/transcribe_networks.h"
#include "moss/transcribe_text.h"

#include "tts-cpp/moss/transcribe.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tts_cpp::moss::detail;

namespace {

constexpr int SKIP = 77;
constexpr double MEL_COSINE = 0.99999;
constexpr double STAGE_COSINE = 0.999;
constexpr double TOKEN_AGREEMENT = 0.95;
constexpr int PARITY_THREADS = 8;
constexpr int PREFILL_BATCH = 256;

int failures = 0;

void check(bool condition, const std::string & label) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", label.c_str());
        failures++;
    }
}

template <typename T>
std::vector<T> read_bin(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("cannot read " + path.string());
    }
    const std::streamsize bytes = input.tellg();
    input.seekg(0);
    std::vector<T> data((size_t) bytes / sizeof(T));
    input.read(reinterpret_cast<char *>(data.data()), bytes);
    return data;
}

std::string read_text(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
}

double cosine(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size() || a.empty()) {
        return -2.0;
    }
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += (double) a[i] * b[i];
        na += (double) a[i] * a[i];
        nb += (double) b[i] * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
}

void report(const char * stage, double value, double threshold) {
    std::printf("%-22s %.6f (threshold %.5f)\n", stage, value, threshold);
    check(value >= threshold, std::string(stage) + " diverges from the PyTorch reference");
}

double token_agreement(const std::vector<int32_t> & a, const std::vector<int32_t> & b) {
    const size_t count = std::min(a.size(), b.size());
    size_t same = 0;
    while (same < count && a[same] == b[same]) {
        same++;
    }
    return (double) same / (double) std::max<size_t>(std::max(a.size(), b.size()), 1);
}

struct Fixture {
    TranscribeModel & model;
    std::filesystem::path dir;
    std::vector<float> audio;
};

std::vector<float> first_chunk_mel(const Fixture & fixture) {
    const TranscribeMel mel(fixture.model.config().audio, read_mel_filters(fixture.model));
    const size_t count = std::min(fixture.audio.size(), (size_t) fixture.model.config().audio.chunk_samples);
    return mel.chunk(fixture.audio.data(), count);
}

void test_mel(const Fixture & fixture) {
    report("log-mel chunk 0", cosine(first_chunk_mel(fixture), read_bin<float>(fixture.dir / "mel_chunk0.bin")),
           MEL_COSINE);
}

void test_encoder(const Fixture & fixture) {
    const std::vector<float> reference = read_bin<float>(fixture.dir / "mel_chunk0.bin");
    const int tokens = transcribe_chunk_tokens(fixture.model.config(),
            std::min(fixture.audio.size(), (size_t) fixture.model.config().audio.chunk_samples));
    const TranscribeChunkEncoding encoding = encode_audio_chunk(fixture.model, reference, tokens, true);
    report("encoder chunk 0", cosine(encoding.encoder_states, read_bin<float>(fixture.dir / "encoder_chunk0.bin")),
           STAGE_COSINE);
    const std::vector<float> embeddings = read_bin<float>(fixture.dir / "audio_embeddings.bin");
    const std::vector<float> head(embeddings.begin(), embeddings.begin() + (std::ptrdiff_t) encoding.embeddings.size());
    report("adaptor chunk 0", cosine(encoding.embeddings, head), STAGE_COSINE);
}

void test_prompt(const Fixture & fixture) {
    const TranscribeTokenizer tokenizer(fixture.model);
    const TranscribeConfig & config = fixture.model.config();
    const int audio_tokens = transcribe_audio_tokens(config, fixture.audio.size());
    check(transcribe_prompt(config, tokenizer, audio_tokens, "") == read_bin<int32_t>(fixture.dir / "prompt_ids.bin"),
          "prompt ids reproduce the Hugging Face processor");
}

void test_prefill(const Fixture & fixture) {
    const std::vector<int32_t> prompt = read_bin<int32_t>(fixture.dir / "prompt_ids.bin");
    TranscribeDecoder decoder(fixture.model, (int) prompt.size() + 1);
    const std::vector<float> logits = decoder.prefill(prompt, read_bin<float>(fixture.dir / "audio_embeddings.bin"),
            PREFILL_BATCH);
    report("prefill logits", cosine(logits, read_bin<float>(fixture.dir / "prefill_logits.bin")), STAGE_COSINE);
}

void test_end_to_end(const Fixture & fixture, const std::string & model_path, bool use_gpu) {
    tts_cpp::moss::TranscribeOptions options;
    options.model_path = model_path;
    options.use_gpu = use_gpu;
    options.n_threads = PARITY_THREADS;
    tts_cpp::moss::TranscribeEngine engine(options);
    tts_cpp::moss::TranscribeRequest request;
    std::vector<int32_t> reference = read_bin<int32_t>(fixture.dir / "generated_ids.bin");
    request.max_new_tokens = (int) reference.size();
    const tts_cpp::moss::TranscribeResult result = engine.transcribe(fixture.audio.data(), fixture.audio.size(),
            engine.sample_rate(), request);
    const std::string expected = strip_whitespace(read_text(fixture.dir / "text.txt"));
    std::printf("transcript match: %s\n", result.text == expected ? "exact" : "differs");
    std::printf("segments: %zu, generated tokens %d (reference %zu)\n", result.segments.size(),
            result.generated_tokens, reference.size() - 1);
    const TranscribeTokenizer tokenizer(fixture.model);
    report("greedy token prefix", token_agreement(tokenizer.encode(result.text), tokenizer.encode(expected)),
           TOKEN_AGREEMENT);
    check(!result.segments.empty(), "the transcript parses into segments");
}

} // namespace

int main() {
    const char * model_path = std::getenv("MOSS_TRANSCRIBE_MODEL");
    const char * reference_dir = std::getenv("MOSS_TRANSCRIBE_REFERENCE_DIR");
    if (!model_path || !reference_dir) {
        std::printf("SKIP: set MOSS_TRANSCRIBE_MODEL and MOSS_TRANSCRIBE_REFERENCE_DIR\n");
        return SKIP;
    }
    try {
        const bool use_gpu = std::getenv("MOSS_TRANSCRIBE_GPU") != nullptr;
        TranscribeModel model(model_path, use_gpu, PARITY_THREADS);
        std::printf("backend: %s\n", model.backend_name());
        validate_transcribe_encoder(model);
        validate_transcribe_decoder(model);
        const std::filesystem::path dir(reference_dir);
        const Fixture fixture{model, dir, read_bin<float>(dir / "audio.bin")};
        test_mel(fixture);
        test_prompt(fixture);
        test_encoder(fixture);
        test_prefill(fixture);
        test_end_to_end(fixture, model_path, use_gpu);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    if (failures == 0) {
        std::printf("moss transcribe parity: OK\n");
        return 0;
    }
    std::fprintf(stderr, "moss transcribe parity: %d failures\n", failures);
    return 1;
}
