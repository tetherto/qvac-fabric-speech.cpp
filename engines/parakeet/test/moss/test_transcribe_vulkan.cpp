#include "transcribe_fixtures.h"
#include "../../../test/moss_precision.h"
#include "moss/transcribe_networks.h"
#include "parakeet/moss_transcribe.h"
#include "parakeet/moss_transcribe_fit.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

using namespace moss_transcribe_fixtures;
using namespace parakeet::moss::detail;

namespace {
void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

void compare(const char * stage, const std::vector<float> & cpu, const std::vector<float> & gpu) {
    require(!cpu.empty() && cpu.size() == gpu.size(), "output shape mismatch");
    double error = 0, norm = 0;
    float worst = 0;
    for (size_t i = 0; i < cpu.size(); ++i) {
        require(std::isfinite(cpu[i]) && std::isfinite(gpu[i]), "nonfinite output");
        const double d = (double) cpu[i] - gpu[i];
        error += d * d;
        norm += (double) cpu[i] * cpu[i];
        worst = std::max(worst, (float) std::fabs(d));
    }
    const double relative = std::sqrt(error / std::max(norm, 1e-20));
    std::printf("%s: max_abs=%g rel_l2=%g\n", stage, worst, relative);
    require(worst < 2e-4f || relative < 0.01, "CPU/Vulkan mismatch");
}

void test(const std::string & path) {
    // Explicit Vulkan must win even when use_gpu is false or another GPU
    // backend is registered first. Missing Vulkan is a failure, not a skip.
    TranscribeModel cpu(path, false, 2), gpu(path, false, 2, "vulkan");
    require(std::string(gpu.backend_name()).find("Vulkan") == 0, "Vulkan was not selected");
    std::vector<float> mel(N_MELS * CHUNK_FRAMES);
    for (size_t i = 0; i < mel.size(); ++i) mel[i] = std::sin((float) i * 0.31f);
    const auto c = encode_audio_chunk(cpu, mel, CHUNK_TOKENS, true);
    const auto g = encode_audio_chunk(gpu, mel, CHUNK_TOKENS, true);
    compare("ASR encoder", c.encoder_states, g.encoder_states);
    compare("ASR adaptor", c.embeddings, g.embeddings);
    compare("ASR partial chunk", encode_audio_chunk(cpu, mel, 2, false).embeddings,
            encode_audio_chunk(gpu, mel, 2, false).embeddings);
    std::vector<int32_t> ids{TOKEN_IM_START, byte_token('a'), TOKEN_AUDIO_START};
    ids.insert(ids.end(), CHUNK_TOKENS, TOKEN_AUDIO_PAD);
    ids.push_back(TOKEN_AUDIO_END);
    ids.push_back(byte_token('b'));
    for (int batch : {1, 3, (int) ids.size()}) {
        TranscribeDecoder dc(cpu, 512), dg(gpu, 512);
        compare("ASR prefill", dc.prefill(ids, c.embeddings, batch), dg.prefill(ids, c.embeddings, batch));
        for (int i = 0; i < 5; ++i) compare("ASR decode", dc.step(byte_token('a')), dg.step(byte_token('a')));
    }
    bool rejected = false;
    try { TranscribeModel unavailable(path, true, 2, "missing-backend"); }
    catch (const std::runtime_error &) { rejected = true; }
    require(rejected, "explicit unavailable backend silently fell back");

    parakeet::moss::TranscribeOptions options;
    options.model_path = path;
    options.backend = "vulkan";
    parakeet::moss::TranscribeEngine engine(options);
    require(std::string(engine.backend_name()).find("Vulkan") == 0, "engine lost backend request");
    const std::vector<float> pcm(CHUNK_SAMPLES + 17, 0.1f);
    const auto result = engine.transcribe(pcm.data(), pcm.size(), SAMPLE_RATE);
    require(!result.cancelled && result.audio_tokens > 0, "engine transcription failed");
    const auto fit = parakeet::moss::fit_params(options, {}, 0.02, 0);
    require(fit.status != parakeet::FitStatus::Error && !fit.device_is_cpu &&
            fit.device_name.find("Vulkan") == 0 && fit.device.weights_bytes > 0,
            "memory projection lost explicit Vulkan selection");
    options.backend = "missing-backend";
    const auto unavailable_fit = parakeet::moss::fit_params(options, {}, 0.02, 0);
    require(unavailable_fit.status == parakeet::FitStatus::Error &&
            unavailable_fit.reason == "no-backend-device", "memory projection silently fell back");
}
} // namespace

int main() {
    std::filesystem::path path;
    int rc = 0;
    try {
        for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0}) {
            path = write_transcribe_model("vulkan", 26342);
            moss_fixture_precision(path, type);
            std::printf("precision: %s\n", ggml_type_name(type));
            test(path.string());
            std::filesystem::remove(path);
        }
        std::puts("MOSS Transcribe CPU/Vulkan parity: OK");
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        rc = 1;
    }
    std::error_code ec;
    std::filesystem::remove(path, ec);
    return rc;
}
