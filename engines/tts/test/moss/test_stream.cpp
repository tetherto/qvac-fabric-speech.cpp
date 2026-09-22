#include "gguf_fixtures.h"

#include "tts-cpp/moss/engine.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

using namespace moss_fixtures;

namespace {

int failures = 0;

void check(bool condition, const std::string & label) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", label.c_str());
        failures++;
    }
}

tts_cpp::moss::EngineOptions fixture_options(const std::filesystem::path & backbone,
                                             const std::filesystem::path & decoder) {
    tts_cpp::moss::EngineOptions options;
    options.backbone_path = backbone.string();
    options.decoder_path = decoder.string();
    options.n_threads = 1;
    options.context = 512;
    options.max_new_tokens = 24;
    options.stream_chunk_frames = 2;
    return options;
}

float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        return 1.0f;
    }
    float worst = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        worst = std::max(worst, std::fabs(a[i] - b[i]));
    }
    return worst;
}

void test_stream_matches_batch(tts_cpp::moss::Engine & engine) {
    const tts_cpp::moss::SynthesisResult batch = engine.synthesize("hi");
    check(!batch.pcm.empty(), "batch synthesis produces PCM");

    std::vector<float> streamed;
    size_t chunks = 0;
    const tts_cpp::moss::SynthesisResult stream = engine.synthesize_stream("hi",
            [&](const float * samples, size_t count, int rate) {
                check(rate == batch.sample_rate, "chunk sample rate matches batch");
                streamed.insert(streamed.end(), samples, samples + count);
                chunks++;
                return true;
            });
    check(!stream.cancelled, "streaming completes without cancellation");
    check(stream.pcm.empty(), "streaming leaves the result PCM to the callback");
    check(chunks >= 1, "streaming emits at least one chunk");
    check(stream.first_audio_ms > 0, "first audio latency is measured");
    check(stream.generated_frames == batch.generated_frames,
            "stream and batch share the seeded trajectory");
    check(streamed.size() == batch.pcm.size(), "streamed sample count matches batch");
    check(max_abs_diff(streamed, batch.pcm) < 1e-4f, "streamed audio matches the batch render");
}

void test_stream_callback_cancels(tts_cpp::moss::Engine & engine) {
    size_t chunks = 0;
    const tts_cpp::moss::SynthesisResult result = engine.synthesize_stream("hi",
            [&](const float *, size_t, int) {
                chunks++;
                return false;
            });
    check(result.cancelled, "returning false from the callback cancels the stream");
    check(chunks == 1, "no further chunks after the callback declines");
}

} // namespace

int main() {
    const auto backbone = write_backbone("stream-backbone", [](gguf_context *) {});
    const auto decoder = write_decoder("stream-decoder", N_VQ, 3 * D_MODEL,
            [](gguf_context *) {});
    int rc = 0;
    try {
        tts_cpp::moss::Engine engine(fixture_options(backbone, decoder));
        test_stream_matches_batch(engine);
        test_stream_callback_cancels(engine);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        rc = 1;
    }
    std::filesystem::remove(backbone);
    std::filesystem::remove(decoder);
    if (rc != 0) {
        return rc;
    }
    if (failures == 0) {
        std::printf("moss streaming: OK\n");
        return 0;
    }
    std::fprintf(stderr, "moss streaming: %d failures\n", failures);
    return 1;
}
