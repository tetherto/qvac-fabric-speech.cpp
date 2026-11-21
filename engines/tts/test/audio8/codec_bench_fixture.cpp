// Build a tiny decoder and obtain reference PCM through decode_codes directly.
// The integration test runs the unmodified benchmark executable separately.
#include "audio8/internal.h"
#include "audio8/window_stream.h"
#include "gguf.h"
#include "tiny_lm.h"
#include "tts-cpp/audio8/engine.h"
#include "tts-cpp/audio8/fit.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <future>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <vector>

namespace codec = tts_cpp::audio8::detail;
namespace fs = std::filesystem;

static void require(bool ok, const std::string & message) {
    if (!ok) throw std::runtime_error(message);
}

static void write_decoder(const fs::path & path, int max_frames = 8) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({1024 * 1024, nullptr, false}), ggml_free);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> file(gguf_init_empty(), gguf_free);
    require(ctx && file, "fixture allocation failed");
    gguf_set_val_str(file.get(), "general.architecture", "audio8-codec");
    gguf_set_val_str(file.get(), "audio8.codec.part", "decoder");
    auto u32 = [&](const char * key, uint32_t value) {
        gguf_set_val_u32(file.get(), (std::string("audio8.codec.") + key).c_str(), value);
    };
    auto f32 = [&](const char * key, float value) {
        gguf_set_val_f32(file.get(), (std::string("audio8.codec.") + key).c_str(), value);
    };
    u32("sample_rate", 8000); u32("frame_size", 4); u32("num_codebooks", 2);
    u32("residual_codebooks", 1); u32("latent_dim", 4); u32("codebook_dim", 4);
    u32("semantic_codebook_size", 4); u32("residual_codebook_size", 4); u32("max_frames", max_frames);
    f32("convnext_norm_eps", 1e-5f); f32("snake_epsilon", 1e-5f);
    const int32_t unused = 0;
    for (const auto * key : {"encoder_strides", "decoder_strides", "residual_dilations"})
        gguf_set_arr_data(file.get(), (std::string("audio8.codec.") + key).c_str(),
                          GGUF_TYPE_INT32, &unused, 0);
    // One tiny transformer layer and no DAC stages. The two production
    // stride-2 resamplers produce exactly four samples per input frame.
    u32("post_tf.n_layer", 1); u32("post_tf.n_head", 1); u32("post_tf.n_kv", 1);
    u32("post_tf.dim", 4); u32("post_tf.inter", 4); u32("post_tf.head_dim", 4);
    u32("post_tf.window", 8); f32("post_tf.rope_theta", 10000); f32("post_tf.norm_eps", 1e-5f);
    auto tensor = [&](const std::string & name, std::initializer_list<int64_t> dims,
                      bool constant = false, float value = 0.0f) {
        const std::vector<int64_t> shape(dims);
        ggml_tensor * t = ggml_new_tensor(ctx.get(), GGML_TYPE_F32,
                                          static_cast<int>(shape.size()), shape.data());
        ggml_set_name(t, name.c_str());
        auto * data = static_cast<float *>(t->data);
        for (int64_t i = 0; i < ggml_nelements(t); ++i)
            data[i] = constant ? value : static_cast<float>(i % 7 - 3) * 0.04f;
        gguf_add_tensor(file.get(), t);
    };
    auto conv = [&](const std::string & name, int in, int out, int taps) {
        tensor(name + "/weight", {in, out, taps});
        tensor(name + "/bias", {out});
    };
    for (const auto * bank : {"q/sem/0", "q/res/0"}) {
        tensor(std::string(bank) + "/codebook/weight", {4, 4});
        conv(std::string(bank) + "/out_proj", 4, 4, 1);
    }
    tensor("q/post/norm/weight", {4}, true, 1.0f);
    const std::string block = "q/post/blk/0/";
    tensor(block + "wq", {4, 4});
    tensor(block + "wk", {4, 4});
    tensor(block + "wv", {4, 4});
    tensor(block + "wo", {4, 4});
    tensor(block + "w1", {4, 4});
    tensor(block + "w2", {4, 4});
    tensor(block + "w3", {4, 4});
    for (const auto * suffix : {"attn_norm", "ffn_norm", "attn_scale", "ffn_scale"})
        tensor(block + suffix, {4}, true, 1.0f);
    tensor("rope/post_tf_cos", {2, max_frames}, true, 1.0f);
    tensor("rope/post_tf_sin", {2, max_frames}, true, 0.0f);
    for (int stage = 0; stage < 2; ++stage) {
        const std::string base = "q/up/" + std::to_string(stage);
        conv(base + "/0/conv", 4, 4, 2);
        tensor(base + "/1/dwconv/conv/weight", {4, 1});
        tensor(base + "/1/dwconv/conv/bias", {4});
        tensor(base + "/1/norm/weight", {4}, true, 1.0f);
        tensor(base + "/1/norm/bias", {4});
        conv(base + "/1/pwconv1", 4, 4, 1);
        conv(base + "/1/pwconv2", 4, 4, 1);
        tensor(base + "/1/gamma", {4}, true, 0.1f);
    }
    conv("dec/0/conv", 4, 4, 1);
    tensor("dec/1/alpha", {4}, true, 1.0f);
    conv("dec/2/conv", 4, 1, 1);
    require(gguf_write_to_file(file.get(), path.u8string().c_str(), false),
            "cannot write fixture decoder");
}

struct owned_codec {
    codec::codec_model model;
    ~owned_codec() { codec::free_codec(model); }
};

static std::vector<int32_t> streaming_frames(int count) {
    std::vector<int32_t> frames;
    for (int frame = 0; frame < count; ++frame) {
        frames.push_back(frame % 4);
        frames.push_back((frame + 1) % 4);
    }
    return frames;
}

static void feed_stream(codec::window_stream & stream, int count) {
    std::string error;
    for (int frame = 1; frame <= count; ++frame) require(stream.advance(frame, &error), error);
}

static void compare_pcm(const std::vector<float> & got, const std::vector<float> & expected) {
    require(got.size() == expected.size(), "streaming sample count mismatch");
    for (size_t index = 0; index < got.size(); ++index) {
        require(std::isfinite(got[index]) && std::abs(got[index] - expected[index]) < 1e-5f,
                "streaming waveform differs from batch");
    }
}

// Fail a chosen prediction to exercise replay before PCM and fatal failure
// after PCM with the production decoder and the real stream worker.
class failing_sidecar final : public codec::codec_coreml_model {
  public:
    explicit failing_sidecar(int fail_on) : fail_on_(fail_on) {}
    int window() const override { return 16; }
    const char * label() const override { return "coreml-test"; }
    bool synthesize(const float *, float * pcm) const override {
        if (++calls_ == fail_on_)
            return false;
        std::fill_n(pcm, window() * 4, 0.5f);
        return true;
    }
    void retire(const char *) const override { retired.set_value(); }
    mutable std::promise<void> retired;

  private:
    int fail_on_;
    mutable int calls_ = 0;
};

static void check_stream_failure(codec::codec_model & model,
                                 const std::vector<int32_t> & frames,
                                 const std::vector<int32_t> & rows,
                                 const std::vector<float> & expected,
                                 int fail_on) {
    auto sidecar = std::make_shared<failing_sidecar>(fail_on);
    auto retired = sidecar->retired.get_future();
    const codec::codec_coreml_call call{sidecar};
    int delivered = 0;
    auto stream = codec::open_synthesis_stream(
        model, call, frames, 1, {},
        [&](const std::vector<float> &) { ++delivered; });
    require(stream && stream->asynchronous(),
            "selected Core ML stream unavailable");
    const int count = static_cast<int>(frames.size()) / model.hp.num_codebooks;
    feed_stream(*stream, count);
    std::vector<float> pcm;
    std::string error;
    codec::decode_timing timing;
    const bool ok = codec::finish_synthesis_stream(
        model, call, *stream, rows.data(), count, 1, {}, pcm, &error, &timing);
    require(retired.wait_for(std::chrono::seconds(0)) ==
                std::future_status::ready,
            "failed sidecar was not retired");
    if (fail_on == 1) {
        require(ok && delivered == 0 && timing.synthesis_backend == "ggml" &&
                    pcm == expected,
                "failed first prediction did not replay wholly on ggml");
    } else {
        require(!ok && delivered == 1 &&
                    error == "audio8: codec failed after PCM delivery began",
                "stream did not fail after delivering PCM");
    }
}

static void
check_stream_failure_before_finish(codec::codec_model & model,
                                   const std::vector<int32_t> & frames,
                                   const std::vector<int32_t> & rows) {
    for (bool unwind : {false, true}) {
        auto sidecar = std::make_shared<failing_sidecar>(1);
        auto retired = sidecar->retired.get_future();
        const codec::codec_coreml_call call{sidecar};
        std::atomic<bool> cancel{false};
        const auto cancelled = [&] { return cancel.load(); };
        auto stream = codec::open_synthesis_stream(
            model, call, frames, 1, cancelled, [](const auto &) {});
        require(static_cast<bool>(stream), "Core ML stream unavailable");
        const int count =
            static_cast<int>(frames.size()) / model.hp.num_codebooks;
        feed_stream(*stream, count);
        require(retired.wait_for(std::chrono::seconds(5)) ==
                    std::future_status::ready,
                "worker did not retire its failed prediction before finish");
        cancel = true;
        if (unwind) {
            stream.reset();
            continue;
        }
        std::vector<float> pcm;
        std::string error;
        require(
            !codec::finish_synthesis_stream(model, call, *stream, rows.data(),
                                            count, 1, cancelled, pcm, &error,
                                            nullptr) &&
                error == codec::CANCELLED && pcm.empty() && !stream->emitted(),
            "cancelled stream replayed or delivered PCM");
    }
}

static void check_streaming_decoder(const fs::path & root) {
    constexpr int frames_count = 51;
    const auto path = root / "stream-decoder.gguf";
    write_decoder(path, frames_count);
    owned_codec loaded;
    std::string error;
    require(codec::load_codec(path.string(), 0, "cpu", loaded.model, &error), error);
    const auto frames = streaming_frames(frames_count);
    const auto rows = codec::codebook_rows(frames, loaded.model.hp.num_codebooks, 0, frames_count);
    std::vector<float> expected;
    require(codec::decode_codes(loaded.model, rows.data(), frames_count, 1, {}, expected, &error), error);
    std::vector<float> delivered;
    auto stream = codec::open_synthesis_stream(
        loaded.model, {}, frames, 1, {}, [&](const std::vector<float> & chunk) {
            delivered.insert(delivered.end(), chunk.begin(), chunk.end());
        });
    require(static_cast<bool>(stream), "ggml stream unavailable");
    feed_stream(*stream, frames_count);
    require(!delivered.empty(), "ggml PCM delayed until finish");
    std::vector<float> result;
    require(codec::finish_synthesis_stream(loaded.model, {}, *stream,
                                           rows.data(), frames_count, 1, {},
                                           result, &error, nullptr),
            error);
    require(delivered == result, "callback PCM differs from returned PCM");
    compare_pcm(delivered, expected);
    check_stream_failure(loaded.model, frames, rows, expected, 1);
    check_stream_failure(loaded.model, frames, rows, expected, 2);
    check_stream_failure_before_finish(loaded.model, frames, rows);

    loaded.model.synthesis_block_frames = 4;
    int calls = 0;
    result.clear();
    require(!codec::decode_codes(
                loaded.model, rows.data(), frames_count, 1,
                [&] { return calls++ >= 1; }, result, &error) &&
                error == codec::CANCELLED && calls == 2 &&
                result.size() == 16 &&
                std::equal(result.begin(), result.end(), expected.begin()),
            "batched cancellation did not keep exactly the first ggml block");
    fs::remove(path);
}

static void check_utf8_paths(const fs::path & root, const fs::path & decoder) {
    using namespace tts_cpp;
    const auto dir = root / fs::u8path(u8"\u97f3\u58f0-\u00e9");
    fs::create_directory(dir);
    const auto codec_path = dir / fs::u8path(u8"\u5fa9\u53f7.gguf");
    fs::copy_file(decoder, codec_path);

    codec::codec_header header;
    std::string error;
    require(codec::peek_codec_header(codec_path.u8string(), header, &error),
            error);
    require(header.part == "decoder", "wrong decoder header");

    audio8_test::tiny_lm lm;
    lm.num_codebooks = header.hp.num_codebooks;
    lm.codebook_size = header.hp.semantic_codebook_size;
    lm.semantic_end = lm.semantic_begin + lm.codebook_size - 1;
    lm.max_seq_len = 127;
    lm.ascii_tokenizer = true;
    const auto lm_path = (dir / fs::u8path(u8"\u8a00\u8a9e.gguf")).u8string();
    require(!audio8_test::write_tiny_lm_gguf(lm, lm_path).empty(),
            "cannot write UTF-8 LM fixture");

    audio8::FitOptions fit;
    fit.lm_gguf_path = lm_path;
    fit.codec_decoder_gguf_path = codec_path.u8string();
    fit.backend = "cpu";
    fit.max_frames = 3;
    fit.coreml.use = CoremlOptions::Use::off;
    const auto too_long = audio8::fit_params(fit);
    require(too_long.status == FitStatus::Error &&
                too_long.reason == "workload-too-large",
            "the default projection prompt fit the short LM");
    fit.prompt_tokens = 124;
    const auto projection = audio8::fit_params(fit);
    require(projection.status == FitStatus::Success, projection.reason);

    audio8::EngineOptions options;
    options.lm_gguf_path = lm_path;
    options.codec_decoder_gguf_path = codec_path.u8string();
    options.backend = "cpu";
    options.n_threads = 1;
    options.max_frames = 3;
    options.greedy = true;
    options.coreml.use = CoremlOptions::Use::off;
    options.coreml.memory_budget_bytes = 1LL << 30;
    audio8::Engine engine(options);
    const auto result = engine.synthesize("a");
    require(result.frames > 0 && result.frames <= options.max_frames &&
                result.pcm.size() ==
                    static_cast<size_t>(result.frames) * header.hp.frame_size &&
                result.codec_synthesis_backend == "ggml",
            "budgeted Core ML-off synthesis failed through UTF-8 paths");
}

int main(int argc, char ** argv) {
    try {
        require(argc == 2, "expected fixture directory");
        const fs::path root(argv[1]);
        const auto decoder = root / "decoder.gguf";
        write_decoder(decoder);
        const std::vector<int32_t> rows{0, 2, 1, 1, 3, 2};
        std::ofstream codes(root / "codes.csv", std::ios::binary);
        codes << "0,1\n2,3\n1,2\n";
        codes.close();
        require(static_cast<bool>(codes), "cannot write fixture codes");
        owned_codec loaded;
        std::string error;
        require(codec::load_codec(decoder.string(), 0, "cpu", loaded.model, &error), error);
        std::vector<float> pcm;
        require(codec::decode_codes(loaded.model, rows.data(), 3, 1, {}, pcm, &error), error);
        require(pcm.size() == 12, "fixture PCM geometry");
        // Hexadecimal text is an independent, lossless float representation.
        // Python packs these values as LE F32 to check every exported byte.
        const auto close_file = [](std::FILE * file) { std::fclose(file); };
        std::unique_ptr<std::FILE, decltype(close_file)> reference(
            std::fopen((root / "reference.txt").string().c_str(), "w"), close_file);
        require(static_cast<bool>(reference), "cannot open reference PCM");
        for (float sample : pcm) {
            require(std::isfinite(sample), "nonfinite fixture PCM");
            require(std::fprintf(reference.get(), "%a\n", static_cast<double>(sample)) > 0,
                    "cannot write reference PCM");
        }
        require(std::fflush(reference.get()) == 0, "cannot flush reference PCM");
        check_streaming_decoder(root);
        check_utf8_paths(root, decoder);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "codec bench fixture: %s\n", error.what());
        return 1;
    }
}
