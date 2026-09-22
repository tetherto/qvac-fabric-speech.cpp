#include "gguf_fixtures.h"

#include "moss/codec.h"
#include "moss/delay_lm.h"
#include "tts-cpp/moss/engine.h"

#include <cstdio>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <string>

using tts_cpp::moss::detail::Codec;
using tts_cpp::moss::detail::DelayLM;
using namespace moss_fixtures;

namespace {

int failures = 0;

void check(bool condition, const std::string & label) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", label.c_str());
        failures++;
    }
}

void expect_backbone_failure(const char * name,
        const std::function<void(gguf_context *)> & mutate, const std::string & expected) {
    const auto path = write_backbone(name, mutate);
    try {
        DelayLM model(path.string(), false, 1, 64);
        check(false, std::string(name) + ": accepted invalid GGUF");
    } catch (const std::runtime_error & e) {
        check(std::string(e.what()).find(expected) != std::string::npos,
                std::string(name) + ": wrong failure: " + e.what());
    }
    std::filesystem::remove(path);
}

void expect_decoder_failure(const char * name, int64_t qkv_rows,
        const std::function<void(gguf_context *)> & mutate_meta, const std::string & expected) {
    const auto path = write_decoder(name, N_VQ, qkv_rows, mutate_meta);
    try {
        Codec codec(path.string(), false, 1);
        check(false, std::string(name) + ": accepted invalid GGUF");
    } catch (const std::runtime_error & e) {
        check(std::string(e.what()).find(expected) != std::string::npos,
                std::string(name) + ": wrong failure: " + e.what());
    }
    std::filesystem::remove(path);
}

void test_valid_backbone() {
    const auto path = write_backbone("valid-backbone", [](gguf_context *) {});
    DelayLM model(path.string(), false, 1, 64);
    check(model.config().text_vocab == TEXT_VOCAB, "backbone text vocab from tensor");
    check(model.config().audio_vocab == AUDIO_HEAD, "backbone audio vocab from tensor");
    check(model.config().n_vq == N_VQ, "backbone n_vq");
    std::filesystem::remove(path);
}

void test_valid_decoder() {
    const auto path = write_decoder("valid-decoder", N_VQ, 3 * D_MODEL, [](gguf_context *) {});
    Codec codec(path.string(), false, 1);
    check(!codec.is_encoder(), "decoder role");
    check(codec.num_quantizers() == N_VQ, "decoder quantizer count");
    check(codec.samples_per_frame() == OUT_DIM, "decoder upsample from patch product");
    const std::vector<int32_t> narrow(4, 1);
    check(codec.decode(narrow, 1).size() == (size_t) (4 * OUT_DIM),
            "partial-channel decode renders one sample block per frame");
    std::filesystem::remove(path);
}

void test_engine_rejects_quantizer_mismatch() {
    const auto backbone = write_backbone("engine-backbone", [](gguf_context *) {});
    const auto decoder = write_decoder("engine-decoder", N_VQ - 1, 3 * D_MODEL,
            [](gguf_context *) {});
    tts_cpp::moss::EngineOptions options;
    options.backbone_path = backbone.string();
    options.decoder_path = decoder.string();
    options.n_threads = 1;
    options.context = 64;
    try {
        tts_cpp::moss::Engine engine(options);
        check(false, "engine accepted a decoder with mismatched quantizers");
    } catch (const std::runtime_error & e) {
        check(std::string(e.what()).find("fewer quantizers") != std::string::npos,
                std::string("engine mismatch: wrong failure: ") + e.what());
    }
    std::filesystem::remove(backbone);
    std::filesystem::remove(decoder);
}

void test_engine_accepts_matching_pair() {
    const auto backbone = write_backbone("engine-backbone-ok", [](gguf_context *) {});
    const auto decoder = write_decoder("engine-decoder-ok", N_VQ, 3 * D_MODEL,
            [](gguf_context *) {});
    tts_cpp::moss::EngineOptions options;
    options.backbone_path = backbone.string();
    options.decoder_path = decoder.string();
    options.n_threads = 1;
    options.context = 64;
    tts_cpp::moss::Engine engine(options);
    check(engine.sample_rate() == 24000, "engine sample rate");
    std::filesystem::remove(backbone);
    std::filesystem::remove(decoder);
}

} // namespace

int main() {
    try {
        test_valid_backbone();
        test_valid_decoder();
        test_engine_accepts_matching_pair();
        test_engine_rejects_quantizer_mismatch();

        expect_backbone_failure("zero heads", [](gguf_context * f) {
            gguf_set_val_u32(f, "moss-tts-delay.attention.head_count", 0);
        }, "invalid model geometry");
        expect_backbone_failure("token id outside vocab", [](gguf_context * f) {
            gguf_set_val_u32(f, "moss-tts-delay.audio_assistant_gen_slot_token_id", 999);
        }, "outside the text vocabulary");
        expect_backbone_failure("huge token id", [](gguf_context * f) {
            gguf_set_val_u32(f, "moss-tts-delay.audio_end_token_id", 0xFFFFFFFFu);
        }, "outside the text vocabulary");
        expect_backbone_failure("pad code outside audio vocab", [](gguf_context * f) {
            gguf_set_val_u32(f, "moss-tts-delay.audio_pad_code", 99);
        }, "outside the audio vocabulary");

        expect_decoder_failure("zero patch size", 3 * D_MODEL, [](gguf_context * f) {
            gguf_set_val_u32(f, "moss-tts-audio-decoder.decoder.1.patch_size", 0);
        }, "invalid patch size");
        expect_decoder_failure("zero quantizers", 3 * D_MODEL, [](gguf_context * f) {
            gguf_set_val_u32(f, "moss-tts-audio-decoder.quantizer.num_quantizers", 0);
        }, "invalid quantizer geometry");
        expect_decoder_failure("oversized codebook_size", 3 * D_MODEL, [](gguf_context * f) {
            gguf_set_val_u32(f, "moss-tts-audio-decoder.quantizer.codebook_size", 4 * CODE_SIZE);
        }, "does not match the declared quantizer geometry");
        expect_decoder_failure("truncated qkv tensor", 2 * D_MODEL, [](gguf_context *) {},
                "unexpected dimensions");
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    if (failures == 0) {
        std::printf("moss load validation: OK\n");
        return 0;
    }
    std::fprintf(stderr, "moss load validation: %d failures\n", failures);
    return 1;
}
