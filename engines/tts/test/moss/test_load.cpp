#include "moss/codec.h"
#include "moss/delay_lm.h"
#include "tts-cpp/moss/engine.h"

#include "gguf.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

using tts_cpp::moss::detail::Codec;
using tts_cpp::moss::detail::DelayLM;

namespace {

constexpr int N_EMBD     = 16;
constexpr int N_FF       = 32;
constexpr int HEAD_DIM   = 8;
constexpr int N_HEADS    = 2;
constexpr int N_KV_HEADS = 1;
constexpr int N_VQ       = 2;
constexpr int TEXT_VOCAB = 16;
constexpr int AUDIO_HEAD = 8;
constexpr int D_MODEL    = 8;
constexpr int RVQ_DIM    = 8;
constexpr int CODE_DIM   = 4;
constexpr int CODE_SIZE  = 8;
constexpr int OUT_DIM    = 4;

int failures = 0;

void check(bool condition, const std::string & label) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", label.c_str());
        failures++;
    }
}

std::filesystem::path temp_gguf(const char * tag) {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
        ("moss-" + std::string(tag) + "-" + std::to_string(stamp) + ".gguf");
}

struct GgufBuilder {
    gguf_context * file;
    ggml_context * tensors;

    GgufBuilder() : file(gguf_init_empty()),
                    tensors(ggml_init({16 * 1024 * 1024, nullptr, false})) {}

    ~GgufBuilder() {
        ggml_free(tensors);
        gguf_free(file);
    }

    void add1(const std::string & name, int64_t n) {
        ggml_tensor * t = ggml_new_tensor_1d(tensors, GGML_TYPE_F32, n);
        std::memset(t->data, 0, ggml_nbytes(t));
        ggml_set_name(t, name.c_str());
        gguf_add_tensor(file, t);
    }

    void add2(const std::string & name, int64_t ne0, int64_t ne1) {
        ggml_tensor * t = ggml_new_tensor_2d(tensors, GGML_TYPE_F32, ne0, ne1);
        std::memset(t->data, 0, ggml_nbytes(t));
        ggml_set_name(t, name.c_str());
        gguf_add_tensor(file, t);
    }

    void write(const std::filesystem::path & path) {
        if (!gguf_write_to_file(file, path.string().c_str(), false)) {
            throw std::runtime_error("cannot write test GGUF");
        }
    }
};

void add_backbone_meta(gguf_context * f) {
    gguf_set_val_str(f, "general.architecture", "moss-tts-delay");
    gguf_set_val_u32(f, "moss-tts-delay.block_count", 1);
    gguf_set_val_u32(f, "moss-tts-delay.embedding_length", N_EMBD);
    gguf_set_val_u32(f, "moss-tts-delay.feed_forward_length", N_FF);
    gguf_set_val_u32(f, "moss-tts-delay.attention.head_count", N_HEADS);
    gguf_set_val_u32(f, "moss-tts-delay.attention.head_count_kv", N_KV_HEADS);
    gguf_set_val_u32(f, "moss-tts-delay.attention.key_length", HEAD_DIM);
    gguf_set_val_u32(f, "moss-tts-delay.context_length", 512);
    gguf_set_val_u32(f, "moss-tts-delay.n_vq", N_VQ);
    gguf_set_val_u32(f, "moss-tts-delay.audio_vocab_size", AUDIO_HEAD - 1);
    gguf_set_val_u32(f, "moss-tts-delay.audio_pad_code", AUDIO_HEAD - 1);
    gguf_set_val_u32(f, "moss-tts-delay.audio_start_token_id", 3);
    gguf_set_val_u32(f, "moss-tts-delay.audio_end_token_id", 4);
    gguf_set_val_u32(f, "moss-tts-delay.audio_user_slot_token_id", 5);
    gguf_set_val_u32(f, "moss-tts-delay.audio_assistant_gen_slot_token_id", 6);
    gguf_set_val_u32(f, "moss-tts-delay.audio_assistant_delay_slot_token_id", 7);
    gguf_set_val_u32(f, "moss-tts-delay.sampling_rate", 24000);

    std::vector<std::string> tokens(TEXT_VOCAB);
    tokens[0] = "<|endoftext|>";
    tokens[1] = "<|im_start|>";
    tokens[2] = "<|im_end|>";
    for (int i = 3; i < TEXT_VOCAB; ++i) {
        tokens[(size_t) i] = "tok" + std::to_string(i);
    }
    std::vector<const char *> token_ptrs;
    for (const std::string & token : tokens) {
        token_ptrs.push_back(token.c_str());
    }
    gguf_set_arr_str(f, "tokenizer.ggml.tokens", token_ptrs.data(), token_ptrs.size());
    const char * merges[] = {"a b"};
    gguf_set_arr_str(f, "tokenizer.ggml.merges", merges, 1);
}

void add_backbone_tensors(GgufBuilder & b) {
    b.add2("token_embd.weight", N_EMBD, TEXT_VOCAB);
    b.add1("output_norm.weight", N_EMBD);
    b.add2("output.weight", N_EMBD, TEXT_VOCAB);
    for (int i = 0; i < N_VQ; ++i) {
        b.add2("token_embd_audio." + std::to_string(i) + ".weight", N_EMBD, AUDIO_HEAD);
        b.add2("output_audio." + std::to_string(i) + ".weight", N_EMBD, AUDIO_HEAD);
    }
    b.add1("blk.0.attn_norm.weight", N_EMBD);
    b.add2("blk.0.attn_q.weight", N_EMBD, (int64_t) N_HEADS * HEAD_DIM);
    b.add2("blk.0.attn_k.weight", N_EMBD, (int64_t) N_KV_HEADS * HEAD_DIM);
    b.add2("blk.0.attn_v.weight", N_EMBD, (int64_t) N_KV_HEADS * HEAD_DIM);
    b.add2("blk.0.attn_output.weight", (int64_t) N_HEADS * HEAD_DIM, N_EMBD);
    b.add1("blk.0.attn_q_norm.weight", HEAD_DIM);
    b.add1("blk.0.attn_k_norm.weight", HEAD_DIM);
    b.add1("blk.0.ffn_norm.weight", N_EMBD);
    b.add2("blk.0.ffn_gate.weight", N_EMBD, N_FF);
    b.add2("blk.0.ffn_up.weight", N_EMBD, N_FF);
    b.add2("blk.0.ffn_down.weight", N_FF, N_EMBD);
}

std::filesystem::path write_backbone(const char * tag,
        const std::function<void(gguf_context *)> & mutate_meta) {
    GgufBuilder b;
    add_backbone_meta(b.file);
    add_backbone_tensors(b);
    mutate_meta(b.file);
    const auto path = temp_gguf(tag);
    b.write(path);
    return path;
}

void add_decoder_meta(gguf_context * f, int num_quantizers) {
    gguf_set_val_str(f, "general.architecture", "moss-tts-audio-decoder");
    gguf_set_val_u32(f, "moss-tts-audio-decoder.sampling_rate", 24000);
    gguf_set_val_u32(f, "moss-tts-audio-decoder.quantizer.input_dim", RVQ_DIM);
    gguf_set_val_u32(f, "moss-tts-audio-decoder.quantizer.rvq_dim", RVQ_DIM);
    gguf_set_val_u32(f, "moss-tts-audio-decoder.quantizer.output_dim", OUT_DIM);
    gguf_set_val_u32(f, "moss-tts-audio-decoder.quantizer.num_quantizers", (uint32_t) num_quantizers);
    gguf_set_val_u32(f, "moss-tts-audio-decoder.quantizer.codebook_size", CODE_SIZE);
    gguf_set_val_u32(f, "moss-tts-audio-decoder.quantizer.codebook_dim", CODE_DIM);
    gguf_set_val_u32(f, "moss-tts-audio-decoder.decoder.block_count", 2);
    gguf_set_val_str(f, "moss-tts-audio-decoder.decoder.0.module_type", "Transformer");
    gguf_set_val_u32(f, "moss-tts-audio-decoder.decoder.0.input_dimension", OUT_DIM);
    gguf_set_val_u32(f, "moss-tts-audio-decoder.decoder.0.output_dimension", OUT_DIM);
    gguf_set_val_u32(f, "moss-tts-audio-decoder.decoder.0.d_model", D_MODEL);
    gguf_set_val_u32(f, "moss-tts-audio-decoder.decoder.0.num_heads", 2);
    gguf_set_val_u32(f, "moss-tts-audio-decoder.decoder.0.num_layers", 1);
    gguf_set_val_u32(f, "moss-tts-audio-decoder.decoder.0.context", 8);
    gguf_set_val_str(f, "moss-tts-audio-decoder.decoder.1.module_type", "PatchedPretransform");
    gguf_set_val_u32(f, "moss-tts-audio-decoder.decoder.1.patch_size", OUT_DIM);
}

void add_decoder_tensors(GgufBuilder & b, int num_quantizers, int64_t qkv_rows) {
    b.add2("quantizer.output_proj.weight", RVQ_DIM, OUT_DIM);
    for (int iq = 0; iq < num_quantizers; ++iq) {
        const std::string prefix = "quantizer.quantizers." + std::to_string(iq) + ".";
        b.add2(prefix + "codebook.weight", CODE_DIM, CODE_SIZE);
        b.add2(prefix + "out_proj.weight", CODE_DIM, RVQ_DIM);
    }
    b.add2("blk.0.input_proj.weight", OUT_DIM, D_MODEL);
    b.add2("blk.0.output_proj.weight", D_MODEL, OUT_DIM);
    b.add2("blk.0.layer.0.attn_qkv.weight", D_MODEL, qkv_rows);
    b.add2("blk.0.layer.0.attn_output.weight", D_MODEL, D_MODEL);
    b.add2("blk.0.layer.0.ffn_up.weight", D_MODEL, 2 * D_MODEL);
    b.add2("blk.0.layer.0.ffn_down.weight", 2 * D_MODEL, D_MODEL);
    b.add1("blk.0.layer.0.attn_norm.weight", D_MODEL);
    b.add1("blk.0.layer.0.attn_norm.bias", D_MODEL);
    b.add1("blk.0.layer.0.ffn_norm.weight", D_MODEL);
    b.add1("blk.0.layer.0.ffn_norm.bias", D_MODEL);
}

std::filesystem::path write_decoder(const char * tag, int num_quantizers, int64_t qkv_rows,
        const std::function<void(gguf_context *)> & mutate_meta) {
    GgufBuilder b;
    add_decoder_meta(b.file, num_quantizers);
    mutate_meta(b.file);
    add_decoder_tensors(b, num_quantizers, qkv_rows);
    const auto path = temp_gguf(tag);
    b.write(path);
    return path;
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
    std::filesystem::remove(path);
}

void test_engine_rejects_quantizer_mismatch() {
    const auto backbone = write_backbone("engine-backbone", [](gguf_context *) {});
    const auto decoder = write_decoder("engine-decoder", N_VQ + 1, 3 * D_MODEL,
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
        check(std::string(e.what()).find("decoder quantizers do not match") != std::string::npos,
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
            gguf_set_val_u32(f, "moss-tts-delay.audio_assistant_gen_slot_token_id", 99);
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
