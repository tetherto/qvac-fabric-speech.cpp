#include "parakeet_npu.h"

#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

void fails(const std::function<void()> & action, const char * message) {
    try {
        action();
    } catch (const std::exception &) {
        return;
    }
    throw std::runtime_error(message);
}

class InactiveEncoder final : public parakeet::NpuEncoder {
public:
    int fixed_mel_frames() const override { return 4; }
    std::string backend_name() const override { return "inactive"; }
    void encode(const float *, int, int, int, parakeet::EncoderOutputs &) override {
        throw std::runtime_error("inactive test encoder must not run");
    }
};

void test_disabled() {
    parakeet::ParakeetCtcModel model;
    parakeet::NpuEncoderOptions options;
    model.npu_encoder = std::make_shared<InactiveEncoder>();
    parakeet::configure_npu_encoder(model, options, 1);
    check(!model.npu_encoder, "empty options retained the previous encoder");
    options.provider_options["config_file"] = "config.json";
    fails([&] { parakeet::configure_npu_encoder(model, options, 1); }, "orphan provider options accepted");
#ifndef PARAKEET_USE_RYZENAI
    options.model_path = "encoder.onnx";
    fails([&] { parakeet::configure_npu_encoder(model, options, 1); }, "disabled build accepted NPU request");
#endif
}

#ifdef PARAKEET_USE_RYZENAI
void test_provider_initialization(const std::string & fixture) {
    parakeet::ParakeetCtcModel model;
    model.mel_cfg.n_mels = 2;
    model.encoder_cfg.d_model = 2;
    parakeet::NpuEncoderOptions options;
    options.model_path = fixture;
    try {
        parakeet::make_onnx_encoder(model, options, parakeet::OnnxEncoderProvider::VitisAI, 1);
    } catch (const std::exception & error) {
        check(std::string(error.what()).find("DefaultLogger") == std::string::npos,
              "provider initialized before its ONNX Runtime environment");
    }
}

void test_encoding(const std::string & fixture) {
    parakeet::ParakeetCtcModel model;
    model.mel_cfg.n_mels = 2;
    model.encoder_cfg.d_model = 2;
    parakeet::NpuEncoderOptions options;
    options.model_path = fixture;
    auto encoder = parakeet::make_onnx_encoder(model, options, parakeet::OnnxEncoderProvider::CPU, 1);
    check(encoder->fixed_mel_frames() == 4, "wrong frame capacity");
    check(encoder->backend_name() == "ONNX-CPU", "wrong provider label");
    const std::vector<float> input{1, 10, 2, 20, 3, 30, 4, 40, 5, 50};
    parakeet::EncoderOutputs output;
    encoder->encode(input.data(), 3, 3, 2, output);
    check(output.encoder_out == std::vector<float>({1, 10, 2, 20, 3, 30}), "padded input layout changed");
    check(output.n_enc_frames == 3 && output.d_model == 2, "wrong valid output dimensions");
    encoder->encode(input.data(), 4, 2, 2, output);
    check(output.encoder_out == std::vector<float>({1, 10, 2, 20}), "encoded length was ignored");
    encoder->encode(input.data(), 4, 4, 2, output);
    check(output.encoder_out == std::vector<float>({1, 10, 2, 20, 3, 30, 4, 40}),
          "full input layout changed");
    fails([&] { encoder->encode(input.data(), 5, 5, 2, output); }, "oversized input accepted");
    fails([&] { encoder->encode(input.data(), 2, 3, 2, output); }, "invalid length accepted");
    fails([&] { encoder->encode(input.data(), 2, 0, 2, output); }, "zero length accepted");
    fails([&] { encoder->encode(input.data(), 2, 2, 3, output); }, "wrong mel count accepted");
    fails([&] { encoder->encode(nullptr, 2, 2, 2, output); }, "null input accepted");
    model.mel_cfg.n_mels = 3;
    fails([&] { parakeet::make_onnx_encoder(model, options, parakeet::OnnxEncoderProvider::CPU, 1); },
          "incompatible GGUF mel geometry accepted");
    model.mel_cfg.n_mels = 2;
    model.encoder_cfg.d_model = 3;
    fails([&] { parakeet::make_onnx_encoder(model, options, parakeet::OnnxEncoderProvider::CPU, 1); },
          "incompatible GGUF encoder geometry accepted");
    options.model_path += ".missing";
    fails([&] { parakeet::make_onnx_encoder(model, options, parakeet::OnnxEncoderProvider::CPU, 1); },
          "missing ONNX model accepted");
}

void test_recreated_sessions(const std::string & fixture) {
    test_encoding(fixture);
    test_encoding(fixture);
}
#endif

} // namespace

int main(int argc, char ** argv) {
    try {
        test_disabled();
#ifdef PARAKEET_USE_RYZENAI
        check(argc == 2, "fixture path required");
        test_provider_initialization(argv[1]);
        test_recreated_sessions(argv[1]);
#else
        (void)argc;
        (void)argv;
#endif
        std::puts("NPU encoder tests passed");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
