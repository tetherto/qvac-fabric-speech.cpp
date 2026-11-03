#include "parakeet_npu.h"
#include "parakeet_ctc.h"

#include <stdexcept>

#ifdef PARAKEET_USE_RYZENAI
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <limits>
#include <unordered_map>
#include <vector>

namespace parakeet {
namespace {

constexpr const char * kSignal = "audio_signal";
constexpr const char * kLength = "length";
constexpr const char * kOutputs = "outputs";
constexpr const char * kOutputLength = "encoded_lengths";

Ort::SessionOptions session_options(const NpuEncoderOptions & options, OnnxEncoderProvider provider,
                                    int n_threads) {
    Ort::SessionOptions session;
    if (n_threads > 0) session.SetIntraOpNumThreads(n_threads);
    if (provider == OnnxEncoderProvider::VitisAI) {
        std::unordered_map<std::string, std::string> values(options.provider_options.begin(),
                                                            options.provider_options.end());
        session.AppendExecutionProvider_VitisAI(values);
    }
    return session;
}

Ort::Session create_session(Ort::Env & environment, const NpuEncoderOptions & options,
                            OnnxEncoderProvider provider, int n_threads) {
    auto configuration = session_options(options, provider, n_threads);
    const auto path = std::filesystem::u8path(options.model_path);
    return Ort::Session(environment, path.c_str(), configuration);
}

void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error("parakeet ONNX encoder: " + message);
}

size_t tensor_index(const Ort::Session & session, bool input, const char * name) {
    Ort::AllocatorWithDefaultOptions allocator;
    const auto count = input ? session.GetInputCount() : session.GetOutputCount();
    for (size_t index = 0; index < count; ++index) {
        auto actual_name = input ? session.GetInputNameAllocated(index, allocator)
                                 : session.GetOutputNameAllocated(index, allocator);
        if (std::string(actual_name.get()) == name) return index;
    }
    throw std::runtime_error(std::string("parakeet ONNX encoder: missing tensor '") + name + "'");
}

std::vector<int64_t> validate_tensor(const Ort::Session & session, bool input, const char * name,
                                     ONNXTensorElementDataType type, size_t rank) {
    const auto index = tensor_index(session, input, name);
    auto information = input ? session.GetInputTypeInfo(index) : session.GetOutputTypeInfo(index);
    auto tensor = information.GetTensorTypeAndShapeInfo();
    require(tensor.GetElementType() == type, std::string("wrong element type for '") + name + "'");
    auto shape = tensor.GetShape();
    require(shape.size() == rank, std::string("wrong rank for '") + name + "'");
    return shape;
}

size_t element_count(int64_t frames, int64_t channels) {
    require(frames > 0 && channels > 0 &&
                static_cast<uint64_t>(frames) <= std::vector<float>().max_size() / channels,
            "tensor dimensions exceed addressable storage");
    return static_cast<size_t>(frames) * static_cast<size_t>(channels);
}

std::vector<float> transpose_encoder(const float * source, int64_t frames, int d_model, int64_t stride) {
    std::vector<float> output(element_count(frames, d_model));
    for (int64_t t = 0; t < frames; ++t) {
        for (int d = 0; d < d_model; ++d) {
            output[static_cast<size_t>(t) * d_model + d] = source[d * stride + t];
        }
    }
    return output;
}

class OnnxEncoder final : public NpuEncoder {
public:
    OnnxEncoder(const ParakeetCtcModel & model, const NpuEncoderOptions & options,
                OnnxEncoderProvider provider, int n_threads)
        : environment_(ORT_LOGGING_LEVEL_WARNING, "parakeet-npu"),
          session_(create_session(environment_, options, provider, n_threads)), n_mels_(model.mel_cfg.n_mels),
          d_model_(model.encoder_cfg.d_model), provider_(provider) {
        require(n_mels_ > 0 && d_model_ > 0, "invalid GGUF encoder dimensions");
        require(session_.GetInputCount() == 2 && session_.GetOutputCount() >= 2,
                "expected audio_signal, length -> outputs, encoded_lengths");
        const auto signal = validate_tensor(session_, true, kSignal, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 3);
        const auto length = validate_tensor(session_, true, kLength, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, 1);
        const auto output =
            validate_tensor(session_, false, kOutputs, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 3);
        const auto output_length =
            validate_tensor(session_, false, kOutputLength, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, 1);
        require(signal[0] == 1 && signal[1] == n_mels_, "input geometry does not match the GGUF");
        require(output[0] == 1 && output[1] == d_model_, "output geometry does not match the GGUF");
        require(length[0] == 1 && output_length[0] == 1, "expected one sequence length");
        require(signal[2] == -1 || (signal[2] > 0 && signal[2] <= std::numeric_limits<int>::max()),
                "invalid mel frame dimension");
        fixed_frames_ = signal[2] > 0 ? static_cast<int>(signal[2]) : 0;
    }

    int fixed_mel_frames() const override { return fixed_frames_; }

    std::string backend_name() const override {
        return provider_ == OnnxEncoderProvider::VitisAI ? "VitisAI" : "ONNX-CPU";
    }

    void encode(const float * mel, int frames, int valid_frames, int n_mels,
                EncoderOutputs & output) override {
        require(mel && n_mels == n_mels_ && frames > 0 && valid_frames > 0 && valid_frames <= frames,
                "invalid mel input");
        const int capacity = fixed_frames_ > 0 ? fixed_frames_ : frames;
        require(frames <= capacity, "input exceeds the ONNX encoder's frame dimension");
        auto signal = transpose_mel(mel, frames, n_mels, capacity);
        int64_t length = valid_frames;
        const std::array<int64_t, 3> signal_shape{1, n_mels, capacity};
        const std::array<int64_t, 1> length_shape{1};
        auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::array<Ort::Value, 2> inputs{
            Ort::Value::CreateTensor<float>(memory, signal.data(), signal.size(), signal_shape.data(),
                                            signal_shape.size()),
            Ort::Value::CreateTensor<int64_t>(memory, &length, 1, length_shape.data(), length_shape.size())};
        const std::array<const char *, 2> names{kSignal, kLength};
        const std::array<const char *, 2> outputs{kOutputs, kOutputLength};
        auto result = session_.Run(Ort::RunOptions{nullptr}, names.data(), inputs.data(), inputs.size(),
                                   outputs.data(), outputs.size());
        read_output(result, output);
    }

private:
    static std::vector<float> transpose_mel(const float * mel, int frames, int n_mels, int capacity) {
        std::vector<float> signal(element_count(capacity, n_mels), 0.0f);
        for (int m = 0; m < n_mels; ++m) {
            for (int t = 0; t < frames; ++t) {
                signal[static_cast<size_t>(m) * capacity + t] = mel[static_cast<size_t>(t) * n_mels + m];
            }
        }
        return signal;
    }

    void read_output(const std::vector<Ort::Value> & result, EncoderOutputs & output) const {
        const auto shape = result[0].GetTensorTypeAndShapeInfo().GetShape();
        require(shape.size() == 3 && shape[0] == 1 && shape[1] == d_model_ && shape[2] > 0,
                "invalid encoder result geometry");
        require(result[1].GetTensorTypeAndShapeInfo().GetElementCount() == 1,
                "invalid encoded_lengths result");
        const int64_t frames = result[1].GetTensorData<int64_t>()[0];
        require(frames > 0 && frames <= shape[2] && frames <= std::numeric_limits<int>::max(),
                "invalid encoded length");
        output = EncoderOutputs{};
        output.d_model = d_model_;
        output.n_enc_frames = static_cast<int>(frames);
        output.encoder_out = transpose_encoder(result[0].GetTensorData<float>(), frames, d_model_, shape[2]);
    }

    Ort::Env environment_;
    Ort::Session session_;
    int n_mels_;
    int d_model_;
    int fixed_frames_ = 0;
    OnnxEncoderProvider provider_;
};

} // namespace
} // namespace parakeet
#endif

namespace parakeet {

std::shared_ptr<NpuEncoder> make_onnx_encoder(const ParakeetCtcModel & model,
                                              const NpuEncoderOptions & options, OnnxEncoderProvider provider,
                                              int n_threads) {
#ifdef PARAKEET_USE_RYZENAI
    try {
        return std::make_shared<OnnxEncoder>(model, options, provider, n_threads);
    } catch (const Ort::Exception & error) {
        throw std::runtime_error(std::string("parakeet ONNX encoder: ") + error.what());
    }
#else
    (void)model;
    (void)options;
    (void)provider;
    (void)n_threads;
    throw std::runtime_error("AMD NPU encoder requires a build with PARAKEET_RYZENAI=ON");
#endif
}

} // namespace parakeet
