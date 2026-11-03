#pragma once

#include "parakeet/npu.h"
#include "parakeet_ctc.h"

#include <memory>
#include <stdexcept>
#include <string>

namespace parakeet {

struct EncoderOutputs;
struct ParakeetCtcModel;

class NpuEncoder {
public:
    virtual ~NpuEncoder() = default;
    virtual int fixed_mel_frames() const = 0;
    virtual std::string backend_name() const = 0;
    virtual void encode(const float * mel, int frames, int valid_frames, int n_mels,
                        EncoderOutputs & output) = 0;
};

enum class OnnxEncoderProvider { VitisAI, CPU };

std::shared_ptr<NpuEncoder> make_onnx_encoder(const ParakeetCtcModel & model,
                                              const NpuEncoderOptions & options, OnnxEncoderProvider provider,
                                              int n_threads);

inline void configure_npu_encoder(ParakeetCtcModel & model, const NpuEncoderOptions & options,
                                  int n_threads) {
    if (options.model_path.empty()) {
        if (!options.provider_options.empty()) {
            throw std::invalid_argument("NPU provider options require an ONNX encoder model_path");
        }
        model.npu_encoder.reset();
        return;
    }
#ifdef PARAKEET_USE_RYZENAI
    model.npu_encoder = make_onnx_encoder(model, options, OnnxEncoderProvider::VitisAI, n_threads);
#else
    (void)n_threads;
    throw std::runtime_error("AMD NPU encoder requires a build with PARAKEET_RYZENAI=ON");
#endif
}

} // namespace parakeet
