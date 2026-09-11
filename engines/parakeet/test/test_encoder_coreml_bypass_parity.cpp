// Sortformer v2.1 AOSC Core ML block-stack parity against the ggml bypass path.

#include "parakeet_ctc.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <sortformer-v2.1.gguf> [min-cos]\n", argv[0]);
        return 2;
    }
    const double min_cos = argc > 2 ? std::atof(argv[2]) : 0.99;

    parakeet::ParakeetCtcModel model;
    if (int rc = parakeet::load_from_gguf(argv[1], model, 0, 999, true); rc != 0) {
        return 3;
    }
    if (!parakeet::model_bypass_encoder_on_coreml(model)) {
        std::fprintf(stderr, "[coreml-bypass-parity] SKIP: bypass sidecar inactive\n");
        return 0;
    }
    const int capacity = parakeet::model_coreml_bypass_fixed_frames(model);
    const int D = model.encoder_cfg.d_model;
    if (capacity <= 0 || D <= 0) return 4;

    const int frame_counts[] = {capacity, capacity > 8 ? capacity - 7 : capacity};
    for (const int T : frame_counts) {
        std::vector<float> input((size_t) T * D);
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = 1.0e-3f * (float) ((int) (i * 2654435761u) % 1000 - 500);
        }

        parakeet::EncoderOutputs ggml;
        if (int rc = parakeet::run_encoder_bypass_pre_encode(
                model, input.data(), T, D, ggml,
                /*max_layers=*/model.encoder_cfg.n_layers); rc != 0) return 5;
        if (ggml.used_coreml) return 6;

        parakeet::EncoderOutputs coreml;
        if (int rc = parakeet::run_encoder_bypass_pre_encode(
                model, input.data(), T, D, coreml); rc != 0) return 7;
        if (!coreml.used_coreml || coreml.encoder_out.size() != ggml.encoder_out.size()) return 8;

        double dot = 0.0, na = 0.0, nb = 0.0;
        for (size_t i = 0; i < ggml.encoder_out.size(); ++i) {
            const double a = ggml.encoder_out[i];
            const double b = coreml.encoder_out[i];
            if (!std::isfinite(b)) return 9;
            dot += a * b;
            na += a * a;
            nb += b * b;
        }
        const double cosine = dot / (std::sqrt(na) * std::sqrt(nb));
        std::fprintf(stderr, "[coreml-bypass-parity] T=%d/%d D=%d cosine=%.6f\n",
                     T, capacity, D, cosine);
        if (cosine < min_cos) return 1;
    }
    return 0;
}
