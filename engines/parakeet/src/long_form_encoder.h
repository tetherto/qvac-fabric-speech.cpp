#pragma once

#include "long_form.h"
#include "parakeet/engine.h"
#include "parakeet_ctc.h"

#include <atomic>

namespace parakeet {

struct WindowedEncoderStats {
    double encoder_ms = 0.0;
    int coreml_windows = 0;
    int ggml_windows = 0;
};

LongFormPlan resolve_long_form_plan(const ParakeetCtcModel & model,
                                    const EngineOptions & opts,
                                    int n_mel_frames);

int run_encoder_windowed(ParakeetCtcModel & model,
                         const float * mel, int n_mel_frames, int n_mels,
                         const LongFormPlan & plan,
                         std::atomic<bool> & cancel_flag,
                         EncoderOutputs & out,
                         WindowedEncoderStats & stats);

}  // namespace parakeet
