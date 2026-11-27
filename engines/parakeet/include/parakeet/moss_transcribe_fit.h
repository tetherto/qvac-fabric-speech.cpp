#pragma once

#include "parakeet/fit.h"
#include "parakeet/moss_transcribe.h"

namespace parakeet::moss {

// An active Core ML encoder sidecar adds a conservative memory envelope to
// the ggml projection. A bundle without a readable weights file, or a backend
// with a separate memory pool, returns "coreml-memory-estimate-unavailable".

PARAKEET_API FitResult fit_params(const TranscribeOptions & options, const TranscribeRequest & request,
                                 double audio_seconds, uint64_t margin_bytes);

}
