#pragma once
#include "tts-cpp/pocket/engine.h"
#include "tts-cpp/fit.h"

namespace tts_cpp::pocket {
struct FitOptions {
    EngineOptions engine;
    // Uses the real tokenizer/chunker and the runtime's maximum frame count.
    // The result bounds native batch synthesis, and therefore native streaming.
    std::string text;
    uint64_t margin_bytes = 256ull * 1024 * 1024;
    // Optional application ceiling, further limited by current available RAM.
    uint64_t memory_budget_bytes = 0;
};

// CPU-only memory preflight. GGUF tensor metadata is read, never tensor data;
// actual runtime graphs are priced without allocating or executing their buffers.
// Prepared voice metadata and frontend JSON are validated. Reference WAV headers
// are inspected (uncompressed WAV, 1..64 channels, at most 64 MiB / 30 seconds).
// This is a memory projection, not validation of weight/voice payloads or quality.
// Host estimates conservatively include load staging, container overhead, thread
// stacks, and native PCM buffers. SDK/JS copies and application buffers are outside
// this native engine projection. Never throws; see tts-cpp/fit.h for status codes.
TTS_CPP_API FitResult fit_params(const FitOptions & options);
}
