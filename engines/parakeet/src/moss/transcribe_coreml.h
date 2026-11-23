#pragma once

#include "encoder_route.h"
#include "moss/transcribe_model.h"
#include "parakeet/coreml_options.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace parakeet::moss::detail {

constexpr const char * COREML_DISABLE_ENV = "MOSS_COREML_DISABLE";
constexpr const char * COREML_STRICT_ENV = "MOSS_COREML_STRICT";
constexpr const char * COREML_UNITS_ENV = "MOSS_COREML_COMPUTE_UNITS";
constexpr const char * GGML_ENCODER_BACKEND = "ggml";
constexpr const char * MIXED_ENCODER_BACKEND = "mixed";

std::string transcribe_encoder_sidecar_path(const std::string & model_path);
std::vector<float> frame_major_mel(const std::vector<float> & mel, int n_mels, int frames);

// The encoder route one window acquires when it starts (its unit of work):
// a Core ML route when the sidecar is ready, otherwise the ggml route
// (marked pending while the sidecar is still generated or loaded).
using TranscribeRouteSource = std::function<EncoderRoute()>;

// The Core ML encoder sidecar of one model on the shared runtime: an adjacent
// <stem>-encoder.mlmodelc, or the encoder generated from the GGUF into the
// cache directory when there is none. It loads in the background after the
// model has loaded. Builds without PARAKEET_COREML keep the encoder on ggml
// and ignore the options.
class TranscribeCoremlEncoder {
public:
    // Before the model loads: records the GGUF's identity, so the encoder is
    // generated only from the file the model loaded its weights from.
    TranscribeCoremlEncoder(const std::string & model_path, const CoremlOptions & options, bool memory_degraded);
    ~TranscribeCoremlEncoder();
    TranscribeCoremlEncoder(const TranscribeCoremlEncoder &) = delete;
    TranscribeCoremlEncoder & operator=(const TranscribeCoremlEncoder &) = delete;

    // After the model loaded: registers the sidecar's slot for the model's
    // window and issues the construction group (with wait = construct,
    // waits for it). `cancel` (the engine's flag) stops a window's wait.
    void attach(const TranscribeConfig & config, const std::atomic<bool> * cancel);

    // One window's route: raised to `now`, then the resolved wait mode
    // (none: no blocking; call: until ready or terminal; use = require throws
    // when the sidecar cannot serve).
    EncoderRoute acquire();
    // The sidecar is loaded and ready now (never requests, waits or throws).
    bool ready() const;

    std::string status_json() const;
    std::string construction_json() const;
    bool wait(int64_t timeout_ms);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Encodes the 30 s windows of a run: each window acquires its route; a route
// that holds the sidecar runs the window on Core ML, any other on ggml. A
// failed prediction retires the sidecar and reruns the window on ggml, so the
// run reports "mixed". Under use = require a window the sidecar cannot serve
// throws instead.
class TranscribeAudioEncoder {
public:
    TranscribeAudioEncoder(TranscribeModel & model, TranscribeRouteSource routes);

    void begin_run();
    std::vector<float> encode(const std::vector<float> & mel, int tokens);
    // "" before any window; otherwise "ggml", the Core ML label or "mixed".
    std::string run_backend() const;
    // A window of this run ran on ggml because the sidecar was not ready.
    bool run_pending() const;

private:
    EncoderEligibility eligibility(const EncoderRoute & route) const;
    std::vector<float> leading_rows(const std::vector<float> & rows, int tokens) const;

    TranscribeModel & model_;
    TranscribeRouteSource routes_;
    EncoderBackendTally tally_;
};

} // namespace parakeet::moss::detail
