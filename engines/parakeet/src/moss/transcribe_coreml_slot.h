#pragma once

// MOSS-Transcribe's audio encoder sidecar on the shared Core ML runtime: the
// slot request (an adjacent prebuilt bundle, or the embedded template
// generated from the GGUF when there is none) and its fit charge.
//
// Portable C++ against the runtime's namespace, like the rest of Parakeet's
// runtime glue: no Core ML here, so the routing and the fit are tested on
// every platform with fake loaders.

#include "coreml/parakeet_coreml_sidecars.h"
#include "coreml_fit.h"

#include <cstdint>
#include <string>
#include <vector>

namespace parakeet::moss::detail {

namespace pcm = ::parakeet::pcm;

// The options resolve with MOSS_COREML_DISABLE, MOSS_COREML_STRICT and
// MOSS_COREML_COMPUTE_UNITS.
constexpr const char * TRANSCRIBE_COREML_ENV_PREFIX = "MOSS";
// The embedded template of the encoder and adaptor: one 30 s window,
// 3000 mel frames in, 375 adaptor tokens out.
constexpr const char * TRANSCRIBE_ENCODER_TEMPLATE = "moss-transcribe-enc-f3000";
constexpr const char * TRANSCRIBE_ENCODER_STAGE = "encoder";

// The window one prediction takes and returns. The loader accepts a bundle
// only when its input and output are exactly this window.
struct TranscribeEncoderWindow {
    int n_mels = 0;  // mel bands
    int frames = 0;  // mel frames per window
    int tokens = 0;  // adaptor tokens per window
    int width = 0;   // embedding width
};

// The slot's geometry: the loader kind, then the window. It is part of the
// runtime's sharing key, so only loaders that accept the same window share a
// model.
std::vector<int64_t> transcribe_encoder_geometry(const TranscribeEncoderWindow & window);
// The window a LoadRequest's geometry carries; false when it is not one.
bool transcribe_encoder_window_of(const std::vector<int64_t> & geometry, TranscribeEncoderWindow & window);

// Bytes of one window's input and output at f32 (the I/O the fit charges for
// a prebuilt bundle).
uint64_t transcribe_encoder_io_bytes(const TranscribeEncoderWindow & window);

// The embedded template probed against `gguf_path`. Nothing is read, and the
// candidate is unavailable, when the build has no generator or Core ML is
// disabled for this instance.
pcm::generation::Candidate probe_transcribe_encoder_template(const CoremlSidecars & sidecars,
                                                             const std::string & gguf_path);

// The slot request: stage "encoder", the adjacent <stem>-encoder.mlmodelc,
// the window as the loader's geometry, the probed template (null: prebuilt
// only) and the identity of the GGUF the model loaded its weights from. The
// stage's default compute units are `all` (today's prebuilt default), or the
// template's default when only the template can serve the stage.
SidecarRequest transcribe_encoder_request(const std::string & gguf_path, const TranscribeEncoderWindow & window,
                                          const pcm::generation::Candidate * generated, pcm::LoadFn load_fn,
                                          const pcm::SourceStat & source);

// The encoder's Core ML charge for fit, from the route plan() gives the
// request (never from the cache, so it is the same before and after the
// first load).
struct TranscribeEncoderFit {
    bool enabled = false;        // Core ML is enabled for this instance
    std::string disabled_reason; // why not ("build", "use=off", MOSS_COREML_DISABLE, "memory", ...)
    pcm::Route route = pcm::Route::none;
    // Resident and the one load transient, margin applied: a prebuilt bundle
    // at its expanded fp16 geometry with the 1.25 margin (the defaults until
    // its envelope is measured); a generated one from its template's
    // `memory` section.
    CoremlCharge charge;
    bool validated = false;      // a generated route: the template's measured envelope
};

TranscribeEncoderFit transcribe_encoder_fit(const CoremlSidecars & sidecars, const std::string & gguf_path,
                                            const TranscribeEncoderWindow & window, uint64_t encoder_parameters);

}  // namespace parakeet::moss::detail
