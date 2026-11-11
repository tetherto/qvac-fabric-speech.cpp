#pragma once

// Core ML part of the Parakeet fit projection (PLAN §2.13). Pure and
// portable: the route comes only from options and adjacent sidecar presence,
// never from the Core ML cache or warm hints, so the projection is identical
// before and after the first load (test-parakeet-coreml-route "fit-stable").

#include "fit_util.h"
#include "parakeet/fit.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <string>
#include <system_error>

namespace parakeet {

// Inputs of the encoder sidecar envelope.
struct CoremlEncoderEnvelopeInputs {
    uint64_t weight_file_bytes = 0;    // <sidecar>/weights/weight.bin
    uint64_t encoder_parameters = 0;   // encoder.* parameters in the GGUF
    uint64_t io_bytes = 0;             // worst-window input + output, f32
    bool     ane = false;              // cpu_and_ane units
};

// §2.13 defaults until the model's PR measures them:
//  - resident runtime weights at their expanded fp16 geometry (a palettized
//    prebuilt is never charged at its compressed file size): the larger of
//    the weight file and 2 bytes per GGUF encoder parameter;
//  - I/O backing, input and output: 2 x io_bytes;
//  - one load transient: 0.25 x weight file on cpu_and_ane, 0.5 x otherwise;
//  - margin x1.25 on resident + transient.
inline uint64_t coreml_encoder_envelope_bytes(const CoremlEncoderEnvelopeInputs & in) {
    using fitutil::sat_add;
    using fitutil::sat_mul;
    const uint64_t runtime_weights =
        std::max(in.weight_file_bytes, sat_mul(in.encoder_parameters, (uint64_t) 2));
    const uint64_t resident = sat_add(runtime_weights, sat_mul(in.io_bytes, (uint64_t) 2));
    const uint64_t transient = in.ane ? in.weight_file_bytes / 4 : in.weight_file_bytes / 2;
    const uint64_t total = sat_add(resident, transient);
    return sat_add(total, total / 4);
}

// The encoder route fit charges: prebuilt when Core ML is enabled and the
// adjacent sidecar directory exists (today's path rules), none otherwise.
struct CoremlEncoderFitRoute {
    bool prebuilt = false;
    uint64_t weight_file_bytes = 0;
};

inline CoremlEncoderFitRoute coreml_encoder_fit_route(const std::string & sidecar_dir,
                                                      bool coreml_enabled) {
    CoremlEncoderFitRoute r;
    std::error_code ec;
    if (!coreml_enabled || !std::filesystem::is_directory(sidecar_dir, ec)) return r;
    r.prebuilt = true;
    const uintmax_t size = std::filesystem::file_size(
        std::filesystem::path(sidecar_dir) / "weights" / "weight.bin", ec);
    if (!ec) r.weight_file_bytes = static_cast<uint64_t>(size);
    return r;
}

// Fills the two verdicts from today's verdict. `required` is what today's
// verdict charged (it includes the diarizer sidecars when the device shares
// host memory); `encoder_envelope` is the encoder sidecar's envelope (0 with
// no route). The top-level `fits` is left as it is: prebuilt routes keep
// today's fit results until their model's PR validates them (§2.13).
inline void apply_coreml_verdicts(FitResult & r, uint64_t required, bool coreml_enabled,
                                  const std::string & disabled_reason, bool encoder_prebuilt,
                                  uint64_t encoder_envelope) {
    using fitutil::sat_add;
    const uint64_t diarizers = r.device_shares_host_memory ? r.coreml_bytes : 0;
    r.ggml_required_bytes = required >= diarizers ? required - diarizers : 0;
    r.ggml_only_fits = r.ggml_required_bytes <= r.device_free_bytes;
    r.coreml.encoder_route = encoder_prebuilt ? "prebuilt" : "none";
    r.coreml.validated = false;
    if (!coreml_enabled) {
        r.coreml.bytes = 0;
        r.coreml.disabled_reason = disabled_reason.empty() ? std::string("use=off") : disabled_reason;
        r.coreml.fits = r.ggml_only_fits;
        return;
    }
    r.coreml.disabled_reason.clear();
    r.coreml.bytes = sat_add(diarizers, encoder_prebuilt ? encoder_envelope : 0);
    r.coreml.fits = sat_add(r.ggml_required_bytes, r.coreml.bytes) <= r.device_free_bytes;
}

}  // namespace parakeet
