#pragma once

// Core ML part of the Parakeet fit projection. Pure and
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
#include <vector>

namespace parakeet {

// The resolved compute units of the encoder load (the runtime's
// ComputeUnits, mirrored here so the projection stays portable).
enum class CoremlFitUnits { all, cpu_and_ane, cpu_and_gpu, cpu_only };

// Inputs of the encoder sidecar envelope.
struct CoremlEncoderEnvelopeInputs {
    uint64_t       weight_file_bytes = 0;   // <sidecar>/weights/weight.bin
    uint64_t       encoder_parameters = 0;  // encoder.* parameters in the GGUF
    uint64_t       io_bytes = 0;            // worst-window input + output, f32
    CoremlFitUnits units = CoremlFitUnits::all;
};

// The client-side load transient (`load_peak`) by compute-unit class, until
// the model's own measurement replaces it:
//  - cpu_and_ane: 0.25 x weight file (cold ANE load peaked at 0.19x);
//  - all, cpu_and_gpu: 1.0 x weight file (update 2026-11-11: Parakeet TDT
//    0.6B fp16 measured +1.08 GiB over a 1.2 GB weight file, ~0.9x, and the
//    palettized prebuilt ~0.95x);
//  - cpu_only: 1.0 x as well. The CPU load builds the same expanded weights
//    and nothing has measured it lower, so it is charged like `all`.
inline uint64_t coreml_encoder_load_transient_bytes(uint64_t weight_file_bytes, CoremlFitUnits units) {
    switch (units) {
        case CoremlFitUnits::cpu_and_ane: return weight_file_bytes / 4;
        case CoremlFitUnits::all:
        case CoremlFitUnits::cpu_and_gpu:
        case CoremlFitUnits::cpu_only: return weight_file_bytes;
    }
    return weight_file_bytes;
}

// Defaults until the model's own measurements replace them:
//  - resident runtime weights at their expanded fp16 geometry (a palettized
//    prebuilt is never charged at its compressed file size): the larger of
//    the weight file and 2 bytes per GGUF encoder parameter;
//  - I/O backing, input and output: 2 x io_bytes;
//  - one load transient by unit class (above);
//  - margin x1.25 on resident + transient.
// One Core ML program the encoder stage keeps resident. Programs a model
// keeps loaded together (TranscribePsy EU's route programs, Sortformer's
// encoder and bypass bundles) are charged together.
struct CoremlFitProgram {
    uint64_t       weight_file_bytes = 0;    // <bundle>/weights/weight.bin
    uint64_t       runtime_floor_bytes = 0;  // expanded fp16 floor of a full-encoder program, else 0
    uint64_t       io_bytes = 0;             // worst-window input + output, f32
    int            functions = 1;            // functions loaded from the bundle, each charged its weights
    CoremlFitUnits units = CoremlFitUnits::all;
};

// Every resident program (runtime weights per loaded function, since loaded
// functions are not assumed to share weight memory until measured, plus
// 2 x I/O backing), one load transient (the largest: the runtime runs one
// Core ML load at a time), and the 1.25 margin on the sum.
inline uint64_t coreml_programs_envelope_bytes(const std::vector<CoremlFitProgram> & programs) {
    using fitutil::sat_add;
    using fitutil::sat_mul;
    uint64_t resident = 0;
    uint64_t transient = 0;
    for (const CoremlFitProgram & p : programs) {
        const uint64_t weights = std::max(p.weight_file_bytes, p.runtime_floor_bytes);
        resident = sat_add(resident, sat_mul(weights, (uint64_t) std::max(1, p.functions)));
        resident = sat_add(resident, sat_mul(p.io_bytes, (uint64_t) 2));
        transient = std::max(transient, coreml_encoder_load_transient_bytes(p.weight_file_bytes, p.units));
    }
    const uint64_t total = sat_add(resident, transient);
    return sat_add(total, total / 4);
}

inline uint64_t coreml_encoder_envelope_bytes(const CoremlEncoderEnvelopeInputs & in) {
    CoremlFitProgram p;
    p.weight_file_bytes = in.weight_file_bytes;
    p.runtime_floor_bytes = fitutil::sat_mul(in.encoder_parameters, (uint64_t) 2);
    p.io_bytes = in.io_bytes;
    p.units = in.units;
    return coreml_programs_envelope_bytes({p});
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

// A stage's Core ML charge in two parts: what stays resident, and the one
// load transient it adds while a load runs. The runtime runs one Core ML job
// per library image, so a model charges its stages' residents together and
// only the largest transient among them.
struct CoremlCharge {
    uint64_t resident = 0;
    uint64_t transient = 0;
};

// The encoder programs' charge, both parts with the 1.25 margin.
inline CoremlCharge coreml_programs_charge(const std::vector<CoremlFitProgram> & programs) {
    using fitutil::sat_add;
    using fitutil::sat_mul;
    CoremlCharge c;
    for (const CoremlFitProgram & p : programs) {
        const uint64_t weights = std::max(p.weight_file_bytes, p.runtime_floor_bytes);
        const uint64_t functions = (uint64_t) std::max(1, p.functions);
        c.resident = sat_add(c.resident, sat_mul(weights, functions));
        c.resident = sat_add(c.resident, sat_mul(sat_mul(p.io_bytes, (uint64_t) 2), functions));
        c.transient = std::max(c.transient, coreml_encoder_load_transient_bytes(p.weight_file_bytes, p.units));
    }
    c.resident = sat_add(c.resident, c.resident / 4);
    c.transient = sat_add(c.transient, c.transient / 4);
    return c;
}

// Fills the two verdicts. `required` is today's verdict charge (with the
// device sharing host memory it includes today's diarizer charge,
// r.coreml_bytes); the ggml-only total excludes that. The Core ML-inclusive
// total adds the residents of the stages Core ML would activate (`diarizers`
// and `encoder`, counted only on shared memory for the diarizers as today)
// and the largest single transient. The top-level `fits` is left as it is:
// prebuilt routes keep today's fit results until their model's PR validates
// them.
inline void apply_coreml_verdicts(FitResult & r, uint64_t required, bool coreml_enabled,
                                  const std::string & disabled_reason, const CoremlCharge & diarizers,
                                  bool encoder_prebuilt, const CoremlCharge & encoder) {
    using fitutil::sat_add;
    const uint64_t legacy_diarizers = r.device_shares_host_memory ? r.coreml_bytes : 0;
    r.ggml_required_bytes = required >= legacy_diarizers ? required - legacy_diarizers : 0;
    r.ggml_only_fits = r.ggml_required_bytes <= r.device_free_bytes;
    r.coreml.encoder_route = encoder_prebuilt ? "prebuilt" : "none";
    r.coreml.validated = false;
    if (!coreml_enabled) {
        r.coreml.bytes = 0;
        r.coreml.disabled_reason = disabled_reason.empty() ? std::string("use=off") : disabled_reason;
        r.coreml.fits = r.ggml_only_fits;
        return;
    }
    const CoremlCharge diar = r.device_shares_host_memory ? diarizers : CoremlCharge{};
    const CoremlCharge enc = encoder_prebuilt ? encoder : CoremlCharge{};
    r.coreml.disabled_reason.clear();
    r.coreml.bytes = sat_add(sat_add(diar.resident, enc.resident), std::max(diar.transient, enc.transient));
    r.coreml.fits = sat_add(r.ggml_required_bytes, r.coreml.bytes) <= r.device_free_bytes;
}

}  // namespace parakeet
