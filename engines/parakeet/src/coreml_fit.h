#pragma once

// Core ML part of the Parakeet fit projection. Pure and
// portable: the route comes only from options and adjacent sidecar presence,
// never from the Core ML cache or warm hints, so the projection is identical
// before and after the first load (test-parakeet-coreml-route "fit-stable").

#include "fit_util.h"
#include "parakeet/fit.h"
#include "parakeet_coreml_path.h"

#include <algorithm>
#include <cmath>
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
inline uint64_t coreml_encoder_envelope_bytes(const CoremlEncoderEnvelopeInputs & in) {
    using fitutil::sat_add;
    using fitutil::sat_mul;
    const uint64_t runtime_weights =
        std::max(in.weight_file_bytes, sat_mul(in.encoder_parameters, (uint64_t) 2));
    const uint64_t resident = sat_add(runtime_weights, sat_mul(in.io_bytes, (uint64_t) 2));
    const uint64_t transient = coreml_encoder_load_transient_bytes(in.weight_file_bytes, in.units);
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

// One diarizer capacity of the generated-diarizer charge. A prebuilt
// capacity (generated = false) only shapes which capacities a range loads
// (coreml_diarizer_bytes charges it); a generated one carries its
// template's unmargined parts for the resolved units.
struct CoremlDiarizerCharge {
    int      frames = 0;
    bool     generated = false;
    uint64_t resident_bytes = 0;
    uint64_t transient_bytes = 0;
    double   margin = 1.0;
};

// The Core ML envelope of the generated capacities that calls in `ranges`
// can load, selected over every capacity (prebuilt and generated) by
// coreml_diarizer_loaded_capacities, or of every generated capacity when
// `every_capacity` (prewarm = all loads them all at construction): the sum of
// their resident parts plus the largest single transient (prepares and loads
// run one at a time), times the largest margin among them. 0 when no
// generated capacity is loaded.
//
// A capacity that fails to load (or never loads: a status-only template)
// hands its calls to the smallest larger capacity that has not failed, which
// then loads in its place (DiarizerSlots skips terminal capacities). So each
// selected capacity is charged at the largest generated capacity its calls
// can reach that way: itself, or a larger one below the next selected
// capacity (which is charged on its own); above the largest selected one,
// every larger capacity.
inline uint64_t coreml_generated_diarizer_envelope(const std::vector<CoremlDiarizerCharge> & capacities,
                                                   const std::vector<CoremlFrameRange> & ranges,
                                                   bool every_capacity = false) {
    using fitutil::sat_add;
    std::vector<int> sorted;
    sorted.reserve(capacities.size());
    for (const CoremlDiarizerCharge & c : capacities) sorted.push_back(c.frames);
    std::sort(sorted.begin(), sorted.end());
    const std::vector<int> loaded = every_capacity ? sorted : coreml_diarizer_loaded_capacities(sorted, ranges);
    uint64_t resident = 0;
    uint64_t transient = 0;
    double margin = 1.0;
    for (size_t i = 0; i < loaded.size(); ++i) {
        const int from = loaded[i];
        const bool bounded = i + 1 < loaded.size();
        const int below = bounded ? loaded[i + 1] : 0;
        uint64_t slot = 0;
        for (const CoremlDiarizerCharge & c : capacities) {
            if (!c.generated || c.frames < from || (bounded && c.frames >= below)) continue;
            slot = std::max(slot, c.resident_bytes);
            transient = std::max(transient, c.transient_bytes);
            margin = std::max(margin, c.margin);
        }
        resident = sat_add(resident, slot);
    }
    const uint64_t total = sat_add(resident, transient);
    return total == 0 ? 0 : fitutil::sat_u64_from_double(std::ceil(static_cast<double>(total) * margin));
}

// Fills the two verdicts from today's verdict. `required` is what today's
// verdict charged (it includes the prebuilt diarizer sidecars when the device
// shares host memory); `encoder_envelope` is the encoder sidecar's envelope (0
// with no route); `generated_diarizers` is the Core ML envelope of the
// diarizer capacities on route generate (validated templates only).
// `coreml.validated` holds when generated capacities are charged and nothing
// else is: a prebuilt charge in the bytes is not a validated envelope. The
// top-level `fits` is left as it is: prebuilt routes keep today's fit results
// until their model's PR validates them, and a generated route never turns a
// working ggml verdict into a refusal (memory-degrade resolves use = off).
inline void apply_coreml_verdicts(FitResult & r, uint64_t required, bool coreml_enabled,
                                  const std::string & disabled_reason, bool encoder_prebuilt,
                                  uint64_t encoder_envelope, uint64_t generated_diarizers = 0) {
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
    r.coreml.bytes = sat_add(sat_add(diarizers, encoder_prebuilt ? encoder_envelope : 0), generated_diarizers);
    // Validated only when every charged byte comes from a validated template
    // envelope: prebuilt sidecars keep today's unvalidated charge.
    r.coreml.validated = generated_diarizers > 0 && diarizers == 0 && !encoder_prebuilt;
    r.coreml.fits = sat_add(r.ggml_required_bytes, r.coreml.bytes) <= r.device_free_bytes;
}

}  // namespace parakeet
