// Compute-unit selection shared by the Core ML encoder, diarizer and decoder
// wrappers.
//
// Apple-only, Objective-C++ header: included by the .mm wrappers under
// PARAKEET_USE_COREML only.

#pragma once

#import <CoreML/CoreML.h>

#include <cstdlib>
#include <cstring>
#include <string>

// Maps resolved compute units (PLAN B.5 vocabulary: `all`, `cpu_and_ane`,
// `cpu_and_gpu`, `cpu_only`) to MLComputeUnits and writes the matching stats
// label. The engine resolves the units once (CoremlOptions::compute_units,
// then PARAKEET_COREML_COMPUTE_UNITS, then the stage default) and passes them
// here; loaders never read the environment themselves (§2.9, §2.12).
//
// `units` null or unrecognised selects `prefer_neural_engine ? cpu_and_ane :
// all`: the graphs exported for the Neural Engine (the fixed-window encoder,
// the diarizers and the batched TDT decoder) are fully ANE-shaped, and
// MLComputeUnitsAll is only a preference that may resolve to the GPU. The
// generic encoder exports keep MLComputeUnitsAll, since their few
// GPU-preferred operations would otherwise fall back to the CPU.
inline MLComputeUnits parakeet_coreml_units_for(const char * units,
                                                bool prefer_neural_engine,
                                                std::string * label) {
    if (units != nullptr && std::strcmp(units, "all") == 0) {
        *label = "coreml-all";
        return MLComputeUnitsAll;
    }
    if (units != nullptr && std::strcmp(units, "cpu_and_gpu") == 0) {
        *label = "coreml-cpu-gpu";
        return MLComputeUnitsCPUAndGPU;
    }
    if (units != nullptr && std::strcmp(units, "cpu_only") == 0) {
        *label = "coreml-cpu";
        return MLComputeUnitsCPUOnly;
    }
    const bool want_ane =
        prefer_neural_engine || (units != nullptr && std::strcmp(units, "cpu_and_ane") == 0);
    if (want_ane) {
        if (@available(macOS 13.0, iOS 16.0, tvOS 16.0, watchOS 9.0, *)) {
            *label = "coreml-ane";
            return MLComputeUnitsCPUAndNeuralEngine;
        }
    }
    *label = "coreml-all";
    return MLComputeUnitsAll;
}

// The synchronous loaders that stay outside the shared runtime (the TDT
// decoder bundle and the MOSS-Transcribe encoder) still take their units from
// an environment variable; this reads it and defers to the function above.
inline MLComputeUnits parakeet_coreml_requested_units(const char * units_env,
                                                      bool prefer_neural_engine,
                                                      std::string * label) {
    const char * env = units_env != nullptr ? std::getenv(units_env) : nullptr;
    return parakeet_coreml_units_for(env, prefer_neural_engine, label);
}
