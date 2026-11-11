#pragma once

// Device tests that compare a Core ML sidecar against ggml wait for every
// sidecar at construction. Sidecars load in the background (PLAN §3.2.0), so
// without this the first calls would run on ggml while the load is in flight.

#include "parakeet/coreml_options.h"

inline parakeet::CoremlOptions coreml_ready_options() {
    parakeet::CoremlOptions options;
    options.wait = parakeet::CoremlOptions::Wait::construct;
    options.prewarm = parakeet::CoremlOptions::Prewarm::all;
    return options;
}
