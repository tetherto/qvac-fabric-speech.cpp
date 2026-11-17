#pragma once

// The Core ML part of a fit verdict, shared by the tts engines whose stages run
// on the shared Core ML runtime (Parler's DAC, Supertonic's vocoder).
//
// `required` is the ggml projection (which keeps the stage's ggml residency:
// the ggml stage serves calls until the sidecar is ready); a generated route
// adds the stage's Core ML envelope. Memory-degrade: with a budget, when the
// Core ML-inclusive requirement exceeds the capacity the verdict uses,
// min(device_free, budget), and the ggml one fits it, the engine resolves
// use = off, so Core ML never turns a working ggml verdict into a refusal. No
// budget, no memory-degrade.

#include "fit_util.h"

#include <algorithm>
#include <cstdint>
#include <optional>

namespace tts_cpp {
namespace detail {

struct coreml_fit_verdict {
    bool ggml_only_fits = false;
    bool coreml_fits    = false;
    bool degraded       = false;  // memory-degrade: use = off, reason "memory"
    bool fits           = false;  // the verdict for what the engine will run
    uint64_t charged    = 0;      // what `fits` compares with device_free
};

inline coreml_fit_verdict decide_coreml_fit(uint64_t required, uint64_t coreml_bytes, bool generate,
                                            uint64_t device_free, const std::optional<int64_t> & budget) {
    coreml_fit_verdict v;
    const uint64_t with_coreml = fitutil::sat_add(required, generate ? coreml_bytes : 0);
    v.ggml_only_fits = required <= device_free;
    v.coreml_fits = with_coreml <= device_free;
    if (generate && budget.has_value()) {
        const uint64_t cap = std::min(device_free, (uint64_t) std::max<int64_t>(*budget, 0));
        v.degraded = with_coreml > cap && required <= cap;
    }
    v.charged = generate && !v.degraded ? with_coreml : required;
    v.fits = v.charged <= device_free;
    return v;
}

}  // namespace detail
}  // namespace tts_cpp
