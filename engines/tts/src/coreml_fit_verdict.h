#pragma once

// The Core ML part of a fit verdict, shared by the tts engines whose stages run
// on the shared Core ML runtime (Parler's DAC, Supertonic's vocoder).
//
// `required` is the ggml projection (which keeps the stage's ggml residency:
// the ggml stage serves calls until the sidecar is ready); a generated route
// adds the stage's Core ML envelope. With a budget, a generated route decides
// every verdict on one capacity, min(device_free, budget): both projections,
// the memory-degrade (the Core ML-inclusive requirement exceeds it and the
// ggml one fits it: the engine resolves use = off, so Core ML never turns a
// working ggml verdict into a refusal) and the final verdict. Without a budget
// or a generated route the capacity is device_free and nothing degrades.

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
    uint64_t charged    = 0;      // what `fits` compares with `capacity`
    uint64_t capacity   = 0;      // device_free, or min(device_free, budget)
};

inline coreml_fit_verdict decide_coreml_fit(uint64_t required, uint64_t coreml_bytes, bool generate,
                                            uint64_t device_free, const std::optional<int64_t> & budget) {
    coreml_fit_verdict v;
    const uint64_t with_coreml = fitutil::sat_add(required, generate ? coreml_bytes : 0);
    v.capacity = device_free;
    if (generate && budget.has_value()) {
        v.capacity = std::min(device_free, (uint64_t) std::max<int64_t>(*budget, 0));
    }
    v.ggml_only_fits = required <= v.capacity;
    v.coreml_fits = with_coreml <= v.capacity;
    v.degraded = generate && budget.has_value() && !v.coreml_fits && v.ggml_only_fits;
    v.charged = generate && !v.degraded ? with_coreml : required;
    v.fits = v.charged <= v.capacity;
    return v;
}

}  // namespace detail
}  // namespace tts_cpp
