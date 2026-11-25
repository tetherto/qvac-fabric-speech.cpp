#pragma once

// The Core ML-inclusive fit verdict of the ACE-Step VAE decoder, model-free so
// it is unit-testable: the ggml projection's budget checks and the same checks
// with the decoder's envelope added, plus the optional memory budget.

#include "acestep/fit_pools.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

namespace tts_cpp::acestep {

// Per pipeline phase, what the generated decoder adds to host memory.
struct CoremlPhaseCharge {
    uint64_t every_phase = 0;  // charged in every phase
    uint64_t vae_decode  = 0;  // charged in the VAE decode phase only
};

// The transient (a generation or load can outlive the VAE phase) in every
// phase; the resident decoder in the VAE decode phase only, unless it is held
// in every phase (every stage resident).
inline CoremlPhaseCharge acestep_coreml_phase_charge(bool generate, uint64_t resident, uint64_t transient,
                                                     bool resident_in_every_phase) {
    CoremlPhaseCharge c;
    if (!generate) return c;
    c.every_phase = fitutil::sat_add(transient, resident_in_every_phase ? resident : 0);
    c.vae_decode  = resident_in_every_phase ? 0 : resident;
    return c;
}

struct CoremlFitVerdict {
    bool ggml_only_fits = true;
    bool coreml_fits    = true;
    bool degraded       = false;  // memory-degrade: use = off, the ggml verdict
    bool fits           = true;   // the verdict reported
};

// `ggml` and `coreml` are fit_budget_checks() of the two projections; the
// first entry is the shared host pool. With a generated route and a budget,
// the budget caps that pool's free bytes for both verdicts before either is
// decided; memory-degrade (use = off) applies only when the capped ggml
// verdict holds and the capped Core ML-inclusive one does not, and the
// verdict reported is the one of the projection the engine will run. Without
// a generated route the ggml verdict is today's, uncapped.
inline CoremlFitVerdict acestep_decide_coreml_fit(const std::vector<FitBudgetCheck> & ggml,
                                                  const std::vector<FitBudgetCheck> & coreml, bool generate,
                                                  const std::optional<int64_t> & budget) {
    const bool capped = generate && budget.has_value();
    auto hold = [&](std::vector<FitBudgetCheck> checks) {
        if (capped && !checks.empty()) {
            checks[0].free_bytes = std::min<uint64_t>(checks[0].free_bytes, (uint64_t) std::max<int64_t>(*budget, 0));
        }
        return fit_checks_hold(checks);
    };
    CoremlFitVerdict v;
    v.ggml_only_fits = hold(ggml);
    v.coreml_fits    = generate ? hold(coreml) : v.ggml_only_fits;
    v.degraded       = capped && !v.coreml_fits && v.ggml_only_fits;
    v.fits           = generate && !v.degraded ? v.coreml_fits : v.ggml_only_fits;
    return v;
}

}  // namespace tts_cpp::acestep
