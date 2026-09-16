#pragma once
// Fixed-width window plan for the Core ML codec sidecar; see docs/audio8.md
// ("Core ML codec sidecar") for why dropping `context` leading frames and
// zero-padding on the right are exact.

#include <vector>

namespace tts_cpp {
namespace audio8 {
namespace detail {

struct coreml_window {
    int begin = 0;       // first post frame fed to the sidecar
    int filled = 0;      // frames of real data in the window; the rest is zero
    int core_begin = 0;  // frames the window's output is kept for: [core_begin, core_end)
    int core_end = 0;
};

// Cores tile [0, n_frames); every core after the first starts >= context frames
// into its window. Empty when context >= window or n_frames <= 0.
inline std::vector<coreml_window> plan_coreml_windows(int n_frames, int window, int context) {
    std::vector<coreml_window> plan;
    if (n_frames <= 0 || window <= 0 || context < 0 || context >= window) return plan;
    if (n_frames <= window) {
        plan.push_back({0, n_frames, 0, n_frames});
        return plan;
    }
    int core_begin = 0;
    for (;;) {
        int begin = core_begin == 0 ? 0 : core_begin - context;
        if (begin + window >= n_frames) {
            begin = n_frames - window;
            plan.push_back({begin, window, core_begin, n_frames});
            return plan;
        }
        plan.push_back({begin, window, core_begin, begin + window});
        core_begin = begin + window;
    }
}

}  // namespace detail
}  // namespace audio8
}  // namespace tts_cpp
