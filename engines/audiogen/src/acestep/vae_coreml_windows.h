#pragma once

#include <vector>

namespace tts_cpp::acestep {

struct VaeCoremlWindow {
    int win_a;
    int core_a;
    int core_b;
};

inline int vae_coreml_window_overlap(int window_frames) {
    constexpr int MAX_OVERLAP = 48;
    constexpr int MIN_OVERLAP = 8;
    const int scaled = (window_frames - MIN_OVERLAP) / 2;
    if (scaled >= MAX_OVERLAP) return MAX_OVERLAP;
    if (scaled < MIN_OVERLAP) return MIN_OVERLAP;
    return scaled;
}

inline std::vector<VaeCoremlWindow> vae_coreml_plan_windows(int T_latent, int window_frames, int overlap) {
    std::vector<VaeCoremlWindow> plan;
    if (T_latent < window_frames || window_frames <= 2 * overlap) {
        return plan;
    }
    const int core_stride = window_frames - 2 * overlap;
    int prev_core_b = 0;
    for (int i = 0;; ++i) {
        int win_a = i * core_stride;
        if (win_a > T_latent - window_frames) {
            win_a = T_latent - window_frames;
        }
        const int win_b  = win_a + window_frames;
        const int core_a = prev_core_b;
        const int core_b = (win_b == T_latent) ? T_latent : win_b - overlap;
        plan.push_back({win_a, core_a, core_b});
        prev_core_b = core_b;
        if (win_b == T_latent) {
            return plan;
        }
    }
}

}  // namespace tts_cpp::acestep
