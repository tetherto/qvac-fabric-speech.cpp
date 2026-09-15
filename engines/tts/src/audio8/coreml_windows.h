#pragma once
// How the Core ML codec sidecar walks an utterance.
//
// The exported synthesis stack has one fixed input width, `window` frames of
// the post-transformer latent, and every convolution in it is causal, so a
// window's output is exact from the first frame whose receptive field lies
// inside the window: the first `context` frames of any window that does not
// start at zero only exist to give the causal convolutions real history, and
// are dropped, exactly as the ggml block path drops its carried context. A
// window that runs past the end of the utterance is zero-padded on the right,
// which changes nothing before the padding because nothing in the stack looks
// forward.
//
// Pure arithmetic, compiled on every platform so it can be unit-tested without
// Apple frameworks or a model.

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

// Every window is exactly `window` frames wide and lies inside [0, n_frames),
// except a single window on an utterance shorter than the width, which is
// zero-padded. The cores tile [0, n_frames) exactly, and every core after the
// first starts at least `context` frames into its window. Empty when the
// width cannot carry the context (context >= window) or there is nothing to
// synthesise.
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
