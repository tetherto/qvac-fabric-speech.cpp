// Equivalence and cache-integrity for the fast-AR graph cache against the
// dual-path dispatch in src/sched_dispatch.{h,cpp}.
//
// fast_pass keeps one built graph per position and replays it, which is only
// sound on the direct path: ggml_backend_sched_alloc_graph resets one shared
// arena and rewrites node->src[] in place, so a graph handed to the scheduler
// cannot be kept, and a kept graph cannot be handed to the scheduler. The same
// hazard is spelled out for T3 in test_t3_sched_equivalence.cpp.
//
// Sequence, same model and same inputs each phase:
//   A   direct path, which fills the cache
//   B   TTS_CPP_FORCE_SCHED=1, which must bypass the cache and still run
//   A'  direct again, which must still match A -- the cache survived B
//
// Logits are compared byte for byte. Bit-exactness is the bar; do not relax it
// to a tolerance, because both paths run the same graph on the same backend.
//
// This is a regression net for the class, not a proof of the guard: on a
// backend that supports every node, phase B reaches the scheduler only because
// fast_pass consults the force hook, and a build that ignored the hook would
// still pass by staying on the direct path. What caught the original defect --
// one allocator shared by every cached graph, so reserving a larger one dangled
// the rest -- was test-audio8-{engine,timing,cli-verbose} segfaulting.

#include "audio8/internal.h"
#include "audio8/sampling.h"
#include "gpu_arm.h"
#include "test_env_portable.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace tts_cpp::audio8::detail;

namespace {

constexpr int GPU_LAYERS = 99;
constexpr int THREADS = 2;

int failures = 0;

void fail(const std::string & what) {
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
}

int requested_layers() {
    return audio8_test::is_gpu_test() ? GPU_LAYERS : 0;
}

// Every position's logits, in order, as the frame produced them.
using logit_trace = std::vector<std::vector<float>>;

// The picker records what it was handed and always answers 0, so the frame
// walks the same positions with the same inputs in every phase.
code_picker recording_picker(logit_trace & trace) {
    return [&trace](const std::vector<float> & logits, int) {
        trace.push_back(logits);
        return 0;
    };
}

bool run_frame(lm_model & model, logit_trace & trace) {
    std::vector<int32_t> codes;
    std::vector<float> prime(static_cast<size_t>(model.hp.fast_hidden), 0.0f);
    std::string error;
    if (!fast_step(model, prime, model.hp.semantic_begin, THREADS,
                   recording_picker(trace), codes, &error)) {
        fail("fast_step failed: " + error);
        return false;
    }
    return true;
}

bool same_trace(const logit_trace & left, const logit_trace & right) {
    if (left.size() != right.size()) return false;
    for (size_t index = 0; index < left.size(); ++index) {
        if (left[index].size() != right[index].size()) return false;
        if (std::memcmp(left[index].data(), right[index].data(),
                        left[index].size() * sizeof(float)) != 0) {
            return false;
        }
    }
    return true;
}

void expect_same(const logit_trace & got, const logit_trace & want, const char * what) {
    if (same_trace(got, want)) return;
    fail(std::string(what) + ": fast-AR logits differ");
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <lm.gguf>\n", argv[0]);
        return 1;
    }

    lm_model model;
    std::string error;
    if (!load_lm(argv[1], requested_layers(), model, &error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    if (!audio8_test::check_requested_gpu("fast cache sched", model.backend)) {
        free_lm(model);
        return 1;
    }

    logit_trace direct, scheduled, direct_again;
    unsetenv("TTS_CPP_FORCE_SCHED");
    const bool ok_a = run_frame(model, direct);
    setenv("TTS_CPP_FORCE_SCHED", "1", 1);
    const bool ok_b = run_frame(model, scheduled);
    unsetenv("TTS_CPP_FORCE_SCHED");
    const bool ok_c = run_frame(model, direct_again);

    if (ok_a && ok_b) expect_same(scheduled, direct, "forced scheduler");
    if (ok_a && ok_c) expect_same(direct_again, direct, "direct after scheduler");
    if (direct.empty()) fail("the frame produced no fast-AR positions");

    std::printf("backend: %s, %zu positions compared\n", ggml_backend_name(model.backend),
                direct.size());
    free_lm(model);

    if (failures == 0) {
        std::fprintf(stderr, "audio8 fast cache sched: PASS\n");
        return 0;
    }
    std::fprintf(stderr, "audio8 fast cache sched: %d failure(s)\n", failures);
    return 1;
}
