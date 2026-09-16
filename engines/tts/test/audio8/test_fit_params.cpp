// Fit-projection parity tests (include/tts-cpp/audio8/fit.h): assert that
// the metadata-only memory projection matches what a REAL load and REAL runs
// actually allocate, byte for byte where the projection is exact by
// construction.
//
// Two modes:
//
//   * No arguments (the always-on CI form): a tiny but complete audio8-lm
//     GGUF is synthesized on the fly (same trick as test_gguf_load.cpp), so
//     the LM-side parity gates run with no model fixture at all:
//       1. projected LM weight bytes == the buffer a real load_lm allocates;
//       2. projected KV-slab bytes == the slow+fast cache buffers;
//       3. projected slow arena (max of prompt prefill and deepest decode
//          step) == the gallocr buffer real slow_step calls reserve;
//       4. projected fast arena == the gallocr buffer real fast_step
//          reserves;
//       5. a missing / wrong-architecture model is Error, never Success.
//
//   * With arguments <lm.gguf> <decoder.gguf> [encoder.gguf] [n_gpu_layers]:
//     the full-pipeline gates on real fixtures -- everything above via
//     fit_params, plus codec parity: projected decode arenas (latent +
//     planned synthesis block) == what a real decode_codes leaves in
//     model.allocr / model.block_allocr; and near-INT_MAX workloads
//     (prompt_tokens, reference_seconds) are Error/"workload-too-large",
//     never a sign-overflowed graph shape.
//
// Exit 0 on success; non-zero with a FAIL line per broken invariant.

#include "tts-cpp/audio8/fit.h"

#include "test_env_portable.h"
#include "tiny_lm.h"

#include "audio8/graph.h"
#include "audio8/internal.h"
#include "fit_price.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace tts_cpp::audio8::detail;

namespace {

int g_failures = 0;

void fail(const std::string & what) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
}

void expect(bool cond, const std::string & what) {
    if (!cond) fail(what);
}

void expect_eq(uint64_t projected, uint64_t real, const std::string & what) {
    if (projected != real) {
        fail(what + ": projected " + std::to_string(projected) +
             " != allocated " + std::to_string(real));
    }
}

// Price a freshly built LM graph the way the projector does (fit_price.h
// mirrors prepare_graph's dispatch).
bool price(lm_model & lm, scratch & build, ::tts_cpp::detail::fit_graph_price & out) {
    return build.ok() &&
           ::tts_cpp::detail::fit_price_graph(lm.backend, build.graph,
                                              2 * AUDIO8_MAX_NODES, out);
}

void run_synthetic_lm_gates() {
    const audio8_test::tiny_lm p;
    const std::string path = audio8_test::write_tiny_lm_gguf(
        p, (fs::temp_directory_path() / "test-audio8-fit-tiny-lm.gguf").string());

    // Metadata-only vs real load on the same (CPU) backend.
    lm_model mm, real;
    fit_load_measure lmm;
    std::string error;
    if (!load_lm_metadata_only(path, /*n_gpu_layers=*/0, mm, lmm, &error)) {
        fail("load_lm_metadata_only failed: " + error);
        return;
    }
    if (!load_lm(path, /*n_gpu_layers=*/0, real, &error)) {
        fail("real load_lm failed: " + error);
        free_lm(mm);
        return;
    }

    // 1 + 2. Weight and KV parity, byte for byte.
    expect_eq(lmm.weights_bytes, ggml_backend_buffer_get_size(real.buffer_w),
              "LM weights parity");
    expect_eq(lmm.kv_bytes,
              (uint64_t) ggml_backend_buffer_get_size(real.slow_kv.buffer) +
                  ggml_backend_buffer_get_size(real.fast_kv.buffer),
              "LM KV parity");

    // 3. Slow arena parity: real prefill + deepest decode step vs the
    //    projection of the same two shapes (one gallocr, so max).
    const int width = p.num_codebooks + 1;
    const int prompt_w = 6;
    std::vector<int32_t> frames((size_t) width * prompt_w, 0);
    for (int c = 0; c < prompt_w; ++c) frames[(size_t) c * width] = 1;
    std::vector<float> sem_logits, fast_input;
    if (!slow_step(real, frames.data(), prompt_w, /*n_past=*/0, /*n_threads=*/2,
                   sem_logits, fast_input, &error) ||
        !slow_step(real, frames.data(), /*width=*/1, /*n_past=*/prompt_w, 2,
                   sem_logits, fast_input, &error)) {
        fail("real slow_step failed: " + error);
    } else {
        ::tts_cpp::detail::fit_graph_price prefill, step;
        {
            scratch build(AUDIO8_MAX_NODES);
            slow_graph_outputs outs;
            build_slow_graph(mm, build, prompt_w, 0, outs);
            if (!price(mm, build, prefill)) fail("pricing the prefill graph failed");
        }
        {
            scratch build(AUDIO8_MAX_NODES);
            slow_graph_outputs outs;
            build_slow_graph(mm, build, 1, prompt_w, outs);
            if (!price(mm, build, step)) fail("pricing the step graph failed");
        }
        if (prefill.host_bytes == 0 && step.host_bytes == 0) {
            expect_eq(std::max(prefill.device_bytes, step.device_bytes),
                      ggml_gallocr_get_buffer_size(real.slow_allocr, 0),
                      "LM slow arena parity");
        }
    }

    // 4. Fast arena parity: a real whole-frame fast_step vs the projected
    //    prime/last maximum.
    {
        std::vector<int32_t> codes;
        std::vector<float> prime_in((size_t) p.hidden, 0.0f);
        const code_picker pick = [](const std::vector<float> &, int) { return 0; };
        if (!fast_step(real, prime_in, p.semantic_begin, 2, pick, codes, &error)) {
            fail("real fast_step failed: " + error);
        } else {
            ::tts_cpp::detail::fit_graph_price prime, last;
            {
                scratch build(AUDIO8_MAX_NODES);
                build_fast_fit_graph(mm, build, 0, /*prime=*/true);
                if (!price(mm, build, prime)) fail("pricing the fast prime graph failed");
            }
            {
                scratch build(AUDIO8_MAX_NODES);
                build_fast_fit_graph(mm, build, p.num_codebooks - 1, /*prime=*/false);
                if (!price(mm, build, last)) fail("pricing the fast step graph failed");
            }
            if (prime.host_bytes == 0 && last.host_bytes == 0) {
                expect_eq(std::max(prime.device_bytes, last.device_bytes),
                          ggml_gallocr_get_buffer_size(real.fast_allocr, 0),
                          "LM fast arena parity");
            }
        }
    }

    free_lm(real);
    free_lm(mm);

    // 5. Errors surface as Error, never Success.
    {
        tts_cpp::audio8::FitOptions bad;
        bad.lm_gguf_path            = path + ".does-not-exist";
        bad.codec_decoder_gguf_path = path;
        tts_cpp::FitResult fr = tts_cpp::audio8::fit_params(bad);
        expect(fr.status == tts_cpp::FitStatus::Error, "missing LM was not Error");
        expect(fr.reason == "model-unreadable",
               "missing LM reason was '" + fr.reason + "'");
        // The LM GGUF is not a codec GGUF: wrong architecture must also be
        // Error, not a projection of garbage.
        tts_cpp::audio8::FitOptions wrong;
        wrong.lm_gguf_path            = path;
        wrong.codec_decoder_gguf_path = path;
        fr = tts_cpp::audio8::fit_params(wrong);
        expect(fr.status == tts_cpp::FitStatus::Error,
               "wrong-architecture decoder was not Error");
    }

    fs::remove(path);
}

void run_fixture_gates(const std::string & lm_path, const std::string & dec_path,
                       const std::string & enc_path, int n_gpu_layers) {
    tts_cpp::audio8::FitOptions fopts;
    fopts.lm_gguf_path            = lm_path;
    fopts.codec_decoder_gguf_path = dec_path;
    fopts.codec_encoder_gguf_path = enc_path;
    fopts.n_gpu_layers            = n_gpu_layers;
    fopts.prompt_tokens           = 16;
    fopts.max_frames              = 64;

    const tts_cpp::FitResult fit = tts_cpp::audio8::fit_params(fopts);
    expect(fit.status != tts_cpp::FitStatus::Error,
           "fit_params returned Error (" + fit.reason + ") for readable fixtures");
    expect(fit.device.weights_bytes > 0,       "projected weights_bytes == 0");
    expect(fit.device.state_bytes > 0,         "projected state_bytes == 0");
    expect(fit.device.lm_compute_bytes > 0,    "projected lm_compute_bytes == 0");
    expect(fit.device.codec_compute_bytes > 0, "projected codec_compute_bytes == 0");
    expect(!fit.report.empty(),                "empty report");
    if (g_failures) return;
    std::printf("%s", fit.report.c_str());

    // Codec decode parity: a real decode leaves the latent arena in
    // model.allocr and the block arena in model.block_allocr; the projection
    // of the same frame count must match both (their sum, since both stay
    // resident), byte for byte on the direct-dispatch path.
    {
        codec_model real;
        std::string error;
        if (!load_codec(dec_path, n_gpu_layers, real, &error)) {
            fail("real load_codec failed: " + error);
            return;
        }
        const int n_frames = 16;
        std::vector<int32_t> codes((size_t) real.hp.num_codebooks * n_frames, 0);
        std::vector<float> pcm;
        if (!decode_codes(real, codes.data(), n_frames, /*n_threads=*/2,
                          cancel_hook(), pcm, &error)) {
            fail("real decode_codes failed: " + error);
        } else {
            codec_model mm;
            fit_load_measure dm;
            if (!load_codec_metadata_only(dec_path, n_gpu_layers, mm, dm, &error)) {
                fail("load_codec_metadata_only failed: " + error);
            } else {
                expect_eq(dm.weights_bytes, ggml_backend_buffer_get_size(real.buffer_w),
                          "codec weights parity");
                codec_fit_measure cm;
                if (!measure_decode_memory(mm, n_frames, cm)) {
                    fail("measure_decode_memory failed");
                } else if (cm.host_bytes == 0) {
                    const uint64_t real_arenas =
                        (uint64_t) ggml_gallocr_get_buffer_size(real.allocr, 0) +
                        ggml_gallocr_get_buffer_size(real.block_allocr, 0);
                    expect_eq(cm.device_bytes, real_arenas, "codec decode arena parity");
                }
                free_codec(mm);
            }
        }
        free_codec(real);
    }

    // Near-INT_MAX workloads are rejected strictly (Error /
    // "workload-too-large"), never priced through wrapped-int graph shapes --
    // a preflight that crashes on the inputs it exists to reject violates its
    // own contract.
    {
        tts_cpp::audio8::FitOptions huge = fopts;
        huge.prompt_tokens = std::numeric_limits<int>::max();
        const tts_cpp::FitResult fr = tts_cpp::audio8::fit_params(huge);
        expect(fr.status == tts_cpp::FitStatus::Error,
               "near-INT_MAX prompt_tokens was not Error");
        expect(fr.reason == "workload-too-large",
               "near-INT_MAX prompt_tokens reason was '" + fr.reason + "'");
    }
    if (!enc_path.empty()) {
        // A reference whose sample count exceeds int must trip the widened
        // encode_positions product guard (or the RoPE-table check, whichever
        // bites first), never sign-overflow inside encoder_positions.
        tts_cpp::audio8::FitOptions huge = fopts;
        huge.reference_seconds = 2.0e5f;  // >= 2^31 samples at any real rate
        const tts_cpp::FitResult fr = tts_cpp::audio8::fit_params(huge);
        expect(fr.status == tts_cpp::FitStatus::Error,
               "over-int reference_seconds was not Error");
        expect(fr.reason == "workload-too-large",
               "over-int reference_seconds reason was '" + fr.reason + "'");
    }
}

}  // namespace

int main(int argc, char ** argv) {
    // The arena gates compare against a real ggml decode, which a staged Core
    // ML sidecar would replace.
    setenv("AUDIO8_COREML_DISABLE", "1", 1);
    if (argc >= 3) {
        run_fixture_gates(argv[1], argv[2], argc > 3 ? argv[3] : "",
                          argc > 4 ? std::atoi(argv[4]) : 0);
    } else {
        run_synthetic_lm_gates();
    }
    if (g_failures == 0) {
        std::printf("test-audio8-fit-params: all checks passed\n");
    }
    return g_failures;
}
