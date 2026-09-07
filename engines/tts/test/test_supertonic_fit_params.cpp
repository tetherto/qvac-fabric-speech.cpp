// Fit-projection parity tests (include/tts-cpp/supertonic/fit.h): assert that
// the metadata-only memory projection matches what a REAL load and REAL
// reservations actually allocate, byte for byte where the projection is exact
// by construction.
//
// Usage: test-supertonic-fit-params <supertonic.gguf> [n_gpu_layers=99]
//
// Gates (all against the real GGUF):
//   1. projected weight bytes (buffer_w + buffer_w_extra) == the buffers a
//      real load allocates, byte for byte -- including the per-tensor
//      storage-type decisions and the GPU pre-transpose roster;
//   2. every stage measure (text encoder cache set / duration / vector
//      estimator loop graph / vocoder) == the real reservation of the same
//      graphs (the parity probes), byte for byte on the direct path;
//   3. fit_params end-to-end: quadratic growth with text_tokens (the relpos
//      masks), growth with audio_seconds, the resolved-CPU refusal
//      (compute-path-not-supported -- never a guessed FITS), near-INT_MAX
//      rejection, and Error (never Success) on unreadable models.
//
// The vector-estimator gates require a non-CPU backend (the CPU multi-cache
// path is refused by design); on a GPU-less host those gates degrade to
// checking the refusal itself.
//
// Exit 0 on success; non-zero with a FAIL line per broken invariant.

#include "tts-cpp/supertonic/fit.h"

#include "supertonic_internal.h"

#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>

using namespace tts_cpp::supertonic::detail;

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

void run_gates(const std::string & path, int n_gpu_layers) {
    // ── Loads: metadata-only twin vs real ───────────────────────────────────
    supertonic_model mm, real;
    supertonic_fit_load_measure fm;
    if (!load_supertonic_gguf_metadata_only(path, mm, n_gpu_layers, /*f16_weights=*/-1,
                                            supertonic_precision::F32, /*vulkan_device=*/0,
                                            {}, fm)) {
        fail("load_supertonic_gguf_metadata_only failed");
        return;
    }
    if (!load_supertonic_gguf(path, real, n_gpu_layers)) {
        fail("real load_supertonic_gguf failed");
        free_supertonic_model(mm);
        return;
    }
    const bool on_cpu = model_prefers_cpu_kernels(real);

    // 1. Weight parity, byte for byte, per buffer.
    expect(real.buffer_w != nullptr, "real load produced no weight buffer");
    if (real.buffer_w) {
        expect_eq(fm.weights_bytes, ggml_backend_buffer_get_size(real.buffer_w),
                  "buffer_w parity");
    }
    if (real.buffer_w_extra) {
        expect_eq(fm.extra_bytes, ggml_backend_buffer_get_size(real.buffer_w_extra),
                  "buffer_w_extra parity");
    } else {
        expect_eq(fm.extra_bytes, 0, "extra bytes on a path with no extra buffer");
    }

    // 2. Stage arena parity: size-only measure (metadata model) vs real
    //    reservation (real model) of the same graphs.
    const int L = 32;
    const int latent_len = 64;
    const int steps = real.hparams.default_steps;
    std::string error;
    {
        uint64_t meas = 0, probe = 0;
        if (!supertonic_fit_measure_text_encoder(mm, L, meas, &error)) {
            fail("text encoder measure failed: " + error);
        } else if (!supertonic_fit_parity_probe_text_encoder(real, L, probe, &error)) {
            fail("text encoder probe failed: " + error);
        } else if (probe > 0) {
            expect_eq(meas, probe, "text encoder cache-set parity");
        }
    }
    {
        uint64_t meas = 0, probe = 0;
        if (!supertonic_fit_measure_duration(mm, L + 1, meas, &error)) {
            fail("duration measure failed: " + error);
        } else if (!supertonic_fit_parity_probe_duration(real, L + 1, probe, &error)) {
            fail("duration probe failed: " + error);
        } else if (probe > 0) {
            expect_eq(meas, probe, "duration cache parity");
        }
    }
    if (!on_cpu) {
        uint64_t meas = 0, probe = 0;
        if (!supertonic_fit_measure_vector(mm, latent_len, L, steps, meas, &error)) {
            fail("vector estimator measure failed: " + error);
        } else if (!supertonic_fit_parity_probe_vector(real, latent_len, L, steps,
                                                       probe, &error)) {
            fail("vector estimator probe failed: " + error);
        } else if (probe > 0) {
            expect_eq(meas, probe, "vector estimator loop-graph parity");
        }
    } else {
        uint64_t meas = 0;
        expect(!supertonic_fit_measure_vector(mm, latent_len, L, steps, meas, &error),
               "CPU vector-estimator path was not refused by the measure");
    }
    {
        uint64_t meas_dev = 0, meas_host = 0, probe = 0;
        if (!supertonic_fit_measure_vocoder(mm, latent_len, meas_dev, meas_host, &error)) {
            fail("vocoder measure failed: " + error);
        } else if (!supertonic_fit_parity_probe_vocoder(real, latent_len, probe, &error)) {
            fail("vocoder probe failed: " + error);
        } else if (probe > 0 && meas_host == 0) {
            expect_eq(meas_dev, probe, "vocoder cache parity");
        }
    }

    free_supertonic_model(real);
    free_supertonic_model(mm);

    // ── fit_params end-to-end ────────────────────────────────────────────────
    tts_cpp::supertonic::FitOptions fopts;
    fopts.model_gguf_path = path;
    fopts.n_gpu_layers    = n_gpu_layers;
    fopts.text_tokens     = 32;
    fopts.audio_seconds   = 4.0f;

    const tts_cpp::FitResult fit = tts_cpp::supertonic::fit_params(fopts);
    if (on_cpu) {
        // The CPU compute path is refused by design -- never a guessed FITS.
        expect(fit.status == tts_cpp::FitStatus::Error &&
                   fit.reason == "compute-path-not-supported",
               "resolved-CPU projection was not refused (" + fit.reason + ")");
    } else {
        expect(fit.status != tts_cpp::FitStatus::Error,
               "fit_params returned Error (" + fit.reason + ") for a readable model");
        expect(fit.device.weights_bytes > 0,       "projected weights_bytes == 0");
        expect(fit.device.lm_compute_bytes > 0,    "projected stage-cache bytes == 0");
        expect(fit.device.codec_compute_bytes > 0, "projected vocoder bytes == 0");
        expect(fit.host_bytes > 0,                 "projected host_bytes == 0");
        expect(!fit.report.empty(),                "empty report");
        if (g_failures == 0) std::printf("%s", fit.report.c_str());

        // The relpos masks make the text-encoder caches quadratic in the
        // text length; a longer utterance grows the estimator/vocoder.
        tts_cpp::supertonic::FitOptions more = fopts;
        more.text_tokens = 96;
        const tts_cpp::FitResult big_text = tts_cpp::supertonic::fit_params(more);
        expect(big_text.device.lm_compute_bytes > fit.device.lm_compute_bytes,
               "stage caches did not grow with text_tokens");
        more = fopts;
        more.audio_seconds = 12.0f;
        const tts_cpp::FitResult big_audio = tts_cpp::supertonic::fit_params(more);
        expect(big_audio.device.total_bytes > fit.device.total_bytes,
               "projection did not grow with audio_seconds");
    }

    // The explicit CPU request must refuse, never guess.
    {
        tts_cpp::supertonic::FitOptions cpu = fopts;
        cpu.n_gpu_layers = 0;
        const tts_cpp::FitResult fr = tts_cpp::supertonic::fit_params(cpu);
        expect(fr.status == tts_cpp::FitStatus::Error &&
                   fr.reason == "compute-path-not-supported",
               "CPU request was not refused (" + fr.reason + ")");
    }

    // Strict workload / argument rejection.
    {
        tts_cpp::supertonic::FitOptions huge = fopts;
        huge.text_tokens = std::numeric_limits<int>::max();
        const tts_cpp::FitResult fr = tts_cpp::supertonic::fit_params(huge);
        expect(fr.status == tts_cpp::FitStatus::Error &&
                   fr.reason == "workload-too-large",
               "near-INT_MAX text_tokens was not workload-too-large");
    }
    {
        tts_cpp::supertonic::FitOptions bad = fopts;
        bad.precision = "int4";
        const tts_cpp::FitResult fr = tts_cpp::supertonic::fit_params(bad);
        expect(fr.status == tts_cpp::FitStatus::Error &&
                   fr.reason == "invalid-arguments",
               "junk precision was not invalid-arguments");
    }
    {
        tts_cpp::supertonic::FitOptions bad = fopts;
        bad.model_gguf_path = path + ".does-not-exist";
        const tts_cpp::FitResult fr = tts_cpp::supertonic::fit_params(bad);
        expect(fr.status == tts_cpp::FitStatus::Error &&
                   fr.reason == "model-unreadable",
               "missing model was not model-unreadable (" + fr.reason + ")");
    }
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: %s <supertonic.gguf> [n_gpu_layers=99]\n"
                     "(fixture-gated: the vocoder/estimator channel plans are hardcoded\n"
                     "to the real architecture, so there is no tiny synthetic form)\n",
                     argv[0]);
        return 2;
    }
    const int n_gpu_layers = argc > 2 ? std::atoi(argv[2]) : 99;
    run_gates(argv[1], n_gpu_layers);
    if (g_failures == 0) {
        std::printf("test-supertonic-fit-params: all checks passed\n");
    }
    return g_failures;
}
