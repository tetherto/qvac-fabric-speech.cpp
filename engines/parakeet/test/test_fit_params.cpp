// Fit-projection parity tests (include/parakeet/fit.h): assert that the
// metadata-only memory projection matches what a REAL load actually
// allocates, byte for byte where the projection is exact by construction:
//
//   1. fit_params returns a projection (never Error) for a readable GGUF,
//      with non-zero weight and encoder-compute figures and a report;
//   2. weights parity -- the projected weight bytes equal the allocated
//      weight buffer bytes of a real load_from_gguf on the same backend
//      (both sides run the same buffer-type sizing, so this is exact);
//   3. encoder-compute parity -- the projected worst-case window graph equals
//      the gallocr buffer a real run_encoder reserves at the same mel frame
//      count (ggml_gallocr_reserve_n_size is reserve's size-only twin);
//   4. device-side saturation -- once the audio exceeds one long-form window,
//      the projected device bytes stop growing (only host extras keep
//      growing), which is the property callers rely on to fit long audio --
//      and the windowed projection prices the RESIDENT graph cache
//      (run_encoder keeps up to 3 window graphs alive), not a single graph;
//   5. a missing model file is Error/"model-unreadable", never Success;
//   6. Sortformer only: the projected head bytes (gallocr measure + CPU-side
//      graph-input term) equal the scheduler reservation a real diarize
//      actually makes -- byte for byte on CPU, bounded from above within one
//      device-side x_in slot on GPU (the sched's input copy is reusable, a
//      user input leaf is not; strict direction only) -- and the measure
//      itself must WORK on a metadata-only model (regression: the sched's
//      own size-only reserve aborts there, which used to crash fit_params on
//      every Sortformer GGUF);
//   7. Nemotron only: the projected encoder compute decomposes into the
//      window graph + prompt-projection graph + streaming step graph +
//      per-chunk pre-encode graph, and each component equals the buffer the
//      real path allocates -- the prompt graph's gallocr after a real
//      run_nemotron_prompt_projection, the step graph's gallocr after real
//      cache-aware stream steps, and the shared sched's buffers after the
//      steady-state per-chunk subsampling run. Plus: an unsupported
//      nemotron_chunk_ms is Error/"invalid-arguments", the default (largest)
//      operating point bounds every explicit one, and the device projection
//      GROWS with audio length (the full-length prompt conditioning),
//      unlike the saturating legacy families.
//
// Usage: test-fit-params <model.gguf> [n_gpu_layers]
// (CMake registers the CPU form, n_gpu_layers omitted = 0 -- the only backend
// every CI lane has; pass e.g. 99 manually to check parity on a GPU backend.)
// Exit 0 on success; non-zero with a FAIL line per broken invariant.

#include "parakeet/fit.h"
#include "long_form.h"
#include "parakeet_ctc.h"
#include "parakeet_sortformer.h"
#include "parakeet_tdt.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

// Same three-stride-2-conv recurrence run_encoder uses (see
// parakeet_fit.cpp::encoder_out_frames).
int encoder_out_frames(const parakeet::EncoderConfig & enc, long long n_mel_frames) {
    auto next = [&](long long len) {
        return enc.causal_downsampling ? (len / 2 + 1) : ((len - 1) / 2 + 1);
    };
    long long t = next(next(next(n_mel_frames)));
    return t < 1 ? 1 : (int) t;
}

}  // namespace

namespace {

int g_failures = 0;

void fail(const std::string & what) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
}

void expect(bool cond, const std::string & what) {
    if (!cond) fail(what);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }
    const std::string model_path   = argv[1];
    const int         n_gpu_layers = argc > 2 ? std::atoi(argv[2]) : 0;

    constexpr float kAudioSeconds = 10.0f;

    parakeet::FitOptions fopts;
    fopts.model_gguf_path = model_path;
    fopts.audio_seconds   = kAudioSeconds;
    fopts.n_gpu_layers    = n_gpu_layers;

    // 1. A readable GGUF always yields a projection.
    const parakeet::FitResult fit = parakeet::fit_params(fopts);
    expect(fit.status != parakeet::FitStatus::Error,
           "fit_params returned Error (" + fit.reason + ") for a readable model");
    expect(fit.device.weights_bytes > 0,        "projected weights_bytes == 0");
    expect(fit.device.encoder_compute_bytes > 0,"projected encoder_compute_bytes == 0");
    expect(fit.device.total_bytes >= fit.device.weights_bytes + fit.device.encoder_compute_bytes,
           "projected total < weights + encoder compute");
    expect(!fit.report.empty(),                 "empty report");
    expect(!fit.model_type.empty(),             "empty model_type");
    expect(fit.device_total_bytes > 0,          "device_total_bytes == 0");
    if (g_failures) {
        return g_failures;  // nothing below is meaningful without a projection
    }
    std::printf("%s", fit.report.c_str());

    // 2 + 3. Parity against a real load on the same (CPU) backend.
    {
        parakeet::ParakeetCtcModel model;
        if (parakeet::load_from_gguf(model_path, model, /*n_threads=*/0,
                                     n_gpu_layers, /*verbose=*/false) != 0) {
            fail("real load_from_gguf failed");
            return g_failures;
        }

        const size_t real_weights = parakeet::model_weights_buffer_bytes(model);
        expect(real_weights == fit.device.weights_bytes,
               "weights parity: projected " + std::to_string(fit.device.weights_bytes) +
               " != allocated " + std::to_string(real_weights));

        // Reproduce fit_params' workload arithmetic: 10 s fits one window on
        // every shipped model, so the projected graph is the single-pass graph
        // at ceil(seconds * sample_rate / hop) mel frames.
        const int n_mels = model.mel_cfg.n_mels;
        const int n_mel_frames = (int) std::ceil(
            (double) kAudioSeconds * model.mel_cfg.sample_rate / model.mel_cfg.hop_length);
        std::vector<float> zero_mel((size_t) n_mel_frames * n_mels, 0.0f);
        parakeet::EncoderOutputs enc_out;
        if (parakeet::run_encoder(model, zero_mel.data(), n_mel_frames, n_mels,
                                  enc_out, /*max_layers=*/-1,
                                  /*capture_intermediates=*/false) != 0) {
            fail("real run_encoder failed");
            return g_failures;
        }
        const size_t real_compute = parakeet::model_encoder_compute_buffer_bytes(model);
        if (fit.model_type != "nemotron") {
            expect(real_compute == fit.device.encoder_compute_bytes,
                   "encoder compute parity: projected " +
                   std::to_string(fit.device.encoder_compute_bytes) +
                   " != reserved " + std::to_string(real_compute));
        }

        // 7. Nemotron: the projected encoder compute decomposes into window
        //    graph + prompt graph + streaming step graph + per-chunk
        //    pre-encode graph, and each component matches what the real path
        //    allocates.
        if (fit.model_type == "nemotron") {
            const auto & rcs = model.nemotron_cfg.allowed_right_context_frames;
            const int rc_max = *std::max_element(rcs.begin(), rcs.end());
            const int stream_enc = rc_max + 1;
            const int total_enc = encoder_out_frames(model.encoder_cfg, n_mel_frames);
            const int prompt_frames = std::max(total_enc, stream_enc);

            // 7a. Composition: reproduce fit_params' component sum with the
            //     same measure APIs (default operating point = largest rc).
            size_t prompt_bytes = 0;
            parakeet::NemotronStreamFitMeasure nsm;
            if (parakeet::measure_nemotron_prompt_compute(model, prompt_frames,
                                                          prompt_bytes) != 0 ||
                parakeet::nemotron_measure_stream(model, rc_max, nsm) != 0) {
                fail("nemotron measure APIs failed on a real-loaded model");
                return g_failures;
            }
            const size_t expected_enc = real_compute + prompt_bytes +
                                        nsm.device_step_graph_bytes +
                                        nsm.device_subsampling_bytes;
            expect(expected_enc == fit.device.encoder_compute_bytes,
                   "nemotron encoder compute composition: projected " +
                   std::to_string(fit.device.encoder_compute_bytes) +
                   " != window " + std::to_string(real_compute) +
                   " + prompt " + std::to_string(prompt_bytes) +
                   " + step " + std::to_string(nsm.device_step_graph_bytes) +
                   " + pre-encode " + std::to_string(nsm.device_subsampling_bytes));

            // 7b. Prompt-graph parity: a real projection at the same frame
            //     count must allocate exactly the projected buffer.
            {
                const int32_t prompt_id =
                    parakeet::resolve_nemotron_prompt_id(model, "");
                std::vector<float> projected;
                if (parakeet::run_nemotron_prompt_projection(
                        model, enc_out.encoder_out.data(), enc_out.n_enc_frames,
                        enc_out.d_model, prompt_id, projected) != 0) {
                    fail("real run_nemotron_prompt_projection failed");
                } else {
                    expect(enc_out.n_enc_frames == prompt_frames,
                           "prompt parity precondition: real projection ran at " +
                           std::to_string(enc_out.n_enc_frames) +
                           " frames, projection priced " +
                           std::to_string(prompt_frames));
                    const size_t real_prompt =
                        parakeet::model_nemotron_prompt_buffer_bytes(model);
                    expect(real_prompt == prompt_bytes,
                           "nemotron prompt parity: projected " +
                           std::to_string(prompt_bytes) + " != allocated " +
                           std::to_string(real_prompt));
                }
            }

            // 7c. Streaming step-graph parity: drive a real cache-aware
            //     session at the projected operating point to its steady
            //     state (two chunks -- the second carries the 9-frame mel
            //     history, the per-chunk peak) and compare the session's
            //     step-graph gallocr and, on single-backend runs, the shared
            //     sched's subsampling reservation.
            {
                parakeet::TdtRuntimeWeights rt;
                parakeet::NemotronStreamState state;
                if (parakeet::tdt_prepare_runtime(model, rt) != 0 ||
                    parakeet::init_nemotron_stream_state(model, "", rc_max,
                                                         state) != 0) {
                    fail("nemotron stream session init failed");
                } else {
                    const int factor = model.encoder_cfg.subsampling_factor;
                    const int feed_frames = (1 + factor * rc_max)      // first chunk
                                          + factor * (rc_max + 1);     // one steady chunk
                    std::vector<float> zero_chunk(
                        (size_t) feed_frames * n_mels, 0.0f);
                    int steps = 0;
                    if (parakeet::append_nemotron_mel_frames(
                            state, zero_chunk.data(), feed_frames, n_mels) != 0) {
                        fail("nemotron append_nemotron_mel_frames failed");
                    }
                    for (; steps < 2; ++steps) {
                        std::vector<float> sig;
                        int nf = 0;
                        if (parakeet::next_nemotron_processed_signal(
                                state, n_mels, /*finalize=*/false, sig, nf) != 1) {
                            break;
                        }
                        parakeet::NemotronStreamStepResult step;
                        if (parakeet::run_nemotron_stream_step(
                                model, rt, sig.data(), nf, n_mels,
                                /*finalize=*/false, state, step) != 0) {
                            fail("real nemotron stream step failed");
                            break;
                        }
                    }
                    expect(steps == 2, "nemotron stream did not reach steady state (" +
                           std::to_string(steps) + " steps)");
                    const size_t real_step =
                        parakeet::nemotron_stream_graph_buffer_bytes(state);
                    expect(real_step == nsm.device_step_graph_bytes,
                           "nemotron step-graph parity: projected " +
                           std::to_string(nsm.device_step_graph_bytes) +
                           " != allocated " + std::to_string(real_step));
                    // On the Nemotron path the shared scheduler is used ONLY
                    // by the per-chunk subsampling graph (encoder, prompt,
                    // step, and decode graphs own their gallocrs), so the
                    // sched buffer sum isolates it on every backend --
                    // including any per-op CPU fallback the projection's
                    // single-buffer measure would miss.
                    {
                        ggml_backend_sched_t sched = parakeet::model_sched(model);
                        size_t real_sub = 0;
                        const int nb = sched ? ggml_backend_sched_get_n_backends(sched) : 0;
                        for (int i = 0; i < nb; ++i) {
                            real_sub += ggml_backend_sched_get_buffer_size(
                                sched, ggml_backend_sched_get_backend(sched, i));
                        }
                        const size_t projected_sub =
                            nsm.device_subsampling_bytes +
                            nsm.host_subsampling_input_bytes;
                        expect(real_sub == projected_sub,
                               "nemotron pre-encode parity: projected " +
                               std::to_string(projected_sub) +
                               " != sched reserved " + std::to_string(real_sub));
                    }
                }
            }
        }

        // 4b. The windowed projection prices run_encoder's RESIDENT graph
        //     cache (first + middle + last window graphs stay alive in the
        //     3-slot LRU), so at >= 3 windows it must exceed the single
        //     worst-window measurement.
        if (fit.model_type != "sortformer") {
            const long long total_mel_600 = (long long) std::ceil(
                600.0 * model.mel_cfg.sample_rate / model.mel_cfg.hop_length);
            const parakeet::LongFormPlan plan = parakeet::resolve_long_form_plan_frames(
                0, 0, model.encoder_cfg.pos_emb_max_len,
                model.encoder_cfg.subsampling_factor, total_mel_600);
            if (plan.enabled) {
                const int middle_mel =
                    (plan.center_frames + 2 * plan.context_frames) * plan.sub;
                size_t single_window = 0;
                if (parakeet::measure_encoder_compute(model, middle_mel, n_mels,
                                                      single_window) != 0) {
                    fail("measure_encoder_compute failed at the middle window length");
                } else {
                    parakeet::FitOptions w = fopts;
                    w.audio_seconds = 600.0f;
                    const parakeet::FitResult fw = parakeet::fit_params(w);
                    expect(fw.status != parakeet::FitStatus::Error,
                           "600 s windowed projection errored");
                    expect(fw.device.encoder_compute_bytes > single_window,
                           "windowed projection does not price the resident graph cache: " +
                           std::to_string(fw.device.encoder_compute_bytes) +
                           " <= single worst window " + std::to_string(single_window));
                }
            }
        }

        // 6. Sortformer head parity: the projection prices the head with a
        //    gallocr on the head backend plus an explicit CPU-side
        //    graph-input term (the scheduler's size-only reserve cannot run
        //    on a metadata-only model); a real diarize's scheduler
        //    reservation must equal that sum byte for byte. The shared sched
        //    is used only by the head on this path, so the buffer sum
        //    isolates it on every backend -- including any per-op CPU
        //    fallback the single-buffer measure would miss.
        if (fit.model_type == "sortformer") {
            const long long total_mel = (long long) std::ceil(
                (double) kAudioSeconds * model.mel_cfg.sample_rate / model.mel_cfg.hop_length);
            const int T = encoder_out_frames(model.encoder_cfg, total_mel);
            const int D = model.encoder_cfg.sortformer_fc_d_model;
            size_t head_active = 0, head_host_input = 0;
            if (parakeet::sortformer_measure_head(model, T, head_active,
                                                  head_host_input) != 0) {
                fail("sortformer_measure_head failed on a real-loaded model");
            } else if (!parakeet::model_sortformer_on_cpu(model)) {
                expect(head_active == fit.device.decoder_compute_bytes,
                       "sortformer head projection: fit decoder compute " +
                       std::to_string(fit.device.decoder_compute_bytes) +
                       " != measured head " + std::to_string(head_active));
            }
            std::vector<float> zero_enc((size_t) T * D, 0.0f);
            parakeet::SortformerDiarizationOptions sopts;
            parakeet::SortformerDiarizationResult  sres;
            if (parakeet::sortformer_diarize_ggml(model, zero_enc.data(), T, D,
                                                  sopts, sres) != 0) {
                fail("real sortformer_diarize_ggml failed");
            } else if (!parakeet::model_sortformer_on_cpu(model)) {
                ggml_backend_sched_t sched = parakeet::model_sched(model);
                size_t real_head = 0;
                const int nb = sched ? ggml_backend_sched_get_n_backends(sched) : 0;
                for (int i = 0; i < nb; ++i) {
                    real_head += ggml_backend_sched_get_buffer_size(
                        sched, ggml_backend_sched_get_backend(sched, i));
                }
                const size_t projected_head = head_active + head_host_input;
                if (nb <= 1) {
                    // CPU-only sched: the reservation IS the gallocr layout.
                    expect(real_head == projected_head,
                           "sortformer head parity: projected " +
                           std::to_string(projected_head) +
                           " != sched reserved " + std::to_string(real_head));
                } else {
                    // GPU sched: the device-side input copy is a reusable
                    // node, so the real reservation may come in below the
                    // measure by at most one x_in slot -- never above
                    // (strict direction). The slack term allows for buffer
                    // alignment padding on the device side.
                    const size_t x_in_slack =
                        (size_t) T * (size_t) D * sizeof(float) + 1024;
                    expect(real_head <= projected_head,
                           "sortformer head UNDER-projection: projected " +
                           std::to_string(projected_head) +
                           " < sched reserved " + std::to_string(real_head));
                    expect(projected_head <= real_head + x_in_slack,
                           "sortformer head over-projection beyond the input "
                           "slot: projected " + std::to_string(projected_head) +
                           " > reserved " + std::to_string(real_head) +
                           " + x_in slack " + std::to_string(x_in_slack));
                }
            }
        }
    }

    // 4. Device-side projection saturates at the long-form window; host extras
    //    keep growing with the audio. (Sortformer diarization does not window
    //    -- its device bytes DO grow -- but the CI fixtures here are
    //    transcription models. Nemotron's device projection also grows: the
    //    locale-prompt projection graph runs over the FULL stitched encoder
    //    output, so it scales with audio_seconds by design.)
    if (fit.model_type != "sortformer") {
        parakeet::FitOptions a = fopts;  a.audio_seconds = 600.0f;
        parakeet::FitOptions b = fopts;  b.audio_seconds = 7200.0f;
        const parakeet::FitResult fa = parakeet::fit_params(a);
        const parakeet::FitResult fb = parakeet::fit_params(b);
        expect(fa.status != parakeet::FitStatus::Error, "600 s projection errored");
        expect(fb.status != parakeet::FitStatus::Error, "7200 s projection errored");
        if (fit.model_type == "nemotron") {
            expect(fb.device.total_bytes > fa.device.total_bytes,
                   "nemotron device projection did not grow with audio length "
                   "(the full-length prompt projection must scale): 600 s -> " +
                   std::to_string(fa.device.total_bytes) + ", 7200 s -> " +
                   std::to_string(fb.device.total_bytes));
        } else {
            expect(fa.device.total_bytes == fb.device.total_bytes,
                   "device projection did not saturate at the long-form window: 600 s -> " +
                   std::to_string(fa.device.total_bytes) + ", 7200 s -> " +
                   std::to_string(fb.device.total_bytes));
        }
        expect(fb.host_bytes > fa.host_bytes,
               "host extras did not grow with audio length");
    }

    // 7d. Nemotron operating-point knob: an unsupported chunk_ms is
    //     Error/"invalid-arguments" (mirroring Engine::stream_start), and the
    //     default (largest) operating point bounds every explicit one.
    if (fit.model_type == "nemotron") {
        parakeet::FitOptions bad = fopts;
        bad.nemotron_chunk_ms = 123;  // not an allowed operating point
        const parakeet::FitResult fr = parakeet::fit_params(bad);
        expect(fr.status == parakeet::FitStatus::Error,
               "unsupported nemotron_chunk_ms was not Error");
        expect(fr.reason == "invalid-arguments",
               "unsupported nemotron_chunk_ms reason was '" + fr.reason + "'");

        parakeet::FitOptions smallest = fopts;
        smallest.nemotron_chunk_ms = 80;  // smallest trained operating point
        const parakeet::FitResult fs = parakeet::fit_params(smallest);
        expect(fs.status != parakeet::FitStatus::Error,
               "chunk_ms=80 projection errored (" + fs.reason + ")");
        expect(fs.device.encoder_compute_bytes <= fit.device.encoder_compute_bytes,
               "default operating point does not bound chunk_ms=80: " +
               std::to_string(fs.device.encoder_compute_bytes) + " > " +
               std::to_string(fit.device.encoder_compute_bytes));
        expect(fs.host_bytes <= fit.host_bytes,
               "default operating point does not bound chunk_ms=80 host extras");
    }

    // 5. Errors are reported as Error, never Success.
    {
        parakeet::FitOptions bad = fopts;
        bad.model_gguf_path = model_path + ".does-not-exist";
        const parakeet::FitResult fr = parakeet::fit_params(bad);
        expect(fr.status == parakeet::FitStatus::Error, "missing model was not Error");
        expect(fr.reason == "model-unreadable",
               "missing model reason was '" + fr.reason + "'");
        expect(!fr.fits, "missing model reported fits");
    }

    if (g_failures == 0) {
        std::printf("test-fit-params: all checks passed\n");
    }
    return g_failures;
}
