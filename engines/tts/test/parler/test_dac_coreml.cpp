// Parler DAC on the Core ML sidecar (src/parler/dac_coreml.h).
//
//   test-parler-dac-coreml --gguf MODEL.gguf [--fallback-gguf MODEL.gguf]
//                          [--ref-dir parler-ref] [--min-snr 45] [--min-speech-snr 45]
//                          [--cache-dir DIR] [--require-coreml]
//
// Every build (fake sidecars, no Core ML needed):
//   * the host quantizer the sidecar decodes from equals ggml's latent;
//   * a sidecar that fails mid-range (on its second window) is retired once
//     and the WHOLE range is redone on ggml -- bit-identical to a ggml-only
//     decode, nothing of the failed call's earlier windows survives -- and
//     later calls stay on ggml;
//   * PARLER_COREML_STRICT turns the failure into an error (for later calls
//     too), and a missing sidecar into a throw at attach;
//   * empty ranges and out-of-range codes keep their ggml-path behaviour;
//   * the pending seam (model.dac_pending, acquired once per call): a call
//     whose sidecar is not ready runs on ggml, bit-identical, and reports
//     pending; a sidecar that becomes ready between two streamed chunks makes
//     the synthesis "mixed" while each chunk stays on one backend; a failure
//     retires through the seam and the call is redone on ggml; strict turns a
//     sidecar that cannot serve, or fails, into an error.
// TTS_CPP_COREML builds with `<stem>-dac.mlmodelc` next to --gguf:
//   * fit_params refuses the projection (coreml-memory-estimate-unavailable)
//     unless PARLER_COREML_DISABLE is set;
//   * the sidecar attaches, reports a coreml-* label, and decodes whole
//     utterances (several windows plus a partial tail, and one shorter than a
//     window) and arbitrary sub-ranges within --min-snr of the ggml CPU decode;
//   * with --ref-dir holding dump-reference.py's case0 fixtures, case0 against
//     the HF wav (--min-speech-snr, 45 dB, for speech; bounded absolute error
//     for near-silence);
//   * an Engine reports the sidecar (dac_on_coreml, dac_backend, whole and
//     streamed) and PARLER_COREML_DISABLE keeps it on ggml;
//   * --fallback-gguf: a sidecar whose window cannot hold its context is
//     refused at attach (and throws under PARLER_COREML_STRICT).
// TTS_CPP_COREML builds with --cache-dir and no sidecar next to --gguf (the
// generated route, template parler-dac-w128, macOS 15 or later):
//   * the DAC sidecar is generated from the GGUF into the cache, compiled and
//     loaded (status route generated, ready); a second engine on the same GGUF
//     adopts it (route cached) without generating again;
//   * the same --min-snr / --min-speech-snr gates as above on the generated
//     sidecar, and the Engine reports it per call;
//   * fit charges the generated envelope (route generate, validated); a
//     budget only the ggml projection fits resolves use = off (fit and Engine).
// --memory-schedule DIR (TTS_CPP_COREML, generated route): ggml decodes run
// while the sidecar is generated, compiled, loaded and warmed into DIR; the
// footprint increment over the warm ggml baseline must stay within the
// envelope fit charges (the memory validation behind template.json
// memory.validated_for). Use a fresh DIR: an existing artifact is adopted.
// Exit 77 (skip) without --gguf; --require-coreml fails instead of skipping
// the Core ML gates (CI).

#include "tts-cpp/parler/engine.h"
#include "tts-cpp/parler/fit.h"
#include "parler/internal.h"
#include "coreml_sidecar_path.h"
#include "npy.h"
#include "../test_env_portable.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <thread>

#if defined(__APPLE__)
#include <mach/mach.h>
#endif
#include <memory>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tts_cpp::parler::detail;

namespace {

int g_failures = 0;

void check(bool ok, const std::string & what) {
    fprintf(stderr, "  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_failures;
}

// Fills every window with a constant, or fails on call `fail_on` (1-based).
class fake_sidecar final : public parler_dac_sidecar {
public:
    fake_sidecar(int window, int hop, int fail_on) : window_(window), hop_(hop), fail_on_(fail_on) {}
    int window() const override { return window_; }
    const char * label() const override { return "coreml-fake"; }
    bool decode(const float *, const float * valid, float * pcm) override {
        ++calls;
        if (calls == fail_on_) return false;
        for (int t = 0; t < window_; ++t) {
            if (valid[t] != 0.0f && valid[t] != 1.0f) return false;
        }
        std::fill(pcm, pcm + (size_t) window_ * hop_, 0.25f);
        return true;
    }
    int calls = 0;

private:
    int window_, hop_, fail_on_;
};

double snr_db(const std::vector<float> & got, const std::vector<float> & ref, double & max_abs) {
    double sig = 0, noise = 0;
    max_abs = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = (double) got[i] - ref[i];
        sig += (double) ref[i] * ref[i];
        noise += d * d;
        max_abs = std::max(max_abs, std::fabs(d));
    }
    if (!std::isfinite(noise)) return NAN;
    return noise > 0 ? 10.0 * std::log10(sig / noise) : INFINITY;
}

// Relative SNR is ill-conditioned near silence: the Indic Hindi greedy
// fixture is mostly DC at -78.5 dBFS. Keep speech on the relative gate and
// bound BOTH RMS (-70 dBFS) and peak (-60 dBFS) error on quiet HF fixtures.
// Random-code parity always uses the requested relative gate.
struct waveform_quality {
    double snr = NAN;
    double peak = 0;
    double ref_rms = 0;
    double error_rms = 0;
    bool quiet = false;
    bool pass = false;
};

waveform_quality hf_quality(const std::vector<float> & pcm, const std::vector<float> & ref,
                            double min_snr) {
    waveform_quality q;
    if (pcm.empty() || pcm.size() != ref.size()) return q;
    double signal = 0, error = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(pcm[i]) || !std::isfinite(ref[i])) return q;
        const double d = (double) pcm[i] - ref[i];
        signal += (double) ref[i] * ref[i];
        error += d * d;
        q.peak = std::max(q.peak, std::fabs(d));
    }
    q.ref_rms = std::sqrt(signal / ref.size());
    q.error_rms = std::sqrt(error / ref.size());
    q.snr = error > 0 ? 10.0 * std::log10(signal / error) : INFINITY;
    q.quiet = q.ref_rms < 1e-3; // below -60 dBFS, based only on the reference
    q.pass = q.quiet ? (q.error_rms <= std::pow(10.0, -70.0 / 20) && q.peak <= 1e-3)
                     : q.snr >= min_snr;
    return q;
}

void waveform_quality_gates() {
    const std::vector<float> silence(100, 0.0f), speech(100, 0.1f);
    check(hf_quality(silence, silence, 45).pass, "quiet gate: exact silence");
    check(hf_quality(std::vector<float>(100, 1.5e-4f), silence, 45).pass,
          "quiet gate: bounded fp16 floor");
    check(!hf_quality(std::vector<float>(100, 4e-4f), silence, 45).pass,
          "quiet gate: excessive RMS rejected");
    auto spike = silence;
    spike[0] = 2e-3f;
    check(!hf_quality(spike, silence, 45).pass, "quiet gate: isolated peak rejected");
    check(!hf_quality(std::vector<float>(100, 0.101f), speech, 45).pass,
          "speech gate: relative SNR required");
    check(hf_quality(std::vector<float>(100, 0.1001f), speech, 45).pass,
          "speech gate: small error passes");
    const std::vector<float> below(100, 0.000999f), above(100, 0.001001f);
    check(hf_quality(below, below, 45).quiet && !hf_quality(above, above, 45).quiet,
          "quiet gate: reference RMS selects the branch at -60 dBFS");
    auto transient = silence;
    transient[0] = 0.0079f; // low RMS, but the quiet gate must preserve this peak
    check(!hf_quality(silence, transient, 45).pass,
          "quiet gate: erasing a quiet reference transient fails");
    check(!hf_quality({}, silence, 45).pass, "quiet gate: wrong length rejected");
    spike[0] = NAN;
    check(!hf_quality(spike, silence, 45).pass && !hf_quality(silence, spike, 45).pass,
          "quiet gate: nonfinite PCM and reference rejected");
}

// A one-frame decode is 512 samples of mostly masked context: its fp16 SNR
// is ill-conditioned (53.3 dB on the mini DAC, whose generated weights are
// byte-identical to the exporter's) and is held to the default gate; every
// longer case is held to --min-snr.
constexpr double SINGLE_FRAME_MIN_SNR = 45.0;

double gate_for(int n_frames, double min_snr) {
    return n_frames > 1 ? min_snr : std::min(min_snr, SINGLE_FRAME_MIN_SNR);
}

std::vector<int32_t> random_codes(const parler_hparams & hp, int n_frames, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> code(0, hp.dac_codebook_size - 1);
    std::vector<int32_t> codes((size_t) hp.dac_n_q * n_frames);
    for (int32_t & c : codes) c = code(rng);
    return codes;
}

bool decode(const parler_model & model, const std::vector<int32_t> & codes, int n_frames,
            std::vector<float> & pcm, std::vector<float> * latent = nullptr, int begin = 0,
            int end = -1, std::string * backend = nullptr, std::string * err = nullptr) {
    return parler_dac_decode(model, codes.data(), n_frames, 2, pcm, latent, begin, end, err, backend);
}

void fake_sidecar_gates(parler_model & model) {
    fprintf(stderr, "fake sidecar gates\n");
    const parler_hparams & hp = model.hparams;
    const int rf = parler_dac_rf_frames(model);
    const int window = 2 * rf + 8;
    const int n_frames = 3 * (window - 2 * rf) + 5;
    const std::vector<int32_t> codes = random_codes(hp, n_frames, 11);

    std::vector<float> ref, ref_lat;
    std::string backend;
    check(decode(model, codes, n_frames, ref, &ref_lat, 0, -1, &backend) && backend == "ggml",
          "ggml reference decode reports ggml");

    // host quantizer == ggml's latent; the fake's pcm is its constant
    auto ok_fake = std::make_shared<fake_sidecar>(window, hp.dac_hop, 0);
    check(parler_attach_dac_sidecar(model, ok_fake), "attach a fake sidecar");
    std::vector<float> pcm, lat;
    check(decode(model, codes, n_frames, pcm, &lat, 0, -1, &backend) && backend == "coreml-fake",
          "sidecar decode reports the sidecar label");
    check(ok_fake->calls == 4, "sidecar walks ceil(n / (W - 2 rf)) windows (" +
          std::to_string(ok_fake->calls) + ")");
    double max_abs = 0;
    for (size_t i = 0; i < lat.size() && i < ref_lat.size(); ++i) {
        max_abs = std::max(max_abs, (double) std::fabs(lat[i] - ref_lat[i]));
    }
    check(lat.size() == ref_lat.size() && max_abs < 1e-4,
          "host RVQ latent == ggml latent (max_abs " + std::to_string(max_abs) + ")");
    check(pcm.size() == ref.size() && pcm.front() == 0.25f && pcm.back() == 0.25f,
          "sidecar pcm covers exactly the range");
    check(decode(model, codes, n_frames, pcm, nullptr, 7, 7) && pcm.empty(), "empty range");
    std::vector<int32_t> bad = codes;
    bad[3] = hp.dac_codebook_size;
    check(!decode(model, bad, n_frames, pcm) && model.dac_sidecar, "out-of-range code rejected before the sidecar");

    // fails on its second window: retired once, whole range redone on ggml
    auto failing = std::make_shared<fake_sidecar>(window, hp.dac_hop, 2);
    check(parler_attach_dac_sidecar(model, failing), "attach a failing fake sidecar");
    setenv(PARLER_COREML_STRICT_ENV, "1", 1);
    check(decode(model, codes, n_frames, pcm, nullptr, 0, -1, &backend) && backend == "ggml",
          "mid-range failure uses captured non-strict policy despite later strict environment");
    check(pcm == ref, "fallback output is bit-identical to the ggml-only decode");
    check(!model.dac_sidecar && failing->calls == 2, "failing sidecar retired after one failure");
    check(decode(model, codes, n_frames, pcm, nullptr, 5, n_frames - 3, &backend) && backend == "ggml" &&
          std::equal(pcm.begin(), pcm.end(), ref.begin() + 5 * hp.dac_hop),
          "later ranges stay on ggml");

    // strict: a failure is an error, a missing sidecar throws at attach
    setenv(PARLER_COREML_STRICT_ENV, "1", 1);
    auto strict_fail = std::make_shared<fake_sidecar>(window, hp.dac_hop, 1);
    parler_attach_dac_sidecar(model, strict_fail);
    unsetenv(PARLER_COREML_STRICT_ENV);
    std::string err;
    check(!decode(model, codes, n_frames, pcm, nullptr, 0, -1, nullptr, &err) && pcm.empty() &&
          err.find(PARLER_COREML_STRICT_ENV) != std::string::npos && !model.dac_sidecar,
          "strict: captured policy fails the call despite later environment removal");
    check(!decode(model, codes, n_frames, pcm) && pcm.empty(), "strict: later calls keep failing after a retirement");
    bool threw = false;
    try {
        parler_attach_dac_sidecar(model, std::string("/nonexistent/parler-none-f32.gguf"),
                                  parler_resolve_coreml_policy("cpu_only", true));
    } catch (const std::runtime_error &) {
        threw = true;
    }
    check(threw, "strict: no sidecar throws at attach");
    setenv(PARLER_COREML_STRICT_ENV, "1", 1);
    auto explicit_fallback = std::make_shared<fake_sidecar>(window, hp.dac_hop, 1);
    check(parler_attach_dac_sidecar(model, explicit_fallback,
                                   parler_resolve_coreml_policy("cpu_only", false)),
          "explicit false attaches independently of strict environment");
    check(decode(model, codes, n_frames, pcm) && pcm == ref,
          "explicit false recovers on ggml despite strict environment");
    unsetenv(PARLER_COREML_STRICT_ENV);
    check(decode(model, codes, n_frames, pcm) && pcm == ref, "retired non-strict sidecar stays on ggml");
    check(!parler_attach_dac_sidecar(model, std::string("/nonexistent/parler-none-f32.gguf")),
          "no sidecar: attach reports ggml");
    setenv(PARLER_COREML_DISABLE_ENV, "1", 1);
    threw = false;
    try {
        parler_attach_dac_sidecar(model, std::string("/nonexistent/parler-none-f32.gguf"),
                                  parler_resolve_coreml_policy("cpu_only", true));
    } catch (const std::runtime_error &) {
        threw = true;
    }
    check(threw, "disabled plus strict retains legacy missing-sidecar failure");
    check(!parler_attach_dac_sidecar(model, std::string("/nonexistent/parler-none-f32.gguf"),
                                    parler_resolve_coreml_policy("cpu_only", false)),
          "global disable permits ggml with explicit non-strict policy");
    unsetenv(PARLER_COREML_DISABLE_ENV);
}

// A pending seam: `ready_after` calls return nothing (pending), then the
// sidecar; `strict` makes acquire throw while pending. `block_until_cancel`
// makes acquire wait, like a Core ML load that is never ready, until the
// caller's cancel flag is set.
class fake_pending final : public parler_dac_pending {
public:
    fake_pending(std::shared_ptr<parler_dac_sidecar> sidecar, int ready_after, bool strict)
        : sidecar_(std::move(sidecar)), ready_after_(ready_after), strict_(strict) {}
    parler_dac_call acquire(int, const std::atomic<bool> * cancel) override {
        ++acquires;
        if (block_until_cancel) {
            saw_cancel_flag = cancel != nullptr;
            const auto give_up = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            while (cancel && !cancel->load() && std::chrono::steady_clock::now() < give_up) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        parler_dac_call call;
        if (retired || acquires <= ready_after_) {
            if (strict_) throw std::runtime_error("fake: not ready; strict");
            call.pending = !retired;
            return call;
        }
        call.sidecar = sidecar_;
        call.retire = [this](const std::string & why) {
            retired = true;
            retire_reason = why;
        };
        return call;
    }
    bool strict() const override { return strict_; }
    int acquires = 0;
    bool retired = false;
    bool block_until_cancel = false;
    bool saw_cancel_flag = false;
    std::string retire_reason;

private:
    std::shared_ptr<parler_dac_sidecar> sidecar_;
    int ready_after_;
    bool strict_;
};

void pending_seam_gates(parler_model & model) {
    fprintf(stderr, "pending seam gates\n");
    const parler_hparams & hp = model.hparams;
    const int rf = parler_dac_rf_frames(model);
    const int window = 2 * rf + 8;
    const int n_frames = 3 * (window - 2 * rf) + 5;
    const std::vector<int32_t> codes = random_codes(hp, n_frames, 23);
    model.dac_sidecar.reset();
    model.dac_sidecar_retired = false;
    model.dac_pending.reset();
    std::vector<float> ref;
    check(decode(model, codes, n_frames, ref), "ggml reference decode");
    if (!parler_dac_rvq_load(model, model.dac_rvq, nullptr)) {
        check(false, "host quantizer loads");
        return;
    }

    // pending: ggml, bit-equal, reported
    auto sidecar = std::make_shared<fake_sidecar>(window, hp.dac_hop, 0);
    auto seam = std::make_shared<fake_pending>(sidecar, 1, false);
    model.dac_pending = seam;
    std::vector<float> pcm;
    std::string backend, err;
    bool pending = false;
    check(parler_dac_decode(model, codes.data(), n_frames, 2, pcm, nullptr, 0, -1, &err, &backend, &pending) &&
              backend == "ggml" && pending && pcm == ref && sidecar->calls == 0,
          "pending: the call runs on ggml, bit-identical, and reports pending");

    // ready between two streamed chunks: each chunk on one backend, the
    // synthesis mixed
    const int split = n_frames / 2;
    std::string label;
    std::vector<float> chunk;
    seam->acquires = 0;
    auto seam2 = std::make_shared<fake_pending>(sidecar, 1, false);
    model.dac_pending = seam2;
    bool p1 = false, p2 = false;
    std::string b1, b2;
    const bool ok1 = parler_dac_decode(model, codes.data(), n_frames, 2, chunk, nullptr, 0, split, &err, &b1, &p1);
    const bool chunk1_ggml = ok1 && std::equal(chunk.begin(), chunk.end(), ref.begin());
    label = parler_merge_dac_backend(label, b1);
    const bool ok2 = parler_dac_decode(model, codes.data(), n_frames, 2, chunk, nullptr, split, n_frames, &err, &b2,
                                       &p2);
    const bool chunk2_sidecar =
        ok2 && !chunk.empty() && std::all_of(chunk.begin(), chunk.end(), [](float v) { return v == 0.25f; });
    label = parler_merge_dac_backend(label, b2);
    check(chunk1_ggml && p1 && b1 == "ggml" && chunk2_sidecar && !p2 && b2 == "coreml-fake" && label == "mixed" &&
              seam2->acquires == 2,
          "ready between chunks: chunk 1 ggml (pending), chunk 2 on the sidecar, synthesis " + label);

    // a failure retires through the seam; the whole call is redone on ggml
    auto failing = std::make_shared<fake_sidecar>(window, hp.dac_hop, 2);
    auto seam3 = std::make_shared<fake_pending>(failing, 0, false);
    model.dac_pending = seam3;
    check(parler_dac_decode(model, codes.data(), n_frames, 2, pcm, nullptr, 0, -1, &err, &backend, &pending) &&
              backend == "ggml" && !pending && pcm == ref && seam3->retired,
          "failure: retired through the seam, the whole call redone on ggml (bit-identical)");
    check(parler_dac_decode(model, codes.data(), n_frames, 2, pcm, nullptr, 0, -1, &err, &backend, &pending) &&
              backend == "ggml" && pcm == ref,
          "after a retirement later calls stay on ggml");

    // strict: not ready, and a failure, are errors
    model.dac_pending = std::make_shared<fake_pending>(sidecar, 1, true);
    err.clear();
    check(!parler_dac_decode(model, codes.data(), n_frames, 2, pcm, nullptr, 0, -1, &err, &backend) &&
              pcm.empty() && err.find("strict") != std::string::npos,
          "strict: a sidecar that cannot serve the call fails it");
    auto strict_failing = std::make_shared<fake_sidecar>(window, hp.dac_hop, 1);
    auto seam4 = std::make_shared<fake_pending>(strict_failing, 0, true);
    model.dac_pending = seam4;
    err.clear();
    check(!parler_dac_decode(model, codes.data(), n_frames, 2, pcm, nullptr, 0, -1, &err, &backend) &&
              pcm.empty() && seam4->retired && err.find(PARLER_COREML_STRICT_ENV) != std::string::npos,
          "strict: a failing sidecar is retired and the call fails");

    // cancellation interrupts a wait for the sidecar: the call returns soon
    // after the flag is set, reports cancellation and decodes nothing
    const int sidecar_calls = sidecar->calls;
    auto blocking = std::make_shared<fake_pending>(sidecar, 1 << 30, false);
    blocking->block_until_cancel = true;
    model.dac_pending = blocking;
    std::atomic<bool> cancel{false};
    std::thread canceller([&cancel] {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        cancel.store(true);
    });
    const auto t0 = std::chrono::steady_clock::now();
    err.clear();
    pcm.assign(4, 1.0f);
    const bool ok_cancel = parler_dac_decode(model, codes.data(), n_frames, 2, pcm, nullptr, 0, -1, &err, &backend,
                                             &pending, &cancel);
    const double cancel_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    canceller.join();
    check(!ok_cancel && blocking->saw_cancel_flag && err == "cancelled" && pcm.empty() && cancel_ms < 5000 &&
              sidecar->calls == sidecar_calls,
          "cancel: a blocked acquisition returns once cancelled, the call reports '" + err + "' after " +
              std::to_string((int) cancel_ms) + " ms");
    model.dac_pending.reset();
}

void coreml_gates(parler_model & model, const std::string & gguf, const std::string & ref_dir,
                  double min_snr, double min_speech_snr) {
    fprintf(stderr, "Core ML sidecar gates (%s)\n", parler_dac_sidecar_path(gguf).c_str());
    model.dac_sidecar.reset();
    if (!parler_attach_dac_sidecar(model, gguf)) {
        check(false, "sidecar attaches");
        return;
    }
    const std::string label = model.dac_sidecar->label();
    check(label.rfind("coreml-", 0) == 0, "label " + label);
    const parler_hparams & hp = model.hparams;
    const int window = model.dac_sidecar->window();
    const int keep = window - 2 * parler_dac_rf_frames(model);

    for (const int n_frames : {2 * keep + keep / 2, window / 3, 1}) {
        const std::vector<int32_t> codes = random_codes(hp, n_frames, 100 + n_frames);
        auto sidecar = model.dac_sidecar;
        model.dac_sidecar.reset();
        std::vector<float> ref, pcm;
        decode(model, codes, n_frames, ref);
        model.dac_sidecar = sidecar;
        std::string backend;
        const bool ok = decode(model, codes, n_frames, pcm, nullptr, 0, -1, &backend);
        double max_abs = 0;
        const double snr = ok && pcm.size() == ref.size() ? snr_db(pcm, ref, max_abs) : NAN;
        check(ok && backend == label && snr >= gate_for(n_frames, min_snr),
              std::to_string(n_frames) + " frames vs ggml: SNR " + std::to_string(snr) +
              " dB, max_abs " + std::to_string(max_abs));
        if (n_frames > keep) {
            // three sub-ranges that split windows differently stitch to the whole
            const int a = keep / 3, b = keep + keep / 2;
            std::vector<float> stitched, part;
            for (const auto & r : {std::make_pair(0, a), std::make_pair(a, b), std::make_pair(b, n_frames)}) {
                decode(model, codes, n_frames, part, nullptr, r.first, r.second);
                stitched.insert(stitched.end(), part.begin(), part.end());
            }
            const double ssnr = stitched.size() == ref.size() ? snr_db(stitched, ref, max_abs) : NAN;
            check(ssnr >= min_snr, "sub-ranges stitch: SNR " + std::to_string(ssnr) + " dB");
        }
    }
    check(model.dac_sidecar != nullptr, "sidecar still attached");

    tts_cpp::parler::FitOptions fit;
    fit.model_gguf_path = gguf;
    check(tts_cpp::parler::fit_params(fit).reason == "coreml-memory-estimate-unavailable",
          "fit refuses to project an enabled sidecar");
    setenv(PARLER_COREML_DISABLE_ENV, "1", 1);
    const std::string reason = tts_cpp::parler::fit_params(fit).reason;
    unsetenv(PARLER_COREML_DISABLE_ENV);
    check(reason != "coreml-memory-estimate-unavailable", "PARLER_COREML_DISABLE: fit projects ggml (" + reason + ")");

    if (!ref_dir.empty() && std::ifstream(ref_dir + "/case0_codes.npy").good() &&
        std::ifstream(ref_dir + "/case0_wav_greedy.npy").good()) {
        npy_array codes_npy = npy_load(ref_dir + "/case0_codes.npy");
        npy_array wav_npy = npy_load(ref_dir + "/case0_wav_greedy.npy");
        const int n_frames = (int) codes_npy.shape[2];
        const int64_t * c64 = reinterpret_cast<const int64_t *>(codes_npy.data.data());
        std::vector<int32_t> codes(codes_npy.n_elements());
        for (size_t i = 0; i < codes.size(); ++i) codes[i] = (int32_t) c64[i];
        std::vector<float> pcm;
        std::string backend;
        const bool ok = decode(model, codes, n_frames, pcm, nullptr, 0, -1, &backend);
        const float * w = npy_as_f32(wav_npy);
        const std::vector<float> want(w, w + wav_npy.n_elements());
        const auto q = hf_quality(pcm, want, min_speech_snr);
        check(ok && backend == label && q.pass, "case0 vs HF wav (" + backend + "): SNR " + std::to_string(q.snr) +
              " dB, max_abs " + std::to_string(q.peak) + ", ref RMS " +
              std::to_string(q.ref_rms) + ", error RMS " + std::to_string(q.error_rms) +
              (q.quiet ? " (quiet absolute gate)" : " (relative gate)"));
    }
}

// The public surface: an Engine on the model attaches the sidecar and reports
// it per call, streamed or not; PARLER_COREML_DISABLE keeps it on ggml.
void engine_gates(const std::string & gguf) {
    fprintf(stderr, "Engine gates\n");
    tts_cpp::parler::EngineOptions opts;
    opts.model_gguf_path = gguf;
    opts.default_description = "A calm voice.";
    opts.max_frames = 40;
    opts.stream_chunk_frames = 12;
    // The sidecar loads in the background; wait for it at construction so
    // the first call is deterministic.
    opts.coreml.wait = tts_cpp::CoremlOptions::Wait::construct;
    for (const bool disabled : {false, true}) {
        if (disabled) setenv(PARLER_COREML_DISABLE_ENV, "1", 1);
        tts_cpp::parler::Engine engine(opts);
        const auto whole = engine.synthesize("a b");
        const auto streamed = engine.synthesize("a b", opts.default_description,
                                                [](const float *, std::size_t, int, bool) {});
        const std::string want = disabled ? "ggml" : "coreml-";
        check(engine.dac_on_coreml() == !disabled &&
              whole.dac_backend.rfind(want, 0) == 0 && streamed.dac_backend.rfind(want, 0) == 0,
              std::string(disabled ? "disabled" : "enabled") + ": dac_on_coreml " +
              (engine.dac_on_coreml() ? "true" : "false") + ", dac_backend " + whole.dac_backend + " / " +
              streamed.dac_backend);
        if (disabled) unsetenv(PARLER_COREML_DISABLE_ENV);
    }
}

#ifdef TTS_CPP_USE_COREML
// The route-independent decode gates against ggml for whatever model.dac_pending
// serves (a generated sidecar here).
void pending_parity_gates(parler_model & model, const std::string & ref_dir, double min_snr,
                          double min_speech_snr) {
    auto seam = model.dac_pending;
    const parler_dac_call probe = seam->acquire(1, nullptr);
    if (!probe.sidecar) {
        check(false, "the generated sidecar is ready");
        return;
    }
    const std::string label = probe.sidecar->label();
    const int window = probe.sidecar->window();
    check(label.rfind("coreml-", 0) == 0, "label " + label + ", window " + std::to_string(window));
    const parler_hparams & hp = model.hparams;
    const int keep = window - 2 * parler_dac_rf_frames(model);
    for (const int n_frames : {2 * keep + keep / 2, window / 3, 1}) {
        const std::vector<int32_t> codes = random_codes(hp, n_frames, 100 + n_frames);
        model.dac_pending.reset();
        std::vector<float> ref, pcm;
        decode(model, codes, n_frames, ref);
        model.dac_pending = seam;
        std::string backend;
        const bool ok = decode(model, codes, n_frames, pcm, nullptr, 0, -1, &backend);
        double max_abs = 0;
        const double snr = ok && pcm.size() == ref.size() ? snr_db(pcm, ref, max_abs) : NAN;
        check(ok && backend == label && snr >= gate_for(n_frames, min_snr),
              std::to_string(n_frames) + " frames vs ggml: SNR " + std::to_string(snr) + " dB, max_abs " +
                  std::to_string(max_abs));
    }
    if (!ref_dir.empty() && std::ifstream(ref_dir + "/case0_codes.npy").good() &&
        std::ifstream(ref_dir + "/case0_wav_greedy.npy").good()) {
        npy_array codes_npy = npy_load(ref_dir + "/case0_codes.npy");
        npy_array wav_npy = npy_load(ref_dir + "/case0_wav_greedy.npy");
        const int n_frames = (int) codes_npy.shape[2];
        const int64_t * c64 = reinterpret_cast<const int64_t *>(codes_npy.data.data());
        std::vector<int32_t> codes(codes_npy.n_elements());
        for (size_t i = 0; i < codes.size(); ++i) codes[i] = (int32_t) c64[i];
        std::vector<float> pcm;
        std::string backend;
        const bool ok = decode(model, codes, n_frames, pcm, nullptr, 0, -1, &backend);
        const float * w = npy_as_f32(wav_npy);
        const std::vector<float> want(w, w + wav_npy.n_elements());
        const auto q = hf_quality(pcm, want, min_speech_snr);
        check(ok && backend == label && q.pass, "case0 vs HF wav (" + backend + "): SNR " + std::to_string(q.snr) +
              " dB, max_abs " + std::to_string(q.peak) + (q.quiet ? " (quiet absolute gate)" : " (relative gate)"));
    }
}

bool contains(const std::string & s, const std::string & needle) { return s.find(needle) != std::string::npos; }

// prewarm none issues an empty construction group and leaves the slot idle
// until the first call: wait = construct with use = require returns without
// throwing, and wait() returns false at once instead of polling an idle slot.
// Nothing is generated or loaded (a scratch cache stays empty).
void idle_slot_gates(parler_model & model, const std::string & gguf) {
    fprintf(stderr, "prewarm none gates\n");
    if (::tts_cpp::detail::coreml_sidecar_exists(parler_dac_sidecar_path(gguf))) {
        fprintf(stderr, "  a prebuilt sidecar is next to the GGUF: skipped\n");
        return;
    }
    const std::string cache =
        (std::filesystem::temp_directory_path() /
         ("parler-dac-idle-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())))
            .string();
    tts_cpp::CoremlOptions opts;
    opts.cache_dir = cache;
    opts.use = tts_cpp::CoremlOptions::Use::require;
    opts.wait = tts_cpp::CoremlOptions::Wait::construct;
    opts.prewarm = tts_cpp::CoremlOptions::Prewarm::none;
    const parler_dac_route route = parler_dac_plan_route(gguf, opts, parler_dac_rvq_bytes(model));
    if (route.route != "generate") {
        fprintf(stderr, "  route %s (%s: %s): skipped\n", route.route.c_str(), route.state.c_str(),
                route.reason.c_str());
        return;
    }
    model.dac_sidecar.reset();
    model.dac_pending.reset();
    std::shared_ptr<parler_dac_coreml> dac;
    std::string what;
    try {
        dac = parler_dac_coreml::create(model, gguf, opts);
    } catch (const std::exception & e) {
        what = e.what();
    }
    check(dac != nullptr, "{use: require, wait: construct, prewarm: none} constructs" +
                              (what.empty() ? std::string() : ": " + what));
    if (dac) {
        const std::string status = dac->status_json();
        const auto t0 = std::chrono::steady_clock::now();
        const bool ready = dac->wait(-1);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        check(contains(status, "\"state\":\"idle\"") && !ready && ms < 1000,
              "an idle slot: wait(-1) returns false after " + std::to_string((int) ms) + " ms; " +
                  status.substr(0, std::min<size_t>(status.size(), 200)));
    }
    model.dac_pending.reset();
    dac.reset();
    std::error_code ec;
    std::filesystem::remove_all(cache, ec);
}

void generated_gates(parler_model & model, const std::string & gguf, const std::string & cache_dir,
                     const std::string & ref_dir, double min_snr, double min_speech_snr) {
    fprintf(stderr, "generated sidecar gates (cache %s)\n", cache_dir.c_str());
    if (::tts_cpp::detail::coreml_sidecar_exists(parler_dac_sidecar_path(gguf))) {
        check(false, "no prebuilt sidecar next to --gguf (the generated route needs none)");
        return;
    }
    tts_cpp::CoremlOptions opts;
    opts.cache_dir = cache_dir;
    opts.wait = tts_cpp::CoremlOptions::Wait::construct;
    opts.use = tts_cpp::CoremlOptions::Use::require;
    const parler_dac_route route = parler_dac_plan_route(gguf, opts, parler_dac_rvq_bytes(model));
    check(route.route == "generate" && route.validated && route.envelope_bytes > 0,
          "route generate (" + route.route + " " + route.state + " " + route.reason + "), envelope " +
              std::to_string(route.envelope_bytes) + " bytes" + (route.validated ? ", validated" : ""));
    model.dac_sidecar.reset();
    model.dac_pending.reset();
    std::shared_ptr<parler_dac_coreml> dac;
    try {
        dac = parler_dac_coreml::create(model, gguf, opts);
    } catch (const std::exception & e) {
        check(false, std::string("generate, compile and load: ") + e.what());
        return;
    }
    const std::string status = dac->status_json();
    fprintf(stderr, "  status: %s\n", status.c_str());
    check(dac->wait(0) && contains(status, "\"state\":\"ready\"") &&
              (contains(status, "\"route\":\"generated\"") || contains(status, "\"route\":\"cached\"")),
          "generated, compiled and loaded (ready)");
    pending_parity_gates(model, ref_dir, min_snr, min_speech_snr);

    // A second owner of the same GGUF, template and cache adopts the artifact.
    parler_model second;
    std::string err;
    if (parler_load_gguf(gguf, second, 0, &err)) {
        auto again = parler_dac_coreml::create(second, gguf, opts);
        const std::string st2 = again->status_json();
        check(again->wait(0) && (contains(st2, "\"route\":\"cached\"") || contains(st2, "\"route\":\"generated\"")),
              "a second engine adopts the artifact: " + st2.substr(0, std::min<size_t>(st2.size(), 160)));
        parler_free_model(second);
    }

    tts_cpp::parler::FitOptions fit;
    fit.model_gguf_path = gguf;
    fit.coreml = opts;
    const tts_cpp::FitResult fr = tts_cpp::parler::fit_params(fit);
    check(fr.coreml.route == "generate" && fr.coreml.bytes == route.envelope_bytes && fr.coreml.validated,
          "fit charges the generated envelope (" + fr.coreml.route + ", " + std::to_string(fr.coreml.bytes) +
              " bytes, " + fr.reason + ")");

    // Near the budget: one only the ggml projection fits resolves use = off
    // (memory-degrade); a roomy one keeps Core ML.
    const uint64_t ggml_need = fr.device.total_bytes + fit.margin_bytes +
                               (fr.device_shares_host_memory ? fr.host_bytes : 0);
    tts_cpp::parler::FitOptions near = fit;
    near.coreml.memory_budget_bytes = (int64_t) (ggml_need + fr.coreml.bytes / 2);
    const tts_cpp::FitResult degraded = tts_cpp::parler::fit_params(near);
    tts_cpp::parler::FitOptions roomy = fit;
    roomy.coreml.memory_budget_bytes = (int64_t) (ggml_need + 2 * fr.coreml.bytes);
    const tts_cpp::FitResult kept = tts_cpp::parler::fit_params(roomy);
    check(degraded.coreml.disabled_reason == "memory" && kept.coreml.disabled_reason.empty(),
          "near-budget memory-degrade: '" + degraded.coreml.disabled_reason + "' below, '" +
              kept.coreml.disabled_reason + "' with room");
    // Near the free memory with a budget far above it: a margin that leaves
    // room for ggml but not for ggml + Core ML degrades on the free memory,
    // and the fit stands.
    const uint64_t ggml_no_margin = ggml_need - fit.margin_bytes;
    if (fr.device_free_bytes > ggml_no_margin + fr.coreml.bytes) {
        tts_cpp::parler::FitOptions tight = fit;
        tight.coreml.memory_budget_bytes = (int64_t) 1 << 50;
        tight.margin_bytes = fr.device_free_bytes - ggml_no_margin - fr.coreml.bytes / 2;
        const tts_cpp::FitResult t = tts_cpp::parler::fit_params(tight);
        check(t.coreml.disabled_reason == "memory" && t.status == tts_cpp::FitStatus::Success &&
                  t.coreml.ggml_only_fits && !t.coreml.fits,
              "near the free memory, huge budget: '" + t.coreml.disabled_reason + "', " + t.reason);
    } else {
        check(false, "free memory too small for the near-free-memory memory-degrade case");
    }

    tts_cpp::parler::EngineOptions eo;
    eo.model_gguf_path = gguf;
    eo.default_description = "A calm voice.";
    eo.max_frames = 40;
    eo.stream_chunk_frames = 12;
    eo.coreml = opts;
    {
        tts_cpp::parler::Engine engine(eo);
        const auto whole = engine.synthesize("a b");
        const auto streamed =
            engine.synthesize("a b", eo.default_description, [](const float *, std::size_t, int, bool) {});
        check(engine.dac_on_coreml() && whole.dac_backend.rfind("coreml-", 0) == 0 &&
                  streamed.dac_backend.rfind("coreml-", 0) == 0 && !whole.coreml_pending &&
                  !streamed.coreml_pending,
              "Engine: dac_backend " + whole.dac_backend + " / " + streamed.dac_backend);
    }
    {
        tts_cpp::parler::EngineOptions low = eo;
        low.coreml.use = tts_cpp::CoremlOptions::Use::automatic;
        low.coreml.memory_budget_bytes = near.coreml.memory_budget_bytes;
        tts_cpp::parler::Engine engine(low);
        const std::string st = engine.coreml_status();
        const auto whole = engine.synthesize("a b");
        check(contains(st, "\"state\":\"disabled\"") && whole.dac_backend == "ggml",
              "Engine under a near budget: DAC on ggml (" + whole.dac_backend + "), " +
                  st.substr(0, std::min<size_t>(st.size(), 200)));
    }
    model.dac_pending.reset();
}

// Footprint of this process (phys_footprint), sampled while the DAC sidecar
// is generated, compiled, loaded and warmed in the background and ggml decodes
// run in a loop on this thread: the increment over the warm ggml baseline
// must stay within the envelope fit charges. ANECompilerService's resident
// size is sampled too (it sets the runtime's cold-load memory reserve).
uint64_t footprint_now() {
    task_vm_info_data_t info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t) &info, &count) != KERN_SUCCESS) return 0;
    return info.phys_footprint;
}

uint64_t ane_compiler_rss() {
    uint64_t total = 0;
    FILE * f = popen("ps -axo rss=,comm= 2>/dev/null", "r");
    if (!f) return 0;
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        if (strstr(line, "ANECompilerService")) total += (uint64_t) std::strtoull(line, nullptr, 10) * 1024;
    }
    pclose(f);
    return total;
}

void memory_schedule(parler_model & model, const std::string & gguf, const std::string & cache_dir) {
    fprintf(stderr, "memory schedule (cache %s)\n", cache_dir.c_str());
    tts_cpp::CoremlOptions opts;
    opts.cache_dir = cache_dir;
    const parler_dac_route route = parler_dac_plan_route(gguf, opts, parler_dac_rvq_bytes(model));
    if (route.route != "generate") {
        check(false, "route generate (" + route.route + " " + route.state + " " + route.reason + ")");
        return;
    }
    const parler_hparams & hp = model.hparams;
    const int n_frames = 128;
    const std::vector<int32_t> codes = random_codes(hp, n_frames, 7);
    model.dac_sidecar.reset();
    model.dac_pending.reset();
    std::vector<float> pcm;
    for (int i = 0; i < 2; ++i) decode(model, codes, n_frames, pcm);  // warm the ggml arena
    const uint64_t base = footprint_now();
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> peak{base}, ane_peak{0};
    std::thread sampler([&] {
        int tick = 0;
        while (!stop.load()) {
            const uint64_t f = footprint_now();
            if (f > peak.load()) peak.store(f);
            if (tick++ % 25 == 0) {
                const uint64_t a = ane_compiler_rss();
                if (a > ane_peak.load()) ane_peak.store(a);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    });
    const auto t0 = std::chrono::steady_clock::now();
    auto dac = parler_dac_coreml::create(model, gguf, opts);
    int ggml_calls = 0, coreml_calls = 0;
    std::string backend;
    while (std::chrono::steady_clock::now() - t0 < std::chrono::minutes(5)) {
        decode(model, codes, n_frames, pcm, nullptr, 0, -1, &backend);
        if (backend == "ggml") {
            ++ggml_calls;
        } else if (++coreml_calls >= 5) {
            break;
        }
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    stop.store(true);
    sampler.join();
    const uint64_t inc = peak.load() > base ? peak.load() - base : 0;
    fprintf(stderr, "  status: %s\n", dac->status_json().c_str());
    char line[512];
    std::snprintf(line, sizeof(line),
                  "overlapped schedule: %d ggml + %d Core ML calls in %.1f s; footprint base %.1f MiB, peak +%.1f MiB "
                  "(envelope %.1f MiB, units %s); ANECompilerService peak %.1f MiB",
                  ggml_calls, coreml_calls, secs, base / 1048576.0, inc / 1048576.0, route.envelope_bytes / 1048576.0,
                  route.compute_units.c_str(), ane_peak.load() / 1048576.0);
    check(coreml_calls >= 5 && ggml_calls >= 1 && inc <= route.envelope_bytes, line);
    model.dac_pending.reset();
}
#endif

void fallback_gates(const std::string & gguf) {
    fprintf(stderr, "short-window sidecar gates (%s)\n", parler_dac_sidecar_path(gguf).c_str());
    parler_model model;
    std::string err;
    if (!parler_load_gguf(gguf, model, 0, &err)) {
        check(false, "load " + gguf + ": " + err);
        return;
    }
    check(!parler_attach_dac_sidecar(model, gguf) && !model.dac_sidecar,
          "a window no wider than its context is refused");
    setenv(PARLER_COREML_STRICT_ENV, "1", 1);
    bool threw = false;
    try {
        parler_attach_dac_sidecar(model, gguf);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    unsetenv(PARLER_COREML_STRICT_ENV);
    check(threw, "strict: refused sidecar throws");
    parler_free_model(model);
}

} // namespace

int main(int argc, char ** argv) {
    std::string gguf, fallback, ref_dir, cache_dir, memory_dir;
    double min_snr = 45.0;
    double min_speech_snr = 45.0;
    bool require_coreml = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--gguf") gguf = next();
        else if (a == "--fallback-gguf") fallback = next();
        else if (a == "--ref-dir") ref_dir = next();
        else if (a == "--min-snr") min_snr = std::atof(next().c_str());
        else if (a == "--min-speech-snr") min_speech_snr = std::atof(next().c_str());
        else if (a == "--cache-dir") cache_dir = next();
        else if (a == "--memory-schedule") memory_dir = next();
        else if (a == "--require-coreml") require_coreml = true;
    }
    if (gguf.empty() || !std::ifstream(gguf).good()) {
        fprintf(stderr, "SKIP: no --gguf model (%s)\n", gguf.c_str());
        return 77;
    }
    unsetenv(PARLER_COREML_STRICT_ENV);
    unsetenv(PARLER_COREML_DISABLE_ENV);

    parler_model model;
    std::string err;
    if (!parler_load_gguf(gguf, model, 0, &err)) {
        fprintf(stderr, "FAIL: load %s: %s\n", gguf.c_str(), err.c_str());
        return 1;
    }
    waveform_quality_gates();
    fake_sidecar_gates(model);
    pending_seam_gates(model);
#ifdef TTS_CPP_USE_COREML
    idle_slot_gates(model, gguf);
    if (!memory_dir.empty()) memory_schedule(model, gguf, memory_dir);
    if (!cache_dir.empty()) {
        generated_gates(model, gguf, cache_dir, ref_dir, min_snr, min_speech_snr);
    } else if (::tts_cpp::detail::coreml_sidecar_exists(parler_dac_sidecar_path(gguf))) {
        coreml_gates(model, gguf, ref_dir, min_snr, min_speech_snr);
        engine_gates(gguf);
    } else {
        fprintf(stderr, "no sidecar next to %s: Core ML gates skipped\n", gguf.c_str());
        check(!require_coreml, "--require-coreml: sidecar present");
    }
    if (!fallback.empty()) fallback_gates(fallback);
#else
    (void) ref_dir;
    (void) min_snr;
    (void) min_speech_snr;
    (void) cache_dir;
    (void) memory_dir;
    (void) fallback;
    fprintf(stderr, "built without TTS_CPP_COREML: Core ML gates skipped\n");
    check(!require_coreml, "--require-coreml: built with TTS_CPP_COREML");
#endif
    parler_free_model(model);
    fprintf(stderr, g_failures ? "FAILED (%d)\n" : "OK\n", g_failures);
    return g_failures ? 1 : 0;
}
