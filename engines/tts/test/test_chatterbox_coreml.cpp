// Chatterbox S3Gen Core ML sidecars.
//
//   test-chatterbox-coreml                   options and their legacy aliases,
//                                            naming, function buckets, HiFT
//                                            window plan, call shapes and the
//                                            Metal estimate, the CFM rollback
//                                            with stand-in steps, fit
//   test-chatterbox-coreml --stages-fixture  each sidecar against ggml on the
//                                            same tokens, one stage at a time,
//                                            then all three; routing past the
//                                            widest function and strict mode
//   test-chatterbox-coreml --shape-fixture   every feature shape a sidecar
//                                            declares is checked before
//                                            predicting
//   test-chatterbox-coreml --pending-fixture background loading and routing
//                                            on tiny multi-width sidecars:
//                                            pending calls, width choice,
//                                            load order, bounded waits, the
//                                            resident cap, retirement and
//                                            destruction mid-load
//   test-chatterbox-coreml --engine-fixture  Engine sentence overlap against
//                 [--slow-load]              in-order sequential synthesis,
//                                            cancellation, worker errors, the
//                                            CFM rollback; with --slow-load,
//                                            the first synthesis while every
//                                            function is still loading equals
//                                            a use = off run, through the
//                                            Engine and tts-cli
//
// The stages fixture reads CHATTERBOX_COREML_TEST_S3GEN (an S3Gen GGUF with
// the exported `<stem>-{encoder,cfm,hift}.mlmodelc` next to it); the engine
// fixture also reads CHATTERBOX_COREML_TEST_T3 (the matching T3 GGUF) and,
// with --slow-load, CHATTERBOX_COREML_TEST_CLI (a tts-cli binary); the shape
// and pending fixtures read CHATTERBOX_COREML_TEST_SHAPES (the directory
// prepare_chatterbox_coreml_shape_fixture.py writes). Each reports a skip
// (77) without its inputs or without Core ML.

#include "chatterbox_coreml.h"
#include "coreml_sidecar_path.h"
#include "npy.h"
#include "test_env_portable.h"
#include "text_preprocess.h"
#include "tts-cpp/chatterbox/engine.h"
#include "tts-cpp/chatterbox/fit.h"
#include "tts-cpp/chatterbox/s3gen_pipeline.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef TTS_CPP_USE_COREML
#include <sys/wait.h>
#endif

namespace cb = tts_cpp::chatterbox::detail;
namespace fs = std::filesystem;
using tts_cpp::chatterbox::Engine;
using tts_cpp::chatterbox::EngineOptions;
using tts_cpp::chatterbox::SynthesisResult;

namespace {

constexpr int SKIP = 77;

void require(bool ok, const std::string & what) {
    if (!ok) throw std::runtime_error(what);
}

struct env_guard {
    std::string name;
    bool present;
    std::string value;
    explicit env_guard(const char * n) : name(n), present(std::getenv(n) != nullptr),
                                          value(present ? std::getenv(n) : "") {}
    ~env_guard() {
        if (present) setenv(name.c_str(), value.c_str(), 1);
        else unsetenv(name.c_str());
    }
};

struct temp_dir {
    fs::path path;
    explicit temp_dir(const std::string & tag)
        : path(fs::path(test_tmpdir()) / ("chatterbox-coreml-" + tag + "-" + test_process_tag())) {
        std::error_code error;
        fs::remove_all(path, error);
        fs::create_directories(path);
    }
    ~temp_dir() {
        std::error_code error;
        fs::remove_all(path, error);
    }
};

using tts_cpp::CoremlOptions;

double max_abs_vec(const std::vector<float> & a, const std::vector<float> & b) {
    double m = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) m = std::max(m, (double) std::fabs(a[i] - b[i]));
    return m;
}

template <class E, class F> bool throws(F && f) {
    try {
        f();
    } catch (const E &) {
        return true;
    }
    return false;
}

void check_options() {
    // The legacy s3gen_synthesize_opts / EngineOptions fields are aliases:
    // each fills its CoremlOptions field only when that is unset.
    CoremlOptions none;
    auto merged = cb::merge_legacy_coreml(none, "cpu_only", 1, false);
    require(merged.compute_units == std::string("cpu_only") && merged.use == CoremlOptions::Use::require,
            "legacy units and strict fill unset fields");
    merged = cb::merge_legacy_coreml(none, "", 0, false);
    require(!merged.compute_units && merged.use == CoremlOptions::Use::automatic,
            "an explicit non-strict alias overrides the strict environment variable");
    require(cb::merge_legacy_coreml(none, "", -1, false).use == CoremlOptions::Use::unset,
            "an unset strict alias leaves use unset");
    CoremlOptions set;
    set.compute_units = "all";
    set.use = CoremlOptions::Use::automatic;
    merged = cb::merge_legacy_coreml(set, "cpu_only", 1, false);
    require(merged.compute_units == std::string("all") && merged.use == CoremlOptions::Use::automatic,
            "explicit CoremlOptions fields win over the aliases");
    require(cb::merge_legacy_coreml(set, "", 1, true).use == CoremlOptions::Use::off, "coreml_disabled is use = off");

    s3gen_synthesize_opts sopts;
    sopts.coreml_compute_units = "cpu_and_gpu";
    sopts.coreml_strict = 1;
    merged = cb::call_coreml_options(sopts);
    require(merged.compute_units == std::string("cpu_and_gpu") && merged.use == CoremlOptions::Use::require,
            "a call's options fold its aliases in");

    CoremlOptions bad;
    bad.compute_units = "gpu_please";
    require(throws<std::invalid_argument>([&] { cb::validate_coreml_options(bad); }),
            "invalid explicit units throw in every build");
    require(throws<std::invalid_argument>([&] { cb::merge_legacy_coreml(none, "gpu_please", -1, false);
                                                cb::validate_coreml_options(
                                                    cb::merge_legacy_coreml(none, "gpu_please", -1, false)); }),
            "an invalid legacy unit throws once folded in");
    CoremlOptions negative;
    negative.strict_wait_ms = -1;
    require(throws<std::invalid_argument>([&] { cb::validate_coreml_options(negative); }),
            "a negative strict wait throws");
    for (const char * units : {"all", "cpu_and_ane", "cpu_and_gpu", "cpu_only"}) {
        require(cb::coreml_valid_units(units), std::string("valid units ") + units);
    }
}

void check_naming() {
    require(cb::coreml_stage_path("models/chatterbox-s3gen-q8_0.gguf", cb::COREML_CFM_SUFFIX) ==
                "models/chatterbox-s3gen-cfm.mlmodelc", "turbo CFM sidecar name");
    require(cb::coreml_stage_path("m/chatterbox-s3gen-mtl-f32.gguf", cb::COREML_HIFT_SUFFIX) ==
                "m/chatterbox-s3gen-mtl-hift.mlmodelc", "multilingual HiFT sidecar name");
    require(cb::coreml_stage_path("chatterbox-s3gen.gguf", cb::COREML_ENCODER_SUFFIX) ==
                "chatterbox-s3gen-encoder.mlmodelc", "untagged encoder sidecar name");
}

void check_buckets() {
    const auto frames = cb::coreml_buckets("frames_640,tokens_256,frames_512,frames_,frames_x1,frames_640,main",
                                           cb::COREML_CFM_PREFIX);
    require(frames == std::vector<int>{512, 640}, "frame buckets");
    require(cb::coreml_buckets("tokens_256,tokens_320", cb::COREML_ENCODER_PREFIX) == std::vector<int>{256, 320},
            "token buckets");
    require(cb::coreml_pick_bucket(frames, 1) == 512 && cb::coreml_pick_bucket(frames, 512) == 512 &&
                cb::coreml_pick_bucket(frames, 513) == 640 && cb::coreml_pick_bucket(frames, 641) == -1,
            "bucket selection");
}

void check_windows() {
    const int window = 128, context = cb::HIFT_CONTEXT_FRAMES;
    require(cb::coreml_hift_windows(10, 2 * context, context).empty(), "a window without room for context");
    for (int frames = 1; frames <= 700; ++frames) {
        const auto plan = cb::coreml_hift_windows(frames, window, context);
        int covered = 0;
        for (const auto & w : plan) {
            require(w.keep_begin == covered, "kept spans tile the utterance");
            require(w.keep_end > w.keep_begin && w.start <= w.keep_begin, "non-empty kept span inside its window");
            require(w.keep_end - w.start <= window, "kept span inside the window");
            require(w.start == 0 || w.keep_begin - w.start >= context, "left context");
            require(w.keep_end == frames || w.start + window - w.keep_end >= context, "right context");
            covered = w.keep_end;
        }
        require(covered == frames, "every frame kept once");
        require(frames > window || plan.size() == 1, "one call when the utterance fits");
    }
}

void check_shapes_and_estimate() {
    // A batch call: prompt + speech + 3 lookahead tokens, two frames each.
    auto batch = cb::s3gen_shape(250, 120, 2, true, true);
    require(batch.tokens == 373 && batch.frames == 746, "batch call shape");
    // A non-final streaming chunk: no lookahead, the last 6 frames trimmed.
    auto chunk = cb::s3gen_shape(250, 25, 2, false, false);
    require(chunk.tokens == 275 && chunk.frames == 544, "streaming chunk shape");

    // The call shape follows the call's own prompt precedence and drops
    // tokens outside the speech vocabulary, as s3gen_synthesize_to_wav does.
    cb::s3gen_gguf_facts facts;
    facts.builtin_prompt_tokens = 250;
    facts.speech_vocab = 6561;
    s3gen_synthesize_opts sopts;
    const std::vector<int32_t> tokens = {1, 2, 6561, 7000, -1, 3};
    require(cb::s3gen_call_shape_for(sopts, tokens, facts).tokens == 250 + 3 + 3, "built-in voice prompt");
    const std::vector<int32_t> prompt(40, 5);
    sopts.prompt_token_view_data = prompt.data();
    sopts.prompt_token_view_size = prompt.size();
    require(cb::s3gen_call_shape_for(sopts, tokens, facts).tokens == 40 + 3 + 3, "a prompt view replaces it");
    s3gen_synthesize_opts embedding_only;
    embedding_only.embedding_override.assign(192, 0.0f);
    require(cb::s3gen_call_shape_for(embedding_only, tokens, facts).tokens == 0 + 3 + 3,
            "conditioning without a prompt has no prompt tokens");

    // The Metal estimate is the measured per-variant line, never negative.
    require(cb::s3gen_estimate_metal_ms_frames(true, 0) == 0.0 && cb::s3gen_estimate_metal_ms_frames(false, 0) == 0.0,
            "the estimate is clamped at zero");
    const double turbo = cb::s3gen_estimate_metal_ms_frames(true, 1000);
    const double mtl = cb::s3gen_estimate_metal_ms_frames(false, 1000);
    require(std::fabs(turbo - (-1347.8 + 2.9867 * 1000)) < 1e-6 && std::fabs(mtl - (-1992.6 + 9.2146 * 1000)) < 1e-6,
            "per-variant Metal cost lines");
    require(cb::s3gen_estimate_metal_ms_frames(true, 1200) > turbo, "the estimate grows with the frames");
}

// The CFM integration with stand-in steps: a Core ML failure at a later step
// restores the initial noise and integrates every step on ggml, so the result
// equals a whole ggml integration; nothing in it draws random numbers.
void check_cfm_rollback() {
    std::mt19937 rng(11);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> z0(64);
    for (float & v : z0) v = gauss(rng);
    const size_t steps = 4;
    // The two backends differ slightly, as fp16 Core ML and ggml do.
    auto make_step = [&](int fail_at, std::vector<std::pair<size_t, bool>> & trace) {
        return [fail_at, &trace](size_t s, std::vector<float> & z, bool on_coreml) -> int {
            trace.emplace_back(s, on_coreml);
            if (on_coreml && (int) s == fail_at) throw cb::coreml_stage_failure("stand-in failure");
            const float rate = on_coreml ? 0.251f : 0.25f;
            for (size_t i = 0; i < z.size(); ++i) z[i] += rate * std::sin(z[i] + (float) s);
            return 0;
        };
    };
    std::vector<std::pair<size_t, bool>> ggml_trace;
    std::vector<float> ggml_z = z0;
    require(cb::cfm_integrate(ggml_z, steps, false, make_step(-1, ggml_trace), [](auto &, size_t) {
                throw std::runtime_error("no failure expected");
            }) == 0,
            "ggml integration");

    std::vector<std::pair<size_t, bool>> coreml_trace;
    std::vector<float> coreml_z = z0;
    cb::cfm_integrate(coreml_z, steps, true, make_step(-1, coreml_trace), [](auto &, size_t) {});
    require(max_abs_vec(coreml_z, ggml_z) > 0.0, "the stand-in backends differ");

    for (int fail_at : {1, 2, 3}) {
        std::vector<std::pair<size_t, bool>> trace;
        std::vector<float> z = z0;
        int failures = 0;
        size_t failed_step = 0;
        const int rc = cb::cfm_integrate(z, steps, true, make_step(fail_at, trace),
                                         [&](const cb::coreml_stage_failure &, size_t s) {
                                             ++failures;
                                             failed_step = s;
                                         });
        require(rc == 0 && failures == 1 && failed_step == (size_t) fail_at, "one failure, reported at its step");
        require(z == ggml_z, "a failure at step " + std::to_string(fail_at + 1) +
                                 " gives exactly the whole ggml integration from the initial noise");
        // Core ML for steps 0..fail_at, then every step again on ggml.
        size_t i = 0;
        for (; i <= (size_t) fail_at; ++i) require(trace[i] == std::make_pair(i, true), "Core ML until the failure");
        for (size_t s = 0; s < steps; ++s, ++i) require(trace[i] == std::make_pair(s, false), "then ggml from step 0");
        require(trace.size() == i, "nothing after the ggml pass");
    }

    // A failure handler that stops the call (strict) propagates; a step's
    // non-zero result (a cancel) stops the integration.
    std::vector<std::pair<size_t, bool>> trace;
    std::vector<float> z = z0;
    require(throws<std::runtime_error>([&] {
                cb::cfm_integrate(z, steps, true, make_step(1, trace),
                                  [](const cb::coreml_stage_failure & e, size_t) { throw std::runtime_error(e.what()); });
            }),
            "strict stops the integration");
    require(cb::cfm_integrate(z, steps, false,
                              [](size_t s, std::vector<float> &, bool) { return s == 1 ? 2 : 0; },
                              [](auto &, size_t) {}) == 2,
            "a step's result stops the integration");
}

void check_warm_estimates() {
    require(cb::coreml_expected_speech_tokens(0) == 0 && cb::coreml_expected_speech_tokens(100) == 115,
            "speech-token estimate");
    // 60 tokens streamed in 25s, first chunk 10: cumulative windows, or
    // capped by 25 tokens of left context.
    require(cb::coreml_stream_windows(60, 10, 25, 0) == std::vector<int>{10, 35, 60}, "cumulative windows");
    require(cb::coreml_stream_windows(60, 10, 25, 25) == std::vector<int>{10, 35, 50}, "left-context windows");
    require(cb::coreml_stream_windows(0, 10, 25, 0).empty() && cb::coreml_stream_windows(5, 0, 0, 0).empty(),
            "no windows");
}

bool same_fit(const tts_cpp::FitResult & a, const tts_cpp::FitResult & b) {
    return a.status == b.status && a.fits == b.fits && a.reason == b.reason && a.model_variant == b.model_variant &&
           a.device_name == b.device_name && a.report == b.report;
}

void check_fit_and_detection() {
    env_guard disabled(cb::COREML_DISABLE_ENV);
    unsetenv(cb::COREML_DISABLE_ENV);
    temp_dir dir("fit");
    tts_cpp::chatterbox::FitOptions options;
    options.t3_gguf_path = (dir.path / "chatterbox-t3-turbo-q8_0.gguf").string();
    options.s3gen_gguf_path = (dir.path / "chatterbox-s3gen-q8_0.gguf").string();
    options.backend = "cpu";
    const auto plain = tts_cpp::chatterbox::fit_params(options);
    require(plain.reason != "coreml-memory-estimate-unavailable", "no sidecar, no Core ML refusal");
    require(!cb::coreml_prebuilt_route(options.s3gen_gguf_path, {}), "nothing to detect");

    for (const char * suffix : {cb::COREML_ENCODER_SUFFIX, cb::COREML_CFM_SUFFIX, cb::COREML_HIFT_SUFFIX}) {
        const std::string sidecar = cb::coreml_stage_path(options.s3gen_gguf_path, suffix);
        fs::create_directory(sidecar);
        const auto result = tts_cpp::chatterbox::fit_params(options);
#ifdef TTS_CPP_USE_COREML
        require(cb::coreml_prebuilt_route(options.s3gen_gguf_path, {}), "sidecar detected");
        require(result.status == tts_cpp::FitStatus::Error && !result.fits &&
                    result.reason == "coreml-memory-estimate-unavailable" && !result.report.empty(),
                std::string("fit refuses with an enabled ") + suffix + " sidecar");
        // The verdict depends on presence and options only, never on loads.
        require(same_fit(tts_cpp::chatterbox::fit_params(options), result), "fit is stable");
#else
        require(result.reason != "coreml-memory-estimate-unavailable", "no Core ML refusal without Core ML");
#endif
        // use = off: exactly the fit without any sidecar.
        auto off = options;
        off.coreml.use = CoremlOptions::Use::off;
        require(same_fit(tts_cpp::chatterbox::fit_params(off), plain), "fit with use = off is unchanged");
        auto legacy = options;
        setenv(cb::COREML_DISABLE_ENV, "1", 1);
        require(same_fit(tts_cpp::chatterbox::fit_params(legacy), plain),
                "CHATTERBOX_COREML_DISABLE restores the native projection");
        unsetenv(cb::COREML_DISABLE_ENV);
        fs::remove(sidecar);
    }
    auto bad = options;
    bad.coreml.compute_units = "gpu_please";
    require(tts_cpp::chatterbox::fit_params(bad).reason == "invalid-arguments", "invalid Core ML options");
}

// ---------------------------------------------------------------- fixture --

struct run_result {
    std::vector<float> wav;
    std::vector<float> mu;
    std::vector<float> mel;
    std::string backend;
};

std::vector<float> load_npy(const std::string & path) {
    npy_array a = npy_load(path);
    const float * data = reinterpret_cast<const float *>(a.data.data());
    return std::vector<float>(data, data + a.n_elements());
}

run_result synthesize(const std::string & gguf, const std::vector<int32_t> & tokens, const fs::path & scratch,
                      bool disabled, int strict) {
    s3gen_synthesize_opts opts;
    opts.s3gen_gguf_path = gguf;
    opts.n_gpu_layers = 99;
    opts.seed = 42;
    opts.coreml_disabled = disabled;
    opts.coreml_strict = strict;
    run_result r;
    opts.pcm_out = &r.wav;
    opts.s3gen_backend_out = &r.backend;
    opts.dump_mel_path = (scratch / "dump.npy").string();
    if (s3gen_synthesize_to_wav(tokens, opts) != 0) throw std::runtime_error("s3gen_synthesize_to_wav failed");
    r.mu = load_npy((scratch / "dump_mu.npy").string());
    r.mel = load_npy((scratch / "dump.npy").string());
    return r;
}

double cosine(const std::vector<float> & a, const std::vector<float> & b) {
    require(a.size() == b.size() && !a.empty(), "compared tensors differ in size");
    double ab = 0, aa = 0, bb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        ab += (double) a[i] * b[i];
        aa += (double) a[i] * a[i];
        bb += (double) b[i] * b[i];
    }
    return ab / (std::sqrt(aa * bb) + 1e-30);
}

double max_abs(const std::vector<float> & a, const std::vector<float> & b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, (double) std::fabs(a[i] - b[i]));
    return m;
}

// Directory holding a link to the GGUF and links to the named sidecars only.
fs::path stage_dir(const fs::path & root, const std::string & gguf, const std::vector<const char *> & suffixes,
                   const std::string & tag) {
    const fs::path dir = root / tag;
    fs::create_directories(dir);
    const fs::path link = dir / fs::path(gguf).filename();
    fs::create_symlink(fs::absolute(gguf), link);
    for (const char * suffix : suffixes) {
        const fs::path sidecar = cb::coreml_stage_path(gguf, suffix);
        fs::create_directory_symlink(fs::absolute(sidecar), cb::coreml_stage_path(link.string(), suffix));
    }
    return dir;
}

std::vector<int32_t> fixture_tokens(int count) {
    std::mt19937 rng(7);
    std::uniform_int_distribution<int32_t> token(0, 6560);
    std::vector<int32_t> tokens((size_t) count);
    for (auto & t : tokens) t = token(rng);
    return tokens;
}

int stages_fixture() {
#ifndef TTS_CPP_USE_COREML
    std::puts("SKIP: built without TTS_CPP_COREML");
    return SKIP;
#else
    const char * env = std::getenv("CHATTERBOX_COREML_TEST_S3GEN");
    if (!env || !*env) {
        std::puts("SKIP: CHATTERBOX_COREML_TEST_S3GEN is not set");
        return SKIP;
    }
    const std::string gguf = env;
    for (const char * suffix : {cb::COREML_ENCODER_SUFFIX, cb::COREML_CFM_SUFFIX, cb::COREML_HIFT_SUFFIX}) {
        if (!fs::exists(cb::coreml_stage_path(gguf, suffix))) {
            std::printf("SKIP: %s is missing\n", cb::coreml_stage_path(gguf, suffix).c_str());
            return SKIP;
        }
    }
    env_guard units(cb::COREML_UNITS_ENV), strict(cb::COREML_STRICT_ENV), disabled(cb::COREML_DISABLE_ENV);
    unsetenv(cb::COREML_UNITS_ENV);
    unsetenv(cb::COREML_STRICT_ENV);
    unsetenv(cb::COREML_DISABLE_ENV);
    temp_dir root("stages");
    const std::vector<int32_t> tokens = fixture_tokens(90);

    const fs::path base = stage_dir(root.path, gguf, {}, "ggml");
    const std::string base_gguf = (base / fs::path(gguf).filename()).string();
    const run_result ref = synthesize(base_gguf, tokens, base, true, -1);
    require(ref.backend == cb::GGML_STAGE_BACKEND, "ggml reference reports ggml");

    auto stage = [&](const char * suffix, const char * name) {
        const fs::path dir = stage_dir(root.path, gguf, {suffix}, name);
        return synthesize((dir / fs::path(gguf).filename()).string(), tokens, dir, false, 1);
    };
    const run_result enc = stage(cb::COREML_ENCODER_SUFFIX, "encoder");
    const double enc_cos = cosine(enc.mu, ref.mu);
    std::printf("[encoder] backend=%s mu cosine=%.6f max_abs=%.3e\n", enc.backend.c_str(), enc_cos,
                max_abs(enc.mu, ref.mu));
    require(enc.backend == cb::MIXED_STAGE_BACKEND && enc_cos > 0.9999, "encoder parity");

    const run_result cfm = stage(cb::COREML_CFM_SUFFIX, "cfm");
    const double cfm_mu = cosine(cfm.mu, ref.mu), cfm_cos = cosine(cfm.mel, ref.mel);
    std::printf("[cfm] mel cosine=%.6f max_abs=%.3e\n", cfm_cos, max_abs(cfm.mel, ref.mel));
    require(cfm_mu > 0.999999 && cfm_cos > 0.999, "CFM parity");

    const run_result hift = stage(cb::COREML_HIFT_SUFFIX, "hift");
    const double hift_cos = cosine(hift.wav, ref.wav);
    std::printf("[hift] waveform cosine=%.6f max_abs=%.3e\n", hift_cos, max_abs(hift.wav, ref.wav));
    require(cosine(hift.mel, ref.mel) > 0.999999 && hift_cos > 0.995, "HiFT parity");

    const fs::path all_dir = stage_dir(root.path, gguf,
                                       {cb::COREML_ENCODER_SUFFIX, cb::COREML_CFM_SUFFIX, cb::COREML_HIFT_SUFFIX}, "all");
    const std::string all_gguf = (all_dir / fs::path(gguf).filename()).string();
    const run_result all = synthesize(all_gguf, tokens, all_dir, false, 1);
    const double all_cos = cosine(all.mel, ref.mel);
    std::printf("[all] backend=%s mel cosine=%.6f waveform cosine=%.6f\n", all.backend.c_str(), all_cos,
                cosine(all.wav, ref.wav));
    require(all.backend == "coreml-ane" && all_cos > 0.999 && all.wav.size() == ref.wav.size(), "all stages");

    // Past the widest encoder/CFM function those stages stay on ggml (HiFT
    // windows any length); strict turns that fallback into an error.
    const std::vector<int32_t> long_tokens = fixture_tokens(1200);
    const run_result past = synthesize(all_gguf, long_tokens, all_dir, false, 0);
    std::printf("[routing] %zu tokens: backend=%s\n", long_tokens.size(), past.backend.c_str());
    require(past.backend == cb::MIXED_STAGE_BACKEND, "fallback past the widest function");
    bool threw = false;
    try {
        synthesize(all_gguf, long_tokens, all_dir, false, 1);
    } catch (const std::runtime_error & e) {
        threw = std::string(e.what()).find("cannot serve") != std::string::npos;
    }
    require(threw, "strict refuses an input past the widest function");
    s3gen_unload();
    return 0;
#endif
}

// ---------------------------------------------------------- shape fixture --

// The features each stage packs, as prepare_chatterbox_coreml_shape_fixture.py
// writes them: good sidecars declare exactly these (encoder tokens_8, CFM
// frames_16, HiFT window 32); `<stage>-<feature>` declares one of them with a
// doubled leading dimension.
const std::vector<std::pair<std::string, std::vector<std::string>>> SHAPE_FEATURES = {
    {"encoder", {"embedding", "valid", "mu"}},
    {"cfm", {"x", "mu", "cond", "spks", "t_emb", "valid", "velocity"}},
    {"hift", {"mel", "source", "valid", "spectrum"}},
};

// Shape-fixture sidecars on the CPU, each acquisition waiting for its load.
CoremlOptions fixture_options() {
    CoremlOptions o;
    o.compute_units = "cpu_only";
    o.wait = CoremlOptions::Wait::call;
    o.prewarm = CoremlOptions::Prewarm::none;
    return o;
}

// One call of `stage` on inputs narrower than its function; true when served.
bool serve(cb::s3gen_sidecars & sidecars, const std::string & stage) {
    const auto plan = sidecars.acquire({6, 12}, std::chrono::steady_clock::now());
    require(plan != nullptr, "a plan");
    std::mt19937 rng(3);
    std::normal_distribution<float> noise(0.0f, 1.0f);
    auto random = [&](size_t n) {
        std::vector<float> v(n);
        for (float & x : v) x = noise(rng);
        return v;
    };
    if (stage == "encoder") {
        constexpr int tokens = 6;
        std::vector<float> mu;
        const bool ok = cb::plan_encode(*plan, random((size_t) tokens * 512), tokens, 2, mu);
        require(!ok || mu.size() == (size_t) tokens * 2 * 80, "encoder output size");
        return ok;
    }
    if (stage == "cfm") {
        constexpr int frames = 12;
        const auto x = random(80 * frames), mu = random(80 * frames), cond = random(80 * frames);
        const auto spks = random(80), t_emb = random(1024);
        std::vector<std::vector<float>> out;
        // Like the pipeline: the CFG pair, of which a batch-1 sidecar uses the first.
        const bool ok = cb::plan_velocity(*plan, 0, 1, frames, {x.data(), x.data()}, {mu.data(), mu.data()},
                                          {cond.data(), cond.data()}, {spks.data(), spks.data()},
                                          {t_emb.data(), t_emb.data()}, out);
        require(!ok || (out.size() == 1 && out[0].size() == (size_t) 80 * frames), "CFM output size");
        return ok;
    }
    constexpr int frames = 40;  // two HiFT windows
    const size_t stft = (size_t) frames * cb::HIFT_STFT_PER_FRAME + 1;
    std::vector<float> spectrum;
    const bool ok = cb::plan_vocode(*plan, random((size_t) 80 * frames), frames, random(cb::HIFT_STFT_BINS * stft),
                                    spectrum);
    require(!ok || spectrum.size() == cb::HIFT_STFT_BINS * stft, "HiFT output size");
    return ok;
}

int shape_fixture() {
#ifndef TTS_CPP_USE_COREML
    std::puts("SKIP: built without TTS_CPP_COREML");
    return SKIP;
#else
    const char * env = std::getenv("CHATTERBOX_COREML_TEST_SHAPES");
    if (!env || !*env) {
        std::puts("SKIP: CHATTERBOX_COREML_TEST_SHAPES is not set");
        return SKIP;
    }
    const fs::path dir = env;
    {
        auto good = cb::s3gen_sidecars::create((dir / "good-s3gen.gguf").string(), fixture_options());
        require(good != nullptr, "the good sidecars are found");
        for (const auto & [stage, features] : SHAPE_FEATURES) {
            require(serve(*good, stage), "the matching " + stage + " sidecar serves");
        }
    }
    // The shared runner copies each feature by its declared shape, so every
    // one must be rejected before predicting, not only the ones the old
    // checks covered.
    for (const auto & [stage, features] : SHAPE_FEATURES) {
        for (const std::string & feature : features) {
            const std::string stem = stage + "-" + feature + "-s3gen";
            require(fs::exists(dir / (stem + "-" + stage + ".mlmodelc")), "fixture has " + stem);
            auto sidecars = cb::s3gen_sidecars::create((dir / (stem + ".gguf")).string(), fixture_options());
            require(sidecars != nullptr, "the " + stem + " sidecar is found");
            std::string error;
            try {
                serve(*sidecars, stage);
            } catch (const std::runtime_error & e) {
                error = e.what();
            }
            std::printf("[%s %s] %s\n", stage.c_str(), feature.c_str(), error.c_str());
            require(error.find("declares " + feature + " ") != std::string::npos,
                    stage + " rejects a sidecar declaring a larger " + feature);
        }
    }
    return 0;
#endif
}

// -------------------------------------------------------- pending fixture --

#ifdef TTS_CPP_USE_COREML
using steady = std::chrono::steady_clock;

double ms_since(steady::time_point t0) {
    return std::chrono::duration<double, std::milli>(steady::now() - t0).count();
}

bool wait_for(const std::function<bool()> & done, int timeout_ms) {
    const auto until = steady::now() + std::chrono::milliseconds(timeout_ms);
    while (!done()) {
        if (steady::now() > until) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return true;
}

// No function of the set is queued, loading or warming.
bool settled(const cb::s3gen_sidecars & set) {
    const std::string json = set.status_json();
    for (const char * busy : {"\"state\":\"queued\"", "\"state\":\"loading\"", "\"state\":\"warming\"",
                              "\"state\":\"deferred(memory)\""}) {
        if (json.find(busy) != std::string::npos) return false;
    }
    return true;
}

void settle(const cb::s3gen_sidecars & set, const std::string & what) {
    if (!wait_for([&] { return settled(set); }, 30000)) {
        std::printf("%s\n", set.status_json().c_str());
        throw std::runtime_error(what + ": the loads finish");
    }
}

// A directory holding links to the named routing-fixture sidecars under a
// fresh path (its own registry entries); returns the GGUF path the sidecars
// sit next to (the GGUF itself need not exist).
std::string routing_case(const fs::path & fixture, const fs::path & root, const std::string & name,
                         const std::vector<const char *> & suffixes) {
    const fs::path dir = root / name;
    fs::create_directories(dir);
    for (const char * suffix : suffixes) {
        fs::create_directory_symlink(fs::absolute(fixture / (std::string("routing-s3gen") + suffix + ".mlmodelc")),
                                     dir / (std::string("routing-s3gen") + suffix + ".mlmodelc"));
    }
    return (dir / "routing-s3gen.gguf").string();
}

CoremlOptions routing_options(const std::string & cache_dir = {},
                              CoremlOptions::Prewarm prewarm = CoremlOptions::Prewarm::none) {
    CoremlOptions o;
    o.compute_units = "cpu_only";
    o.wait = CoremlOptions::Wait::none;
    o.prewarm = prewarm;
    if (!cache_dir.empty()) o.cache_dir = cache_dir;
    return o;
}

cb::s3gen_call_shape shape(int tokens) {
    return cb::s3gen_call_shape{tokens, 2 * tokens};
}

cb::plan_stage_view view(const std::shared_ptr<S3genCoremlPlan> & plan, cb::s3gen_stage stage) {
    return cb::plan_view(*plan, stage);
}
#endif

int pending_fixture() {
#ifndef TTS_CPP_USE_COREML
    std::puts("SKIP: built without TTS_CPP_COREML");
    return SKIP;
#else
    const char * env = std::getenv("CHATTERBOX_COREML_TEST_SHAPES");
    if (!env || !*env) {
        std::puts("SKIP: CHATTERBOX_COREML_TEST_SHAPES is not set");
        return SKIP;
    }
    const fs::path fixture = env;
    for (const char * suffix : {cb::COREML_ENCODER_SUFFIX, cb::COREML_CFM_SUFFIX, cb::COREML_HIFT_SUFFIX}) {
        if (!fs::exists(fixture / (std::string("routing-s3gen") + suffix + ".mlmodelc"))) {
            std::printf("SKIP: the fixture has no routing-s3gen%s.mlmodelc (rerun "
                        "prepare_chatterbox_coreml_shape_fixture.py)\n", suffix);
            return SKIP;
        }
    }
    env_guard units(cb::COREML_UNITS_ENV), strict(cb::COREML_STRICT_ENV), disabled(cb::COREML_DISABLE_ENV);
    env_guard cache("QVAC_COREML_CACHE_DIR");
    unsetenv(cb::COREML_UNITS_ENV);
    unsetenv(cb::COREML_STRICT_ENV);
    unsetenv(cb::COREML_DISABLE_ENV);
    unsetenv("QVAC_COREML_CACHE_DIR");
    temp_dir root("pending");
    const std::vector<const char *> all = {cb::COREML_ENCODER_SUFFIX, cb::COREML_CFM_SUFFIX, cb::COREML_HIFT_SUFFIX};
    auto & hooks = cb::coreml_test();
    using S = cb::s3gen_stage;

    // 1. A call whose functions are still loading runs every stage on ggml,
    //    marked pending, without blocking; a later call gets the models.
    {
        auto set = cb::s3gen_sidecars::create(routing_case(fixture, root.path, "pending", all), routing_options());
        require(set && set->buckets(S::cfm) == std::vector<int>{16, 24, 32, 40} &&
                    set->buckets(S::encoder) == std::vector<int>{8, 12, 16, 20} && set->has(S::hift),
                "the routing sidecars list their widths");
        hooks.load_delay_ms = 1000;
        const auto t0 = steady::now();
        auto plan = set->acquire(shape(8), steady::now() + std::chrono::milliseconds(10));
        const double took = ms_since(t0);
        for (S s : {S::encoder, S::cfm, S::hift}) {
            require(!view(plan, s).model && view(plan, s).pending && view(plan, s).label == cb::GGML_STAGE_BACKEND,
                    std::string("pending ") + cb::to_string(s) + " runs on ggml");
        }
        require(cb::plan_pending(*plan) && took < 200, "a pending call is marked and never blocks");
        hooks.load_delay_ms = 0;
        settle(*set, "pending");
        plan = set->acquire(shape(8), steady::now());
        require(view(plan, S::encoder).bucket == 8 && view(plan, S::cfm).bucket == 16 && view(plan, S::hift).model &&
                    view(plan, S::cfm).label == "coreml-cpu" && !cb::plan_pending(*plan),
                "the next call runs on the loaded functions");
        std::printf("[pending] acquired in %.1f ms, all three on ggml; the next call on Core ML\n", took);
    }

    // 2. A short sentence whose widths are ready, then a longer one whose
    //    widths are still loading: the first runs on Core ML, the second on
    //    ggml (a narrower ready function never serves a wider call).
    {
        auto set = cb::s3gen_sidecars::create(routing_case(fixture, root.path, "short-long", all), routing_options());
        set->predict({shape(8)});
        settle(*set, "short-long");
        hooks.load_delay_ms = 800;
        auto short_plan = set->acquire(shape(8), steady::now());
        auto long_plan = set->acquire(shape(12), steady::now());
        require(view(short_plan, S::cfm).bucket == 16 && view(short_plan, S::encoder).bucket == 8,
                "the short sentence runs on its ready widths");
        require(!view(long_plan, S::cfm).model && view(long_plan, S::cfm).pending && !view(long_plan, S::encoder).model,
                "the longer sentence runs on ggml while its widths load");
        require(view(long_plan, S::hift).model, "HiFT serves every length");
        hooks.load_delay_ms = 0;
        settle(*set, "short-long");
        std::printf("[short-long] ready widths served, the longer sentence on ggml\n");
    }

    // 3. The actual length exceeds the prediction: the predicted width is
    //    loaded, the call needs a wider one; that one is requested at once
    //    and the call runs on ggml meanwhile.
    {
        auto set = cb::s3gen_sidecars::create(routing_case(fixture, root.path, "longer", all), routing_options());
        set->predict({shape(8)});
        settle(*set, "longer");
        hooks.clear_loads();
        hooks.load_delay_ms = 500;
        auto plan = set->acquire(shape(16), steady::now());
        require(!view(plan, S::cfm).model && !view(plan, S::encoder).model, "the longer call runs on ggml");
        hooks.load_delay_ms = 0;
        settle(*set, "longer");
        const auto loads = hooks.loads();
        require(std::find(loads.begin(), loads.end(), "cfm/frames_32") != loads.end() &&
                    std::find(loads.begin(), loads.end(), "encoder/tokens_16") != loads.end(),
                "the actual widths are loaded");
        require(set->state(S::cfm, 32) == "ready", "and ready for the next call");
        std::printf("[longer] actual widths loaded after the call\n");
    }

    // 4. A CFM failure at a later Euler step (stand-in steps; the real
    //    pipeline is the engine fixture's).
    check_cfm_rollback();
    std::printf("[cfm-rollback] a later-step failure equals a whole ggml integration from z0\n");

    // 5. A failure during an overlapped call: the plan defers it (the worker
    //    hands the call back), the function is retired process-wide, and the
    //    redo's fresh plan no longer uses it (here a wider ready function).
    {
        auto set = cb::s3gen_sidecars::create(routing_case(fixture, root.path, "retire", all), routing_options());
        set->predict({shape(8), shape(12)});
        settle(*set, "retire");
        auto plan = set->acquire(shape(8), steady::now());
        require(view(plan, S::cfm).bucket == 16, "the overlapped call holds frames_16");
        cb::defer_coreml_failures(*plan);
        require(cb::coreml_failures_deferred(*plan), "the worker's plan defers failures");
        hooks.fail_stage = (int) S::cfm;
        std::vector<float> x(80 * 16, 0.0f), spks(80, 0.0f), t_emb(1024, 0.0f);
        std::vector<std::vector<float>> out;
        bool failed = false;
        try {
            cb::plan_velocity(*plan, 0, 1, 16, {x.data()}, {x.data()}, {x.data()}, {spks.data()}, {t_emb.data()}, out);
        } catch (const cb::coreml_stage_failure &) {
            failed = true;
        }
        require(failed && hooks.fail_stage == -1, "the injected prediction failure surfaces as a stage failure");
        cb::plan_retire(*plan, S::cfm, "injected");
        plan.reset();
        auto redo = set->acquire(shape(8), steady::now());
        require(view(redo, S::cfm).bucket == 24 && view(redo, S::encoder).bucket == 8,
                "the redo runs CFM on the next ready width, the other stages unchanged");
        require(set->state(S::cfm, 16) == "retired", "the failed function is retired");
        std::printf("[retire] frames_16 retired; the redo runs on frames_24\n");
    }

    // 6. Dropping the sidecars while a load is in flight returns at once.
    {
        auto set = cb::s3gen_sidecars::create(routing_case(fixture, root.path, "destroy", all), routing_options());
        hooks.clear_loads();
        hooks.load_delay_ms = 1500;
        set->predict({shape(8)});
        require(wait_for([&] { return !hooks.loads().empty(); }, 5000), "a load is in flight");
        const auto t0 = steady::now();
        set.reset();
        const double took = ms_since(t0);
        std::printf("[destroy] dropped mid-load in %.2f ms\n", took);
        require(took < 100.0, "destruction mid-load returns in under 100 ms");
        hooks.load_delay_ms = 0;
        std::this_thread::sleep_for(std::chrono::milliseconds(1700));  // let the started load end
    }

    // 7. The exact width is still loading and a wider one is ready: the wider
    //    one serves the call (padding masked).
    {
        auto set = cb::s3gen_sidecars::create(routing_case(fixture, root.path, "wider", all), routing_options());
        set->predict({shape(20)});
        settle(*set, "wider");
        hooks.load_delay_ms = 800;
        auto plan = set->acquire(shape(12), steady::now());
        require(view(plan, S::cfm).bucket == 40 && view(plan, S::encoder).bucket == 20 && !cb::plan_pending(*plan),
                "a ready wider function serves the call");
        hooks.load_delay_ms = 0;
        settle(*set, "wider");
        require(set->state(S::cfm, 24) == "ready", "the exact width loaded meanwhile");
        std::printf("[wider] frames_40 served a 24-frame call\n");
    }

    // 8. The hinted widths are loading at construction and the first call
    //    needs another one: it is the next load, right after the one in
    //    flight.
    {
        const std::string cache_dir = (root.path / "order-cache").string();
        const std::string gguf = routing_case(fixture, root.path, "order", {cb::COREML_CFM_SUFFIX});
        {
            auto set = cb::s3gen_sidecars::create(gguf, routing_options(cache_dir));
            set->predict({shape(8), shape(12), shape(16)});
            settle(*set, "order (hints)");
        }
        hooks.clear_loads();
        hooks.load_delay_ms = 400;
        auto set = cb::s3gen_sidecars::create(gguf, routing_options(cache_dir, CoremlOptions::Prewarm::minimal));
        set->start_construction();
        require(wait_for([&] { return hooks.loads().size() == 1; }, 5000), "a construction load is in flight");
        auto plan = set->acquire(shape(20), steady::now());
        require(!view(plan, S::cfm).model, "the call runs on ggml");
        hooks.load_delay_ms = 0;
        settle(*set, "order");
        const auto loads = hooks.loads();
        std::string order;
        for (const auto & l : loads) order += (order.empty() ? "" : " ") + l;
        std::printf("[order] %s\n", order.c_str());
        const std::set<std::string> hinted = {"cfm/frames_16", "cfm/frames_24", "cfm/frames_32"};
        require(loads.size() == 4 && hinted.count(loads[0]) == 1 && loads[1] == "cfm/frames_40",
                "the call's width loads second, right after the one in flight");
        require(std::set<std::string>(loads.begin() + 2, loads.end()).size() == 2, "then the other hinted widths");
    }

    // 9. A stale hint: the slow load is waited for only until the deadline,
    //    the hint cools down, and the next call does not wait at all.
    {
        const std::string cache_dir = (root.path / "stale-cache").string();
        const std::string gguf = routing_case(fixture, root.path, "stale", {cb::COREML_CFM_SUFFIX});
        // Two processes' worth of loads: the first records a cold load, the
        // second (hinted) an observed warm load time.
        for (int round = 0; round < 2; ++round) {
            auto set = cb::s3gen_sidecars::create(gguf, routing_options(cache_dir));
            set->predict({shape(8)});
            settle(*set, "stale (hint)");
        }
        hooks.load_delay_ms = 2000;
        auto set = cb::s3gen_sidecars::create(gguf, routing_options(cache_dir));
        auto t0 = steady::now();
        auto plan = set->acquire(shape(8), t0 + std::chrono::milliseconds(300));
        const double first = ms_since(t0);
        t0 = steady::now();
        auto again = set->acquire(shape(8), t0 + std::chrono::milliseconds(300));
        const double second = ms_since(t0);
        std::printf("[stale] first call waited %.0f ms (deadline 300), the next %.0f ms\n", first, second);
        require(!view(plan, S::cfm).model && first >= 250 && first < 450,
                "the warm-hinted slow load is waited for until the deadline, not beyond");
        require(!view(again, S::cfm).model && second < 100, "the stale hint gives the next call no wait");
        hooks.load_delay_ms = 0;
        settle(*set, "stale");
    }

    // 10. The resident cap never releases a held or a predicted width.
    {
        const std::string gguf = routing_case(fixture, root.path, "cap", {cb::COREML_CFM_SUFFIX});
        auto set = cb::s3gen_sidecars::create(gguf, routing_options());
        set->predict({shape(8)});
        settle(*set, "cap");
        auto held = set->acquire(shape(8), steady::now());
        require(view(held, S::cfm).bucket == 16, "a call holds frames_16");
        set->predict({shape(12)});
        settle(*set, "cap");
        size_t most = 0;
        for (int tokens : {16, 20, 16, 20}) {
            set->acquire(shape(tokens), steady::now());
            require(wait_for([&] {
                        most = std::max(most, set->live(S::cfm));
                        return settled(*set);
                    }, 30000),
                    "cap: the loads finish");
            require(set->state(S::cfm, 2 * tokens) == "ready", "the requested width loads");
            require(set->state(S::cfm, 16) == "ready" && set->state(S::cfm, 24) == "ready",
                    "the held and the predicted widths stay loaded");
        }
        std::printf("[cap] at most %zu CFM functions live; frames_16 (held) and frames_24 (predicted) kept\n", most);
        require(most <= cb::COREML_RESIDENT_CAP && set->live(S::cfm) <= cb::COREML_RESIDENT_CAP,
                "never more live functions than the cap");
        set->end_call();
    }
    return 0;
#endif
}

// --------------------------------------------------------- engine fixture --

// Three segments at 60 bytes; with the CFM and HiFT sidecars present the
// Engine renders each segment's S3Gen on a worker while T3 decodes the next.
const char * OVERLAP_TEXT = "The quick brown fox jumps over the lazy dog. "
                            "Core ML renders this sentence while the next one decodes. "
                            "Every segment must come back in order.";
constexpr int OVERLAP_SENTENCE_CHARS = 60;

EngineOptions engine_options(const std::string & t3, const std::string & s3gen) {
    EngineOptions opts;
    opts.t3_gguf_path = t3;
    opts.s3gen_gguf_path = s3gen;
    opts.n_gpu_layers = 99;
    opts.seed = 42;
    opts.max_sentence_chars = OVERLAP_SENTENCE_CHARS;
    return opts;
}

bool same_pcm(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && !a.empty() && max_abs(a, b) == 0.0;
}

#ifdef TTS_CPP_USE_COREML
// The Engine's Core ML functions are all loaded (none queued or loading).
bool engine_settled(const Engine & engine) {
    const std::string json = engine.coreml_status();
    for (const char * busy : {"\"state\":\"queued\"", "\"state\":\"loading\"", "\"state\":\"warming\"",
                              "\"state\":\"deferred(memory)\""}) {
        if (json.find(busy) != std::string::npos) return false;
    }
    return true;
}

std::string read_file(const fs::path & path) {
    std::ifstream in(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

// Runs tts-cli on the overlap text; returns its exit code.
int run_cli(const std::string & cli, const std::string & t3, const std::string & s3gen, const fs::path & wav,
            const fs::path & log, const std::string & extra) {
    const std::string cmd = "'" + cli + "' --model '" + t3 + "' --s3gen-gguf '" + s3gen + "' --text '" +
                            OVERLAP_TEXT + "' --out '" + wav.string() + "' --seed 42 --n-gpu-layers 99 " +
                            "--max-sentence-chars " + std::to_string(OVERLAP_SENTENCE_CHARS) + " --verbose " + extra +
                            " 2> '" + log.string() + "'";
    const int status = std::system(cmd.c_str());
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

// Stage labels tts-cli reported ("[encoder] 12.3 ms (coreml-ane)", ...).
std::set<std::string> cli_labels(const std::string & log) {
    std::set<std::string> labels;
    std::istringstream lines(log);
    std::string line;
    while (std::getline(lines, line)) {
        const bool stage = line.find("[encoder]") != std::string::npos || line.find("[cfm_step") != std::string::npos ||
                           line.find("[hift_decode]") != std::string::npos;
        const size_t open = line.rfind('('), close = line.rfind(')');
        if (stage && open != std::string::npos && close != std::string::npos && close > open) {
            labels.insert(line.substr(open + 1, close - open - 1));
        }
    }
    return labels;
}

// While every function is still loading (held, or delayed past the whole
// run for tts-cli: a slow, cold load), the first
// synthesis runs entirely on ggml and equals a use = off run with the same
// seed, sample for sample; once loaded, later calls run on Core ML. Through
// the Engine and through tts-cli.
void slow_load_checks(const std::string & t3, const std::string & s3gen, const std::string & cli,
                      const fs::path & root) {
    auto & hooks = cb::coreml_test();
    EngineOptions off_opts = engine_options(t3, s3gen);
    off_opts.coreml.use = CoremlOptions::Use::off;
    const SynthesisResult reference = Engine(off_opts).synthesize(OVERLAP_TEXT);
    require(reference.s3gen_backend == cb::GGML_STAGE_BACKEND, "use = off runs on ggml");

    hooks.hold_loads = true;
    EngineOptions slow_opts = engine_options(t3, s3gen);
    slow_opts.coreml.cache_dir = (root / "engine-cache").string();
    Engine engine(slow_opts);
    const SynthesisResult first = engine.synthesize(OVERLAP_TEXT);
    std::printf("[slow-load engine] first call: backend=%s pending=%d, max |first - use=off| = %.3e\n",
                first.s3gen_backend.c_str(), (int) first.coreml_pending,
                first.pcm.size() == reference.pcm.size() ? max_abs(first.pcm, reference.pcm) : -1.0);
    require(first.s3gen_backend == cb::GGML_STAGE_BACKEND && first.coreml_pending,
            "the first call runs on ggml while the functions load");
    require(same_pcm(first.pcm, reference.pcm), "the first call equals a use = off run with the same seed");
    hooks.hold_loads = false;
    require(wait_for([&] { return engine_settled(engine); }, 900000), "the requested functions load");
    const SynthesisResult later = engine.synthesize(OVERLAP_TEXT);
    std::printf("[slow-load engine] later call: backend=%s pending=%d\n", later.s3gen_backend.c_str(),
                (int) later.coreml_pending);
    require(later.s3gen_backend == "coreml-ane" && !later.coreml_pending, "later calls run on Core ML");

    // tts-cli runs its own sentence loop.
    const fs::path off_wav = root / "cli-off.wav", slow_wav = root / "cli-slow.wav", warm_wav = root / "cli-warm.wav";
    const std::string cache = " --coreml-cache-dir '" + (root / "cli-cache").string() + "'";
    require(run_cli(cli, t3, s3gen, off_wav, root / "cli-off.log", "--coreml-use off") == 0, "tts-cli use = off");
    require(run_cli(cli, t3, s3gen, slow_wav, root / "cli-slow.log", "--coreml-slow-load-ms 120000" + cache) == 0,
            "tts-cli with slow loads");
    const std::set<std::string> slow_labels = cli_labels(read_file(root / "cli-slow.log"));
    require(slow_labels == std::set<std::string>{cb::GGML_STAGE_BACKEND}, "tts-cli's first call runs on ggml");
    require(!read_file(off_wav).empty() && read_file(off_wav) == read_file(slow_wav),
            "tts-cli's first call equals its use = off run");
    require(run_cli(cli, t3, s3gen, warm_wav, root / "cli-warm.log", "--coreml-wait call" + cache) == 0,
            "tts-cli once loaded");
    const std::set<std::string> warm_labels = cli_labels(read_file(root / "cli-warm.log"));
    std::printf("[slow-load cli] first call ggml and equal to use = off; loaded call labels:");
    for (const auto & l : warm_labels) std::printf(" %s", l.c_str());
    std::printf("\n");
    require(warm_labels == std::set<std::string>{"coreml-ane"}, "tts-cli runs every stage on Core ML once loaded");
}

// The real CFM latch: a Core ML failure at the second Euler step restores the
// initial noise and integrates on ggml, so the call equals use = off sample
// for sample (only the CFM sidecar is present, so the encoder and HiFT run
// on ggml in both).
void cfm_rollback_check(const std::string & s3gen, const fs::path & root) {
    const fs::path dir = stage_dir(root, s3gen, {cb::COREML_CFM_SUFFIX}, "cfm-rollback");
    const std::string gguf = (dir / fs::path(s3gen).filename()).string();
    const std::vector<int32_t> tokens = fixture_tokens(90);
    auto run = [&](const CoremlOptions & coreml) {
        s3gen_synthesize_opts opts;
        opts.s3gen_gguf_path = gguf;
        opts.n_gpu_layers = 99;
        opts.seed = 42;
        opts.coreml = coreml;
        run_result r;
        opts.pcm_out = &r.wav;
        opts.s3gen_backend_out = &r.backend;
        require(s3gen_synthesize_to_wav(tokens, opts) == 0, "s3gen_synthesize_to_wav");
        return r;
    };
    CoremlOptions off;
    off.use = CoremlOptions::Use::off;
    const run_result reference = run(off);
    CoremlOptions waited;
    waited.wait = CoremlOptions::Wait::call;  // the CFM function is loaded before the call
    waited.compute_units = "cpu_only";
    const run_result served = run(waited);
    require(served.backend == cb::MIXED_STAGE_BACKEND, "CFM runs on Core ML");
    cb::coreml_test().fail_cfm_step = 1;
    const run_result rolled = run(waited);
    require(cb::coreml_test().fail_cfm_step == -1, "the failure was injected");
    std::printf("[cfm-rollback] backend=%s, max |rolled back - use=off| = %.3e\n", rolled.backend.c_str(),
                rolled.wav.size() == reference.wav.size() ? max_abs(rolled.wav, reference.wav) : -1.0);
    require(rolled.backend == cb::GGML_STAGE_BACKEND, "after the failure the whole call ran on ggml");
    require(same_pcm(rolled.wav, reference.wav), "a CFM failure at step 2 equals a whole ggml integration from z0");
}
#endif

int engine_fixture(bool slow_load) {
#ifndef TTS_CPP_USE_COREML
    (void) slow_load;
    std::puts("SKIP: built without TTS_CPP_COREML");
    return SKIP;
#else
    const char * t3_env = std::getenv("CHATTERBOX_COREML_TEST_T3");
    const char * s3gen_env = std::getenv("CHATTERBOX_COREML_TEST_S3GEN");
    if (!t3_env || !*t3_env || !s3gen_env || !*s3gen_env) {
        std::puts("SKIP: CHATTERBOX_COREML_TEST_T3 / CHATTERBOX_COREML_TEST_S3GEN are not set");
        return SKIP;
    }
    const char * cli_env = std::getenv("CHATTERBOX_COREML_TEST_CLI");
    if (slow_load && (!cli_env || !*cli_env || !fs::exists(cli_env))) {
        std::puts("SKIP: --slow-load needs CHATTERBOX_COREML_TEST_CLI (a tts-cli binary)");
        return SKIP;
    }
    const std::string t3 = t3_env, s3gen = s3gen_env;
    for (const char * suffix : {cb::COREML_ENCODER_SUFFIX, cb::COREML_CFM_SUFFIX, cb::COREML_HIFT_SUFFIX}) {
        if (!fs::exists(cb::coreml_stage_path(s3gen, suffix))) {
            std::printf("SKIP: %s is missing\n", cb::coreml_stage_path(s3gen, suffix).c_str());
            return SKIP;
        }
    }
    env_guard units(cb::COREML_UNITS_ENV), strict(cb::COREML_STRICT_ENV), disabled(cb::COREML_DISABLE_ENV);
    env_guard cache("QVAC_COREML_CACHE_DIR");
    unsetenv(cb::COREML_UNITS_ENV);
    unsetenv(cb::COREML_STRICT_ENV);
    unsetenv(cb::COREML_DISABLE_ENV);
    unsetenv("QVAC_COREML_CACHE_DIR");
    temp_dir root("engine");
    const auto segments = tts_cpp::chatterbox::text_preprocess::split_text_for_tts(OVERLAP_TEXT,
                                                                                   OVERLAP_SENTENCE_CHARS);
    require(segments.size() == 3, "the overlap text splits into three segments");

    // First, before anything is loaded in this process.
    if (slow_load) slow_load_checks(t3, s3gen, cli_env, root.path);

    // Overlapped synthesis equals the segments synthesized one call at a time
    // (one segment never overlaps) and crossfaded in order, sample for sample:
    // T3 and the CFM noise are seeded per segment, so only order can differ.
    // wait = call: every sentence waits for its functions, so each one
    // overlaps.
    EngineOptions waited_opts = engine_options(t3, s3gen);
    waited_opts.coreml.wait = CoremlOptions::Wait::call;
    Engine engine(waited_opts);
    const SynthesisResult overlapped = engine.synthesize(OVERLAP_TEXT);
    // Timed again once every function is compiled and loaded, so the cancel
    // points below fall inside a warm request.
    const auto t0 = std::chrono::steady_clock::now();
    require(same_pcm(engine.synthesize(OVERLAP_TEXT).pcm, overlapped.pcm), "overlapped synthesis is repeatable");
    const auto elapsed = std::chrono::steady_clock::now() - t0;
    std::vector<float> sequential;
    std::set<std::string> backends;
    int tokens = 0;
    for (const std::string & segment : segments) {
        const SynthesisResult one = engine.synthesize(segment);
        tts_cpp::chatterbox::text_preprocess::append_pcm_crossfade(sequential, one.pcm, one.sample_rate,
                                                                   engine.options().crossfade_ms);
        backends.insert(one.s3gen_backend);
        tokens += one.t3_tokens;
    }
    std::printf("[overlap] %zu samples in %.2f s, backend=%s, max |overlapped - sequential| = %.3e\n",
                overlapped.pcm.size(), std::chrono::duration<double>(elapsed).count(),
                overlapped.s3gen_backend.c_str(),
                overlapped.pcm.size() == sequential.size() ? max_abs(overlapped.pcm, sequential) : -1.0);
    require(backends == std::set<std::string>{"coreml-ane"} && overlapped.s3gen_backend == "coreml-ane",
            "every segment and the aggregate report the Core ML backend");
    require(overlapped.t3_tokens == tokens, "overlapped T3 token count matches the segments'");
    require(same_pcm(overlapped.pcm, sequential), "overlapped PCM equals the in-order sequential PCM");

    // Engine::cancel() at points across the request: each run either finishes
    // or throws a cancellation (from T3, the S3Gen worker or the segment
    // boundary), never anything else, and the next request is unaffected.
    int cancelled = 0;
    for (double fraction : {0.2, 0.45, 0.7}) {
        std::thread canceller([&] {
            std::this_thread::sleep_for(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed * fraction));
            engine.cancel();
        });
        std::string error;
        try {
            engine.synthesize(OVERLAP_TEXT);
        } catch (const std::runtime_error & e) {
            error = e.what();
        }
        canceller.join();
        std::printf("[cancel at %.0f%%] %s\n", fraction * 100, error.empty() ? "finished" : error.c_str());
        require(error.empty() || error.find("cancelled") != std::string::npos, "a cancel surfaces as a cancellation");
        cancelled += !error.empty();
    }
    require(cancelled > 0, "at least one cancel lands mid-request");
    require(same_pcm(engine.synthesize(OVERLAP_TEXT).pcm, overlapped.pcm), "a cancelled engine synthesizes again");

    // A CFM sidecar that cannot serve (here, a package without frames_*
    // functions): strict mode refuses the call, and without strict the
    // segments run CFM on ggml (so they do not overlap) and aggregate as
    // mixed.
    const fs::path broken = stage_dir(root.path, s3gen, {cb::COREML_ENCODER_SUFFIX, cb::COREML_HIFT_SUFFIX}, "broken");
    const std::string broken_gguf = (broken / fs::path(s3gen).filename()).string();
    fs::create_directory_symlink(fs::absolute(cb::coreml_stage_path(s3gen, cb::COREML_HIFT_SUFFIX)),
                                 cb::coreml_stage_path(broken_gguf, cb::COREML_CFM_SUFFIX));
    EngineOptions strict_opts = engine_options(t3, broken_gguf);
    strict_opts.coreml_strict = true;
    std::string error;
    try {
        Engine(strict_opts).synthesize(OVERLAP_TEXT);
    } catch (const std::runtime_error & e) {
        error = e.what();
    }
    std::printf("[worker error] %s\n", error.c_str());
    require(error.find("cannot serve") != std::string::npos, "the worker's strict-mode error reaches the caller");
    EngineOptions lenient_opts = engine_options(t3, broken_gguf);
    lenient_opts.coreml_strict = false;
    const SynthesisResult mixed = Engine(lenient_opts).synthesize(OVERLAP_TEXT);
    std::printf("[fallback] %zu samples, backend=%s\n", mixed.pcm.size(), mixed.s3gen_backend.c_str());
    require(!mixed.pcm.empty() && mixed.s3gen_backend == cb::MIXED_STAGE_BACKEND,
            "a CFM fallback aggregates as mixed");

    cfm_rollback_check(s3gen, root.path);

    // A prediction failure on the worker during the overlap: the worker hands
    // the segment back, the dispatching thread renders it again, and the
    // result equals a later run with the same function retired (last: the
    // retirement is process-wide).
    cb::coreml_test().fail_stage = (int) cb::s3gen_stage::cfm;
    const SynthesisResult redone = engine.synthesize(OVERLAP_TEXT);
    require(cb::coreml_test().fail_stage == -1, "the worker's CFM prediction failed");
    const SynthesisResult after = engine.synthesize(OVERLAP_TEXT);
    std::printf("[overlap retire] redone backend=%s, later backend=%s\n", redone.s3gen_backend.c_str(),
                after.s3gen_backend.c_str());
    require(same_pcm(redone.pcm, after.pcm), "the redone segment equals a run with the function retired");
    s3gen_unload();
    return 0;
#endif
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc > 1 && std::string(argv[1]) == "--stages-fixture") return stages_fixture();
        if (argc > 1 && std::string(argv[1]) == "--shape-fixture") return shape_fixture();
        if (argc > 1 && std::string(argv[1]) == "--pending-fixture") return pending_fixture();
        if (argc > 1 && std::string(argv[1]) == "--engine-fixture") {
            return engine_fixture(argc > 2 && std::string(argv[2]) == "--slow-load");
        }
        check_options();
        check_naming();
        check_buckets();
        check_windows();
        check_warm_estimates();
        check_shapes_and_estimate();
        check_cfm_rollback();
        check_fit_and_detection();
        std::puts("test-chatterbox-coreml: ok");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "test-chatterbox-coreml: FAIL: %s\n", e.what());
        return 1;
    }
}
