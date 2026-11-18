// Pocket Core ML sidecars: the FlowLM step and the Mimi decoder.
//
//   test-pocket-coreml                  policy, naming, RoPE tables, sidecar
//                                       detection, fit refusal, option checks
//   test-pocket-coreml --shape-fixture  a tiny real step reproduces its float32
//                                       reference after the prefix is written
//                                       into its state and after a reset clears
//                                       a NaN row, and a random full-size Mimi
//                                       decoder streams its reference;
//                                       every declared shape, the schema, the
//                                       checkpoint hash and state presence are
//                                       checked before predicting
//   test-pocket-coreml --engine-fixture the real sidecars against ggml FlowLM
//                                       and Mimi, the ggml replay of frames a
//                                       retired step generated, and the
//                                       Engine's routing, labels and strict mode
//   test-pocket-coreml --generated BUNDLE --cache-dir DIR [--memory-schedule DIR]
//                                       sidecars generated from a bundle's
//                                       GGUFs (no prebuilt sidecar next to
//                                       them): routes and fit, a call that
//                                       runs on ggml while the sidecars are
//                                       pending (matching Core ML off),
//                                       readiness at construction, parity of
//                                       the generated step and decoder against
//                                       ggml, adoption by a second Engine, fit
//                                       unchanged by a populated cache, strict
//                                       mode; with --memory-schedule, the
//                                       overlapped memory schedule against the
//                                       templates' constants
//
// The shape fixture reads POCKET_COREML_TEST_SHAPES (the directory
// test/prepare_pocket_coreml_fixture.py writes); the engine fixture reads
// POCKET_COREML_TEST_BUNDLE (a Pocket bundle directory with the exported
// flow-lm-step.mlmodelc and mimi-decoder.mlmodelc next to their GGUFs). Each reports a skip (77)
// without its inputs or without Core ML.

#include "pocket/coreml.h"
#include "pocket/flow_lm.h"
#include "pocket/frontend.h"
#include "pocket/mimi.h"
#include "coreml_sidecar_path.h"
#include "json.hpp"
#include "npy.h"
#include "../test_env_portable.h"
#include "tts-cpp/pocket/engine.h"
#include "tts-cpp/pocket/fit.h"

#ifdef TTS_CPP_USE_COREML
#include "moss/coreml/runner.h"
#include "memory_probe.h"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>

#if defined(__APPLE__)
#include <sys/clonefile.h>
#include <sys/sysctl.h>
#endif
#ifndef _WIN32
#include <sys/wait.h>
#endif
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace pd = tts_cpp::pocket::detail;
namespace fs = std::filesystem;
using tts_cpp::pocket::Engine;
using tts_cpp::pocket::EngineOptions;

namespace {

constexpr int SKIP = 77;

void require(bool ok, const std::string & what) {
    if (!ok) throw std::runtime_error(what);
}

template <class F> bool throws(F fn) {
    try { fn(); } catch (const std::exception &) { return true; }
    return false;
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
        : path(fs::path(test_tmpdir()) / ("pocket-coreml-" + tag + "-" + test_process_tag())) {
        std::error_code error;
        fs::remove_all(path, error);
        fs::create_directories(path);
    }
    ~temp_dir() {
        std::error_code error;
        fs::remove_all(path, error);
    }
};

double max_error(const std::vector<float> & got, const std::vector<float> & want) {
    require(got.size() == want.size(), "size mismatch");
    double error = 0, scale = 1;
    for (size_t i = 0; i < got.size(); ++i) {
        require(std::isfinite(got[i]), "non-finite output");
        error = std::max(error, std::abs(double(got[i]) - want[i]));
        scale = std::max(scale, std::abs(double(want[i])));
    }
    return error / scale;
}

double snr_db(const std::vector<float> & got, const std::vector<float> & want) {
    require(got.size() == want.size(), "size mismatch");
    double signal = 0, noise = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        require(std::isfinite(got[i]), "non-finite output");
        signal += double(want[i]) * want[i];
        noise += (double(got[i]) - want[i]) * (double(got[i]) - want[i]);
    }
    return 10 * std::log10(signal / std::max(noise, 1e-30));
}

double cosine(const std::vector<float> & a, const std::vector<float> & b) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) { dot += double(a[i]) * b[i]; na += double(a[i]) * a[i]; nb += double(b[i]) * b[i]; }
    return dot / std::sqrt(na * nb);
}

void check_policy() {
    env_guard units(pd::COREML_UNITS_ENV), strict(pd::COREML_STRICT_ENV), disabled(pd::COREML_DISABLE_ENV);
    unsetenv(pd::COREML_UNITS_ENV);
    unsetenv(pd::COREML_STRICT_ENV);
    unsetenv(pd::COREML_DISABLE_ENV);
    auto policy = pd::resolve_coreml_policy();
    require(policy.compute_units.empty() && !policy.strict && !policy.disabled, "default policy");
    require(pd::coreml_units(policy, pd::COREML_STEP_DEFAULT_UNITS) == "cpu_and_gpu" &&
            pd::coreml_units(policy, pd::COREML_DECODER_DEFAULT_UNITS) == "cpu_and_ane", "per-sidecar default units");

    setenv(pd::COREML_UNITS_ENV, "all", 1);
    setenv(pd::COREML_STRICT_ENV, "0", 1);  // presence means strict
    policy = pd::resolve_coreml_policy();
    require(policy.compute_units == "all" && policy.strict, "inherited policy");
    require(pd::coreml_units(policy, pd::COREML_DECODER_DEFAULT_UNITS) == "all", "set units apply to every sidecar");
    policy = pd::resolve_coreml_policy("cpu_only", false);
    require(policy.compute_units == "cpu_only" && !policy.strict, "explicit policy overrides the environment");

    setenv(pd::COREML_UNITS_ENV, "gpu_please", 1);
    require(pd::resolve_coreml_policy().compute_units.empty(), "invalid inherited units keep the defaults");
    require(throws([] { pd::resolve_coreml_policy("gpu_please"); }), "invalid explicit units throw");

    setenv(pd::COREML_DISABLE_ENV, "1", 1);
    require(pd::resolve_coreml_policy().disabled, "POCKET_COREML_DISABLE");
}

void check_naming() {
    require(pd::coreml_step_path("models/pocket/flow-lm.gguf") == "models/pocket/flow-lm-step.mlmodelc", "bundle name");
    require(pd::coreml_step_path("m/flow-lm-f16.gguf") == "m/flow-lm-step.mlmodelc", "quant tag dropped");
    require(pd::coreml_decoder_path("models/pocket/mimi.gguf") == "models/pocket/mimi-decoder.mlmodelc", "decoder name");
}

void check_rope() {
    std::vector<float> cos, sin;
    pd::coreml_rope(0, 8, 10000.0f, cos, sin);
    require(cos.size() == 4 && std::all_of(cos.begin(), cos.end(), [](float c) { return c == 1.0f; }) &&
            std::all_of(sin.begin(), sin.end(), [](float s) { return s == 0.0f; }), "position 0");
    pd::coreml_rope(37, 8, 10000.0f, cos, sin);
    for (int i = 0; i < 4; ++i) {
        const double theta = 37.0 * std::pow(10000.0, -2.0 * i / 8);
        require(std::abs(cos[size_t(i)] - std::cos(theta)) < 1e-6 && std::abs(sin[size_t(i)] - std::sin(theta)) < 1e-6,
                "adjacent-pair frequencies");
    }
}

// Detection is presence-only, so an empty sidecar directory next to a GGUF
// path that does not exist is enough; fit refuses before reading anything.
void check_detection_and_fit() {
    env_guard units(pd::COREML_UNITS_ENV), strict(pd::COREML_STRICT_ENV), disabled(pd::COREML_DISABLE_ENV);
    unsetenv(pd::COREML_UNITS_ENV);
    unsetenv(pd::COREML_STRICT_ENV);
    unsetenv(pd::COREML_DISABLE_ENV);
    temp_dir dir("detect");
    const std::string gguf = (dir.path / "flow-lm.gguf").string(), mimi_gguf = (dir.path / "mimi.gguf").string();
    require(!pd::coreml_step_enabled(gguf, pd::resolve_coreml_policy()), "no sidecar");
    require(!pd::coreml_decoder_enabled(mimi_gguf, pd::resolve_coreml_policy()), "no decoder sidecar");
    // The decoder alone makes fit refuse too.
    fs::create_directories(pd::coreml_decoder_path(mimi_gguf));
    {
        tts_cpp::pocket::FitOptions decoder_only;
        decoder_only.text = "Hi.";
        decoder_only.engine.flow_lm_path = gguf;
        decoder_only.engine.mimi_path = mimi_gguf;
        decoder_only.engine.frontend_path = (dir.path / "frontend.json").string();
        decoder_only.engine.voice_path = (dir.path / "voice.gguf").string();
#ifdef TTS_CPP_USE_COREML
        require(pd::coreml_decoder_enabled(mimi_gguf, pd::resolve_coreml_policy()), "decoder sidecar detected");
        require(tts_cpp::pocket::fit_params(decoder_only).reason == "coreml-memory-estimate-unavailable",
                "fit refuses the Core ML decoder");
#else
        require(!pd::coreml_decoder_enabled(mimi_gguf, pd::resolve_coreml_policy()), "no Core ML decoder in this build");
#endif
    }
    fs::create_directories(pd::coreml_step_path(gguf));
    tts_cpp::pocket::FitOptions fit;
    fit.text = "Hi.";
    fit.engine.flow_lm_path = gguf;
    fit.engine.mimi_path = (dir.path / "mimi.gguf").string();
    fit.engine.frontend_path = (dir.path / "frontend.json").string();
    fit.engine.voice_path = (dir.path / "voice.gguf").string();
#ifdef TTS_CPP_USE_COREML
    require(pd::coreml_step_enabled(gguf, pd::resolve_coreml_policy()), "sidecar detected");
    require(tts_cpp::pocket::fit_params(fit).reason == "coreml-memory-estimate-unavailable", "fit refuses Core ML");
    setenv(pd::COREML_DISABLE_ENV, "1", 1);
    require(!pd::coreml_step_enabled(gguf, pd::resolve_coreml_policy()), "disabled sidecar");
    require(tts_cpp::pocket::fit_params(fit).reason != "coreml-memory-estimate-unavailable", "disabled sidecar is fitted");
#else
    require(!pd::coreml_step_enabled(gguf, pd::resolve_coreml_policy()), "no Core ML in this build");
    require(tts_cpp::pocket::fit_params(fit).reason != "coreml-memory-estimate-unavailable", "no Core ML refusal");
#endif
    fit.engine.coreml_compute_units = "gpu_please";
    require(tts_cpp::pocket::fit_params(fit).reason == "invalid-arguments", "fit rejects invalid compute units");
    // Rejected before any model file is opened.
    std::string error;
    try { Engine engine(fit.engine); } catch (const std::exception & e) { error = e.what(); }
    require(error.find("coreml_compute_units") != std::string::npos, "Engine rejects invalid compute units: " + error);
}

// Options, routes and the fit arithmetic of the shared Core ML runtime, with no
// model: an explicit cpu_and_ane for the step is refused when options are
// resolved; the environment's or the legacy all-stage cpu_and_ane leaves the
// step on its default units; the envelope sums the generated stages' resident
// parts and charges one transient; memory-degrade needs a budget.
void check_coreml_options() {
    env_guard units(pd::COREML_UNITS_ENV), disabled(pd::COREML_DISABLE_ENV);
    unsetenv(pd::COREML_UNITS_ENV);
    unsetenv(pd::COREML_DISABLE_ENV);
    pd::CoremlEngineOptions o;
    o.flow_lm_path = "missing/flow-lm.gguf";
    o.mimi_path = "missing/mimi.gguf";
    o.coreml.compute_units = "cpu_and_ane";
    std::string error;
    try { pd::coreml_plan_route(pd::CoremlStage::step, o); } catch (const std::invalid_argument & e) { error = e.what(); }
    require(error.find("cannot run on cpu_and_ane") != std::string::npos, "explicit cpu_and_ane step refused: " + error);
    require(!throws([&] { pd::coreml_plan_route(pd::CoremlStage::decoder, o); }), "cpu_and_ane decoder accepted");
    o.coreml.compute_units.reset();
    o.step_units = "cpu_and_ane";
    require(throws([&] { pd::coreml_plan_route(pd::CoremlStage::step, o); }), "explicit step units cpu_and_ane refused");
    o.step_units.clear();
    o.decoder_units = "gpu_please";
    error.clear();
    try { pd::coreml_plan_route(pd::CoremlStage::step, o); } catch (const std::invalid_argument & e) { error = e.what(); }
    require(error.find("coreml_decoder_units") != std::string::npos, "invalid decoder units named: " + error);
    o.decoder_units.clear();
    o.coreml.strict_wait_ms = -1;
    require(throws([&] { pd::coreml_plan_route(pd::CoremlStage::step, o); }), "negative strict_wait_ms");
    o.coreml.strict_wait_ms.reset();
    o.legacy_units = "cpu_and_ane";
#ifdef TTS_CPP_USE_COREML
    require(pd::coreml_plan_route(pd::CoremlStage::step, o).compute_units == "cpu_and_gpu",
            "legacy all-stage cpu_and_ane leaves the step on cpu_and_gpu");
    require(pd::coreml_plan_route(pd::CoremlStage::decoder, o).compute_units == "cpu_and_ane",
            "legacy all-stage units reach the decoder");
    o.legacy_units.clear();
    setenv(pd::COREML_UNITS_ENV, "cpu_and_ane", 1);
    require(pd::coreml_plan_route(pd::CoremlStage::step, o).compute_units == "cpu_and_gpu",
            "POCKET_COREML_COMPUTE_UNITS=cpu_and_ane leaves the step on cpu_and_gpu");
    unsetenv(pd::COREML_UNITS_ENV);
    const auto none = pd::coreml_plan_route(pd::CoremlStage::step, o);
    require(none.route == "none", "no GGUF, no prebuilt sidecar: route none, got " + none.route);
#endif
    // Envelope: resident parts summed, one transient, the larger margin.
    pd::CoremlRoute a, b;
    a.route = b.route = "generate";
    a.resident_bytes = 100; a.transient_bytes = 40; a.margin = 1.0;
    b.resident_bytes = 10; b.transient_bytes = 70; b.margin = 1.0;
    require(pd::coreml_envelope_bytes(a, b) == 180, "envelope = resident sum + largest transient");
    b.margin = 1.25;
    require(pd::coreml_envelope_bytes(a, b) == 225, "envelope with the larger margin");
    b.route = "none";
    require(pd::coreml_envelope_bytes(a, b) == 140, "only generated stages are charged");
    a.route = "prebuilt";
    require(pd::coreml_envelope_bytes(a, b) == 0, "no generated stage, no envelope");
    // Verdicts: memory-degrade only with a budget that the ggml projection fits.
    auto v = pd::decide_coreml_fit(100, 50, true, 120, std::nullopt);
    require(v.ggml_only_fits && !v.coreml_fits && !v.degraded && !v.fits && v.charged == 150,
            "without a budget the Core ML-inclusive verdict stands");
    v = pd::decide_coreml_fit(100, 50, true, 1000, int64_t(120));
    require(v.degraded && v.fits && v.charged == 100, "a budget only ggml fits degrades");
    v = pd::decide_coreml_fit(100, 50, true, 120, int64_t(1) << 50);
    require(v.degraded && v.fits, "degrade is decided on min(free, budget)");
    v = pd::decide_coreml_fit(100, 50, true, 1000, int64_t(90));
    require(!v.degraded && !v.ggml_only_fits && !v.coreml_fits && !v.fits,
            "a budget below the ggml projection refuses, however much memory is free");
    v = pd::decide_coreml_fit(100, 50, true, 1000, int64_t(160));
    require(!v.degraded && v.fits && v.charged == 150, "a budget both fit keeps Core ML");
    v = pd::decide_coreml_fit(100, 50, false, 120, int64_t(110));
    require(!v.degraded && v.fits && v.charged == 100, "no generated stage, no degrade");
    v = pd::decide_coreml_fit(100, 50, false, 1000, int64_t(90));
    require(!v.fits, "without a generated stage the budget still caps the ggml verdict");
    // One ceiling: the smaller of the fit's budget and the Core ML budget.
    require(!pd::effective_fit_budget(0, std::nullopt), "no budget set");
    require(pd::effective_fit_budget(0, int64_t(70)) == int64_t(70), "the Core ML budget alone");
    require(pd::effective_fit_budget(80, std::nullopt) == int64_t(80), "the fit's budget alone enables degrade");
    require(pd::effective_fit_budget(80, int64_t(70)) == int64_t(70) &&
                pd::effective_fit_budget(60, int64_t(70)) == int64_t(60),
            "both set: the smaller");
    require(pd::effective_fit_budget(~uint64_t(0), std::nullopt) == INT64_MAX, "a huge fit budget saturates");
    std::puts("[options] Core ML options, routes and fit arithmetic: ok");
}

std::vector<float> npy_floats(const fs::path & path) {
    const npy_array array = npy_load(path.string());
    const float * data = npy_as_f32(array);
    return std::vector<float>(data, data + array.n_elements());
}

#ifdef TTS_CPP_USE_COREML
// The random full-size decoder streams its eager reference one frame per
// call (past the attention window), in calls of several frames, and again
// after reset(); each defective sidecar is rejected at load.
void check_decoder_fixture(const fs::path & root) {
    const fs::path reference = root / "good" / "reference";
    std::ifstream config_file(reference / "config.json");
    const auto config = nlohmann::json::parse(config_file);
    const std::string source = config["source"];
    const auto sidecar = [&](const std::string & variant) { return (root / variant / "mimi-decoder.mlmodelc").string(); };
    const auto latents = npy_floats(reference / "latents.npy"), want = npy_floats(reference / "pcm.npy");
    for (const char * units : {"cpu_only", pd::COREML_DECODER_DEFAULT_UNITS}) {
        pd::CoremlDecoder decoder(sidecar("good"), units, source);
        const auto one_by_one = decoder.decode(latents);
        const double snr = snr_db(one_by_one, want);
        decoder.reset();
        std::vector<float> split;
        const size_t frame = pd::CoremlDecoderGeometry::latent;
        for (size_t at = 0, count = 3; at < latents.size(); at += count * frame, count = count % 5 + 1) {
            const size_t end = std::min(latents.size(), at + count * frame);
            const auto part = decoder.decode(std::vector<float>(latents.begin() + at, latents.begin() + end));
            split.insert(split.end(), part.begin(), part.end());
        }
        std::printf("[mimi good %s] SNR %.1f dB vs the eager stream; split calls %.2e from one-by-one\n", units, snr,
                    max_error(split, one_by_one));
        // Broken state feedback scores <= 6 dB on these weights. The Neural
        // Engine computes them at ~25 dB (the real checkpoint at ~46 dB).
        require(snr > (std::string(units) == "cpu_only" ? 30 : 15),
                std::string("tiny decoder diverges from its reference on ") + units);
        require(max_error(split, one_by_one) < 1e-6, "split decode calls change the stream");
    }
    // Strict mode keeps a decoder whose prediction failed; the next chunk's
    // reset() must clear the non-finite stream state. The Neural Engine
    // decodes a NaN latent to finite PCM, so the failure is forced on the CPU.
    {
        pd::CoremlDecoder decoder(sidecar("good"), "cpu_only", source);
        require(throws([&] { decoder.decode(std::vector<float>(pd::CoremlDecoderGeometry::latent, NAN)); }),
                "a NaN latent decoded");
        decoder.reset();
        require(snr_db(decoder.decode(latents), want) > 30, "reset after a failure leaves the stream broken");
    }
    require(throws([&] { pd::CoremlDecoder decoder(sidecar("good"), "cpu_only", "another-mimi"); }),
            "a decoder of another checkpoint");
    { pd::CoremlDecoder probe(sidecar("probe"), "cpu_only", source); }
    std::vector<std::string> variants = {"latents", "cos", "sin", "mask", "keys", "values", "pcm", "new_keys",
                                         "new_values", "source", "stateful"};
    for (const char * h : {"h_upsample", "h_conv0", "h_convtr0", "h_residual0", "h_convtr1", "h_residual1", "h_convtr2",
                           "h_residual2", "h_final"}) {
        variants.push_back(h);
        variants.push_back(std::string("new_") + h);
    }
    for (const auto & variant : variants) {
        std::string error;
        try {
            pd::CoremlDecoder decoder(sidecar(variant), "cpu_only", source);
        } catch (const std::exception & e) {
            error = e.what();
        }
        require(!error.empty(), "accepted the mimi " + variant + " variant");
        std::printf("[mimi %s] rejected: %s\n", variant.c_str(), error.c_str());
    }
}
#endif

int shape_fixture() {
#ifndef TTS_CPP_USE_COREML
    std::puts("SKIP: built without Core ML (TTS_CPP_COREML=OFF)");
    return SKIP;
#else
    const char * root_env = std::getenv("POCKET_COREML_TEST_SHAPES");
    if (root_env == nullptr || !*root_env) {
        std::puts("SKIP: set POCKET_COREML_TEST_SHAPES to the prepare_pocket_coreml_fixture.py output");
        return SKIP;
    }
    const fs::path root(root_env), reference = root / "good" / "reference";
    std::ifstream config_file(reference / "config.json");
    const auto config = nlohmann::json::parse(config_file);
    pd::FlowConfig cfg;
    cfg.dim = config["dim"]; cfg.heads = config["heads"]; cfg.layers = config["layers"];
    cfg.ff_dim = config["ff_dim"]; cfg.latent_dim = config["latent_dim"]; cfg.flow_dim = config["flow_dim"];
    cfg.flow_depth = config["flow_depth"]; cfg.vocab_size = config["vocab_size"]; cfg.rope_base = config["rope_base"];
    const int context = config["context"], prefix = config["prefix"], frames = config["frames"];
    const std::string source = config["source"];
    const auto sidecar = [&](const char * variant) { return (root / variant / "flow-lm-step.mlmodelc").string(); };

    // The real step reproduces its float32 reference on every placement.
    const auto keys = npy_floats(reference / "prefix_keys.npy"), values = npy_floats(reference / "prefix_values.npy");
    const auto latents = npy_floats(reference / "latents.npy"), noises = npy_floats(reference / "noises.npy");
    const auto hidden_ref = npy_floats(reference / "hidden.npy"), next_ref = npy_floats(reference / "next.npy");
    pd::CacheRows rows;
    rows.count = prefix;
    const size_t layer_size = size_t(prefix) * cfg.dim;
    for (int l = 0; l < cfg.layers; ++l) {
        rows.keys.emplace_back(keys.begin() + l * layer_size, keys.begin() + (l + 1) * layer_size);
        rows.values.emplace_back(values.begin() + l * layer_size, values.begin() + (l + 1) * layer_size);
    }
    const auto at = [](const std::vector<float> & all, int row, int width) {
        return std::vector<float>(all.begin() + size_t(row) * width, all.begin() + size_t(row + 1) * width);
    };
    for (const char * units : {"cpu_only", pd::COREML_STEP_DEFAULT_UNITS}) {
        pd::CoremlStep step(sidecar("good"), units, cfg, source);
        require(step.context() == context, "context read from the sidecar");
        std::vector<float> hidden, next;
        const auto replay = [&](const char * tag) {
            for (int f = 0; f < frames; ++f) {
                step.step(prefix + f, at(latents, f, cfg.latent_dim), at(noises, f, cfg.latent_dim), hidden, next);
                const double eh = max_error(hidden, at(hidden_ref, f, cfg.dim));
                const double en = max_error(next, at(next_ref, f, cfg.latent_dim));
                std::printf("[%s %s] frame %d: hidden error %.2e, next latent error %.2e\n", tag, units, f, eh, en);
                require(eh < 3e-2 && en < 3e-2, std::string("tiny step diverges from its reference on ") + units);
            }
        };
        step.write_rows(rows);
        replay("good");

        // A failed prediction can leave NaN in a row past the prefix (here
        // written directly). Rewriting the prefix keeps it: the blended write
        // multiplies it by zero and the attention mask only adds to its score.
        // reset() clears it and the frames replay to the reference.
        const std::vector<std::vector<float>> nan_rows(size_t(cfg.layers), std::vector<float>(size_t(cfg.dim), NAN));
        step.write_rows({prefix + 1, 1, nan_rows, nan_rows});
        require(throws([&] { replay("poisoned"); }), "a NaN row past the prefix left the step finite");
        step.write_rows(rows);
        require(throws([&] { replay("prefix rewritten"); }), "rewriting the prefix cleared a NaN row");
        step.reset(rows);
        replay("reset");

        const std::vector<float> zeros(size_t(cfg.latent_dim));
        require(throws([&] { step.step(context, zeros, zeros, hidden, next); }), "position past the context");
        require(throws([&] { pd::CacheRows late = rows; late.start = context - 1; step.write_rows(late); }),
                "rows past the context");
    }
    require(throws([&] { pd::CoremlStep step(sidecar("good"), "cpu_only", cfg, "another-checkpoint"); }),
            "a sidecar of another checkpoint");

    // The probe declares the expected features; each other variant has one defect.
    { pd::CoremlStep probe(sidecar("probe"), "cpu_only", cfg, source); }
    for (const char * variant : {"latent", "noise", "cos", "sin", "mask", "write_mask", "hidden", "next_latent", "keys",
                                 "values", "schema", "source", "stateless"}) {
        std::string error;
        try {
            pd::CoremlStep step(sidecar(variant), "cpu_only", cfg, source);
        } catch (const std::exception & e) {
            error = e.what();
        }
        require(!error.empty(), std::string("accepted the ") + variant + " variant");
        std::printf("[%s] rejected: %s\n", variant, error.c_str());
    }

    // The runner bounds every state write by the declared buffer.
    moss_coreml_model * model = moss_coreml_load(sidecar("probe").c_str(), "cpu_only");
    require(model != nullptr, "probe loads in the runner");
    int64_t dims[8] = {};
    require(moss_coreml_state_dims(model, "keys", dims, 8) == 4 && dims[2] == context, "state dims");
    const int64_t head_dim = cfg.dim / cfg.heads;
    std::vector<float> row(size_t(cfg.heads * head_dim), 1.0f);
    const int64_t count[4] = {1, cfg.heads, 1, head_dim};
    const int64_t inside[4] = {0, 0, context - 1, 0}, outside[4] = {0, 0, context, 0};
    const int status_inside = moss_coreml_write_state(model, "keys", row.data(), inside, count, 4);
    const int status_outside = moss_coreml_write_state(model, "keys", row.data(), outside, count, 4);
    const int status_rank = moss_coreml_write_state(model, "keys", row.data(), inside, count, 3);
    const int status_name = moss_coreml_write_state(model, "kv", row.data(), inside, count, 4);
    moss_coreml_free(model);
    require(status_inside == 0 && status_outside != 0 && status_rank != 0 && status_name != 0, "state write bounds");
    check_decoder_fixture(root / "mimi");
    std::puts("test-pocket-coreml --shape-fixture: ok");
    return 0;
#endif
}

EngineOptions bundle_options(const fs::path & dir) {
    EngineOptions opts;
    opts.flow_lm_path = (dir / "flow-lm.gguf").string();
    opts.mimi_path = (dir / "mimi.gguf").string();
    opts.frontend_path = (dir / "frontend.json").string();
    opts.voice_path = (dir / "voice.gguf").string();
    return opts;
}

// The sidecar against ggml F32 FlowLM, teacher-forced on ggml's latents.
void check_stage_parity(const EngineOptions & opts, const std::string & step_path = {}) {
    pd::FlowLM lm(opts.flow_lm_path, 2048, 4);
    const auto voice = lm.read_voice(opts.voice_path);
    pd::CoremlStep step(step_path.empty() ? pd::coreml_step_path(opts.flow_lm_path) : step_path,
                        pd::COREML_STEP_DEFAULT_UNITS, lm.config(), lm.source_hash());
    step.write_rows({0, voice.frames, voice.keys, voice.values});
    lm.restore_voice(voice);
    pd::Frontend frontend(opts.frontend_path);
    lm.prefill(lm.text_embeddings(frontend.encode("The quick brown fox jumps over the lazy dog.")));
    step.write_rows(lm.cache_rows(voice.frames, lm.position() - voice.frames));
    std::mt19937 rng(7);
    std::normal_distribution<float> normal(0.0f, std::sqrt(0.3f));
    std::vector<float> previous, hidden, next, noise(size_t(lm.config().latent_dim));
    double worst_cosine = 1, worst_eos = 0, worst_latent = 0;
    for (int f = 0; f < 48; ++f) {
        for (auto & x : noise) x = normal(rng);
        const int position = lm.position();
        const auto condition = lm.advance(previous);
        const auto expected = lm.sample(condition.hidden, noise, 1);
        step.step(position, previous.empty() ? lm.bos_latent() : previous, noise, hidden, next);
        worst_cosine = std::min(worst_cosine, cosine(hidden, condition.hidden));
        worst_eos = std::max(worst_eos, double(std::abs(lm.eos_logit(hidden) - condition.eos_logit)));
        worst_latent = std::max(worst_latent, max_error(next, expected));
        previous = expected;
    }
    std::printf("[stage] 48 frames: hidden cosine >= %.6f, EOS logit error <= %.4f, next latent error <= %.4f\n",
                worst_cosine, worst_eos, worst_latent);
    require(worst_cosine > 0.9995 && worst_eos < 0.1 && worst_latent < 0.05, "Core ML step diverges from ggml FlowLM");
}

// A retired sidecar's frames are prefilled into the ggml cache in one batch;
// the next frame must match frame-by-frame ggml decoding.
void check_replay(const EngineOptions & opts) {
    pd::FlowLM lm(opts.flow_lm_path, 2048, 4);
    const auto voice = lm.read_voice(opts.voice_path);
    const auto tokens = pd::Frontend(opts.frontend_path).encode("Replay the generated frames.");
    lm.restore_voice(voice);
    lm.prefill(lm.text_embeddings(tokens));
    std::vector<float> previous, fed;
    std::mt19937 rng(11);
    std::normal_distribution<float> normal(0.0f, 0.5f);
    std::vector<float> noise(size_t(lm.config().latent_dim));
    for (int f = 0; f < 6; ++f) {
        const auto input = previous.empty() ? lm.bos_latent() : previous;
        fed.insert(fed.end(), input.begin(), input.end());
        const auto condition = lm.advance(previous);
        for (auto & x : noise) x = normal(rng);
        previous = lm.sample(condition.hidden, noise, 1);
    }
    const auto sequential = lm.advance(previous);
    lm.restore_voice(voice);
    lm.prefill(lm.text_embeddings(tokens));
    lm.prefill(lm.audio_embeddings(fed));
    const auto replayed = lm.advance(previous);
    const double error = max_error(replayed.hidden, sequential.hidden);
    std::printf("[replay] hidden error %.2e, EOS %.4f vs %.4f\n", error, replayed.eos_logit, sequential.eos_logit);
    require(error < 1e-3 && std::abs(replayed.eos_logit - sequential.eos_logit) < 1e-2, "ggml replay diverges");
}

// The real decoder against ggml Mimi on the same streamed latents.
void check_decoder_parity(const EngineOptions & opts, const std::string & decoder_path = {}) {
    pd::Mimi mimi(opts.mimi_path, 4);
    pd::CoremlDecoder decoder(decoder_path.empty() ? pd::coreml_decoder_path(opts.mimi_path) : decoder_path,
                              pd::COREML_DECODER_DEFAULT_UNITS, mimi.source_hash());
    std::mt19937 rng(3);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> latents(size_t(40) * pd::Mimi::latent_dim());
    for (auto & x : latents) x = normal(rng);
    mimi.reset_decoder();
    std::vector<float> want;
    for (size_t at = 0; at < latents.size(); at += size_t(8) * pd::Mimi::latent_dim()) {
        const auto part = mimi.decode(std::vector<float>(latents.begin() + at, latents.begin() + at + 8 * pd::Mimi::latent_dim()));
        want.insert(want.end(), part.begin(), part.end());
    }
    const double snr = snr_db(decoder.decode(latents), want);
    std::printf("[decoder] %s: 40 frames, SNR %.1f dB vs ggml Mimi\n", decoder.label().c_str(), snr);
    require(snr > 35, "Core ML Mimi decoder diverges from ggml Mimi");
}

void check_engine(const EngineOptions & prebuilt) {
    env_guard units(pd::COREML_UNITS_ENV), strict(pd::COREML_STRICT_ENV), disabled(pd::COREML_DISABLE_ENV);
    unsetenv(pd::COREML_UNITS_ENV);
    unsetenv(pd::COREML_STRICT_ENV);
    unsetenv(pd::COREML_DISABLE_ENV);
    // The sidecars load in the background: pin readiness so every comparison
    // below runs on Core ML.
    EngineOptions base = prebuilt;
    base.coreml.wait = tts_cpp::CoremlOptions::Wait::construct;
    const std::string text = "The quick brown fox jumps over the lazy dog. It barked twice.";
    Engine coreml(base);
    require(coreml.flow_lm_on_coreml() && coreml.mimi_on_coreml(), "sidecars loaded");
    const auto batch = coreml.synthesize(text);
    std::vector<float> streamed;
    const auto stream = coreml.synthesize_stream(text, [&](const float * pcm, size_t n, int) {
        streamed.insert(streamed.end(), pcm, pcm + n); return true;
    });
    require(batch.flow_lm_backend.rfind("coreml-", 0) == 0 && stream.flow_lm_backend == batch.flow_lm_backend,
            "Core ML label, got " + batch.flow_lm_backend);
    require(batch.mimi_backend.rfind("coreml-", 0) == 0 && stream.mimi_backend == batch.mimi_backend,
            "Core ML Mimi label, got " + batch.mimi_backend);
    require(max_error(streamed, batch.pcm) < 1e-4, "stream and batch differ on Core ML");

    setenv(pd::COREML_DISABLE_ENV, "1", 1);
    Engine ggml(base);
    unsetenv(pd::COREML_DISABLE_ENV);
    const auto reference = ggml.synthesize(text);
    require(!ggml.flow_lm_on_coreml() && !ggml.mimi_on_coreml() && reference.flow_lm_backend == "ggml" &&
            reference.mimi_backend == "ggml", "POCKET_COREML_DISABLE keeps ggml");
    std::printf("[engine] %s: %d frames; ggml: %d frames\n", batch.flow_lm_backend.c_str(), batch.generated_frames,
                reference.generated_frames);
    require(std::abs(batch.generated_frames - reference.generated_frames) <= 4, "Core ML changes the speech length");

    // Several chunks in one call: each copies its text rows and starts at a
    // lower position than the previous chunk's last frame.
    EngineOptions chunked = base;
    chunked.max_tokens = 12;
    const auto several = Engine(chunked).synthesize(text);
    setenv(pd::COREML_DISABLE_ENV, "1", 1);
    const auto several_ggml = Engine(chunked).synthesize(text);
    unsetenv(pd::COREML_DISABLE_ENV);
    std::printf("[engine] %d-token chunks: %s %d frames; ggml %d frames\n", chunked.max_tokens,
                several.flow_lm_backend.c_str(), several.generated_frames, several_ggml.generated_frames);
    require(several.flow_lm_backend == batch.flow_lm_backend && several.mimi_backend == batch.mimi_backend,
            "multi-chunk call left Core ML");
    require(std::abs(several.generated_frames - several_ggml.generated_frames) <= 8, "multi-chunk speech length");

    EngineOptions steps = base;
    steps.steps = 2;
    require(Engine(steps).synthesize("Hi.").flow_lm_backend == "mixed", "steps > 1 run the flow on ggml");

    // A chunk past the sidecar's context: strict refuses before any PCM.
    EngineOptions long_chunk = base;
    long_chunk.coreml_strict = true;
    long_chunk.max_tokens = 1024;
    long_chunk.context = 4096;
    std::string words;
    for (int i = 0; i < 40; ++i) words += "one two three four five six seven eight nine ten ";
    Engine strict_engine(long_chunk);
    int callbacks = 0;
    std::string error;
    try {
        strict_engine.synthesize_stream(words, [&](const float *, size_t, int) { ++callbacks; return true; });
    } catch (const std::exception & e) {
        error = e.what();
    }
    require(error.find("Core ML strict policy") != std::string::npos,
            "strict mode serves a chunk past the sidecar's context: " + error);
    require(callbacks == 0, "strict refusal after PCM");

    tts_cpp::pocket::FitOptions fit;
    fit.engine = base;
    fit.text = text;
    require(tts_cpp::pocket::fit_params(fit).reason == "coreml-memory-estimate-unavailable", "fit refuses the sidecar");
}

#ifdef TTS_CPP_USE_COREML
using Json = nlohmann::json;
using CoremlOptions = tts_cpp::CoremlOptions;

double mib(uint64_t bytes) { return bytes / 1048576.0; }

std::string host_chip() {
    char brand[128] = {};
    size_t size = sizeof(brand);
    return sysctlbyname("machdep.cpu.brand_string", brand, &size, nullptr, 0) == 0 ? std::string(brand) : "unknown";
}

Json status_of(const Engine & e) { return Json::parse(e.coreml_status()); }

const Json & stage_status(const Json & status, const std::string & stage) {
    for (const Json & s : status.at("stages")) if (s.at("stage") == stage) return s;
    throw std::runtime_error("no status for stage " + stage + ": " + status.dump());
}

// The overlapped memory schedule: ggml synthesis runs in a loop on this thread
// while both sidecars are generated, compiled, loaded and warmed into
// `cache_dir` in the background, until three calls ran both stages on Core ML.
// One sampler reads the footprint every 2 ms and the slots' states every
// 10 ms; another reads ANECompilerService every 20 ms. The runtime runs one
// Core ML job at a time, so each sample belongs to at most one stage's prepare
// or load. Held to the templates' constants:
//   prepare (per stage)    the peak over the footprint just before that
//                          stage's generation began, against
//                          generation_peak_bytes + 8 MiB + compile_peak_bytes;
//   load+warm (per stage)  the peak over the footprint just before its load
//                          began, against load_peak_bytes for its units;
//   steady                 both ready, Core ML calls running: the peak over the
//                          footprint just before the first load, against the
//                          two resident parts;
//   whole run              the peak over the baseline, against the envelope;
//   compiler               ANECompilerService's growth while a stage on the
//                          Neural Engine loads and warms, against that
//                          template's ane_compile_reserve_bytes.
struct stage_mem {
    uint64_t prepare = 0, load_warm = 0;
    bool prepared = false, loaded = false;
};

int stage_phase(const std::string & state) {
    if (state == "digesting" || state == "generating" || state == "compiling") return 1;
    if (state == "loading" || state == "warming") return 2;
    if (state == "ready") return 3;
    return 0;
}

// The GGUF is replaced, or removed, after the ggml weights were loaded and
// before the sidecars are made: FlowLM is loaded through a symlink, which is
// then pointed at a clone of the file (same bytes, another inode), and Mimi
// through a clone that is then deleted. Generation must not follow either
// path: both stages fail with source_changed, nothing is generated, and the
// construction does not throw.
void source_changed_gate(const EngineOptions & bundle) {
    std::puts("[source] replaced and removed GGUFs");
    const fs::path dir = fs::temp_directory_path() / ("pocket-source-changed-" + test_process_tag());
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir / "cache");
    const fs::path link = dir / "flow-lm.gguf", clone = dir / "flow-lm-clone.gguf", mimi_copy = dir / "mimi.gguf";
    fs::create_symlink(fs::absolute(bundle.flow_lm_path), link);
    require(::clonefile(fs::absolute(bundle.mimi_path).c_str(), mimi_copy.c_str(), 0) == 0, "clone of mimi.gguf");
    EngineOptions opts = bundle;
    opts.flow_lm_path = link.string();
    opts.mimi_path = mimi_copy.string();
    opts.coreml.cache_dir = (dir / "cache").string();
    opts.coreml.wait = CoremlOptions::Wait::construct;
    auto co = pd::coreml_engine_options(opts);
    pd::FlowLM lm(opts.flow_lm_path, 2048, 4, nullptr,
                  [&](int fd, const std::string & p) { co.step_source = pd::coreml_loaded_source(fd, p); });
    pd::Mimi mimi(opts.mimi_path, 4, nullptr,
                  [&](int fd, const std::string & p) { co.decoder_source = pd::coreml_loaded_source(fd, p); });
    require(co.step_source && co.decoder_source, "both loads record the file they read");
    require(::clonefile(fs::absolute(bundle.flow_lm_path).c_str(), clone.c_str(), 0) == 0, "clone of flow-lm.gguf");
    fs::remove(link);
    fs::create_symlink(clone, link);
    fs::remove(mimi_copy);
    std::unique_ptr<pd::CoremlStages> stages;
    try {
        stages = pd::CoremlStages::create(co, lm.config(), lm.source_hash(), 64, false);
    } catch (const std::exception & e) {
        require(false, std::string("construction after the replacement throws: ") + e.what());
    }
    require(stages != nullptr, "stages are made");
    const Json st = Json::parse(stages->status_json());
    std::printf("[source] step %s, mimi %s\n", stage_status(st, "step").dump().substr(0, 160).c_str(),
                stage_status(st, "mimi").dump().substr(0, 160).c_str());
    require(!stages->step_ready() && !stages->decoder_ready() &&
                stage_status(st, "step").dump().find("source_changed") != std::string::npos &&
                stage_status(st, "mimi").dump().find("source_changed") != std::string::npos,
            "a replaced or removed GGUF is not generated from: " + st.dump());
    require(!fs::exists(dir / "cache" / "qvac-coreml-v1" / "artifacts") ||
                fs::is_empty(dir / "cache" / "qvac-coreml-v1" / "artifacts"),
            "nothing is published");
    stages.reset();
    fs::remove_all(dir, ec);
}

void memory_schedule(const EngineOptions & bundle, const std::string & cache_dir) {
    std::printf("[memory] schedule (cache %s)\n", cache_dir.c_str());
    EngineOptions opts = bundle;
    opts.coreml.cache_dir = cache_dir;
    auto co = pd::coreml_engine_options(opts);
    const auto step_route = pd::coreml_plan_route(pd::CoremlStage::step, co);
    const auto decoder_route = pd::coreml_plan_route(pd::CoremlStage::decoder, co);
    require(step_route.route == "generate" && decoder_route.route == "generate",
            "memory schedule: both routes generate (" + step_route.route + " " + step_route.reason + "; " +
                decoder_route.route + " " + decoder_route.reason + ")");
    // The Engine's two stages on their own: the ggml FlowLM and Mimi are
    // loaded and their arenas warmed before the baseline, then the sidecars
    // are generated, compiled, loaded and warmed in the background (prewarm
    // predicted requests both at once) while ggml keeps working.
    pd::FlowLM lm(opts.flow_lm_path, 2048, 4, nullptr,
                  [&](int fd, const std::string & p) { co.step_source = pd::coreml_loaded_source(fd, p); });
    pd::Mimi mimi(opts.mimi_path, 4, nullptr,
                  [&](int fd, const std::string & p) { co.decoder_source = pd::coreml_loaded_source(fd, p); });
    const auto voice = lm.read_voice(opts.voice_path);
    const auto tokens = pd::Frontend(opts.frontend_path).encode("The quick brown fox jumps over the lazy dog.");
    constexpr int FRAMES = 16;
    std::mt19937 rng(5);
    std::normal_distribution<float> normal(0.0f, 0.5f);
    std::vector<float> noise(size_t(lm.config().latent_dim));
    auto ggml_unit = [&] {
        lm.restore_voice(voice);
        lm.prefill(lm.text_embeddings(tokens));
        std::vector<float> previous, latents;
        for (int f = 0; f < FRAMES; ++f) {
            for (auto & x : noise) x = normal(rng);
            const auto condition = lm.advance(previous);
            previous = lm.sample(condition.hidden, noise, 1);
            const auto d = lm.denormalize(previous);
            latents.insert(latents.end(), d.begin(), d.end());
        }
        mimi.reset_decoder();
        const size_t batch = size_t(mimi.max_decode_frames()) * pd::Mimi::latent_dim();
        for (size_t at = 0; at < latents.size(); at += batch)
            mimi.decode(std::vector<float>(latents.begin() + at, latents.begin() + std::min(latents.size(), at + batch)));
    };
    ggml_unit();
    ggml_unit();
    std::atomic<bool> stop{false};
    std::atomic<int> active{-1};       // 0 step, 1 decoder, -1 none
    std::atomic<int> active_phase{0};  // stage_phase of the active stage
    std::atomic<bool> both_ready{false};
    std::atomic<const pd::CoremlStages *> watched{nullptr};
    struct sample { uint64_t footprint; int stage, phase; bool ready; };
    std::vector<sample> samples;
    samples.reserve(1 << 17);
    std::vector<qvac_coreml_memory::AneSample> ane_samples;
    const uint64_t base = qvac_coreml_memory::phys_footprint();
    std::thread sampler([&] {
        for (int tick = 0; !stop.load(); ++tick) {
            const pd::CoremlStages * st_ = watched.load();
            if (st_ != nullptr && tick % 5 == 0) {
                const Json st = Json::parse(st_->status_json());
                int stage = -1, phase = 0, ready = 0;
                for (const Json & s : st.at("stages")) {
                    const int p = stage_phase(s.at("state").get<std::string>());
                    if (p == 3) ++ready;
                    if (p == 1 || p == 2) {
                        stage = s.at("stage") == "step" ? 0 : 1;
                        phase = p;
                    }
                }
                active.store(stage);
                active_phase.store(phase);
                both_ready.store(ready == 2);
            }
            samples.push_back({qvac_coreml_memory::phys_footprint(), active.load(), active_phase.load(), both_ready.load()});
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    });
    std::thread ane_sampler([&] {
        while (!stop.load()) {
            // The decoder is the stage the Neural Engine compiles.
            ane_samples.push_back({qvac_coreml_memory::ane_compiler_rss(), active.load() == 1 && active_phase.load() == 2});
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    });
    const auto t0 = std::chrono::steady_clock::now();
    int ggml_calls = 0, coreml_calls = 0;
    {
        auto stages = pd::CoremlStages::create(co, lm.config(), lm.source_hash(), voice.frames + 2, false);
        watched.store(stages.get());
        const pd::CacheRows voice_rows{0, voice.frames, voice.keys, voice.values};
        while (std::chrono::steady_clock::now() - t0 < std::chrono::minutes(5)) {
            if (!stages->step_ready() || !stages->decoder_ready()) {
                ggml_unit();
                ++ggml_calls;
                continue;
            }
            // One unit on both sidecars: the text rows, FRAMES step
            // predictions, then the frames through the decoder.
            auto step = stages->acquire_step(FRAMES, nullptr);
            auto decoder = stages->acquire_decoder(FRAMES, nullptr);
            require(step.step && decoder.model, "memory schedule: ready sidecars acquired");
            lm.restore_voice(voice);
            lm.prefill(lm.text_embeddings(tokens));
            int position = lm.position();
            std::vector<float> previous = lm.bos_latent(), hidden, next, latents;
            {
                auto lock = step.lock();
                if (!step.step->primed()) step.step->prime(voice_rows);
                step.step->write_rows(lm.cache_rows(voice.frames, position - voice.frames));
                for (int f = 0; f < FRAMES; ++f, ++position) {
                    for (auto & x : noise) x = normal(rng);
                    step.step->step(position, previous, noise, hidden, next);
                    previous = next;
                    const auto d = lm.denormalize(next);
                    latents.insert(latents.end(), d.begin(), d.end());
                }
            }
            pd::CoremlDecoder stream(decoder.model, decoder.label);
            {
                auto lock = decoder.lock();
                stream.decode(latents);
            }
            if (++coreml_calls >= 3) break;
        }
        std::printf("  status: %s\n", stages->status_json().c_str());
        watched.store(nullptr);
        stop.store(true);
        sampler.join();
        ane_sampler.join();
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    // Attribution: each stage's prepare and load+warm over the footprint just
    // before that phase began; the steady state over the footprint just before
    // the first load.
    stage_mem mem[2];
    uint64_t prev = base, start = base, steady = 0, total = 0;
    std::optional<uint64_t> first_load;
    int last_stage = -1, last_phase = 0;
    for (const sample & s : samples) {
        if (s.stage != last_stage || s.phase != last_phase) start = prev;
        if (s.stage >= 0 && s.phase == 2 && !first_load) first_load = start;
        const uint64_t inc = s.footprint > start ? s.footprint - start : 0;
        if (s.stage >= 0 && s.phase == 1) { mem[s.stage].prepared = true; mem[s.stage].prepare = std::max(mem[s.stage].prepare, inc); }
        if (s.stage >= 0 && s.phase == 2) { mem[s.stage].loaded = true; mem[s.stage].load_warm = std::max(mem[s.stage].load_warm, inc); }
        if (s.ready && s.stage < 0) {
            const uint64_t from = first_load.value_or(base);
            steady = std::max(steady, s.footprint > from ? s.footprint - from : 0);
        }
        total = std::max(total, s.footprint > base ? s.footprint - base : 0);
        prev = s.footprint;
        last_stage = s.stage;
        last_phase = s.phase;
    }
    const auto ane = qvac_coreml_memory::ane_compiler_growth(ane_samples);
    const auto baseline = qvac_coreml_memory::ane_compiler_baseline(ane);
    const uint64_t envelope = pd::coreml_envelope_bytes(step_route, decoder_route);
    const pd::CoremlRoute * routes[2] = {&step_route, &decoder_route};
    const char * names[2] = {"step", "mimi"};
    std::vector<std::string> failures;
    auto over = [&](const std::string & what, uint64_t got, uint64_t charged) {
        if (got > charged) failures.push_back(what + ": " + std::to_string(got) + " bytes over the charged " + std::to_string(charged));
    };
    std::printf("  memory (MiB; each phase over the footprint just before it began, steady over the footprint just "
                "before the first load, whole run over the warm ggml baseline %.1f):\n", mib(base));
    for (int i = 0; i < 2; ++i) {
        if (!mem[i].prepared) failures.push_back(std::string("the schedule never sampled the ") + names[i] + " prepare");
        if (!mem[i].loaded) failures.push_back(std::string("the schedule never sampled the ") + names[i] + " load");
        std::printf("    %-5s prepare %7.1f  charged %7.1f   load+warm %7.1f  charged %7.1f  (%s)\n", names[i],
                    mib(mem[i].prepare), mib(routes[i]->prepare_peak_bytes), mib(mem[i].load_warm),
                    mib(routes[i]->load_peak_bytes), routes[i]->compute_units.c_str());
        over(std::string(names[i]) + " prepare", mem[i].prepare, routes[i]->prepare_peak_bytes);
        over(std::string(names[i]) + " load and warm-up", mem[i].load_warm, routes[i]->load_peak_bytes);
    }
    const uint64_t resident = step_route.resident_bytes + decoder_route.resident_bytes;
    std::printf("    steady  %7.1f  charged %7.1f (both resident parts)\n", mib(steady), mib(resident));
    std::printf("    whole   %7.1f  charged %7.1f (envelope)\n", mib(total), mib(envelope));
    std::printf("    ANECompilerService growth while the decoder loaded %.1f, reserve %.1f (before the load %.1f MiB, a %s baseline; peak %.1f)\n",
                mib(ane.growth), mib(decoder_route.ane_compile_reserve_bytes), mib(ane.before),
                qvac_coreml_memory::ane_baseline_name(baseline), mib(ane.peak));
    over("steady state (resident)", steady, resident);
    over("whole run (envelope)", total, envelope);
    if (decoder_route.ane_compile_reserve_bytes > 0) over("ANECompilerService growth (reserve)", ane.growth, decoder_route.ane_compile_reserve_bytes);
    std::printf("  memory report: {\"chip\":\"%s\",\"base\":%llu,\"step_prepare\":%llu,\"step_load_warm\":%llu,"
                "\"mimi_prepare\":%llu,\"mimi_load_warm\":%llu,\"steady\":%llu,\"total\":%llu,"
                "\"ane_compiler_before_load\":%llu,\"ane_compiler_growth\":%llu,\"ane_compiler_peak\":%llu,"
                "\"ane_compiler_baseline\":\"%s\",\"samples\":%zu,\"ggml_calls\":%d,\"coreml_calls\":%d,\"seconds\":%.1f}\n",
                host_chip().c_str(), (unsigned long long) base, (unsigned long long) mem[0].prepare,
                (unsigned long long) mem[0].load_warm, (unsigned long long) mem[1].prepare,
                (unsigned long long) mem[1].load_warm, (unsigned long long) steady, (unsigned long long) total,
                (unsigned long long) ane.before, (unsigned long long) ane.growth, (unsigned long long) ane.peak,
                qvac_coreml_memory::ane_baseline_name(baseline), samples.size(), ggml_calls, coreml_calls, secs);
    for (const auto & f : failures) std::printf("  [FAIL] memory schedule: %s\n", f.c_str());
    require(coreml_calls >= 3 && ggml_calls >= 1, "memory schedule: ggml calls while pending, then Core ML calls");
    require(failures.empty(), "memory schedule: a phase exceeded its charge");
}

int generated_fixture(const std::string & bundle, const std::string & cache_dir, const std::string & memory_dir) {
    env_guard units(pd::COREML_UNITS_ENV), strict(pd::COREML_STRICT_ENV), disabled(pd::COREML_DISABLE_ENV);
    unsetenv(pd::COREML_UNITS_ENV);
    unsetenv(pd::COREML_STRICT_ENV);
    unsetenv(pd::COREML_DISABLE_ENV);
    const EngineOptions plain = bundle_options(bundle);
    require(!tts_cpp::detail::coreml_sidecar_exists(pd::coreml_step_path(plain.flow_lm_path)) &&
                !tts_cpp::detail::coreml_sidecar_exists(pd::coreml_decoder_path(plain.mimi_path)),
            "the bundle must have no prebuilt sidecar (the generated route needs none)");
    if (!memory_dir.empty()) {
        memory_schedule(plain, memory_dir);
        std::puts("test-pocket-coreml --generated --memory-schedule: ok");
        return 0;
    }
    EngineOptions opts = plain;
    opts.coreml.cache_dir = cache_dir;
    const auto co = pd::coreml_engine_options(opts);
    const auto step_route = pd::coreml_plan_route(pd::CoremlStage::step, co);
    const auto decoder_route = pd::coreml_plan_route(pd::CoremlStage::decoder, co);
    std::printf("[routes] step %s %s (%s) %s; mimi %s %s (%s) %s\n", step_route.route.c_str(),
                step_route.compute_units.c_str(), step_route.validated ? "validated" : "unvalidated",
                step_route.reason.c_str(), decoder_route.route.c_str(), decoder_route.compute_units.c_str(),
                decoder_route.validated ? "validated" : "unvalidated", decoder_route.reason.c_str());
    require(step_route.route == "generate" && step_route.validated && step_route.compute_units == "cpu_and_gpu",
            "step route generate, validated, cpu_and_gpu");
    require(decoder_route.route == "generate" && decoder_route.validated && decoder_route.compute_units == "cpu_and_ane",
            "decoder route generate, validated, cpu_and_ane");

    const std::string text = "The quick brown fox jumps over the lazy dog.";
    tts_cpp::pocket::FitOptions fit;
    fit.engine = opts;
    fit.text = text;
    const auto fit_before = tts_cpp::pocket::fit_params(fit);
    const uint64_t envelope = pd::coreml_envelope_bytes(step_route, decoder_route);
    std::printf("[fit] %s, Core ML envelope %llu bytes (%.1f MiB)%s\n", fit_before.reason.c_str(),
                (unsigned long long) fit_before.coreml.bytes, mib(fit_before.coreml.bytes),
                fit_before.coreml.validated ? ", validated" : "");
    require(fit_before.status == tts_cpp::FitStatus::Success && fit_before.coreml.route == "generate" &&
                fit_before.coreml.bytes == envelope && envelope > 0 && fit_before.coreml.validated &&
                fit_before.coreml.disabled_reason.empty(),
            "fit charges the generated envelope: " + fit_before.report);

    // Core ML off: the reference every pending call must equal.
    EngineOptions off = opts;
    off.coreml.use = CoremlOptions::Use::off;
    Engine ggml(off);
    const auto reference = ggml.synthesize(text);
    require(reference.flow_lm_backend == "ggml" && reference.mimi_backend == "ggml" && !reference.coreml_pending,
            "use = off runs ggml");

    // Readiness pinned to "not yet": nothing is requested before the first
    // call (prewarm none) and a call never waits for an unhinted load (wait
    // none), so the first single-chunk call runs on ggml and matches Core ML
    // off.
    const bool cold = !fs::exists(fs::path(cache_dir) / "qvac-coreml-v1" / "artifacts");
    if (cold) {
        EngineOptions pending = opts;
        pending.coreml.prewarm = CoremlOptions::Prewarm::none;
        pending.coreml.wait = CoremlOptions::Wait::none;
        Engine engine(pending);
        const Json idle = status_of(engine);
        require(stage_status(idle, "step").at("state") == "idle" && stage_status(idle, "mimi").at("state") == "idle",
                "prewarm none: both stages idle after construction: " + idle.dump());
        require(!engine.coreml_wait(0), "nothing ready yet");
        const auto first = engine.synthesize(text);
        std::printf("[pending] first call: FlowLM %s, Mimi %s, pending %d\n", first.flow_lm_backend.c_str(),
                    first.mimi_backend.c_str(), int(first.coreml_pending));
        require(first.flow_lm_backend == "ggml" && first.mimi_backend == "ggml" && first.coreml_pending,
                "the first call runs on ggml while the sidecars are pending");
        // The CPU ggml path differs between processes and instances only in
        // float rounding (about 2e-7), so "the same as Core ML off" is a
        // tolerance, far below what one Core ML frame changes.
        require(first.pcm.size() == reference.pcm.size() && max_error(first.pcm, reference.pcm) < 1e-4,
                "a pending call matches Core ML off");
        require(engine.coreml_wait(-1), "the first call's request makes both sidecars ready: " + engine.coreml_status());
        const Json ready = status_of(engine);
        require(stage_status(ready, "step").at("route") == "generated" &&
                    stage_status(ready, "mimi").at("route") == "generated",
                "a cold cache generates both: " + ready.dump());
        const auto second = engine.synthesize(text);
        require(second.flow_lm_backend == "coreml-gpu" && second.mimi_backend == "coreml-ane" && !second.coreml_pending,
                "once ready, a call runs both stages on Core ML: " + second.flow_lm_backend + " / " + second.mimi_backend);
    } else {
        std::puts("[pending] skipped: the cache already holds the artifacts (use a fresh --cache-dir)");
    }

    // Readiness pinned to "ready" at construction (wait construct): every
    // call runs on Core ML; a second Engine adopts the cached artifacts.
    EngineOptions waited = opts;
    waited.coreml.wait = CoremlOptions::Wait::construct;
    Engine engine(waited);
    require(engine.flow_lm_on_coreml() && engine.mimi_on_coreml(), "wait construct: both ready after construction");
    const Json st = status_of(engine);
    const std::string step_path = stage_status(st, "step").at("path");
    const std::string decoder_path = stage_status(st, "mimi").at("path");
    require(stage_status(st, "step").at("route") == "cached" && stage_status(st, "mimi").at("route") == "cached",
            "a second Engine adopts the cached artifacts: " + st.dump());
    const auto batch = engine.synthesize(text);
    std::printf("[engine] %s / %s: %d frames; ggml %d frames\n", batch.flow_lm_backend.c_str(), batch.mimi_backend.c_str(),
                batch.generated_frames, reference.generated_frames);
    require(batch.flow_lm_backend == "coreml-gpu" && batch.mimi_backend == "coreml-ane", "Core ML labels");
    require(std::abs(batch.generated_frames - reference.generated_frames) <= 4, "Core ML changes the speech length");
    std::vector<float> streamed;
    engine.synthesize_stream(text, [&](const float * pcm, size_t n, int) { streamed.insert(streamed.end(), pcm, pcm + n); return true; });
    require(max_error(streamed, batch.pcm) < 1e-4, "stream and batch differ on the generated sidecars");

    // Parity of the generated artifacts against ggml (the prebuilt sidecars'
    // gates, on the generated bundles).
    check_stage_parity(plain, step_path);
    check_decoder_parity(plain, decoder_path);
    source_changed_gate(plain);

    // Fit is the same with a populated cache.
    const auto fit_after = tts_cpp::pocket::fit_params(fit);
    require(fit_after.coreml.route == fit_before.coreml.route && fit_after.coreml.bytes == fit_before.coreml.bytes &&
                fit_after.coreml.validated == fit_before.coreml.validated &&
                fit_after.device.total_bytes == fit_before.device.total_bytes && fit_after.host_bytes == fit_before.host_bytes,
            "fit is unchanged by a populated cache");

    // Strict: use = require serves a call on the cached sidecars.
    EngineOptions strict_opts = opts;
    strict_opts.coreml.use = CoremlOptions::Use::require;
    Engine strict_engine(strict_opts);
    const auto strict_result = strict_engine.synthesize(text);
    require(strict_result.flow_lm_backend == "coreml-gpu" && strict_result.mimi_backend == "coreml-ane",
            "strict waits for and runs the sidecars");
    // Memory-degrade: a budget only the ggml projection fits drops Core ML.
    EngineOptions degrade = opts;
    degrade.coreml.wait = CoremlOptions::Wait::construct;
    {
        tts_cpp::pocket::FitOptions f = fit;
        f.engine.coreml.memory_budget_bytes = int64_t(1) << 50;
        const auto probe = tts_cpp::pocket::fit_params(f);
        const uint64_t ggml_required = probe.device.total_bytes + (probe.device_shares_host_memory ? probe.host_bytes : 0) +
                                       f.margin_bytes;
        // A budget between the ggml projection and the Core ML-inclusive one.
        f.engine.coreml.memory_budget_bytes = int64_t(ggml_required + probe.coreml.bytes / 2);
        const auto degraded = tts_cpp::pocket::fit_params(f);
        std::printf("[degrade] budget %.1f MiB: %s, Core ML %s\n", mib(uint64_t(*f.engine.coreml.memory_budget_bytes)),
                    degraded.reason.c_str(), degraded.coreml.disabled_reason.c_str());
        require(degraded.coreml.disabled_reason == "memory" && degraded.fits, "a near budget degrades to ggml in fit");
        degrade.coreml.memory_budget_bytes = f.engine.coreml.memory_budget_bytes;
        degrade.fit_text = text;
    }
    Engine degraded_engine(degrade);
    const Json dst = status_of(degraded_engine);
    require(stage_status(dst, "step").at("state") == "disabled" && stage_status(dst, "step").at("route") == "none" &&
                stage_status(dst, "mimi").at("state") == "disabled" && stage_status(dst, "mimi").at("route") == "none",
            "the Engine resolves use = off under memory-degrade: " + dst.dump());
    require(degraded_engine.synthesize(text).flow_lm_backend == "ggml", "a degraded Engine runs ggml");
    {
        // A fit assessed with a larger headroom than the default: the budget
        // holds ggml + Core ML with the default margin, but only ggml with
        // this one. The Engine, given the same text and margin, must reach
        // that fit's verdict (memory-degrade) and run ggml.
        tts_cpp::pocket::FitOptions wide = fit;
        wide.engine.coreml.memory_budget_bytes = int64_t(1) << 50;
        const auto probe = tts_cpp::pocket::fit_params(wide);
        const uint64_t ggml_required =
            probe.device.total_bytes + (probe.device_shares_host_memory ? probe.host_bytes : 0) + fit.margin_bytes;
        wide.margin_bytes = fit.margin_bytes + probe.coreml.bytes;
        wide.engine.coreml.memory_budget_bytes = int64_t(ggml_required + probe.coreml.bytes + (64ull << 20));
        const auto assessed = tts_cpp::pocket::fit_params(wide);
        tts_cpp::pocket::FitOptions as_default = wide;
        as_default.margin_bytes = fit.margin_bytes;
        const auto by_default = tts_cpp::pocket::fit_params(as_default);
        require(assessed.coreml.disabled_reason == "memory" && by_default.coreml.disabled_reason.empty(),
                "a budget only the larger margin degrades: '" + assessed.coreml.disabled_reason + "' with it, '" +
                    by_default.coreml.disabled_reason + "' with the default");
        EngineOptions low = opts;
        low.coreml.wait = CoremlOptions::Wait::construct;
        low.coreml.memory_budget_bytes = wide.engine.coreml.memory_budget_bytes;
        low.fit_text = text;
        low.fit_margin_bytes = wide.margin_bytes;
        Engine engine_low(low);
        const Json lst = status_of(engine_low);
        const auto low_result = engine_low.synthesize(text);
        std::printf("[degrade] margin %.1f MiB: Engine FlowLM %s, Mimi %s\n", mib(wide.margin_bytes),
                    low_result.flow_lm_backend.c_str(), low_result.mimi_backend.c_str());
        require(stage_status(lst, "step").at("state") == "disabled" &&
                    stage_status(lst, "mimi").at("state") == "disabled" && low_result.flow_lm_backend == "ggml" &&
                    low_result.mimi_backend == "ggml",
                "Engine with the fit's nondefault margin: memory-degrade as that fit (" + low_result.flow_lm_backend +
                    " / " + low_result.mimi_backend + "), " + lst.dump());
    }
    std::puts("test-pocket-coreml --generated: ok");
    return 0;
}
#endif

// Budgets below the ggml projection, between it and the Core ML-inclusive
// one, and above both, through the public fit (each budget source, and both
// with the smaller winning) and, with `cli`, through pocket-cli --fit. One
// ceiling holds every verdict: below refuses, between degrades, above keeps
// Core ML (when the build generates).
int fit_budgets(const fs::path & bundle, const std::string & cli) {
    EngineOptions e = bundle_options(bundle);
    const fs::path cache = fs::temp_directory_path() / ("pocket-fit-budgets-" + test_process_tag());
    e.coreml.cache_dir = cache.string();
    tts_cpp::pocket::FitOptions f;
    f.engine = e;
    f.text = "The quick brown fox jumps over the lazy dog.";
    const auto probe = tts_cpp::pocket::fit_params(f);
    require(probe.status != tts_cpp::FitStatus::Error, "fit of the bundle: " + probe.report);
    if (!probe.device_shares_host_memory) {
        std::puts("SKIP: the selected device does not share host memory");
        return SKIP;
    }
    const uint64_t MIB = 1ull << 20;
    const uint64_t required = probe.device.total_bytes + probe.host_bytes + f.margin_bytes;
    const uint64_t envelope = probe.coreml.route == "generate" ? probe.coreml.bytes : 0;
    std::printf("[budgets] ggml %.1f MiB, Core ML envelope %.1f MiB (route %s), free %.1f MiB\n",
                double(required) / MIB, double(envelope) / MIB, probe.coreml.route.c_str(),
                double(probe.device_free_bytes) / MIB);
    const uint64_t below = required - 16 * MIB, between = required + envelope / 2,
                   above = required + envelope + 64 * MIB;
    if (probe.device_free_bytes < above + 256 * MIB) {
        std::puts("SKIP: not enough free memory for the budget above both projections");
        return SKIP;
    }
    struct Case {
        const char * name;
        uint64_t budget;
        bool fits;
        bool degraded;
    };
    std::vector<Case> cases = {{"below", below, false, false}, {"above", above, true, false}};
    if (envelope) cases.push_back({"between", between, true, true});
    for (const Case & c : cases) {
        for (int source = 0; source < 4; ++source) {
            tts_cpp::pocket::FitOptions g = f;
            const char * label = "";
            if (source == 0) { g.memory_budget_bytes = c.budget; label = "fit budget"; }
            if (source == 1) { g.engine.coreml.memory_budget_bytes = int64_t(c.budget); label = "Core ML budget"; }
            if (source == 2) {
                g.memory_budget_bytes = c.budget;
                g.engine.coreml.memory_budget_bytes = int64_t(c.budget);
                label = "both";
            }
            if (source == 3) {
                g.memory_budget_bytes = c.budget;
                g.engine.coreml.memory_budget_bytes = int64_t(c.budget + envelope + 1024 * MIB);
                label = "both, fit's smaller";
            }
            const auto r = tts_cpp::pocket::fit_params(g);
            const bool degraded = r.coreml.disabled_reason == "memory";
            std::printf("[budgets] %s (%s): %s%s\n", c.name, label, r.reason.c_str(), degraded ? ", memory-degrade" : "");
            require(r.fits == c.fits && degraded == c.degraded && r.device_free_bytes <= c.budget,
                    std::string("budget ") + c.name + " via " + label + ": " + r.report);
        }
    }
#ifndef _WIN32
    if (!cli.empty()) {
        const fs::path report = cache.string() + "-report.json";
        for (const Case & c : cases) {
            const uint64_t budget_mib = c.name == std::string("below") ? required / MIB - 16
                                        : c.name == std::string("between") ? (required + envelope / 2) / MIB
                                                                           : (required + envelope) / MIB + 64;
            const std::string cmd = "'" + cli + "' --fit --model-dir '" + bundle.string() + "' --text '" + f.text +
                                    "' --coreml-cache-dir '" + cache.string() + "' --fit-budget-mib " +
                                    std::to_string(budget_mib) + " --report '" + report.string() + "' > /dev/null";
            const int status = std::system(cmd.c_str());
            const int code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
            const nlohmann::json j = nlohmann::json::parse(std::ifstream(report));
            const bool degraded = j.at("coreml").at("disabled_reason") == "memory";
            std::printf("[budgets] pocket-cli %s (%llu MiB): exit %d%s\n", c.name, (unsigned long long) budget_mib, code,
                        degraded ? ", memory-degrade" : "");
            require(code == (c.fits ? 0 : 1) && j.at("fits") == c.fits && degraded == c.degraded,
                    std::string("pocket-cli --fit budget ") + c.name + ": " + j.dump());
        }
        fs::remove(report);
    }
#else
    (void) cli;
#endif
    std::error_code ec;
    fs::remove_all(cache, ec);
    std::puts("test-pocket-coreml --fit-budgets: ok");
    return 0;
}

int engine_fixture() {
#ifndef TTS_CPP_USE_COREML
    std::puts("SKIP: built without Core ML (TTS_CPP_COREML=OFF)");
    return SKIP;
#else
    const char * dir = std::getenv("POCKET_COREML_TEST_BUNDLE");
    if (dir == nullptr || !*dir) {
        std::puts("SKIP: set POCKET_COREML_TEST_BUNDLE to a Pocket bundle with flow-lm-step.mlmodelc");
        return SKIP;
    }
    const auto opts = bundle_options(dir);
    require(tts_cpp::detail::coreml_sidecar_exists(pd::coreml_step_path(opts.flow_lm_path)), "bundle has no step sidecar");
    require(tts_cpp::detail::coreml_sidecar_exists(pd::coreml_decoder_path(opts.mimi_path)), "bundle has no Mimi decoder sidecar");
    check_stage_parity(opts);
    check_decoder_parity(opts);
    check_replay(opts);
    check_engine(opts);
    std::puts("test-pocket-coreml --engine-fixture: ok");
    return 0;
#endif
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc > 1 && std::string(argv[1]) == "--shape-fixture") return shape_fixture();
        if (argc > 1 && std::string(argv[1]) == "--engine-fixture") return engine_fixture();
        if (argc > 2 && std::string(argv[1]) == "--fit-budgets")
            return fit_budgets(argv[2], argc > 4 && std::string(argv[3]) == "--cli" ? argv[4] : "");
        if (argc > 1 && std::string(argv[1]) == "--generated") {
#ifndef TTS_CPP_USE_COREML
            std::puts("SKIP: built without Core ML (TTS_CPP_COREML=OFF)");
            return SKIP;
#else
            std::string bundle, cache, memory;
            for (int i = 2; i < argc; ++i) {
                const std::string a = argv[i];
                if (a == "--cache-dir" && i + 1 < argc) cache = argv[++i];
                else if (a == "--memory-schedule" && i + 1 < argc) memory = argv[++i];
                else if (bundle.empty()) bundle = a;
                else throw std::runtime_error("unknown argument " + a);
            }
            if (bundle.empty() || (cache.empty() && memory.empty()))
                throw std::runtime_error("--generated BUNDLE --cache-dir DIR | --memory-schedule DIR");
            return generated_fixture(bundle, cache, memory);
#endif
        }
        check_policy();
        check_coreml_options();
        check_naming();
        check_rope();
        check_detection_and_fit();
        std::puts("test-pocket-coreml: ok");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "test-pocket-coreml: FAIL: %s\n", e.what());
        return 1;
    }
}
