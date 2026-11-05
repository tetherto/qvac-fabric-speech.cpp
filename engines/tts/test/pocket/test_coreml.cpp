// Pocket FlowLM Core ML step sidecar.
//
//   test-pocket-coreml                  policy, naming, RoPE tables, sidecar
//                                       detection, fit refusal, option checks
//   test-pocket-coreml --shape-fixture  a tiny real step reproduces its float32
//                                       reference after the prefix is written
//                                       into its state; every declared shape,
//                                       the schema, the checkpoint hash and
//                                       the state are checked before predicting
//   test-pocket-coreml --engine-fixture the real step against ggml FlowLM frame
//                                       by frame, the ggml replay of frames a
//                                       retired sidecar generated, and the
//                                       Engine's routing, labels and strict mode
//
// The shape fixture reads POCKET_COREML_TEST_SHAPES (the directory
// test/prepare_pocket_coreml_fixture.py writes); the engine fixture reads
// POCKET_COREML_TEST_BUNDLE (a Pocket bundle directory with the exported
// flow-lm-step.mlmodelc next to flow-lm.gguf). Each reports a skip (77)
// without its inputs or without Core ML.

#include "pocket/coreml.h"
#include "pocket/flow_lm.h"
#include "pocket/frontend.h"
#include "coreml_sidecar_path.h"
#include "json.hpp"
#include "npy.h"
#include "../test_env_portable.h"
#include "tts-cpp/pocket/engine.h"
#include "tts-cpp/pocket/fit.h"

#ifdef TTS_CPP_USE_COREML
#include "moss/coreml/runner.h"
#endif

#include <algorithm>
#include <cmath>
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
    require(policy.compute_units == "cpu_and_gpu" && !policy.strict && !policy.disabled, "default policy");

    setenv(pd::COREML_UNITS_ENV, "all", 1);
    setenv(pd::COREML_STRICT_ENV, "0", 1);  // presence means strict
    policy = pd::resolve_coreml_policy();
    require(policy.compute_units == "all" && policy.strict, "inherited policy");
    policy = pd::resolve_coreml_policy("cpu_only", false);
    require(policy.compute_units == "cpu_only" && !policy.strict, "explicit policy overrides the environment");

    setenv(pd::COREML_UNITS_ENV, "gpu_please", 1);
    require(pd::resolve_coreml_policy().compute_units == "cpu_and_gpu", "invalid inherited units keep the default");
    require(throws([] { pd::resolve_coreml_policy("gpu_please"); }), "invalid explicit units throw");

    setenv(pd::COREML_DISABLE_ENV, "1", 1);
    require(pd::resolve_coreml_policy().disabled, "POCKET_COREML_DISABLE");
}

void check_naming() {
    require(pd::coreml_step_path("models/pocket/flow-lm.gguf") == "models/pocket/flow-lm-step.mlmodelc", "bundle name");
    require(pd::coreml_step_path("m/flow-lm-f16.gguf") == "m/flow-lm-step.mlmodelc", "quant tag dropped");
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
    const std::string gguf = (dir.path / "flow-lm.gguf").string();
    require(!pd::coreml_step_enabled(gguf, pd::resolve_coreml_policy()), "no sidecar");
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

std::vector<float> npy_floats(const fs::path & path) {
    const npy_array array = npy_load(path.string());
    const float * data = npy_as_f32(array);
    return std::vector<float>(data, data + array.n_elements());
}

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
    for (const char * units : {"cpu_only", pd::COREML_DEFAULT_UNITS}) {
        pd::CoremlStep step(sidecar("good"), units, cfg, source);
        require(step.context() == context, "context read from the sidecar");
        step.write_rows(rows);
        std::vector<float> hidden, next;
        for (int f = 0; f < frames; ++f) {
            const auto at = [](const std::vector<float> & all, int row, int width) {
                return std::vector<float>(all.begin() + size_t(row) * width, all.begin() + size_t(row + 1) * width);
            };
            step.step(prefix + f, at(latents, f, cfg.latent_dim), at(noises, f, cfg.latent_dim), hidden, next);
            const double eh = max_error(hidden, at(hidden_ref, f, cfg.dim));
            const double en = max_error(next, at(next_ref, f, cfg.latent_dim));
            std::printf("[good %s] frame %d: hidden error %.2e, next latent error %.2e\n", units, f, eh, en);
            require(eh < 3e-2 && en < 3e-2, std::string("tiny step diverges from its reference on ") + units);
        }
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
void check_stage_parity(const EngineOptions & opts) {
    pd::FlowLM lm(opts.flow_lm_path, 2048, 4);
    const auto voice = lm.read_voice(opts.voice_path);
    pd::CoremlStep step(pd::coreml_step_path(opts.flow_lm_path), pd::COREML_DEFAULT_UNITS, lm.config(), lm.source_hash());
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

void check_engine(const EngineOptions & base) {
    env_guard units(pd::COREML_UNITS_ENV), strict(pd::COREML_STRICT_ENV), disabled(pd::COREML_DISABLE_ENV);
    unsetenv(pd::COREML_UNITS_ENV);
    unsetenv(pd::COREML_STRICT_ENV);
    unsetenv(pd::COREML_DISABLE_ENV);
    const std::string text = "The quick brown fox jumps over the lazy dog. It barked twice.";
    Engine coreml(base);
    require(coreml.flow_lm_on_coreml(), "sidecar loaded");
    const auto batch = coreml.synthesize(text);
    std::vector<float> streamed;
    const auto stream = coreml.synthesize_stream(text, [&](const float * pcm, size_t n, int) {
        streamed.insert(streamed.end(), pcm, pcm + n); return true;
    });
    require(batch.flow_lm_backend.rfind("coreml-", 0) == 0 && stream.flow_lm_backend == batch.flow_lm_backend,
            "Core ML label, got " + batch.flow_lm_backend);
    require(max_error(streamed, batch.pcm) < 1e-4, "stream and batch differ on Core ML");

    setenv(pd::COREML_DISABLE_ENV, "1", 1);
    Engine ggml(base);
    unsetenv(pd::COREML_DISABLE_ENV);
    const auto reference = ggml.synthesize(text);
    require(!ggml.flow_lm_on_coreml() && reference.flow_lm_backend == "ggml", "POCKET_COREML_DISABLE keeps ggml");
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
    require(several.flow_lm_backend == batch.flow_lm_backend, "multi-chunk call left Core ML");
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
    check_stage_parity(opts);
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
        check_policy();
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
