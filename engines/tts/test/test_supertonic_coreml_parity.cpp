// Supertonic Core ML vocoder sidecar: parity against the ggml vocoder graph
// through the production dispatch, the load-status and per-call backend
// reports, and the fallbacks (disabled, strict without a sidecar, invalid
// sidecar).
//
// Prebuilt sidecar (no arguments): skips (77) without TTS_CPP_COREML or a
// staged SUPERTONIC_COREML_TEST_MODELS_DIR holding a Supertonic GGUF with a
// compiled `<model>-vocoder.mlmodelc` next to it.
//
// Generated sidecar:
//   test-supertonic-coreml-parity --gguf MODEL.gguf --cache-dir DIR
//       [--min-cosine 0.999] [--require-coreml] [--memory-schedule DIR2]
// generates the vocoder sidecar from MODEL.gguf into DIR (no sidecar may sit
// next to it) and holds it to the ggml vocoder of the same GGUF at
// --min-cosine on a padded, an exact-window and a stitched multi-window
// latent; then a second owner adopts the artifact (route cached), a GGUF
// replaced after the load is not generated from, an Engine synthesizes on it, fit charges the template's envelope (and degrades to
// ggml near a budget, with the F16 weight deny list projected), and a Max or Ultra brand string keeps a generated
// sidecar off unless use = require. --require-coreml fails (instead of
// skipping, 77) when the route is not generate. --memory-schedule runs the
// overlapped memory schedule into the fresh cache DIR2 and holds each phase
// to the template constant that charges it.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
constexpr int SKIP_EXIT_CODE = 77;
}

#ifndef TTS_CPP_USE_COREML

int main() {
    std::fprintf(stderr, "[supertonic-coreml-parity] built without TTS_CPP_COREML; skipping\n");
    return SKIP_EXIT_CODE;
}

#else

#include "memory_probe.h"
#include "memory_schedule.h"
#include "supertonic_coreml_path.h"
#include "supertonic_coreml_vocoder.h"
#include "supertonic_internal.h"
#include "supertonic_vocoder_coreml.h"
#include "tts-cpp/supertonic/engine.h"
#include "tts-cpp/supertonic/fit.h"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <thread>

#include <sys/clonefile.h>

using namespace tts_cpp::supertonic::detail;
namespace fs = std::filesystem;

namespace {

constexpr double MIN_COSINE = 0.999;
// A brand whose device class the template does not prefer ggml on: the memory
// schedule measures the generated route on every host, Max and Ultra included,
// with the default (automatic) policy.
constexpr const char * UNGATED_BRAND = "Apple M2";
constexpr int    N_THREADS  = 4;
constexpr int    DEGRADE_STEPS = 3;  // not the GGUF's default

int g_failures = 0;

void fail(const std::string & what) {
    std::fprintf(stderr, "[supertonic-coreml-parity] FAIL: %s\n", what.c_str());
    ++g_failures;
}

void expect(bool ok, const std::string & what) {
    if (!ok) fail(what);
}

struct scoped_env {
    const char * name;
    scoped_env(const char * n, const char * value) : name(n) { setenv(name, value, 1); }
    ~scoped_env() { unsetenv(name); }
};

std::string find_supertonic_gguf(const std::string & dir) {
    const char * candidates[] = {"supertonic2.gguf",      "supertonic3.gguf",
                                 "supertonic.gguf",       "supertonic2-f32.gguf",
                                 "supertonic2-f16.gguf",  "supertonic2-q8_0.gguf",
                                 "supertonic3-f32.gguf",  "supertonic3-q8_0.gguf"};
    for (const char * name : candidates) {
        const std::string path = dir + "/" + name;
        if (fs::exists(path)) return path;
    }
    return {};
}

std::vector<float> make_latent(int channels, int latent_len, uint32_t seed) {
    std::vector<float> latent((size_t) channels * latent_len);
    uint32_t state = seed;
    for (float & v : latent) {
        state = state * 1664525u + 1013904223u;
        // Uniform in [-2, 2): wide enough to exercise the stack, bounded so
        // the fp16 sidecar stays in a well-conditioned range.
        v = ((float) (state >> 8) / 8388608.0f) * 2.0f - 2.0f;
    }
    return latent;
}

double cosine(const std::vector<float> & a, const std::vector<float> & b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += (double) a[i] * (double) b[i];
        na  += (double) a[i] * (double) a[i];
        nb  += (double) b[i] * (double) b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
}

struct owned_model {
    supertonic_model model;
    bool loaded = false;
    ~owned_model() { if (loaded) free_supertonic_model(model); }
};

bool load(const std::string & gguf, owned_model & owned) {
    try {
        owned.loaded = load_supertonic_gguf(gguf, owned.model, /*n_gpu_layers=*/0);
        if (owned.loaded) supertonic_set_n_threads(owned.model, N_THREADS);
    } catch (const std::exception & e) {
        fail("load_supertonic_gguf(" + gguf + "): " + e.what());
        return false;
    }
    if (!owned.loaded) fail("load_supertonic_gguf(" + gguf + ") returned false");
    return owned.loaded;
}

void check_parity(const supertonic_model & ggml_model, const supertonic_model & coreml_model,
                  int latent_len, uint32_t seed) {
    const std::string tag = "latent_len=" + std::to_string(latent_len);
    const std::vector<float> latent =
        make_latent(ggml_model.hparams.latent_channels, latent_len, seed);

    std::vector<float> wav_ggml, wav_coreml;
    std::string error, backend;
    if (!supertonic_vocoder_forward_ggml(ggml_model, latent.data(), latent_len, wav_ggml,
                                         &error, &backend)) {
        fail(tag + ": ggml vocoder failed: " + error);
        return;
    }
    expect(backend == "ggml", tag + ": ggml reference reports backend " + backend);

    scoped_env strict("SUPERTONIC_COREML_STRICT", "1");
    if (!supertonic_vocoder_forward_ggml(coreml_model, latent.data(), latent_len, wav_coreml,
                                         &error, &backend)) {
        fail(tag + ": Core ML vocoder failed: " + error);
        return;
    }
    expect(backend.rfind("coreml", 0) == 0, tag + ": Core ML run reports backend " + backend);
    expect(wav_coreml.size() == wav_ggml.size(),
           tag + ": size mismatch (" + std::to_string(wav_coreml.size()) + " vs " +
           std::to_string(wav_ggml.size()) + ")");
    if (wav_coreml.size() != wav_ggml.size()) return;

    const double cos = cosine(wav_ggml, wav_coreml);
    std::fprintf(stderr, "[supertonic-coreml-parity] %s cosine=%.6f\n", tag.c_str(), cos);
    expect(cos >= MIN_COSINE, tag + ": cosine " + std::to_string(cos) + " below " +
           std::to_string(MIN_COSINE));
}

void check_engine_report(const std::string & gguf) {
    using tts_cpp::supertonic::Engine;
    using tts_cpp::supertonic::EngineOptions;
    using tts_cpp::supertonic::SynthesisResult;
    try {
        EngineOptions opts;
        opts.model_gguf_path = gguf;
        opts.n_threads = N_THREADS;
        // The prebuilt sidecar loads in the background; wait for it.
        opts.coreml.wait = tts_cpp::CoremlOptions::Wait::construct;
        Engine engine(opts);
        expect(engine.vocoder_on_coreml(), "engine: vocoder_on_coreml() is false");
        SynthesisResult result = engine.synthesize("Core ML parity check.");
        expect(result.vocoder_synthesis_backend.rfind("coreml", 0) == 0,
               "engine: SynthesisResult::vocoder_synthesis_backend is " + result.vocoder_synthesis_backend);
        expect(!result.pcm.empty(), "engine: empty PCM");

        // Streaming with every chunk on the sidecar keeps the sidecar label
        // rather than degrading to "mixed".
        EngineOptions stream_opts = opts;
        stream_opts.stream_chunk_tokens = 40;
        Engine stream_engine(stream_opts);
        int chunks = 0;
        SynthesisResult streamed = stream_engine.synthesize(
            "The quick brown fox jumps over the lazy dog. The curious cat watches "
            "from the warm windowsill. A lone bell rings across the quiet valley.",
            [&](const float *, std::size_t, int, bool) { ++chunks; });
        expect(chunks >= 2, "engine: streaming produced " + std::to_string(chunks) +
                            " chunk(s); the multi-chunk label path never ran");
        expect(streamed.vocoder_synthesis_backend.rfind("coreml", 0) == 0,
               "engine: streamed vocoder_synthesis_backend is " +
               streamed.vocoder_synthesis_backend);
    } catch (const std::exception & e) {
        fail(std::string("engine synthesis: ") + e.what());
    }
}

// A CPU-backed engine (n_gpu_layers = 0) with a sidecar attached must warm it:
// the ctor pre-warm's vocoder pass shows up as a "vocoder,coreml" profile row
// before any operator-visible synthesize().
void check_cpu_warm_up_reaches_sidecar(const std::string & gguf) {
    using tts_cpp::supertonic::Engine;
    using tts_cpp::supertonic::EngineOptions;
    std::error_code ec;
    const fs::path csv = fs::temp_directory_path(ec) / "supertonic-coreml-warmup.csv";
    fs::remove(csv, ec);
    supertonic_profile_csv_set_path(csv.string().c_str());
    try {
        EngineOptions opts;
        opts.model_gguf_path = gguf;
        opts.n_threads = N_THREADS;
        opts.prewarm_text = "Warm up the vocoder sidecar.";
        opts.coreml.wait = tts_cpp::CoremlOptions::Wait::construct;
        Engine engine(opts);
        expect(engine.vocoder_on_coreml(), "warm-up: sidecar not attached");
    } catch (const std::exception & e) {
        fail(std::string("warm-up engine: ") + e.what());
    }
    supertonic_profile_csv_set_path(nullptr);
    std::ifstream rows(csv);
    std::string line;
    bool warmed = false;
    while (std::getline(rows, line)) {
        if (line.rfind("vocoder,coreml,", 0) == 0) {
            warmed = true;
            break;
        }
    }
    expect(warmed, "warm-up: the ctor pre-warm left the Core ML vocoder cold "
                   "(no vocoder,coreml profile row)");
    fs::remove(csv, ec);
}

void check_disabled_fallback(const std::string & gguf, int latent_len) {
    scoped_env disable("SUPERTONIC_COREML_DISABLE", "1");
    owned_model owned;
    if (!load(gguf, owned)) return;
    expect(!owned.model.vocoder_on_coreml,
           "SUPERTONIC_COREML_DISABLE: sidecar still attached");
    const std::vector<float> latent =
        make_latent(owned.model.hparams.latent_channels, latent_len, 7u);
    std::vector<float> wav;
    std::string error, backend;
    expect(supertonic_vocoder_forward_ggml(owned.model, latent.data(), latent_len, wav,
                                           &error, &backend),
           "SUPERTONIC_COREML_DISABLE: ggml vocoder failed: " + error);
    expect(backend == "ggml", "SUPERTONIC_COREML_DISABLE: backend is " + backend);

    // Strict mode with no sidecar attached must fail rather than silently
    // measure ggml.
    scoped_env strict("SUPERTONIC_COREML_STRICT", "1");
    std::vector<float> refused;
    expect(!supertonic_vocoder_forward_ggml(owned.model, latent.data(), latent_len, refused,
                                            &error, &backend),
           "strict without a sidecar: forward did not fail");
}

// A sidecar path that exists but is not a Core ML model must leave the load
// on ggml and the forward on the fallback.
void check_invalid_sidecar(const std::string & gguf, int latent_len) {
    std::error_code ec;
    const fs::path dir = fs::temp_directory_path(ec) / "supertonic-coreml-invalid";
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    if (ec) {
        fail("invalid sidecar: cannot stage " + dir.string());
        return;
    }
    const fs::path linked = dir / "supertonic-invalid.gguf";
    fs::create_symlink(fs::absolute(gguf), linked, ec);
    if (ec) {
        fail("invalid sidecar: cannot link the GGUF: " + ec.message());
        return;
    }
    fs::create_directories(coreml_vocoder_sidecar_path(linked.string()), ec);

    owned_model owned;
    if (!load(linked.string(), owned)) return;
    expect(!owned.model.vocoder_on_coreml, "invalid sidecar: load-status reports attached");
    const std::vector<float> latent =
        make_latent(owned.model.hparams.latent_channels, latent_len, 5u);
    std::vector<float> wav;
    std::string error, backend;
    expect(supertonic_vocoder_forward_ggml(owned.model, latent.data(), latent_len, wav,
                                           &error, &backend),
           "invalid sidecar: ggml fallback failed: " + error);
    expect(backend == "ggml", "invalid sidecar: backend is " + backend);
    fs::remove_all(dir, ec);
}

// ---- generated sidecar ----------------------------------------------------------

std::string status_field(const std::string & status, const std::string & field) {
    const std::string key = "\"" + field + "\":\"";
    const size_t at = status.find(key);
    if (at == std::string::npos) return {};
    const size_t begin = at + key.size();
    return status.substr(begin, status.find('"', begin) - begin);
}

double mib(uint64_t bytes) { return bytes / 1048576.0; }

// A model whose vocoder runs on the shared Core ML runtime, like the Engine's.
struct managed_model {
    owned_model owned;
    std::shared_ptr<supertonic_vocoder_coreml> stage;
};

bool load_managed(const std::string & gguf, const tts_cpp::CoremlOptions & o, managed_model & m) {
    m.owned.model.vocoder_coreml_managed = true;
    if (!load(gguf, m.owned)) return false;
    try {
        m.stage = supertonic_vocoder_coreml::create(m.owned.model, gguf, o);
    } catch (const std::exception & e) {
        fail(std::string("create the vocoder stage: ") + e.what());
        return false;
    }
    return m.stage != nullptr;
}

void check_generated_parity(const supertonic_model & ggml_model, const supertonic_model & coreml_model,
                            int latent_len, uint32_t seed, double min_cosine) {
    const std::string tag = "generated, latent_len=" + std::to_string(latent_len);
    const std::vector<float> latent = make_latent(ggml_model.hparams.latent_channels, latent_len, seed);
    std::vector<float> wav_ggml, wav_coreml;
    std::string error, backend;
    if (!supertonic_vocoder_forward_ggml(ggml_model, latent.data(), latent_len, wav_ggml, &error, &backend)) {
        fail(tag + ": ggml vocoder failed: " + error);
        return;
    }
    bool pending = true;
    if (!supertonic_vocoder_forward_ggml(coreml_model, latent.data(), latent_len, wav_coreml, &error, &backend,
                                         &pending)) {
        fail(tag + ": Core ML vocoder failed: " + error);
        return;
    }
    expect(backend.rfind("coreml", 0) == 0 && !pending, tag + ": the call ran on " + backend);
    expect(wav_coreml.size() == wav_ggml.size(), tag + ": size mismatch");
    if (wav_coreml.size() != wav_ggml.size()) return;
    const double cos = cosine(wav_ggml, wav_coreml);
    std::fprintf(stderr, "[supertonic-coreml-parity] %s %s cosine=%.6f (gate %.4f)\n", tag.c_str(), backend.c_str(),
                 cos, min_cosine);
    expect(cos >= min_cosine, tag + ": cosine " + std::to_string(cos) + " below " + std::to_string(min_cosine));
}

// The GGUF is replaced after the weights were loaded and before the sidecar
// is made: the model is loaded through a symlink, which is then pointed at a
// clone of the file (same bytes, another inode). The prepare job must not
// follow the path to the replacement: it fails with source_changed, nothing
// is generated, and with use = automatic the vocoder runs on ggml. Loaded
// through the same symlink without the replacement, the stage adopts the
// artifact in `cache_dir` (the symlink resolves to the loaded file). The
// brand is an ungated one, so the generated route is planned on every host.
void check_replaced_gguf(const std::string & gguf, const std::string & cache_dir) {
    std::error_code ec;
    const fs::path dir = fs::temp_directory_path(ec) /
                         ("supertonic-replaced-" +
                          std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const fs::path link = dir / "model.gguf", clone = dir / "clone.gguf", cache = dir / "cache";
    fs::create_directories(dir, ec);
    fs::create_symlink(fs::absolute(gguf), link, ec);
    if (ec) {
        fail("replaced GGUF: cannot link the GGUF: " + ec.message());
        fs::remove_all(dir, ec);
        return;
    }
    tts_cpp::CoremlOptions o;
    o.wait = tts_cpp::CoremlOptions::Wait::construct;
    o.use = tts_cpp::CoremlOptions::Use::automatic;
    for (const bool replace : {false, true}) {
        const std::string what = replace ? "a GGUF replaced after the load" : "a GGUF loaded through a symlink";
        owned_model owned;
        owned.model.vocoder_coreml_managed = true;
        if (!load(link.string(), owned)) break;
        if (replace) {
            bool replaced = ::clonefile(fs::absolute(gguf).c_str(), clone.c_str(), 0) == 0;
            if (replaced) {
                fs::remove(link, ec);
                fs::create_symlink(clone, link, ec);
                replaced = !ec;
            }
            if (!replaced) {
                fail("replaced GGUF: cannot point the symlink at a clone");
                break;
            }
        }
        o.cache_dir = replace ? cache.string() : cache_dir;
        std::shared_ptr<supertonic_vocoder_coreml> stage;
        try {
            stage = supertonic_vocoder_coreml::create(owned.model, link.string(), o, false, UNGATED_BRAND);
        } catch (const std::exception & e) {
            fail(what + ": create: " + e.what());
            break;
        }
        const std::string st = stage ? stage->status_json() : std::string();
        const int latent_len = 8;
        const std::vector<float> latent = make_latent(owned.model.hparams.latent_channels, latent_len, 23u);
        std::vector<float> wav;
        std::string error, backend;
        const bool ok =
            supertonic_vocoder_forward_ggml(owned.model, latent.data(), latent_len, wav, &error, &backend);
        if (replace) {
            expect(stage && !stage->wait(0) && st.find("source_changed") != std::string::npos && ok &&
                       backend == "ggml",
                   what + " is not generated from (vocoder on " + backend + "): " + st.substr(0, 300));
        } else {
            expect(stage && stage->wait(0) && ok && backend.rfind("coreml", 0) == 0,
                   what + " adopts the cached sidecar (vocoder on " + backend + "): " + st.substr(0, 300));
        }
        std::fprintf(stderr, "[supertonic-coreml-parity] %s: state %s, vocoder on %s\n", what.c_str(),
                     status_field(st, "state").c_str(), backend.c_str());
        owned.model.vocoder_pending.reset();
        stage.reset();
    }
    fs::remove_all(dir, ec);
}

void check_engine_generated(const std::string & gguf, const tts_cpp::CoremlOptions & o) {
    using tts_cpp::supertonic::Engine;
    using tts_cpp::supertonic::EngineOptions;
    try {
        EngineOptions opts;
        opts.model_gguf_path = gguf;
        opts.n_threads = N_THREADS;
        opts.coreml = o;
        opts.coreml.wait = tts_cpp::CoremlOptions::Wait::construct;
        Engine engine(opts);
        expect(engine.vocoder_on_coreml() && engine.coreml_wait(0), "engine: the generated sidecar is not ready");
        const std::string status = engine.coreml_status();
        expect(status_field(status, "state") == "ready", "engine: status " + status);
        auto result = engine.synthesize("Core ML generated vocoder check.");
        expect(result.vocoder_synthesis_backend.rfind("coreml", 0) == 0 && !result.coreml_pending,
               "engine: vocoder ran on " + result.vocoder_synthesis_backend);
        expect(!result.pcm.empty(), "engine: empty PCM");
    } catch (const std::exception & e) {
        fail(std::string("engine on the generated sidecar: ") + e.what());
    }
}

// An Engine whose vocoder resolved to ggml synthesizes on ggml with
// SUPERTONIC_COREML_STRICT set: the stage's policy is resolved once, at
// construction, from the explicit options (and memory-degrade), and no call
// reads the environment again. `o` has no budget for use = off; with one it
// is the memory-degrade case.
void check_ggml_route_ignores_strict_env(const std::string & gguf, const tts_cpp::CoremlOptions & o,
                                         const std::string & what,
                                         tts_cpp::supertonic::Precision precision =
                                             tts_cpp::supertonic::Precision::Auto,
                                         int steps = 0, int vulkan_device = 0) {
    using tts_cpp::supertonic::Engine;
    using tts_cpp::supertonic::EngineOptions;
    scoped_env strict("SUPERTONIC_COREML_STRICT", "1");
    try {
        EngineOptions opts;
        opts.model_gguf_path = gguf;
        opts.n_threads = N_THREADS;
        opts.n_gpu_layers = 99;  // what fit projects
        opts.precision = precision;
        opts.steps = steps;
        opts.vulkan_device = vulkan_device;
        opts.coreml = o;
        Engine engine(opts);
        const std::string status = engine.coreml_status();
        expect(status_field(status, "state") == "disabled", what + ": status " + status);
        auto result = engine.synthesize("The strict variable is ignored here.");
        expect(result.vocoder_synthesis_backend == "ggml" && !result.pcm.empty(),
               what + ": vocoder ran on '" + result.vocoder_synthesis_backend + "'");
        std::fprintf(stderr, "[supertonic-coreml-parity] %s with SUPERTONIC_COREML_STRICT: synthesized on %s\n",
                     what.c_str(), result.vocoder_synthesis_backend.c_str());
    } catch (const std::exception & e) {
        fail(what + " with SUPERTONIC_COREML_STRICT: " + e.what());
    }
}

// A CPU-backed Engine with prewarm none leaves the vocoder slot idle until a
// call requests it; its prewarm_text warm-up is such a call, so the slot must
// have left idle once construction returns.
void check_cpu_warm_up_requests_idle_sidecar(const std::string & gguf, const tts_cpp::CoremlOptions & o) {
    using tts_cpp::supertonic::Engine;
    using tts_cpp::supertonic::EngineOptions;
    try {
        EngineOptions opts;
        opts.model_gguf_path = gguf;
        opts.n_threads = N_THREADS;
        opts.n_gpu_layers = 0;
        opts.coreml = o;
        opts.coreml.prewarm = tts_cpp::CoremlOptions::Prewarm::none;
        opts.coreml.wait = tts_cpp::CoremlOptions::Wait::none;
        {
            Engine idle(opts);
            const std::string state = status_field(idle.coreml_status(), "state");
            expect(state == "idle", "CPU engine without warm-up: state " + state + " (expected idle)");
        }
        opts.prewarm_text = "Warm up the vocoder sidecar.";
        Engine engine(opts);
        const std::string state = status_field(engine.coreml_status(), "state");
        expect(!state.empty() && state != "idle", "CPU engine warm-up from an idle slot: state " + state);
        engine.coreml_wait(-1);
        std::fprintf(stderr, "[supertonic-coreml-parity] CPU warm-up requested the idle sidecar (state %s after "
                     "construction)\n", state.c_str());
    } catch (const std::exception & e) {
        fail(std::string("CPU warm-up engine: ") + e.what());
    }
}

// Fit charges the generated route's envelope; with a budget that only the
// ggml projection fits, it degrades to ggml (use = off) instead of refusing,
// and an Engine with that budget synthesizes on ggml. Fit runs with a
// precision and a step count other than the defaults, and the Engine with the
// same ones must reach the same degrade, as must one given the larger
// headroom a fit was assessed with. With `required` (the
// generated route is required on this host) a fit that cannot be projected or
// a host without the free memory the near-budget case needs fails instead of
// skipping.
void check_fit(const std::string & gguf, const tts_cpp::CoremlOptions & o, const supertonic_vocoder_route & route,
               bool required) {
    tts_cpp::supertonic::FitOptions f;
    f.model_gguf_path = gguf;
    f.n_gpu_layers = 99;  // the fused GPU path is the one fit projects
    f.precision = "f32";
    f.steps = DEGRADE_STEPS;
    f.coreml = o;
    const tts_cpp::FitResult r = tts_cpp::supertonic::fit_params(f);
    if (r.status == tts_cpp::FitStatus::Error) {
        if (required) fail("fit: " + r.reason);
        else std::fprintf(stderr, "[supertonic-coreml-parity] fit: %s (not projected on this backend)\n", r.reason.c_str());
        return;
    }
    expect(r.coreml.route == "generate" && r.coreml.bytes == route.envelope_bytes && r.coreml.disabled_reason.empty(),
           "fit: route " + r.coreml.route + ", Core ML " + std::to_string(r.coreml.bytes) + " bytes (envelope " +
               std::to_string(route.envelope_bytes) + ")");
    uint64_t required_bytes = r.device.total_bytes + f.margin_bytes;
    if (r.device_shares_host_memory) required_bytes += r.host_bytes;
    if (required_bytes + r.coreml.bytes / 2 > r.device_free_bytes) {
        if (required) {
            fail("fit near the budget: free memory " + std::to_string(r.device_free_bytes) + " is below " +
                 std::to_string(required_bytes + r.coreml.bytes / 2));
        } else {
            std::fprintf(stderr, "[supertonic-coreml-parity] fit near the budget: not run (free memory too low)\n");
        }
        return;
    }
    tts_cpp::supertonic::FitOptions near = f;
    near.coreml.memory_budget_bytes = (int64_t) (required_bytes + r.coreml.bytes / 2);
    const tts_cpp::FitResult d = tts_cpp::supertonic::fit_params(near);
    expect(d.coreml.disabled_reason == "memory" && d.fits && d.coreml.ggml_only_fits,
           "fit near the budget: disabled_reason '" + d.coreml.disabled_reason + "', fits " +
               std::to_string(d.fits));
    std::fprintf(stderr, "[supertonic-coreml-parity] fit: envelope %.1f MiB, near-budget degrade ok\n",
                 mib(r.coreml.bytes));
    check_ggml_route_ignores_strict_env(gguf, near.coreml, "memory-degrade", tts_cpp::supertonic::Precision::F32,
                                        DEGRADE_STEPS);
    // The automatic Vulkan adapter (-1), which the loader accepts and ignores
    // off Vulkan, needs the same memory: fit projects it and the Engine
    // degrades the same way.
    tts_cpp::supertonic::FitOptions near_auto = near;
    near_auto.vulkan_device = -1;
    const tts_cpp::FitResult da = tts_cpp::supertonic::fit_params(near_auto);
    expect(da.status != tts_cpp::FitStatus::Error && da.coreml.disabled_reason == "memory",
           "fit near the budget, automatic adapter: " + da.reason + ", disabled_reason '" +
               da.coreml.disabled_reason + "'");
    check_ggml_route_ignores_strict_env(gguf, near.coreml, "memory-degrade, automatic adapter",
                                        tts_cpp::supertonic::Precision::F32, DEGRADE_STEPS, -1);
    // A budget the fit cannot assess (the per-step estimator path it does not
    // model): nothing shows the generated sidecar fits, so it stays off.
    {
        scoped_env per_step("SUPERTONIC_DISABLE_LOOP_GRAPH", "1");
        tts_cpp::supertonic::FitOptions roomy = f;
        roomy.coreml.memory_budget_bytes = (int64_t) 1 << 50;
        const tts_cpp::FitResult u = tts_cpp::supertonic::fit_params(roomy);
        expect(u.status == tts_cpp::FitStatus::Error && u.reason == "compute-path-not-supported",
               "fit on the per-step path: " + u.reason);
        check_ggml_route_ignores_strict_env(gguf, roomy.coreml, "budget without a projection",
                                            tts_cpp::supertonic::Precision::F32, DEGRADE_STEPS);
    }

    // A fit assessed with a larger headroom than the default: the budget
    // holds ggml + Core ML with the default 256 MiB margin, but only ggml
    // with this one. The Engine, given the same margin, must reach the same
    // verdict (memory-degrade) as that fit.
    tts_cpp::supertonic::FitOptions wide = f;
    wide.margin_bytes = f.margin_bytes + r.coreml.bytes;
    const uint64_t wide_budget = required_bytes + r.coreml.bytes + (64ull << 20);
    wide.coreml.memory_budget_bytes = (int64_t) wide_budget;
    if (wide_budget > r.device_free_bytes) {
        if (required) {
            fail("fit with a larger margin: free memory " + std::to_string(r.device_free_bytes) + " is below " +
                 std::to_string(wide_budget));
        } else {
            std::fprintf(stderr, "[supertonic-coreml-parity] fit with a larger margin: not run (free memory too low)\n");
        }
        return;
    }
    const tts_cpp::FitResult assessed = tts_cpp::supertonic::fit_params(wide);
    tts_cpp::supertonic::FitOptions as_default = wide;
    as_default.margin_bytes = f.margin_bytes;
    const tts_cpp::FitResult by_default = tts_cpp::supertonic::fit_params(as_default);
    expect(assessed.coreml.disabled_reason == "memory" && by_default.coreml.disabled_reason.empty(),
           "a budget only the larger margin degrades: '" + assessed.coreml.disabled_reason + "' with it, '" +
               by_default.coreml.disabled_reason + "' with the default");
    try {
        tts_cpp::supertonic::EngineOptions eo;
        eo.model_gguf_path = gguf;
        eo.n_threads = N_THREADS;
        eo.n_gpu_layers = 99;
        eo.precision = tts_cpp::supertonic::Precision::F32;
        eo.steps = DEGRADE_STEPS;
        eo.coreml = wide.coreml;
        eo.fit_margin_bytes = wide.margin_bytes;
        tts_cpp::supertonic::Engine engine(eo);
        const std::string status = engine.coreml_status();
        auto result = engine.synthesize("The larger margin keeps the vocoder on ggml.");
        expect(status_field(status, "state") == "disabled" && result.vocoder_synthesis_backend == "ggml" &&
                   !result.pcm.empty(),
               "Engine with the fit's nondefault margin: memory-degrade as that fit (vocoder on '" +
                   result.vocoder_synthesis_backend + "'), " + status.substr(0, 200));
        std::fprintf(stderr, "[supertonic-coreml-parity] fit with a %.0f MiB margin: Engine state %s, vocoder on %s\n",
                     mib(wide.margin_bytes), status_field(status, "state").c_str(),
                     result.vocoder_synthesis_backend.c_str());
    } catch (const std::exception & e) {
        fail(std::string("Engine with the fit's nondefault margin: ") + e.what());
    }
}

// The F16 weight deny list changes the weight storage, so fit projects it and
// the Engine decides memory-degrade with it. A budget between the two storage
// policies (ggml + Core ML fits with F16 weights, not with the denied tensors
// kept at F32) keeps generation without the deny list and degrades with it,
// and an Engine given the list and that budget synthesizes on ggml.
void check_fit_deny_list(const std::string & gguf, const tts_cpp::CoremlOptions & o, bool required) {
    tts_cpp::supertonic::FitOptions f;
    f.model_gguf_path = gguf;
    f.n_gpu_layers = 99;
    f.precision = "f32";
    f.f16_weights = 1;
    f.steps = DEGRADE_STEPS;
    f.coreml = o;
    tts_cpp::supertonic::FitOptions denied = f;
    denied.f16_weights_deny_list = {"convnext"};
    const tts_cpp::FitResult base = tts_cpp::supertonic::fit_params(f);
    const tts_cpp::FitResult deny = tts_cpp::supertonic::fit_params(denied);
    if (base.status == tts_cpp::FitStatus::Error || deny.status == tts_cpp::FitStatus::Error) {
        if (required) fail("fit with a deny list: " + base.reason + " / " + deny.reason);
        return;
    }
    const uint64_t MiB = 1ull << 20;
    expect(deny.device.weights_bytes > base.device.weights_bytes + MiB,
           "fit with a deny list projects " + std::to_string(deny.device.weights_bytes) +
               " weight bytes, without it " + std::to_string(base.device.weights_bytes));
    uint64_t need = deny.device.total_bytes + denied.margin_bytes;
    if (deny.device_shares_host_memory) need += deny.host_bytes;
    // Core ML + ggml with the deny list exceeds it by 1 MiB; ggml alone fits.
    const uint64_t budget = need + deny.coreml.bytes - MiB;
    if (budget + 256 * MiB > deny.device_free_bytes) {
        if (required) {
            fail("fit with a deny list: free memory " + std::to_string(deny.device_free_bytes) + " is below " +
                 std::to_string(budget + 256 * MiB));
        } else {
            std::fprintf(stderr, "[supertonic-coreml-parity] fit with a deny list: not run (free memory too low)\n");
        }
        return;
    }
    f.coreml.memory_budget_bytes = (int64_t) budget;
    denied.coreml.memory_budget_bytes = (int64_t) budget;
    const tts_cpp::FitResult base_b = tts_cpp::supertonic::fit_params(f);
    const tts_cpp::FitResult deny_b = tts_cpp::supertonic::fit_params(denied);
    expect(base_b.coreml.disabled_reason.empty() && deny_b.coreml.disabled_reason == "memory" && deny_b.fits,
           "a budget between the storage policies: '" + base_b.coreml.disabled_reason + "' without the deny list, '" +
               deny_b.coreml.disabled_reason + "' with it");
    try {
        tts_cpp::supertonic::EngineOptions eo;
        eo.model_gguf_path = gguf;
        eo.n_threads = N_THREADS;
        eo.n_gpu_layers = 99;
        eo.precision = tts_cpp::supertonic::Precision::F32;
        eo.f16_weights = 1;
        eo.f16_weights_deny_list = denied.f16_weights_deny_list;
        eo.steps = DEGRADE_STEPS;
        eo.coreml = denied.coreml;
        tts_cpp::supertonic::Engine engine(eo);
        const std::string status = engine.coreml_status();
        auto result = engine.synthesize("The deny list keeps the vocoder on ggml.");
        expect(status_field(status, "state") == "disabled" && result.vocoder_synthesis_backend == "ggml" &&
                   !result.pcm.empty(),
               "Engine with the deny list near the budget: vocoder on '" + result.vocoder_synthesis_backend + "', " +
                   status.substr(0, 200));
        std::fprintf(stderr,
                     "[supertonic-coreml-parity] deny list: weights %.1f -> %.1f MiB, Engine state %s, vocoder on %s\n",
                     mib(base.device.weights_bytes), mib(deny.device.weights_bytes),
                     status_field(status, "state").c_str(), result.vocoder_synthesis_backend.c_str());
    } catch (const std::exception & e) {
        fail(std::string("Engine with the deny list near the budget: ") + e.what());
    }
}

// A generated sidecar stays off on the device classes the template prefers
// ggml on, unless use = require.
void check_device_gating(const std::string & gguf, const std::string & cache_dir) {
    tts_cpp::CoremlOptions o;
    o.cache_dir = cache_dir;
    for (const char * brand : {"Apple M3 Ultra", "Apple M4 Max", "Apple M4 Max (Virtual)"}) {
        const supertonic_vocoder_route r = supertonic_vocoder_plan_route(gguf, o, brand);
        expect(r.route == "none" && r.state == "unsupported" && r.reason == "device",
               std::string(brand) + ": route " + r.route + " " + r.state + " " + r.reason);
    }
    for (const char * brand : {"Apple M2", "Apple M4 Pro", "Apple M1"}) {
        const supertonic_vocoder_route r = supertonic_vocoder_plan_route(gguf, o, brand);
        expect(r.route == "generate", std::string(brand) + ": route " + r.route + " " + r.reason);
    }
    tts_cpp::CoremlOptions req = o;
    req.use = tts_cpp::CoremlOptions::Use::require;
    expect(supertonic_vocoder_plan_route(gguf, req, "Apple M3 Ultra").route == "generate",
           "use = require generates on an Ultra");
}

// The backend label a sidecar opened for `units` reports.
std::string label_for_units(const std::string & units) {
    if (units == "cpu_and_ane") return "coreml-ane";
    if (units == "cpu_and_gpu") return "coreml-gpu";
    if (units == "cpu_only") return "coreml-cpu";
    return "coreml-all";
}

// The overlapped memory schedule: ggml vocoder calls on this thread while the
// sidecar is generated, compiled, loaded and warmed in the background into a
// fresh `cache_dir`, until five calls have run on Core ML. One sampler thread
// reads the footprint and the slot's state every millisecond (the warm-up
// prediction takes a few); another reads ANECompilerService every 20 ms,
// each query bracketed by two reads of the slot state. 1.25 x every phase's
// measurement must stay within the template constant that charges it, and a
// failed probe, a failed call or one on another backend fails the run
// (engines/coreml/test/memory_schedule.h).
void memory_schedule(const std::string & gguf, const std::string & cache_dir) {
    using namespace qvac_coreml_memory;
    const std::string chip = supertonic_cpu_brand();
    std::fprintf(stderr, "[supertonic-coreml-parity] memory schedule (cache %s, host %s)\n", cache_dir.c_str(),
                 chip.c_str());
    tts_cpp::CoremlOptions opts;
    opts.cache_dir = cache_dir;
    const supertonic_vocoder_route route = supertonic_vocoder_plan_route(gguf, opts, UNGATED_BRAND);
    if (route.route != "generate") {
        fail("memory schedule: route " + route.route + " " + route.state + " " + route.reason);
        return;
    }
    owned_model owned;
    owned.model.vocoder_coreml_managed = true;
    if (!load(gguf, owned)) return;
    supertonic_model & model = owned.model;
    const int latent_len = 150;  // three windows
    const std::vector<float> latent = make_latent(model.hparams.latent_channels, latent_len, 29u);
    std::vector<float> wav;
    std::string error, backend;
    for (int i = 0; i < 2; ++i) {  // warm the ggml arena
        const bool ok = supertonic_vocoder_forward_ggml(model, latent.data(), latent_len, wav, &error, &backend);
        if (classify_call(ok, backend, SUPERTONIC_VOCODER_GGML_BACKEND, "") != schedule_call::ggml) {
            fail("memory schedule: the ggml warm-up call succeeds (" + backend + (ok ? "" : ": " + error) + ")");
            return;
        }
    }
    uint64_t base = 0;
    const bool base_ok = phys_footprint(base);
    size_t footprint_failures = 0;
    std::atomic<bool> stop{false};
    std::atomic<int> phase{(int) mem_phase::ggml};
    std::atomic<const supertonic_vocoder_coreml *> watched{nullptr};
    std::vector<AneSample> ane_samples;
    std::vector<mem_sample> samples;
    samples.reserve(1 << 16);
    std::thread sampler([&] {
        while (!stop.load()) {
            const supertonic_vocoder_coreml * v = watched.load();
            if (v != nullptr) phase.store((int) mem_phase_of(v->state()));
            uint64_t fp = 0;
            if (phys_footprint(fp)) {
                samples.push_back({fp, (mem_phase) phase.load()});
            } else {
                ++footprint_failures;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });
    // Each compiler query is bracketed by its own reads of the slot state
    // (not the footprint sampler's cached phase): a query that crossed the
    // load's start or end is dropped (counted, reported).
    size_t ane_crossing = 0;
    std::thread ane_sampler([&] {
        const auto slot_phase = [&] {
            const supertonic_vocoder_coreml * v = watched.load();
            return v ? mem_phase_of(v->state()) : mem_phase::ggml;
        };
        while (!stop.load()) {
            const mem_phase before = slot_phase();
            uint64_t a = 0;
            const bool ok = ane_compiler_rss(a);
            const std::optional<bool> loading = ane_query_loading(before, slot_phase());
            if (loading) {
                ane_samples.push_back({a, *loading, ok});
            } else {
                ++ane_crossing;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    });
    const auto t0 = std::chrono::steady_clock::now();
    std::shared_ptr<supertonic_vocoder_coreml> stage;
    try {
        stage = supertonic_vocoder_coreml::create(model, gguf, opts, false, UNGATED_BRAND);
    } catch (const std::exception & e) {
        fail(std::string("memory schedule: create: ") + e.what());
    }
    watched.store(stage.get());
    int ggml_calls = 0, coreml_calls = 0, bad_calls = 0;
    const std::string coreml_label = label_for_units(route.compute_units);
    while (stage && std::chrono::steady_clock::now() - t0 < std::chrono::minutes(5)) {
        bool pending = false;
        backend.clear();
        const bool ok =
            supertonic_vocoder_forward_ggml(model, latent.data(), latent_len, wav, &error, &backend, &pending);
        const schedule_call k = classify_call(ok, backend, SUPERTONIC_VOCODER_GGML_BACKEND, coreml_label);
        if (k == schedule_call::bad) {
            ++bad_calls;
            fail("memory schedule: every overlapped call succeeds on ggml or " + coreml_label + " (" +
                 (ok ? backend : "failed on " + backend + ": " + error) + ")");
            break;
        }
        if (k == schedule_call::ggml) {
            ++ggml_calls;
        } else if (++coreml_calls >= 5) {
            break;
        }
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    stop.store(true);
    sampler.join();
    ane_sampler.join();
    if (!stage) return;
    std::fprintf(stderr, "  status: %s\n", stage->status_json().c_str());

    const mem_breakdown b = mem_breakdown_of(samples, base);
    const AneGrowth ane = ane_compiler_growth(ane_samples);
    const AneBaseline baseline = ane_compiler_baseline(ane);
    char before_load[96];
    if (baseline == AneBaseline::unknown) {
        std::snprintf(before_load, sizeof(before_load), "no sample before the load, an unknown baseline");
    } else {
        std::snprintf(before_load, sizeof(before_load), "before the load %.1f MiB, a %s baseline", mib(ane.before),
                      ane_baseline_name(baseline));
    }
    uint64_t ane_peak = 0;
    for (const auto & a : ane_samples) ane_peak = std::max(ane_peak, a.rss);
    const mem_charges c{route.prepare_peak_bytes, route.load_peak_bytes, route.resident_bytes,
                        route.ane_compile_reserve_bytes, route.envelope_bytes};
    const uint64_t other_resident = route.resident_bytes - route.workspace_bytes;
    const uint64_t workspace = b.steady > other_resident ? b.steady - other_resident : 0;
    std::fprintf(stderr,
                 "  memory (MiB; each phase over the footprint just before it began, steady over the footprint just "
                 "before the load, whole run over the warm ggml baseline %.1f; a phase needs x%.2f its "
                 "measurement):\n"
                 "    prepare    %7.1f  needs %7.1f  charged %7.1f  (generate %.1f, compile %.1f)\n"
                 "    load+warm  %7.1f  needs %7.1f  charged %7.1f  (load %.1f, warm-up %.1f)\n"
                 "    steady     %7.1f  needs %7.1f  charged %7.1f  (workspace %.1f, charged %.1f; over the baseline "
                 "%.1f)\n"
                 "    whole run  %7.1f  needs %7.1f  charged %7.1f  (the envelope: margined phases)\n"
                 "    ANECompilerService size while loading and warming %.1f, needs %.1f, reserve %.1f (growth %.1f, "
                 "%s; whole run %.1f)\n",
                 mib(base), MEASURED_MARGIN, mib(b.prepare), MEASURED_MARGIN * mib(b.prepare), mib(c.prepare),
                 mib(b.generate), mib(b.compile), mib(b.load_warm), MEASURED_MARGIN * mib(b.load_warm), mib(c.load),
                 mib(b.load), mib(b.warm), mib(b.steady), MEASURED_MARGIN * mib(b.steady), mib(c.resident),
                 mib(workspace), mib(route.workspace_bytes), mib(b.steady_over_base), mib(b.total), mib(b.total),
                 mib(c.envelope), mib(ane.peak), MEASURED_MARGIN * mib(ane.peak), mib(c.ane_reserve),
                 mib(ane.growth), before_load, mib(ane_peak));
    std::fprintf(stderr,
                 "  memory report: {\"chip\":\"%s\",\"template\":\"%s\",\"units\":\"%s\",\"base\":%llu,\"prepare\":%llu,"
                 "\"generate\":%llu,\"compile\":%llu,\"load_warm\":%llu,\"load\":%llu,\"warm\":%llu,\"steady\":%llu,"
                 "\"steady_over_base\":%llu,\"workspace\":%llu,\"total\":%llu,\"ane_compiler_load_peak\":%llu,"
                 "\"ane_compiler_before_load\":%llu,\"ane_compiler_growth\":%llu,\"ane_compiler_peak\":%llu,"
                 "\"ane_compiler_baseline\":\"%s\",\"samples\":%zu,\"ane_compiler_loading_samples\":%zu,"
                 "\"ane_compiler_crossing\":%zu}\n",
                 chip.c_str(), route.template_id.c_str(), route.compute_units.c_str(), (unsigned long long) base,
                 (unsigned long long) b.prepare, (unsigned long long) b.generate, (unsigned long long) b.compile,
                 (unsigned long long) b.load_warm, (unsigned long long) b.load, (unsigned long long) b.warm,
                 (unsigned long long) b.steady, (unsigned long long) b.steady_over_base,
                 (unsigned long long) workspace, (unsigned long long) b.total, (unsigned long long) ane.peak,
                 (unsigned long long) ane.before, (unsigned long long) ane.growth, (unsigned long long) ane_peak,
                 ane_baseline_name(baseline), samples.size(), ane.loading_samples, ane_crossing);
    const bool ane_units = route.compute_units == "all" || route.compute_units == "cpu_and_ane";
    std::vector<std::string> failures = probe_failures(base_ok, footprint_failures, ane_units, ane);
    for (const std::string & f : mem_failures(b, ane.peak, c)) failures.push_back(f);
    for (const std::string & f : failures) fail("memory schedule: " + f);
    std::fprintf(stderr,
                 "[supertonic-coreml-parity] overlapped schedule: %d ggml + %d Core ML calls in %.1f s; footprint base "
                 "%.1f MiB, peak +%.1f MiB (envelope %.1f MiB, units %s); ANECompilerService %.1f MiB while loading, "
                 "grew %.1f MiB (%s)\n",
                 ggml_calls, coreml_calls, secs, mib(base), mib(b.total), mib(route.envelope_bytes),
                 route.compute_units.c_str(), mib(ane.peak), mib(ane.growth), before_load);
    expect(coreml_calls >= 5 && ggml_calls >= 1 && bad_calls == 0,
           "memory schedule: " + std::to_string(ggml_calls) + " ggml and " + std::to_string(coreml_calls) +
               " Core ML calls");
    model.vocoder_pending.reset();
}

int generated_main(const std::string & gguf, const std::string & cache_dir, double min_cosine, bool require_coreml,
                   const std::string & memory_dir) {
    unsetenv("SUPERTONIC_COREML_DISABLE");
    unsetenv("SUPERTONIC_COREML_STRICT");
    if (fs::exists(coreml_vocoder_sidecar_path(gguf))) {
        fail("a prebuilt sidecar sits next to " + gguf + "; the generated route needs none");
        return 1;
    }
    if (!memory_dir.empty()) {
        memory_schedule(gguf, memory_dir);
        std::fprintf(stderr, g_failures ? "test_supertonic_coreml_parity: %d failure(s)\n"
                                        : "test_supertonic_coreml_parity: OK\n", g_failures);
        return g_failures ? 1 : 0;
    }
    tts_cpp::CoremlOptions o;
    o.cache_dir = cache_dir;
    const supertonic_vocoder_route route = supertonic_vocoder_plan_route(gguf, o);
    std::fprintf(stderr, "[supertonic-coreml-parity] %s: route %s%s%s template %s units %s envelope %.1f MiB%s\n",
                 gguf.c_str(), route.route.c_str(), route.reason.empty() ? "" : " ",
                 route.reason.c_str(), route.template_id.c_str(), route.compute_units.c_str(),
                 mib(route.envelope_bytes), route.validated ? " (validated)" : "");
    if (route.route != "generate") {
        // A Max or Ultra host keeps the generated sidecar off by default; the
        // gates below need it, so they run with use = require there.
        const bool device = route.reason == "device";
        if (!device) {
            if (require_coreml) fail("route " + route.route + " (" + route.state + ": " + route.reason + ")");
            return require_coreml ? 1 : SKIP_EXIT_CODE;
        }
        std::fprintf(stderr, "[supertonic-coreml-parity] this device class prefers ggml; gates run with use = require\n");
        o.use = tts_cpp::CoremlOptions::Use::require;
    }

    owned_model ggml_owned;
    {
        scoped_env disable("SUPERTONIC_COREML_DISABLE", "1");
        if (!load(gguf, ggml_owned)) return 1;
    }
    tts_cpp::CoremlOptions first = o;
    first.wait = tts_cpp::CoremlOptions::Wait::construct;
    first.use = tts_cpp::CoremlOptions::Use::require;
    managed_model gen;
    const auto t0 = std::chrono::steady_clock::now();
    if (!load_managed(gguf, first, gen)) return 1;
    const double ready_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const std::string status = gen.stage->status_json();
    std::fprintf(stderr, "[supertonic-coreml-parity] ready after %.1f s: %s\n", ready_s, status.c_str());
    expect(status_field(status, "state") == "ready", "first owner: state " + status_field(status, "state"));
    const std::string first_route = status_field(status, "route");
    expect(first_route == "generated" || first_route == "cached", "first owner: route " + first_route);

    const int window = 64;
    check_generated_parity(ggml_owned.model, gen.owned.model, window / 2, 11u, min_cosine);
    check_generated_parity(ggml_owned.model, gen.owned.model, window, 13u, min_cosine);
    check_generated_parity(ggml_owned.model, gen.owned.model, 3 * window - window / 3, 17u, min_cosine);

    // A second owner adopts the published artifact.
    managed_model second;
    if (load_managed(gguf, first, second)) {
        const std::string st = second.stage->status_json();
        expect(status_field(st, "route") == "cached" && status_field(st, "state") == "ready",
               "second owner: " + st);
    }

    check_replaced_gguf(gguf, cache_dir);
    check_engine_generated(gguf, o);
    check_cpu_warm_up_requests_idle_sidecar(gguf, o);
    tts_cpp::CoremlOptions off = o;
    off.use = tts_cpp::CoremlOptions::Use::off;
    check_ggml_route_ignores_strict_env(gguf, off, "use = off");
    check_fit(gguf, o, supertonic_vocoder_plan_route(gguf, o), require_coreml);
    check_fit_deny_list(gguf, o, require_coreml);
    check_device_gating(gguf, cache_dir);

    if (g_failures == 0) {
        std::printf("test_supertonic_coreml_parity: OK\n");
        return 0;
    }
    std::fprintf(stderr, "test_supertonic_coreml_parity: %d failure(s)\n", g_failures);
    return 1;
}

}  // namespace

int main(int argc, char ** argv) {
    std::string gguf, cache_dir, memory_dir;
    double min_cosine = MIN_COSINE;
    bool require_coreml = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--gguf") gguf = next();
        else if (a == "--cache-dir") cache_dir = next();
        else if (a == "--memory-schedule") memory_dir = next();
        else if (a == "--min-cosine") min_cosine = std::atof(next().c_str());
        else if (a == "--require-coreml") require_coreml = true;
        else {
            std::fprintf(stderr, "[supertonic-coreml-parity] unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    if (!gguf.empty()) {
        if (cache_dir.empty() && memory_dir.empty()) {
            std::fprintf(stderr, "[supertonic-coreml-parity] --gguf needs --cache-dir or --memory-schedule\n");
            return 2;
        }
        return generated_main(gguf, cache_dir, min_cosine, require_coreml, memory_dir);
    }

    const char * env_dir = std::getenv("SUPERTONIC_COREML_TEST_MODELS_DIR");
    if (env_dir == nullptr) {
        std::fprintf(stderr,
                     "[supertonic-coreml-parity] SUPERTONIC_COREML_TEST_MODELS_DIR is not "
                     "set; skipping\n");
        return SKIP_EXIT_CODE;
    }
    const std::string dir = env_dir;
    gguf = find_supertonic_gguf(dir);
    if (gguf.empty()) {
        std::fprintf(stderr,
                     "[supertonic-coreml-parity] no Supertonic GGUF under %s; skipping\n",
                     dir.c_str());
        return SKIP_EXIT_CODE;
    }
    const std::string sidecar = coreml_vocoder_sidecar_path(gguf);
    if (!fs::exists(sidecar)) {
        std::fprintf(stderr, "[supertonic-coreml-parity] no sidecar at %s; skipping\n",
                     sidecar.c_str());
        return SKIP_EXIT_CODE;
    }

    owned_model ggml_owned;
    {
        scoped_env disable("SUPERTONIC_COREML_DISABLE", "1");
        if (!load(gguf, ggml_owned)) return 1;
    }
    expect(!ggml_owned.model.vocoder_on_coreml, "disabled load still attached the sidecar");

    owned_model coreml_owned;
    if (!load(gguf, coreml_owned)) return 1;
    if (!coreml_owned.model.vocoder_on_coreml) {
        fail("sidecar exists at " + sidecar + " but did not attach");
        return 1;
    }

    const int window =
        (int) supertonic_coreml_vocoder_window_frames(coreml_owned.model.coreml_vocoder);
    const int context = supertonic_coreml_vocoder_context_frames(coreml_owned.model);
    std::fprintf(stderr, "[supertonic-coreml-parity] %s window=%d context=%d\n",
                 sidecar.c_str(), window, context);
    expect(window > context, "exported window does not clear the causal context");

    // Shorter than the window (single padded call), exactly the window, and
    // long enough to need several stitched windows.
    check_parity(ggml_owned.model, coreml_owned.model, std::max(1, window / 2), 11u);
    check_parity(ggml_owned.model, coreml_owned.model, window, 13u);
    check_parity(ggml_owned.model, coreml_owned.model, 3 * window - window / 3, 17u);

    check_disabled_fallback(gguf, std::max(1, window / 2));
    check_invalid_sidecar(gguf, std::max(1, window / 2));
    check_engine_report(gguf);
    check_cpu_warm_up_reaches_sidecar(gguf);

    if (g_failures == 0) {
        std::printf("test_supertonic_coreml_parity: OK\n");
        return 0;
    }
    std::fprintf(stderr, "test_supertonic_coreml_parity: %d failure(s)\n", g_failures);
    return 1;
}

#endif  // TTS_CPP_USE_COREML
