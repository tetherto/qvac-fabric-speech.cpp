// The Supertonic vocoder call's hand-off between its Core ML sidecar and the
// ggml graph (supertonic_vocoder_forward_ggml), with test doubles and no model
// weights: a ready sidecar serves the call, a pending one leaves it to ggml, a
// failed prediction retires the sidecar and fails under the strict policy, a
// throwing acquisition fails the call, a cancelled wait cancels it, a
// blocking readiness wait ends on a cancel made after it started (a later
// wait cannot take that cancel back), and the
// strict policy is taken once, from the stage, never from the environment
// (SUPERTONIC_COREML_STRICT is read only for a sidecar attached at load, and
// only by Core ML builds).
//
// Without weights the ggml graph cannot run, so a call that should reach it
// passes latent_len = 0: the graph rejects that length ("latent_len must be
// positive"), which no policy refusal says.

#include "supertonic_internal.h"
#include "supertonic_vocoder_coreml.h"

#include "../test/test_env_portable.h"

#include "ggml.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace tts_cpp::supertonic::detail;

namespace {

int g_failures = 0;

void check(bool ok, const std::string & what) {
    std::fprintf(stderr, "  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_failures;
}

struct scoped_env {
    const char * name;
    scoped_env(const char * n, const char * value) : name(n) { setenv(name, value, 1); }
    ~scoped_env() { unsetenv(name); }
};

constexpr const char * GGML_LENGTH_ERROR = "latent_len must be positive";
constexpr int WINDOW = 64;  // the generated sidecar's window
constexpr int LATENT_FRAMES = 160;  // several windows
constexpr float SIDECAR_SAMPLE = 0.25f;

// A model with the vocoder geometry the hand-off reads (hyperparameters and
// the convolution kernel sizes behind the window context), and no weights.
struct shape_model {
    ggml_context * ctx = nullptr;
    supertonic_model model;
    shape_model() {
        ggml_init_params params{};
        params.mem_size = 64 * ggml_tensor_overhead();
        params.no_alloc = true;
        ctx = ggml_init(params);
        supertonic_vocoder_weights & v = model.vocoder;
        v.embed_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 7, 24, 512);
        for (auto & block : v.convnext) block.dw_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 7, 1, 512);
        v.head1_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 512, 2048);
        v.head2_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 2048, 512);
    }
    ~shape_model() {
        model.vocoder_pending.reset();
        ggml_free(ctx);
    }
    size_t samples_per_frame() const {
        return (size_t) model.hparams.base_chunk_size * model.hparams.ttl_chunk_compress_factor;
    }
};

class fake_sidecar final : public supertonic_vocoder_sidecar {
public:
    explicit fake_sidecar(size_t samples_per_frame, bool ok) : spf_(samples_per_frame), ok_(ok) {}
    int window() const override { return WINDOW; }
    const char * label() const override { return "coreml-fake"; }
    bool synthesize(const float *, int64_t, int64_t, int64_t, int64_t, int64_t keep, float * wav) override {
        ++predictions;
        if (!ok_) return false;
        for (size_t i = 0; i < (size_t) keep * spf_; ++i) wav[i] = SIDECAR_SAMPLE;
        return true;
    }
    int predictions = 0;

private:
    size_t spf_;
    bool ok_;
};

enum class acquire_mode { ready, failing, pending, throws };

class fake_pending final : public supertonic_vocoder_pending {
public:
    fake_pending(size_t samples_per_frame, acquire_mode mode, bool strict)
        : sidecar(std::make_shared<fake_sidecar>(samples_per_frame, mode != acquire_mode::failing)),
          mode_(mode),
          strict_(strict) {}
    supertonic_vocoder_call acquire(int, const std::atomic<bool> *) override {
        ++acquisitions;
        supertonic_vocoder_call call;
        switch (mode_) {
            case acquire_mode::ready:
            case acquire_mode::failing:
                call.sidecar = sidecar;
                call.retire = [this](const std::string & reason) { retired.push_back(reason); };
                break;
            case acquire_mode::pending:
                call.pending = true;
                break;
            case acquire_mode::throws:
                throw std::runtime_error("fake: the sidecar cannot serve this call");
        }
        return call;
    }
    bool strict() const override { return strict_; }

    std::shared_ptr<fake_sidecar> sidecar;
    int acquisitions = 0;
    std::vector<std::string> retired;

private:
    acquire_mode mode_;
    bool strict_;
};

struct outcome {
    bool ok = false;
    std::string error, backend;
    bool pending = false;
    std::vector<float> wav;
};

outcome forward(const supertonic_model & model, int latent_len, const std::atomic<bool> * cancel = nullptr) {
    outcome o;
    const std::vector<float> latent((size_t) model.hparams.latent_channels * (size_t) (latent_len > 0 ? latent_len : 1),
                                    0.0f);
    o.ok = supertonic_vocoder_forward_ggml(model, latent.data(), latent_len, o.wav, &o.error, &o.backend, &o.pending,
                                           cancel);
    return o;
}

bool reached_ggml(const outcome & o) {
    return !o.ok && o.error.find(GGML_LENGTH_ERROR) != std::string::npos && o.backend == "ggml";
}

std::string describe(const outcome & o) {
    return std::string(o.ok ? "ok" : "failed") + ", backend '" + o.backend + "', error '" + o.error + "'";
}

void fake_stage_gates() {
    std::fprintf(stderr, "hand-off with a fake stage\n");
    {
        shape_model m;
        auto stage = std::make_shared<fake_pending>(m.samples_per_frame(), acquire_mode::ready, false);
        m.model.vocoder_coreml_managed = true;
        m.model.vocoder_pending = stage;
        const outcome o = forward(m.model, LATENT_FRAMES);
        bool filled = o.wav.size() == LATENT_FRAMES * m.samples_per_frame();
        for (float s : o.wav) filled = filled && s == SIDECAR_SAMPLE;
        check(o.ok && o.backend == "coreml-fake" && !o.pending && filled && stage->acquisitions == 1 &&
                  stage->sidecar->predictions >= 3,
              "a ready sidecar serves every window of the call (" + std::to_string(stage->sidecar->predictions) +
                  " predictions): " + describe(o));
    }
    {
        shape_model m;
        auto stage = std::make_shared<fake_pending>(m.samples_per_frame(), acquire_mode::pending, true);
        m.model.vocoder_coreml_managed = true;
        m.model.vocoder_pending = stage;
        const outcome o = forward(m.model, 0);
        check(reached_ggml(o) && o.pending, "a pending sidecar leaves the call to ggml, marked pending: " + describe(o));
    }
    {
        shape_model m;
        auto stage = std::make_shared<fake_pending>(m.samples_per_frame(), acquire_mode::failing, true);
        m.model.vocoder_coreml_managed = true;
        m.model.vocoder_pending = stage;
        const outcome o = forward(m.model, LATENT_FRAMES);
        check(!o.ok && o.error.find("strict") != std::string::npos && stage->retired.size() == 1 &&
                  stage->retired[0] == "prediction failed" && o.wav.empty(),
              "a failed prediction retires the sidecar and fails the call under strict: " + describe(o));
    }
    for (bool strict : {false, true}) {
        // The runtime's wait returns early, with nothing to run on, once the
        // flag is set.
        shape_model m;
        auto stage = std::make_shared<fake_pending>(m.samples_per_frame(), acquire_mode::pending, strict);
        m.model.vocoder_coreml_managed = true;
        m.model.vocoder_pending = stage;
        std::atomic<bool> cancel{true};
        const outcome o = forward(m.model, LATENT_FRAMES, &cancel);
        check(!o.ok && o.error == "cancelled" && o.wav.empty(),
              std::string("a wait ended by cancellation cancels the call, not a ggml run") +
                  (strict ? " (strict): " : ": ") + describe(o));
    }
    {
        shape_model m;
        auto stage = std::make_shared<fake_pending>(m.samples_per_frame(), acquire_mode::throws, true);
        m.model.vocoder_coreml_managed = true;
        m.model.vocoder_pending = stage;
        const outcome o = forward(m.model, LATENT_FRAMES);
        check(!o.ok && o.error == "fake: the sidecar cannot serve this call",
              "an acquisition that throws fails the call with its reason: " + describe(o));
    }
}

// SUPERTONIC_COREML_STRICT never reaches a call of a model whose vocoder the
// Core ML runtime manages: the stage resolved its policy at construction.
void strict_environment_gates() {
    std::fprintf(stderr, "the strict environment variable\n");
    scoped_env strict("SUPERTONIC_COREML_STRICT", "1");
    {
        // An Engine or bench model in a build without Core ML (no stage at
        // all), or a bench run without --coreml-*.
        shape_model m;
        m.model.vocoder_coreml_managed = true;
        const outcome o = forward(m.model, 0);
        check(reached_ggml(o), "a managed model without a Core ML stage ignores the variable: " + describe(o));
    }
    {
        // A stage whose resolved policy is not strict.
        shape_model m;
        auto stage = std::make_shared<fake_pending>(m.samples_per_frame(), acquire_mode::pending, false);
        m.model.vocoder_coreml_managed = true;
        m.model.vocoder_pending = stage;
        const outcome o = forward(m.model, 0);
        check(reached_ggml(o), "a stage that is not strict ignores the variable: " + describe(o));
    }
#ifdef TTS_CPP_USE_COREML
    {
        // The sidecar attached at load keeps its legacy switch.
        shape_model m;
        const outcome o = forward(m.model, 0);
        check(!o.ok && o.error.find("SUPERTONIC_COREML_STRICT") != std::string::npos,
              "a model without the Core ML runtime and no sidecar attached at load still refuses: " + describe(o));
    }
#else
    {
        shape_model m;
        const outcome o = forward(m.model, 0);
        check(reached_ggml(o), "a build without Core ML ignores the variable: " + describe(o));
    }
#endif
}

// The real stage, on routes that resolve to ggml while the variable is set:
// each installs itself, is not strict, and leaves every call to ggml.
void stage_route_gates() {
    std::fprintf(stderr, "the stage on a none route\n");
    scoped_env strict("SUPERTONIC_COREML_STRICT", "1");
    struct route_case {
        const char * what;
        tts_cpp::CoremlOptions::Use use;
        bool memory_degrade;
    };
    const route_case cases[] = {
        {"use = off", tts_cpp::CoremlOptions::Use::off, false},
        {"use = automatic without a cache directory", tts_cpp::CoremlOptions::Use::automatic, false},
        {"memory-degrade (use unset)", tts_cpp::CoremlOptions::Use::unset, true},
    };
    for (const route_case & c : cases) {
        shape_model m;
        m.model.vocoder_coreml_managed = true;
        tts_cpp::CoremlOptions o;
        o.use = c.use;
        std::shared_ptr<supertonic_vocoder_coreml> stage;
        std::string threw;
        try {
            stage = supertonic_vocoder_coreml::create(m.model, "no-such-model.gguf", o, c.memory_degrade, "Apple M2");
        } catch (const std::exception & e) {
            threw = e.what();
        }
#ifdef TTS_CPP_USE_COREML
        check(threw.empty() && stage && stage->route().route == "none" && !stage->strict() &&
                  m.model.vocoder_pending == stage,
              std::string(c.what) + ": constructs, not strict, installed on the model" +
                  (threw.empty() ? "" : " (threw: " + threw + ")"));
#else
        check(threw.empty() && !stage, std::string(c.what) + ": a build without Core ML has no stage");
#endif
        const outcome out = forward(m.model, 0);
        check(reached_ggml(out), std::string(c.what) + ": the call runs on ggml: " + describe(out));
    }
}

// Blocking Core ML waits and cancel(), as Engine::coreml_wait uses them:
// each waiter reads the cancel count when it starts. Waiter A is asleep
// between two polls while cancel() runs and waiter B starts; A must still end
// on that cancel (a wait that starts later cannot take it back), and B, which
// started after it, waits for its own timeout.
void wait_cancel_gates() {
    std::fprintf(stderr, "blocking waits and cancel()\n");
    using clock = std::chrono::steady_clock;
    std::atomic<uint64_t> cancels{0};
    std::atomic<int> a_polls{0};
    std::atomic<bool> b_started{false};
    bool a_ready = true, b_ready = true;
    double a_ms = 0.0, b_ms = 0.0;
    auto ms_since = [](clock::time_point t0) {
        return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    };
    std::thread a([&] {
        const auto t0 = clock::now();
        const uint64_t since = cancels.load();
        a_ready = supertonic_wait_until_ready(
            [&] {
                if (a_polls.fetch_add(1) == 1) {
                    while (!b_started.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                // Ready after 3 s: reached only if the cancel was lost.
                return ms_since(t0) > 3000.0 ? 1 : 0;
            },
            -1, &cancels, since);
        a_ms = ms_since(t0);
    });
    while (a_polls.load() < 2) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    cancels.fetch_add(1);  // Engine::cancel()
    std::thread b([&] {
        const auto t0 = clock::now();
        const uint64_t since = cancels.load();
        b_started.store(true);
        b_ready = supertonic_wait_until_ready([] { return 0; }, 100, &cancels, since);
        b_ms = ms_since(t0);
    });
    a.join();
    b.join();
    check(!a_ready && a_ms < 1000.0,
          "a wait asleep across cancel() and a later wait's start ends on that cancel (" + std::to_string(a_ms) +
              " ms)");
    check(!b_ready && b_ms >= 100.0,
          "a wait started after cancel() waits for its own timeout (" + std::to_string(b_ms) + " ms)");

    // A cancel after both waits started ends both.
    std::atomic<int> started{0};
    bool r1 = true, r2 = true;
    auto waiter = [&](bool & r) {
        const uint64_t since = cancels.load();
        started.fetch_add(1);
        r = supertonic_wait_until_ready([] { return 0; }, 3000, &cancels, since);
    };
    const auto t0 = clock::now();
    std::thread w1(waiter, std::ref(r1)), w2(waiter, std::ref(r2));
    while (started.load() < 2) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    cancels.fetch_add(1);
    w1.join();
    w2.join();
    check(!r1 && !r2 && ms_since(t0) < 1000.0, "one cancel() ends every wait that started before it");
    check(supertonic_wait_until_ready([] { return 1; }, 0, &cancels, cancels.load()) &&
              !supertonic_wait_until_ready([] { return -1; }, -1, &cancels, cancels.load()),
          "ready returns true, a slot nothing will move returns false at once");
}

}  // namespace

int main() {
    unsetenv("SUPERTONIC_COREML_DISABLE");
    unsetenv("SUPERTONIC_COREML_STRICT");
    unsetenv("SUPERTONIC_COREML_COMPUTE_UNITS");
    unsetenv("QVAC_COREML_CACHE_DIR");
    fake_stage_gates();
    strict_environment_gates();
    stage_route_gates();
    wait_cancel_gates();
    std::fprintf(stderr, g_failures ? "FAILED (%d)\n" : "OK\n", g_failures);
    return g_failures ? 1 : 0;
}
