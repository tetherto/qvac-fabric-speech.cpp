// Core ML codec sidecar: parity against the ggml synthesis through the
// production dispatch, the stream that overlaps it with generation, the public
// load-status and per-call backend reports, and the fallbacks (no sidecar,
// invalid sidecar, sidecar that loads but cannot serve). Skips (77) without TTS_CPP_COREML, a staged
// AUDIO8_COREML_TEST_MODELS_DIR, or a compiled sidecar next to the decoder
// GGUF there. Optional: AUDIO8_COREML_FALLBACK_MODELS_DIR (a sidecar whose
// window cannot carry the causal context) and AUDIO8_COREML_TEST_LM_GGUF (a
// real language model, for a synthesis through the public Engine).

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
    std::fprintf(stderr, "[coreml-parity] built without TTS_CPP_COREML; skipping\n");
    return SKIP_EXIT_CODE;
}

#else

#include "audio8/coreml_path.h"
#include "audio8/coreml_windows.h"
#include "audio8/internal.h"
#include "audio8/window_stream.h"
#include "backend_util.h"
#include "gguf.h"
#include "test_env_portable.h"
#include "tiny_lm.h"
#include "memory_probe.h"
#include "nlohmann/json.hpp"
#include "qvac_coreml/generation.h"
#include "qvac_coreml/manager.h"
#include "tts-cpp/audio8/engine.h"
#include "tts-cpp/audio8/fit.h"

#include <chrono>
#include <filesystem>
#include <future>
#include <fstream>
#include <memory>
#include <thread>

using namespace tts_cpp::audio8::detail;
namespace fs = std::filesystem;
using tts_cpp::CoremlOptions;
namespace pcm = QVAC_COREML_NS;

namespace {

constexpr double MIN_COSINE   = 0.999;
// Streamed and batch synthesis both run on the sidecar; they differ only in
// the rounding of post passes over different spans.
constexpr double STREAM_MIN_COSINE = 0.99999;
constexpr int    SHORT_FRAMES = 10;
constexpr int    LONG_FRAMES  = 200;
constexpr int    N_THREADS    = 4;

int g_failures = 0;

void fail(const std::string & what) {
    std::fprintf(stderr, "[coreml-parity] FAIL: %s\n", what.c_str());
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

std::string find_decoder_gguf(const std::string & dir) {
    const char * candidates[] = {"audio8-codec-decoder-f32.gguf", "audio8-codec-decoder-f16.gguf",
                                 "audio8-codec-decoder-q8_0.gguf", "audio8-codec-decoder.gguf"};
    for (const char * name : candidates) {
        const std::string path = dir + "/" + name;
        if (fs::exists(path)) return path;
    }
    return {};
}

std::vector<int32_t> make_codes(const codec_hparams & hp, int n_frames, uint32_t seed) {
    std::vector<int32_t> codes(static_cast<size_t>(hp.num_codebooks) * n_frames);
    uint32_t state = seed;
    for (int book = 0; book < hp.num_codebooks; ++book) {
        const int limit = book == 0 ? hp.semantic_codebook_size : hp.residual_codebook_size;
        for (int frame = 0; frame < n_frames; ++frame) {
            state = state * 1664525u + 1013904223u;
            codes[static_cast<size_t>(book) * n_frames + frame] =
                static_cast<int32_t>((state >> 8) % static_cast<uint32_t>(limit));
        }
    }
    return codes;
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

double max_abs_err(const std::vector<float> & a, const std::vector<float> & b) {
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i) worst = std::max(worst, (double) std::fabs(a[i] - b[i]));
    return worst;
}

struct owned_codec {
    codec_model model;
    bool loaded = false;
    ~owned_codec() { free_codec(model); }
};

bool load(const std::string & gguf, owned_codec & codec,
          CoremlOptions options = {}) {
    options.wait = CoremlOptions::Wait::construct;
    std::string error;
    codec.loaded = load_codec(gguf, /*n_gpu_layers=*/0, /*backend=*/"",
                              codec.model, &error, options);
    if (!codec.loaded) fail("load_codec(" + gguf + "): " + error);
    return codec.loaded;
}

bool decode(codec_model & model, const std::vector<int32_t> & codes,
            int n_frames, const cancel_hook & cancel, std::vector<float> & pcm,
            decode_timing & timing, std::string & error,
            const codec_coreml_call * call = nullptr) {
    return decode_codes(model, codes.data(), n_frames, N_THREADS, cancel, pcm,
                        &error, nullptr, &timing, call);
}

bool check_parity(const char * tag, codec_model & ggml, codec_model & coreml, int n_frames) {
    const std::vector<int32_t> codes = make_codes(ggml.hp, n_frames, 0x2468ace1u);
    std::vector<float> pcm_ggml, pcm_coreml;
    decode_timing t_ggml, t_coreml;
    std::string error;
    {
        scoped_env disabled("AUDIO8_COREML_DISABLE", "1");
        if (!decode(ggml, codes, n_frames, nullptr, pcm_ggml, t_ggml, error)) {
            fail(std::string(tag) + ": ggml decode: " + error);
            return false;
        }
    }
    {
        scoped_env strict("AUDIO8_COREML_STRICT", "1");
        if (!decode(coreml, codes, n_frames, nullptr, pcm_coreml, t_coreml, error)) {
            fail(std::string(tag) + ": Core ML decode: " + error);
            return false;
        }
    }
    expect(t_ggml.synthesis_backend == "ggml" && t_coreml.synthesis_backend.rfind("coreml", 0) == 0,
           std::string(tag) + ": backends were '" + t_ggml.synthesis_backend + "' and '" +
               t_coreml.synthesis_backend + "'");
    const size_t want = static_cast<size_t>(n_frames) * ggml.hp.frame_size;
    if (pcm_ggml.size() != want || pcm_coreml.size() != want) {
        fail(std::string(tag) + ": sizes ggml=" + std::to_string(pcm_ggml.size()) +
             " coreml=" + std::to_string(pcm_coreml.size()) + " want=" + std::to_string(want));
        return false;
    }
    const double cos = cosine(pcm_ggml, pcm_coreml);
    std::fprintf(stderr,
                 "[coreml-parity] %s: frames=%d window=%d samples=%zu cosine=%.7f max_abs=%.3e "
                 "(min cosine %.4f, %s)\n",
                 tag, n_frames, t_coreml.block_frames, want, cos, max_abs_err(pcm_ggml, pcm_coreml),
                 MIN_COSINE, t_coreml.synthesis_backend.c_str());
    expect(cos >= MIN_COSINE, std::string(tag) + ": cosine below the gate");
    return cos >= MIN_COSINE;
}

void check_route_policy(const std::string & gguf) {
    CoremlOptions options;
    options.use = CoremlOptions::Use::require;
    owned_codec disabled_codec;
    {
        scoped_env disabled("AUDIO8_COREML_DISABLE", "1");
        if (!load(gguf, disabled_codec, options))
            return;
    }
    expect(!disabled_codec.model.coreml->wait(0),
           "disable must override require");
    const auto call = disabled_codec.model.coreml->acquire(nullptr);
    expect(!call.model && !call.pending && !call.strict,
           "disable must latch ggml");
    expect(disabled_codec.model.coreml->status().find(
               "\"state\":\"disabled\"") != std::string::npos,
           "disabled status was not published");

    options.use = CoremlOptions::Use::automatic;
    options.generate = CoremlOptions::Generate::never;
    options.compute_units = "cpu_and_ane";
    options.prewarm = CoremlOptions::Prewarm::none;
    owned_codec idle;
    if (!load(gguf, idle, options))
        return;
    const auto status = idle.model.coreml->status();
    expect(!idle.model.coreml->wait(0) &&
               status.find("idle") != std::string::npos,
           "prewarm none must leave the slot idle");
    expect(status.find("cpu_and_ane") != std::string::npos,
           "the explicit compute units were not reflected in status");
}

void check_cancellation(codec_model & coreml) {
    const std::vector<int32_t> codes = make_codes(coreml.hp, LONG_FRAMES, 99u);
    int asked = 0;
    const cancel_hook after_one = [&asked] { return asked++ >= 1; };
    std::vector<float> pcm;
    decode_timing timing;
    std::string error;
    scoped_env strict("AUDIO8_COREML_STRICT", "1");
    const bool ran = decode(coreml, codes, LONG_FRAMES, after_one, pcm, timing, error);
    expect(!ran && error == CANCELLED, "cancel " + std::string(ran ? "was ignored" : "stopped with '" + error + "'"));
    const int window = coreml.coreml->acquire(nullptr).model->window();
    const size_t one_window = static_cast<size_t>(std::min(window, LONG_FRAMES)) * coreml.hp.frame_size;
    expect(pcm.size() == one_window, "cancel kept " + std::to_string(pcm.size()) +
                                         " samples, expected one window of " + std::to_string(one_window));
    if (g_failures == 0) std::fprintf(stderr, "[coreml-parity] cancel: one window kept (%zu samples)\n", pcm.size());
}

void place_frame(const std::vector<int32_t> & rows, int books, int n_frames, int frame,
                 std::vector<int32_t> & frames) {
    for (int book = 0; book < books; ++book) {
        frames[static_cast<size_t>(frame) * books + book] =
            rows[static_cast<size_t>(book) * n_frames + frame];
    }
}

// Codebook-major codes back to the frame-major order the language model emits.
std::vector<int32_t> frame_major(const std::vector<int32_t> & rows, int books, int n_frames) {
    std::vector<int32_t> frames(rows.size());
    for (int frame = 0; frame < n_frames; ++frame) place_frame(rows, books, n_frames, frame, frames);
    return frames;
}

bool stream_frames(window_stream & stream, int n_frames, std::string & error) {
    for (int frame = 1; frame <= n_frames; ++frame) {
        if (!stream.advance(frame, &error)) return false;
    }
    return true;
}

int leading_windows(codec_model & model, int n_frames) {
    const int window = model.coreml->acquire(nullptr).model->window();
    return static_cast<int>(
               plan_coreml_windows(n_frames, window, synthesis_context_frames(model)).size()) -
           1;
}

// The engine's path: frames offered one at a time while they are generated,
// the rest after the last one. It has to stream every window that becomes
// final early and stitch the waveform the batch windows produce.
void check_streamed(codec_model & coreml, int n_frames) {
    const std::vector<int32_t> rows = make_codes(coreml.hp, n_frames, 0x13579bdfu);
    const std::vector<int32_t> frames = frame_major(rows, coreml.hp.num_codebooks, n_frames);
    std::vector<float> batch, streamed;
    decode_timing t_batch, t_stream;
    std::string error;
    expect(decode(coreml, rows, n_frames, nullptr, batch, t_batch, error),
           "batch decode failed: " + error);
    const auto call = coreml.coreml->acquire(nullptr);
    const std::unique_ptr<window_stream> stream =
        open_synthesis_stream(coreml, call, frames, N_THREADS, nullptr);
    if (!stream) {
        fail("no synthesis stream opened although a sidecar is attached");
        return;
    }
    expect(stream_frames(*stream, n_frames, error), "streamed advance failed: " + error);
    expect(finish_synthesis_stream(coreml, call, *stream, rows.data(), n_frames,
                                   N_THREADS, nullptr, streamed, &error,
                                   &t_stream),
           "streamed finish failed: " + error);
    expect(t_stream.streamed_windows == leading_windows(coreml, n_frames),
           "streamed " + std::to_string(t_stream.streamed_windows) + " windows, expected " +
               std::to_string(leading_windows(coreml, n_frames)));
    expect(t_stream.synthesis_backend.rfind("coreml", 0) == 0,
           "streamed synthesis ran on " + t_stream.synthesis_backend);
    expect(streamed.size() == batch.size(), "streamed and batch lengths differ");
    if (streamed.size() != batch.size()) return;
    const double cos = cosine(streamed, batch);
    std::fprintf(stderr, "[coreml-parity] streamed: frames=%d windows during generation=%d "
                         "cosine=%.7f max_abs=%.3e against batch (min cosine %.5f)\n",
                 n_frames, t_stream.streamed_windows, cos, max_abs_err(streamed, batch),
                 STREAM_MIN_COSINE);
    expect(cos >= STREAM_MIN_COSINE, "streamed: cosine below the gate");
}

void check_streamed_cancel(codec_model & coreml) {
    const std::vector<int32_t> rows = make_codes(coreml.hp, LONG_FRAMES, 11u);
    const std::vector<int32_t> frames = frame_major(rows, coreml.hp.num_codebooks, LONG_FRAMES);
    const auto call = coreml.coreml->acquire(nullptr);
    const std::unique_ptr<window_stream> stream = open_synthesis_stream(
        coreml, call, frames, N_THREADS, [] { return true; });
    if (!stream) {
        fail("no synthesis stream opened for the cancel check");
        return;
    }
    std::vector<float> pcm;
    std::string error;
    stream_frames(*stream, LONG_FRAMES, error);
    const bool ran =
        finish_synthesis_stream(coreml, call, *stream, rows.data(), LONG_FRAMES,
                                N_THREADS, nullptr, pcm, &error, nullptr);
    expect(!ran && error == CANCELLED,
           "streamed cancel " + std::string(ran ? "was ignored" : "stopped with '" + error + "'"));
}

// A language model the public Engine accepts next to this decoder: the two
// GGUFs must agree on the codebook count and the semantic codebook size.
std::string write_matching_tiny_lm(const codec_hparams & hp,
                                   int context = 384) {
    audio8_test::tiny_lm p;
    p.num_codebooks = hp.num_codebooks;
    p.codebook_size = hp.semantic_codebook_size;
    p.max_seq_len = context;
    p.ascii_tokenizer = true;
    p.semantic_end = p.semantic_begin + p.codebook_size - 1;
    p.eos = p.semantic_end + 1;
    p.vocab = p.eos + 1;
    return audio8_test::write_tiny_lm_gguf(
        p, test_tmpdir() + "/test-audio8-coreml-tiny-lm-" + test_process_tag() +
               "-" + std::to_string(context) + ".gguf");
}

tts_cpp::audio8::EngineOptions engine_options(const std::string & lm, const std::string & decoder) {
    tts_cpp::audio8::EngineOptions opts;
    opts.lm_gguf_path            = lm;
    opts.codec_decoder_gguf_path = decoder;
    opts.n_threads               = N_THREADS;
    opts.greedy                  = true;
    opts.max_frames              = SHORT_FRAMES;
    opts.coreml.wait = CoremlOptions::Wait::construct;
    return opts;
}

void check_public_load_status(const std::string & lm, const std::string & decoder, bool want_loaded,
                              const char * tag) {
    try {
        tts_cpp::audio8::Engine engine(engine_options(lm, decoder));
        expect(engine.codec_on_coreml() == want_loaded,
               std::string(tag) + ": Engine::codec_on_coreml() = " +
                   (engine.codec_on_coreml() ? "true" : "false") + ", expected " +
                   (want_loaded ? "true" : "false"));
        expect(engine.backend_name() == "CPU",
               std::string(tag) + ": backend_name() should stay the ggml backend, got " + engine.backend_name());
    } catch (const std::exception & e) {
        fail(std::string(tag) + ": Engine construction threw: " + e.what());
    }
}

void check_public_synthesis_backend(const std::string & lm, const std::string & decoder,
                                    const char * want_prefix, bool want_attached_after,
                                    const char * tag) {
    try {
        tts_cpp::audio8::Engine engine(engine_options(lm, decoder));
        expect(engine.codec_on_coreml(), std::string(tag) + ": sidecar not attached at load");
        const tts_cpp::audio8::SynthesisResult result = engine.synthesize("Hello from the parity test.");
        expect(result.codec_synthesis_backend.rfind(want_prefix, 0) == 0,
               std::string(tag) + ": codec_synthesis_backend = '" + result.codec_synthesis_backend +
                   "', expected prefix '" + want_prefix + "'");
        expect(engine.codec_on_coreml() == want_attached_after,
               std::string(tag) + ": codec_on_coreml() after synthesis = " +
                   (engine.codec_on_coreml() ? "true" : "false"));
        std::fprintf(stderr, "[coreml-parity] %s: %d frames, codec synthesis on %s (sidecar attached after: %s)\n",
                     tag, result.frames, result.codec_synthesis_backend.c_str(),
                     engine.codec_on_coreml() ? "yes" : "no");
    } catch (const std::exception & e) {
        fail(std::string(tag) + ": synthesis threw: " + e.what());
    }
}

// The staged decoder next to a sidecar directory that is not a model.
std::string stage_invalid_sidecar(const std::string & gguf) {
    const fs::path dir = fs::path(test_tmpdir()) / ("test-audio8-coreml-invalid-" + test_process_tag());
    fs::remove_all(dir);
    fs::create_directories(dir);
    const fs::path link = dir / fs::path(gguf).filename();
    fs::create_symlink(fs::absolute(gguf), link);
    const fs::path bogus = fs::path(coreml_codec_sidecar_path(link.string()));
    fs::create_directories(bogus);
    std::ofstream(bogus / "coremldata.bin") << "not a model";
    return link.string();
}

void check_invalid_sidecar(const std::string & gguf, const std::string & lm) {
    const std::string staged = stage_invalid_sidecar(gguf);
    owned_codec codec;
    if (!load(staged, codec)) return;
    expect(!codec.model.coreml->wait(0), "an invalid sidecar was attached");
    const std::vector<int32_t> codes = make_codes(codec.model.hp, SHORT_FRAMES, 3u);
    std::vector<float> pcm;
    decode_timing timing;
    std::string error;
    expect(decode(codec.model, codes, SHORT_FRAMES, nullptr, pcm, timing, error),
           "decode next to an invalid sidecar failed: " + error);
    expect(timing.synthesis_backend == "ggml", "invalid sidecar: synthesis ran on " + timing.synthesis_backend);
    check_public_load_status(lm, staged, /*want_loaded=*/false, "invalid sidecar");
    fs::remove_all(fs::path(staged).parent_path());
}

// A sidecar that loads but cannot serve any utterance (window <= causal
// context): fails under STRICT, otherwise falls back to ggml bit-exactly and is
// retired, so the next call goes straight to ggml with no sidecar attached.
void check_unservable_sidecar(const std::string & dir, const std::string & lm) {
    const std::string gguf = find_decoder_gguf(dir);
    if (gguf.empty() || !fs::exists(coreml_codec_sidecar_path(gguf))) {
        fail("AUDIO8_COREML_FALLBACK_MODELS_DIR has no decoder GGUF + sidecar");
        return;
    }
    owned_codec codec;
    if (!load(gguf, codec)) return;
    expect(codec.model.coreml->wait(0),
           "the fallback fixture's sidecar did not load");
    const auto call = codec.model.coreml->acquire(nullptr);
    if (!call.model)
        return;
    const int window = call.model->window();
    const int context = synthesis_context_frames(codec.model);
    expect(window <= context, "the fallback fixture's window (" + std::to_string(window) +
                                  ") carries the context (" + std::to_string(context) + "); it would serve");

    const std::vector<int32_t> codes = make_codes(codec.model.hp, SHORT_FRAMES, 5u);
    const std::vector<int32_t> no_frames;
    expect(!open_synthesis_stream(codec.model, call, no_frames, N_THREADS,
                                  nullptr),
           "a sidecar that cannot carry the context opened a synthesis stream");
    std::string error;
    check_public_load_status(lm, gguf, /*want_loaded=*/true,
                             "unservable sidecar");
    try {
        auto opts = engine_options(lm, gguf);
        opts.coreml.use = CoremlOptions::Use::require;
        tts_cpp::audio8::Engine engine(opts);
        (void)engine.synthesize("ab");
        fail("strict public synthesis accepted an unservable sidecar");
    } catch (const std::runtime_error & e) {
        expect(std::string(e.what()).find(
                   "required Core ML synthesis unavailable") !=
                   std::string::npos,
               std::string("strict public synthesis reported: ") + e.what());
    }
    std::vector<float> pcm_fallback, pcm_ggml, pcm_retired;
    decode_timing t_fallback, t_ggml, t_retired;
    expect(decode(codec.model, codes, SHORT_FRAMES, nullptr, pcm_fallback,
                  t_fallback, error, &call),
           "fallback decode failed: " + error);
    expect(t_fallback.synthesis_backend == "ggml",
           "fallback synthesis ran on " + t_fallback.synthesis_backend);
    expect(!codec.model.coreml->wait(0),
           "the failed sidecar was not retired after the fallback");
    expect(decode(codec.model, codes, SHORT_FRAMES, nullptr, pcm_retired, t_retired, error) &&
               t_retired.synthesis_backend == "ggml",
           "decode after retirement did not go straight to ggml: " + error);
    {
        scoped_env disabled("AUDIO8_COREML_DISABLE", "1");
        expect(decode(codec.model, codes, SHORT_FRAMES, nullptr, pcm_ggml, t_ggml, error),
               "reference decode failed: " + error);
    }
    expect(pcm_fallback == pcm_ggml && pcm_retired == pcm_ggml,
           "fallback outputs are not bit-identical to the ggml decode");
    std::fprintf(stderr, "[coreml-parity] unservable sidecar (window %d <= context %d): loaded, fell back to ggml, retired\n",
                 window, context);
}

using json = nlohmann::json;

unsigned long long trace_cast(uint64_t value) {
    return static_cast<unsigned long long>(value);
}
using tts_cpp::audio8::Engine;
using tts_cpp::audio8::fit_params;
using tts_cpp::audio8::FitOptions;

json stage(const std::string & status) {
    return json::parse(status).at("stages").at(0);
}

template <class Predicate> bool poll_until(Predicate pred) {
    using namespace std::chrono;
    const auto deadline = steady_clock::now() + seconds(5);
    while (!pred()) {
        if (steady_clock::now() >= deadline)
            return false;
        std::this_thread::sleep_for(milliseconds(5));
    }
    return true;
}

std::atomic<bool> gate_started{false}, gate_released{false};

pcm::LoadResult hold_scheduler(const pcm::LoadRequest &) {
    using namespace std::chrono;
    gate_started = true;
    const auto deadline = steady_clock::now() + seconds(45);
    while (!gate_released && steady_clock::now() < deadline)
        std::this_thread::sleep_for(milliseconds(5));
    pcm::LoadResult result;
    result.model = std::make_shared<int>(1);
    return result;
}

void check_delayed_synthesis(const std::string & lm, const std::string & gguf) {
    using namespace std::chrono_literals;
    pcm::SlotSpec spec;
    spec.route = pcm::Route::prebuilt;
    spec.stage = "audio8-test-gate";
    spec.artifact_path = gguf + ".test-gate";
    spec.units = pcm::ComputeUnits::cpu_only;
    spec.load_fn = hold_scheduler;
    const auto gate = pcm::Manager::instance()->make_slot(spec, {});
    struct release_gate {
        ~release_gate() { gate_released = true; }
    } release;
    gate->request(pcm::Priority::now);
    if (!poll_until([] { return gate_started.load(); })) {
        fail("could not hold the Core ML scheduler gate");
        return;
    }

    auto opts = engine_options(lm, gguf);
    opts.coreml = {};
    Engine engine(opts);
    auto off = opts;
    off.coreml.use = CoremlOptions::Use::off;
    const auto reference = Engine(off).synthesize("ab");
    const auto pending = engine.synthesize("ab");
    expect(
        !gate_released && pending.coreml_pending &&
            pending.codec_synthesis_backend == "ggml" &&
            pending.codes == reference.codes && pending.pcm == reference.pcm,
        "default public synthesis did not return a complete pending ggml call");

    for (auto use :
         {CoremlOptions::Use::automatic, CoremlOptions::Use::require}) {
        auto options = opts.coreml;
        options.use = use;
        options.wait = CoremlOptions::Wait::call;
        options.prewarm = CoremlOptions::Prewarm::none;
        options.strict_wait_ms = 5000;
        owned_codec codec;
        std::string load_error;
        if (!load_codec(gguf, 0, "cpu", codec.model, &load_error, options)) {
            fail("delayed decoder load failed: " + load_error);
            return;
        }
        const auto codes = make_codes(codec.model.hp, SHORT_FRAMES, 37u);
        std::atomic<bool> cancel{false};
        auto result = std::async(std::launch::async, [&] {
            std::vector<float> pcm;
            std::string error;
            int blocks = 0;
            const bool ok = decode_codes(
                codec.model, codes.data(), SHORT_FRAMES, N_THREADS,
                [&] {
                    ++blocks;
                    return false;
                },
                pcm, &error, nullptr, nullptr, nullptr, &cancel);
            return !ok && error == CANCELLED && pcm.empty() && blocks == 0;
        });
        expect(poll_until([&] {
                   return stage(codec.model.coreml->status()).at("state") ==
                          "queued";
               }),
               "direct batch did not wait behind the gate");
        expect(result.wait_for(0ms) == std::future_status::timeout,
               "direct batch returned before cancellation");
        cancel = true;
        if (result.wait_for(2s) != std::future_status::ready) {
            fail("direct batch did not observe cancellation while waiting");
            gate_released = true;
            return;
        }
        expect(result.get(),
               "direct batch cancellation touched the block hook or PCM");
    }

    auto strict = opts;
    strict.coreml.use = CoremlOptions::Use::require;
    strict.coreml.prewarm = CoremlOptions::Prewarm::none;
    strict.coreml.strict_wait_ms = 5000;
    Engine prewarming(strict);
    auto prewarm = std::async(std::launch::async, [&] {
        return json::parse(
            prewarming.coreml_prewarm(CoremlOptions::Prewarm::all, true));
    });
    expect(poll_until([&] {
               return stage(prewarming.coreml_status()).at("state") == "queued";
           }),
           "public prewarm did not wait behind the gate");
    expect(prewarm.wait_for(0ms) == std::future_status::timeout,
           "public prewarm returned before cancellation");
    prewarming.cancel();
    if (prewarm.wait_for(2s) != std::future_status::ready) {
        fail("Engine::cancel did not interrupt the public prewarm wait");
        gate_released = true;
        return;
    }
    const auto prewarm_result = prewarm.get();
    expect(prewarm_result.at("success") == false &&
               prewarm_result.at("nothing_to_prepare") == false,
           "cancelled public prewarm returned an unexpected group result");
    Engine required(strict);
    auto result = std::async(std::launch::async, [&] {
        try {
            (void)required.synthesize("ab");
        } catch (const std::runtime_error & e) {
            return std::string(e.what());
        }
        return std::string{};
    });
    expect(poll_until([&] {
               return stage(required.coreml_status()).at("state") == "queued";
           }),
           "strict Engine did not wait behind the gate");
    expect(result.wait_for(0ms) == std::future_status::timeout,
           "strict Engine returned before cancellation");
    required.cancel();
    if (result.wait_for(2s) != std::future_status::ready) {
        fail("Engine::cancel did not interrupt strict acquisition");
        gate_released = true;
        return;
    }
    expect(result.get() == CANCELLED,
           "strict Engine did not report cancellation");
    expect(gate->state() == pcm::SlotState::loading,
           "the scheduler gate released before cancellation was observed");
    gate_released = true;
    if (!engine.coreml_wait(60000)) {
        fail("default public slot did not become ready after releasing the "
             "gate");
        return;
    }
    const auto automatic_ready = engine.synthesize("ab");
    const auto strict_ready = required.synthesize("ab");
    for (const auto * audio : {&automatic_ready, &strict_ready}) {
        expect(!audio->coreml_pending &&
                   audio->codec_synthesis_backend == "coreml-all" &&
                   audio->codes == reference.codes &&
                   audio->pcm.size() == reference.pcm.size() &&
                   std::all_of(audio->pcm.begin(), audio->pcm.end(),
                               [](float value) { return std::isfinite(value); }),
               "ready public call did not produce complete Core ML audio");
    }
    expect(automatic_ready.pcm == strict_ready.pcm,
           "default and required ready routes returned different audio");
    std::printf("public and direct delayed acquisition, cancellation, and "
                "ready routing tested\n");
}

FitOptions fit_options(const std::string & lm, const std::string & gguf,
                       const CoremlOptions & options) {
    FitOptions fit;
    fit.lm_gguf_path = lm;
    fit.codec_decoder_gguf_path = gguf;
    fit.max_frames = LONG_FRAMES;
    fit.coreml = options;
    return fit;
}

void check_short_context_prebuilt(const std::string & gguf,
                                  const codec_hparams & hp) {
    const auto lm = write_matching_tiny_lm(hp, 127);
    auto opts = engine_options(lm, gguf);
    auto fit = fit_options(lm, gguf, opts.coreml);
    fit.max_frames = opts.max_frames;
    const auto too_long = fit_params(fit);
    expect(too_long.status == tts_cpp::FitStatus::Error &&
               too_long.reason == "workload-too-large",
           "the default fit prompt fit the short context fixture");
    fit.prompt_tokens = 125;
    const auto sized = fit_params(fit);
    expect(sized.fits && sized.coreml.route == "prebuilt" &&
               sized.coreml.bytes == 0,
           "the prebuilt route changed its static fit");
    opts.coreml.memory_budget_bytes = 1LL << 30;
    const auto result = Engine(opts).synthesize("ab");
    expect(result.frames > 0 && !result.pcm.empty() &&
               result.codec_synthesis_backend == "coreml-all",
           "budgeted prebuilt Engine did not synthesize with a short context");
    fs::remove(lm);
}

void check_source_changes(const std::string & gguf, const fs::path & work,
                          CoremlOptions options) {
    options.wait = CoremlOptions::Wait::call;
    for (bool replace : {false, true}) {
        const auto dir = work / (replace ? "replacement" : "mutation");
        fs::create_directories(dir);
        const auto local = dir / fs::path(gguf).filename();
        fs::copy_file(gguf, local);
        options.cache_dir = (dir / "cache").string();
        options.use = replace ? CoremlOptions::Use::require
                              : CoremlOptions::Use::automatic;
        owned_codec codec;
        std::string error;
        int observations = 0;
        const auto change_during_upload = [&] {
            ++observations;
            if (replace) {
                fs::rename(local, dir / "retained.gguf");
                std::unique_ptr<gguf_context, decltype(&gguf_free)> replacement(
                    gguf_init_empty(), gguf_free);
                if (!replacement ||
                    !gguf_write_to_file(replacement.get(), local.c_str(), false))
                    throw std::runtime_error("cannot write replacement fixture");
                return;
            }
            std::unique_ptr<gguf_context, decltype(&gguf_free)> header(
                gguf_init_from_file(local.c_str(), {true, nullptr}), gguf_free);
            if (!header)
                throw std::runtime_error("cannot read mutation fixture");
            const auto offset =
                static_cast<std::streamoff>(gguf_get_data_offset(header.get()));
            std::fstream file(local,
                              std::ios::binary | std::ios::in | std::ios::out);
            file.exceptions(std::ios::badbit | std::ios::failbit);
            file.seekg(offset);
            const char byte = static_cast<char>(file.get());
            file.seekp(offset);
            file.put(byte ^ 1);
            file.close();
            fs::last_write_time(local, fs::last_write_time(local) -
                                           std::chrono::seconds(2));
        };
        const bool loaded =
            load_codec(local.string(), 0, "cpu", codec.model, &error, options,
                       codec_budget::unprojected, change_during_upload);
        expect(loaded && observations == 1,
               "source-change fixture load: " + error);
        if (!loaded)
            continue;
        const auto codes = make_codes(codec.model.hp, SHORT_FRAMES, 99u);
        std::vector<float> audio;
        decode_timing timing;
        try {
            const bool ok = decode(codec.model, codes, SHORT_FRAMES, nullptr,
                                   audio, timing, error);
            expect(!replace && ok && timing.synthesis_backend == "ggml" &&
                       !audio.empty(),
                   "changed source did not preserve the automatic/strict "
                   "contract");
        } catch (const pcm::CoremlError & e) {
            expect(replace && e.state == pcm::SlotState::failed,
                   std::string("unexpected changed-source acquisition: ") +
                       e.what());
        }
        const auto status = stage(codec.model.coreml->status());
        const auto artifacts = dir / "cache" / "qvac-coreml-v1" / "artifacts";
        expect(status.dump().find("source_changed") != std::string::npos &&
                   status.at("label") == "ggml" &&
                   fs::is_directory(artifacts) && fs::is_empty(artifacts),
               "changed source could reach generation: " + status.dump());
        std::printf("source identity after %s: %s\n",
                    replace ? "pathname replacement"
                            : "in-place upload mutation",
                    status.at("state").get<std::string>().c_str());
    }
    fs::remove_all(work);
}

bool same_fit(const tts_cpp::FitResult & lhs, const tts_cpp::FitResult & rhs) {
    return lhs.status == rhs.status && lhs.coreml.route == rhs.coreml.route &&
           lhs.coreml.validated == rhs.coreml.validated &&
           lhs.coreml.bytes == rhs.coreml.bytes &&
           lhs.coreml.disabled_reason == rhs.coreml.disabled_reason &&
           lhs.coreml.ggml_only_fits == rhs.coreml.ggml_only_fits &&
           lhs.coreml.fits == rhs.coreml.fits;
}

uint64_t ggml_need(const tts_cpp::FitResult & fit,
                   const FitOptions & workload) {
    return fit.device.total_bytes + workload.margin_bytes +
           (fit.device_shares_host_memory ? fit.host_bytes : 0);
}

void check_fit_stable(const FitOptions & workload,
                      const tts_cpp::FitResult & cold, const fs::path & cache) {
    const int previous_failures = g_failures;
    expect(same_fit(cold, fit_params(workload)),
           "fit changed after generation");
    int hints = 0;
    for (const auto & file :
         fs::directory_iterator(cache / "qvac-coreml-v1" / "hints")) {
        if (file.path().extension() != ".json")
            continue;
        json record;
        std::ifstream(file.path()) >> record;
        // The runtime has already written a fresh hint. Change only its expiry.
        expect(record.at("stale_until").is_null(),
               "warm-load hint was already stale");
        record["stale_until"] = "2099-01-01T00:00:00Z";
        std::ofstream(file.path()) << record;
        ++hints;
    }
    expect(hints > 0, "generated route wrote no warm-load hint");
    expect(same_fit(cold, fit_params(workload)),
           "fit changed with a stale hint");
    std::printf("test-audio8-fit-stable: %s, %d stale hint(s) tested\n",
                g_failures == previous_failures ? "PASS" : "FAIL", hints);
}

struct memory_trace {
    uint64_t whole = 0;
    uint64_t prepare = 0;
    uint64_t load = 0;
    uint64_t ready = 0;
    int prepare_overlap = 0;
    int load_overlap = 0;
    int missing = 0;
    std::vector<qvac_coreml_memory::AneSample> compiler;
};

std::atomic<uint64_t> test_available_bytes{0};

void check_prebuilt_reserve_boundary(const fs::path & dir) {
    const auto bundle = dir / "codec.mlmodelc";
    fs::create_directories(bundle / "weights");
    const uint64_t missing = prebuilt_codec_ane_reserve(bundle.string());
    expect(missing >= (512ULL << 20),
           "missing prebuilt weights bypassed cold admission");
    if (!missing)
        return;

    const auto weights = bundle / "weights" / "weight.bin";
    std::ofstream(weights).close();
    expect(prebuilt_codec_ane_reserve(bundle.string()) == missing,
           "an empty prebuilt weight file bypassed cold admission");
    constexpr uint64_t WEIGHT_BYTES = 20ULL << 20;
    fs::resize_file(weights, WEIGHT_BYTES);
    const uint64_t known = prebuilt_codec_ane_reserve(bundle.string());
    expect(known == (32ULL << 20),
           "prebuilt reserve did not use the known weight size");
    fs::remove(weights);
    fs::create_directory(weights);
    expect(prebuilt_codec_ane_reserve(bundle.string()) == missing,
           "non-file prebuilt weight path did not get a conservative reserve");

    pcm::Config config;
    config.available_memory = [] { return test_available_bytes.load(); };
    config.pin_image = false;
    config.memory_recheck = std::chrono::milliseconds(10);
    const auto manager = pcm::Manager::create(config);
    pcm::SlotSpec spec;
    spec.route = pcm::Route::prebuilt;
    spec.stage = "audio8-test-reserve";
    spec.units = pcm::ComputeUnits::all;
    spec.ane_compile_reserve_bytes = missing;
    spec.artifact_path = bundle.string();
    spec.load_fn = [](const pcm::LoadRequest &) {
        pcm::LoadResult result;
        result.model = std::make_shared<int>(1);
        return result;
    };
    const auto slot = manager->make_slot(spec, {});
    test_available_bytes = missing - 1;
    slot->request(pcm::Priority::now);
    expect(poll_until([&] {
               return slot->state() == pcm::SlotState::deferred_memory;
           }) &&
               manager->stats().loads_started == 0,
           "cold prebuilt was admitted one byte below the reserve");
    test_available_bytes = missing;
    manager->notify();
    expect(poll_until([&] { return slot->state() == pcm::SlotState::ready; }) &&
               manager->stats().loads_started == 1,
           "cold prebuilt was not admitted exactly at the reserve");
    fs::remove_all(dir);
}

// Two concurrent things only: the existing codec call on the producer and
// the shared runtime's prepare/load job. The sampler never executes a model.
bool check_overlap(owned_codec & codec, bool generated, uint64_t budget,
                   uint64_t compiler_reserve) {
    using namespace std::chrono;
    if (!tts_cpp::detail::backend_is_metal(codec.model.backend)) {
        fail("the overlapped decoder did not select Metal");
        return false;
    }
    const auto device = ggml_backend_get_device(codec.model.backend);
    std::printf("Audio8 overlap backend=%s device=%s\n",
                ggml_backend_dev_name(device),
                ggml_backend_dev_description(device));
    const auto rows = make_codes(codec.model.hp, LONG_FRAMES, 0x49a2u);
    const codec_coreml_call ggml;
    std::vector<float> pcm;
    decode_timing timing;
    std::string error;
    for (int i = 0; i < 2; ++i) {
        if (!decode(codec.model, rows, LONG_FRAMES, nullptr, pcm, timing, error,
                    &ggml)) {
            fail("could not warm the ggml baseline: " + error);
            return false;
        }
    }
    const uint64_t base = qvac_coreml_memory::phys_footprint();
    if (base == 0) {
        fail("could not read the process footprint baseline");
        return false;
    }
    std::atomic<bool> stop{false}, inference{false};
    const auto stage_state = [&] {
        return stage(codec.model.coreml->status())
            .at("state")
            .get<std::string>();
    };
    const auto loading = [](const std::string & state) {
        return state == "loading" || state == "warming";
    };
    std::promise<void> baseline_sampled;
    auto baseline_ready = baseline_sampled.get_future();
    auto sampler = std::async(std::launch::async, [&] {
        memory_trace trace;
        uint64_t rss = 0;
        const bool initial = qvac_coreml_memory::ane_compiler_rss(rss);
        trace.compiler.push_back({rss, false, initial});
        baseline_sampled.set_value();
        unsigned tick = 0;
        while (!stop.load()) {
            const auto state = stage_state();
            const uint64_t total = qvac_coreml_memory::phys_footprint();
            if (total == 0)
                ++trace.missing;
            const uint64_t delta = total > base ? total - base : 0;
            trace.whole = std::max(trace.whole, delta);
            if (state == "digesting" || state == "generating" ||
                state == "compiling") {
                trace.prepare = std::max(trace.prepare, delta);
                if (inference.load())
                    ++trace.prepare_overlap;
            } else if (loading(state)) {
                trace.load = std::max(trace.load, delta);
                if (inference.load())
                    ++trace.load_overlap;
            } else if (state == "ready") {
                trace.ready = std::max(trace.ready, delta);
            }
            if (++tick % 4 == 0 && state != "ready") {
                const bool before = loading(state);
                const bool sampled = qvac_coreml_memory::ane_compiler_rss(rss);
                if (before == loading(stage_state()))
                    trace.compiler.push_back({rss, before, sampled});
            }
            std::this_thread::sleep_for(milliseconds(4));
        }
        return trace;
    });
    baseline_ready.wait();

    int pending_calls = 0, ready_calls = 0;
    const auto deadline = steady_clock::now() + minutes(3);
    try {
        while (steady_clock::now() < deadline && ready_calls < 2) {
            const auto call = codec.model.coreml->acquire(nullptr);
            if (!(call.model || call.pending)) {
                fail("generation ended before the slot became ready: " +
                     codec.model.coreml->status());
                break;
            }
            inference.store(true);
            const bool ok = decode(codec.model, rows, LONG_FRAMES, nullptr, pcm,
                                   timing, error, &call);
            inference.store(false);
            if (!ok || timing.synthesis_backend !=
                           (call.model ? call.model->label() : "ggml")) {
                fail("a synthesis switched routes or failed during overlap: " +
                     error);
                break;
            }
            if (call.pending)
                ++pending_calls;
            if (call.model)
                ++ready_calls;
        }
    } catch (const std::exception & e) {
        fail(e.what());
    }
    stop.store(true);
    const auto trace = sampler.get();
    const auto compiler =
        qvac_coreml_memory::ane_compiler_growth(trace.compiler);
    const bool ready = pending_calls > 0 && ready_calls == 2;
    expect(ready, "cold generation did not produce both pending ggml and ready "
                  "Core ML calls");
    expect(trace.missing == 0 && (!generated || trace.prepare_overlap > 0) &&
               trace.load_overlap > 0 && trace.ready > 0,
           "memory trace missed preparation or load overlapped with inference");
    const bool sampled = compiler.has_before && compiler.loading_samples >= 2 &&
                         compiler.failed_samples == 0;
    const auto baseline = qvac_coreml_memory::ane_compiler_baseline(compiler);
    const bool has_growth = compiler.peak > 0 && compiler.growth > 0;
    const bool clean_sample =
        sampled && has_growth &&
        baseline == qvac_coreml_memory::AneBaseline::clean;
    const bool reserve_covers_peak =
        compiler.peak > 0 && 1.25 * compiler.peak <= compiler_reserve;
    expect(
        sampled,
        "memory trace did not sample ANECompilerService reliably during load");
    expect(compiler_reserve > 0,
           "ANE-capable route had no runtime compiler reserve");
    if (generated)
        expect(trace.whole <= budget,
               "overlapped footprint exceeded the fit envelope");
    const char * require_clean = std::getenv("AUDIO8_REQUIRE_CLEAN_MEMORY");
    if (require_clean && std::string(require_clean) == "1") {
        expect(clean_sample, "clean ANE compiler growth could not be measured");
        expect(reserve_covers_peak,
               "ANE compiler peak plus margin exceeded its reserve");
    }
    std::printf(
        "audio8 memory bytes: baseline=%llu prepare=%llu load=%llu ready=%llu "
        "peak=%llu "
        "budget=%llu compiler_before=%llu compiler_peak=%llu "
        "compiler_growth=%llu "
        "compiler_reserve=%llu reserve_covers_peak_plus_margin=%s "
        "compiler_baseline=%s compiler_evidence=%s compiler_samples=%zu "
        "compiler_failures=%zu "
        "prepare_overlap=%d load_overlap=%d pending_calls=%d ready_calls=%d\n",
        trace_cast(base), trace_cast(trace.prepare), trace_cast(trace.load),
        trace_cast(trace.ready), trace_cast(trace.whole), trace_cast(budget),
        trace_cast(compiler.before), trace_cast(compiler.peak),
        trace_cast(compiler.growth), trace_cast(compiler_reserve),
        reserve_covers_peak ? "yes" : "no",
        qvac_coreml_memory::ane_baseline_name(baseline),
        clean_sample ? "clean-observation" : "unqualified",
        compiler.loading_samples, compiler.failed_samples,
        trace.prepare_overlap, trace.load_overlap, pending_calls, ready_calls);
    return ready;
}

void check_generated(const std::string & gguf, const fs::path & cache) {
    expect(!fs::exists(coreml_codec_sidecar_path(gguf)),
           "generated fixture must have no adjacent sidecar");
    expect(!fs::exists(cache) || fs::is_empty(cache),
           "generated fixture needs an empty cache");
    if (g_failures)
        return;
    fs::create_directories(cache);

    CoremlOptions options;
    options.cache_dir = cache.string();
    options.compute_units = "all";
    options.prewarm = CoremlOptions::Prewarm::none;
    options.wait = CoremlOptions::Wait::none;
    pcm::Options runtime_options;
    const auto candidates =
        pcm::generation::probe({"audio8-codec-w64"}, gguf,
                               pcm::resolve_coreml(runtime_options, "AUDIO8"));
    if (candidates.empty() || !candidates.front().matched) {
        fail("the generated fixture did not match the Audio8 template");
        return;
    }
    const auto reserve = pcm::generation::ane_compile_reserve_bytes(
        candidates.front().memory, pcm::ComputeUnits::all);

    owned_codec ggml;
    CoremlOptions disabled;
    disabled.use = CoremlOptions::Use::off;
    if (!load(gguf, ggml, disabled))
        return;
    const std::string lm = write_matching_tiny_lm(ggml.model.hp);
    const auto workload = fit_options(lm, gguf, options);
    const auto cold = fit_params(workload);
    expect(cold.fits && cold.coreml.route == "generate" &&
               cold.coreml.validated && cold.coreml.bytes > 0 &&
               cold.coreml.disabled_reason.empty(),
           "fit could not price the matched generated route: " + cold.report);
    if (!cold.coreml.bytes)
        return;

    // This fixture uses 64-frame windows and a 25% template margin. A
    // shorter workload drops one queued input and one retained PCM window.
    constexpr uint64_t window = 64;
    const int step = static_cast<int>(window) -
                     synthesis_context_frames(ggml.model);
    const uint64_t window_bytes =
        window * sizeof(float) * (ggml.model.hp.latent_dim +
                                   ggml.model.hp.frame_size);
    const auto shorter = fit_codec_coreml(
        gguf, options, 0, UINT64_MAX, LONG_FRAMES - step, ggml.model);
    expect(shorter.route == "generate" &&
               cold.coreml.bytes == shorter.bytes + window_bytes * 5 / 4,
           "generated stream buffers did not receive the template margin");

    auto near = workload;
    near.coreml.memory_budget_bytes =
        ggml_need(cold, workload) + cold.coreml.bytes / 2;
    const auto degraded = fit_params(near);
    expect(degraded.fits && degraded.coreml.ggml_only_fits &&
               degraded.coreml.disabled_reason == "memory",
           "near-budget fit did not degrade to ggml");

    auto opts = engine_options(lm, gguf);
    opts.max_frames = LONG_FRAMES;
    opts.coreml = near.coreml;
    {
        Engine degraded_engine(opts);
        const auto status = stage(degraded_engine.coreml_status());
        expect(status.at("state") == "disabled" &&
                   !degraded_engine.codec_on_coreml(),
               "construction did not apply the same near-budget decision");
    }

    {
        const auto dir = fs::path(lm).parent_path() / "budget-route-change";
        fs::create_directories(dir);
        const auto local = dir / "codec.gguf";
        fs::create_symlink(fs::absolute(gguf), local);
        const auto prebuilt = coreml_codec_sidecar_path(local.string());
        fs::create_directory(prebuilt);
        expect(!codec_needs_budget_projection(local.string(), near.coreml),
               "initial prebuilt unexpectedly needed a generation projection");
        owned_codec changed_route;
        std::string load_error;
        const bool loaded = load_codec(
            local.string(), 0, "cpu", changed_route.model, &load_error,
            near.coreml, codec_budget::unprojected, [&] {
                if (!fs::remove(prebuilt))
                    throw std::runtime_error("could not remove prebuilt fixture");
            });
        expect(loaded, "could not load changed-route fixture: " + load_error);
        if (loaded) {
            const auto status = stage(changed_route.model.coreml->status());
            expect(status.at("state") == "unsupported(unprojected_budget)",
                   "a budgeted route changed to generation without a projection");
        }
    }

    owned_codec generated;
    std::string error;
    // Retain a Metal fallback during generation and the load. Parity uses a
    // separate CPU load after this schedule.
    if (!load_codec(gguf, 99, "auto", generated.model, &error, options)) {
        fail("load generated fixture: " + error);
        return;
    }
    expect(!generated.model.coreml->wait(0),
           "prewarm none started work before demand");
    if (!check_overlap(generated, true, cold.coreml.bytes, reserve))
        return;
    const auto ready = stage(generated.model.coreml->status());
    expect(ready.at("route") == "generated" && ready.at("state") == "ready",
           "the empty cache did not produce a ready generated sidecar");

    owned_codec cpu;
    options.use = CoremlOptions::Use::require;
    if (!load(gguf, cpu, options))
        return;
    (void)cpu.model.coreml->acquire(nullptr);
    expect(stage(cpu.model.coreml->status()).at("route") == "cached",
           "second slot did not adopt the existing artifact");
    check_parity("generated short", ggml.model, cpu.model, SHORT_FRAMES);
    check_parity("generated long", ggml.model, cpu.model, LONG_FRAMES);
    check_streamed(cpu.model, LONG_FRAMES);

    opts.coreml = options;
    opts.max_frames = 70;
    {
        Engine engine(opts);
        const auto construction =
            json::parse(engine.coreml_construction_json());
        expect(construction.at("success") == true &&
                   construction.at("nothing_to_prepare") == true,
               "prewarm none did not report an empty construction group");
        const auto prewarm = json::parse(
            engine.coreml_prewarm(CoremlOptions::Prewarm::all, true));
        expect(prewarm.at("success") == true &&
                   prewarm.at("nothing_to_prepare") == false &&
                   engine.codec_on_coreml(),
               "explicit prewarm did not reach ready");
        const auto result = [&] {
            scoped_env batch("AUDIO8_COREML_STREAM_DISABLE", "1");
            return engine.synthesize("ab");
        }();
        std::vector<float> delivered;
        const auto streamed = engine.synthesize_stream(
            "ab", [&](const float * pcm, size_t n, int) {
                delivered.insert(delivered.end(), pcm, pcm + n);
                return true;
            });
        expect(!result.coreml_pending && !streamed.coreml_pending &&
                   result.codec_synthesis_backend == "coreml-all" &&
                   streamed.codec_synthesis_backend == "coreml-all",
               "public generated synthesis and stream must both use the ready "
               "Core ML codec");
        const double cos = result.pcm.size() == streamed.pcm.size()
                               ? cosine(result.pcm, streamed.pcm)
                               : -1;
        expect(
            result.frames > 64 && result.codes == streamed.codes &&
                delivered == streamed.pcm && cos >= STREAM_MIN_COSINE,
            "public streamed audio differed from after-generation synthesis");
        std::printf(
            "public stream vs after generation: frames=%d cosine=%.7f\n",
            result.frames, cos);
    }
    check_fit_stable(workload, cold, cache);
    check_source_changes(gguf, cache / "source-changes", options);
    fs::remove(lm);
}

void check_prebuilt_overlap(const std::string & gguf) {
    CoremlOptions options;
    options.compute_units = "all";
    options.generate = CoremlOptions::Generate::never;
    options.prewarm = CoremlOptions::Prewarm::none;
    options.wait = CoremlOptions::Wait::none;
    owned_codec codec;
    std::string error;
    if (!load_codec(gguf, 99, "auto", codec.model, &error, options)) {
        fail("prebuilt fixture load failed: " + error);
        return;
    }
    const auto reserve =
        prebuilt_codec_ane_reserve(coreml_codec_sidecar_path(gguf));
    if (check_overlap(codec, false, 0, reserve)) {
        expect(stage(codec.model.coreml->status()).at("route") == "prebuilt",
               "background prebuilt load used a different route");
    }
}

void check_unvalidated(const std::string & gguf, const std::string & cache) {
    CoremlOptions options;
    options.cache_dir = cache;
    const auto candidates = pcm::generation::probe(
        {"audio8-codec-w64"}, gguf, pcm::resolve_coreml({}, "AUDIO8"));
    expect(candidates.size() == 1 && candidates.front().matched &&
               candidates.front().memory.validated_units.empty(),
           "the shipped Audio8 template gained a generation allowlist");
    owned_codec codec;
    if (!load(gguf, codec, options))
        return;
    const auto status = stage(codec.model.coreml->status());
    expect(status.at("state") == "unsupported(unvalidated_units)" &&
               status.at("compute_units") == "all",
           "unvalidated generation should be suppressed: " + status.dump());
    const auto call = codec.model.coreml->acquire(nullptr);
    expect(!call.model && !call.pending && !call.strict,
           "suppressed route should latch ggml immediately");

    options.use = CoremlOptions::Use::require;
    owned_codec strict;
    if (!load(gguf, strict, options))
        return;
    try {
        (void)strict.model.coreml->acquire(nullptr);
        fail("require unexpectedly accepted an unsupported generated route");
    } catch (const std::runtime_error & e) {
        expect(std::string(e.what()).find("unvalidated_units") !=
                   std::string::npos,
               "strict failure did not name the unavailable route");
    }
    const auto lm = write_matching_tiny_lm(codec.model.hp);
    const auto fit = fit_params(fit_options(lm, gguf, options));
    expect(fit.coreml.route == "none" && !fit.coreml.validated &&
               fit.coreml.bytes == 0 && fit.fits == fit.coreml.ggml_only_fits,
           "unvalidated route changed the ggml fit");
    fs::remove(lm);

    options.use = CoremlOptions::Use::automatic;
    options.generate = CoremlOptions::Generate::never;
    owned_codec disabled;
    if (!load(gguf, disabled, options))
        return;
    const auto disabled_status = stage(disabled.model.coreml->status());
    expect(disabled_status.at("state") == "disabled" &&
               disabled_status.at("route") == "none",
           "generate=never should disable a matching template: " +
               disabled_status.dump());
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc == 3 && std::string(argv[1]) == "--prebuilt-overlap") {
        check_prebuilt_overlap(argv[2]);
        std::printf("%s\n", g_failures == 0 ? "PASS" : "FAIL");
        return g_failures == 0 ? 0 : 1;
    }
    if (argc == 4) {
        try {
            const std::string mode = argv[1];
            if (mode == "--generated")
                check_generated(argv[2], argv[3]);
            else if (mode == "--unvalidated")
                check_unvalidated(argv[2], argv[3]);
            else
                fail("unknown test mode: " + mode);
        } catch (const std::exception & e) {
            fail(e.what());
        }
        std::printf("%s\n", g_failures == 0 ? "PASS" : "FAIL");
        return g_failures == 0 ? 0 : 1;
    }
    if (argc != 1)
        return 2;

    const char * models_dir = std::getenv("AUDIO8_COREML_TEST_MODELS_DIR");
    if (models_dir == nullptr || *models_dir == '\0') {
        std::fprintf(stderr, "[coreml-parity] AUDIO8_COREML_TEST_MODELS_DIR not set; skipping\n");
        return SKIP_EXIT_CODE;
    }
    const std::string gguf = find_decoder_gguf(models_dir);
    if (gguf.empty()) {
        std::fprintf(stderr, "[coreml-parity] no codec decoder GGUF under %s; skipping\n", models_dir);
        return SKIP_EXIT_CODE;
    }
    const std::string sidecar = coreml_codec_sidecar_path(gguf);
    if (!fs::exists(sidecar)) {
        std::fprintf(stderr, "[coreml-parity] no sidecar at %s; skipping\n", sidecar.c_str());
        return SKIP_EXIT_CODE;
    }

    check_route_policy(gguf);
    codec_header header;
    std::string error;
    if (!peek_codec_header(gguf, header, &error)) {
        fail(error);
        return 1;
    }
    const std::string tiny_lm = write_matching_tiny_lm(header.hp);
    check_delayed_synthesis(tiny_lm, gguf);
    check_short_context_prebuilt(gguf, header.hp);
    check_prebuilt_reserve_boundary(
        fs::path(test_tmpdir()) /
        ("audio8-prebuilt-reserve-" + test_process_tag()));

    owned_codec ggml;
    {
        scoped_env disabled("AUDIO8_COREML_DISABLE", "1");
        if (!load(gguf, ggml)) return 1;
    }
    owned_codec coreml;
    {
        scoped_env strict("AUDIO8_COREML_STRICT", "1");
        if (!load(gguf, coreml)) return 1;
    }
    if (!coreml.model.coreml || !coreml.model.coreml->wait(0)) {
        fail("sidecar at " + sidecar + " did not initialise");
        return 1;
    }
    const int window = coreml.model.coreml->acquire(nullptr).model->window();
    const int context = synthesis_context_frames(coreml.model);
    std::fprintf(stderr, "[coreml-parity] sidecar %s: window %d frames, causal context %d frames\n",
                 sidecar.c_str(), window, context);
    if (context >= window) {
        fail("the window cannot carry the causal context");
        return 1;
    }

    check_parity("short", ggml.model, coreml.model, SHORT_FRAMES);
    check_parity("long", ggml.model, coreml.model, LONG_FRAMES);
    check_cancellation(coreml.model);
    check_streamed(coreml.model, SHORT_FRAMES);
    check_streamed(coreml.model, LONG_FRAMES);
    check_streamed_cancel(coreml.model);

    check_public_load_status(tiny_lm, gguf, /*want_loaded=*/true, "sidecar present");
    {
        scoped_env disabled("AUDIO8_COREML_DISABLE", "1");
        check_public_load_status(tiny_lm, gguf, /*want_loaded=*/false, "sidecar disabled");
    }
    check_invalid_sidecar(gguf, tiny_lm);
    if (const char * fallback_dir = std::getenv("AUDIO8_COREML_FALLBACK_MODELS_DIR"); fallback_dir && *fallback_dir) {
        check_unservable_sidecar(fallback_dir, tiny_lm);
    } else {
        std::fprintf(stderr, "[coreml-parity] AUDIO8_COREML_FALLBACK_MODELS_DIR not set; unservable-sidecar checks not run\n");
    }
    if (const char * real_lm = std::getenv("AUDIO8_COREML_TEST_LM_GGUF"); real_lm && *real_lm) {
        check_public_synthesis_backend(real_lm, gguf, "coreml", /*want_attached_after=*/true,
                                       "sidecar present, public synthesis");
    }
    fs::remove(tiny_lm);

    std::printf("\n%s\n", g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}

#endif  // TTS_CPP_USE_COREML
