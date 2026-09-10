#include "tts-cpp/pocket/engine.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <thread>

using namespace tts_cpp::pocket;
template<class F> void rejects(const char * name, F fn) {
    bool threw = false; try { fn(); } catch (const std::exception &) { threw = true; }
    if (!threw) throw std::runtime_error(std::string("accepted ")+name);
}
static void compare(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.empty() || a.size() != b.size()) throw std::runtime_error("PCM length mismatch");
    double error = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) throw std::runtime_error("non-finite PCM");
        error = std::max(error, std::abs(double(a[i])-b[i]));
    }
    if (error > 0.0001) throw std::runtime_error("stream/batch/reuse differ");
}
int main(int argc, char ** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: test-pocket-engine model-directory flow-lm.gguf\n"); return 2; }
    try {
        const std::string dir = argv[1];
        EngineOptions opts; opts.flow_lm_path = argv[2]; opts.mimi_path = dir+"/mimi.gguf";
        opts.frontend_path = dir+"/frontend.json"; opts.voice_path = dir+"/voice.gguf";
        Engine engine(opts);
        const std::string text = "Hello! We can generate speech with Fabric.";
        const auto batch = engine.synthesize(text);
        if (batch.cancelled || batch.sample_rate != 24000 || batch.generated_frames <= 0 || batch.first_audio_ms <= 0) throw std::runtime_error("invalid synthesis result");
        std::vector<float> streamed; int callbacks = 0;
        const auto stream = engine.synthesize_stream(text, [&](const float * p, size_t n, int rate) {
            if (rate != 24000 || !n || n > 16*1920) throw std::runtime_error("invalid callback chunk");
            streamed.insert(streamed.end(), p, p+n);
            if (++callbacks == 1) {
                rejects("reentrant synthesis", [&] { engine.synthesize("Nested request."); });
                bool rejected = false;
                std::thread other([&] { try { engine.synthesize("Concurrent request."); } catch (const std::exception &) { rejected = true; } });
                other.join(); if (!rejected) throw std::runtime_error("concurrent call accepted");
            }
            return true;
        });
        if (stream.cancelled || !stream.pcm.empty() || callbacks < 2) throw std::runtime_error("invalid stream result");
        compare(batch.pcm, streamed);
        callbacks = 0;
        auto cancelled = engine.synthesize_stream(text, [&](const float *, size_t, int) { ++callbacks; return false; });
        if (!cancelled.cancelled || callbacks != 1) throw std::runtime_error("callback cancellation failed");
        compare(engine.synthesize(text).pcm, batch.pcm);
        cancelled = engine.synthesize_stream(text, [&](const float *, size_t, int) {
            std::thread stop([&] { engine.cancel(); }); stop.join(); return true;
        });
        if (!cancelled.cancelled) throw std::runtime_error("cross-thread cancellation failed");
        rejects("callback exception", [&] { engine.synthesize_stream(text, [](const float *, size_t, int) -> bool { throw std::runtime_error("callback failure"); }); });
        rejects("empty text", [&] { engine.synthesize("  \n"); });
        rejects("invalid UTF-8", [&] { engine.synthesize(std::string(1, char(0xff))); });
        compare(engine.synthesize(text).pcm, batch.pcm);
        for (int rate : {8000, 44100}) {
            opts.output_sample_rate = rate; Engine resampled(opts);
            if (resampled.sample_rate() != rate) throw std::runtime_error("wrong advertised rate");
            const auto output = resampled.synthesize(text); std::vector<float> chunks;
            resampled.synthesize_stream(text, [&](const float * p, size_t n, int sr) {
                if (sr != rate) throw std::runtime_error("wrong callback rate");
                chunks.insert(chunks.end(), p, p+n); return true;
            });
            compare(output.pcm, chunks);
            if (output.sample_rate != rate || std::abs(double(output.pcm.size())/rate-double(batch.pcm.size())/24000) > 1.0/rate)
                throw std::runtime_error("resampling changed duration");
        }
        opts.output_sample_rate = 0;
        rejects("invalid sample rate", [&] { Engine invalid(opts); });
        opts.output_sample_rate = 24000;
        opts.eos_threshold = 1e30f;
        Engine no_eos(opts);
        rejects("producer EOS failure", [&] { no_eos.synthesize("Hello."); });
        rejects("producer EOS failure reuse", [&] { no_eos.synthesize("Hello."); });
        opts.eos_threshold = -4; opts.steps = 0;
        rejects("invalid steps", [&] { Engine invalid(opts); });
        std::puts("Pocket Engine synthesis, streaming, cancellation, reentry, concurrency and error recovery passed"); return 0;
    } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
