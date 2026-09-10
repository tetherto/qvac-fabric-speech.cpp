#define TTS_CPP_BUILD
#include "tts-cpp/pocket/engine.h"
#include "pocket/flow_lm.h"
#include "pocket/mimi.h"
#include "pocket/frontend.h"
#include "pocket/reference_audio.h"
#include "voice_features.h"
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>

namespace tts_cpp::pocket {
namespace {
using Clock = std::chrono::steady_clock;
[[noreturn]] void fail(const std::string & text) { throw std::runtime_error("Pocket TTS: "+text); }
double milliseconds(Clock::time_point begin) { return std::chrono::duration<double, std::milli>(Clock::now()-begin).count(); }
// Explicit transform avoids implementation-defined std::normal_distribution.
class Noise {
    std::mt19937 random_;
    bool spare_ = false;
    double saved_ = 0;
public:
    explicit Noise(uint32_t seed) : random_(seed) {}
    double normal() {
        if (spare_) { spare_ = false; return saved_; }
        const double u = (double(random_())+0.5)/4294967296.0;
        const double v = (double(random_())+0.5)/4294967296.0;
        const double r = std::sqrt(-2*std::log(u)), angle = 6.2831853071795864769*v;
        saved_ = r*std::sin(angle); spare_ = true; return r*std::cos(angle);
    }
    std::vector<float> sample(int width, float temperature, float clamp) {
        std::vector<float> values(width);
        const double sigma = std::sqrt(double(temperature));
        for (auto & x : values) {
            // Inverse-CDF sampling for very narrow truncation would be more
            // efficient. Bound rejection so hostile options cannot hang.
            int tries = 0;
            do { x = float(normal()*sigma); if (++tries > 100000) fail("noise truncation is too narrow"); }
            while (clamp > 0 && std::abs(x) > clamp);
        }
        return values;
    }
};
}

struct Engine::Impl {
    EngineOptions opts;
    std::unique_ptr<detail::FlowLM> lm;
    std::unique_ptr<detail::Mimi> mimi;
    std::unique_ptr<detail::Frontend> frontend;
    detail::VoiceState voice;
    std::atomic<bool> operating{false};
    std::atomic<bool> cancelled{false};
    std::condition_variable changed;

    explicit Impl(const EngineOptions & o) : opts(o) {
        if (o.output_sample_rate < 8000 || o.output_sample_rate > 192000) fail("output sample rate must be 8000..192000 Hz");
        if (o.steps < 1 || o.steps > 64 || !std::isfinite(o.temperature) || o.temperature < 0 || o.temperature > 10 ||
            !std::isfinite(o.noise_clamp) || o.noise_clamp < 0 || !std::isfinite(o.eos_threshold) ||
            o.frames_after_eos < -1 || o.frames_after_eos > 100 || o.max_tokens < 1 || o.max_tokens > 1024)
            fail("invalid synthesis options");
        if (o.voice_path.empty() == o.reference_audio_path.empty()) fail("supply exactly one prepared voice or reference audio file");
        std::unique_ptr<detail::ReferenceAudio> reference;
        if (!o.reference_audio_path.empty()) reference.reset(new detail::ReferenceAudio(o.reference_audio_path));
        frontend.reset(new detail::Frontend(o.frontend_path));
        lm.reset(new detail::FlowLM(o.flow_lm_path, o.context, o.n_threads));
        mimi.reset(new detail::Mimi(o.mimi_path, o.n_threads));
        if (frontend->source_hash() != lm->source_hash() || mimi->source_hash() != lm->source_hash() ||
            frontend->vocab_size() != lm->config().vocab_size || lm->config().latent_dim != mimi->latent_dim()) fail("model artifacts do not belong to the same checkpoint");
        if (!o.voice_path.empty()) voice = lm->read_voice(o.voice_path);
        else {
            auto pcm = reference->read_mono();
            const int rate = reference->sample_rate();
            if (rate != 24000) pcm = ::resample_sinc(pcm, rate, 24000);
            lm->prefill(lm->voice_embeddings(mimi->encode(pcm))); voice = lm->capture_voice();
        }
    }

    SynthesisResult generate(const std::string & text, const AudioCallback & callback) {
        bool idle = false;
        if (!operating.compare_exchange_strong(idle, true)) fail("concurrent or reentrant synthesis on one engine is unsupported");
        struct Release { std::atomic<bool> & flag; ~Release() { flag.store(false); } } release{operating};
        cancelled.store(false);
        if (!callback) fail("stream callback is empty");
        const auto begin = Clock::now();
        const auto chunks = frontend->split(text, opts.max_tokens);
        if (chunks.empty()) fail("no text chunks");
        // Validate all chunks before delivering any PCM.
        for (const auto & chunk : chunks) {
            const auto tokens = frontend->encode(chunk.text);
            const int max_frames = std::ceil((tokens.size()/3.0+2)*12.5);
            if (voice.frames+tokens.size()+max_frames > size_t(opts.context)) fail("text chunk exceeds the context; add sentence punctuation or increase context");
        }
        SynthesisResult result; result.sample_rate = opts.output_sample_rate;
        Noise noise(opts.seed); OutputResampler resampler(24000, opts.output_sample_rate);
        bool delivered = false;
        const auto emit = [&](const std::vector<float> & pcm) {
            if (pcm.empty() || cancelled.load()) return;
            if (!delivered) { result.first_audio_ms = milliseconds(begin); delivered = true; }
            if (!callback(pcm.data(), pcm.size(), opts.output_sample_rate)) {
                cancelled.store(true); changed.notify_all();
            }
        };
        for (const auto & chunk : chunks) {
            if (cancelled.load()) break;
            lm->restore_voice(voice); mimi->reset_decoder();
            const auto tokens = frontend->encode(chunk.text);
            const int max_frames = std::ceil((tokens.size()/3.0+2)*12.5);
            const int tail = opts.frames_after_eos >= 0 ? opts.frames_after_eos : chunk.tail_frames;
            struct Queue {
                std::mutex mutex;
                std::deque<std::vector<float>> frames;
                bool done = false, stop = false;
                std::exception_ptr error;
            } queue;
            std::thread producer([&] {
                try {
                    lm->prefill(lm->text_embeddings(tokens));
                    std::vector<float> previous; int eos = -1; bool ended = false;
                    for (int frame = 0; frame < max_frames && !cancelled.load(); ++frame) {
                        { std::lock_guard<std::mutex> g(queue.mutex); if (queue.stop) break; }
                        const auto condition = lm->advance(previous);
                        previous = lm->sample(condition.hidden, noise.sample(lm->config().latent_dim, opts.temperature, opts.noise_clamp), opts.steps);
                        if (condition.eos_logit > opts.eos_threshold && eos < 0) eos = frame;
                        if (eos >= 0 && frame >= eos+tail) { ended = true; break; }
                        auto latent = lm->denormalize(previous);
                        std::unique_lock<std::mutex> g(queue.mutex);
                        changed.wait(g, [&] { return queue.frames.size() < 16 || queue.stop || cancelled.load(); });
                        if (queue.stop || cancelled.load()) break;
                        queue.frames.push_back(std::move(latent)); changed.notify_all();
                    }
                    std::lock_guard<std::mutex> g(queue.mutex);
                    if (!ended && !queue.stop && !cancelled.load()) fail("generation reached its estimated limit without completing EOS");
                } catch (...) {
                    std::lock_guard<std::mutex> g(queue.mutex); queue.error = std::current_exception();
                }
                { std::lock_guard<std::mutex> g(queue.mutex); queue.done = true; }
                changed.notify_all();
            });
            try {
                while (!cancelled.load()) {
                    std::vector<float> latents;
                    {
                        std::unique_lock<std::mutex> g(queue.mutex);
                        changed.wait(g, [&] { return !queue.frames.empty() || queue.done || cancelled.load(); });
                        if (queue.error) std::rethrow_exception(queue.error);
                        if (cancelled.load() || (queue.frames.empty() && queue.done)) break;
                        while (!queue.frames.empty() && latents.size() < size_t(16*lm->config().latent_dim)) {
                            auto & frame = queue.frames.front(); latents.insert(latents.end(), frame.begin(), frame.end()); queue.frames.pop_front();
                        }
                        changed.notify_all();
                    }
                    auto pcm = mimi->decode(latents);
                    if (cancelled.load()) break;
                    result.generated_frames += latents.size()/lm->config().latent_dim;
                    emit(resampler.process(pcm));
                }
                { std::lock_guard<std::mutex> g(queue.mutex); queue.stop = true; }
                changed.notify_all(); producer.join();
                if (queue.error && !cancelled.load()) std::rethrow_exception(queue.error);
            } catch (...) {
                { std::lock_guard<std::mutex> g(queue.mutex); queue.stop = true; }
                changed.notify_all(); if (producer.joinable()) producer.join(); throw;
            }
        }
        if (!cancelled.load()) emit(resampler.finish());
        result.cancelled = cancelled.load(); result.generation_ms = milliseconds(begin); return result;
    }
};
Engine::Engine(const EngineOptions & options) : impl_(new Impl(options)) {}
Engine::~Engine() = default;
int Engine::sample_rate() const noexcept { return impl_->opts.output_sample_rate; }
std::string Engine::model_source_sha256() const { return impl_->lm->source_hash(); }
void Engine::cancel() noexcept { impl_->cancelled.store(true); impl_->changed.notify_all(); }
SynthesisResult Engine::synthesize_stream(const std::string & text, const AudioCallback & callback) { return impl_->generate(text, callback); }
SynthesisResult Engine::synthesize(const std::string & text) {
    std::vector<float> pcm;
    auto result = synthesize_stream(text, [&](const float * p, size_t n, int) { pcm.insert(pcm.end(), p, p+n); return true; });
    result.pcm = std::move(pcm); return result;
}
}
