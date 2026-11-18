#define TTS_CPP_BUILD
#include "tts-cpp/pocket/engine.h"
#include "tts-cpp/pocket/fit.h"
#include "pocket/flow_lm.h"
#include "pocket/mimi.h"
#include "pocket/frontend.h"
#include "pocket/generation.h"
#include "pocket/reference_audio.h"
#include "pocket/backend.h"
#include "pocket/coreml.h"
#include "voice_features.h"
#include "backend_selection.h"
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
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
    std::string device_name;
    // Both stages on the shared Core ML runtime; null without TTS_CPP_COREML.
    std::unique_ptr<detail::CoremlStages> coreml;
    std::vector<float> bos;
    // Used by the synthesizing thread: this chunk's Mimi decoder stream.
    detail::CoremlDecoderUse decoder_use;
    std::unique_ptr<detail::CoremlDecoder> decoder;
    std::vector<float> chunk_latents;  // decoded on Core ML this chunk, for a ggml replay
    // Where this call's stages ran.
    bool flow_coreml = false, flow_ggml = false, mimi_coreml = false, mimi_ggml = false;
    std::string flow_label, mimi_label;
    std::atomic<bool> pending{false};  // a chunk ran on ggml because a sidecar was not ready
    std::atomic<bool> operating{false};
    std::atomic<bool> cancelled{false};
    // Set by cancel() together with `cancelled`; a blocking coreml_wait()
    // clears and watches only this one.
    std::atomic<bool> wait_cancelled{false};
    std::condition_variable changed;

    explicit Impl(const EngineOptions & o) : opts(o) {
        detail::backend_request(o.backend);
        detail::CoremlEngineOptions coreml_options = detail::coreml_engine_options(o);
        // Validates the Core ML options in every build (an explicit invalid
        // value throws even without Core ML).
        detail::coreml_plan_route(detail::CoremlStage::step, coreml_options);
        detail::coreml_plan_route(detail::CoremlStage::decoder, coreml_options);
#ifdef TTS_CPP_USE_COREML
        // Fit's memory-degrade: with a budget, Core ML is dropped when only the
        // ggml projection fits. Decided before the weights load, so it reads
        // the same free memory fit_params reports, for the workload and
        // headroom the caller's fit assessed.
        bool degrade = false;
        if (o.coreml.memory_budget_bytes) {
            FitOptions fit;
            fit.engine = o;
            fit.text = o.fit_text;
            fit.margin_bytes = o.fit_margin_bytes;
            degrade = fit_params(fit).coreml.disabled_reason == "memory";
        }
#endif
        if (o.output_sample_rate < 8000 || o.output_sample_rate > 192000) fail("output sample rate must be 8000..192000 Hz");
        if (o.steps < 1 || o.steps > 64 || !std::isfinite(o.temperature) || o.temperature < 0 || o.temperature > 10 ||
            !std::isfinite(o.noise_clamp) || o.noise_clamp < 0 || !std::isfinite(o.eos_threshold) ||
            o.frames_after_eos < -1 || o.frames_after_eos > 100 || o.max_tokens < 1 || o.max_tokens > 1024)
            fail("invalid synthesis options");
        if (o.voice_path.empty() == o.reference_audio_path.empty()) fail("supply exactly one prepared voice or reference audio file");
        std::unique_ptr<detail::ReferenceAudio> reference;
        if (!o.reference_audio_path.empty()) reference.reset(new detail::ReferenceAudio(o.reference_audio_path));
        if (!o.backends_dir.empty()) ::tts_cpp::detail::set_backends_directory(o.backends_dir);
        auto backend = detail::select_backend(o.use_gpu, o.backend);
        const auto device = ggml_backend_get_device(backend.get());
        device_name = ggml_backend_dev_description(device);
        frontend.reset(new detail::Frontend(o.frontend_path));
        // Each load records the file it read (Core ML generation asks for it).
        lm.reset(new detail::FlowLM(o.flow_lm_path, o.context, o.n_threads, device, [&](int fd, const std::string & p) {
            coreml_options.step_source = detail::coreml_loaded_source(fd, p);
        }));
        mimi.reset(new detail::Mimi(o.mimi_path, o.n_threads, device, [&](int fd, const std::string & p) {
            coreml_options.decoder_source = detail::coreml_loaded_source(fd, p);
        }));
        if (frontend->source_hash() != lm->source_hash() || mimi->source_hash() != lm->source_hash() ||
            frontend->vocab_size() != lm->config().vocab_size || lm->config().latent_dim != mimi->latent_dim()) fail("model artifacts do not belong to the same checkpoint");
        if (!o.voice_path.empty()) voice = lm->read_voice(o.voice_path);
        else {
            auto pcm = reference->read_mono();
            const int rate = reference->sample_rate();
            if (rate != mimi->sample_rate()) pcm = ::resample_sinc(pcm, rate, mimi->sample_rate());
            lm->prefill(lm->voice_embeddings(mimi->encode(pcm))); voice = lm->capture_voice();
        }
#ifdef TTS_CPP_USE_COREML
        // A chunk needs the voice plus at least one token and one frame.
        coreml = detail::CoremlStages::create(coreml_options, lm->config(), lm->source_hash(), voice.frames + 2, degrade);
        if (coreml && coreml->step_route().route != "none") bos = lm->bos_latent();
#endif
    }

    [[noreturn]] void fail_strict(const std::string & reason) const {
        fail(reason + " (Core ML strict policy: coreml_strict / " + detail::COREML_STRICT_ENV + ")");
    }
    // A failed prediction retires the sidecar for every Engine on it; strict
    // mode then fails the call instead of falling back.
    void retire_decoder(const std::exception & e) {
        if (decoder_use.retire) decoder_use.retire(e.what());
        const bool strict = coreml && coreml->decoder_strict();
        decoder.reset();
        decoder_use = {};
        if (strict) fail_strict(e.what());
        std::fprintf(stderr, "Pocket TTS: %s; retiring the Core ML Mimi decoder, Mimi continues on ggml\n", e.what());
    }
    detail::CacheRows voice_rows() const { return {0, voice.frames, voice.keys, voice.values}; }
    void retire_step(detail::CoremlStepUse & use, const std::exception & e) {
        if (use.retire) use.retire(e.what());
        use = {};
        if (coreml && coreml->step_strict()) fail_strict(e.what());
        std::fprintf(stderr, "Pocket TTS: %s; retiring the Core ML step sidecar, FlowLM continues on ggml\n", e.what());
    }

    struct Chunk {
        std::vector<int> tokens;
        detail::FrameBudget budget;
    };
    struct Queue {
        std::mutex mutex;
        std::deque<std::vector<float>> frames;
        bool done = false, stop = false;
        std::exception_ptr error;
    };
    std::vector<Chunk> prepare_chunks(const std::string & text) {
        std::vector<Chunk> prepared;
        size_t longest = 0;
        for (const auto & chunk : frontend->split(text, opts.max_tokens)) {
            auto tokens = frontend->encode(chunk.text);
            const int tail = opts.frames_after_eos >= 0 ? opts.frames_after_eos : chunk.tail_frames;
            const auto budget = detail::frame_budget(tokens.size(), tail);
            const size_t positions = voice.frames+tokens.size()+budget.max_frames();
            if (positions > size_t(opts.context))
                fail("text chunk exceeds the context; add sentence punctuation or increase context");
            prepared.push_back({std::move(tokens), budget});
            longest = std::max(longest, positions);
        }
        // Strict mode refuses a chunk past the step's context before any PCM
        // (the acquisition waits for the step).
        if (coreml && coreml->step_strict()) {
            auto use = coreml->acquire_step(0, &cancelled);
            if (use.step && longest > size_t(use.step->context()))
                fail_strict("text chunk needs " + std::to_string(longest) + " positions; the Core ML step sidecar holds " +
                            std::to_string(use.step->context()) + " (lower max_tokens or export a longer --context)");
        }
        if (prepared.empty()) fail("no text chunks");
        return prepared;
    }
    bool enqueue(Queue & queue, std::vector<float> latent) {
        std::unique_lock<std::mutex> lock(queue.mutex);
        changed.wait(lock, [&] { return queue.frames.size() < size_t(mimi->max_decode_frames()) || queue.stop || cancelled.load(); });
        if (queue.stop || cancelled.load()) return false;
        queue.frames.push_back(std::move(latent));
        changed.notify_all();
        return true;
    }
    // The Core ML step serves a chunk whose every position fits its state. It
    // is acquired once, here on the producer thread; a model this Engine has
    // not used yet gets the voice rows first. A failing prediction retires it;
    // the frames it generated are prefilled into the ggml cache and the chunk
    // continues on ggml.
    void generate_frames(const Chunk & chunk, Noise & noise, Queue & queue) {
        const int start = lm->position();
        lm->prefill(lm->text_embeddings(chunk.tokens));
        int position = lm->position();
        detail::CoremlStepUse use;
        if (coreml) use = coreml->acquire_step(chunk.budget.max_frames(), &cancelled);
        if (use.pending) pending.store(true);
        bool on_coreml = use.step && position+chunk.budget.max_frames() <= use.step->context();
        if (on_coreml) {
            try {
                auto lock = use.lock();
                // The voice prefix never changes and no frame writes below it.
                if (!use.step->primed()) use.step->prime(voice_rows());
                use.step->write_rows(lm->cache_rows(start, position-start));
            }
            catch (const std::exception & e) { retire_step(use, e); on_coreml = false; }
        }
        std::vector<float> previous, fed, hidden, next;
        detail::FrameProgress progress(chunk.budget);
        while (progress.has_next()) {
            if (cancelled.load()) return;
            { std::lock_guard<std::mutex> lock(queue.mutex); if (queue.stop) return; }
            const auto frame_noise = noise.sample(lm->config().latent_dim, opts.temperature, opts.noise_clamp);
            float eos = 0;
            if (on_coreml) {
                const auto & input = previous.empty() ? bos : previous;
                try { auto lock = use.lock(); use.step->step(position, input, frame_noise, hidden, next); }
                catch (const std::exception & e) {
                    on_coreml = false;
                    retire_step(use, e);
                    if (!fed.empty()) lm->prefill(lm->audio_embeddings(fed));
                }
                if (on_coreml) {
                    fed.insert(fed.end(), input.begin(), input.end()); ++position; flow_coreml = true;
                    flow_label = use.label;
                    eos = lm->eos_logit(hidden);
                    if (opts.steps > 1) { next = lm->sample(hidden, frame_noise, opts.steps); flow_ggml = true; }
                }
            }
            if (!on_coreml) {
                const auto condition = lm->advance(previous);
                eos = condition.eos_logit; flow_ggml = true;
                next = lm->sample(condition.hidden, frame_noise, opts.steps);
            }
            previous.swap(next);
            if (progress.finish_frame(eos > opts.eos_threshold)) return;
            if (!enqueue(queue, lm->denormalize(previous))) return;
        }
        if (!cancelled.load()) fail("generation reached its estimated limit without completing EOS");
    }
    void produce(const Chunk & chunk, Noise & noise, Queue & queue) noexcept {
        try { generate_frames(chunk, noise, queue); }
        catch (...) {
            std::lock_guard<std::mutex> lock(queue.mutex); queue.error = std::current_exception();
        }
        { std::lock_guard<std::mutex> lock(queue.mutex); queue.done = true; }
        changed.notify_all();
    }
    std::vector<float> dequeue(Queue & queue) {
        std::unique_lock<std::mutex> lock(queue.mutex);
        changed.wait(lock, [&] { return !queue.frames.empty() || queue.done || cancelled.load(); });
        if (queue.error) std::rethrow_exception(queue.error);
        if (cancelled.load()) return {};
        std::vector<float> latents;
        const size_t batch_size = size_t(mimi->max_decode_frames())*mimi->latent_dim();
        while (!queue.frames.empty() && latents.size() < batch_size) {
            auto & frame = queue.frames.front();
            latents.insert(latents.end(), frame.begin(), frame.end());
            queue.frames.pop_front();
        }
        changed.notify_all();
        return latents;
    }
    // A failing Core ML decode retires the decoder: the chunk's earlier
    // frames are decoded again on ggml (output discarded) to rebuild Mimi's
    // streaming state, then ggml decodes this batch.
    std::vector<float> decode(const std::vector<float> & latents) {
        if (decoder) {
            try {
                std::vector<float> pcm;
                { auto lock = decoder_use.lock(); pcm = decoder->decode(latents); }
                chunk_latents.insert(chunk_latents.end(), latents.begin(), latents.end());
                mimi_coreml = true;
                mimi_label = decoder_use.label;
                return pcm;
            } catch (const std::exception & e) {
                retire_decoder(e);
                const size_t batch = size_t(mimi->max_decode_frames())*mimi->latent_dim();
                for (size_t at = 0; at < chunk_latents.size(); at += batch)
                    mimi->decode(std::vector<float>(chunk_latents.begin()+at,
                                                    chunk_latents.begin()+std::min(chunk_latents.size(), at+batch)));
                chunk_latents.clear();
            }
        }
        mimi_ggml = true;
        return mimi->decode(latents);
    }
    using Emit = std::function<void(const std::vector<float> &)>;
    void drain(Queue & queue, OutputResampler & resampler, SynthesisResult & result, const Emit & emit) {
        while (!cancelled.load()) {
            auto latents = dequeue(queue);
            if (latents.empty()) break;
            auto pcm = decode(latents);
            if (cancelled.load()) break;
            result.generated_frames += latents.size()/mimi->latent_dim();
            emit(resampler.process(pcm));
        }
    }
    void stop_producer(Queue & queue, std::thread & producer) {
        { std::lock_guard<std::mutex> lock(queue.mutex); queue.stop = true; }
        changed.notify_all();
        if (producer.joinable()) producer.join();
    }
    void generate_chunk(const Chunk & chunk, Noise & noise, OutputResampler & resampler,
                        SynthesisResult & result, const Emit & emit) {
        lm->restore_voice(voice); mimi->reset_decoder();
        // The Mimi decoder is acquired once per chunk, on this (the calling) thread.
        decoder.reset();
        decoder_use = {};
        if (coreml) decoder_use = coreml->acquire_decoder(chunk.budget.max_frames(), &cancelled);
        if (decoder_use.pending) pending.store(true);
        if (decoder_use.model) decoder = std::make_unique<detail::CoremlDecoder>(decoder_use.model, decoder_use.label);
        chunk_latents.clear();
        Queue queue;
        std::thread producer([&] { produce(chunk, noise, queue); });
        try { drain(queue, resampler, result, emit); }
        catch (...) { stop_producer(queue, producer); throw; }
        stop_producer(queue, producer);
        if (queue.error && !cancelled.load()) std::rethrow_exception(queue.error);
    }
    void generate_chunks(const std::vector<Chunk> & chunks, Noise & noise, OutputResampler & resampler,
                         SynthesisResult & result, const Emit & emit) {
        for (const auto & chunk : chunks) {
            if (cancelled.load()) break;
            generate_chunk(chunk, noise, resampler, result, emit);
        }
    }
    SynthesisResult generate(const std::string & text, const AudioCallback & callback) {
        bool idle = false;
        if (!operating.compare_exchange_strong(idle, true)) fail("concurrent or reentrant synthesis on one engine is unsupported");
        struct Release { std::atomic<bool> & flag; ~Release() { flag.store(false); } } release{operating};
        cancelled.store(false);
        if (!callback) fail("stream callback is empty");
        const auto begin = Clock::now();
        // Validate every chunk, including its EOS tail, before delivering PCM.
        const auto chunks = prepare_chunks(text);
        SynthesisResult result; result.sample_rate = opts.output_sample_rate;
        Noise noise(opts.seed); OutputResampler resampler(mimi->sample_rate(), opts.output_sample_rate);
        flow_coreml = flow_ggml = mimi_coreml = mimi_ggml = false;
        flow_label.clear(); mimi_label.clear(); pending.store(false);
        bool delivered = false;
        const auto emit = [&](const std::vector<float> & pcm) {
            if (pcm.empty() || cancelled.load()) return;
            if (!delivered) { result.first_audio_ms = milliseconds(begin); delivered = true; }
            if (!callback(pcm.data(), pcm.size(), opts.output_sample_rate)) {
                cancelled.store(true); changed.notify_all();
            }
        };
        generate_chunks(chunks, noise, resampler, result, emit);
        if (!cancelled.load()) emit(resampler.finish());
        const auto placement = [](bool coreml, bool ggml, const std::string & label) {
            return !coreml ? detail::GGML_STAGE_BACKEND : ggml ? detail::MIXED_STAGE_BACKEND : label;
        };
        result.flow_lm_backend = placement(flow_coreml, flow_ggml, flow_label);
        result.mimi_backend = placement(mimi_coreml, mimi_ggml, mimi_label);
        result.coreml_pending = pending.load();
        // The chunk's handles are released with the call.
        decoder.reset();
        decoder_use = {};
        result.cancelled = cancelled.load(); result.generation_ms = milliseconds(begin); return result;
    }
};
Engine::Engine(const EngineOptions & options) : impl_(new Impl(options)) {}
Engine::~Engine() = default;
int Engine::sample_rate() const noexcept { return impl_->opts.output_sample_rate; }
const char * Engine::backend_name() const noexcept { return flow_lm_backend_name(); }
const char * Engine::flow_lm_backend_name() const noexcept { return impl_->lm->backend_name(); }
const char * Engine::mimi_backend_name() const noexcept { return impl_->mimi->backend_name(); }
bool Engine::flow_lm_on_coreml() const noexcept { return impl_->coreml && impl_->coreml->step_ready(); }
bool Engine::mimi_on_coreml() const noexcept { return impl_->coreml && impl_->coreml->decoder_ready(); }
std::string Engine::coreml_status() const { return impl_->coreml ? impl_->coreml->status_json() : std::string("{}"); }
bool Engine::coreml_wait(int64_t timeout_ms) const {
    if (!impl_->coreml) return false;
    // A readiness poll has nothing to cancel and touches no flag.
    if (timeout_ms == 0) return impl_->coreml->wait(0, nullptr);
    impl_->wait_cancelled.store(false);
    return impl_->coreml->wait(timeout_ms, &impl_->wait_cancelled);
}
const char * Engine::device_name() const noexcept { return impl_->device_name.c_str(); }
std::string Engine::model_source_sha256() const { return impl_->lm->source_hash(); }
void Engine::cancel() noexcept {
    impl_->cancelled.store(true); impl_->wait_cancelled.store(true); impl_->changed.notify_all();
}
SynthesisResult Engine::synthesize_stream(const std::string & text, const AudioCallback & callback) { return impl_->generate(text, callback); }
SynthesisResult Engine::synthesize(const std::string & text) {
    std::vector<float> pcm;
    auto result = synthesize_stream(text, [&](const float * p, size_t n, int) { pcm.insert(pcm.end(), p, p+n); return true; });
    result.pcm = std::move(pcm); return result;
}
}
