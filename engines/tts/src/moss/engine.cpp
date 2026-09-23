#include "tts-cpp/moss/engine.h"

#include "moss/codec.h"
#include "moss/delay_lm.h"
#include "moss/frontend.h"
#include "moss/generation.h"

#include "backend_selection.h"
#include "dr_wav.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <functional>
#include <mutex>
#include <random>
#include <stdexcept>

namespace tts_cpp::moss {
namespace {

using detail::AudioSegments;
using detail::Codec;
using detail::DelayLM;
using detail::DelayLogits;
using detail::DelayRow;
using detail::DelayState;
using detail::Frontend;
using detail::SamplingConfig;

constexpr int CONTEXT_HEADROOM = 8;
constexpr int SINGLE_DECODE_MAX_FRAMES = 750;
constexpr int TERMINATION_ROWS = 2;

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("moss engine: " + message);
}

double elapsed_ms(std::chrono::steady_clock::time_point since) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - since).count();
}

constexpr uint64_t MAX_REFERENCE_WAV_BYTES = 64ull * 1024 * 1024;
constexpr uint64_t MAX_REFERENCE_SECONDS   = 60;
constexpr unsigned int MAX_REFERENCE_CHANNELS = 64;
constexpr size_t REFERENCE_READ_BLOCK_FRAMES  = 4096;

void require_reference_file_size(const std::string & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    const std::streamoff bytes = input ? (std::streamoff) input.tellg() : -1;
    if (bytes <= 0) {
        fail("cannot read reference WAV: " + path);
    }
    if ((uint64_t) bytes > MAX_REFERENCE_WAV_BYTES) {
        fail("reference WAV must be at most 64 MiB on disk");
    }
}

void validate_reference_header(const drwav & wav, int expected_rate) {
    if ((int) wav.sampleRate != expected_rate) {
        fail("reference WAV must be sampled at " + std::to_string(expected_rate) +
             " Hz; there is no resampling");
    }
    if (wav.channels == 0 || wav.channels > MAX_REFERENCE_CHANNELS) {
        fail("reference WAV channel count is unsupported");
    }
    if (wav.totalPCMFrameCount == 0 ||
        wav.totalPCMFrameCount > (uint64_t) wav.sampleRate * MAX_REFERENCE_SECONDS) {
        fail("reference WAV must hold between one sample and " +
             std::to_string(MAX_REFERENCE_SECONDS) + " seconds of audio");
    }
}

void downmix_frames(const std::vector<float> & interleaved, size_t count, unsigned int channels,
                    std::vector<float> & mono, size_t start) {
    for (size_t frame = 0; frame < count; ++frame) {
        float sum = 0.0f;
        for (unsigned int channel = 0; channel < channels; ++channel) {
            sum += interleaved[frame * channels + channel];
        }
        mono[start + frame] = sum / (float) channels;
    }
}

std::vector<float> read_mono_frames(drwav & wav) {
    std::vector<float> mono((size_t) wav.totalPCMFrameCount, 0.0f);
    std::vector<float> interleaved(REFERENCE_READ_BLOCK_FRAMES * wav.channels);
    for (size_t start = 0; start < mono.size();) {
        const size_t count = std::min(REFERENCE_READ_BLOCK_FRAMES, mono.size() - start);
        if (drwav_read_pcm_frames_f32(&wav, count, interleaved.data()) != count) {
            fail("reference WAV is truncated");
        }
        downmix_frames(interleaved, count, wav.channels, mono, start);
        start += count;
    }
    return mono;
}

std::vector<float> read_reference_wav(const std::string & path, int expected_rate) {
    require_reference_file_size(path);
    drwav wav{};
    if (!drwav_init_file(&wav, path.c_str(), nullptr)) {
        fail("cannot read reference WAV: " + path);
    }
    try {
        validate_reference_header(wav, expected_rate);
        std::vector<float> mono = read_mono_frames(wav);
        drwav_uninit(&wav);
        return mono;
    } catch (...) {
        drwav_uninit(&wav);
        throw;
    }
}

} // namespace

struct Engine::Impl {
    EngineOptions options;
    SamplingConfig sampling;
    std::unique_ptr<DelayLM> backbone;
    std::unique_ptr<Frontend> frontend;
    std::unique_ptr<Codec> decoder;
    std::vector<int32_t> reference_codes;
    int reference_frames = 0;
    std::atomic<bool> cancel_requested{false};
    std::mutex synthesis_mutex;

    void load() {
        if (options.backbone_path.empty() || options.decoder_path.empty()) {
            fail("backbone_path and decoder_path are required");
        }
        if (options.max_new_tokens < 1) {
            fail("max_new_tokens must be positive");
        }
        if (options.duration_tokens < 0) {
            fail("duration_tokens must not be negative");
        }
        if (!options.backends_dir.empty()) {
            ::tts_cpp::detail::set_backends_directory(options.backends_dir);
        }
        sampling.text_temperature  = options.text_temperature;
        sampling.text_top_p        = options.text_top_p;
        sampling.text_top_k        = options.text_top_k;
        sampling.audio_temperature = options.audio_temperature;
        sampling.audio_top_p       = options.audio_top_p;
        sampling.audio_top_k       = options.audio_top_k;
        sampling.audio_repetition_penalty = options.audio_repetition_penalty;
        backbone.reset(new DelayLM(options.backbone_path, options.use_gpu, options.n_threads,
                options.context));
        frontend.reset(new Frontend(*backbone));
        decoder.reset(new Codec(options.decoder_path, options.use_gpu, options.n_threads));
        if (decoder->is_encoder()) {
            fail("decoder_path points at an encoder checkpoint");
        }
        if (decoder->num_quantizers() != backbone->config().n_vq) {
            fail("decoder quantizers do not match the backbone channels");
        }
        validate_duration_budget();
        load_reference();
    }

    void validate_duration_budget() const {
        if (options.duration_tokens == 0) {
            return;
        }
        const int needed = options.duration_tokens + backbone->config().n_vq - 1 + TERMINATION_ROWS;
        if (needed > options.max_new_tokens) {
            fail("duration_tokens needs " + std::to_string(needed) +
                 " generation steps with the delay drain; raise max_new_tokens");
        }
    }

    void load_reference() {
        if (options.reference_audio_path.empty()) {
            return;
        }
        if (options.encoder_path.empty()) {
            fail("voice cloning needs encoder_path next to reference_audio_path");
        }
        Codec encoder(options.encoder_path, options.use_gpu, options.n_threads);
        if (!encoder.is_encoder()) {
            fail("encoder_path points at a decoder checkpoint");
        }
        if (encoder.num_quantizers() != backbone->config().n_vq) {
            fail("encoder quantizers do not match the backbone channels");
        }
        const std::vector<float> pcm = read_reference_wav(options.reference_audio_path,
                encoder.sample_rate());
        reference_codes = encoder.encode(pcm);
        reference_frames = (int) (reference_codes.size() / (size_t) encoder.num_quantizers());
        if (reference_frames == 0) {
            fail("reference audio is shorter than one codec frame");
        }
    }

    void generate_rows(const std::vector<DelayRow> & prompt, DelayState & state,
                       std::mt19937 & rng, SynthesisResult & result,
                       const std::function<bool()> & after_step) {
        DelayLogits logits = backbone->prefill(prompt);
        for (int step = 0; step < options.max_new_tokens; ++step) {
            if (cancel_requested) {
                result.cancelled = true;
                return;
            }
            const DelayRow row = state.step(logits, sampling, rng);
            result.generated_frames += 1;
            if (after_step && !after_step()) {
                result.cancelled = true;
                return;
            }
            if (state.stopping()) {
                return;
            }
            logits = backbone->step(row);
        }
    }

    struct StreamProgress {
        std::vector<int32_t> pending;
        int scanned_frames = 0;
        int emitted_frames = 0;
        bool in_segment = false;
        bool audio_emitted = false;
    };

    int pending_frames(const StreamProgress & progress) const {
        return (int) (progress.pending.size() / (size_t) backbone->config().n_vq);
    }

    std::vector<int32_t> take_pending(StreamProgress & progress, int frames) const {
        const size_t count = (size_t) frames * (size_t) backbone->config().n_vq;
        std::vector<int32_t> taken(progress.pending.begin(),
                progress.pending.begin() + (std::ptrdiff_t) count);
        progress.pending.erase(progress.pending.begin(), progress.pending.begin() + (std::ptrdiff_t) count);
        return taken;
    }

    bool deliver(const std::vector<float> & pcm, StreamProgress & progress, const AudioCallback & callback,
                 const std::chrono::steady_clock::time_point & start, SynthesisResult & result) {
        if (pcm.empty()) {
            return true;
        }
        if (!progress.audio_emitted) {
            progress.audio_emitted = true;
            result.first_audio_ms = elapsed_ms(start);
        }
        return callback(pcm.data(), pcm.size(), decoder->sample_rate());
    }

    bool decode_pending(StreamProgress & progress, int frames, const AudioCallback & callback,
                        const std::chrono::steady_clock::time_point & start, SynthesisResult & result) {
        const std::vector<int32_t> codes = take_pending(progress, frames);
        const auto decode_start = std::chrono::steady_clock::now();
        const std::vector<float> pcm = decoder->decode_stream(codes);
        result.decode_ms += elapsed_ms(decode_start);
        progress.emitted_frames += frames;
        return deliver(pcm, progress, callback, start, result);
    }

    bool emit_full_chunks(StreamProgress & progress, const AudioCallback & callback,
                          const std::chrono::steady_clock::time_point & start, SynthesisResult & result) {
        const int chunk = std::max(1, options.stream_chunk_frames);
        while (pending_frames(progress) >= chunk) {
            if (!decode_pending(progress, chunk, callback, start, result)) {
                return false;
            }
        }
        return true;
    }

    bool close_segment(StreamProgress & progress, const AudioCallback & callback,
                       const std::chrono::steady_clock::time_point & start, SynthesisResult & result) {
        progress.in_segment = false;
        const int remaining = pending_frames(progress);
        return remaining == 0 || decode_pending(progress, remaining, callback, start, result);
    }

    bool consume_frame(const std::vector<int32_t> & frame, StreamProgress & progress,
                       const AudioCallback & callback,
                       const std::chrono::steady_clock::time_point & start, SynthesisResult & result) {
        const int n_vq = backbone->config().n_vq;
        if (detail::frame_is_pad(frame, 0, n_vq, backbone->config().audio_pad_code)) {
            return !progress.in_segment || close_segment(progress, callback, start, result);
        }
        if (!progress.in_segment) {
            decoder->begin_decode_stream();
            progress.in_segment = true;
        }
        progress.pending.insert(progress.pending.end(), frame.begin(), frame.end());
        return emit_full_chunks(progress, callback, start, result);
    }

    bool stream_ready_frames(const DelayState & state, int prompt_frames, StreamProgress & progress,
                             const AudioCallback & callback,
                             const std::chrono::steady_clock::time_point & start, SynthesisResult & result) {
        const int available = state.available_frames(prompt_frames);
        for (; progress.scanned_frames < available; ++progress.scanned_frames) {
            if (!consume_frame(state.frame_codes(prompt_frames, progress.scanned_frames), progress,
                    callback, start, result)) {
                return false;
            }
        }
        return true;
    }

    void decode_batch(const AudioSegments & segments, SynthesisResult & result) {
        const auto decode_start = std::chrono::steady_clock::now();
        result.pcm = detail::decode_segments(*decoder, segments, SINGLE_DECODE_MAX_FRAMES);
        result.decode_ms = elapsed_ms(decode_start);
    }

    std::vector<DelayRow> build_checked_prompt(const std::string & text) {
        const std::vector<DelayRow> prompt = frontend->build_prompt(backbone->config(), text,
                options.language, options.duration_tokens, reference_codes, reference_frames);
        if ((int) prompt.size() + options.max_new_tokens + CONTEXT_HEADROOM > backbone->context()) {
            fail("prompt plus max_new_tokens exceeds the context; raise the context option");
        }
        return prompt;
    }

    SynthesisResult run(const std::string & text) {
        std::unique_lock<std::mutex> lock(synthesis_mutex, std::try_to_lock);
        if (!lock.owns_lock()) {
            fail("synthesis already in progress on this instance");
        }
        if (text.empty()) {
            fail("text must not be empty");
        }
        cancel_requested = false;

        const std::vector<DelayRow> prompt = build_checked_prompt(text);
        std::mt19937 rng(options.seed);
        DelayState state(backbone->config(), prompt, frontend->tokens().pad,
                frontend->tokens().im_end);
        SynthesisResult result;

        const auto generation_start = std::chrono::steady_clock::now();
        backbone->reset();
        generate_rows(prompt, state, rng, result, nullptr);
        result.generation_ms = elapsed_ms(generation_start);
        if (result.cancelled) {
            return result;
        }

        const AudioSegments segments = state.generated_audio((int) prompt.size());
        if (segments.empty()) {
            fail("the model produced no audio frames");
        }
        if (cancel_requested) {
            result.cancelled = true;
            return result;
        }
        decode_batch(segments, result);
        result.sample_rate = decoder->sample_rate();
        result.cancelled = cancel_requested;
        return result;
    }

    SynthesisResult run_stream(const std::string & text, const AudioCallback & callback) {
        std::unique_lock<std::mutex> lock(synthesis_mutex, std::try_to_lock);
        if (!lock.owns_lock()) {
            fail("synthesis already in progress on this instance");
        }
        if (text.empty()) {
            fail("text must not be empty");
        }
        if (!callback) {
            fail("streaming synthesis needs a callback");
        }
        cancel_requested = false;

        const std::vector<DelayRow> prompt = build_checked_prompt(text);
        const int prompt_frames = (int) prompt.size();
        std::mt19937 rng(options.seed);
        DelayState state(backbone->config(), prompt, frontend->tokens().pad,
                frontend->tokens().im_end);
        SynthesisResult result;
        StreamProgress progress;

        const auto start = std::chrono::steady_clock::now();
        backbone->reset();
        generate_rows(prompt, state, rng, result, [&]() {
            return stream_ready_frames(state, prompt_frames, progress, callback, start, result);
        });
        if (!result.cancelled) {
            const bool delivered =
                    stream_ready_frames(state, prompt_frames, progress, callback, start, result) &&
                    close_segment(progress, callback, start, result);
            result.cancelled = !delivered;
        }
        result.generation_ms = elapsed_ms(start) - result.decode_ms;
        result.sample_rate = decoder->sample_rate();
        result.cancelled = result.cancelled || cancel_requested;
        if (!result.cancelled && progress.emitted_frames == 0) {
            fail("the model produced no audio frames");
        }
        return result;
    }
};

Engine::Engine(const EngineOptions & options) : impl_(new Impl) {
    impl_->options = options;
    impl_->load();
}

Engine::~Engine() = default;

SynthesisResult Engine::synthesize(const std::string & text) {
    return impl_->run(text);
}

SynthesisResult Engine::synthesize_stream(const std::string & text, const AudioCallback & callback) {
    return impl_->run_stream(text, callback);
}

void Engine::cancel() noexcept {
    impl_->cancel_requested = true;
}

int Engine::sample_rate() const noexcept {
    return impl_->decoder->sample_rate();
}

const char * Engine::backend_name() const noexcept {
    return impl_->backbone->backend_name();
}

} // namespace tts_cpp::moss
