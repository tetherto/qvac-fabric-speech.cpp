#include "tts-cpp/moss/speech.h"

#include "moss/speech_codec.h"
#include "moss/speech_generation.h"
#include "moss/speech_lm.h"
#include "moss/speech_prompt.h"
#include "moss/speech_request.h"

#include "backend_selection.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <mutex>
#include <random>
#include <stdexcept>

namespace tts_cpp::moss {
namespace {

using detail::SpeechCodec;
using detail::SpeechLM;
using detail::SpeechRow;
using detail::SpeechTextTokenizer;
using detail::SpeechTurn;
using detail::SpeechTurnRole;

constexpr int DEFAULT_MAX_NEW_TOKENS = 1000;
constexpr int MAX_NEW_TOKENS = 16384;
constexpr int MIN_NEW_TOKENS = 10;
constexpr int PREFILL_BATCH_TOKENS = 256;
constexpr size_t MAX_MESSAGES = 256;
constexpr size_t MAX_TEXT_BYTES = 16384;
constexpr int MIN_SAMPLE_RATE = 8000;
constexpr int MAX_SAMPLE_RATE = 192000;
constexpr double MAX_AUDIO_SECONDS = 600.0;
constexpr float MAX_TEMPERATURE = 10.0f;
constexpr float MAX_REPLY_SECONDS = 3600.0f;

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("moss speech: " + message);
}

double elapsed_ms(std::chrono::steady_clock::time_point since) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - since).count();
}

SpeechTurnRole turn_role(SpeechRole role) {
    switch (role) {
        case SpeechRole::System:    return SpeechTurnRole::System;
        case SpeechRole::Assistant: return SpeechTurnRole::Assistant;
        case SpeechRole::User:      break;
    }
    return SpeechTurnRole::User;
}

void validate_audio(const std::vector<float> & audio, int sample_rate, const char * what) {
    if (sample_rate < MIN_SAMPLE_RATE || sample_rate > MAX_SAMPLE_RATE) {
        fail(std::string(what) + " sample rate must be in [8000, 192000] Hz");
    }
    if ((double) audio.size() / sample_rate > MAX_AUDIO_SECONDS) {
        fail(std::string(what) + " is longer than " + std::to_string((int) MAX_AUDIO_SECONDS) + " s");
    }
}

void validate_message(const SpeechMessage & message) {
    if (message.audio.empty() == message.text.empty()) {
        fail("each message carries exactly one of text or audio");
    }
    if (message.text.size() > MAX_TEXT_BYTES) {
        fail("message text must be at most " + std::to_string(MAX_TEXT_BYTES) + " bytes");
    }
    if (!message.audio.empty()) {
        validate_audio(message.audio, message.sample_rate, "message audio");
    }
}

void validate_messages(const std::vector<SpeechMessage> & messages) {
    if (messages.empty() || messages.size() > MAX_MESSAGES) {
        fail("a request needs 1.." + std::to_string(MAX_MESSAGES) + " messages");
    }
    for (const SpeechMessage & message : messages) {
        validate_message(message);
    }
    if (messages.back().role != SpeechRole::User) {
        fail("the last message must be the user's turn");
    }
}

void validate_sampling(const SpeechRequest & request) {
    if (!request.greedy && !(request.temperature > 0.0f && request.temperature <= MAX_TEMPERATURE)) {
        fail("temperature must be in (0, 10]");
    }
    if (!(request.top_p > 0.0f && request.top_p <= 1.0f) || request.top_k < 0) {
        fail("top_p must be in (0, 1] and top_k must not be negative");
    }
    if (request.max_new_tokens < 0 || request.max_new_tokens > MAX_NEW_TOKENS) {
        fail("max_new_tokens must be 0.." + std::to_string(MAX_NEW_TOKENS));
    }
    if (!std::isfinite(request.max_reply_seconds) || request.max_reply_seconds < 0.0f ||
        request.max_reply_seconds > MAX_REPLY_SECONDS) {
        fail("max_reply_seconds must be in [0, 3600]");
    }
}

detail::SpeechSampling sampling_of(const SpeechRequest & request) {
    return {!request.greedy, request.temperature, request.top_p, request.top_k};
}

} // namespace

namespace detail {

void validate_speech_request(const SpeechRequest & request) {
    validate_messages(request.messages);
    validate_sampling(request);
    if (!request.voice.empty()) {
        validate_audio(request.voice, request.voice_sample_rate, "voice prompt");
    }
}

} // namespace detail

struct SpeechEngine::Impl {
    SpeechOptions options;
    std::unique_ptr<SpeechLM> lm;
    std::unique_ptr<SpeechCodec> codec;
    std::unique_ptr<SpeechTextTokenizer> tokenizer;
    std::atomic<bool> cancel_requested{false};
    std::mutex respond_mutex;

    void load() {
        if (options.model_path.empty() || options.codec_path.empty()) {
            fail("model_path and codec_path are required");
        }
        if (!options.backends_dir.empty()) {
            ::tts_cpp::detail::set_backends_directory(options.backends_dir);
        }
        lm = std::make_unique<SpeechLM>(options.model_path, options.use_gpu, options.n_threads);
        codec = std::make_unique<SpeechCodec>(options.codec_path, options.use_gpu, options.n_threads);
        tokenizer = std::make_unique<SpeechTextTokenizer>(lm->tokenizer_tokens(), lm->tokenizer_merges(),
                lm->tokenizer_types());
    }

    float tokens_per_second() const {
        const detail::SpeechVqConfig & vq = codec->tokenizer().config();
        return (float) vq.sample_rate / (float) vq.samples_per_token();
    }

    SpeechTurn turn_of(const SpeechMessage & message) {
        SpeechTurn turn{turn_role(message.role), message.text, {}, !message.audio.empty()};
        if (turn.is_audio) {
            turn.audio_codes = codec->encode(message.audio, message.sample_rate);
        }
        return turn;
    }

    std::vector<SpeechTurn> turns_of(const std::vector<SpeechMessage> & messages) {
        std::vector<SpeechTurn> turns;
        for (const SpeechMessage & message : messages) {
            turns.push_back(turn_of(message));
        }
        return turns;
    }

    detail::SpeechLimits limits_of(const SpeechRequest & request) const {
        const int max_new = request.max_new_tokens > 0 ? request.max_new_tokens : DEFAULT_MAX_NEW_TOKENS;
        const int reply_codes = (int) std::ceil(request.max_reply_seconds * tokens_per_second());
        return {max_new, MIN_NEW_TOKENS, reply_codes};
    }

    bool keep_going(const SpeechProgress & progress, int generated, int limit) const {
        return !cancel_requested && (!progress || progress(generated, limit));
    }

    bool generate(detail::SpeechGenerationState & state, detail::SpeechLogits logits, const SpeechRequest & request,
                  const detail::SpeechLimits & limits, const SpeechProgress & progress) {
        std::mt19937 rng(request.seed);
        while (!state.stopping()) {
            const SpeechRow row = state.next(std::move(logits), sampling_of(request), rng);
            if (!keep_going(progress, (int) state.generated().size(), limits.max_new_tokens)) {
                return false;
            }
            if (!state.stopping()) {
                logits = lm->step(row);
            }
        }
        return true;
    }

    const detail::SpeechVoice & reply_voice(const SpeechRequest & request, detail::SpeechVoice & custom) {
        if (request.voice.empty()) {
            return codec->default_voice();
        }
        custom = codec->voice(request.voice, request.voice_sample_rate);
        return custom;
    }

    void speak(SpeechResult & result, const std::vector<int32_t> & codes, const SpeechRequest & request) {
        const auto decode_start = std::chrono::steady_clock::now();
        detail::SpeechVoice custom;
        detail::SpeechDecodeOptions decode;
        decode.cancel = &cancel_requested;
        result.pcm = codec->decode(codes, reply_voice(request, custom), decode);
        result.decode_ms = elapsed_ms(decode_start);
        result.cancelled = cancel_requested.load();
    }

    SpeechResult run(const SpeechRequest & request, const SpeechProgress & progress) {
        SpeechResult result;
        result.sample_rate = codec->sample_rate();
        const auto encode_start = std::chrono::steady_clock::now();
        const std::vector<SpeechRow> prompt = detail::speech_prompt(lm->config(), *tokenizer,
                turns_of(request.messages), !request.text_reply);
        result.encode_ms = elapsed_ms(encode_start);
        result.prompt_tokens = (int) prompt.size();
        const detail::SpeechLimits limits = limits_of(request);
        if ((int64_t) prompt.size() + limits.max_new_tokens > lm->config().n_ctx_train) {
            fail("the conversation and max_new_tokens exceed the model context");
        }

        const auto prefill_start = std::chrono::steady_clock::now();
        lm->begin(result.prompt_tokens + limits.max_new_tokens);
        detail::SpeechLogits logits = lm->prefill(prompt, PREFILL_BATCH_TOKENS);
        result.prefill_ms = elapsed_ms(prefill_start);

        const auto generate_start = std::chrono::steady_clock::now();
        detail::SpeechGenerationState state(lm->config().tokens, prompt.back(), limits);
        result.cancelled = !generate(state, std::move(logits), request, limits, progress);
        result.generate_ms = elapsed_ms(generate_start);
        result.generated_tokens = (int) state.generated().size();
        result.truncated = state.truncated();
        result.text = detail::speech_reply_text(*tokenizer, state.generated());
        const std::vector<int32_t> codes = detail::speech_reply_codes(state.generated(), lm->config().tokens);
        result.reply_tokens = (int) codes.size();
        if (!result.cancelled && !request.text_reply) {
            speak(result, codes, request);
        }
        return result;
    }

    SpeechResult respond(const SpeechRequest & request, const SpeechProgress & progress) {
        std::unique_lock<std::mutex> lock(respond_mutex, std::try_to_lock);
        if (!lock.owns_lock()) {
            fail("a response is already in progress on this instance");
        }
        cancel_requested = false;
        detail::validate_speech_request(request);
        return run(request, progress);
    }
};

SpeechEngine::SpeechEngine(const SpeechOptions & options) : impl_(new Impl) {
    impl_->options = options;
    impl_->load();
}

SpeechEngine::~SpeechEngine() = default;

SpeechResult SpeechEngine::respond(const SpeechRequest & request, const SpeechProgress & progress) {
    return impl_->respond(request, progress);
}

void SpeechEngine::cancel() noexcept {
    impl_->cancel_requested = true;
}

int SpeechEngine::sample_rate() const noexcept {
    return impl_->codec->sample_rate();
}

float SpeechEngine::tokens_per_second() const noexcept {
    return impl_->tokens_per_second();
}

const char * SpeechEngine::backend_name() const noexcept {
    return impl_->lm->backend_name();
}

} // namespace tts_cpp::moss
