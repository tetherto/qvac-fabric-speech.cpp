#include "parakeet/moss_transcribe.h"
#include "parakeet/moss_transcribe_fit.h"

#include "moss/transcribe_audio.h"
#include "moss/transcribe_coreml.h"
#include "moss/transcribe_model.h"
#include "moss/transcribe_networks.h"
#include "moss/transcribe_text.h"
#include "moss/transcribe_request.h"

#include "parakeet_ctc.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <stdexcept>

namespace parakeet::moss {
namespace {

using detail::TranscribeDecoder;
using detail::TranscribeMel;
using detail::TranscribeModel;
using detail::TranscribeTokenizer;

constexpr double DEGRADE_FIT_SECONDS = 300.0;
constexpr uint64_t DEGRADE_FIT_MARGIN_BYTES = 256ull * 1024 * 1024;

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("moss transcribe: " + message);
}

double elapsed_ms(std::chrono::steady_clock::time_point since) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - since).count();
}

int32_t greedy_token(const std::vector<float> & logits) {
    return (int32_t) std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
}

// Memory-degrade: with a budget, an instance whose Core ML-inclusive fit
// fails where the ggml-only fit passes resolves use = off ("memory"), so
// Core ML never turns a working ggml verdict into a refusal. Without a budget
// the options apply as given. The verdict comes from the totals the host
// assessed (TranscribeOptions::coreml_fit) when it passed them; otherwise
// from a projection of 300 s of audio with the default request and a 256 MiB
// margin (the parakeet::FitOptions defaults).
bool coreml_memory_degraded(const TranscribeOptions & options) {
#ifdef PARAKEET_USE_COREML
    if (!options.coreml.memory_budget_bytes.has_value() || *options.coreml.memory_budget_bytes < 0) return false;
    CoremlFitTotals totals;
    if (options.coreml_fit.has_value()) {
        totals = *options.coreml_fit;
    } else {
        const FitResult fit = fit_params(options, TranscribeRequest{}, DEGRADE_FIT_SECONDS, DEGRADE_FIT_MARGIN_BYTES);
        if (fit.status == FitStatus::Error) return false;
        totals.ggml_required_bytes = fit.ggml_required_bytes;
        totals.coreml_bytes = fit.coreml.bytes;
    }
    const uint64_t budget = static_cast<uint64_t>(*options.coreml.memory_budget_bytes);
    const uint64_t coreml_required = totals.ggml_required_bytes + totals.coreml_bytes;
    const bool ggml_fits = totals.ggml_required_bytes <= budget;
    const bool coreml_fits = coreml_required >= totals.ggml_required_bytes && coreml_required <= budget;
    return ggml_fits && !coreml_fits;
#else
    (void) options;
    return false;
#endif
}

} // namespace

struct TranscribeEngine::Impl {
    TranscribeOptions options;
    std::unique_ptr<TranscribeModel> model;
    std::unique_ptr<TranscribeTokenizer> tokenizer;
    std::unique_ptr<TranscribeMel> mel;
    std::unique_ptr<detail::TranscribeAudioEncoder> audio_encoder;
    std::atomic<bool> cancel_requested{false};
    std::mutex transcription_mutex;
    // Declared last: destroyed first, so no window holds the sidecar while
    // the model goes away.
    std::unique_ptr<detail::TranscribeCoremlEncoder> coreml;

    void load() {
        if (options.model_path.empty()) {
            fail("model_path is required");
        }
        if (!options.backends_dir.empty()) {
            ::parakeet::set_backends_directory(options.backends_dir);
        }
        coreml = std::make_unique<detail::TranscribeCoremlEncoder>(options.model_path, options.coreml,
                                                                   coreml_memory_degraded(options));
        model = std::make_unique<TranscribeModel>(options.model_path, options.use_gpu, options.n_threads, options.backend);
        detail::validate_transcribe_encoder(*model);
        detail::validate_transcribe_decoder(*model);
        tokenizer = std::make_unique<TranscribeTokenizer>(*model);
        mel = std::make_unique<TranscribeMel>(model->config().audio, detail::read_mel_filters(*model));
        attach_audio_encoder();
    }

    void attach_audio_encoder() {
        coreml->attach(model->config(), &cancel_requested);
        detail::TranscribeCoremlEncoder * sidecar = coreml.get();
        audio_encoder = std::make_unique<detail::TranscribeAudioEncoder>(*model, [sidecar] {
            return sidecar->acquire();
        });
    }

    void require_context(int prompt_tokens, int limit) const {
        const int64_t needed = (int64_t) prompt_tokens + limit;
        if (needed > model->config().text.n_ctx_train) {
            fail("audio and max_new_tokens need " + std::to_string(needed) + " positions; the decoder holds " +
                 std::to_string(model->config().text.n_ctx_train));
        }
    }

    void validate(const float * pcm, size_t samples, int sample_rate, const TranscribeRequest & request) const {
        if (sample_rate != model->config().audio.sample_rate) {
            fail("audio must be sampled at " + std::to_string(model->config().audio.sample_rate) + " Hz");
        }
        if (pcm == nullptr || samples == 0) {
            fail("audio must not be empty");
        }
        if (request.prompt.size() > detail::TRANSCRIBE_MAX_PROMPT_BYTES) {
            fail("prompt must be at most " + std::to_string(detail::TRANSCRIBE_MAX_PROMPT_BYTES) + " bytes");
        }
    }

    std::vector<float> chunk_embeddings(const float * pcm, size_t samples, size_t chunk) {
        const size_t length = (size_t) model->config().audio.chunk_samples;
        const size_t first = chunk * length;
        const size_t count = std::min(length, samples - first);
        const int tokens = detail::transcribe_chunk_tokens(model->config(), count);
        return audio_encoder->encode(mel->chunk(pcm + first, count), tokens);
    }

    bool encode_audio(const float * pcm, size_t samples, std::vector<float> & embeddings) {
        const size_t chunks = detail::transcribe_chunk_count(model->config().audio, samples);
        for (size_t chunk = 0; chunk < chunks; ++chunk) {
            if (cancel_requested) {
                return false;
            }
            const std::vector<float> encoded = chunk_embeddings(pcm, samples, chunk);
            embeddings.insert(embeddings.end(), encoded.begin(), encoded.end());
        }
        return true;
    }

    bool keep_going(const TranscribeProgress & progress, int generated, int limit) const {
        if (cancel_requested) {
            return false;
        }
        return !progress || progress(generated, limit);
    }

    bool generate(TranscribeDecoder & decoder, std::vector<float> logits, int limit,
                  const TranscribeProgress & progress, std::vector<int32_t> & generated) {
        const int32_t end = model->config().tokens.im_end;
        for (int count = 0; count < limit; ++count) {
            const int32_t token = greedy_token(logits);
            if (token == end) {
                return true;
            }
            generated.push_back(token);
            if (!keep_going(progress, (int) generated.size(), limit)) {
                return false;
            }
            if (count + 1 < limit) {
                logits = decoder.step(token);
            }
        }
        return true;
    }

    void finish(TranscribeResult & result, const std::vector<int32_t> & generated) const {
        result.generated_tokens = (int) generated.size();
        result.text = detail::strip_whitespace(tokenizer->decode(generated));
        result.segments = detail::parse_transcript(result.text);
    }

    TranscribeResult run(const float * pcm, size_t samples, const TranscribeRequest & request,
                         const TranscribeProgress & progress) {
        TranscribeResult result;
        const int limit = detail::resolved_transcribe_limit(model->config(), request);
        result.audio_tokens = detail::transcribe_audio_tokens(model->config(), samples);
        const std::vector<int32_t> prompt = detail::transcribe_prompt(model->config(), *tokenizer,
                result.audio_tokens, detail::resolved_transcribe_prompt(model->config(), request));
        result.prompt_tokens = (int) prompt.size();
        require_context(result.prompt_tokens, limit);

        const auto encode_start = std::chrono::steady_clock::now();
        std::vector<float> embeddings;
        audio_encoder->begin_run();
        result.cancelled = !encode_audio(pcm, samples, embeddings);
        result.encode_ms = elapsed_ms(encode_start);
        result.encoder_backend = audio_encoder->run_backend();
        result.coreml_pending = audio_encoder->run_pending();
        if (result.cancelled) {
            return result;
        }

        const auto prefill_start = std::chrono::steady_clock::now();
        TranscribeDecoder decoder(*model, result.prompt_tokens + limit);
        std::vector<float> logits = decoder.prefill(prompt, embeddings, detail::TRANSCRIBE_PREFILL_BATCH_TOKENS);
        result.prefill_ms = elapsed_ms(prefill_start);

        const auto decode_start = std::chrono::steady_clock::now();
        std::vector<int32_t> generated;
        result.cancelled = !generate(decoder, std::move(logits), limit, progress, generated);
        result.decode_ms = elapsed_ms(decode_start);
        finish(result, generated);
        return result;
    }

    TranscribeResult transcribe(const float * pcm, size_t samples, int sample_rate, const TranscribeRequest & request,
                                const TranscribeProgress & progress) {
        std::unique_lock<std::mutex> lock(transcription_mutex, std::try_to_lock);
        if (!lock.owns_lock()) {
            fail("transcription already in progress on this instance");
        }
        cancel_requested = false;
        validate(pcm, samples, sample_rate, request);
        return run(pcm, samples, request, progress);
    }
};

TranscribeEngine::TranscribeEngine(const TranscribeOptions & options) : impl_(new Impl) {
    impl_->options = options;
    impl_->load();
}

TranscribeEngine::~TranscribeEngine() = default;

TranscribeResult TranscribeEngine::transcribe(const float * pcm, size_t samples, int sample_rate,
                                              const TranscribeRequest & request,
                                              const TranscribeProgress & progress) {
    return impl_->transcribe(pcm, samples, sample_rate, request, progress);
}

void TranscribeEngine::cancel() noexcept {
    impl_->cancel_requested = true;
}

int TranscribeEngine::sample_rate() const noexcept {
    return impl_->model->config().audio.sample_rate;
}

const char * TranscribeEngine::backend_name() const noexcept {
    return impl_->model->backend_name();
}

bool TranscribeEngine::encoder_on_coreml() const noexcept {
    return impl_->coreml->ready();
}

std::string TranscribeEngine::coreml_status_json() const {
    return impl_->coreml->status_json();
}

std::string TranscribeEngine::coreml_construction_json() const {
    return impl_->coreml->construction_json();
}

bool TranscribeEngine::coreml_wait(int64_t timeout_ms) {
    return impl_->coreml->wait(timeout_ms);
}

} // namespace parakeet::moss
