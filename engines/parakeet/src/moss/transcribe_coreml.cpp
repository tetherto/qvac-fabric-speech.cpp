#include "moss/transcribe_coreml.h"

#include "moss/transcribe_networks.h"
#include "parakeet_coreml_path.h"

#ifdef PARAKEET_USE_COREML
#include "coreml/parakeet-encoder.h"
#include "coreml/parakeet_coreml_load_error.h"
#include "coreml/parakeet_coreml_sidecars.h"
#include "moss/transcribe_coreml_slot.h"
#include "parakeet_ctc.h"
#include "parakeet_log.h"

#include <cstdio>
#endif

#include <stdexcept>

namespace parakeet::moss::detail {
namespace {

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("moss transcribe: " + message);
}

int chunk_token_capacity(const TranscribeConfig & config) {
    return config.encoder.n_ctx / config.merge_size;
}

void scatter_band(const std::vector<float> & mel, int band, int n_mels, int frames, std::vector<float> & out) {
    const size_t row = (size_t) band * frames;
    for (int frame = 0; frame < frames; ++frame) {
        out[(size_t) frame * n_mels + band] = mel[row + frame];
    }
}

#ifdef PARAKEET_USE_COREML
bool matches_window(const parakeet_coreml_context * context, const TranscribeEncoderWindow & window) {
    return parakeet_coreml_fixed_mel_frames(context, window.n_mels) == window.frames &&
           parakeet_coreml_fixed_output_rows(context, window.width) == window.tokens;
}

// Runs on the runtime's load lane with the resolved compute units; never
// reads the environment. A bundle whose input or output is not exactly the
// model's window is a verified incompatibility. A generated bundle
// (request.tmpl set) loads under its cache entry's lease.
pcm::LoadResult load_transcribe_encoder(const pcm::LoadRequest & request) {
    pcm::LoadResult result;
    if (!pcm::generation::begin_load(request, result)) return result;
    TranscribeEncoderWindow window;
    const bool known = transcribe_encoder_window_of(request.geometry, window);
    parakeet_coreml_load_error error{};
    parakeet_coreml_context * context = parakeet_coreml_init_with_units_error(
        request.artifact_path.c_str(), pcm::to_string(request.units), &error);
    if (context != nullptr && (!known || !matches_window(context, window))) {
        error = parakeet_coreml_load_error{};
        error.incompatible = 1;
        std::snprintf(error.message, sizeof(error.message), "window %lld mel frames -> %lld tokens, expected %d -> %d",
                      (long long) parakeet_coreml_fixed_mel_frames(context, window.n_mels),
                      (long long) parakeet_coreml_fixed_output_rows(context, window.width), window.frames,
                      window.tokens);
        parakeet_coreml_free(context);
        context = nullptr;
    }
    if (context == nullptr) {
        result.error = sidecar_load_error("Core ML " + request.stage, error);
        PARAKEET_LOG_WARN("moss transcribe: failed to load the Core ML encoder at '%s' (%s); using ggml\n",
                          request.artifact_path.c_str(), result.error.message.c_str());
        // The framework refused a generated bundle (or it is not the model's
        // window): end_load quarantines it, unless the error is a resource
        // one, and the slot prepares it again.
        result.rejected = request.tmpl != nullptr;
        pcm::generation::end_load(request, result);
        return result;
    }
    PARAKEET_LOG_INFO("moss transcribe: Core ML encoder loaded from '%s'\n", request.artifact_path.c_str());
    result.model = std::shared_ptr<void>(context, [](void * p) {
        parakeet_coreml_free(static_cast<parakeet_coreml_context *>(p));
    });
    pcm::generation::end_load(request, result);
    return result;
}
#endif

} // namespace

#ifdef PARAKEET_USE_COREML
struct TranscribeCoremlEncoder::Impl {
    std::string model_path;
    CoremlSidecars sidecars;
    pcm::SourceStat before;
    bool before_ok = false;
    // Read by the slot while it is registered; kept for the model's life.
    pcm::generation::Candidate candidate;
    std::shared_ptr<pcm::Slot> slot;
    int n_mels = 0;
    const std::atomic<bool> * cancel = nullptr;

    Impl(const std::string & path, const CoremlOptions & options, bool memory_degraded)
        : model_path(path), sidecars(options, nullptr, memory_degraded, TRANSCRIBE_COREML_ENV_PREFIX) {
        before_ok = pcm::generation::source_stat(path, before);
    }

    void attach(const TranscribeConfig & config, const std::atomic<bool> * flag) {
        cancel = flag;
        n_mels = config.audio.n_mels;
        if (!sidecars.resolved().enabled()) {
            PARAKEET_LOG_INFO("moss transcribe: Core ML disabled (%s); using ggml\n",
                              sidecars.resolved().disabled_reason.c_str());
        }
        // The identity of the GGUF the weights came from: path samples before
        // the first open and after the last read (the model opens it twice).
        pcm::SourceStat after;
        const bool after_ok = pcm::generation::source_stat(model_path, after);
        const pcm::SourceStat source = pcm::generation::loaded_source(before, before_ok, after, after_ok, model_path);
        const TranscribeEncoderWindow window{config.audio.n_mels, config.audio.chunk_frames,
                                             chunk_token_capacity(config), config.text.n_embd};
        candidate = probe_transcribe_encoder_template(sidecars, model_path);
        const SidecarRequest request =
            transcribe_encoder_request(model_path, window, &candidate, &load_transcribe_encoder, source);
        slot = sidecars.add(request);
        const pcm::SlotStatus status = slot->status();
        if (slot->spec().route == pcm::Route::generate) {
            PARAKEET_LOG_INFO("moss transcribe: Core ML encoder is generated from the GGUF into '%s'\n",
                              sidecars.resolved().cache_dir.c_str());
        } else if (slot->spec().route == pcm::Route::prebuilt) {
            PARAKEET_LOG_INFO("moss transcribe: Core ML encoder available at '%s'\n", request.path.c_str());
        } else if (sidecars.resolved().enabled()) {
            PARAKEET_LOG_INFO("moss transcribe: no Core ML encoder (%s); using ggml\n",
                              pcm::state_label(status.state, status.reason).c_str());
        }
        sidecars.start();
    }

    EncoderRoute acquire() {
        if (!slot) return {};
        return route_from_acquired(slot->acquire(pcm::Demand::now, std::nullopt, cancel), coreml_encoder_model_ops(),
                                   n_mels, sidecars.resolved().strict());
    }

    bool ready() const {
        return slot && static_cast<bool>(acquire_if_ready(*slot).model);
    }
};

TranscribeCoremlEncoder::TranscribeCoremlEncoder(const std::string & model_path, const CoremlOptions & options,
                                                 bool memory_degraded)
    : impl_(new Impl(model_path, options, memory_degraded)) {}

void TranscribeCoremlEncoder::attach(const TranscribeConfig & config, const std::atomic<bool> * cancel) {
    impl_->attach(config, cancel);
}

EncoderRoute TranscribeCoremlEncoder::acquire() { return impl_->acquire(); }

bool TranscribeCoremlEncoder::ready() const { return impl_->ready(); }

std::string TranscribeCoremlEncoder::status_json() const { return impl_->sidecars.status_json(); }

std::string TranscribeCoremlEncoder::construction_json() const { return impl_->sidecars.construction_json(); }

bool TranscribeCoremlEncoder::wait(int64_t timeout_ms) {
    return impl_->sidecars.wait_construction(timeout_ms, impl_->cancel);
}
#else
namespace {

// A group with nothing to prepare: finished and successful at once.
std::string empty_construction_json() {
    return "{\"group\":\"construction\",\"nothing_to_prepare\":true,\"skipped_route_none\":[],\"slots\":[],"
           "\"success\":true}";
}

} // namespace

struct TranscribeCoremlEncoder::Impl {};

TranscribeCoremlEncoder::TranscribeCoremlEncoder(const std::string &, const CoremlOptions &, bool) {}

void TranscribeCoremlEncoder::attach(const TranscribeConfig &, const std::atomic<bool> *) {}

EncoderRoute TranscribeCoremlEncoder::acquire() { return EncoderRoute::ggml(); }

bool TranscribeCoremlEncoder::ready() const { return false; }

std::string TranscribeCoremlEncoder::status_json() const { return "{}"; }

std::string TranscribeCoremlEncoder::construction_json() const { return empty_construction_json(); }

bool TranscribeCoremlEncoder::wait(int64_t) { return true; }
#endif

// Drops the slot; never waits for a load in flight.
TranscribeCoremlEncoder::~TranscribeCoremlEncoder() = default;

std::string transcribe_encoder_sidecar_path(const std::string & model_path) {
    return ::parakeet::coreml_encoder_sidecar_path(model_path);
}

std::vector<float> frame_major_mel(const std::vector<float> & mel, int n_mels, int frames) {
    if (mel.size() != (size_t) n_mels * frames) {
        fail("mel window has the wrong size for the Core ML encoder");
    }
    std::vector<float> out(mel.size());
    for (int band = 0; band < n_mels; ++band) {
        scatter_band(mel, band, n_mels, frames, out);
    }
    return out;
}

TranscribeAudioEncoder::TranscribeAudioEncoder(TranscribeModel & model, TranscribeRouteSource routes)
    : model_(model), routes_(std::move(routes)) {}

void TranscribeAudioEncoder::begin_run() {
    tally_ = EncoderBackendTally{};
}

// Every window is a full 30 s window (the last one zero-padded), so a
// sidecar of exactly the model's window serves every one of them.
EncoderEligibility TranscribeAudioEncoder::eligibility(const EncoderRoute & route) const {
    if (!route.holds()) return EncoderEligibility::ggml;
    return route.geometry().fixed_mel_frames == model_.config().audio.chunk_frames ? EncoderEligibility::coreml
                                                                                    : EncoderEligibility::cannot_serve;
}

std::vector<float> TranscribeAudioEncoder::encode(const std::vector<float> & mel, int tokens) {
    const TranscribeConfig & config = model_.config();
    const EncoderRoute route = routes_ ? routes_() : EncoderRoute::ggml();
    tally_.add_route(route);
    std::vector<float> embeddings;
    bool used_coreml = false;
    dispatch_on_route(
        route, eligibility(route),
        [&] {
            const int capacity = chunk_token_capacity(config);
            if (tokens < 1 || tokens > capacity) fail("chunk token count is outside the Core ML encoder window");
            const std::vector<float> frames = frame_major_mel(mel, config.audio.n_mels, config.audio.chunk_frames);
            std::vector<float> rows((size_t) capacity * config.text.n_embd, 0.0f);
            const int rc = route.encode(config.audio.chunk_frames, config.audio.chunk_frames, config.audio.n_mels,
                                        frames.data(), capacity, config.text.n_embd, rows.data());
            if (rc == 0) embeddings = leading_rows(rows, tokens);
            return rc;
        },
        [&] {
            embeddings = encode_audio_chunk(model_, mel, tokens, false).embeddings;
            return 0;
        },
        used_coreml);
    tally_.add(used_coreml, route.coreml_label());
    return embeddings;
}

std::string TranscribeAudioEncoder::run_backend() const {
    if (tally_.coreml + tally_.ggml == 0) {
        return std::string();
    }
    return tally_.label();
}

bool TranscribeAudioEncoder::run_pending() const {
    return tally_.pending;
}

std::vector<float> TranscribeAudioEncoder::leading_rows(const std::vector<float> & rows, int tokens) const {
    const size_t count = (size_t) tokens * model_.config().text.n_embd;
    return std::vector<float>(rows.begin(), rows.begin() + (std::ptrdiff_t) count);
}

} // namespace parakeet::moss::detail
