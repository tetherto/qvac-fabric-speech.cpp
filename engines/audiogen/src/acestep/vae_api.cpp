#include "audiogen-cpp/acestep/vae.h"

#include "vae_ggml.h"              // internal: VaeModel + vae_model_*
#include "vae_access.h"
#include "vae_coreml.h"
#include "vae_coreml_windows.h"
#include "vae_encode_windows.h"

#include "acestep/backend_registry.h"
#include "acestep/engine_backends.h"

#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>

namespace tts_cpp::acestep {

struct Vae::Impl {
    ggml_backend_t backend = nullptr;  // owned
    VaeModel *     model   = nullptr;
    std::string    backend_name = "CPU";
    // The decoder's Core ML stage: owned by this Vae when loaded standalone,
    // shared with the Engine otherwise. Null: ggml only.
    std::shared_ptr<detail::vae_coreml_stage> coreml;
    // The ggml weights came from the file the Core ML decoder is generated
    // from (vae_coreml_stage::serves_loaded_source). When false, no decode of
    // this Vae acquires the decoder.
    bool coreml_source_ok = true;

    mutable std::mutex  last_mu;
    mutable std::string last_backend = detail::VAE_GGML_BACKEND;
    mutable bool        last_pending = false;

    ~Impl() {
        if (model) vae_model_free(model);
        if (backend) ggml_backend_free(backend);
    }
};

namespace {

enum class CoremlDecodeStatus { done, too_short, failed, cancelled };

void copy_window_core(const VaeCoremlWindow & w, const std::vector<float> & pcm_win, std::vector<float> & pcm_out) {
    const size_t samples_per_frame = (size_t) VAE_ENCODER_UPSAMPLE * VAE_PCM_CHANNELS;
    const size_t core_off = (size_t) (w.core_a - w.win_a) * samples_per_frame;
    const size_t core_len = (size_t) (w.core_b - w.core_a) * samples_per_frame;
    const size_t dst      = (size_t) w.core_a * samples_per_frame;
    std::copy(pcm_win.begin() + core_off, pcm_win.begin() + core_off + core_len, pcm_out.begin() + dst);
}

// The whole decode on the acquired decoder, in fixed overlapped windows of its
// exported length (the last one end-aligned), with the trimmed-core stitching
// of the ggml chunked decode.
CoremlDecodeStatus coreml_decode(detail::vae_coreml_decoder & dec, const float * latent, int T_latent,
                                 std::vector<float> & pcm_out, const Vae::ProgressCb & on_progress) {
    const int window_frames = dec.window();
    const std::vector<VaeCoremlWindow> plan = vae_coreml_plan_windows(T_latent, window_frames, VAE_COREML_OVERLAP);
    if (plan.empty()) return CoremlDecodeStatus::too_short;

    pcm_out.assign((size_t) T_latent * VAE_ENCODER_UPSAMPLE * VAE_PCM_CHANNELS, 0.0f);
    std::vector<float> pcm_win((size_t) window_frames * VAE_ENCODER_UPSAMPLE * VAE_PCM_CHANNELS);
    const int n_windows = (int) plan.size();
    for (int i = 0; i < n_windows; ++i) {
        const VaeCoremlWindow & w = plan[i];
        if (!dec.decode(latent + (size_t) w.win_a * VAE_LATENT_CHANNELS, pcm_win.data())) {
            pcm_out.clear();
            return CoremlDecodeStatus::failed;
        }
        copy_window_core(w, pcm_win, pcm_out);
        if (on_progress && !on_progress(i + 1, n_windows)) {
            pcm_out.clear();
            return CoremlDecodeStatus::cancelled;
        }
    }
    return CoremlDecodeStatus::done;
}

}  // namespace

Vae::Vae() : impl_(std::make_unique<Impl>()) {}
Vae::~Vae() = default;

std::unique_ptr<Vae> detail::VaeAccess::load(const std::string & gguf_path, const VaeOptions & opts,
                                             std::shared_ptr<vae_coreml_stage> stage) {
    std::unique_ptr<Vae> v(new Vae());
    v->impl_->coreml = std::move(stage);

    // The two custom ops (col2im_1d, snake) have CPU, Metal and Vulkan kernels
    // in the fabric ggml, so the decode/encode graph can run on a GPU
    // backend. n_gpu_layers > 0 opts in (Metal on Apple, Vulkan elsewhere);
    // falls back to CPU when no GPU backend is registered/available. Backends are
    // acquired through the ggml registry (see backend_registry.h) so the CPU path
    // resolves on arm64 dlopen (GGML_CPU_ALL_VARIANTS) builds too. When Vae::load
    // runs standalone (not via Engine), backends_dir loads the modules first.
    load_backends(opts.backends_dir);

    // Resolution shared with the engine and the memory-fit projection
    // (engine_backends.h), so all three agree by construction.
    BackendRequest request;
    request.n_gpu_layers   = opts.n_gpu_layers;
    request.backend        = opts.backend;
    request.n_threads      = opts.n_threads;
    request.verbose        = opts.verbose;
    ggml_backend_t backend = resolve_vae_backend(request);
    if (!backend) throw std::runtime_error("acestep-vae: failed to init backend '" + opts.backend + "'");
    v->impl_->backend = backend;

    // The identity of the file the weights were read from, compared with the
    // one the decoder's slot generates from, at every (re)load.
    const std::shared_ptr<detail::vae_coreml_stage> & stage_ref = v->impl_->coreml;
    bool source_ok = true;
    VaeModel * model = vae_model_load(gguf_path, backend, opts.with_encoder, opts.verbose, [&](int fd) {
        if (stage_ref && stage_ref->active()) source_ok = stage_ref->serves_loaded_source(fd);
    });
    if (!model) throw std::runtime_error("acestep-vae: failed to load VAE GGUF: " + gguf_path);
    v->impl_->model = model;
    v->impl_->coreml_source_ok = source_ok;
    if (!source_ok) {
        fprintf(stderr,
                "[acestep-vae] %s is not the file the Core ML VAE decoder is generated from (it was replaced "
                "after the engine was created); this VAE decodes on ggml%s\n",
                gguf_path.c_str(), v->impl_->coreml->strict() ? ", which the Core ML strict policy refuses" : "");
    }
    const char * bn   = ggml_backend_name(backend);
    v->impl_->backend_name = bn ? bn : "CPU";
    return v;
}

bool detail::VaeAccess::coreml_source_ok(const Vae & v) { return v.impl_->coreml_source_ok; }

std::unique_ptr<Vae> Vae::load(const std::string & gguf_path, const VaeOptions & opts) {
    // The Core ML stage first: it validates the options, may throw under
    // strict, and starts any construction work (prewarm all) while the ggml
    // weights load. A standalone Vae has no low-memory mode, so a construction
    // load is kept.
    std::shared_ptr<detail::vae_coreml_stage> stage =
        detail::vae_coreml_stage::create(gguf_path, opts.coreml,
                                         /*memory_degrade=*/opts.coreml.memory_budget_bytes.has_value() &&
                                             opts.coreml.memory_degrade.value_or(false),
                                         /*release_after_construction=*/false);
    std::unique_ptr<Vae> v = detail::VaeAccess::load(gguf_path, opts, stage);
    if (opts.verbose && stage) {
        fprintf(stderr, "[acestep-vae] Core ML decoder: route %s%s%s\n", stage->route().route.c_str(),
                stage->route().reason.empty() ? "" : ", ", stage->route().reason.c_str());
    }
    return v;
}

std::vector<float> Vae::decode(const std::vector<float> & latent, int T_latent,
                               const ProgressCb & on_progress) const {
    if (T_latent <= 0 || (int) latent.size() < T_latent * 64) return {};
    std::string backend = detail::VAE_GGML_BACKEND;
    bool        pending = false;
    auto finish = [&](std::vector<float> pcm) {
        std::lock_guard<std::mutex> lock(impl_->last_mu);
        impl_->last_backend = backend;
        impl_->last_pending = pending;
        return pcm;
    };

    // A VAE loaded from another file than the decoder's never acquires it.
    if (impl_->coreml && impl_->coreml->active() && !impl_->coreml_source_ok) {
        if (impl_->coreml->strict()) {
            fprintf(stderr, "[acestep-vae] the loaded VAE GGUF is not the file the Core ML decoder is generated "
                            "from and Core ML strict policy is enabled; failing instead of the ggml decode\n");
            return finish({});
        }
    } else if (impl_->coreml && impl_->coreml->active()) {
        // The backend of the whole decode is chosen here, once: the decoder
        // this acquisition returns, or ggml.
        detail::vae_coreml_call call;
        try {
            call = impl_->coreml->acquire(T_latent);
        } catch (const std::exception & e) {
            fprintf(stderr, "[acestep-vae] %s\n", e.what());
            return finish({});
        }
        pending = call.pending;
        if (call.cancelled) return finish({});  // neither backend: the caller cancelled
        if (call.decoder) {
            const bool strict = impl_->coreml->strict();
            std::vector<float> pcm;
            switch (coreml_decode(*call.decoder, latent.data(), T_latent, pcm, on_progress)) {
                case CoremlDecodeStatus::done:
                    backend = call.decoder->label();
                    return finish(std::move(pcm));
                case CoremlDecodeStatus::cancelled:
                    backend = call.decoder->label();
                    return finish({});
                case CoremlDecodeStatus::too_short:
                    if (strict) {
                        fprintf(stderr, "[acestep-vae] T_latent=%d is shorter than the Core ML window (%d) and "
                                        "Core ML strict policy is enabled; failing instead of the ggml decode\n",
                                T_latent, call.decoder->window());
                        return finish({});
                    }
                    break;  // today's rule: a latent shorter than one window decodes on ggml
                case CoremlDecodeStatus::failed:
                    // Retire the sidecar and redo the whole decode on ggml, so
                    // no output mixes a failed decode's windows.
                    fprintf(stderr, "[acestep-vae] Core ML VAE decoder (%s) failed; retiring it%s\n",
                            call.decoder->label(), strict ? "" : ", decoding on ggml");
                    if (call.retire) call.retire("a Core ML VAE prediction failed");
                    if (strict) return finish({});
                    break;
            }
        }
    }
    std::vector<float> pcm;
    const int T_audio = vae_model_decode(impl_->model, latent.data(), T_latent, pcm, on_progress);
    if (T_audio < 0) return finish({});
    return finish(std::move(pcm));
}

std::vector<float> Vae::encode(const std::vector<float> & pcm_interleaved, int frames,
                               int * T_latent_out,
                               const ProgressCb & on_progress) const {
    const int window_count = vae_encode_window_count(frames);
    int window_index = 0;
    const VaeWindowEncoder encode = [this, &on_progress, &window_index, window_count](
                                        const float * pcm, int window_frames,
                                        std::vector<float> & latent) {
        ProgressCb window_progress;
        if (on_progress) {
            window_progress = [&on_progress, &window_index, window_count](
                                  int done, int total) {
                return on_progress(window_index * total + done,
                                   window_count * total);
            };
        }
        const int encoded = vae_model_encode(
            impl_->model, pcm, window_frames, latent, window_progress);
        ++window_index;
        return encoded;
    };
    return encode_vae_pcm_bounded(pcm_interleaved, frames, encode, T_latent_out);
}

std::string Vae::last_decode_backend() const {
    std::lock_guard<std::mutex> lock(impl_->last_mu);
    return impl_->last_backend;
}

bool Vae::last_decode_coreml_pending() const {
    std::lock_guard<std::mutex> lock(impl_->last_mu);
    return impl_->last_pending;
}

std::string Vae::coreml_status() const {
    return impl_->coreml ? impl_->coreml->status_json() : std::string("{}");
}

bool        Vae::has_encoder() const   { return vae_model_has_encoder(impl_->model); }
int         Vae::sample_rate() const   { return 48000; }
int         Vae::upsample_factor() const { return 1920; }
std::string Vae::backend_name() const  { return impl_->backend_name; }

} // namespace tts_cpp::acestep
