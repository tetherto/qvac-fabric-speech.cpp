#include "audiogen-cpp/minimax/engine.h"

#include "minimax/backend.h"
#include "minimax/logic.h"
#include "minimax/mm3-pipeline.h"
#include "minimax/request-utils.h"

#include <atomic>
#include <cmath>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>

namespace tts_cpp::minimax {
namespace {

constexpr const char * kModelLicense = "MiniMax-Music3 Community License";

std::recursive_mutex & engine_mutex() {
    static std::recursive_mutex mutex;
    return mutex;
}

int & active_engine_count() {
    static int count = 0;
    return count;
}

EngineOptions resolve_model_paths(const EngineOptions & input) {
    EngineOptions options = input;
    const detail::ModelPair pair =
        detail::resolve_model_pair(options.model_dir, options.lm_model_path, options.synth_model_path);
    options.lm_model_path = pair.lm;
    options.synth_model_path = pair.synth;
    return options;
}

detail::ModelCompatibility model_compatibility(const MM3Model & model) {
    const MM3LmConfig & lm = model.lm_cfg;
    const MM3SynthConfig & synth = model.synth_cfg;
    detail::ModelCompatibility result;
    result.lm_embedding = lm.embedding_length;
    result.lm_codebooks = lm.num_codebooks;
    result.lm_acoustic_vocab = lm.acoustic_vocab_size;
    result.frame_rate = lm.frame_rate;
    result.max_audio_frames = lm.max_audio_frames;
    result.max_prompt_tokens = lm.max_prompt_tokens;
    result.depth_embedding = synth.depth.embedding_length;
    result.depth_codebooks = synth.depth.num_codebooks;
    result.depth_acoustic_vocab = synth.depth.audio_vocab_size;
    result.condition_layers = synth.cond.num_layers;
    result.condition_hidden = synth.cond.hidden_dim;
    result.condition_out = synth.cond.out_dim;
    result.condition_rate = {static_cast<int>(synth.cond.input_sampling_rate),
                             static_cast<int>(synth.cond.input_hop_length),
                             static_cast<int>(synth.cond.output_sampling_rate),
                             static_cast<int>(synth.cond.output_hop_length)};
    result.dit_condition = synth.dit.condition_dim;
    result.dit_channels = synth.dit.in_channels;
    result.window_frames = synth.dit.window_frames;
    result.hop_frames = synth.dit.hop_frames;
    result.window_latents = synth.dit.window_latents;
    result.hop_latents = synth.dit.hop_latents;
    result.vocoder_latent_channels = synth.voc.latent_channels;
    result.vocoder_sampling_rate = synth.voc.sampling_rate;
    result.vocoder_channels = synth.voc.channels;
    result.vocoder_upsample = synth.voc.total_upsample;
    result.components = synth.components;
    return result;
}

void probe_model_files(MM3Model & model, const EngineOptions & options) {
    model.models_dir = options.model_dir;
    mm3_probe_file(options.lm_model_path, &model.lm_file, &model.lm_cfg, nullptr, &model.meta_errors);
    mm3_probe_file(options.synth_model_path, &model.synth_file, nullptr, &model.synth_cfg, &model.meta_errors);
    if (model.lm_file.arch != "qwen3") {
        model.meta_errors.push_back("LM general.architecture must be qwen3");
    }
    if (model.synth_file.arch != "mm3") {
        model.meta_errors.push_back("synth general.architecture must be mm3");
    }
    if (model.lm_file.license != kModelLicense || model.synth_file.license != kModelLicense) {
        model.meta_errors.push_back("both GGUF files must declare the MiniMax-Music3 Community License");
    }
    if (model.synth_cfg.components.size() != 4) {
        model.meta_errors.push_back("synth GGUF must contain depth, cond, dit, and vocoder components");
    }
    const std::vector<std::string> compatibility_errors =
        detail::validate_model_compatibility(model_compatibility(model));
    model.meta_errors.insert(model.meta_errors.end(), compatibility_errors.begin(), compatibility_errors.end());
    if (!model.meta_errors.empty()) {
        throw std::runtime_error("minimax engine: " + model.meta_errors.front());
    }
}

int resolve_thread_count(int requested) {
    if (requested > 0) {
        return requested;
    }
    const unsigned int available = std::thread::hardware_concurrency();
    return available > 0 ? static_cast<int>(available) : 1;
}

uint64_t resolve_seed(int64_t seed) {
    if (seed >= 0) {
        return static_cast<uint64_t>(seed);
    }
    std::random_device random;
    return (static_cast<uint64_t>(random()) << 32) ^ static_cast<uint64_t>(random());
}

void interleave_planar_stereo(const std::vector<float> & planar, int64_t samples, std::vector<float> & interleaved) {
    interleaved.resize(static_cast<size_t>(samples) * 2);
    for (int64_t sample = 0; sample < samples; ++sample) {
        interleaved[static_cast<size_t>(sample) * 2] = planar[static_cast<size_t>(sample)];
        interleaved[static_cast<size_t>(sample) * 2 + 1] =
            planar[static_cast<size_t>(samples + sample)];
    }
}

int64_t progress_current(const MM3GenProgress & progress) {
    if (progress.window < 0 || progress.n_windows <= 1) {
        return progress.step;
    }
    return progress.window * progress.n_steps + progress.step;
}

int64_t progress_total(const MM3GenProgress & progress) {
    if (progress.window < 0 || progress.n_windows <= 1) {
        return progress.n_steps;
    }
    return progress.n_windows * progress.n_steps;
}

void release_graphs() {
    mm3_lm_free(&g_mm3_lm);
    mm3_depth_free(&g_mm3_depth);
    mm3_cond_free(&g_mm3_cond);
    mm3_dit_free(&g_mm3_dit);
    mm3_vocoder_free(&g_mm3_voc);
}

bool abort_when_cancelled(void * user_data) {
    return static_cast<std::atomic<bool> *>(user_data)->load();
}

class AbortScope {
public:
    AbortScope(ggml_backend_t backend, ggml_backend_t cpu_backend, std::atomic<bool> & cancelled)
        : backend_(backend), cpu_backend_(cpu_backend) {
        backend_set_abort_handler(cpu_backend_, abort_when_cancelled, &cancelled);
        if (backend_ != cpu_backend_) {
            backend_set_abort_handler(backend_, abort_when_cancelled, &cancelled);
        }
    }

    ~AbortScope() {
        backend_set_abort_handler(cpu_backend_, nullptr, nullptr);
        if (backend_ != cpu_backend_) {
            backend_set_abort_handler(backend_, nullptr, nullptr);
        }
    }

private:
    ggml_backend_t backend_;
    ggml_backend_t cpu_backend_;
};

class GenerationScope {
public:
    explicit GenerationScope(bool & generating) : generating_(generating) {
        if (generating_) {
            throw std::logic_error("minimax engine: recursive generate() is not allowed");
        }
        generating_ = true;
    }

    ~GenerationScope() {
        generating_ = false;
    }

private:
    bool & generating_;
};

// A cancellation is consumed by the generation that observes it, never erased
// at startup: cancel() arriving between a caller's own precheck and generate()
// entry must cancel that run instead of stalling until the first progress
// callback re-arms the flag.
class CancellationScope {
public:
    explicit CancellationScope(std::atomic<bool> & cancelled) : cancelled_(cancelled) {}

    ~CancellationScope() {
        cancelled_.store(false);
    }

private:
    std::atomic<bool> & cancelled_;
};

}

struct Engine::Impl {
    EngineOptions options;
    MM3Model model;
    mutable MM3Tokenizer tokenizer;
    mutable std::atomic<bool> cancelled{false};
    mutable bool generating = false;
    bool registered = false;

    ~Impl() {
        std::lock_guard<std::recursive_mutex> lock(engine_mutex());
        release_graphs();
        mm3_unload(&model);
        if (registered) {
            --active_engine_count();
        }
    }
};

Engine::Engine() : impl_(std::make_unique<Impl>()) {}
Engine::~Engine() = default;

std::unique_ptr<Engine> Engine::create(const EngineOptions & input) {
    std::lock_guard<std::recursive_mutex> lock(engine_mutex());
    if (!detail::engine_instance_available(active_engine_count())) {
        throw std::runtime_error("minimax engine: only one active engine instance is supported");
    }
    std::unique_ptr<Engine> engine(new Engine());
    engine->impl_->options = resolve_model_paths(input);
    engine->impl_->options.n_threads = resolve_thread_count(input.n_threads);
    backend_configure_cpu(engine->impl_->options.n_threads, input.backends_dir);
    backend_configure_device(input.device);
    probe_model_files(engine->impl_->model, engine->impl_->options);
    std::string error;
    if (!mm3_load(&engine->impl_->model, &error)) {
        throw std::runtime_error("minimax engine: " + error);
    }
    engine->impl_->registered = true;
    ++active_engine_count();
    return engine;
}

GenerateResult Engine::generate(const GenerateParams & params, const ProgressFn & progress) const {
    std::lock_guard<std::recursive_mutex> lock(engine_mutex());
    GenerationScope generation_scope(impl_->generating);
    CancellationScope cancellation_scope(impl_->cancelled);
    AbortScope abort_scope(impl_->model.backend, impl_->model.cpu_backend, impl_->cancelled);

    const int model_max_frames = static_cast<int>(impl_->model.lm_cfg.max_audio_frames);
    const int64_t max_frames = detail::validate_frames(params.max_frames, model_max_frames);
    const int steps = detail::resolve_flow_steps(params.inference_steps);
    if (steps > 1000) {
        throw std::invalid_argument("inference steps must be in 1..1000");
    }
    const float cfg_scale =
        params.cfg_scale > 0.0f ? params.cfg_scale : impl_->model.synth_cfg.flow.cfg_scale;
    if (!std::isfinite(cfg_scale) || cfg_scale <= 0.0f) {
        throw std::invalid_argument("CFG scale must be finite and greater than zero");
    }

    if (impl_->cancelled.load()) {
        return {};
    }

    MM3GenRequest request;
    request.prompt = detail::build_prompt(params.caption, params.lyrics);
    request.max_frames = max_frames;
    request.seed = resolve_seed(params.seed);
    request.steps = steps;
    request.cfg_flow = cfg_scale;
    request.should_cancel = [this] { return impl_->cancelled.load(); };

    MM3ProgressCb callback;
    if (progress) {
        callback = [this, &progress](const MM3GenProgress & state) {
            if (!progress(state.stage, progress_current(state), progress_total(state))) {
                impl_->cancelled.store(true);
            }
        };
    }

    MM3GenResult generated;
    std::string error;
    if (!mm3_generate(impl_->model, request, &impl_->tokenizer, callback, &generated, &error)) {
        if (error == MM3_ERR_CANCELLED || impl_->cancelled.load()) {
            return {};
        }
        throw std::runtime_error("minimax engine: " + error);
    }

    GenerateResult result;
    interleave_planar_stereo(generated.audio, generated.n_samples, result.pcm);
    result.sample_rate = generated.sample_rate;
    result.channels = 2;
    result.emitted_frames = generated.frames;
    result.ar_ms = generated.ar_ms;
    result.condition_ms = generated.cond_ms;
    result.flow_ms = generated.flow_ms;
    result.vocoder_ms = generated.voc_ms;
    result.total_ms = generated.total_ms;
    return result;
}

void Engine::cancel() const {
    impl_->cancelled.store(true);
}

int Engine::sample_rate() const {
    return static_cast<int>(impl_->model.synth_cfg.voc.sampling_rate);
}

std::string Engine::backend_name() const {
    const ggml_backend_t backend = impl_->model.backend;
    if (!backend || backend == impl_->model.cpu_backend) {
        return "CPU";
    }
    const char * name = tts_cpp::acestep::backend_reg_name(backend);
    return name && *name ? name : "CPU";
}

GpuFallbackReason Engine::gpu_fallback_reason() const {
    return g_backend_gpu_fallback_reason;
}

}
