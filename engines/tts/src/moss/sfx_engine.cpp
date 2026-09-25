#include "tts-cpp/moss/sound_effect.h"

#include "moss/sfx_model.h"
#include "moss/sfx_networks.h"
#include "moss/sfx_prompt.h"
#include "moss/sfx_sampler.h"
#include "moss/sfx_tokenizer.h"

#include "backend_selection.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <mutex>
#include <stdexcept>

namespace tts_cpp::moss {
namespace {

using detail::SfxDitSession;
using detail::SfxModel;
using detail::SfxTokenizer;

constexpr int MAX_STEPS = 1000;
constexpr float MAX_GUIDANCE = 50.0f;
constexpr float MAX_SHIFT = 100.0f;
constexpr int DECODE_WINDOW_FRAMES = 256;
constexpr float UNGUIDED = 1.0f;
constexpr int TENTHS_PER_SECOND = 10;
constexpr size_t MAX_PROMPT_BYTES = 8192;

struct PromptContexts {
    std::vector<float> positive;
    std::vector<float> negative;
};

bool is_valid_override(float value) {
    return std::isfinite(value) && value >= 0.0f;
}

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("moss sfx: " + message);
}

double elapsed_ms(std::chrono::steady_clock::time_point since) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - since).count();
}

struct ResolvedRequest {
    std::string conditioning;
    std::string negative_prompt;
    int tenths = 0;
    int steps = 0;
    float guidance = 0.0f;
    float shift = 0.0f;
    uint32_t seed = 0;
};

} // namespace

struct SoundEffectEngine::Impl {
    SoundEffectOptions options;
    std::unique_ptr<SfxModel> model;
    std::unique_ptr<SfxTokenizer> tokenizer;
    std::atomic<bool> cancel_requested{false};
    std::mutex generation_mutex;

    void load() {
        if (options.model_path.empty()) {
            fail("model_path is required");
        }
        if (!options.backends_dir.empty()) {
            ::tts_cpp::detail::set_backends_directory(options.backends_dir);
        }
        model = std::make_unique<SfxModel>(options.model_path, options.use_gpu, options.n_threads);
        validate_networks();
        tokenizer = std::make_unique<SfxTokenizer>(*model);
    }

    void validate_networks() const {
        detail::validate_text_encoder(*model);
        detail::validate_dit(*model);
        detail::validate_vae(*model);
    }

    ResolvedRequest resolve(const SoundEffectRequest & request) const {
        validate_overrides(request);
        const detail::SfxConfig & config = model->config();
        ResolvedRequest resolved;
        resolved.tenths = detail::seconds_in_tenths(request.seconds);
        resolved.conditioning = detail::clean_prompt(detail::duration_prompt(request.prompt, resolved.tenths));
        resolved.negative_prompt = detail::clean_prompt(request.negative_prompt);
        resolved.steps = request.steps > 0 ? request.steps : config.default_steps;
        resolved.guidance = request.guidance > 0.0f ? request.guidance : config.default_guidance;
        resolved.shift = request.shift > 0.0f ? request.shift : config.sigma_shift;
        resolved.seed = request.seed;
        validate_prompt(request.prompt);
        validate(resolved);
        return resolved;
    }

    void validate_overrides(const SoundEffectRequest & request) const {
        if (request.steps < 0) {
            fail("steps must be 1.." + std::to_string(MAX_STEPS) + ", or 0 for the model default");
        }
        if (!is_valid_override(request.guidance)) {
            fail("guidance must be in [1, " + std::to_string((int) MAX_GUIDANCE) + "], or 0 for the model default");
        }
        if (!is_valid_override(request.shift)) {
            fail("shift must be in (0, " + std::to_string((int) MAX_SHIFT) + "], or 0 for the model default");
        }
        if (!std::isfinite(request.seconds)) {
            fail("seconds must be a finite number");
        }
        if (request.prompt.size() > MAX_PROMPT_BYTES || request.negative_prompt.size() > MAX_PROMPT_BYTES) {
            fail("prompts must be at most " + std::to_string(MAX_PROMPT_BYTES) + " bytes");
        }
    }

    void validate_prompt(const std::string & prompt) const {
        if (detail::clean_prompt(prompt).empty()) {
            fail("prompt must not be empty");
        }
    }

    void validate(const ResolvedRequest & request) const {
        const float max_seconds = model->config().max_seconds;
        if (request.tenths < 1 || (float) request.tenths > max_seconds * TENTHS_PER_SECOND) {
            fail("seconds must be in (0, " + std::to_string(max_seconds) + "]");
        }
        if (request.steps > MAX_STEPS) {
            fail("steps must be 1.." + std::to_string(MAX_STEPS));
        }
        if (request.guidance < UNGUIDED || request.guidance > MAX_GUIDANCE) {
            fail("guidance must be in [1, " + std::to_string((int) MAX_GUIDANCE) + "]");
        }
        if (request.shift > MAX_SHIFT) {
            fail("shift must be in (0, " + std::to_string((int) MAX_SHIFT) + "]");
        }
    }

    std::vector<float> step_velocity(SfxDitSession & dit, const std::vector<float> & latents, float timestep,
                                     const std::vector<float> & positive, const std::vector<float> & negative,
                                     float guidance) {
        std::vector<float> conditioned = dit.velocity(latents, timestep, positive);
        if (guidance == UNGUIDED) {
            return conditioned;
        }
        return detail::guided_velocity(conditioned, dit.velocity(latents, timestep, negative), guidance);
    }

    bool keep_going(const SoundEffectProgress & progress, int step, int total) const {
        if (cancel_requested) {
            return false;
        }
        return !progress || progress(step, total);
    }

    void denoise_step(SfxDitSession & dit, std::vector<float> & latents, const ResolvedRequest & request,
                      const std::vector<float> & sigmas, int step, const std::vector<float> & positive,
                      const std::vector<float> & negative) {
        const float sigma = sigmas[(size_t) step];
        const float next = step + 1 < request.steps ? sigmas[(size_t) step + 1] : 0.0f;
        const float timestep = sigma * (float) model->config().train_timesteps;
        detail::euler_step(latents, step_velocity(dit, latents, timestep, positive, negative, request.guidance),
                sigma, next);
    }

    bool denoise(std::vector<float> & latents, const ResolvedRequest & request, const std::vector<float> & positive,
                 const std::vector<float> & negative, const SoundEffectProgress & progress) {
        const std::vector<float> sigmas = detail::flow_sigmas(request.steps, request.shift);
        SfxDitSession dit(*model, model->config().latent_frames());
        return run_steps(dit, latents, request, sigmas, positive, negative, progress);
    }

    bool run_steps(SfxDitSession & dit, std::vector<float> & latents, const ResolvedRequest & request,
                   const std::vector<float> & sigmas, const std::vector<float> & positive,
                   const std::vector<float> & negative, const SoundEffectProgress & progress) {
        for (int step = 0; step < request.steps; ++step) {
            denoise_step(dit, latents, request, sigmas, step, positive, negative);
            if (!keep_going(progress, step + 1, request.steps)) {
                return false;
            }
        }
        return true;
    }

    int64_t output_samples(int tenths) const {
        return (int64_t) model->config().vae.sample_rate * tenths / TENTHS_PER_SECOND;
    }

    int kept_frames(int64_t samples) const {
        const detail::SfxConfig & config = model->config();
        const int hop = config.vae.hop_length();
        return (int) std::min<int64_t>(config.latent_frames(), (samples + hop - 1) / hop);
    }

    std::vector<float> decode(const std::vector<float> & latents, int tenths) {
        const int64_t samples = output_samples(tenths);
        std::vector<float> pcm = detail::decode_latents(*model, latents, model->config().latent_frames(),
                kept_frames(samples), DECODE_WINDOW_FRAMES,
                [this](int, int) { return !cancel_requested.load(); });
        pcm.resize(std::min(pcm.size(), (size_t) samples));
        return pcm;
    }

    PromptContexts encode_prompts(const ResolvedRequest & request) {
        PromptContexts contexts;
        contexts.positive = detail::encode_text(*model, tokenizer->encode(request.conditioning));
        contexts.negative = detail::encode_text(*model, tokenizer->encode(request.negative_prompt));
        return contexts;
    }

    std::vector<float> initial_noise(uint32_t seed) const {
        const detail::SfxConfig & config = model->config();
        return detail::gaussian_noise((size_t) config.latent_frames() * config.dit.in_channels, seed);
    }

    void finish_decode(SoundEffectResult & result, const std::vector<float> & latents, int tenths) {
        const auto decode_start = std::chrono::steady_clock::now();
        result.pcm = decode(latents, tenths);
        result.decode_ms = elapsed_ms(decode_start);
        result.cancelled = cancel_requested;
        if (result.cancelled) {
            result.pcm.clear();
        }
    }

    SoundEffectResult run(const ResolvedRequest & request, const SoundEffectProgress & progress) {
        SoundEffectResult result;
        result.sample_rate = model->config().vae.sample_rate;
        const auto text_start = std::chrono::steady_clock::now();
        const PromptContexts contexts = encode_prompts(request);
        result.text_ms = elapsed_ms(text_start);

        const auto diffusion_start = std::chrono::steady_clock::now();
        std::vector<float> latents = initial_noise(request.seed);
        const bool finished = denoise(latents, request, contexts.positive, contexts.negative, progress);
        result.diffusion_ms = elapsed_ms(diffusion_start);
        result.cancelled = !finished;
        if (finished) {
            finish_decode(result, latents, request.tenths);
        }
        return result;
    }

    SoundEffectResult generate(const SoundEffectRequest & request, const SoundEffectProgress & progress) {
        std::unique_lock<std::mutex> lock(generation_mutex, std::try_to_lock);
        if (!lock.owns_lock()) {
            fail("generation already in progress on this instance");
        }
        cancel_requested = false;
        return run(resolve(request), progress);
    }
};

SoundEffectEngine::SoundEffectEngine(const SoundEffectOptions & options) : impl_(new Impl) {
    impl_->options = options;
    impl_->load();
}

SoundEffectEngine::~SoundEffectEngine() = default;

SoundEffectResult SoundEffectEngine::generate(const SoundEffectRequest & request,
                                              const SoundEffectProgress & progress) {
    return impl_->generate(request, progress);
}

void SoundEffectEngine::cancel() noexcept {
    impl_->cancel_requested = true;
}

int SoundEffectEngine::sample_rate() const noexcept {
    return impl_->model->config().vae.sample_rate;
}

float SoundEffectEngine::max_seconds() const noexcept {
    return impl_->model->config().max_seconds;
}

const char * SoundEffectEngine::backend_name() const noexcept {
    return impl_->model->backend_name();
}

} // namespace tts_cpp::moss
