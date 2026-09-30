// Model-backed quality regression for the MiniMax-Music3 synthesis stack.
//
// Guards the two failure modes that shipped "valid WAV but noise" audio:
//   1. a flipped DiT velocity (the removed mm3.dit.output_negated readback
//      negation) — the Euler trajectory then never reaches the data manifold;
//   2. a stale DiT condition upload — the graph allocator recycles input
//      blocks between computes, so conditioning must be re-uploaded on every
//      forward or steps 1..N-1 run on garbage.
//
// Both defects leave the final flow latents far from the data manifold: the
// learned latent distribution has a per-element std of ~2.2, while a broken
// trajectory stalls near the ~1.0 std of the Gaussian noise it started from
// (measured 1.39 for both historical bugs). A healthy run must also keep the
// DiT deterministic across repeated computes with interleaved CFG branches.
//
// Requires AUDIOGEN_TEST_MINIMAX_MODELS_DIR; exits 77 (ctest skip) otherwise.
// With --q4 the model directory comes from AUDIOGEN_TEST_MINIMAX_Q4_MODELS_DIR
// instead, so the same checks gate a quantized pair in a separate process
// (the static LM/DiT graph caches are not rebuilt for a second in-process load).

#include "minimax/backend.h"
#include "minimax/logic.h"
#include "minimax/mm3-pipeline.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char * expression) {
    if (condition) return;
    ++failures;
    std::fprintf(stderr, "FAIL %s\n", expression);
}

#define CHECK(condition) check((condition), #condition)

double latent_std(const std::vector<float> & latents) {
    if (latents.empty()) return 0.0;
    double sum = 0.0, sum_sq = 0.0;
    for (float value : latents) {
        sum += value;
        sum_sq += (double) value * value;
    }
    const double mean = sum / (double) latents.size();
    return std::sqrt(sum_sq / (double) latents.size() - mean * mean);
}

// The tiled vocoder decode must be bit-identical to a single-shot decode:
// MM3_VOC_OVERLAP frames of context exceed the conv stack's receptive field,
// so every tile interior reproduces the full-length computation exactly.
bool vocoder_tiling_is_bit_exact(const MM3Model & model, std::string * error) {
    const int64_t L  = (int64_t) model.synth_cfg.dit.window_latents;
    const int64_t FC = (int64_t) model.synth_cfg.voc.fold_channels;
    if (L <= MM3_VOC_CHUNK) {
        *error = "window_latents does not exercise the tiled path";
        return false;
    }
    std::vector<float> latents;
    tts_cpp::minimax::detail::fill_noise(11, 0, latents, L * 2 * FC);

    std::vector<float> single, tiled;
    const int64_t T = L * (int64_t) model.synth_cfg.voc.total_upsample;
    single.assign((size_t) (2 * T), 0.0f);
    tiled.assign((size_t) (2 * T), 0.0f);
    if (!mm3_vocoder_prepare(model, &g_mm3_voc, error) ||
        !mm3_vocoder_decode_tiled(model, latents, L, single, L, MM3_VOC_OVERLAP, error) ||
        !mm3_vocoder_decode_tiled(model, latents, L, tiled, MM3_VOC_CHUNK, MM3_VOC_OVERLAP, error)) {
        return false;
    }
    if (std::memcmp(single.data(), tiled.data(), single.size() * sizeof(float)) != 0) {
        *error = "tiled vocoder output differs from single-shot output";
        return false;
    }
    return true;
}

bool dit_is_deterministic_across_computes(const MM3Model & model, std::string * error) {
    const int64_t L = 64;
    const int64_t N = (int64_t) model.synth_cfg.dit.in_channels * L;
    const int64_t CN = (int64_t) model.synth_cfg.dit.condition_dim * L;
    std::vector<float> latents, condition;
    tts_cpp::minimax::detail::fill_noise(7, 0, latents, N);
    tts_cpp::minimax::detail::fill_noise(7, 1, condition, CN);
    std::vector<float> first((size_t) N), first_unconditional((size_t) N);
    std::vector<float> swapped((size_t) N), swapped_unconditional((size_t) N);
    std::vector<float> repeat((size_t) N), repeat_unconditional((size_t) N);
    const std::vector<MM3DitBranch> swapped_branches = {
        { MM3_DIT_UNCONDITIONED_GATE, swapped_unconditional.data() },
        { MM3_DIT_CONDITIONED_GATE, swapped.data() },
    };
    if (!mm3_dit_prepare(model, &g_mm3_dit, error) ||
        !mm3_dit_run_cfg(model, &g_mm3_dit, latents.data(), condition.data(), 0.5f, L, first.data(),
                         first_unconditional.data(), error) ||
        !mm3_dit_run(model, &g_mm3_dit, latents.data(), condition.data(), swapped_branches, 0.5f, L, error) ||
        !mm3_dit_run_cfg(model, &g_mm3_dit, latents.data(), condition.data(), 0.5f, L, repeat.data(),
                         repeat_unconditional.data(), error)) {
        return false;
    }
    if (first != repeat || first_unconditional != repeat_unconditional) {
        if (error) {
            *error = "DiT output changed between identical computes";
        }
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char ** argv) {
    const bool   q4_run  = argc > 1 && std::strcmp(argv[1], "--q4") == 0;
    const char * env_var = q4_run ? "AUDIOGEN_TEST_MINIMAX_Q4_MODELS_DIR" : "AUDIOGEN_TEST_MINIMAX_MODELS_DIR";
    const char * models_dir = std::getenv(env_var);
    if (!models_dir || !*models_dir) {
        std::fprintf(stderr, "SKIP set %s to run\n", env_var);
        return 77;
    }

    backend_configure_cpu(4, std::getenv("AUDIOGEN_TEST_BACKENDS_DIR") ? std::getenv("AUDIOGEN_TEST_BACKENDS_DIR") : "");
    backend_configure_device("");  // cpu unless MM3_DEVICE overrides

    MM3Model model;
    const tts_cpp::minimax::detail::ModelPair pair =
        tts_cpp::minimax::detail::resolve_model_pair(models_dir, "", "");
    model.models_dir = models_dir;
    mm3_probe_file(pair.lm, &model.lm_file, &model.lm_cfg, nullptr, &model.meta_errors);
    mm3_probe_file(pair.synth, &model.synth_file, nullptr, &model.synth_cfg, &model.meta_errors);
    CHECK(model.meta_errors.empty());
    std::string error;
    if (!mm3_load(&model, &error)) {
        std::fprintf(stderr, "FAIL mm3_load: %s\n", error.c_str());
        return 1;
    }

    if (!dit_is_deterministic_across_computes(model, &error)) {
        std::fprintf(stderr, "FAIL dit determinism: %s\n", error.c_str());
        ++failures;
    }

    if (!vocoder_tiling_is_bit_exact(model, &error)) {
        std::fprintf(stderr, "FAIL vocoder tiling: %s\n", error.c_str());
        ++failures;
    }

    MM3GenRequest request;
    request.prompt = tts_cpp::minimax::detail::build_prompt("A short warm piano note.", "");
    request.max_frames = 48;
    request.seed = 7;
    request.steps = 30;
    request.cfg_flow = model.synth_cfg.flow.cfg_scale > 0 ? model.synth_cfg.flow.cfg_scale : 1.7f;
    request.keep_window_latents = true;

    MM3Tokenizer tokenizer;
    MM3GenResult result;
    if (!mm3_generate(model, request, &tokenizer, nullptr, &result, &error)) {
        std::fprintf(stderr, "FAIL mm3_generate: %s\n", error.c_str());
        mm3_unload(&model);
        return 1;
    }

    CHECK(result.frames > 0);
    CHECK(result.n_windows == 1);
    CHECK(result.n_samples > 0);
    CHECK(!result.ids_cond_used.empty());
    CHECK(!result.has_nan);
    CHECK(result.peak > 0.005f);
    CHECK(result.rms > 0.001);
    CHECK(result.window_latents.size() == 1);

    // On-manifold check: a correct flow trajectory ends at the learned latent
    // distribution (std ~2.2 for 300-frame songs, ~1.9 for this short prompt);
    // the historical velocity-sign and stale-condition bugs both stalled near
    // the noise distribution (std ~1.4).
    const double std_dev = latent_std(result.window_latents[0]);
    std::fprintf(stderr, "[quality] final latent std %.4f (healthy ~1.9-2.4, broken ~1.4)\n", std_dev);
    CHECK(std_dev > 1.7);
    CHECK(std_dev < 3.2);

    mm3_unload(&model);
    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::fprintf(stderr, "OK\n");
    return 0;
}
