#include "minimax/coreml-models.h"
#include "minimax/mm3-pipeline.h"
#include "minimax_tiny_model.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <stdexcept>

namespace {

constexpr int64_t PROMPT_TOKENS = 3;
constexpr int64_t REQUEST_POSITIONS = 12;
constexpr int64_t DIT_FRAMES = 3;
constexpr float ABSOLUTE_TOLERANCE = 0.003f;
constexpr float REPEAT_TOLERANCE = 1e-6f;
constexpr double PRODUCTION_MAX_RELATIVE_RMS = 0.002;
constexpr double PRODUCTION_MIN_COSINE = 0.9999;

void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

void release_graphs() {
    mm3_lm_free(&g_mm3_lm);
    mm3_depth_free(&g_mm3_depth);
    mm3_cond_free(&g_mm3_cond);
    mm3_dit_free(&g_mm3_dit);
    mm3_vocoder_free(&g_mm3_voc);
}

std::vector<float> values(size_t count) {
    std::vector<float> result(count);
    for (size_t i = 0; i < count; ++i) result[i] = std::sin(static_cast<float>(i + 1) * 0.17f);
    return result;
}

void compare(const std::vector<float> & expected, const std::vector<float> & actual,
             float tolerance, const char * label) {
    require(expected.size() == actual.size(), std::string(label) + " output size mismatch");
    float maximum = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        require(std::isfinite(actual[i]), std::string(label) + " non-finite output");
        maximum = std::max(maximum, std::fabs(expected[i] - actual[i]));
    }
    std::fprintf(stderr, "%s: max error %.9g (gate %.9g)\n", label, maximum, tolerance);
    require(maximum <= tolerance, std::string(label) + " exceeds parity tolerance");
}

void compare_production(const std::vector<float> & expected, const std::vector<float> & actual, const char * label) {
    require(expected.size() == actual.size(), std::string(label) + " output size mismatch");
    double reference_energy = 0, actual_energy = 0, dot = 0, error_energy = 0, maximum = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        require(std::isfinite(actual[i]), std::string(label) + " non-finite output");
        const double delta = actual[i] - expected[i];
        reference_energy += static_cast<double>(expected[i]) * expected[i];
        actual_energy += static_cast<double>(actual[i]) * actual[i];
        dot += static_cast<double>(expected[i]) * actual[i];
        error_energy += delta * delta;
        maximum = std::max(maximum, std::fabs(delta));
    }
    require(reference_energy > 0 && actual_energy > 0, "production parity needs nonzero reference outputs");
    const double relative_rms = std::sqrt(error_energy / reference_energy);
    const double cosine = dot / std::sqrt(reference_energy * actual_energy);
    std::fprintf(stderr, "%s: max error %.9g, relative RMS %.9g, cosine %.12g\n", label, maximum, relative_rms, cosine);
    require(relative_rms <= PRODUCTION_MAX_RELATIVE_RMS && cosine >= PRODUCTION_MIN_COSINE,
            std::string(label) + " exceeds production mixed-precision tolerances");
}

void compare_stage(const std::vector<float> & expected, const std::vector<float> & actual, bool real, const char * label) {
    if (real) compare_production(expected, actual, label);
    else compare(expected, actual, ABSOLUTE_TOLERANCE, label);
}

void append(std::vector<float> & output, const std::vector<float> & values) {
    output.insert(output.end(), values.begin(), values.end());
}

std::vector<float> run_lm(MM3Model & model) {
    const size_t hidden_size = model.lm_cfg.embedding_length;
    const size_t logits_size = model.lm_cfg.semantic_vocab_size + 1;
    std::vector<float> hidden(hidden_size * 2), logits(logits_size * 2), output;
    const int32_t conditional[] = {1, 2, 3};
    const int32_t unconditional[] = {4, 5, 6};
    std::vector<int32_t> acoustic(model.lm_cfg.num_codebooks - 1);
    for (size_t i = 0; i < acoustic.size(); ++i) acoustic[i] = static_cast<int32_t>(i * model.lm_cfg.acoustic_vocab_size);
    std::string error;
    require(mm3_lm_prepare(model, &g_mm3_lm, REQUEST_POSITIONS, &error), error);
    require(mm3_lm_prefill(model, &g_mm3_lm, conditional, unconditional, PROMPT_TOKENS,
                          hidden.data(), logits.data(), &error), error);
    append(output, hidden);
    append(output, logits);
    for (int i = 0; i < 3; ++i) {
        require(mm3_lm_decode(model, &g_mm3_lm, model.lm_cfg.semantic_vocab_offset + i, acoustic.data(),
                             hidden.data(), logits.data(), nullptr, &error), error);
        append(output, hidden);
        append(output, logits);
    }
    return output;
}

std::vector<float> run_depth(MM3Model & model) {
    const auto hidden = values(model.lm_cfg.embedding_length * 2);
    std::vector<int32_t> forced(model.synth_cfg.depth.num_codebooks - 1, 0);
    MM3DepthFrame frame;
    std::string error;
    std::vector<float> output;
    for (int i = 0; i < 2; ++i) {
        require(mm3_depth_decode_frame(model, hidden.data(), hidden.data() + model.lm_cfg.embedding_length,
                                       i, forced.data(), &frame, &error), error);
        append(output, frame.hiddens);
        append(output, frame.logits_cond);
        append(output, frame.logits_uncond);
    }
    return output;
}

std::vector<float> run_dit(MM3Model & model) {
    auto latent = values(model.synth_cfg.dit.in_channels * DIT_FRAMES);
    const auto condition = values(model.synth_cfg.dit.condition_dim * DIT_FRAMES);
    std::vector<float> output(latent.size() * 2);
    std::string error;
    require(mm3_dit_prepare(model, &g_mm3_dit, &error), error);
    require(mm3_dit_run_cfg(model, &g_mm3_dit, latent.data(), condition.data(), 0.5f, DIT_FRAMES,
                            output.data(), output.data() + latent.size(), &error), error);
    return output;
}

void verify_lm_failure_recovery(MM3Model & model, const std::vector<float> & expected) {
    const size_t hidden = model.lm_cfg.embedding_length;
    require(model.lm.token_embd->type == GGML_TYPE_F32, "recovery fixture requires F32 embeddings");
    std::vector<float> original(hidden), poisoned(hidden, std::numeric_limits<float>::quiet_NaN());
    const size_t offset = model.lm.token_embd->nb[1];
    ggml_backend_tensor_get(model.lm.token_embd, original.data(), offset, hidden * sizeof(float));
    ggml_backend_tensor_set(model.lm.token_embd, poisoned.data(), offset, hidden * sizeof(float));
    const int32_t conditional[] = {1, 2, 3}, unconditional[] = {4, 5, 6};
    std::vector<float> output_hidden(hidden * 2), logits((model.lm_cfg.semantic_vocab_size + 1) * 2);
    std::string error;
    const bool succeeded = mm3_lm_prefill(model, &g_mm3_lm, conditional, unconditional, PROMPT_TOKENS,
                                         output_hidden.data(), logits.data(), &error);
    ggml_backend_tensor_set(model.lm.token_embd, original.data(), offset, hidden * sizeof(float));
    require(!succeeded && error.find("Core ML prediction failed") != std::string::npos,
            "non-finite LM prediction must fail before consuming outputs");
    mm3_coreml_clear_errors(model.lm.blk);
    compare(expected, run_lm(model), REPEAT_TOLERANCE, "LM recovery after failed prediction");
}

void verify_depth_failure_recovery(MM3Model & model, const std::vector<float> & expected) {
    std::vector<float> poisoned(model.lm_cfg.embedding_length * 2, std::numeric_limits<float>::quiet_NaN());
    std::vector<int32_t> forced(model.synth_cfg.depth.num_codebooks - 1, 0);
    MM3DepthFrame frame;
    std::string error;
    const bool succeeded = mm3_depth_decode_frame(model, poisoned.data(), poisoned.data() + model.lm_cfg.embedding_length,
                                                  0, forced.data(), &frame, &error);
    require(!succeeded && error.find("Core ML prediction failed") != std::string::npos,
            "non-finite depth prediction must fail before consuming outputs");
    mm3_coreml_clear_errors(model.synth.depth.blk);
    compare(expected, run_depth(model), REPEAT_TOLERANCE, "Depth recovery after failed prediction");
}

void verify_incompatible_outputs(const MM3Model & model, const std::filesystem::path & directory) {
    using namespace tts_cpp::minimax::detail;
    const auto & lm = model.lm_cfg;
    const CoreMLBlockConfig config{"lm", 0, lm.embedding_length, lm.head_count, lm.head_count_kv, lm.key_length};
    const auto hash = coreml_source_fingerprint(model.lm_file.path);
    for (const auto * name : {"bad-output-name.mlmodelc", "bad-output-shape.mlmodelc"}) {
        const auto block = load_coreml_block((directory / name).string(), hash, config);
        require(block != nullptr, std::string("missing incompatible-output fixture: ") + name);
        require(!coreml_block_supports(block, PROMPT_TOKENS, 256, 2), "incompatible outputs must select native block");
        require(coreml_block_error(block).empty(), "compatibility rejection must not abort native fallback");
        require(coreml_block_predictions(block) == 0, "compatibility probes must not count as generated predictions");
    }
}

void verify_fallback_and_errors(MM3Model & model) {
    using namespace tts_cpp::minimax::detail;
    const auto & block = model.lm.blk[0].coreml;
    require(!coreml_block_supports(block, 1, model.lm_cfg.context_length + 1, 2), "oversized cache must select native block");
    require(!coreml_block_supports(block, 1, 256, 3), "unsupported CFG batch must select native block");
    const auto path = model.lm_file.path + "-coreml/lm-block-0.mlmodelc";
    const auto & config = model.lm_cfg;
    require(!load_coreml_block(path, "wrong-source-hash", {"lm", 0, config.embedding_length,
            config.head_count, config.head_count_kv, config.key_length}), "mismatched weights must select native block");
    coreml_block_set_error(*block, "injected prediction error");
    std::string error;
    require(!mm3_coreml_check(model.lm.blk, &error), "prediction error must abort output consumption");
    require(error.find("injected prediction error") != std::string::npos, "prediction error detail was lost");
    mm3_coreml_clear_errors(model.lm.blk);
    require(mm3_coreml_check(model.lm.blk, &error), "a new generation must clear previous errors");
}

void verify_predictions(const MM3Model & model) {
    using tts_cpp::minimax::detail::coreml_block_predictions;
    require(coreml_block_predictions(model.lm.blk[0].coreml) >= 4, "LM did not run Core ML prefill and cached decode");
    require(coreml_block_predictions(model.synth.depth.blk[0].coreml) >= 4, "Depth did not run Core ML across frames");
    require(coreml_block_predictions(model.synth.dit.blk[0].coreml) >= 1, "DiT did not run Core ML CFG");
}

int parity(const std::filesystem::path & directory, bool real) {
    MM3Model model;
    backend_configure_cpu(4, "");
    backend_configure_device(real ? "gpu" : "cpu");
    mm3_probe_file(directory / "mm3-lm-f16.gguf", &model.lm_file, &model.lm_cfg, nullptr, &model.meta_errors);
    mm3_probe_file(directory / "mm3-synth-f16.gguf", &model.synth_file, nullptr, &model.synth_cfg, &model.meta_errors);
    std::string error;
    require(mm3_load(&model, &error), error);
    if (real) {
        model.lm.blk.resize(1);
        model.lm_cfg.block_count = 1;
        model.synth.depth.blk.resize(1);
        model.synth_cfg.depth.block_count = 1;
        model.synth.dit.blk.resize(1);
        model.synth_cfg.dit.block_count = 1;
    }
    try {
        const auto lm = run_lm(model);
        const auto depth = run_depth(model);
        const auto dit = run_dit(model);
        release_graphs();
        tts_cpp::minimax::detail::attach_coreml_blocks(model);
        const char * dit_directory = std::getenv("AUDIOGEN_TEST_MINIMAX_DIT_COREML_DIR");
        if (real && dit_directory) {
            const auto & config = model.synth_cfg.dit;
            const auto hash = tts_cpp::minimax::detail::coreml_source_fingerprint(model.synth_file.path);
            model.synth.dit.blk[0].coreml = tts_cpp::minimax::detail::load_coreml_block(
                std::string(dit_directory) + "/dit-block-0.mlmodelc", hash,
                {"dit", 0, config.embedding_length, config.head_count, config.head_count, config.head_dim});
            require(model.synth.dit.blk[0].coreml != nullptr, "failed to load selected DiT diagnostic sidecar");
        }
        const auto accelerated_lm = run_lm(model);
        const auto accelerated_depth = run_depth(model);
        const auto accelerated_dit = run_dit(model);
        verify_predictions(model);
        verify_fallback_and_errors(model);
        compare_stage(lm, accelerated_lm, real, "LM multi-step");
        compare_stage(depth, accelerated_depth, real, "Depth multi-frame");
        compare_stage(dit, accelerated_dit, real, "DiT CFG");
        compare(accelerated_lm, run_lm(model), REPEAT_TOLERANCE, "LM reset");
        compare(accelerated_depth, run_depth(model), REPEAT_TOLERANCE, "Depth reset");
        if (!real) {
            verify_incompatible_outputs(model, directory);
            verify_lm_failure_recovery(model, accelerated_lm);
            verify_depth_failure_recovery(model, accelerated_depth);
        }
        release_graphs();
        mm3_unload(&model);
    } catch (...) {
        release_graphs();
        mm3_unload(&model);
        throw;
    }
    return 0;
}

}

int main(int argc, char ** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--write-fixtures") {
            const std::filesystem::path directory = argv[2];
            std::filesystem::create_directories(directory);
            require(minimax_tiny::write_lm(directory / "mm3-lm-f16.gguf") &&
                    minimax_tiny::write_synth(directory / "mm3-synth-f16.gguf"), "failed to write fixtures");
            return 0;
        }
        const char * directory = std::getenv("AUDIOGEN_TEST_MINIMAX_COREML_DIR");
        if (!directory) {
            std::fprintf(stderr, "SKIP: set AUDIOGEN_TEST_MINIMAX_COREML_DIR to exported tiny Core ML fixtures\n");
            return 77;
        }
        return parity(directory, argc == 2 && std::string(argv[1]) == "--real");
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
