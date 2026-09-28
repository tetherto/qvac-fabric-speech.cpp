#include "parakeet_diarization_v3.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstring>

namespace parakeet {
namespace {

constexpr float kLayerNormEpsilon = 1.0e-5f;
constexpr size_t kGraphSlots = 4096;

ggml_tensor * layer_norm(
    ggml_context * context, ggml_tensor * input,
    ggml_tensor * weight, ggml_tensor * bias) {
    return ggml_add(context,
        ggml_mul(context, ggml_norm(context, input, kLayerNormEpsilon), weight), bias);
}

ggml_tensor * linear(
    ggml_context * context, ggml_tensor * input,
    ggml_tensor * weight, ggml_tensor * bias) {
    ggml_tensor * output = ggml_mul_mat(context, weight, input);
    return bias ? ggml_add(context, output, bias) : output;
}

ggml_tensor * qkv_view(
    ggml_context * context, ggml_tensor * qkv,
    const NemotronDiarizationConfig & config, int component, int frames) {
    const int head_width = config.encoder_width / config.attention_heads;
    return ggml_view_4d(context, qkv, head_width, config.attention_heads, frames, 1,
        static_cast<size_t>(head_width) * qkv->nb[0], qkv->nb[1], qkv->nb[2],
        static_cast<size_t>(component * config.encoder_width) * qkv->nb[0]);
}

ggml_tensor * rotary_attention(
    ggml_context * context, ggml_tensor * input, ggml_tensor * positions,
    const NemotronDiarizationConfig & config,
    const NemotronDiarizationLayer & layer, int frames) {
    ggml_tensor * qkv = linear(context, input, layer.qkv_w, nullptr);
    ggml_tensor * query = qkv_view(context, qkv, config, 0, frames);
    ggml_tensor * key = qkv_view(context, qkv, config, 1, frames);
    ggml_tensor * value = qkv_view(context, qkv, config, 2, frames);
    const int head_width = config.encoder_width / config.attention_heads;
    const int rotary_width = static_cast<int>(head_width * config.rotary_fraction);
    query = ggml_rope_ext(context, query, positions, nullptr, rotary_width,
        GGML_ROPE_TYPE_NEOX, config.position_limit, config.rope_base,
        1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    key = ggml_rope_ext(context, key, positions, nullptr, rotary_width,
        GGML_ROPE_TYPE_NEOX, config.position_limit, config.rope_base,
        1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    query = ggml_permute(context, query, 0, 2, 1, 3);
    key = ggml_permute(context, key, 0, 2, 1, 3);
    value = ggml_permute(context, value, 0, 2, 1, 3);
    ggml_tensor * attended = ggml_flash_attn_ext(context, query, key, value,
        nullptr, 1.0f / std::sqrt(static_cast<float>(head_width)), 0.0f, 0.0f);
    ggml_tensor * merged = ggml_reshape_2d(context,
        ggml_cont(context, attended), config.encoder_width, frames);
    return linear(context, merged, layer.attention_w, layer.attention_b);
}

ggml_tensor * encoder_layer(
    ggml_context * context, ggml_tensor * input, ggml_tensor * positions,
    const NemotronDiarizationConfig & config,
    const NemotronDiarizationLayer & layer, int frames) {
    ggml_tensor * normalized = layer_norm(
        context, input, layer.norm1_w, layer.norm1_b);
    ggml_tensor * attended = rotary_attention(
        context, normalized, positions, config, layer, frames);
    ggml_tensor * residual = ggml_add(context, input, attended);
    normalized = layer_norm(context, residual, layer.norm2_w, layer.norm2_b);
    ggml_tensor * hidden = linear(context, normalized,
        layer.feed_forward_in_w, layer.feed_forward_in_b);
    hidden = ggml_gelu_erf(context, hidden);
    ggml_tensor * output = linear(context, hidden,
        layer.feed_forward_out_w, layer.feed_forward_out_b);
    return ggml_add(context, residual, output);
}

ggml_tensor * encode(
    ggml_context * context, ggml_tensor * stacked, ggml_tensor * state,
    ggml_tensor * positions, ggml_tensor ** chunk_embeddings,
    const NemotronDiarizationConfig & config,
    const NemotronDiarizationWeights & weights, int frames) {
    *chunk_embeddings = linear(context, stacked, weights.feature_projection, nullptr);
    ggml_tensor * encoded = state
        ? ggml_concat(context, state, *chunk_embeddings, 1)
        : *chunk_embeddings;
    encoded = layer_norm(context, encoded,
        weights.embedding_norm_w, weights.embedding_norm_b);
    for (const auto & layer : weights.layers) {
        encoded = encoder_layer(context, encoded, positions, config, layer, frames);
    }
    return layer_norm(context, encoded,
        weights.final_norm_w, weights.final_norm_b);
}

ggml_tensor * speaker_head(
    ggml_context * context, ggml_tensor * encoded,
    const NemotronDiarizationConfig & config,
    const NemotronDiarizationWeights & weights, int frames) {
    ggml_tensor * projected = linear(context, encoded,
        weights.encoder_projection_w, weights.encoder_projection_b);
    ggml_tensor * convolution_input = ggml_cont(context,
        ggml_permute(context, projected, 1, 0, 2, 3));
    ggml_tensor * convolved = ggml_conv_1d(context,
        weights.upsample_w, convolution_input, 1, 1, 1);
    ggml_tensor * channels_first = ggml_cont(context,
        ggml_permute(context, convolved, 1, 0, 2, 3));
    channels_first = ggml_add(context, channels_first, weights.upsample_b);
    ggml_tensor * upsampled = ggml_reshape_2d(context,
        channels_first, config.output_width, frames * config.subsampling_factor);
    ggml_tensor * hidden = linear(context, ggml_relu(context, upsampled),
        weights.hidden_w, weights.hidden_b);
    ggml_tensor * logits = linear(context, ggml_relu(context, hidden),
        weights.speakers_w, weights.speakers_b);
    return ggml_sigmoid(context, logits);
}

void stack_mel(
    const float * mel, int mel_frames, int mel_bins, int factor,
    std::vector<float> & stacked) {
    const int frames = (mel_frames + factor - 1) / factor;
    stacked.assign(static_cast<size_t>(frames) * factor * mel_bins, 0.0f);
    std::memcpy(stacked.data(), mel,
        static_cast<size_t>(mel_frames) * mel_bins * sizeof(float));
}

void fill_positions(int frames, std::vector<int32_t> & positions) {
    positions.resize(frames);
    for (int frame = 0; frame < frames; ++frame) positions[frame] = frame;
}

int compute_graph(
    const ParakeetCtcModel & model,
    ggml_context * context, ggml_tensor * input, ggml_tensor * state_input,
    ggml_tensor * positions, ggml_tensor * output,
    ggml_tensor * chunk_embeddings, const std::vector<float> & stacked,
    const float * state_values, int state_frames,
    const std::vector<int32_t> & position_values,
    NemotronDiarizationChunk & result) {
    ggml_cgraph * graph = ggml_new_graph_custom(context, kGraphSlots, false);
    ggml_build_forward_expand(graph, output);
    ggml_backend_sched_t scheduler = model_sched(model);
    if (!scheduler) return 1;
    ggml_backend_sched_reset(scheduler);
    if (!ggml_backend_sched_alloc_graph(scheduler, graph)) return 2;
    ggml_backend_tensor_set(input, stacked.data(), 0,
        stacked.size() * sizeof(float));
    if (state_input) {
        ggml_backend_tensor_set(state_input, state_values, 0,
            static_cast<size_t>(state_frames) * model.nemotron_diarization_cfg.encoder_width * sizeof(float));
    }
    ggml_backend_tensor_set(positions, position_values.data(), 0,
        position_values.size() * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(scheduler, graph) != GGML_STATUS_SUCCESS) {
        return 3;
    }
    result.state_frames = state_frames;
    result.chunk_frames = static_cast<int>(chunk_embeddings->ne[1]);
    result.probabilities.resize(static_cast<size_t>(output->ne[1]) *
        model.nemotron_diarization_cfg.speakers);
    result.embeddings.resize(static_cast<size_t>(result.chunk_frames) *
        model.nemotron_diarization_cfg.encoder_width);
    ggml_backend_tensor_get(output, result.probabilities.data(), 0,
        result.probabilities.size() * sizeof(float));
    ggml_backend_tensor_get(chunk_embeddings, result.embeddings.data(), 0,
        result.embeddings.size() * sizeof(float));
    return 0;
}

}

int run_nemotron_diarization_chunk(
    const ParakeetCtcModel & model, const float * mel, int mel_frames,
    const float * state, int state_frames, NemotronDiarizationChunk & output) {
    const auto & config = model.nemotron_diarization_cfg;
    if (!mel || mel_frames <= 0 || state_frames < 0 ||
        (state_frames > 0 && !state) || config.subsampling_factor <= 0) return 1;
    const int chunk_frames = (mel_frames + config.subsampling_factor - 1) /
        config.subsampling_factor;
    const int frames = state_frames + chunk_frames;
    if (frames > config.position_limit) return 2;
    std::vector<float> stacked;
    stack_mel(mel, mel_frames, model.mel_cfg.n_mels,
        config.subsampling_factor, stacked);
    std::vector<int32_t> position_values;
    fill_positions(frames, position_values);
    const size_t overhead = ggml_tensor_overhead() * kGraphSlots +
        ggml_graph_overhead_custom(kGraphSlots, false);
    ggml_init_params parameters = {overhead, nullptr, true};
    ggml_context * context = ggml_init(parameters);
    if (!context) return 3;
    ggml_tensor * input = ggml_new_tensor_2d(context, GGML_TYPE_F32,
        config.subsampling_factor * model.mel_cfg.n_mels, chunk_frames);
    ggml_tensor * state_input = state_frames > 0
        ? ggml_new_tensor_2d(context, GGML_TYPE_F32, config.encoder_width, state_frames)
        : nullptr;
    ggml_tensor * positions = ggml_new_tensor_1d(context, GGML_TYPE_I32, frames);
    ggml_set_input(input);
    if (state_input) ggml_set_input(state_input);
    ggml_set_input(positions);
    ggml_tensor * chunk_embeddings = nullptr;
    ggml_tensor * encoded = encode(context, input, state_input, positions,
        &chunk_embeddings,
        config, model.nemotron_diarization, frames);
    ggml_tensor * speaker_output = speaker_head(context, encoded,
        config, model.nemotron_diarization, frames);
    ggml_set_output(speaker_output);
    ggml_set_output(chunk_embeddings);
    const int result = compute_graph(model, context, input, state_input,
        positions, speaker_output, chunk_embeddings, stacked, state,
        state_frames, position_values, output);
    ggml_free(context);
    return result;
}

int run_nemotron_diarization(
    const ParakeetCtcModel & model, const float * mel, int mel_frames,
    std::vector<float> & probabilities) {
    NemotronDiarizationChunk chunk;
    const int result = run_nemotron_diarization_chunk(
        model, mel, mel_frames, nullptr, 0, chunk);
    if (result != 0) return result;
    probabilities = std::move(chunk.probabilities);
    probabilities.resize(static_cast<size_t>(mel_frames) *
        model.nemotron_diarization_cfg.speakers);
    return 0;
}

}
