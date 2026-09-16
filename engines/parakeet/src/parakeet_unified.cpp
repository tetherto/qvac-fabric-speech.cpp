#include "parakeet_unified.h"

#include "cached_encoder.h"
#include "mel_preprocess.h"
#include "sentencepiece_bpe.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace parakeet {
namespace {

using namespace cached_encoder;

constexpr int kPreEncodeHistoryMelFrames = 8;
constexpr int kSubsamplingHistoryFrames = 1;
constexpr int kDefaultSubsamplingFactor = 8;

struct UnifiedStepGraph {
    ggml_context * context = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t allocator = nullptr;

    ggml_tensor * encoder_input = nullptr;
    ggml_tensor * attention_mask = nullptr;
    ggml_tensor * position_input = nullptr;
    ggml_tensor * encoder_output = nullptr;
    std::vector<ggml_tensor *> channel_cache_inputs;
    std::vector<ggml_tensor *> channel_cache_outputs;
    std::vector<ggml_tensor *> time_cache_inputs;
    std::vector<ggml_tensor *> time_cache_outputs;

    std::vector<float> positions;
    int current_frames = 0;
    int committed_frames = 0;
    int channel_frames = 0;
    int convolution_frames = 0;

    void clear() {
        if (allocator) {
            ggml_gallocr_free(allocator);
            allocator = nullptr;
        }
        if (context) {
            ggml_free(context);
            context = nullptr;
        }
        graph = nullptr;
        encoder_input = nullptr;
        attention_mask = nullptr;
        position_input = nullptr;
        encoder_output = nullptr;
        channel_cache_inputs.clear();
        channel_cache_outputs.clear();
        time_cache_inputs.clear();
        time_cache_outputs.clear();
        positions.clear();
        current_frames = 0;
        committed_frames = 0;
    }

    ~UnifiedStepGraph() {
        clear();
    }
};

int channel_cache_frames(const ParakeetCtcModel & model) {
    return model.unified_cfg.left_context_frames;
}

int convolution_cache_frames(const ParakeetCtcModel & model) {
    return model.unified_cfg.cache_time_steps;
}

int encoder_subsampling_factor(const ParakeetCtcModel & model) {
    return model.encoder_cfg.subsampling_factor > 0
        ? model.encoder_cfg.subsampling_factor
        : kDefaultSubsamplingFactor;
}

ggml_tensor * leading_frames(
    ggml_context * context,
    ggml_tensor * frames,
    int width,
    int count) {
    return ggml_cont(
        context,
        ggml_view_2d(context, frames, width, count, frames->nb[1], 0));
}

ggml_tensor * squeezed_pointwise(
    ggml_context * context,
    ggml_tensor * weight,
    int rows,
    int cols) {
    if (ggml_n_dims(weight) == 2 && weight->ne[0] == rows && weight->ne[1] == cols) {
        return weight;
    }
    return ggml_reshape_2d(context, weight, rows, cols);
}

ggml_tensor * gated_linear_unit(
    ggml_context * context,
    ggml_tensor * projected,
    int width,
    int frames) {
    ggml_tensor * first_half = ggml_cont(
        context,
        ggml_view_2d(context, projected, width, frames, projected->nb[1], 0));
    ggml_tensor * second_half = ggml_cont(
        context,
        ggml_view_2d(
            context,
            projected,
            width,
            frames,
            projected->nb[1],
            static_cast<size_t>(width) * projected->nb[0]));
    return ggml_mul(context, first_half, ggml_sigmoid(context, second_half));
}

ggml_tensor * convolution_norm(
    ggml_context * context,
    ggml_tensor * time_major,
    const BlockWeights & weights,
    const EncoderConfig & config) {
    if (config.conv_norm_type == ConvNormType::LayerNorm) {
        ggml_tensor * channel_major = ggml_cont(
            context, ggml_permute(context, time_major, 1, 0, 2, 3));
        return ggml_silu(
            context,
            layer_norm(
                context,
                channel_major,
                weights.conv_norm_w,
                weights.conv_norm_b,
                config.layer_norm_eps));
    }
    ggml_tensor * scaled = ggml_mul(
        context,
        time_major,
        ggml_reshape_2d(context, weights.conv_bn_scale, 1, config.d_model));
    scaled = ggml_add(
        context,
        scaled,
        ggml_reshape_2d(context, weights.conv_bn_shift, 1, config.d_model));
    scaled = ggml_silu(context, scaled);
    return ggml_cont(context, ggml_permute(context, scaled, 1, 0, 2, 3));
}

ggml_tensor * symmetric_cached_convolution(
    ggml_context * context,
    const ParakeetCtcModel & model,
    ggml_tensor * value,
    ggml_tensor * time_cache,
    const BlockWeights & weights,
    const EncoderConfig & config,
    int current_frames,
    int committed_frames,
    int convolution_frames,
    ggml_tensor ** next_time_cache) {
    const int width = config.d_model;
    ggml_tensor * projected = add_bias(
        context,
        ggml_mul_mat(
            context,
            squeezed_pointwise(context, weights.conv_pw1_w, width, 2 * width),
            value),
        weights.conv_pw1_b);
    ggml_tensor * gated = gated_linear_unit(context, projected, width, current_frames);
    ggml_tensor * time_major = ggml_cont(
        context, ggml_permute(context, gated, 1, 0, 2, 3));
    ggml_tensor * with_history = ggml_concat(context, time_cache, time_major, 0);

    *next_time_cache = ggml_cont(
        context,
        ggml_view_2d(
            context,
            with_history,
            convolution_frames,
            width,
            with_history->nb[1],
            static_cast<size_t>(committed_frames) * with_history->nb[0]));

    ggml_tensor * padded = ggml_pad(context, with_history, convolution_frames, 0, 0, 0);
    ggml_tensor * convolved = ggml_conv_1d_dw(
        context,
        weights.conv_dw_w,
        ensure_contig_on_opencl(context, model, padded),
        1,
        0,
        1);
    if (weights.conv_dw_b) {
        convolved = ggml_add(
            context,
            convolved,
            ggml_reshape_2d(context, weights.conv_dw_b, 1, width));
    }
    ggml_tensor * normalized = convolution_norm(context, convolved, weights, config);
    return add_bias(
        context,
        ggml_mul_mat(
            context,
            squeezed_pointwise(context, weights.conv_pw2_w, width, width),
            normalized),
        weights.conv_pw2_b);
}

ggml_tensor * build_unified_block(
    ggml_context * context,
    const ParakeetCtcModel & model,
    ggml_tensor * value,
    ggml_tensor * channel_cache,
    ggml_tensor * time_cache,
    ggml_tensor * position_input,
    ggml_tensor * attention_mask,
    const BlockWeights & weights,
    const EncoderConfig & config,
    int current_frames,
    int committed_frames,
    int channel_frames,
    int convolution_frames,
    ggml_tensor ** next_channel_cache,
    ggml_tensor ** next_time_cache) {
    ggml_tensor * residual = value;
    ggml_tensor * transformed = feed_forward(
        context,
        value,
        weights.norm_ff1_w,
        weights.norm_ff1_b,
        weights.ff1_l1_w,
        weights.ff1_l1_b,
        weights.ff1_l2_w,
        weights.ff1_l2_b,
        config.layer_norm_eps);
    value = ggml_add(context, residual, ggml_scale(context, transformed, 0.5f));

    residual = value;
    ggml_tensor * normalized_attention = layer_norm(
        context,
        value,
        weights.norm_attn_w,
        weights.norm_attn_b,
        config.layer_norm_eps);
    ggml_tensor * key_value = ggml_concat(context, channel_cache, normalized_attention, 1);
    *next_channel_cache = update_channel_cache(
        context,
        channel_cache,
        leading_frames(context, normalized_attention, config.d_model, committed_frames),
        config.d_model,
        committed_frames,
        channel_frames);
    transformed = cached_attention(
        context,
        normalized_attention,
        key_value,
        position_input,
        attention_mask,
        weights,
        config.n_heads,
        config.head_dim,
        current_frames,
        channel_frames + current_frames);
    value = ggml_add(context, residual, transformed);

    residual = value;
    ggml_tensor * normalized_convolution = layer_norm(
        context,
        value,
        weights.norm_conv_w,
        weights.norm_conv_b,
        config.layer_norm_eps);
    transformed = symmetric_cached_convolution(
        context,
        model,
        normalized_convolution,
        time_cache,
        weights,
        config,
        current_frames,
        committed_frames,
        convolution_frames,
        next_time_cache);
    value = ggml_add(context, residual, transformed);

    residual = value;
    transformed = feed_forward(
        context,
        value,
        weights.norm_ff2_w,
        weights.norm_ff2_b,
        weights.ff2_l1_w,
        weights.ff2_l1_b,
        weights.ff2_l2_w,
        weights.ff2_l2_b,
        config.layer_norm_eps);
    value = ggml_add(context, residual, ggml_scale(context, transformed, 0.5f));
    return layer_norm(
        context,
        value,
        weights.norm_out_w,
        weights.norm_out_b,
        config.layer_norm_eps);
}

ggml_tensor * scaled_encoder_input(
    ggml_context * context,
    ggml_tensor * encoder_input,
    const EncoderConfig & config) {
    if (!config.xscaling) {
        return encoder_input;
    }
    return ggml_scale(context, encoder_input, std::sqrt(static_cast<float>(config.d_model)));
}

void add_layer_caches(
    ggml_context * context,
    const EncoderConfig & config,
    int channel_frames,
    int convolution_frames,
    UnifiedStepGraph & output) {
    ggml_tensor * channel_cache = ggml_new_tensor_2d(
        context, GGML_TYPE_F32, config.d_model, channel_frames);
    ggml_tensor * time_cache = ggml_new_tensor_2d(
        context, GGML_TYPE_F32, convolution_frames, config.d_model);
    ggml_set_input(channel_cache);
    ggml_set_input(time_cache);
    output.channel_cache_inputs.push_back(channel_cache);
    output.time_cache_inputs.push_back(time_cache);
}

ggml_tensor * build_unified_layers(
    ggml_context * context,
    const ParakeetCtcModel & model,
    ggml_tensor * value,
    int current_frames,
    int committed_frames,
    int channel_frames,
    int convolution_frames,
    UnifiedStepGraph & output) {
    const EncoderConfig & config = model.encoder_cfg;
    for (int layer = 0; layer < config.n_layers; ++layer) {
        add_layer_caches(context, config, channel_frames, convolution_frames, output);
        ggml_tensor * next_channel_cache = nullptr;
        ggml_tensor * next_time_cache = nullptr;
        value = build_unified_block(
            context,
            model,
            value,
            output.channel_cache_inputs.back(),
            output.time_cache_inputs.back(),
            output.position_input,
            output.attention_mask,
            model.blocks[layer],
            config,
            current_frames,
            committed_frames,
            channel_frames,
            convolution_frames,
            &next_channel_cache,
            &next_time_cache);
        ggml_set_output(next_channel_cache);
        ggml_set_output(next_time_cache);
        output.channel_cache_outputs.push_back(next_channel_cache);
        output.time_cache_outputs.push_back(next_time_cache);
    }
    return value;
}

void expand_cache_outputs(UnifiedStepGraph & output) {
    for (ggml_tensor * cache : output.channel_cache_outputs) {
        ggml_build_forward_expand(output.graph, cache);
    }
    for (ggml_tensor * cache : output.time_cache_outputs) {
        ggml_build_forward_expand(output.graph, cache);
    }
}

int reserve_step_graph(
    const ParakeetCtcModel & model,
    UnifiedStepGraph & output,
    size_t * measure_bytes) {
    ggml_backend_buffer_type_t buffer_type =
        ggml_backend_get_default_buffer_type(model.backend_active());
    if (measure_bytes) {
        ggml_gallocr_t pricer = ggml_gallocr_new(buffer_type);
        if (!pricer) {
            return -2;
        }
        *measure_bytes = 0;
        ggml_gallocr_reserve_n_size(pricer, output.graph, nullptr, nullptr, measure_bytes);
        ggml_gallocr_free(pricer);
        return 0;
    }
    output.allocator = ggml_gallocr_new(buffer_type);
    if (!output.allocator || !ggml_gallocr_reserve(output.allocator, output.graph)) {
        return -2;
    }
    return 0;
}

int build_step_graph(
    const ParakeetCtcModel & model,
    int current_frames,
    int committed_frames,
    UnifiedStepGraph & output,
    size_t * measure_bytes = nullptr) {
    output.clear();
    const EncoderConfig & config = model.encoder_cfg;
    const int channel_frames = channel_cache_frames(model);
    const int convolution_frames = convolution_cache_frames(model);
    const int key_frames = channel_frames + current_frames;
    const size_t graph_size = GGML_DEFAULT_GRAPH_SIZE * 16;
    ggml_init_params params = {};
    params.mem_size =
        ggml_tensor_overhead() * graph_size +
        ggml_graph_overhead_custom(graph_size, false) +
        128 * 1024;
    params.no_alloc = true;
    output.context = ggml_init(params);
    if (!output.context) {
        return -1;
    }

    ggml_context * context = output.context;
    output.encoder_input = ggml_new_tensor_2d(
        context, GGML_TYPE_F32, config.d_model, current_frames);
    output.attention_mask = ggml_new_tensor_2d(
        context, GGML_TYPE_F32, key_frames, current_frames);
    output.position_input = ggml_new_tensor_2d(
        context, GGML_TYPE_F32, config.d_model, 2 * key_frames - 1);
    ggml_set_input(output.encoder_input);
    ggml_set_input(output.attention_mask);
    ggml_set_input(output.position_input);

    output.channel_cache_inputs.reserve(config.n_layers);
    output.channel_cache_outputs.reserve(config.n_layers);
    output.time_cache_inputs.reserve(config.n_layers);
    output.time_cache_outputs.reserve(config.n_layers);
    output.encoder_output = build_unified_layers(
        context,
        model,
        scaled_encoder_input(context, output.encoder_input, config),
        current_frames,
        committed_frames,
        channel_frames,
        convolution_frames,
        output);
    ggml_set_output(output.encoder_output);

    output.graph = ggml_new_graph_custom(context, graph_size, false);
    expand_cache_outputs(output);
    ggml_build_forward_expand(output.graph, output.encoder_output);

    if (int rc = reserve_step_graph(model, output, measure_bytes); rc != 0) {
        output.clear();
        return rc;
    }
    if (measure_bytes) {
        output.clear();
        return 0;
    }
    output.current_frames = current_frames;
    output.committed_frames = committed_frames;
    output.channel_frames = channel_frames;
    output.convolution_frames = convolution_frames;
    output.positions = relative_positions(key_frames, config.d_model);
    return 0;
}

void open_mask_row(float * row, int first_key, int last_key) {
    for (int key = first_key; key < last_key; ++key) {
        row[key] = 0.0f;
    }
}

int first_visible_cache_slot(
    int query,
    int committed_frames,
    int cache_length,
    int channel_frames) {
    const int chunk_offset = committed_frames > 0 ? (query / committed_frames) * committed_frames : 0;
    return std::max(channel_frames - cache_length, chunk_offset);
}

void fill_attention_mask(
    int cache_length,
    int current_frames,
    int committed_frames,
    int channel_frames,
    std::vector<float> & mask) {
    const int key_frames = channel_frames + current_frames;
    mask.assign(static_cast<size_t>(key_frames) * current_frames, -1.0e30f);
    for (int query = 0; query < current_frames; ++query) {
        float * row = mask.data() + static_cast<size_t>(query) * key_frames;
        const int first_cache = first_visible_cache_slot(
            query, committed_frames, cache_length, channel_frames);
        open_mask_row(row, std::min(first_cache, channel_frames), key_frames);
    }
}

MelConfig raw_mel_config(const MelConfig & config) {
    MelConfig raw = config;
    raw.normalize = MelNormalize::None;
    return raw;
}

}

struct UnifiedStreamState::Impl {
    RnntDecodeState decoder;
    std::unique_ptr<UnifiedStepGraph> graph;
    IncrementalMelState incremental_mel;
    std::vector<float> raw_history;
    std::vector<float> pending_mel;
    std::vector<float> stats_window;
    std::vector<double> mean;
    std::vector<float> inv_std;
    int mel_width = 0;
    int subsampling_factor = kDefaultSubsamplingFactor;
    int stats_window_frames = 0;
    int committed_mel_frames = 0;
};

UnifiedStreamState::UnifiedStreamState() : impl(std::make_unique<Impl>()) {}
UnifiedStreamState::~UnifiedStreamState() = default;
UnifiedStreamState::UnifiedStreamState(UnifiedStreamState &&) noexcept = default;
UnifiedStreamState & UnifiedStreamState::operator=(UnifiedStreamState &&) noexcept = default;

namespace {

int frames_in(const std::vector<float> & mel, int n_mels) {
    return n_mels > 0 ? static_cast<int>(mel.size() / static_cast<size_t>(n_mels)) : 0;
}

void accumulate_mean(const std::vector<float> & window, int frames, int n_mels, std::vector<double> & mean) {
    mean.assign(static_cast<size_t>(n_mels), 0.0);
    for (int t = 0; t < frames; ++t) {
        for (int bin = 0; bin < n_mels; ++bin) {
            mean[bin] += window[static_cast<size_t>(t) * n_mels + bin];
        }
    }
    for (double & value : mean) {
        value /= std::max(1, frames);
    }
}

void accumulate_inv_std(
    const std::vector<float> & window,
    int frames,
    int n_mels,
    const std::vector<double> & mean,
    std::vector<float> & inv_std) {
    std::vector<double> squares(static_cast<size_t>(n_mels), 0.0);
    for (int t = 0; t < frames; ++t) {
        for (int bin = 0; bin < n_mels; ++bin) {
            const double d = window[static_cast<size_t>(t) * n_mels + bin] - mean[bin];
            squares[bin] += d * d;
        }
    }
    inv_std.assign(static_cast<size_t>(n_mels), 1.0f);
    const double denominator = std::max(1, frames - 1);
    for (int bin = 0; bin < n_mels; ++bin) {
        inv_std[bin] = 1.0f / static_cast<float>(std::sqrt(squares[bin] / denominator) + 1e-5);
    }
}

void normalize_frames(
    const float * raw,
    int frames,
    int n_mels,
    const std::vector<double> & mean,
    const std::vector<float> & inv_std,
    std::vector<float> & out) {
    for (int t = 0; t < frames; ++t) {
        for (int bin = 0; bin < n_mels; ++bin) {
            const size_t index = static_cast<size_t>(t) * n_mels + bin;
            out.push_back((raw[index] - static_cast<float>(mean[bin])) * inv_std[bin]);
        }
    }
}

void append_frames(std::vector<float> & target, const float * frames, int count, int n_mels) {
    target.insert(target.end(), frames, frames + static_cast<size_t>(count) * n_mels);
}

void trim_leading_frames(std::vector<float> & target, int keep_frames, int n_mels) {
    const int frames = frames_in(target, n_mels);
    if (frames <= keep_frames) {
        return;
    }
    target.erase(
        target.begin(),
        target.begin() + static_cast<std::ptrdiff_t>(frames - keep_frames) * n_mels);
}

void update_cmvn_statistics(UnifiedStreamState::Impl & impl, const float * incoming, int frames, int n_mels) {
    std::vector<float> window = impl.stats_window;
    append_frames(window, incoming, frames, n_mels);
    trim_leading_frames(window, impl.stats_window_frames, n_mels);
    const int window_frames = frames_in(window, n_mels);
    accumulate_mean(window, window_frames, n_mels, impl.mean);
    accumulate_inv_std(window, window_frames, n_mels, impl.mean, impl.inv_std);
}

void append_normalized_history(UnifiedStreamState::Impl & impl, int n_mels, std::vector<float> & processed) {
    const int history_frames = frames_in(impl.raw_history, n_mels);
    const int missing = kPreEncodeHistoryMelFrames - history_frames;
    processed.assign(static_cast<size_t>(std::max(0, missing)) * n_mels, 0.0f);
    normalize_frames(impl.raw_history.data(), history_frames, n_mels, impl.mean, impl.inv_std, processed);
}

void commit_mel_frames(UnifiedStreamState::Impl & impl, int commit_frames, int n_mels) {
    const float * committed = impl.pending_mel.data();
    append_frames(impl.stats_window, committed, commit_frames, n_mels);
    trim_leading_frames(impl.stats_window, impl.stats_window_frames, n_mels);
    append_frames(impl.raw_history, committed, commit_frames, n_mels);
    trim_leading_frames(impl.raw_history, kPreEncodeHistoryMelFrames, n_mels);
    impl.pending_mel.erase(
        impl.pending_mel.begin(),
        impl.pending_mel.begin() + static_cast<std::ptrdiff_t>(commit_frames) * n_mels);
    impl.committed_mel_frames = commit_frames;
}

int step_mel_frames(const UnifiedStreamState & state) {
    return state.impl->subsampling_factor * (state.chunk_frames + state.right_context_frames);
}

int chunk_mel_frames(const UnifiedStreamState & state) {
    return state.impl->subsampling_factor * state.chunk_frames;
}

bool state_accepts_input(const UnifiedStreamState & state) {
    return state.impl && state.chunk_frames > 0 && !state.cancelled && !state.finalized;
}

void upload_layer_caches(
    const ParakeetCtcModel & model,
    const UnifiedStepGraph & graph,
    const UnifiedStreamState & state) {
    const size_t channel_layer = static_cast<size_t>(graph.channel_frames) * model.encoder_cfg.d_model;
    const size_t time_layer = static_cast<size_t>(graph.convolution_frames) * model.encoder_cfg.d_model;
    for (int layer = 0; layer < model.encoder_cfg.n_layers; ++layer) {
        ggml_backend_tensor_set(
            graph.channel_cache_inputs[layer],
            state.cache_channel.data() + static_cast<size_t>(layer) * channel_layer,
            0,
            channel_layer * sizeof(float));
        ggml_backend_tensor_set(
            graph.time_cache_inputs[layer],
            state.cache_time.data() + static_cast<size_t>(layer) * time_layer,
            0,
            time_layer * sizeof(float));
    }
}

void download_layer_caches(
    const ParakeetCtcModel & model,
    const UnifiedStepGraph & graph,
    UnifiedStreamState & state) {
    const size_t channel_layer = static_cast<size_t>(graph.channel_frames) * model.encoder_cfg.d_model;
    const size_t time_layer = static_cast<size_t>(graph.convolution_frames) * model.encoder_cfg.d_model;
    for (int layer = 0; layer < model.encoder_cfg.n_layers; ++layer) {
        ggml_backend_tensor_get(
            graph.channel_cache_outputs[layer],
            state.cache_channel.data() + static_cast<size_t>(layer) * channel_layer,
            0,
            channel_layer * sizeof(float));
        ggml_backend_tensor_get(
            graph.time_cache_outputs[layer],
            state.cache_time.data() + static_cast<size_t>(layer) * time_layer,
            0,
            time_layer * sizeof(float));
    }
}

void prepare_step_inputs(
    UnifiedStepGraph & graph,
    const std::vector<float> & encoder_input,
    int cache_length) {
    ggml_backend_tensor_set(
        graph.encoder_input, encoder_input.data(), 0, encoder_input.size() * sizeof(float));
    std::vector<float> mask;
    fill_attention_mask(
        cache_length,
        graph.current_frames,
        graph.committed_frames,
        graph.channel_frames,
        mask);
    ggml_backend_tensor_set(graph.attention_mask, mask.data(), 0, mask.size() * sizeof(float));
    ggml_backend_tensor_set(
        graph.position_input, graph.positions.data(), 0, graph.positions.size() * sizeof(float));
}

int ensure_step_graph(
    const ParakeetCtcModel & model,
    UnifiedStreamState & state,
    int current_frames,
    int committed_frames) {
    UnifiedStepGraph * graph = state.impl->graph.get();
    if (graph &&
        graph->current_frames == current_frames &&
        graph->committed_frames == committed_frames) {
        return 0;
    }
    state.impl->graph = std::make_unique<UnifiedStepGraph>();
    return build_step_graph(model, current_frames, committed_frames, *state.impl->graph);
}

int run_encoder_step(
    ParakeetCtcModel & model,
    UnifiedStreamState & state,
    const std::vector<float> & encoder_input,
    int current_frames,
    int committed_frames,
    std::vector<float> & encoder_output) {
    if (int rc = ensure_step_graph(model, state, current_frames, committed_frames); rc != 0) {
        return rc;
    }
    UnifiedStepGraph & graph = *state.impl->graph;
    if (!ggml_gallocr_alloc_graph(graph.allocator, graph.graph)) {
        return -8;
    }
    prepare_step_inputs(graph, encoder_input, state.cache_length);
    upload_layer_caches(model, graph, state);
    if (ggml_backend_graph_compute(model.backend_active(), graph.graph) != GGML_STATUS_SUCCESS) {
        return -9;
    }
    encoder_output.resize(static_cast<size_t>(current_frames) * model.encoder_cfg.d_model);
    ggml_backend_tensor_get(
        graph.encoder_output, encoder_output.data(), 0, encoder_output.size() * sizeof(float));
    download_layer_caches(model, graph, state);
    return 0;
}

int decode_committed_frames(
    const ParakeetCtcModel & model,
    TdtRuntimeWeights & runtime,
    UnifiedStreamState & state,
    const std::vector<float> & encoder_output,
    int committed_frames,
    UnifiedStreamStepResult & result) {
    const int width = model.encoder_cfg.d_model;
    result.encoder_committed.assign(
        encoder_output.begin(),
        encoder_output.begin() + static_cast<std::ptrdiff_t>(committed_frames) * width);
    RnntDecodeOptions options;
    options.max_symbols_per_step = model.encoder_cfg.rnnt_max_symbols_per_step;
    return rnnt_decode_window(
        model,
        runtime,
        result.encoder_committed.data(),
        committed_frames,
        width,
        options,
        state.impl->decoder,
        result.new_token_ids,
        result.decoder_steps);
}

using ProfileClock = std::chrono::steady_clock;

double elapsed_ms(ProfileClock::time_point from, ProfileClock::time_point to) {
    return std::chrono::duration<double, std::milli>(to - from).count();
}

double g_profile_pcm_ms = 0.0;
double g_profile_signal_ms = 0.0;

bool profiling_enabled() {
    static const bool enabled = std::getenv("PARAKEET_UNIFIED_PROFILE") != nullptr;
    return enabled;
}

void log_step_profile(
    ProfileClock::time_point subsampling_start,
    ProfileClock::time_point encoder_start,
    ProfileClock::time_point decode_start,
    ProfileClock::time_point end) {
    if (!profiling_enabled()) return;
    std::fprintf(stderr, "unified-step sub=%.2f enc=%.2f dec=%.2f ms pcm_total=%.1f signal_total=%.1f\n",
                 elapsed_ms(subsampling_start, encoder_start),
                 elapsed_ms(encoder_start, decode_start),
                 elapsed_ms(decode_start, end),
                 g_profile_pcm_ms,
                 g_profile_signal_ms);
}

void write_raw_floats(const std::string & path, const std::vector<float> & values) {
    FILE * file = std::fopen(path.c_str(), "wb");
    if (!file) return;
    std::fwrite(values.data(), sizeof(float), values.size(), file);
    std::fclose(file);
}

void dump_step_for_debug(
    int step_index,
    const std::vector<float> & encoder_input,
    const std::vector<float> & encoder_output,
    const std::vector<float> & encoder_committed,
    int width) {
    const char * directory = std::getenv("PARAKEET_DUMP_UNIFIED_STEPS");
    if (!directory) return;
    const std::string prefix = std::string(directory) + "/step-" + std::to_string(step_index);
    write_raw_floats(prefix + "-input.f32", encoder_input);
    write_raw_floats(prefix + "-output.f32", encoder_output);
    write_raw_floats(prefix + "-committed.f32", encoder_committed);
    (void) width;
}

void advance_state(UnifiedStreamState & state, int committed_frames, int channel_frames, bool finalize) {
    state.cache_length = std::min(channel_frames, state.cache_length + committed_frames);
    ++state.step_index;
    state.emitted_encoder_frames += committed_frames;
    if (finalize) {
        state.finalized = true;
    }
}

}

bool unified_operating_point_supported(
    const ParakeetCtcModel & model,
    int chunk_frames,
    int right_context_frames) {
    const UnifiedStreamingConfig & cfg = model.unified_cfg;
    const bool chunk_ok = std::find(
        cfg.allowed_chunk_frames.begin(), cfg.allowed_chunk_frames.end(), chunk_frames) !=
        cfg.allowed_chunk_frames.end();
    const bool right_ok = std::find(
        cfg.allowed_right_context_frames.begin(),
        cfg.allowed_right_context_frames.end(),
        right_context_frames) != cfg.allowed_right_context_frames.end();
    return cfg.available && chunk_ok && right_ok;
}

void reset_unified_stream(UnifiedStreamState & state) {
    state.cache_channel.clear();
    state.cache_time.clear();
    state.token_ids.clear();
    state.cache_length = 0;
    state.chunk_frames = -1;
    state.right_context_frames = -1;
    state.step_index = 0;
    state.emitted_encoder_frames = 0;
    state.cancelled = false;
    state.finalized = false;
    state.impl = std::make_unique<UnifiedStreamState::Impl>();
}

void cancel_unified_stream(UnifiedStreamState & state) {
    state.cancelled = true;
}

int unified_pending_mel_frames(const UnifiedStreamState & state) {
    if (!state.impl || state.impl->mel_width <= 0) {
        return 0;
    }
    return frames_in(state.impl->pending_mel, state.impl->mel_width);
}

int init_unified_stream_state(
    const ParakeetCtcModel & model,
    int chunk_frames,
    int right_context_frames,
    UnifiedStreamState & state) {
    if (model.model_type != ParakeetModelType::RNNT || !model.unified_cfg.available) {
        return -1;
    }
    if (!unified_operating_point_supported(model, chunk_frames, right_context_frames)) {
        return -2;
    }
    reset_unified_stream(state);
    state.chunk_frames = chunk_frames;
    state.right_context_frames = right_context_frames;
    state.impl->subsampling_factor = encoder_subsampling_factor(model);
    state.impl->stats_window_frames = state.impl->subsampling_factor *
        (channel_cache_frames(model) + chunk_frames + right_context_frames);
    state.cache_channel.assign(
        static_cast<size_t>(model.encoder_cfg.n_layers) * channel_cache_frames(model) * model.encoder_cfg.d_model,
        0.0f);
    state.cache_time.assign(
        static_cast<size_t>(model.encoder_cfg.n_layers) * model.encoder_cfg.d_model * convolution_cache_frames(model),
        0.0f);
    return 0;
}

int append_unified_mel_frames(
    UnifiedStreamState & state,
    const float * mel,
    int n_frames,
    int n_mels) {
    if (!state_accepts_input(state)) {
        return -1;
    }
    if (!mel || n_frames <= 0 || n_mels <= 0) {
        return -2;
    }
    if (state.impl->mel_width != 0 && state.impl->mel_width != n_mels) {
        return -3;
    }
    state.impl->mel_width = n_mels;
    append_frames(state.impl->pending_mel, mel, n_frames, n_mels);
    return 0;
}

int append_unified_pcm(
    const ParakeetCtcModel & model,
    UnifiedStreamState & state,
    const float * samples,
    int n_samples,
    bool finalize) {
    if (!state_accepts_input(state)) {
        return -1;
    }
    const auto started = ProfileClock::now();
    std::vector<float> mel;
    int mel_frames = 0;
    struct Accumulate {
        ProfileClock::time_point from;
        ~Accumulate() { g_profile_pcm_ms += elapsed_ms(from, ProfileClock::now()); }
    } accumulate{started};
    if (int rc = append_log_mel(
            samples,
            n_samples,
            finalize,
            raw_mel_config(model.mel_cfg),
            state.impl->incremental_mel,
            mel,
            mel_frames); rc != 0) {
        return rc;
    }
    if (mel_frames == 0) {
        return 0;
    }
    return append_unified_mel_frames(state, mel.data(), mel_frames, model.mel_cfg.n_mels);
}

int next_unified_processed_signal(
    UnifiedStreamState & state,
    int n_mels,
    bool finalize,
    std::vector<float> & processed_signal,
    int & n_frames) {
    processed_signal.clear();
    n_frames = 0;
    if (state.impl && state.impl->mel_width == 0 && !state.cancelled && !state.finalized) {
        return 0;
    }
    if (!state_accepts_input(state) || n_mels <= 0 || state.impl->mel_width != n_mels) {
        return -1;
    }
    UnifiedStreamState::Impl & impl = *state.impl;
    struct Accumulate {
        ProfileClock::time_point from = ProfileClock::now();
        ~Accumulate() { g_profile_signal_ms += elapsed_ms(from, ProfileClock::now()); }
    } accumulate;
    const int pending_frames = frames_in(impl.pending_mel, n_mels);
    const int required_frames = step_mel_frames(state);
    if ((!finalize && pending_frames < required_frames) || pending_frames == 0) {
        return 0;
    }
    const int consumed_frames = std::min(required_frames, pending_frames);
    const bool last_step = finalize && pending_frames <= required_frames;
    const int commit_frames = last_step ? consumed_frames : std::min(chunk_mel_frames(state), consumed_frames);

    update_cmvn_statistics(impl, impl.pending_mel.data(), consumed_frames, n_mels);
    append_normalized_history(impl, n_mels, processed_signal);
    normalize_frames(impl.pending_mel.data(), consumed_frames, n_mels, impl.mean, impl.inv_std, processed_signal);
    n_frames = frames_in(processed_signal, n_mels);
    commit_mel_frames(impl, commit_frames, n_mels);
    return 1;
}

int run_unified_stream_step(
    ParakeetCtcModel & model,
    TdtRuntimeWeights & runtime,
    const float * processed_signal,
    int n_mel_frames,
    int n_mels,
    bool finalize,
    UnifiedStreamState & state,
    UnifiedStreamStepResult & result) {
    result = UnifiedStreamStepResult{};
    if (state.cancelled) {
        return -1;
    }
    if (state.finalized) {
        return finalize ? 0 : -2;
    }
    if (!state.impl || state.chunk_frames <= 0 || state.right_context_frames < 0) {
        return -3;
    }
    if (!processed_signal || n_mel_frames <= 0) {
        if (finalize) {
            state.finalized = true;
            return 0;
        }
        return -4;
    }
    if (n_mels != model.mel_cfg.n_mels) {
        return -5;
    }

    const auto t_subsampling = std::chrono::steady_clock::now();
    std::vector<float> subsampled;
    int subsampled_frames = 0;
    if (int rc = run_subsampling(model, processed_signal, n_mel_frames, n_mels, subsampled, subsampled_frames);
        rc != 0) {
        return rc;
    }
    const auto t_encoder = std::chrono::steady_clock::now();
    const int current_frames = subsampled_frames - kSubsamplingHistoryFrames;
    if (current_frames <= 0) {
        return -6;
    }
    const int expected_frames = state.chunk_frames + state.right_context_frames;
    if (!finalize && current_frames != expected_frames) {
        return -7;
    }
    const int committed_frames = finalize ? current_frames : state.chunk_frames;

    const int width = model.encoder_cfg.d_model;
    const std::vector<float> encoder_input(
        subsampled.begin() + static_cast<std::ptrdiff_t>(kSubsamplingHistoryFrames) * width,
        subsampled.end());
    std::vector<float> encoder_output;
    if (int rc = run_encoder_step(model, state, encoder_input, current_frames, committed_frames, encoder_output);
        rc != 0) {
        return rc;
    }
    const auto t_decode = std::chrono::steady_clock::now();
    if (int rc = decode_committed_frames(model, runtime, state, encoder_output, committed_frames, result);
        rc != 0) {
        return rc;
    }
    log_step_profile(t_subsampling, t_encoder, t_decode, std::chrono::steady_clock::now());
    dump_step_for_debug(state.step_index, encoder_input, encoder_output, result.encoder_committed, width);
    state.token_ids.insert(state.token_ids.end(), result.new_token_ids.begin(), result.new_token_ids.end());
    result.text = detokenize(model.vocab, state.token_ids);
    result.committed_frames = committed_frames;
    result.provisional_frames = current_frames - committed_frames;
    advance_state(state, committed_frames, channel_cache_frames(model), finalize);
    return 0;
}

size_t unified_stream_graph_buffer_bytes(const UnifiedStreamState & state) {
    if (!state.impl || !state.impl->graph || !state.impl->graph->allocator) {
        return 0;
    }
    return ggml_gallocr_get_buffer_size(state.impl->graph->allocator, 0);
}

}
