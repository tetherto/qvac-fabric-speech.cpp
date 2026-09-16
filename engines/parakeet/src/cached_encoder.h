#pragma once

// Graph builders shared by the cache-aware streaming encoders (Nemotron and
// Unified RNN-T): relative-position attention over a channel cache plus the
// current frames, the feed-forward module, and the channel-cache roll.
// Tensors are (d_model, frames); the definitions live in parakeet_nemotron.cpp.

#include "parakeet_ctc.h"

#include "ggml.h"

#include <vector>

namespace parakeet {
namespace cached_encoder {

ggml_tensor * ensure_contig_on_opencl(
    ggml_context * ctx,
    const ParakeetCtcModel & model,
    ggml_tensor * tensor);

ggml_tensor * add_bias(
    ggml_context * context,
    ggml_tensor * value,
    ggml_tensor * bias);

ggml_tensor * layer_norm(
    ggml_context * context,
    ggml_tensor * value,
    ggml_tensor * weight,
    ggml_tensor * bias,
    float epsilon);

ggml_tensor * feed_forward(
    ggml_context * context,
    ggml_tensor * value,
    ggml_tensor * norm_w,
    ggml_tensor * norm_b,
    ggml_tensor * l1_w,
    ggml_tensor * l1_b,
    ggml_tensor * l2_w,
    ggml_tensor * l2_b,
    float epsilon);

std::vector<float> relative_positions(int total_frames, int width);

ggml_tensor * cached_attention(
    ggml_context * context,
    ggml_tensor * query_input,
    ggml_tensor * key_value_input,
    ggml_tensor * position_input,
    ggml_tensor * attention_mask,
    const BlockWeights & weights,
    int heads,
    int head_width,
    int query_frames,
    int key_frames);

ggml_tensor * update_channel_cache(
    ggml_context * context,
    ggml_tensor * cache,
    ggml_tensor * current,
    int width,
    int current_frames,
    int channel_frames);

}
}
