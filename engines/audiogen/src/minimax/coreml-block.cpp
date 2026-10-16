#include "coreml-block.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>

namespace tts_cpp::minimax::detail {
namespace {

std::vector<float> read_values(const ggml_tensor * tensor) {
    const auto * data = static_cast<const float *>(tensor->data);
    return {data, data + ggml_nelements(tensor)};
}

std::vector<float> read_positions(const ggml_tensor * tensor, int64_t tokens) {
    std::vector<float> result(static_cast<size_t>(tokens));
    for (int64_t i = 0; i < tokens; ++i) {
        result[static_cast<size_t>(i)] = tensor->type == GGML_TYPE_I64
            ? static_cast<float>(static_cast<const int64_t *>(tensor->data)[i])
            : static_cast<float>(static_cast<const int32_t *>(tensor->data)[i]);
    }
    return result;
}

std::vector<CoreMLArray> block_inputs(const ggml_tensor * dst) {
    const ggml_tensor * hidden = dst->src[0];
    const int64_t tokens = hidden->ne[1];
    const int64_t batch = hidden->ne[2];
    std::vector<CoreMLArray> inputs = {
        {"hidden", {batch, tokens, hidden->ne[0]}, read_values(hidden)},
        {"positions", {tokens}, read_positions(dst->src[1], tokens)},
    };
    if (!dst->src[2]) return inputs;
    const ggml_tensor * keys = dst->src[2];
    inputs.push_back({"keys", {batch, keys->ne[2], keys->ne[1], keys->ne[0]}, read_values(keys)});
    inputs.push_back({"values", inputs.back().shape, read_values(dst->src[3])});
    inputs.push_back({"mask", {1, 1, tokens, keys->ne[1]}, dst->src[4] ? read_values(dst->src[4]) : std::vector<float>(static_cast<size_t>(tokens * keys->ne[1]), 0.0f)});
    return inputs;
}

void copy_predictions(ggml_tensor * dst, const std::vector<CoreMLArray> & outputs) {
    size_t offset = 0;
    for (const auto & output : outputs) {
        std::memcpy(static_cast<float *>(dst->data) + offset, output.values.data(), output.values.size() * sizeof(float));
        offset += output.values.size();
    }
}

void predict_block(ggml_tensor * dst, int, int, void * user_data) {
    try {
        auto & block = *static_cast<CoreMLBlock *>(user_data);
        const auto inputs = block_inputs(dst);
        const auto & shape = inputs[0].shape;
        std::vector<CoreMLArray> outputs = {{"hidden_out", shape, {}}};
        if (dst->src[2]) {
            const auto & cache = inputs[2].shape;
            const std::vector<int64_t> new_shape = {shape[0], shape[1], cache[1], cache[3]};
            outputs.push_back({"new_keys", new_shape, {}});
            outputs.push_back({"new_values", new_shape, {}});
        }
        if (coreml_block_predict(block, inputs, outputs)) {
            copy_predictions(dst, outputs);
            return;
        }
    } catch (const std::exception & error) {
        coreml_block_set_error(*static_cast<CoreMLBlock *>(user_data), error.what());
    } catch (...) {
        coreml_block_set_error(*static_cast<CoreMLBlock *>(user_data), "Core ML callback failed");
    }
    std::fill_n(static_cast<float *>(dst->data), ggml_nelements(dst), 0.0f);
}

ggml_tensor * contiguous_f32(ggml_context * ctx, ggml_tensor * tensor) {
    return ggml_cont(ctx, ggml_cast(ctx, tensor, GGML_TYPE_F32));
}

ggml_tensor * cache_view(ggml_context * ctx, ggml_tensor * tensor, int64_t length,
                         int64_t heads, int64_t dim, int64_t batch, bool lm) {
    if (!lm) {
        return ggml_view_4d(ctx, tensor, dim, length, heads, batch,
                            tensor->nb[1], tensor->nb[2], tensor->nb[3], 0);
    }
    return ggml_view_4d(ctx, tensor, dim, length, heads, batch, tensor->nb[1],
                        static_cast<size_t>(dim) * tensor->nb[0],
                        static_cast<size_t>(tensor->ne[1] / batch) * tensor->nb[1], 0);
}

ggml_tensor * new_cache_rows(ggml_context * ctx, ggml_tensor * packed, int64_t offset,
                            int64_t tokens, int64_t heads, int64_t dim, int64_t batch, bool lm) {
    ggml_tensor * result = ggml_view_4d(ctx, packed, dim, heads, tokens, batch,
        static_cast<size_t>(dim) * sizeof(float), static_cast<size_t>(dim * heads) * sizeof(float),
        static_cast<size_t>(dim * heads * tokens) * sizeof(float), static_cast<size_t>(offset) * sizeof(float));
    if (lm) return ggml_reshape_2d(ctx, result, dim * heads, tokens * batch);
    return ggml_cont(ctx, ggml_permute(ctx, result, 0, 2, 1, 3));
}

void write_cache(ggml_context * ctx, ggml_cgraph * graph, ggml_tensor * packed,
                  ggml_tensor * keys, ggml_tensor * values, ggml_tensor * rows,
                  int64_t hidden_size, int64_t tokens, int64_t heads, int64_t dim, int64_t batch, bool lm) {
    const int64_t count = tokens * heads * dim * batch;
    auto * new_keys = new_cache_rows(ctx, packed, hidden_size, tokens, heads, dim, batch, lm);
    auto * new_values = new_cache_rows(ctx, packed, hidden_size + count, tokens, heads, dim, batch, lm);
    ggml_build_forward_expand(graph, ggml_set_rows(ctx, keys, new_keys, rows));
    ggml_build_forward_expand(graph, ggml_set_rows(ctx, values, new_values, rows));
}

}

ggml_tensor * coreml_transformer_block(ggml_context * ctx, ggml_cgraph * graph,
                                      const std::shared_ptr<CoreMLBlock> & block,
                                      ggml_tensor * hidden, ggml_tensor * positions, ggml_tensor * mask,
                                      ggml_tensor * keys, ggml_tensor * values, ggml_tensor * rows,
                                      int64_t length, int64_t heads, int64_t dim, bool lm) {
    const int64_t tokens = hidden->ne[1];
    const int64_t batch = hidden->ne[2];
    if (!coreml_block_supports(block, tokens, length, batch)) return nullptr;
    ggml_tensor * args[] = {ggml_cont(ctx, hidden), positions, nullptr, nullptr, nullptr};
    const int64_t hidden_size = ggml_nelements(hidden);
    int64_t packed_size = hidden_size;
    if (keys) {
        args[2] = contiguous_f32(ctx, cache_view(ctx, keys, length, heads, dim, batch, lm));
        args[3] = contiguous_f32(ctx, cache_view(ctx, values, length, heads, dim, batch, lm));
        args[4] = mask ? contiguous_f32(ctx, mask) : nullptr;
        packed_size += 2 * tokens * heads * dim * batch;
    }
    auto * packed = ggml_custom_4d(ctx, GGML_TYPE_F32, packed_size, 1, 1, 1,
                                   args, keys ? 5 : 2, predict_block, 1, block.get());
    ggml_set_name(packed, "mm3_coreml_transformer");
    if (keys) write_cache(ctx, graph, packed, keys, values, rows, hidden_size, tokens, heads, dim, batch, lm);
    return ggml_view_3d(ctx, packed, hidden->ne[0], tokens, batch,
                        static_cast<size_t>(hidden->ne[0]) * sizeof(float),
                        static_cast<size_t>(hidden->ne[0] * tokens) * sizeof(float), 0);
}

#ifndef AUDIOGEN_USE_COREML
std::string coreml_source_fingerprint(const std::string &) { return {}; }
std::shared_ptr<CoreMLBlock> load_coreml_block(const std::string &, const std::string &, const CoreMLBlockConfig &) { return {}; }
bool coreml_block_supports(const std::shared_ptr<CoreMLBlock> &, int64_t, int64_t, int64_t) { return false; }
bool coreml_block_predict(CoreMLBlock &, const std::vector<CoreMLArray> &, std::vector<CoreMLArray> &) { return false; }
void coreml_block_set_error(CoreMLBlock &, const std::string &) {}
std::string coreml_block_error(const std::shared_ptr<CoreMLBlock> &) { return {}; }
uint64_t coreml_block_predictions(const std::shared_ptr<CoreMLBlock> &) { return 0; }
#endif

}
