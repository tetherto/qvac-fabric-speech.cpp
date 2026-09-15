#include "audio8/graph.h"

#include "backend_util.h"

#include <cmath>

namespace tts_cpp {
namespace audio8 {
namespace detail {
namespace {

const size_t FLOAT = sizeof(float);

size_t graph_arena(int nodes) {
    return static_cast<size_t>(nodes) * ggml_tensor_overhead() +
           ggml_graph_overhead_custom(nodes, /*grads=*/false);
}

ggml_tensor * table_window(ggml_context * ctx, ggml_tensor * table, int first, int count) {
    ggml_tensor * rows = ggml_view_2d(ctx, table, table->ne[0], count, table->nb[1],
                                      static_cast<size_t>(first) * table->nb[1]);
    return ggml_reshape_3d(ctx, rows, table->ne[0], 1, count);
}

ggml_tensor * lower_half(ggml_context * ctx, ggml_tensor * x) {
    return ggml_view_3d(ctx, x, x->ne[0] / 2, x->ne[1], x->ne[2], x->nb[1], x->nb[2], 0);
}

ggml_tensor * upper_half(ggml_context * ctx, ggml_tensor * x) {
    const size_t half = static_cast<size_t>(x->ne[0] / 2) * x->nb[0];
    return ggml_view_3d(ctx, x, x->ne[0] / 2, x->ne[1], x->ne[2], x->nb[1], x->nb[2], half);
}

ggml_tensor * swap_halves(ggml_context * ctx, ggml_tensor * x) {
    return ggml_concat(ctx, upper_half(ctx, x), lower_half(ctx, x), 0);
}

ggml_tensor * project_heads(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * bias,
                            ggml_tensor * x, int head_dim, int heads) {
    return ggml_reshape_3d(ctx, linear(ctx, weight, x, bias), head_dim, heads,
                           static_cast<int>(x->ne[1]));
}

// K is stored position-major so a step appends one contiguous row per position.
ggml_tensor * cache_keys(ggml_context * ctx, const kv_cache & cache,
                         const attention_shape & shape, int total) {
    const size_t layer_bytes = static_cast<size_t>(cache.capacity) * cache.stride * FLOAT;
    return ggml_view_3d(ctx, cache.k, shape.head_dim, shape.n_kv, total,
                        static_cast<size_t>(shape.head_dim) * FLOAT,
                        static_cast<size_t>(cache.stride) * FLOAT,
                        static_cast<size_t>(shape.layer) * layer_bytes);
}

// V is stored channel-major so the value matmul can read rows of positions
// without transposing the whole prefix on every step.
ggml_tensor * cache_values(ggml_context * ctx, const kv_cache & cache,
                           const attention_shape & shape, int total) {
    const size_t column = static_cast<size_t>(cache.capacity) * FLOAT;
    const size_t layer_bytes = column * cache.stride;
    return ggml_view_3d(ctx, cache.v, total, shape.head_dim, shape.n_kv, column,
                        column * shape.head_dim,
                        static_cast<size_t>(shape.layer) * layer_bytes);
}

void append_keys(ggml_context * ctx, ggml_cgraph * graph, const kv_cache & cache,
                 const attention_shape & shape, ggml_tensor * keys) {
    const size_t layer_bytes = static_cast<size_t>(cache.capacity) * cache.stride * FLOAT;
    const size_t offset = static_cast<size_t>(shape.layer) * layer_bytes +
                          static_cast<size_t>(shape.n_past) * cache.stride * FLOAT;
    ggml_tensor * slot = ggml_view_2d(ctx, cache.k, cache.stride, shape.width,
                                      static_cast<size_t>(cache.stride) * FLOAT, offset);
    ggml_tensor * flat = ggml_reshape_2d(ctx, keys, cache.stride, shape.width);
    ggml_build_forward_expand(graph, ggml_cpy(ctx, flat, slot));
}

void append_values(ggml_context * ctx, ggml_cgraph * graph, const kv_cache & cache,
                   const attention_shape & shape, ggml_tensor * values) {
    const size_t column = static_cast<size_t>(cache.capacity) * FLOAT;
    const size_t offset = static_cast<size_t>(shape.layer) * column * cache.stride +
                          static_cast<size_t>(shape.n_past) * FLOAT;
    ggml_tensor * slot = ggml_view_2d(ctx, cache.v, shape.width, cache.stride, column, offset);
    ggml_tensor * flat = ggml_reshape_2d(ctx, values, cache.stride, shape.width);
    ggml_build_forward_expand(graph, ggml_cpy(ctx, ggml_transpose(ctx, flat), slot));
}

ggml_tensor * attend(ggml_context * ctx, ggml_tensor * query, ggml_tensor * keys,
                     ggml_tensor * values, ggml_tensor * mask, int head_dim, int n_head,
                     bool precise_values) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    ggml_tensor * scores = precise_mul_mat(ctx, keys, query);
    ggml_tensor * weights = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f);
    ggml_tensor * blended = multiply_mat(ctx, values, weights, precise_values);
    ggml_tensor * merged = ggml_cont(ctx, ggml_permute(ctx, blended, 0, 2, 1, 3));
    return ggml_reshape_2d(ctx, merged, head_dim * n_head, blended->ne[1]);
}

ggml_tensor * rotated_query(ggml_context * ctx, const attention_weights & weights,
                            ggml_tensor * x, const rope_planes & rope,
                            const attention_shape & shape) {
    ggml_tensor * query = project_heads(ctx, weights.wq, weights.wq_b, x, shape.head_dim,
                                        shape.n_head);
    return ggml_permute(ctx, apply_rope(ctx, query, rope), 0, 2, 1, 3);
}

// GGML_PREC_F32 buys precision the operands still have. A block-quantised
// weight has already spent it, and on CUDA the default path -- the integer dot
// product every CPU build of this tier also takes -- accumulates in f32
// anyway, so the marker only forces a dequantise-to-f32 round trip through
// cuBLAS -- 48% of GPU kernel time, a fifth of the decode's wall, since the
// loop is host-bound rather than GPU-bound. Other backends keep the marker:
// ggml-vulkan reduces quantised matmuls in f16 under PREC_DEFAULT, which is a
// genuine loss rather than the same arithmetic spelled differently.
bool quantised_matmul_is_already_f32(const ggml_tensor * weight) {
    return ggml_is_quantized(weight->type) && weight->buffer &&
           ::tts_cpp::detail::reg_name_is_cuda(
               ::tts_cpp::detail::buft_reg_name(ggml_backend_buffer_get_type(weight->buffer)));
}

}  // namespace

ggml_tensor * precise_mul_mat(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    ggml_tensor * out = ggml_mul_mat(ctx, a, b);
    if (!quantised_matmul_is_already_f32(a)) {
        ggml_mul_mat_set_prec(out, GGML_PREC_F32);
    }
    return out;
}

ggml_tensor * multiply_mat(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b,
                           bool precise) {
    return precise ? precise_mul_mat(ctx, a, b) : ggml_mul_mat(ctx, a, b);
}

scratch::scratch(int nodes) {
    ggml_init_params params = {graph_arena(nodes), nullptr, /*no_alloc=*/true};
    ctx = ggml_init(params);
    if (ctx) graph = ggml_new_graph_custom(ctx, nodes, /*grads=*/false);
}

scratch::~scratch() {
    if (ctx) ggml_free(ctx);
}

ggml_tensor * input_i32(ggml_context * ctx, const char * name, int count) {
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, count);
    ggml_set_name(t, name);
    ggml_set_input(t);
    return t;
}

ggml_tensor * input_f32(ggml_context * ctx, const char * name, int ne0, int ne1) {
    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    ggml_set_name(t, name);
    ggml_set_input(t);
    return t;
}

ggml_tensor * mark_output(ggml_cgraph * graph, ggml_tensor * t) {
    ggml_set_output(t);
    ggml_build_forward_expand(graph, t);
    return t;
}

void write_input(ggml_cgraph * graph, const char * name, const void * data, size_t bytes) {
    ggml_backend_tensor_set(ggml_graph_get_tensor(graph, name), data, 0, bytes);
}

void read_output(ggml_tensor * t, std::vector<float> & out) {
    out.resize(ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
}

void read_ids(ggml_tensor * t, std::vector<int32_t> & out) {
    out.resize(ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
}

bool prepare_graph(ggml_backend_t backend, ::tts_cpp::detail::sched_fallback & sched,
                   ggml_backend_buffer_t weight_buffer, ggml_gallocr_t allocr,
                   ggml_cgraph * graph, const char * stage, bool & use_sched,
                   std::string * error) {
    use_sched = ::tts_cpp::detail::sched_force_enabled() ||
                !::tts_cpp::detail::graph_fully_supported(backend, graph);
    if (!use_sched) {
        if (ggml_gallocr_reserve(allocr, graph) && ggml_gallocr_alloc_graph(allocr, graph)) {
            return true;
        }
        if (error) *error = std::string("audio8: ") + stage + " graph allocation failed";
        return false;
    }
    if (::tts_cpp::detail::graph_has_unsupported_preallocated_op(backend, graph)) {
        if (error) {
            *error = std::string("audio8: ") + stage +
                     " graph has an unsupported persistent-buffer operation";
        }
        return false;
    }
    if (!::tts_cpp::detail::sched_fallback_ensure(
            sched, backend, 2 * AUDIO8_MAX_NODES, {weight_buffer}) ||
        !::tts_cpp::detail::sched_fallback_alloc(sched, graph)) {
        if (error) *error = std::string("audio8: ") + stage + " graph allocation failed";
        return false;
    }
    return true;
}

bool compute_graph(ggml_backend_t backend, ::tts_cpp::detail::sched_fallback & sched,
                   ggml_cgraph * graph, bool use_sched, int n_threads,
                   const char * stage, std::string * error) {
    const ggml_status status =
        use_sched ? ::tts_cpp::detail::sched_fallback_compute(sched, backend, graph, n_threads)
                  : ::tts_cpp::detail::direct_compute(backend, graph, n_threads);
    if (status == GGML_STATUS_SUCCESS) return true;
    if (error) *error = std::string("audio8: ") + stage + " graph compute failed";
    return false;
}

rope_planes rope_window(ggml_context * ctx, ggml_tensor * cos_table,
                        ggml_tensor * sin_table, int first, int count) {
    ggml_tensor * cosines = table_window(ctx, cos_table, first, count);
    ggml_tensor * sines = table_window(ctx, sin_table, first, count);
    return {ggml_concat(ctx, cosines, cosines, 0),
            ggml_concat(ctx, ggml_neg(ctx, sines), sines, 0)};
}

ggml_tensor * apply_rope(ggml_context * ctx, ggml_tensor * x, const rope_planes & rope) {
    return ggml_add(ctx, ggml_mul(ctx, x, rope.cos),
                    ggml_mul(ctx, swap_halves(ctx, x), rope.signed_sin));
}

ggml_tensor * rms_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * weight, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), weight);
}

ggml_tensor * linear(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * x,
                     ggml_tensor * bias) {
    ggml_tensor * out = precise_mul_mat(ctx, weight, x);
    return bias ? ggml_add(ctx, out, bias) : out;
}

ggml_tensor * swiglu(ggml_context * ctx, ggml_tensor * w1, ggml_tensor * w2,
                     ggml_tensor * w3, ggml_tensor * x) {
    ggml_tensor * gated = ggml_swiglu_split(ctx, precise_mul_mat(ctx, w1, x),
                                            precise_mul_mat(ctx, w3, x));
    return precise_mul_mat(ctx, w2, gated);
}

ggml_tensor * attention(ggml_context * ctx, ggml_cgraph * graph,
                        const attention_weights & weights, ggml_tensor * x,
                        const rope_planes & rope, const kv_cache & cache,
                        const attention_shape & shape, ggml_tensor * mask) {
    ggml_tensor * key = project_heads(ctx, weights.wk, weights.wk_b, x, shape.head_dim,
                                      shape.n_kv);
    ggml_tensor * value = project_heads(ctx, weights.wv, weights.wv_b, x, shape.head_dim,
                                        shape.n_kv);
    append_keys(ctx, graph, cache, shape, apply_rope(ctx, key, rope));
    append_values(ctx, graph, cache, shape, value);

    const int total = shape.n_past + shape.width;
    ggml_tensor * query = rotated_query(ctx, weights, x, rope, shape);
    ggml_tensor * keys = ggml_permute(ctx, cache_keys(ctx, cache, shape, total), 0, 2, 1, 3);
    ggml_tensor * values = cache_values(ctx, cache, shape, total);
    ggml_tensor * merged = attend(ctx, query, keys, values, mask, shape.head_dim,
                                  shape.n_head, shape.precise_values);
    return linear(ctx, weights.wo, merged, nullptr);
}

ggml_tensor * windowed_attention(ggml_context * ctx, const attention_weights & weights,
                                 ggml_tensor * x, const rope_planes & rope,
                                 const attention_shape & shape, ggml_tensor * mask) {
    ggml_tensor * key = project_heads(ctx, weights.wk, weights.wk_b, x, shape.head_dim,
                                      shape.n_kv);
    ggml_tensor * value = project_heads(ctx, weights.wv, weights.wv_b, x, shape.head_dim,
                                        shape.n_kv);
    ggml_tensor * query = rotated_query(ctx, weights, x, rope, shape);
    ggml_tensor * keys = ggml_permute(ctx, apply_rope(ctx, key, rope), 0, 2, 1, 3);
    ggml_tensor * values = ggml_cont(ctx, ggml_permute(ctx, value, 1, 2, 0, 3));
    ggml_tensor * merged = attend(ctx, query, keys, values, mask, shape.head_dim,
                                  shape.n_head, shape.precise_values);
    return linear(ctx, weights.wo, merged, nullptr);
}

void fill_causal_mask(float * mask, int keys, int queries, int first_query, int window) {
    for (int query = 0; query < queries; ++query) {
        const int position = first_query + query;
        const int oldest = window > 0 ? position - window + 1 : 0;
        for (int key = 0; key < keys; ++key) {
            const bool visible = key <= position && key >= oldest;
            mask[static_cast<size_t>(query) * keys + key] = visible ? 0.0f : -INFINITY;
        }
    }
}

}  // namespace detail
}  // namespace audio8
}  // namespace tts_cpp
