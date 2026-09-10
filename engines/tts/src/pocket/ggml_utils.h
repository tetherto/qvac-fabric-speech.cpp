#pragma once

#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "pocket/memory.h"
#include "gguf_stream.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace tts_cpp::pocket::detail {

inline void require_finite(const std::vector<float> & x, const std::string & what) {
    for (float value : x) if (!std::isfinite(value)) throw std::runtime_error("pocket: non-finite " + what);
}

// One source tensor at a time, never a second copy of the complete checkpoint.
// Validate payloads before exposing them to graph or host-side arithmetic.
inline void load_float_weight(::tts_cpp::detail::gguf_stream_reader & reader,
                              ggml_tensor * source, ggml_tensor * destination) {
    const auto n = static_cast<size_t>(ggml_nelements(source));
    std::vector<float> values(n);
    if (source->type == GGML_TYPE_F32) {
        if (!reader.to_host(ggml_get_name(source), values.data(), n * sizeof(float)))
            throw std::runtime_error("pocket: weight read failed");
    } else if (source->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> half(n);
        if (!reader.to_host(ggml_get_name(source), half.data(), n * sizeof(ggml_fp16_t)))
            throw std::runtime_error("pocket: half weight read failed");
        ggml_fp16_to_fp32_row(half.data(), values.data(), static_cast<int64_t>(n));
    } else {
        throw std::runtime_error("pocket: unsupported weight type");
    }
    require_finite(values, ggml_get_name(source));
    ggml_backend_tensor_set(destination, values.data(), 0, n * sizeof(float));
}

// A non-null data marker makes ggml's size-only allocator treat persistent
// tensors as externally allocated. It must never be dereferenced or executed.
inline void mark_external(ggml_context * ctx) {
    for (auto * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t))
        t->data = reinterpret_cast<void *>(uintptr_t(0x1000));
}
inline uint64_t price_cpu_graph(ggml_gallocr_t allocator, ggml_cgraph * graph, int threads) {
    size_t arena = 0;
    ggml_gallocr_reserve_n_size(allocator, graph, nullptr, nullptr, &arena);
    const auto plan = ggml_graph_plan(graph, threads, nullptr);
    return uint64_t(arena) + plan.work_size;
}
inline void price_persistent(ggml_backend_t backend, ggml_context * metadata,
                             ggml_context * weights, ggml_context * state,
                             gguf_context * file, MemoryMeasure & out) {
    const auto buft = ggml_backend_get_default_buffer_type(backend);
    out.weights = ggml_backend_alloc_ctx_tensors_from_buft_size(weights, buft);
    out.state = ggml_backend_alloc_ctx_tensors_from_buft_size(state, buft);
    out.metadata = ggml_get_mem_size(metadata) + ggml_get_mem_size(weights) +
                   ggml_get_mem_size(state) + gguf_get_meta_size(file);
    for (auto * t = ggml_get_first_tensor(metadata); t; t = ggml_get_next_tensor(metadata, t)) {
        const auto n = uint64_t(ggml_nelements(t));
        out.load_staging = std::max(out.load_staging, n * (t->type == GGML_TYPE_F16 ? 6 : 4));
    }
    mark_external(weights); mark_external(state);
}

struct PocketGraph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t allocator = nullptr;
    std::vector<ggml_tensor *> updates;
    explicit PocketGraph(ggml_backend_t backend) {
        constexpr int nodes = 32768;
        ctx = ggml_init({metadata_bytes(), nullptr, true});
        if (!ctx) throw std::runtime_error("pocket: graph context allocation failed");
        graph = ggml_new_graph_custom(ctx, nodes, false);
        allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!allocator) { ggml_free(ctx); ctx = nullptr; throw std::runtime_error("pocket: graph allocator failed"); }
    }
    ~PocketGraph() { if (allocator) ggml_gallocr_free(allocator); if (ctx) ggml_free(ctx); }
    PocketGraph(const PocketGraph &) = delete;
    PocketGraph & operator=(const PocketGraph &) = delete;
    static size_t metadata_bytes() {
        return 32768 * ggml_tensor_overhead() + ggml_graph_overhead_custom(32768, false);
    }
    uint64_t prepare(ggml_tensor * output, bool size_only = false, int threads = 1) {
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
        // The old convolution / transformer histories must remain intact until
        // every consumer has run. Delay cache writes until after the output.
        for (auto * update : updates) ggml_build_forward_expand(graph, update);
        if (size_only) return price_cpu_graph(allocator, graph, threads);
        if (!ggml_gallocr_alloc_graph(allocator, graph)) throw std::runtime_error("pocket: graph allocation failed");
        return 0;
    }
    std::vector<float> compute(ggml_backend_t backend, ggml_tensor * output) {
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("pocket: graph execution failed");
        std::vector<float> values(static_cast<size_t>(ggml_nelements(output)));
        ggml_backend_tensor_get(output, values.data(), 0, values.size() * sizeof(float));
        require_finite(values, "graph output");
        return values;
    }
};

inline ggml_tensor * pocket_mm(ggml_context * c, ggml_tensor * w, ggml_tensor * x) {
    auto * result = ggml_mul_mat(c, w, x);
    ggml_mul_mat_set_prec(result, GGML_PREC_F32);
    return result;
}

inline ggml_tensor * pocket_gelu(ggml_context * c, ggml_tensor * x) {
    auto * cubic = ggml_mul(c, ggml_sqr(c, x), x);
    auto * inner = ggml_scale(c, ggml_add(c, x, ggml_scale(c, cubic, 0.044715f)), 0.7978845608028654f);
    return ggml_mul(c, ggml_scale(c, x, 0.5f), ggml_scale_bias(c, ggml_tanh(c, inner), 1, 1));
}

inline std::vector<float> pocket_read(ggml_tensor * t) {
    std::vector<float> result(static_cast<size_t>(ggml_nelements(t)));
    ggml_backend_tensor_get(t, result.data(), 0, result.size() * sizeof(float));
    return result;
}
} // namespace tts_cpp::pocket::detail
