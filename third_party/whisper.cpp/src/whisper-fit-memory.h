#pragma once

#include "ggml-backend.h"
#include "ggml-hexagon.h"

#include <cstdint>
#include <limits>

// A device may expose shared system RAM while having no device-specific
// capacity query (e.g. FastRPC). Keep its tensor accounting on that device,
// but compare the combined host/device requirement with the host RAM pool.
struct whisper_fit_memory_info {
    bool shares_host = false;
    size_t free_bytes = 0;
    size_t total_bytes = 0;
};

inline whisper_fit_memory_info whisper_fit_query_memory(ggml_backend_dev_t dev) {
    ggml_backend_dev_props props = {};
    ggml_backend_dev_get_props(dev, &props);
    whisper_fit_memory_info info;
    info.shares_host = props.type == GGML_BACKEND_DEVICE_TYPE_CPU ||
                       props.type == GGML_BACKEND_DEVICE_TYPE_IGPU || props.memory_unified;
    info.free_bytes = props.memory_free;
    info.total_bytes = props.memory_total;
    if (info.shares_host && info.total_bytes == 0) {
        ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu != nullptr && cpu != dev) {
            ggml_backend_dev_memory(cpu, &info.free_bytes, &info.total_bytes);
        }
    }
    return info;
}

// Price only prospective registry metadata growth: the OS's available-RAM
// snapshot already reflects retained metadata and shared queues. Tensor
// buffers are priced separately by the graph/context allocators. Opaque
// future driver/DSP heaps are still covered only by the margin.
// Call once for the primary backend; the Hexagon query already covers all
// sessions, including sessions used by a VAD context.
inline bool whisper_fit_query_backend_overhead(ggml_backend_t backend, uint64_t max_graph_nodes, uint64_t & bytes) {
    bytes = 0;
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    auto query = reg ? reinterpret_cast<ggml_backend_hexagon_get_runtime_info_v1_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_hexagon_get_runtime_info_v1")) : nullptr;
    if (query == nullptr) {
        return true;
    }
    ggml_backend_hexagon_runtime_info_v1 retained = {}, future = {};
    if (!query(backend, 0, &retained, sizeof(retained)) ||
        !query(backend, max_graph_nodes, &future, sizeof(future)) ||
        retained.version != GGML_HEXAGON_RUNTIME_INFO_VERSION ||
        future.version != GGML_HEXAGON_RUNTIME_INFO_VERSION ||
        retained.session_count == 0 || retained.session_count != future.session_count ||
        future.host_bytes_bound < retained.host_bytes_bound || future.shared_bytes < retained.shared_bytes) {
        return false;
    }
    const uint64_t host = future.host_bytes_bound - retained.host_bytes_bound;
    const uint64_t shared = future.shared_bytes - retained.shared_bytes;
    bytes = shared > std::numeric_limits<uint64_t>::max() - host ?
        std::numeric_limits<uint64_t>::max() : host + shared;
    return true;
}
