#pragma once

#include "backend_selection.h"
#include "backend_util.h"
#include <stdexcept>

namespace tts_cpp::pocket::detail {
inline ggml_backend_t init_backend(bool use_gpu, int threads) {
    auto * backend = use_gpu
        ? ::tts_cpp::detail::init_gpu_backend(1, false, "pocket", 0, false, nullptr,
                                            ::tts_cpp::detail::GpuBackendRequirement::Metal)
        : ::tts_cpp::detail::init_cpu_backend();
    if (!backend) throw std::runtime_error(use_gpu ? "pocket: Metal backend unavailable" : "pocket: CPU backend unavailable");
    ::tts_cpp::detail::backend_set_n_threads(backend, threads);
    return backend;
}
}
