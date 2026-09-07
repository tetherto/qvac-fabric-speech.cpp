#pragma once

// Backend-introspection helpers that work uniformly under both
// GGML_BACKEND_DL=ON and GGML_BACKEND_DL=OFF. The legacy
// ggml_backend_is_cpu / ggml_backend_is_metal entry points live in
// the per-backend shared libraries (libggml-cpu.* / libggml-metal.*),
// so they are unlinkable from libqvac-parakeet under the dynamic-loader
// build mode embedded host applications typically ship with. Routing
// through the registry (ggml_backend_get_device + ggml_backend_dev_*)
// reaches the same answer in both modes.

#include "ggml-backend.h"

#include <cstring>

namespace parakeet {

// Tier ranking for GPU selection (lower = preferred). Pure function so unit
// tests can exercise the ordering against synthesised device topologies
// without a live ggml-backend registry. The classification mirrors the
// bucket-and-try_init walk in parakeet_ctc.cpp::init_gpu_backend so any
// drift between them fails the unit test.
//
// The classification does NOT reason about Adreno-6xx-broken skips or the
// PARAKEET_ALLOW_ADRENO_6XX override — those live in init_gpu_backend
// because they are enumeration-side decisions (skip the device before it
// enters a bucket at all), not tier-ranking decisions.
enum class GpuTier {
    AdrenoOpenCL700Plus  = 0, // Snapdragon 8 Gen 2/3/4 (validated + faster than Vulkan).
    CudaDiscrete         = 1, // NVIDIA dGPU via CUDA; preferred over Vulkan on same card.
    CudaIntegrated       = 2, // Tegra/Jetson (CUDA reported as IGPU); still CUDA-first.
    OtherDiscrete        = 3, // Discrete Vulkan / Metal / etc.
    OtherIntegrated      = 4, // UMA Vulkan (iGPU / Mali / Apple integrated).
    OpenCLOther          = 5, // Non-Adreno OpenCL (desktop OpenCL, unrecognised Adreno).
    NotSelectable        = 6,
};

inline GpuTier gpu_tier_for(const char *                     reg_name,
                            enum ggml_backend_dev_type       dev_type,
                            int                              adreno_version) {
    if (dev_type != GGML_BACKEND_DEVICE_TYPE_GPU &&
        dev_type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
        return GpuTier::NotSelectable;
    }
    const bool integrated = (dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU);
    const bool is_opencl  = reg_name && std::strcmp(reg_name, "OpenCL") == 0;
    const bool is_cuda    = reg_name && std::strcmp(reg_name, "CUDA")   == 0;
    if (is_opencl) {
        if (adreno_version >= 700) return GpuTier::AdrenoOpenCL700Plus;
        return GpuTier::OpenCLOther;
    }
    if (is_cuda) {
        return integrated ? GpuTier::CudaIntegrated : GpuTier::CudaDiscrete;
    }
    return integrated ? GpuTier::OtherIntegrated : GpuTier::OtherDiscrete;
}


inline const char * backend_reg_name(ggml_backend_t b) {
    if (!b) return "";
    ggml_backend_dev_t dev = ggml_backend_get_device(b);
    if (!dev) return "";
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (!reg) return "";
    const char * n = ggml_backend_reg_name(reg);
    return n ? n : "";
}

inline bool backend_is_cpu(ggml_backend_t b) {
    if (!b) return false;
    ggml_backend_dev_t dev = ggml_backend_get_device(b);
    return dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
}

// Which attention graph the encoder builds. Fused attention is compiled in per
// build (PARAKEET_FLASH_ATTN) but used only on the backends it was compared
// against the unfused graph on, CUDA, Metal and Vulkan, and only when the
// backend accepts the node; CPU and OpenCL keep the unfused graph whatever the
// binary carries.
struct AttnPath {
    bool flash_attn    = false;
    bool per_head_mask = false; // the fused kernel accepts one additive mask per head
};

inline bool backend_is_metal(ggml_backend_t b) {
    // Upstream ggml registered the Metal backend as "Metal" until mid-2026,
    // when the registry name changed to "MTL" (GGML_METAL_NAME). The pinned
    // qvac-ext-ggml@speech carries the new name; accept both so the
    // Metal-specific gates keep firing across pin bumps.
    const char * n = backend_reg_name(b);
    return std::strcmp(n, "Metal") == 0 || std::strcmp(n, "MTL") == 0;
}

inline bool backend_is_cuda(ggml_backend_t b) {
    return std::strcmp(backend_reg_name(b), "CUDA") == 0;
}

inline bool backend_is_vulkan(ggml_backend_t b) {
    return std::strcmp(backend_reg_name(b), "Vulkan") == 0;
}

inline bool flash_attn_allowed(bool compiled_in, ggml_backend_t b) {
    return compiled_in && b && (backend_is_cuda(b) || backend_is_metal(b) || backend_is_vulkan(b));
}

inline void backend_set_n_threads(ggml_backend_t b, int n_threads) {
    if (!b || n_threads <= 0) return;
    ggml_backend_dev_t dev = ggml_backend_get_device(b);
    if (!dev) return;
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (!reg) return;
    auto fn = (ggml_backend_set_n_threads_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (fn) fn(b, n_threads);
}

}
