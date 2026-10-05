#pragma once

#include "acestep/backend_registry.h"
#include "acestep/fit_util.h"
#include "acestep/stage_placement.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace tts_cpp::acestep {

inline constexpr size_t FIT_HOST_POOL = 0;

struct FitPool {
    ggml_backend_dev_t device             = nullptr;
    bool               shares_host_memory = false;
    uint64_t           free_bytes         = 0;
    uint64_t           total_bytes        = 0;
};

struct PoolCharge {
    std::vector<uint64_t> bytes;

    explicit PoolCharge(size_t n_pools) : bytes(n_pools, 0) {}

    PoolCharge & add_to(size_t pool, uint64_t b) {
        bytes[pool] = fitutil::sat_add(bytes[pool], b);
        return *this;
    }

    PoolCharge & add_host(uint64_t b) { return add_to(FIT_HOST_POOL, b); }

    PoolCharge & add(const PoolCharge & o) {
        for (size_t i = 0; i < bytes.size(); ++i) {
            bytes[i] = fitutil::sat_add(bytes[i], o.bytes[i]);
        }
        return *this;
    }

    PoolCharge & max_with(const PoolCharge & o) {
        for (size_t i = 0; i < bytes.size(); ++i) {
            bytes[i] = std::max(bytes[i], o.bytes[i]);
        }
        return *this;
    }
};

struct FitBudgetCheck {
    uint64_t free_bytes = 0;
    uint64_t required   = 0;
};

inline bool fit_device_shares_host_memory(enum ggml_backend_dev_type type, const char * reg_name,
                                          uint64_t total_bytes) {
    const bool reports_no_memory_of_its_own = total_bytes == 0;
    return type == GGML_BACKEND_DEVICE_TYPE_CPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU ||
           backend_name_is_metal(reg_name) || reports_no_memory_of_its_own;
}

inline FitPool fit_pool_for_device(ggml_backend_dev_t dev) {
    size_t free_b = 0, total_b = 0;
    ggml_backend_dev_memory(dev, &free_b, &total_b);
    FitPool pool;
    pool.device             = dev;
    pool.free_bytes         = free_b;
    pool.total_bytes        = total_b;
    pool.shares_host_memory =
        fit_device_shares_host_memory(ggml_backend_dev_type(dev), backend_dev_reg_name(dev), total_b);
    return pool;
}

inline size_t fit_pool_find(const std::vector<FitPool> & pools, ggml_backend_dev_t dev) {
    for (size_t i = 0; i < pools.size(); ++i) {
        if (pools[i].device == dev) return i;
    }
    return pools.size();
}

inline size_t fit_pool_of(const std::vector<FitPool> & pools, ggml_backend_dev_t dev) {
    const size_t i = fit_pool_find(pools, dev);
    return i < pools.size() ? i : FIT_HOST_POOL;
}

inline size_t fit_pool_register(std::vector<FitPool> & pools, const FitPool & pool) {
    const size_t i = fit_pool_find(pools, pool.device);
    if (i == pools.size()) pools.push_back(pool);
    return i;
}

inline PoolCharge fit_charge(const std::vector<FitPool> & pools, ggml_backend_dev_t dev, uint64_t bytes) {
    PoolCharge c(pools.size());
    c.add_to(fit_pool_of(pools, dev), bytes);
    return c;
}

inline size_t fit_shared_budget_pool(const std::vector<FitPool> & pools, size_t primary) {
    const FitPool & p                  = pools[primary];
    const bool      primary_has_budget = primary != FIT_HOST_POOL && p.shares_host_memory && p.free_bytes > 0;
    return primary_has_budget ? primary : FIT_HOST_POOL;
}

inline uint64_t fit_shared_peak(const std::vector<FitPool> & pools, const PoolCharge & peak) {
    uint64_t total = 0;
    for (size_t i = 0; i < pools.size(); ++i) {
        if (pools[i].shares_host_memory) total = fitutil::sat_add(total, peak.bytes[i]);
    }
    return total;
}

inline void append_discrete_pool_checks(std::vector<FitBudgetCheck> & checks, const std::vector<FitPool> & pools,
                                        const PoolCharge & peak, uint64_t margin) {
    for (size_t i = 0; i < pools.size(); ++i) {
        if (!pools[i].shares_host_memory) {
            checks.push_back({ pools[i].free_bytes, fitutil::sat_add(peak.bytes[i], margin) });
        }
    }
}

inline std::vector<FitBudgetCheck> fit_budget_checks(const std::vector<FitPool> & pools, const PoolCharge & peak,
                                                     size_t primary, uint64_t margin) {
    std::vector<FitBudgetCheck> checks;
    checks.push_back({ pools[fit_shared_budget_pool(pools, primary)].free_bytes,
                       fitutil::sat_add(fit_shared_peak(pools, peak), margin) });
    append_discrete_pool_checks(checks, pools, peak, margin);
    return checks;
}

inline bool fit_check_holds(const FitBudgetCheck & check) {
    return check.required <= check.free_bytes;
}

inline bool fit_checks_hold(const std::vector<FitBudgetCheck> & checks) {
    return std::all_of(checks.begin(), checks.end(), fit_check_holds);
}

inline uint64_t fit_checks_headroom(const std::vector<FitBudgetCheck> & checks) {
    uint64_t headroom = std::numeric_limits<uint64_t>::max();
    for (const FitBudgetCheck & c : checks) {
        headroom = std::min(headroom, fit_check_holds(c) ? c.free_bytes - c.required : 0);
    }
    return headroom;
}

}  // namespace tts_cpp::acestep
