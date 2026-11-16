#pragma once

// Probes for the overlapped memory schedules that validate a generated Core
// ML sidecar's memory constants: this process's footprint, the resident size
// of ANECompilerService, and that service's growth during this process's
// loads. Header-only. The probes read 0 off macOS. ane_compiler_growth() is
// portable: the tts unit test test-parler-dac-coreml-self-test checks it in
// every build, with no model.

#include <algorithm>
#include <cstdint>
#include <vector>

#if defined(__APPLE__)
#include <mach/mach.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#endif

namespace qvac_coreml_memory {

// This process's phys_footprint in bytes (0 when unavailable).
inline uint64_t phys_footprint() {
#if defined(__APPLE__)
    task_vm_info_data_t info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t) &info, &count) != KERN_SUCCESS) return 0;
    return info.phys_footprint;
#else
    return 0;
#endif
}

// The summed resident size of every ANECompilerService process in bytes (0
// when none runs).
inline uint64_t ane_compiler_rss() {
#if defined(__APPLE__)
    uint64_t total = 0;
    FILE * f = popen("ps -axo rss=,comm= 2>/dev/null", "r");
    if (!f) return 0;
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        if (strstr(line, "ANECompilerService")) total += (uint64_t) std::strtoull(line, nullptr, 10) * 1024;
    }
    pclose(f);
    return total;
#else
    return 0;
#endif
}

// One ANECompilerService sample, and whether this process was loading or
// warming a Core ML model when it was taken.
struct AneSample {
    uint64_t rss = 0;
    bool loading = false;
};

struct AneGrowth {
    uint64_t before = 0;      // the last size sampled before the first load (0 when none precedes it)
    bool has_before = false;  // whether any sample precedes the first load
    uint64_t peak = 0;        // the largest size sampled while loading
    uint64_t growth = 0;      // peak - before
};

// ANECompilerService is a long-lived system process: its absolute size
// includes whatever earlier processes on the host compiled, so a shared CI
// runner reads hundreds of MiB more than a clean host. What this process's
// cold loads add is its growth: the largest size sampled while loading, less
// its last size sampled before the first load. On a clean host the growth
// equals the absolute peak. A service that already holds memory reuses it for
// the next compile, so on such a dirty baseline the growth understates what
// the same compile needs on a clean service; a reserve derived from growth
// must come from clean starts (or from the absolute peak). The size is summed
// over every instance on the host, so another process that compiles for the
// Neural Engine during this process's load counts as growth too.
inline AneGrowth ane_compiler_growth(const std::vector<AneSample> & samples) {
    AneGrowth g;
    bool loaded = false;
    for (const AneSample & s : samples) {
        if (s.loading) {
            loaded = true;
            g.peak = std::max(g.peak, s.rss);
        } else if (!loaded) {
            g.before = s.rss;
            g.has_before = true;
        }
    }
    g.growth = g.peak > g.before ? g.peak - g.before : 0;
    return g;
}

// Up to this size before the first load the service counts as clean: an
// idle instance holds about 1 MiB, one that has compiled a model for anyone
// hundreds of MiB.
constexpr uint64_t ANE_COMPILER_CLEAN_BYTES = 32ull << 20;

// The service's state before the first load: clean (its growth stands for a
// cold compile on a clean host), dirty (it already held memory, so its growth
// understates one) or unknown (no sample before the load, so the growth is
// its whole size while loading). Reports print it next to the size before the
// load, so a dirty or missing baseline is visible.
enum class AneBaseline { clean, dirty, unknown };

inline AneBaseline ane_compiler_baseline(const AneGrowth & g) {
    if (!g.has_before) return AneBaseline::unknown;
    return g.before <= ANE_COMPILER_CLEAN_BYTES ? AneBaseline::clean : AneBaseline::dirty;
}

inline const char * ane_baseline_name(AneBaseline b) {
    switch (b) {
        case AneBaseline::clean: return "clean";
        case AneBaseline::dirty: return "dirty";
        case AneBaseline::unknown: break;
    }
    return "unknown";
}

// Whether a run's growth was measured from a clean service.
inline bool ane_compiler_clean_start(const AneGrowth & g) { return ane_compiler_baseline(g) == AneBaseline::clean; }

}  // namespace qvac_coreml_memory
