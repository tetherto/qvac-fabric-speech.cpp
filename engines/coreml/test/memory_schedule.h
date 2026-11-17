#pragma once

// Phase attribution for the overlapped memory schedules that validate a
// generated Core ML sidecar's memory constants (each engine's test drives the
// schedule; this header holds what every one of them measures the same way).
// Header-only, portable: the tts unit tests check it in every build.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace qvac_coreml_memory {

// The memory schedule (--memory-schedule) samples this process's footprint
// while the generated sidecar is prepared, loaded and warmed in the
// background, and attributes each sample to the stage slot's state when it
// was taken. Each phase is held to the template constant that charges it:
//   prepare     digesting, generating and compiling: the peak over the
//               footprint just before generation began, against
//               generation_peak_bytes + 8 MiB + compile_peak_bytes;
//   load+warm   loading and the warm-up prediction: the peak over the
//               footprint just before the load began, against
//               load_peak_bytes for the resolved compute units;
//   steady      ready, Core ML predictions running: the peak over the
//               footprint just before the load began, against the resident
//               charge (weights, 2 x I/O, workspace_bytes and the engine's
//               adapter copies, such as Parler's host quantizer). Generation memory the allocator still holds when the
//               load begins is the prepare transient's, which the envelope
//               charges next to the resident; the whole run checks the sum;
//   whole run   the peak over the baseline, against the envelope fit charges;
//   compiler    ANECompilerService's growth while loading and warming (its
//               peak less its size just before the first load; see
//               qvac_coreml_memory::ane_compiler_growth), against
//               ane_compile_reserve_bytes (when the units include the
//               Neural Engine). The report prints the size before the load
//               and labels that baseline clean, dirty (a service that already
//               held memory grows less than a clean one would) or unknown (no
//               sample before the load).
// generate, compile, load and warm are also reported on their own, each over
// the footprint just before that state began.
enum class mem_phase { ggml, generate, compile, load, warm, ready };

inline mem_phase mem_phase_of(const std::string & state) {
    if (state == "digesting" || state == "generating") return mem_phase::generate;
    if (state == "compiling") return mem_phase::compile;
    if (state == "loading") return mem_phase::load;
    if (state == "warming") return mem_phase::warm;
    if (state == "ready") return mem_phase::ready;
    return mem_phase::ggml;  // idle, queued, deferred(memory): nothing of the sidecar's yet
}

struct mem_sample {
    uint64_t footprint = 0;
    mem_phase phase = mem_phase::ggml;
};

// Increments in bytes; the flags say which states were sampled.
struct mem_breakdown {
    uint64_t prepare = 0, generate = 0, compile = 0, load = 0, warm = 0, load_warm = 0, steady = 0, total = 0;
    uint64_t steady_over_base = 0;  // reported only
    bool generated = false, compiled = false, loaded = false, warmed = false, ready = false;
};

inline mem_breakdown mem_breakdown_of(const std::vector<mem_sample> & samples, uint64_t base) {
    // 1: prepare (generate, compile); 2: load and warm-up; 0: the rest.
    const auto group = [](mem_phase p) {
        if (p == mem_phase::generate || p == mem_phase::compile) return 1;
        if (p == mem_phase::load || p == mem_phase::warm) return 2;
        return 0;
    };
    const auto over = [](uint64_t v, uint64_t from) { return v > from ? v - from : (uint64_t) 0; };
    mem_breakdown b;
    uint64_t prev = base, phase_start = base, group_start = base;
    // The footprint just before the first load: the steady state is measured
    // from there, so generation memory the allocator still holds when the
    // load begins counts as the prepare's, not as resident.
    std::optional<uint64_t> load_start;
    mem_phase last = mem_phase::ggml;
    for (const mem_sample & s : samples) {
        if (s.phase != last) phase_start = prev;
        if (group(s.phase) != group(last)) group_start = prev;
        if (group(s.phase) == 2 && !load_start) load_start = group_start;
        const uint64_t inc = over(s.footprint, phase_start);
        switch (s.phase) {
            case mem_phase::generate: b.generated = true; b.generate = std::max(b.generate, inc); break;
            case mem_phase::compile: b.compiled = true; b.compile = std::max(b.compile, inc); break;
            case mem_phase::load: b.loaded = true; b.load = std::max(b.load, inc); break;
            case mem_phase::warm: b.warmed = true; b.warm = std::max(b.warm, inc); break;
            case mem_phase::ready:
                b.ready = true;
                b.steady = std::max(b.steady, over(s.footprint, load_start.value_or(base)));
                b.steady_over_base = std::max(b.steady_over_base, over(s.footprint, base));
                break;
            case mem_phase::ggml: break;
        }
        if (group(s.phase) == 1) b.prepare = std::max(b.prepare, over(s.footprint, group_start));
        if (group(s.phase) == 2) b.load_warm = std::max(b.load_warm, over(s.footprint, group_start));
        b.total = std::max(b.total, over(s.footprint, base));
        prev = s.footprint;
        last = s.phase;
    }
    return b;
}

// The template's charges for one run, before the margin, and the envelope.
struct mem_charges {
    uint64_t prepare = 0, load = 0, resident = 0, ane_reserve = 0, envelope = 0;
};

// One line per charge the run exceeds, or per state it never sampled.
inline std::vector<std::string> mem_failures(const mem_breakdown & b, uint64_t ane_compiler_growth, const mem_charges & c) {
    std::vector<std::string> out;
    const std::pair<bool, const char *> states[] = {{b.generated, "generating"}, {b.compiled, "compiling"},
                                                    {b.loaded, "loading"},       {b.warmed, "warming"},
                                                    {b.ready, "ready"}};
    for (const auto & [seen, state] : states) {
        if (!seen) out.push_back(std::string("the schedule never sampled the slot ") + state);
    }
    const auto over = [&out](const char * what, uint64_t got, uint64_t charged) {
        if (got <= charged) return;
        char line[256];
        std::snprintf(line, sizeof(line), "%s: %llu bytes over the charged %llu", what, (unsigned long long) got,
                      (unsigned long long) charged);
        out.push_back(line);
    };
    over("prepare", b.prepare, c.prepare);
    over("load and warm-up", b.load_warm, c.load);
    over("steady state (resident)", b.steady, c.resident);
    over("whole run (envelope)", b.total, c.envelope);
    if (c.ane_reserve > 0) over("ANECompilerService growth (reserve)", ane_compiler_growth, c.ane_reserve);
    return out;
}


}  // namespace qvac_coreml_memory
