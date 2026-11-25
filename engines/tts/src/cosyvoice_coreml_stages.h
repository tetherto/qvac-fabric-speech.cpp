#pragma once

// The flow front-end, the HiFT F0 predictor and the HiFT decoder of CosyVoice3
// on the shared Core ML runtime (see cosyvoice_coreml.h). Compiled into the
// tts-cpp library only.

#include "cosyvoice_coreml.h"
#include "tts-cpp/coreml_options.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

// ---- the front-end, F0 and HiFT stages on the shared Core ML runtime -------

enum class cosyvoice_stage : int { frontend = 0, f0 = 1, hift = 2 };
constexpr int COSYVOICE_STAGE_COUNT = 3;
constexpr const char * COSYVOICE_GGML_LABEL = "ggml";
constexpr const char * COSYVOICE_MIXED_LABEL = "mixed";

// "frontend", "f0", "hift": the status stage names.
const char * cosyvoice_stage_name(cosyvoice_stage s);
// The prebuilt sidecar suffix next to the stage's GGUF: "-frontend" (flow),
// "-f0" (HiFT), "" (HiFT).
const char * cosyvoice_stage_suffix(cosyvoice_stage s);

// One stage's route, decided from options, OS, build, template match and the
// presence of a prebuilt sidecar, never from the cache, so it is the same
// before and after the first generation. For the generated route it carries
// the Core ML memory the stage charges, from its template.
struct cosyvoice_stage_route {
    std::string route = "none";  // prebuilt | generate | none
    std::string state;           // when none: unsupported | disabled | absent | no_template
    std::string reason;
    std::string compute_units;   // resolved
    bool validated = false;      // generate: the template's memory is validated for compute_units
    // generate: the parts before the margin, from the template, and its margin.
    uint64_t resident_bytes = 0;      // weights, 2 x I/O, state, workspace
    uint64_t workspace_bytes = 0;     // included in resident_bytes
    uint64_t prepare_peak_bytes = 0;  // generation and compile transient
    uint64_t load_peak_bytes = 0;     // load and warm-up transient for compute_units
    double margin = 1.25;
    // generate: the free memory a cold Neural Engine load waits for (runtime
    // admission; fit does not charge it). 0 when compute_units has no ANE.
    uint64_t ane_compile_reserve_bytes = 0;
};

// The three stages' routes and the Core ML envelope fit charges for the
// generated ones: each generated stage's resident part plus one transient per
// library image (the largest stage's; the runtime runs one Core ML job at a
// time), each with its template's margin.
struct cosyvoice_coreml_plan {
    cosyvoice_stage_route stages[COSYVOICE_STAGE_COUNT];
    bool strict = false;
    bool any_prebuilt = false;
    bool any_generate = false;
    bool validated = false;  // every generated stage is validated
    uint64_t envelope_bytes = 0;
};

// Throws std::invalid_argument for invalid explicit options.
cosyvoice_coreml_plan cosyvoice_plan_coreml(const std::string & flow_gguf, const std::string & hift_gguf,
                                            const tts_cpp::CoremlOptions & options, bool ignore_prebuilt = false);

// Whether the legacy LM and flow-DiT sidecars may be opened under `options`
// (the COSYVOICE_COREML_* environment included): false when Core ML is off.
bool cosyvoice_coreml_enabled(const tts_cpp::CoremlOptions & options);

// What one stage unit runs on, acquired once at its start and held for the
// whole unit.
struct cosyvoice_stage_call {
    std::shared_ptr<cosyvoice_coreml> model;  // null: the unit runs on ggml
    bool pending = false;  // on ggml only because the sidecar was not ready yet
};

// The front-end, F0 and HiFT stages of one Engine on the shared Core ML
// runtime: the resolved options, the routes, the slots, the construction
// group and status.
class cosyvoice_coreml_stages {
public:
    // Resolves `options` (with the COSYVOICE_* environment) and plans the
    // three routes. Creates a slot per stage whose route is not none and
    // issues the construction group in pipeline order (front-end, F0, HiFT),
    // waiting for it with wait = construct. `memory_degrade` resolves
    // use = off (fit's memory-degrade). Returns null in builds without
    // TTS_CPP_COREML. Throws std::invalid_argument for invalid options and
    // std::runtime_error when use = require cannot be met.
    static std::shared_ptr<cosyvoice_coreml_stages> create(const std::string & flow_gguf, const std::string & hift_gguf,
                                                           const tts_cpp::CoremlOptions & options,
                                                           bool ignore_prebuilt, bool memory_degrade);
    ~cosyvoice_coreml_stages();

    // Once per stage unit. `frames` sizes the bounded wait for a warm load
    // (never longer than the unit would take on ggml). Throws
    // std::runtime_error when use = require and the stage cannot serve the
    // unit (failed, retired, timed out).
    cosyvoice_stage_call acquire(cosyvoice_stage s, int frames, const std::atomic<bool> * cancel);
    bool strict() const;
    const cosyvoice_coreml_plan & plan() const;
    // Whether the stage has a slot (route != none).
    bool active(cosyvoice_stage s) const;
    // Status JSON (schema 1: engine, cache_dir, generation, stages).
    std::string status_json() const;
    // The construction group's result JSON.
    std::string construction_json() const;
    // Issues another group over every stage with a slot (scope none: an empty
    // group) and, with `wait`, blocks until it finishes or `cancels` differs
    // from `since`. Returns its result JSON.
    std::string prewarm_json(tts_cpp::CoremlOptions::Prewarm scope, bool wait,
                             const std::atomic<uint64_t> * cancels, uint64_t since);
    // Blocks until every stage with a slot is ready or one is terminal,
    // `timeout_ms` passes (< 0: no limit) or `cancels` differs from `since`.
    // True when all are ready; false at once while one is idle (not requested
    // yet) or deferred(memory).
    bool wait(int64_t timeout_ms, const std::atomic<uint64_t> * cancels, uint64_t since) const;

    struct impl;

private:
    explicit cosyvoice_coreml_stages(std::unique_ptr<impl> p);
    std::unique_ptr<impl> p_;
};

// Label merge across stages or calls: equal labels stay, anything else is
// "mixed" (an empty accumulator takes the next label).
std::string cosyvoice_merge_label(const std::string & acc, const std::string & next);
