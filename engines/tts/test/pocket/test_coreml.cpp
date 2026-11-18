// Pocket Core ML sidecars: the FlowLM step and the Mimi decoder.
//
//   test-pocket-coreml                  policy, naming, RoPE tables, sidecar
//                                       detection, fit refusal, option checks
//   test-pocket-coreml --shape-fixture  a tiny real step reproduces its float32
//                                       reference after the prefix is written
//                                       into its state and after a reset clears
//                                       a NaN row, and a random full-size Mimi
//                                       decoder streams its reference;
//                                       every declared shape, the schema, the
//                                       checkpoint hash and state presence are
//                                       checked before predicting
//   test-pocket-coreml --engine-fixture the real sidecars against ggml FlowLM
//                                       and Mimi, the ggml replay of frames a
//                                       retired step generated, and the
//                                       Engine's routing, labels and strict mode
//   test-pocket-coreml --generated BUNDLE --cache-dir DIR [--memory-schedule DIR]
//                                       sidecars generated from a bundle's
//                                       GGUFs (no prebuilt sidecar next to
//                                       them): routes and fit, a call that
//                                       runs on ggml while the sidecars are
//                                       pending (matching Core ML off),
//                                       readiness at construction, parity of
//                                       the generated step and decoder against
//                                       ggml, adoption by a second Engine, fit
//                                       unchanged by a populated cache, strict
//                                       mode; with --memory-schedule, the
//                                       overlapped memory schedule against the
//                                       templates' constants
//
// The shape fixture reads POCKET_COREML_TEST_SHAPES (the directory
// test/prepare_pocket_coreml_fixture.py writes); the engine fixture reads
// POCKET_COREML_TEST_BUNDLE (a Pocket bundle directory with the exported
// flow-lm-step.mlmodelc and mimi-decoder.mlmodelc next to their GGUFs). Each reports a skip (77)
// without its inputs or without Core ML.

#include "pocket/coreml.h"
#include "pocket/flow_lm.h"
#include "pocket/frontend.h"
#include "pocket/generation.h"
#include "pocket/mimi.h"
#include "coreml_sidecar_path.h"
#include "json.hpp"
#include "npy.h"
#include "../test_env_portable.h"
#include "tts-cpp/pocket/engine.h"
#include "tts-cpp/pocket/fit.h"

#include "memory_schedule.h"
#ifdef TTS_CPP_USE_COREML
#include "moss/coreml/runner.h"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>

#if defined(__APPLE__)
#include <sys/clonefile.h>
#include <sys/sysctl.h>
#endif
#ifndef _WIN32
#include <sys/wait.h>
#endif
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace pd = tts_cpp::pocket::detail;
namespace fs = std::filesystem;
using tts_cpp::pocket::Engine;
using tts_cpp::pocket::EngineOptions;

namespace {

constexpr int SKIP = 77;

void require(bool ok, const std::string & what) {
    if (!ok) throw std::runtime_error(what);
}

template <class F> bool throws(F fn) {
    try { fn(); } catch (const std::exception &) { return true; }
    return false;
}

struct env_guard {
    std::string name;
    bool present;
    std::string value;
    explicit env_guard(const char * n) : name(n), present(std::getenv(n) != nullptr),
                                          value(present ? std::getenv(n) : "") {}
    ~env_guard() {
        if (present) setenv(name.c_str(), value.c_str(), 1);
        else unsetenv(name.c_str());
    }
};

struct temp_dir {
    fs::path path;
    explicit temp_dir(const std::string & tag)
        : path(fs::path(test_tmpdir()) / ("pocket-coreml-" + tag + "-" + test_process_tag())) {
        std::error_code error;
        fs::remove_all(path, error);
        fs::create_directories(path);
    }
    ~temp_dir() {
        std::error_code error;
        fs::remove_all(path, error);
    }
};

double max_error(const std::vector<float> & got, const std::vector<float> & want) {
    require(got.size() == want.size(), "size mismatch");
    double error = 0, scale = 1;
    for (size_t i = 0; i < got.size(); ++i) {
        require(std::isfinite(got[i]), "non-finite output");
        error = std::max(error, std::abs(double(got[i]) - want[i]));
        scale = std::max(scale, std::abs(double(want[i])));
    }
    return error / scale;
}

double snr_db(const std::vector<float> & got, const std::vector<float> & want) {
    require(got.size() == want.size(), "size mismatch");
    double signal = 0, noise = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        require(std::isfinite(got[i]), "non-finite output");
        signal += double(want[i]) * want[i];
        noise += (double(got[i]) - want[i]) * (double(got[i]) - want[i]);
    }
    return 10 * std::log10(signal / std::max(noise, 1e-30));
}

double cosine(const std::vector<float> & a, const std::vector<float> & b) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) { dot += double(a[i]) * b[i]; na += double(a[i]) * a[i]; nb += double(b[i]) * b[i]; }
    return dot / std::sqrt(na * nb);
}

void check_policy() {
    env_guard units(pd::COREML_UNITS_ENV), strict(pd::COREML_STRICT_ENV), disabled(pd::COREML_DISABLE_ENV);
    unsetenv(pd::COREML_UNITS_ENV);
    unsetenv(pd::COREML_STRICT_ENV);
    unsetenv(pd::COREML_DISABLE_ENV);
    auto policy = pd::resolve_coreml_policy();
    require(policy.compute_units.empty() && !policy.strict && !policy.disabled, "default policy");
    require(pd::coreml_units(policy, pd::COREML_STEP_DEFAULT_UNITS) == "cpu_and_gpu" &&
            pd::coreml_units(policy, pd::COREML_DECODER_DEFAULT_UNITS) == "cpu_and_ane", "per-sidecar default units");

    setenv(pd::COREML_UNITS_ENV, "all", 1);
    setenv(pd::COREML_STRICT_ENV, "0", 1);  // presence means strict
    policy = pd::resolve_coreml_policy();
    require(policy.compute_units == "all" && policy.strict, "inherited policy");
    require(pd::coreml_units(policy, pd::COREML_DECODER_DEFAULT_UNITS) == "all", "set units apply to every sidecar");
    policy = pd::resolve_coreml_policy("cpu_only", false);
    require(policy.compute_units == "cpu_only" && !policy.strict, "explicit policy overrides the environment");

    setenv(pd::COREML_UNITS_ENV, "gpu_please", 1);
    require(pd::resolve_coreml_policy().compute_units.empty(), "invalid inherited units keep the defaults");
    require(throws([] { pd::resolve_coreml_policy("gpu_please"); }), "invalid explicit units throw");

    setenv(pd::COREML_DISABLE_ENV, "1", 1);
    require(pd::resolve_coreml_policy().disabled, "POCKET_COREML_DISABLE");
}

void check_naming() {
    require(pd::coreml_step_path("models/pocket/flow-lm.gguf") == "models/pocket/flow-lm-step.mlmodelc", "bundle name");
    require(pd::coreml_step_path("m/flow-lm-f16.gguf") == "m/flow-lm-step.mlmodelc", "quant tag dropped");
    require(pd::coreml_decoder_path("models/pocket/mimi.gguf") == "models/pocket/mimi-decoder.mlmodelc", "decoder name");
}

void check_rope() {
    std::vector<float> cos, sin;
    pd::coreml_rope(0, 8, 10000.0f, cos, sin);
    require(cos.size() == 4 && std::all_of(cos.begin(), cos.end(), [](float c) { return c == 1.0f; }) &&
            std::all_of(sin.begin(), sin.end(), [](float s) { return s == 0.0f; }), "position 0");
    pd::coreml_rope(37, 8, 10000.0f, cos, sin);
    for (int i = 0; i < 4; ++i) {
        const double theta = 37.0 * std::pow(10000.0, -2.0 * i / 8);
        require(std::abs(cos[size_t(i)] - std::cos(theta)) < 1e-6 && std::abs(sin[size_t(i)] - std::sin(theta)) < 1e-6,
                "adjacent-pair frequencies");
    }
}

// Detection is presence-only, so an empty sidecar directory next to a GGUF
// path that does not exist is enough; fit refuses before reading anything.
namespace qm = qvac_coreml_memory;

// One footprint sample of the Pocket memory schedule, with each stage's slot
// state when it was taken.
struct pocket_mem_sample {
    uint64_t footprint = 0;
    qm::mem_phase step = qm::mem_phase::ggml, mimi = qm::mem_phase::ggml;
};

// The schedule's attribution (qvac_coreml_memory::mem_breakdown_of) per
// stage and for both together. One Core ML job runs at a time, so at most one
// stage prepares, loads or warms at any sample:
//   step, mimi  that stage's prepare and load+warm, each over the footprint
//               just before that phase began (the other stage's work counts as
//               ggml for it);
//   both        ready only once both stages are, otherwise the active
//               stage's phase: its steady state is over the footprint just
//               before the first load of either stage (against both resident
//               parts) and its whole run over the baseline (the envelope).
struct pocket_mem {
    qm::mem_breakdown step, mimi, both;
};

pocket_mem pocket_mem_of(const std::vector<pocket_mem_sample> & samples, uint64_t base) {
    const auto active = [](qm::mem_phase p) { return p != qm::mem_phase::ggml && p != qm::mem_phase::ready; };
    std::vector<qm::mem_sample> step, mimi, both;
    for (const pocket_mem_sample & s : samples) {
        step.push_back({s.footprint, active(s.step) ? s.step : qm::mem_phase::ggml});
        mimi.push_back({s.footprint, active(s.mimi) ? s.mimi : qm::mem_phase::ggml});
        const bool ready = s.step == qm::mem_phase::ready && s.mimi == qm::mem_phase::ready;
        both.push_back({s.footprint, active(s.step)   ? s.step
                                     : active(s.mimi) ? s.mimi
                                     : ready          ? qm::mem_phase::ready
                                                      : qm::mem_phase::ggml});
    }
    return {qm::mem_breakdown_of(step, base), qm::mem_breakdown_of(mimi, base), qm::mem_breakdown_of(both, base)};
}

// A stage's template charges.
struct pocket_stage_charges {
    uint64_t prepare = 0, load = 0, ane_reserve = 0;
};

// One line per charge the run exceeds, or per state it never sampled: each
// stage's prepare and load+warm and (on the Neural Engine) ANECompilerService's
// absolute size while it loaded, 1.25 x measured; the steady state with both
// ready, 1.25 x measured, against both resident parts; the whole run, as
// measured, against the envelope (qvac_coreml_memory::mem_failures).
std::vector<std::string> pocket_mem_failures(const pocket_mem & m, const pocket_stage_charges & step,
                                             const pocket_stage_charges & mimi, uint64_t ane_compiler_load_peak,
                                             uint64_t resident, uint64_t envelope) {
    std::vector<std::string> out;
    const auto stage = [&](const char * name, qm::mem_breakdown b, const pocket_stage_charges & c, uint64_t ane) {
        b.ready = true;  // the steady state is both stages' (below)
        b.steady = b.total = 0;
        // The load and the warm-up are one charged phase; a sample of either
        // is a sample of it. Mimi's warm-up prediction on the Neural Engine
        // takes about a millisecond, shorter than the sampling period.
        b.loaded = b.warmed = b.loaded || b.warmed;
        for (const std::string & f : qm::mem_failures(b, ane, {c.prepare, c.load, 0, c.ane_reserve, 0}))
            out.push_back(std::string(name) + " " + f);
    };
    stage("step", m.step, step, 0);
    stage("mimi", m.mimi, mimi, ane_compiler_load_peak);
    qm::mem_breakdown b = m.both;
    b.generated = b.compiled = b.loaded = b.warmed = true;  // per stage (above)
    b.prepare = b.load_warm = 0;
    for (const std::string & f : qm::mem_failures(b, 0, {0, 0, resident, 0, envelope})) out.push_back("both " + f);
    return out;
}

bool contains_text(const std::vector<std::string> & lines, const std::string & what) {
    for (const std::string & l : lines)
        if (l.find(what) != std::string::npos) return true;
    return false;
}

// The schedule's arithmetic on a synthetic run (no device): per-stage
// attribution, the margin on every measured phase, the whole run as measured,
// and every missing state or probe reading failing the run.
void check_memory_schedule_attribution() {
    using P = qm::mem_phase;
    const std::vector<pocket_mem_sample> run = {
        {100, P::ggml, P::ggml},    {102, P::generate, P::ggml}, {130, P::compile, P::ggml},
        {110, P::ggml, P::generate}, {150, P::ggml, P::compile},  {140, P::load, P::ggml},
        {170, P::warm, P::ggml},     {165, P::ready, P::load},    {190, P::ready, P::warm},
        {185, P::ready, P::ready},   {180, P::ready, P::ready}};
    const pocket_mem m = pocket_mem_of(run, 100);
    require(m.step.prepare == 30 && m.step.load_warm == 20 && m.mimi.prepare == 20 && m.mimi.load_warm == 20,
            "memory schedule: each stage's prepare and load+warm over the footprint just before it began");
    require(m.both.steady == 35 && m.both.total == 90 && m.both.ready,
            "memory schedule: steady with both ready over the first load, whole run over the baseline");
    require(!m.step.ready && !m.mimi.ready, "memory schedule: one stage ready is not the steady state");
    // 1.25 x 30 = 37.5: a 38 charge holds it, 37 does not.
    const pocket_stage_charges step{38, 25, 0}, mimi{25, 25, 125};
    require(pocket_mem_failures(m, step, mimi, 100, 44, 90).empty(), "memory schedule: a run within every charge");
    require(contains_text(pocket_mem_failures(m, {37, 25, 0}, mimi, 100, 44, 90), "step prepare"),
            "memory schedule: the margin applies to the step's prepare");
    require(contains_text(pocket_mem_failures(m, step, {25, 24, 125}, 100, 44, 90), "mimi load and warm-up"),
            "memory schedule: the margin applies to Mimi's load and warm-up");
    require(contains_text(pocket_mem_failures(m, step, mimi, 101, 44, 90), "ANECompilerService"),
            "memory schedule: 1.25 x the compiler's absolute size while loading must fit the reserve");
    require(contains_text(pocket_mem_failures(m, step, mimi, 100, 43, 90), "steady state"),
            "memory schedule: the margin applies to the steady state");
    require(contains_text(pocket_mem_failures(m, step, mimi, 100, 44, 89), "whole run") &&
                !contains_text(pocket_mem_failures(m, step, mimi, 100, 44, 90), "whole run"),
            "memory schedule: the whole run is held to the envelope as measured");
    // A run that never sampled a state fails, and a digest is not a generation.
    std::vector<pocket_mem_sample> partial = run;
    partial[1].step = P::digest;
    require(contains_text(pocket_mem_failures(pocket_mem_of(partial, 100), step, mimi, 100, 44, 90),
                          "step the schedule never sampled the slot generating"),
            "memory schedule: a digest alone does not count as the generation");
    // The load+warm phase needs a sample of either state, not of both.
    partial = run;
    partial[8].mimi = P::load;
    require(pocket_mem_failures(pocket_mem_of(partial, 100), step, mimi, 100, 44, 90).empty(),
            "memory schedule: a load sampled without its (short) warm-up is the load+warm phase");
    partial[7].mimi = partial[8].mimi = P::compile;
    require(contains_text(pocket_mem_failures(pocket_mem_of(partial, 100), step, mimi, 100, 44, 90),
                          "mimi the schedule never sampled the slot loading"),
            "memory schedule: a stage never sampled loading or warming fails");
    partial = run;
    partial.resize(9);
    require(contains_text(pocket_mem_failures(pocket_mem_of(partial, 100), step, mimi, 100, 44, 90),
                          "never sampled the slot ready"),
            "memory schedule: a run without a steady-state sample fails");
    // Missing probe readings fail instead of reading 0.
    qm::AneGrowth none;
    require(qm::probe_failures(false, 0, true, none).size() == 2 && qm::probe_failures(true, 3, false, none).size() == 1,
            "memory schedule: a failed baseline, footprint probe or compiler query fails the run");
    require(qm::ane_query_loading(P::compile, P::load) == std::nullopt &&
                qm::ane_query_loading(P::load, P::warm) == std::optional<bool>(true) &&
                qm::ane_query_loading(P::ready, P::ggml) == std::optional<bool>(false),
            "memory schedule: a compiler query crossing the load's start or end is dropped");
    std::puts("[memory] schedule attribution and margins: ok");
}

void check_detection_and_fit() {
    env_guard units(pd::COREML_UNITS_ENV), strict(pd::COREML_STRICT_ENV), disabled(pd::COREML_DISABLE_ENV);
    unsetenv(pd::COREML_UNITS_ENV);
    unsetenv(pd::COREML_STRICT_ENV);
    unsetenv(pd::COREML_DISABLE_ENV);
    temp_dir dir("detect");
    const std::string gguf = (dir.path / "flow-lm.gguf").string(), mimi_gguf = (dir.path / "mimi.gguf").string();
    require(!pd::coreml_step_enabled(gguf, pd::resolve_coreml_policy()), "no sidecar");
    require(!pd::coreml_decoder_enabled(mimi_gguf, pd::resolve_coreml_policy()), "no decoder sidecar");
    // The decoder alone makes fit refuse too.
    fs::create_directories(pd::coreml_decoder_path(mimi_gguf));
    {
        tts_cpp::pocket::FitOptions decoder_only;
        decoder_only.text = "Hi.";
        decoder_only.engine.flow_lm_path = gguf;
        decoder_only.engine.mimi_path = mimi_gguf;
        decoder_only.engine.frontend_path = (dir.path / "frontend.json").string();
        decoder_only.engine.voice_path = (dir.path / "voice.gguf").string();
#ifdef TTS_CPP_USE_COREML
        require(pd::coreml_decoder_enabled(mimi_gguf, pd::resolve_coreml_policy()), "decoder sidecar detected");
        require(tts_cpp::pocket::fit_params(decoder_only).reason == "coreml-memory-estimate-unavailable",
                "fit refuses the Core ML decoder");
#else
        require(!pd::coreml_decoder_enabled(mimi_gguf, pd::resolve_coreml_policy()), "no Core ML decoder in this build");
#endif
    }
    fs::create_directories(pd::coreml_step_path(gguf));
    tts_cpp::pocket::FitOptions fit;
    fit.text = "Hi.";
    fit.engine.flow_lm_path = gguf;
    fit.engine.mimi_path = (dir.path / "mimi.gguf").string();
    fit.engine.frontend_path = (dir.path / "frontend.json").string();
    fit.engine.voice_path = (dir.path / "voice.gguf").string();
#ifdef TTS_CPP_USE_COREML
    require(pd::coreml_step_enabled(gguf, pd::resolve_coreml_policy()), "sidecar detected");
    require(tts_cpp::pocket::fit_params(fit).reason == "coreml-memory-estimate-unavailable", "fit refuses Core ML");
    setenv(pd::COREML_DISABLE_ENV, "1", 1);
    require(!pd::coreml_step_enabled(gguf, pd::resolve_coreml_policy()), "disabled sidecar");
    require(tts_cpp::pocket::fit_params(fit).reason != "coreml-memory-estimate-unavailable", "disabled sidecar is fitted");
#else
    require(!pd::coreml_step_enabled(gguf, pd::resolve_coreml_policy()), "no Core ML in this build");
    require(tts_cpp::pocket::fit_params(fit).reason != "coreml-memory-estimate-unavailable", "no Core ML refusal");
#endif
    fit.engine.coreml_compute_units = "gpu_please";
    require(tts_cpp::pocket::fit_params(fit).reason == "invalid-arguments", "fit rejects invalid compute units");
    // Rejected before any model file is opened.
    std::string error;
    try { Engine engine(fit.engine); } catch (const std::exception & e) { error = e.what(); }
    require(error.find("coreml_compute_units") != std::string::npos, "Engine rejects invalid compute units: " + error);
}

// Options, routes and the fit arithmetic of the shared Core ML runtime, with no
// model: an explicit cpu_and_ane for the step is refused when options are
// resolved; the environment's or the legacy all-stage cpu_and_ane leaves the
// step on its default units; the envelope sums the generated stages' resident
// parts and charges one transient; memory-degrade needs a budget.
void check_coreml_options() {
    env_guard units(pd::COREML_UNITS_ENV), disabled(pd::COREML_DISABLE_ENV);
    unsetenv(pd::COREML_UNITS_ENV);
    unsetenv(pd::COREML_DISABLE_ENV);
    pd::CoremlEngineOptions o;
    o.flow_lm_path = "missing/flow-lm.gguf";
    o.mimi_path = "missing/mimi.gguf";
    o.coreml.compute_units = "cpu_and_ane";
    std::string error;
    try { pd::coreml_plan_route(pd::CoremlStage::step, o); } catch (const std::invalid_argument & e) { error = e.what(); }
    require(error.find("cannot run on cpu_and_ane") != std::string::npos, "explicit cpu_and_ane step refused: " + error);
    require(!throws([&] { pd::coreml_plan_route(pd::CoremlStage::decoder, o); }), "cpu_and_ane decoder accepted");
    o.coreml.compute_units.reset();
    o.step_units = "cpu_and_ane";
    require(throws([&] { pd::coreml_plan_route(pd::CoremlStage::step, o); }), "explicit step units cpu_and_ane refused");
    o.step_units.clear();
    o.decoder_units = "gpu_please";
    error.clear();
    try { pd::coreml_plan_route(pd::CoremlStage::step, o); } catch (const std::invalid_argument & e) { error = e.what(); }
    require(error.find("coreml_decoder_units") != std::string::npos, "invalid decoder units named: " + error);
    o.decoder_units.clear();
    o.coreml.strict_wait_ms = -1;
    require(throws([&] { pd::coreml_plan_route(pd::CoremlStage::step, o); }), "negative strict_wait_ms");
    o.coreml.strict_wait_ms.reset();
    o.legacy_units = "cpu_and_ane";
#ifdef TTS_CPP_USE_COREML
    require(pd::coreml_plan_route(pd::CoremlStage::step, o).compute_units == "cpu_and_gpu",
            "legacy all-stage cpu_and_ane leaves the step on cpu_and_gpu");
    require(pd::coreml_plan_route(pd::CoremlStage::decoder, o).compute_units == "cpu_and_ane",
            "legacy all-stage units reach the decoder");
    o.legacy_units.clear();
    setenv(pd::COREML_UNITS_ENV, "cpu_and_ane", 1);
    require(pd::coreml_plan_route(pd::CoremlStage::step, o).compute_units == "cpu_and_gpu",
            "POCKET_COREML_COMPUTE_UNITS=cpu_and_ane leaves the step on cpu_and_gpu");
    unsetenv(pd::COREML_UNITS_ENV);
    const auto none = pd::coreml_plan_route(pd::CoremlStage::step, o);
    require(none.route == "none", "no GGUF, no prebuilt sidecar: route none, got " + none.route);
#endif
    // Envelope: resident parts summed, one transient, the larger margin.
    pd::CoremlRoute a, b;
    a.route = b.route = "generate";
    a.resident_bytes = 100; a.transient_bytes = 40; a.margin = 1.0;
    b.resident_bytes = 10; b.transient_bytes = 70; b.margin = 1.0;
    require(pd::coreml_envelope_bytes(a, b) == 180, "envelope = resident sum + largest transient");
    b.margin = 1.25;
    require(pd::coreml_envelope_bytes(a, b) == 225, "envelope with the larger margin");
    b.route = "none";
    require(pd::coreml_envelope_bytes(a, b) == 140, "only generated stages are charged");
    a.route = "prebuilt";
    require(pd::coreml_envelope_bytes(a, b) == 0, "no generated stage, no envelope");
    // Verdicts: memory-degrade only with a budget that the ggml projection fits.
    auto v = pd::decide_coreml_fit(100, 50, true, 120, std::nullopt);
    require(v.ggml_only_fits && !v.coreml_fits && !v.degraded && !v.fits && v.charged == 150,
            "without a budget the Core ML-inclusive verdict stands");
    v = pd::decide_coreml_fit(100, 50, true, 1000, int64_t(120));
    require(v.degraded && v.fits && v.charged == 100, "a budget only ggml fits degrades");
    v = pd::decide_coreml_fit(100, 50, true, 120, int64_t(1) << 50);
    require(v.degraded && v.fits, "degrade is decided on min(free, budget)");
    v = pd::decide_coreml_fit(100, 50, true, 1000, int64_t(90));
    require(!v.degraded && !v.ggml_only_fits && !v.coreml_fits && !v.fits,
            "a budget below the ggml projection refuses, however much memory is free");
    v = pd::decide_coreml_fit(100, 50, true, 1000, int64_t(160));
    require(!v.degraded && v.fits && v.charged == 150, "a budget both fit keeps Core ML");
    v = pd::decide_coreml_fit(100, 50, false, 120, int64_t(110));
    require(!v.degraded && v.fits && v.charged == 100, "no generated stage, no degrade");
    v = pd::decide_coreml_fit(100, 50, false, 1000, int64_t(90));
    require(!v.fits, "without a generated stage the budget still caps the ggml verdict");
    // One ceiling: the smaller of the fit's budget and the Core ML budget.
    require(!pd::effective_fit_budget(0, std::nullopt), "no budget set");
    require(pd::effective_fit_budget(0, int64_t(70)) == int64_t(70), "the Core ML budget alone");
    require(pd::effective_fit_budget(80, std::nullopt) == int64_t(80), "the fit's budget alone enables degrade");
    require(pd::effective_fit_budget(80, int64_t(70)) == int64_t(70) &&
                pd::effective_fit_budget(60, int64_t(70)) == int64_t(60),
            "both set: the smaller");
    require(pd::effective_fit_budget(~uint64_t(0), std::nullopt) == INT64_MAX, "a huge fit budget saturates");
    std::puts("[options] Core ML options, routes and fit arithmetic: ok");
}

std::vector<float> npy_floats(const fs::path & path) {
    const npy_array array = npy_load(path.string());
    const float * data = npy_as_f32(array);
    return std::vector<float>(data, data + array.n_elements());
}

#ifdef TTS_CPP_USE_COREML
// The random full-size decoder streams its eager reference one frame per
// call (past the attention window), in calls of several frames, and again
// after reset(); each defective sidecar is rejected at load.
void check_decoder_fixture(const fs::path & root) {
    const fs::path reference = root / "good" / "reference";
    std::ifstream config_file(reference / "config.json");
    const auto config = nlohmann::json::parse(config_file);
    const std::string source = config["source"];
    const auto sidecar = [&](const std::string & variant) { return (root / variant / "mimi-decoder.mlmodelc").string(); };
    const auto latents = npy_floats(reference / "latents.npy"), want = npy_floats(reference / "pcm.npy");
    for (const char * units : {"cpu_only", pd::COREML_DECODER_DEFAULT_UNITS}) {
        pd::CoremlDecoder decoder(sidecar("good"), units, source);
        const auto one_by_one = decoder.decode(latents);
        const double snr = snr_db(one_by_one, want);
        decoder.reset();
        std::vector<float> split;
        const size_t frame = pd::CoremlDecoderGeometry::latent;
        for (size_t at = 0, count = 3; at < latents.size(); at += count * frame, count = count % 5 + 1) {
            const size_t end = std::min(latents.size(), at + count * frame);
            const auto part = decoder.decode(std::vector<float>(latents.begin() + at, latents.begin() + end));
            split.insert(split.end(), part.begin(), part.end());
        }
        std::printf("[mimi good %s] SNR %.1f dB vs the eager stream; split calls %.2e from one-by-one\n", units, snr,
                    max_error(split, one_by_one));
        // Broken state feedback scores <= 6 dB on these weights. The Neural
        // Engine computes them at ~25 dB (the real checkpoint at ~46 dB).
        require(snr > (std::string(units) == "cpu_only" ? 30 : 15),
                std::string("tiny decoder diverges from its reference on ") + units);
        require(max_error(split, one_by_one) < 1e-6, "split decode calls change the stream");
    }
    // Strict mode keeps a decoder whose prediction failed; the next chunk's
    // reset() must clear the non-finite stream state. The Neural Engine
    // decodes a NaN latent to finite PCM, so the failure is forced on the CPU.
    {
        pd::CoremlDecoder decoder(sidecar("good"), "cpu_only", source);
        require(throws([&] { decoder.decode(std::vector<float>(pd::CoremlDecoderGeometry::latent, NAN)); }),
                "a NaN latent decoded");
        decoder.reset();
        require(snr_db(decoder.decode(latents), want) > 30, "reset after a failure leaves the stream broken");
    }
    require(throws([&] { pd::CoremlDecoder decoder(sidecar("good"), "cpu_only", "another-mimi"); }),
            "a decoder of another checkpoint");
    { pd::CoremlDecoder probe(sidecar("probe"), "cpu_only", source); }
    std::vector<std::string> variants = {"latents", "cos", "sin", "mask", "keys", "values", "pcm", "new_keys",
                                         "new_values", "source", "stateful"};
    for (const char * h : {"h_upsample", "h_conv0", "h_convtr0", "h_residual0", "h_convtr1", "h_residual1", "h_convtr2",
                           "h_residual2", "h_final"}) {
        variants.push_back(h);
        variants.push_back(std::string("new_") + h);
    }
    for (const auto & variant : variants) {
        std::string error;
        try {
            pd::CoremlDecoder decoder(sidecar(variant), "cpu_only", source);
        } catch (const std::exception & e) {
            error = e.what();
        }
        require(!error.empty(), "accepted the mimi " + variant + " variant");
        std::printf("[mimi %s] rejected: %s\n", variant.c_str(), error.c_str());
    }
}
#endif

int shape_fixture() {
#ifndef TTS_CPP_USE_COREML
    std::puts("SKIP: built without Core ML (TTS_CPP_COREML=OFF)");
    return SKIP;
#else
    const char * root_env = std::getenv("POCKET_COREML_TEST_SHAPES");
    if (root_env == nullptr || !*root_env) {
        std::puts("SKIP: set POCKET_COREML_TEST_SHAPES to the prepare_pocket_coreml_fixture.py output");
        return SKIP;
    }
    const fs::path root(root_env), reference = root / "good" / "reference";
    std::ifstream config_file(reference / "config.json");
    const auto config = nlohmann::json::parse(config_file);
    pd::FlowConfig cfg;
    cfg.dim = config["dim"]; cfg.heads = config["heads"]; cfg.layers = config["layers"];
    cfg.ff_dim = config["ff_dim"]; cfg.latent_dim = config["latent_dim"]; cfg.flow_dim = config["flow_dim"];
    cfg.flow_depth = config["flow_depth"]; cfg.vocab_size = config["vocab_size"]; cfg.rope_base = config["rope_base"];
    const int context = config["context"], prefix = config["prefix"], frames = config["frames"];
    const std::string source = config["source"];
    const auto sidecar = [&](const char * variant) { return (root / variant / "flow-lm-step.mlmodelc").string(); };

    // The real step reproduces its float32 reference on every placement.
    const auto keys = npy_floats(reference / "prefix_keys.npy"), values = npy_floats(reference / "prefix_values.npy");
    const auto latents = npy_floats(reference / "latents.npy"), noises = npy_floats(reference / "noises.npy");
    const auto hidden_ref = npy_floats(reference / "hidden.npy"), next_ref = npy_floats(reference / "next.npy");
    pd::CacheRows rows;
    rows.count = prefix;
    const size_t layer_size = size_t(prefix) * cfg.dim;
    for (int l = 0; l < cfg.layers; ++l) {
        rows.keys.emplace_back(keys.begin() + l * layer_size, keys.begin() + (l + 1) * layer_size);
        rows.values.emplace_back(values.begin() + l * layer_size, values.begin() + (l + 1) * layer_size);
    }
    const auto at = [](const std::vector<float> & all, int row, int width) {
        return std::vector<float>(all.begin() + size_t(row) * width, all.begin() + size_t(row + 1) * width);
    };
    for (const char * units : {"cpu_only", pd::COREML_STEP_DEFAULT_UNITS}) {
        pd::CoremlStep step(sidecar("good"), units, cfg, source);
        require(step.context() == context, "context read from the sidecar");
        std::vector<float> hidden, next;
        const auto replay = [&](const char * tag) {
            for (int f = 0; f < frames; ++f) {
                step.step(prefix + f, at(latents, f, cfg.latent_dim), at(noises, f, cfg.latent_dim), hidden, next);
                const double eh = max_error(hidden, at(hidden_ref, f, cfg.dim));
                const double en = max_error(next, at(next_ref, f, cfg.latent_dim));
                std::printf("[%s %s] frame %d: hidden error %.2e, next latent error %.2e\n", tag, units, f, eh, en);
                require(eh < 3e-2 && en < 3e-2, std::string("tiny step diverges from its reference on ") + units);
            }
        };
        step.write_rows(rows);
        replay("good");

        // A failed prediction can leave NaN in a row past the prefix (here
        // written directly). Rewriting the prefix keeps it: the blended write
        // multiplies it by zero and the attention mask only adds to its score.
        // reset() clears it and the frames replay to the reference.
        const std::vector<std::vector<float>> nan_rows(size_t(cfg.layers), std::vector<float>(size_t(cfg.dim), NAN));
        step.write_rows({prefix + 1, 1, nan_rows, nan_rows});
        require(throws([&] { replay("poisoned"); }), "a NaN row past the prefix left the step finite");
        step.write_rows(rows);
        require(throws([&] { replay("prefix rewritten"); }), "rewriting the prefix cleared a NaN row");
        step.reset(rows);
        replay("reset");

        const std::vector<float> zeros(size_t(cfg.latent_dim));
        require(throws([&] { step.step(context, zeros, zeros, hidden, next); }), "position past the context");
        require(throws([&] { pd::CacheRows late = rows; late.start = context - 1; step.write_rows(late); }),
                "rows past the context");
    }
    require(throws([&] { pd::CoremlStep step(sidecar("good"), "cpu_only", cfg, "another-checkpoint"); }),
            "a sidecar of another checkpoint");

    // The probe declares the expected features; each other variant has one defect.
    { pd::CoremlStep probe(sidecar("probe"), "cpu_only", cfg, source); }
    for (const char * variant : {"latent", "noise", "cos", "sin", "mask", "write_mask", "hidden", "next_latent", "keys",
                                 "values", "schema", "source", "stateless"}) {
        std::string error;
        try {
            pd::CoremlStep step(sidecar(variant), "cpu_only", cfg, source);
        } catch (const std::exception & e) {
            error = e.what();
        }
        require(!error.empty(), std::string("accepted the ") + variant + " variant");
        std::printf("[%s] rejected: %s\n", variant, error.c_str());
    }

    // The runner bounds every state write by the declared buffer.
    moss_coreml_model * model = moss_coreml_load(sidecar("probe").c_str(), "cpu_only");
    require(model != nullptr, "probe loads in the runner");
    int64_t dims[8] = {};
    require(moss_coreml_state_dims(model, "keys", dims, 8) == 4 && dims[2] == context, "state dims");
    const int64_t head_dim = cfg.dim / cfg.heads;
    std::vector<float> row(size_t(cfg.heads * head_dim), 1.0f);
    const int64_t count[4] = {1, cfg.heads, 1, head_dim};
    const int64_t inside[4] = {0, 0, context - 1, 0}, outside[4] = {0, 0, context, 0};
    const int status_inside = moss_coreml_write_state(model, "keys", row.data(), inside, count, 4);
    const int status_outside = moss_coreml_write_state(model, "keys", row.data(), outside, count, 4);
    const int status_rank = moss_coreml_write_state(model, "keys", row.data(), inside, count, 3);
    const int status_name = moss_coreml_write_state(model, "kv", row.data(), inside, count, 4);
    moss_coreml_free(model);
    require(status_inside == 0 && status_outside != 0 && status_rank != 0 && status_name != 0, "state write bounds");
    check_decoder_fixture(root / "mimi");
    std::puts("test-pocket-coreml --shape-fixture: ok");
    return 0;
#endif
}

EngineOptions bundle_options(const fs::path & dir) {
    EngineOptions opts;
    opts.flow_lm_path = (dir / "flow-lm.gguf").string();
    opts.mimi_path = (dir / "mimi.gguf").string();
    opts.frontend_path = (dir / "frontend.json").string();
    opts.voice_path = (dir / "voice.gguf").string();
    return opts;
}

// The sidecar against ggml F32 FlowLM, teacher-forced on ggml's latents.
void check_stage_parity(const EngineOptions & opts, const std::string & step_path = {}) {
    pd::FlowLM lm(opts.flow_lm_path, 2048, 4);
    const auto voice = lm.read_voice(opts.voice_path);
    pd::CoremlStep step(step_path.empty() ? pd::coreml_step_path(opts.flow_lm_path) : step_path,
                        pd::COREML_STEP_DEFAULT_UNITS, lm.config(), lm.source_hash());
    step.write_rows({0, voice.frames, voice.keys, voice.values});
    lm.restore_voice(voice);
    pd::Frontend frontend(opts.frontend_path);
    lm.prefill(lm.text_embeddings(frontend.encode("The quick brown fox jumps over the lazy dog.")));
    step.write_rows(lm.cache_rows(voice.frames, lm.position() - voice.frames));
    std::mt19937 rng(7);
    std::normal_distribution<float> normal(0.0f, std::sqrt(0.3f));
    std::vector<float> previous, hidden, next, noise(size_t(lm.config().latent_dim));
    double worst_cosine = 1, worst_eos = 0, worst_latent = 0;
    for (int f = 0; f < 48; ++f) {
        for (auto & x : noise) x = normal(rng);
        const int position = lm.position();
        const auto condition = lm.advance(previous);
        const auto expected = lm.sample(condition.hidden, noise, 1);
        step.step(position, previous.empty() ? lm.bos_latent() : previous, noise, hidden, next);
        worst_cosine = std::min(worst_cosine, cosine(hidden, condition.hidden));
        worst_eos = std::max(worst_eos, double(std::abs(lm.eos_logit(hidden) - condition.eos_logit)));
        worst_latent = std::max(worst_latent, max_error(next, expected));
        previous = expected;
    }
    std::printf("[stage] 48 frames: hidden cosine >= %.6f, EOS logit error <= %.4f, next latent error <= %.4f\n",
                worst_cosine, worst_eos, worst_latent);
    require(worst_cosine > 0.9995 && worst_eos < 0.1 && worst_latent < 0.05, "Core ML step diverges from ggml FlowLM");
}

// A retired sidecar's frames are prefilled into the ggml cache in one batch;
// the next frame must match frame-by-frame ggml decoding.
void check_replay(const EngineOptions & opts) {
    pd::FlowLM lm(opts.flow_lm_path, 2048, 4);
    const auto voice = lm.read_voice(opts.voice_path);
    const auto tokens = pd::Frontend(opts.frontend_path).encode("Replay the generated frames.");
    lm.restore_voice(voice);
    lm.prefill(lm.text_embeddings(tokens));
    std::vector<float> previous, fed;
    std::mt19937 rng(11);
    std::normal_distribution<float> normal(0.0f, 0.5f);
    std::vector<float> noise(size_t(lm.config().latent_dim));
    for (int f = 0; f < 6; ++f) {
        const auto input = previous.empty() ? lm.bos_latent() : previous;
        fed.insert(fed.end(), input.begin(), input.end());
        const auto condition = lm.advance(previous);
        for (auto & x : noise) x = normal(rng);
        previous = lm.sample(condition.hidden, noise, 1);
    }
    const auto sequential = lm.advance(previous);
    lm.restore_voice(voice);
    lm.prefill(lm.text_embeddings(tokens));
    lm.prefill(lm.audio_embeddings(fed));
    const auto replayed = lm.advance(previous);
    const double error = max_error(replayed.hidden, sequential.hidden);
    std::printf("[replay] hidden error %.2e, EOS %.4f vs %.4f\n", error, replayed.eos_logit, sequential.eos_logit);
    require(error < 1e-3 && std::abs(replayed.eos_logit - sequential.eos_logit) < 1e-2, "ggml replay diverges");
}

// The real decoder against ggml Mimi on the same streamed latents.
void check_decoder_parity(const EngineOptions & opts, const std::string & decoder_path = {}) {
    pd::Mimi mimi(opts.mimi_path, 4);
    pd::CoremlDecoder decoder(decoder_path.empty() ? pd::coreml_decoder_path(opts.mimi_path) : decoder_path,
                              pd::COREML_DECODER_DEFAULT_UNITS, mimi.source_hash());
    std::mt19937 rng(3);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> latents(size_t(40) * pd::Mimi::latent_dim());
    for (auto & x : latents) x = normal(rng);
    mimi.reset_decoder();
    std::vector<float> want;
    for (size_t at = 0; at < latents.size(); at += size_t(8) * pd::Mimi::latent_dim()) {
        const auto part = mimi.decode(std::vector<float>(latents.begin() + at, latents.begin() + at + 8 * pd::Mimi::latent_dim()));
        want.insert(want.end(), part.begin(), part.end());
    }
    const double snr = snr_db(decoder.decode(latents), want);
    std::printf("[decoder] %s: 40 frames, SNR %.1f dB vs ggml Mimi\n", decoder.label().c_str(), snr);
    require(snr > 35, "Core ML Mimi decoder diverges from ggml Mimi");
}

void check_engine(const EngineOptions & prebuilt) {
    env_guard units(pd::COREML_UNITS_ENV), strict(pd::COREML_STRICT_ENV), disabled(pd::COREML_DISABLE_ENV);
    unsetenv(pd::COREML_UNITS_ENV);
    unsetenv(pd::COREML_STRICT_ENV);
    unsetenv(pd::COREML_DISABLE_ENV);
    // The sidecars load in the background: pin readiness so every comparison
    // below runs on Core ML.
    EngineOptions base = prebuilt;
    base.coreml.wait = tts_cpp::CoremlOptions::Wait::construct;
    const std::string text = "The quick brown fox jumps over the lazy dog. It barked twice.";
    Engine coreml(base);
    require(coreml.flow_lm_on_coreml() && coreml.mimi_on_coreml(), "sidecars loaded");
    const auto batch = coreml.synthesize(text);
    std::vector<float> streamed;
    const auto stream = coreml.synthesize_stream(text, [&](const float * pcm, size_t n, int) {
        streamed.insert(streamed.end(), pcm, pcm + n); return true;
    });
    require(batch.flow_lm_backend.rfind("coreml-", 0) == 0 && stream.flow_lm_backend == batch.flow_lm_backend,
            "Core ML label, got " + batch.flow_lm_backend);
    require(batch.mimi_backend.rfind("coreml-", 0) == 0 && stream.mimi_backend == batch.mimi_backend,
            "Core ML Mimi label, got " + batch.mimi_backend);
    require(max_error(streamed, batch.pcm) < 1e-4, "stream and batch differ on Core ML");

    setenv(pd::COREML_DISABLE_ENV, "1", 1);
    Engine ggml(base);
    unsetenv(pd::COREML_DISABLE_ENV);
    const auto reference = ggml.synthesize(text);
    require(!ggml.flow_lm_on_coreml() && !ggml.mimi_on_coreml() && reference.flow_lm_backend == "ggml" &&
            reference.mimi_backend == "ggml", "POCKET_COREML_DISABLE keeps ggml");
    std::printf("[engine] %s: %d frames; ggml: %d frames\n", batch.flow_lm_backend.c_str(), batch.generated_frames,
                reference.generated_frames);
    require(std::abs(batch.generated_frames - reference.generated_frames) <= 4, "Core ML changes the speech length");

    // Several chunks in one call: each copies its text rows and starts at a
    // lower position than the previous chunk's last frame.
    EngineOptions chunked = base;
    chunked.max_tokens = 12;
    const auto several = Engine(chunked).synthesize(text);
    setenv(pd::COREML_DISABLE_ENV, "1", 1);
    const auto several_ggml = Engine(chunked).synthesize(text);
    unsetenv(pd::COREML_DISABLE_ENV);
    std::printf("[engine] %d-token chunks: %s %d frames; ggml %d frames\n", chunked.max_tokens,
                several.flow_lm_backend.c_str(), several.generated_frames, several_ggml.generated_frames);
    require(several.flow_lm_backend == batch.flow_lm_backend && several.mimi_backend == batch.mimi_backend,
            "multi-chunk call left Core ML");
    require(std::abs(several.generated_frames - several_ggml.generated_frames) <= 8, "multi-chunk speech length");

    EngineOptions steps = base;
    steps.steps = 2;
    require(Engine(steps).synthesize("Hi.").flow_lm_backend == "mixed", "steps > 1 run the flow on ggml");

    // A chunk past the sidecar's context: strict refuses before any PCM.
    EngineOptions long_chunk = base;
    long_chunk.coreml_strict = true;
    long_chunk.max_tokens = 1024;
    long_chunk.context = 4096;
    std::string words;
    for (int i = 0; i < 40; ++i) words += "one two three four five six seven eight nine ten ";
    Engine strict_engine(long_chunk);
    int callbacks = 0;
    std::string error;
    try {
        strict_engine.synthesize_stream(words, [&](const float *, size_t, int) { ++callbacks; return true; });
    } catch (const std::exception & e) {
        error = e.what();
    }
    require(error.find("Core ML strict policy") != std::string::npos,
            "strict mode serves a chunk past the sidecar's context: " + error);
    require(callbacks == 0, "strict refusal after PCM");

    tts_cpp::pocket::FitOptions fit;
    fit.engine = base;
    fit.text = text;
    require(tts_cpp::pocket::fit_params(fit).reason == "coreml-memory-estimate-unavailable", "fit refuses the sidecar");
}

#ifdef TTS_CPP_USE_COREML
using Json = nlohmann::json;
using CoremlOptions = tts_cpp::CoremlOptions;

double mib(uint64_t bytes) { return bytes / 1048576.0; }

std::string host_chip() {
    char brand[128] = {};
    size_t size = sizeof(brand);
    return sysctlbyname("machdep.cpu.brand_string", brand, &size, nullptr, 0) == 0 ? std::string(brand) : "unknown";
}

Json status_of(const Engine & e) { return Json::parse(e.coreml_status()); }

const Json & stage_status(const Json & status, const std::string & stage) {
    for (const Json & s : status.at("stages")) if (s.at("stage") == stage) return s;
    throw std::runtime_error("no status for stage " + stage + ": " + status.dump());
}

// The overlapped memory schedule: ggml synthesis runs in a loop on this thread
// while both sidecars are generated, compiled, loaded and warmed into
// `cache_dir` in the background, until three calls ran both stages on Core ML.
// One sampler reads the footprint every 2 ms and the slots' states every
// 10 ms; another reads ANECompilerService every 20 ms. The runtime runs one
// Core ML job at a time, so each sample belongs to at most one stage's prepare
// or load. Held to the templates' constants:
//   prepare (per stage)    the peak over the footprint just before that
//                          stage's generation began, against
//                          generation_peak_bytes + 8 MiB + compile_peak_bytes;
//   load+warm (per stage)  the peak over the footprint just before its load
//                          began, against load_peak_bytes for its units;
//   steady                 both ready, Core ML calls running: the peak over the
//                          footprint just before the first load, against the
//                          two resident parts;
//   whole run              the peak over the baseline, against the envelope;
//   compiler               ANECompilerService's growth while a stage on the
//                          Neural Engine loads and warms, against that
//                          template's ane_compile_reserve_bytes.
// The GGUF is replaced, or removed, after the ggml weights were loaded and
// before the sidecars are made: FlowLM is loaded through a symlink, which is
// then pointed at a clone of the file (same bytes, another inode), and Mimi
// through a clone that is then deleted. Generation must not follow either
// path: the step fails with source_changed, Mimi stays on ggml, nothing is
// generated, and the construction does not throw.
void source_changed_gate(const EngineOptions & bundle) {
    std::puts("[source] replaced and removed GGUFs");
    const fs::path dir = fs::temp_directory_path() / ("pocket-source-changed-" + test_process_tag());
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir / "cache");
    const fs::path link = dir / "flow-lm.gguf", clone = dir / "flow-lm-clone.gguf", mimi_copy = dir / "mimi.gguf";
    fs::create_symlink(fs::absolute(bundle.flow_lm_path), link);
    require(::clonefile(fs::absolute(bundle.mimi_path).c_str(), mimi_copy.c_str(), 0) == 0, "clone of mimi.gguf");
    EngineOptions opts = bundle;
    opts.flow_lm_path = link.string();
    opts.mimi_path = mimi_copy.string();
    opts.coreml.cache_dir = (dir / "cache").string();
    opts.coreml.wait = CoremlOptions::Wait::construct;
    auto co = pd::coreml_engine_options(opts);
    pd::FlowLM lm(opts.flow_lm_path, 2048, 4, nullptr,
                  [&](int fd, const std::string & p) { co.step_source = pd::coreml_loaded_source(fd, p); });
    pd::Mimi mimi(opts.mimi_path, 4, nullptr,
                  [&](int fd, const std::string & p) { co.decoder_source = pd::coreml_loaded_source(fd, p); });
    require(co.step_source && co.decoder_source, "both loads record the file they read");
    require(::clonefile(fs::absolute(bundle.flow_lm_path).c_str(), clone.c_str(), 0) == 0, "clone of flow-lm.gguf");
    fs::remove(link);
    fs::create_symlink(clone, link);
    fs::remove(mimi_copy);
    std::unique_ptr<pd::CoremlStages> stages;
    try {
        stages = pd::CoremlStages::create(co, lm.config(), lm.source_hash(), 64, false);
    } catch (const std::exception & e) {
        require(false, std::string("construction after the replacement throws: ") + e.what());
    }
    require(stages != nullptr, "stages are made");
    const Json st = Json::parse(stages->status_json());
    std::printf("[source] step %s, mimi %s\n", stage_status(st, "step").dump().substr(0, 160).c_str(),
                stage_status(st, "mimi").dump().substr(0, 160).c_str());
    // The removed Mimi GGUF no longer matches a template when the stages are
    // planned (no_template); had it vanished after that, its prepare would
    // fail with source_changed. Either way it stays on ggml.
    const std::string mimi_state = stage_status(st, "mimi").at("state");
    require(!stages->step_ready() && !stages->decoder_ready() &&
                stage_status(st, "step").at("state") == "failed" &&
                stage_status(st, "step").dump().find("source_changed") != std::string::npos &&
                (mimi_state == "no_template" ||
                 (mimi_state == "failed" && stage_status(st, "mimi").dump().find("source_changed") != std::string::npos)),
            "a replaced or removed GGUF is not generated from: " + st.dump());
    // The construction group reports the failed step (and lists the decoder,
    // with no route, separately): it did not succeed.
    const Json construction = Json::parse(stages->construction_json());
    bool all_failed = !construction.at("slots").empty();
    for (const Json & slot : construction.at("slots")) all_failed = all_failed && slot.at("state") == "failed";
    require(construction.at("success") == false && construction.at("nothing_to_prepare") == false && all_failed &&
                construction.at("slots").size() + construction.at("skipped_route_none").size() == 2,
            "the construction group reports the failed slot: " + construction.dump());
    require(!fs::exists(dir / "cache" / "qvac-coreml-v1" / "artifacts") ||
                fs::is_empty(dir / "cache" / "qvac-coreml-v1" / "artifacts"),
            "nothing is published");
    stages.reset();
    fs::remove_all(dir, ec);
}

// A chunk needing more positions than the generated step holds (1024) runs
// on ggml whatever the step's readiness, so it neither requests the step nor
// reports coreml_pending; a chunk that fits requests it and, while it is
// still being generated, reports pending. Mimi is kept out of the picture:
// cpu_only is not a validated decoder placement, so its route is none.
void capacity_gate(const EngineOptions & bundle) {
    std::puts("[capacity] oversized and fitting chunks against a pending step");
    const fs::path dir = fs::temp_directory_path() / ("pocket-capacity-" + test_process_tag());
    std::error_code ec;
    fs::remove_all(dir, ec);
    EngineOptions opts = bundle;
    opts.coreml.cache_dir = (dir / "cache").string();
    opts.coreml.prewarm = CoremlOptions::Prewarm::none;
    opts.coreml.wait = CoremlOptions::Wait::none;
    opts.coreml_decoder_units = "cpu_only";
    opts.max_tokens = 1024;
    // A one-chunk text whose positions (voice, tokens and frame budget) exceed
    // the step's 1024 but fit the Engine's context.
    const int voice_frames = pd::FlowLM(opts.flow_lm_path, 2048, 4).read_voice(opts.voice_path).frames;
    const pd::Frontend frontend(opts.frontend_path);
    std::string text = "Pocket reads this long sentence";
    size_t positions = 0;
    while (positions <= 1100) {
        text += " and then the quick brown fox jumps over the lazy dog";
        const auto chunks = frontend.split(text + ".", opts.max_tokens);
        require(chunks.size() == 1, "the long text stays one chunk");
        const auto tokens = frontend.encode(chunks[0].text);
        positions = size_t(voice_frames) + tokens.size() + pd::frame_budget(tokens.size(), chunks[0].tail_frames).max_frames();
    }
    text += ".";
    require(positions <= size_t(opts.context), "the long chunk fits the Engine's context");
    Engine engine(opts);
    const Json st = status_of(engine);
    require(stage_status(st, "mimi").at("route") == "none" && stage_status(st, "step").at("state") == "idle",
            "Mimi has no route (cpu_only unvalidated) and the step is idle: " + st.dump());
    // Only the chunk's start matters (the step is acquired there), so the
    // call is cancelled at its first audio instead of synthesizing the whole
    // long text.
    const auto oversized = engine.synthesize_stream(text, [](const float *, size_t, int) { return false; });
    const Json after = status_of(engine);
    std::printf("[capacity] %zu positions: FlowLM %s, pending %d, step %s\n", positions,
                oversized.flow_lm_backend.c_str(), int(oversized.coreml_pending),
                stage_status(after, "step").at("state").get<std::string>().c_str());
    require(oversized.cancelled && oversized.flow_lm_backend == "ggml" && !oversized.coreml_pending &&
                stage_status(after, "step").at("state") == "idle",
            "an oversized chunk runs ggml without requesting the step or reporting pending: " + after.dump());
    const auto fitting = engine.synthesize("Hello there.");
    std::printf("[capacity] fitting chunk: FlowLM %s, pending %d\n", fitting.flow_lm_backend.c_str(),
                int(fitting.coreml_pending));
    require(fitting.flow_lm_backend == "ggml" && fitting.coreml_pending,
            "a fitting chunk requests the step and reports pending while it is generated");
    engine.coreml_wait(-1);
    fs::remove_all(dir, ec);
}

void memory_schedule(const EngineOptions & bundle, const std::string & cache_dir) {
    std::printf("[memory] schedule (cache %s)\n", cache_dir.c_str());
    EngineOptions opts = bundle;
    opts.coreml.cache_dir = cache_dir;
    auto co = pd::coreml_engine_options(opts);
    const auto step_route = pd::coreml_plan_route(pd::CoremlStage::step, co);
    const auto decoder_route = pd::coreml_plan_route(pd::CoremlStage::decoder, co);
    require(step_route.route == "generate" && decoder_route.route == "generate",
            "memory schedule: both routes generate (" + step_route.route + " " + step_route.reason + "; " +
                decoder_route.route + " " + decoder_route.reason + ")");
    // The Engine's two stages on their own: the ggml FlowLM and Mimi are
    // loaded and their arenas warmed before the baseline, then the sidecars
    // are generated, compiled, loaded and warmed in the background (prewarm
    // predicted requests both at once) while ggml keeps working.
    pd::FlowLM lm(opts.flow_lm_path, 2048, 4, nullptr,
                  [&](int fd, const std::string & p) { co.step_source = pd::coreml_loaded_source(fd, p); });
    pd::Mimi mimi(opts.mimi_path, 4, nullptr,
                  [&](int fd, const std::string & p) { co.decoder_source = pd::coreml_loaded_source(fd, p); });
    const auto voice = lm.read_voice(opts.voice_path);
    const auto tokens = pd::Frontend(opts.frontend_path).encode("The quick brown fox jumps over the lazy dog.");
    constexpr int FRAMES = 16;
    std::mt19937 rng(5);
    std::normal_distribution<float> normal(0.0f, 0.5f);
    std::vector<float> noise(size_t(lm.config().latent_dim));
    auto ggml_unit = [&] {
        lm.restore_voice(voice);
        lm.prefill(lm.text_embeddings(tokens));
        std::vector<float> previous, latents;
        for (int f = 0; f < FRAMES; ++f) {
            for (auto & x : noise) x = normal(rng);
            const auto condition = lm.advance(previous);
            previous = lm.sample(condition.hidden, noise, 1);
            const auto d = lm.denormalize(previous);
            latents.insert(latents.end(), d.begin(), d.end());
        }
        mimi.reset_decoder();
        const size_t batch = size_t(mimi.max_decode_frames()) * pd::Mimi::latent_dim();
        for (size_t at = 0; at < latents.size(); at += batch)
            mimi.decode(std::vector<float>(latents.begin() + at, latents.begin() + std::min(latents.size(), at + batch)));
    };
    ggml_unit();
    ggml_unit();
    std::atomic<bool> stop{false};
    std::atomic<const pd::CoremlStages *> watched{nullptr};
    // Both stages' slot states, read from the status (idle and the like read
    // as ggml: nothing of the sidecar's yet).
    auto phases = [](const pd::CoremlStages * stages) {
        std::pair<qm::mem_phase, qm::mem_phase> out{qm::mem_phase::ggml, qm::mem_phase::ggml};
        if (stages == nullptr) return out;
        const Json st = Json::parse(stages->status_json());
        for (const Json & s : st.at("stages")) {
            const qm::mem_phase p = qm::mem_phase_of(s.at("state").get<std::string>());
            (s.at("stage") == "step" ? out.first : out.second) = p;
        }
        return out;
    };
    std::vector<pocket_mem_sample> samples;
    samples.reserve(1 << 17);
    size_t footprint_failures = 0;
    std::atomic<int> steady_samples{0};
    std::vector<qm::AneSample> ane_samples;
    uint64_t base = 0;
    const bool base_ok = qm::phys_footprint(base);
    std::thread sampler([&] {
        while (!stop.load()) {
            const auto now = phases(watched.load());
            uint64_t footprint = 0;
            if (qm::phys_footprint(footprint)) {
                samples.push_back({footprint, now.first, now.second});
                if (now.first == qm::mem_phase::ready && now.second == qm::mem_phase::ready) ++steady_samples;
            } else {
                ++footprint_failures;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    });
    std::thread ane_sampler([&] {
        while (!stop.load()) {
            // The decoder is the stage the Neural Engine compiles. Each query
            // is bracketed by two state reads; one that crossed the start or
            // the end of the decoder's load is dropped.
            const qm::mem_phase before = phases(watched.load()).second;
            qm::AneSample a;
            a.ok = qm::ane_compiler_rss(a.rss);
            const qm::mem_phase after = phases(watched.load()).second;
            if (const std::optional<bool> loading = qm::ane_query_loading(before, after)) {
                a.loading = *loading;
                ane_samples.push_back(a);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    });
    const auto t0 = std::chrono::steady_clock::now();
    int ggml_calls = 0, coreml_calls = 0;
    {
        auto stages = pd::CoremlStages::create(co, lm.config(), lm.source_hash(), voice.frames + 2, false);
        watched.store(stages.get());
        const pd::CacheRows voice_rows{0, voice.frames, voice.keys, voice.values};
        while (std::chrono::steady_clock::now() - t0 < std::chrono::minutes(5)) {
            if (!stages->step_ready() || !stages->decoder_ready()) {
                ggml_unit();
                ++ggml_calls;
                continue;
            }
            // One unit on both sidecars: the text rows, FRAMES step
            // predictions, then the frames through the decoder.
            auto step = stages->acquire_step(FRAMES, nullptr);
            auto decoder = stages->acquire_decoder(FRAMES, nullptr);
            require(step.step && decoder.model, "memory schedule: ready sidecars acquired");
            lm.restore_voice(voice);
            lm.prefill(lm.text_embeddings(tokens));
            int position = lm.position();
            std::vector<float> previous = lm.bos_latent(), hidden, next, latents;
            {
                auto lock = step.lock();
                if (!step.step->primed()) step.step->prime(voice_rows);
                step.step->write_rows(lm.cache_rows(voice.frames, position - voice.frames));
                for (int f = 0; f < FRAMES; ++f, ++position) {
                    for (auto & x : noise) x = normal(rng);
                    step.step->step(position, previous, noise, hidden, next);
                    previous = next;
                    const auto d = lm.denormalize(next);
                    latents.insert(latents.end(), d.begin(), d.end());
                }
            }
            pd::CoremlDecoder stream(decoder.model, decoder.label);
            {
                auto lock = decoder.lock();
                stream.decode(latents);
            }
            // The steady state: keep running both sidecars for at least a
            // second, so the footprint is sampled with both ready.
            if (++coreml_calls >= 3 && steady_samples.load() >= 100) break;
        }
        std::printf("  status: %s\n", stages->status_json().c_str());
        watched.store(nullptr);
        stop.store(true);
        sampler.join();
        ane_sampler.join();
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    const pocket_mem m = pocket_mem_of(samples, base);
    const auto ane = qm::ane_compiler_growth(ane_samples);
    const auto baseline = qm::ane_compiler_baseline(ane);
    const uint64_t envelope = pd::coreml_envelope_bytes(step_route, decoder_route);
    const uint64_t resident = step_route.resident_bytes + decoder_route.resident_bytes;
    const bool decoder_ane = decoder_route.compute_units == "cpu_and_ane" || decoder_route.compute_units == "all";
    const qm::mem_breakdown * per[2] = {&m.step, &m.mimi};
    const pd::CoremlRoute * routes[2] = {&step_route, &decoder_route};
    const char * names[2] = {"step", "mimi"};
    std::printf("  memory (MiB; each stage's phase over the footprint just before it began, steady with both ready "
                "over the footprint just before the first load, whole run over the warm ggml baseline %.1f; each "
                "phase needs 1.25 x its measurement within its charge, the whole run its measurement):\n",
                mib(base));
    for (int i = 0; i < 2; ++i) {
        std::printf("    %-5s prepare %7.1f  charged %7.1f   load+warm %7.1f  charged %7.1f  (%s; generate %.1f, "
                    "compile %.1f, load %.1f, warm %.1f)\n",
                    names[i], mib(per[i]->prepare), mib(routes[i]->prepare_peak_bytes), mib(per[i]->load_warm),
                    mib(routes[i]->load_peak_bytes), routes[i]->compute_units.c_str(), mib(per[i]->generate),
                    mib(per[i]->compile), mib(per[i]->load), mib(per[i]->warm));
    }
    std::printf("    steady  %7.1f  charged %7.1f (both resident parts)\n", mib(m.both.steady), mib(resident));
    std::printf("    whole   %7.1f  charged %7.1f (envelope)\n", mib(m.both.total), mib(envelope));
    std::printf("    ANECompilerService while the decoder loaded: peak %.1f, reserve %.1f; growth %.1f over %.1f "
                "before the load (a %s baseline)\n",
                mib(ane.peak), mib(decoder_route.ane_compile_reserve_bytes), mib(ane.growth), mib(ane.before),
                qm::ane_baseline_name(baseline));
    std::vector<std::string> failures = qm::probe_failures(base_ok, footprint_failures, decoder_ane, ane);
    for (const std::string & f : pocket_mem_failures(
             m, {step_route.prepare_peak_bytes, step_route.load_peak_bytes, step_route.ane_compile_reserve_bytes},
             {decoder_route.prepare_peak_bytes, decoder_route.load_peak_bytes, decoder_route.ane_compile_reserve_bytes},
             ane.peak, resident, envelope))
        failures.push_back(f);
    if (coreml_calls < 3 || ggml_calls < 1)
        failures.push_back("expected ggml calls while pending, then at least 3 Core ML calls (got " +
                           std::to_string(ggml_calls) + " and " + std::to_string(coreml_calls) + ")");
    std::printf("  memory report: {\"chip\":\"%s\",\"base\":%llu,\"step_prepare\":%llu,\"step_load_warm\":%llu,"
                "\"mimi_prepare\":%llu,\"mimi_load_warm\":%llu,\"steady\":%llu,\"total\":%llu,"
                "\"ane_compiler_before_load\":%llu,\"ane_compiler_growth\":%llu,\"ane_compiler_peak\":%llu,"
                "\"ane_compiler_baseline\":\"%s\",\"samples\":%zu,\"ane_samples\":%zu,\"ggml_calls\":%d,"
                "\"coreml_calls\":%d,\"seconds\":%.1f}\n",
                host_chip().c_str(), (unsigned long long) base, (unsigned long long) m.step.prepare,
                (unsigned long long) m.step.load_warm, (unsigned long long) m.mimi.prepare,
                (unsigned long long) m.mimi.load_warm, (unsigned long long) m.both.steady,
                (unsigned long long) m.both.total, (unsigned long long) ane.before, (unsigned long long) ane.growth,
                (unsigned long long) ane.peak, qm::ane_baseline_name(baseline), samples.size(), ane.loading_samples,
                ggml_calls, coreml_calls, secs);
    for (const auto & f : failures) std::printf("  [FAIL] memory schedule: %s\n", f.c_str());
    require(failures.empty(), "memory schedule: a phase exceeded its charge or was not measured");
}

int generated_fixture(const std::string & bundle, const std::string & cache_dir, const std::string & memory_dir) {
    env_guard units(pd::COREML_UNITS_ENV), strict(pd::COREML_STRICT_ENV), disabled(pd::COREML_DISABLE_ENV);
    unsetenv(pd::COREML_UNITS_ENV);
    unsetenv(pd::COREML_STRICT_ENV);
    unsetenv(pd::COREML_DISABLE_ENV);
    const EngineOptions plain = bundle_options(bundle);
    require(!tts_cpp::detail::coreml_sidecar_exists(pd::coreml_step_path(plain.flow_lm_path)) &&
                !tts_cpp::detail::coreml_sidecar_exists(pd::coreml_decoder_path(plain.mimi_path)),
            "the bundle must have no prebuilt sidecar (the generated route needs none)");
    if (!memory_dir.empty()) {
        memory_schedule(plain, memory_dir);
        std::puts("test-pocket-coreml --generated --memory-schedule: ok");
        return 0;
    }
    EngineOptions opts = plain;
    opts.coreml.cache_dir = cache_dir;
    const auto co = pd::coreml_engine_options(opts);
    const auto step_route = pd::coreml_plan_route(pd::CoremlStage::step, co);
    const auto decoder_route = pd::coreml_plan_route(pd::CoremlStage::decoder, co);
    std::printf("[routes] step %s %s (%s) %s; mimi %s %s (%s) %s\n", step_route.route.c_str(),
                step_route.compute_units.c_str(), step_route.validated ? "validated" : "unvalidated",
                step_route.reason.c_str(), decoder_route.route.c_str(), decoder_route.compute_units.c_str(),
                decoder_route.validated ? "validated" : "unvalidated", decoder_route.reason.c_str());
    require(step_route.route == "generate" && step_route.validated && step_route.compute_units == "cpu_and_gpu",
            "step route generate, validated, cpu_and_gpu");
    require(decoder_route.route == "generate" && decoder_route.validated && decoder_route.compute_units == "cpu_and_ane",
            "decoder route generate, validated, cpu_and_ane");

    const std::string text = "The quick brown fox jumps over the lazy dog.";
    tts_cpp::pocket::FitOptions fit;
    fit.engine = opts;
    fit.text = text;
    const auto fit_before = tts_cpp::pocket::fit_params(fit);
    const uint64_t envelope = pd::coreml_envelope_bytes(step_route, decoder_route);
    std::printf("[fit] %s, Core ML envelope %llu bytes (%.1f MiB)%s\n", fit_before.reason.c_str(),
                (unsigned long long) fit_before.coreml.bytes, mib(fit_before.coreml.bytes),
                fit_before.coreml.validated ? ", validated" : "");
    require(fit_before.status == tts_cpp::FitStatus::Success && fit_before.coreml.route == "generate" &&
                fit_before.coreml.bytes == envelope && envelope > 0 && fit_before.coreml.validated &&
                fit_before.coreml.disabled_reason.empty(),
            "fit charges the generated envelope: " + fit_before.report);

    // Core ML off: the reference every pending call must equal.
    EngineOptions off = opts;
    off.coreml.use = CoremlOptions::Use::off;
    Engine ggml(off);
    const auto reference = ggml.synthesize(text);
    require(reference.flow_lm_backend == "ggml" && reference.mimi_backend == "ggml" && !reference.coreml_pending,
            "use = off runs ggml");

    // Readiness pinned to "not yet": nothing is requested before the first
    // call (prewarm none) and a call never waits for an unhinted load (wait
    // none), so the first single-chunk call runs on ggml and matches Core ML
    // off.
    const bool cold = !fs::exists(fs::path(cache_dir) / "qvac-coreml-v1" / "artifacts");
    if (cold) {
        EngineOptions pending = opts;
        pending.coreml.prewarm = CoremlOptions::Prewarm::none;
        pending.coreml.wait = CoremlOptions::Wait::none;
        Engine engine(pending);
        const Json idle = status_of(engine);
        require(stage_status(idle, "step").at("state") == "idle" && stage_status(idle, "mimi").at("state") == "idle",
                "prewarm none: both stages idle after construction: " + idle.dump());
        require(!engine.coreml_wait(0), "nothing ready yet");
        const auto first = engine.synthesize(text);
        std::printf("[pending] first call: FlowLM %s, Mimi %s, pending %d\n", first.flow_lm_backend.c_str(),
                    first.mimi_backend.c_str(), int(first.coreml_pending));
        require(first.flow_lm_backend == "ggml" && first.mimi_backend == "ggml" && first.coreml_pending,
                "the first call runs on ggml while the sidecars are pending");
        // The CPU ggml path differs between processes and instances only in
        // float rounding (about 2e-7), so "the same as Core ML off" is a
        // tolerance, far below what one Core ML frame changes.
        require(first.pcm.size() == reference.pcm.size() && max_error(first.pcm, reference.pcm) < 1e-4,
                "a pending call matches Core ML off");
        {
            // The sidecars are still being generated. Waiter A blocks; cancel()
            // must end it even though waiter B starts right after the cancel
            // (a single shared flag that B cleared on entry would leave A
            // waiting). B keeps waiting until the sidecars are ready.
            std::atomic<int> a_result{-1};
            std::thread a([&] { a_result.store(engine.coreml_wait(-1) ? 1 : 0); });
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            require(a_result.load() == -1 && !engine.coreml_wait(0),
                    "the sidecars are still pending when waiter A blocks: " + engine.coreml_status());
            engine.cancel();
            std::atomic<int> b_result{-1};
            std::thread b([&] { b_result.store(engine.coreml_wait(-1) ? 1 : 0); });
            const auto cancelled_at = std::chrono::steady_clock::now();
            while (a_result.load() == -1 && std::chrono::steady_clock::now() - cancelled_at < std::chrono::seconds(2))
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            const bool pending_after_cancel = !engine.coreml_wait(0);
            a.join();
            b.join();
            std::printf("[wait] waiter A after cancel: %d (still pending: %d); waiter B: %d\n", a_result.load(),
                        int(pending_after_cancel), b_result.load());
            require(a_result.load() == 0 && pending_after_cancel,
                    "cancel() ends a blocking wait even when another wait starts after it");
            require(b_result.load() == 1, "a wait started after the cancel waits for readiness");
        }
        require(engine.coreml_wait(-1), "the first call's request makes both sidecars ready: " + engine.coreml_status());
        const Json ready = status_of(engine);
        require(stage_status(ready, "step").at("route") == "generated" &&
                    stage_status(ready, "mimi").at("route") == "generated",
                "a cold cache generates both: " + ready.dump());
        const auto second = engine.synthesize(text);
        require(second.flow_lm_backend == "coreml-gpu" && second.mimi_backend == "coreml-ane" && !second.coreml_pending,
                "once ready, a call runs both stages on Core ML: " + second.flow_lm_backend + " / " + second.mimi_backend);
    } else {
        std::puts("[pending] skipped: the cache already holds the artifacts (use a fresh --cache-dir)");
    }

    // Readiness pinned to "ready" at construction (wait construct): every
    // call runs on Core ML; a second Engine adopts the cached artifacts.
    EngineOptions waited = opts;
    waited.coreml.wait = CoremlOptions::Wait::construct;
    Engine engine(waited);
    require(engine.flow_lm_on_coreml() && engine.mimi_on_coreml(), "wait construct: both ready after construction");
    const Json st = status_of(engine);
    const std::string step_path = stage_status(st, "step").at("path");
    const std::string decoder_path = stage_status(st, "mimi").at("path");
    require(stage_status(st, "step").at("route") == "cached" && stage_status(st, "mimi").at("route") == "cached",
            "a second Engine adopts the cached artifacts: " + st.dump());
    const auto batch = engine.synthesize(text);
    std::printf("[engine] %s / %s: %d frames; ggml %d frames\n", batch.flow_lm_backend.c_str(), batch.mimi_backend.c_str(),
                batch.generated_frames, reference.generated_frames);
    require(batch.flow_lm_backend == "coreml-gpu" && batch.mimi_backend == "coreml-ane", "Core ML labels");
    require(std::abs(batch.generated_frames - reference.generated_frames) <= 4, "Core ML changes the speech length");
    std::vector<float> streamed;
    engine.synthesize_stream(text, [&](const float * pcm, size_t n, int) { streamed.insert(streamed.end(), pcm, pcm + n); return true; });
    require(max_error(streamed, batch.pcm) < 1e-4, "stream and batch differ on the generated sidecars");
    {
        const Json construction = Json::parse(engine.coreml_construction());
        require(construction.at("success") == true && construction.at("nothing_to_prepare") == false &&
                    construction.at("slots").size() == 2,
                "wait construct: the construction group succeeded with both sidecars: " + construction.dump());
        // prewarm none: an empty construction group (which succeeds), nothing
        // ready; a later prewarm request prepares both and reports them.
        EngineOptions later = opts;
        later.coreml.prewarm = CoremlOptions::Prewarm::none;
        Engine idle_engine(later);
        const Json empty = Json::parse(idle_engine.coreml_construction());
        require(empty.at("success") == true && empty.at("nothing_to_prepare") == true && !idle_engine.coreml_wait(0),
                "prewarm none: an empty construction group, nothing ready: " + empty.dump());
        const Json group = Json::parse(idle_engine.coreml_prewarm(CoremlOptions::Prewarm::all, true));
        std::printf("[prewarm] after construction: %s\n", group.dump().substr(0, 200).c_str());
        require(group.at("group") == "prewarm" && group.at("success") == true && group.at("slots").size() == 2 &&
                    idle_engine.flow_lm_on_coreml() && idle_engine.mimi_on_coreml(),
                "coreml_prewarm(all, wait) after construction makes both sidecars ready: " + group.dump());
        // Core ML off: both groups are empty and succeed.
        const Json off_construction = Json::parse(ggml.coreml_construction());
        const Json off_prewarm = Json::parse(ggml.coreml_prewarm(CoremlOptions::Prewarm::all, true));
        require(off_construction.at("nothing_to_prepare") == true && off_construction.at("success") == true &&
                    off_prewarm.at("nothing_to_prepare") == true && off_prewarm.at("success") == true &&
                    off_prewarm.at("skipped_route_none").size() == 2,
                "use = off: empty groups that succeed, both stages listed as route none: " + off_prewarm.dump());
    }

    // Parity of the generated artifacts against ggml (the prebuilt sidecars'
    // gates, on the generated bundles).
    check_stage_parity(plain, step_path);
    check_decoder_parity(plain, decoder_path);
    source_changed_gate(plain);
    capacity_gate(plain);

    // Fit is the same with a populated cache.
    const auto fit_after = tts_cpp::pocket::fit_params(fit);
    require(fit_after.coreml.route == fit_before.coreml.route && fit_after.coreml.bytes == fit_before.coreml.bytes &&
                fit_after.coreml.validated == fit_before.coreml.validated &&
                fit_after.device.total_bytes == fit_before.device.total_bytes && fit_after.host_bytes == fit_before.host_bytes,
            "fit is unchanged by a populated cache");

    // Strict: use = require serves a call on the cached sidecars.
    EngineOptions strict_opts = opts;
    strict_opts.coreml.use = CoremlOptions::Use::require;
    Engine strict_engine(strict_opts);
    const auto strict_result = strict_engine.synthesize(text);
    require(strict_result.flow_lm_backend == "coreml-gpu" && strict_result.mimi_backend == "coreml-ane",
            "strict waits for and runs the sidecars");
    // Memory-degrade: a budget only the ggml projection fits drops Core ML.
    EngineOptions degrade = opts;
    degrade.coreml.wait = CoremlOptions::Wait::construct;
    {
        tts_cpp::pocket::FitOptions f = fit;
        f.engine.coreml.memory_budget_bytes = int64_t(1) << 50;
        const auto probe = tts_cpp::pocket::fit_params(f);
        const uint64_t ggml_required = probe.device.total_bytes + (probe.device_shares_host_memory ? probe.host_bytes : 0) +
                                       f.margin_bytes;
        // A budget between the ggml projection and the Core ML-inclusive one.
        f.engine.coreml.memory_budget_bytes = int64_t(ggml_required + probe.coreml.bytes / 2);
        const auto degraded = tts_cpp::pocket::fit_params(f);
        std::printf("[degrade] budget %.1f MiB: %s, Core ML %s\n", mib(uint64_t(*f.engine.coreml.memory_budget_bytes)),
                    degraded.reason.c_str(), degraded.coreml.disabled_reason.c_str());
        require(degraded.coreml.disabled_reason == "memory" && degraded.fits, "a near budget degrades to ggml in fit");
        degrade.coreml.memory_budget_bytes = f.engine.coreml.memory_budget_bytes;
        degrade.fit_text = text;
    }
    Engine degraded_engine(degrade);
    const Json dst = status_of(degraded_engine);
    require(stage_status(dst, "step").at("state") == "disabled" && stage_status(dst, "step").at("route") == "none" &&
                stage_status(dst, "mimi").at("state") == "disabled" && stage_status(dst, "mimi").at("route") == "none",
            "the Engine resolves use = off under memory-degrade: " + dst.dump());
    require(degraded_engine.synthesize(text).flow_lm_backend == "ggml", "a degraded Engine runs ggml");
    {
        // A fit assessed with a larger headroom than the default: the budget
        // holds ggml + Core ML with the default margin, but only ggml with
        // this one. The Engine, given the same text and margin, must reach
        // that fit's verdict (memory-degrade) and run ggml.
        tts_cpp::pocket::FitOptions wide = fit;
        wide.engine.coreml.memory_budget_bytes = int64_t(1) << 50;
        const auto probe = tts_cpp::pocket::fit_params(wide);
        const uint64_t ggml_required =
            probe.device.total_bytes + (probe.device_shares_host_memory ? probe.host_bytes : 0) + fit.margin_bytes;
        wide.margin_bytes = fit.margin_bytes + probe.coreml.bytes;
        wide.engine.coreml.memory_budget_bytes = int64_t(ggml_required + probe.coreml.bytes + (64ull << 20));
        const auto assessed = tts_cpp::pocket::fit_params(wide);
        tts_cpp::pocket::FitOptions as_default = wide;
        as_default.margin_bytes = fit.margin_bytes;
        const auto by_default = tts_cpp::pocket::fit_params(as_default);
        require(assessed.coreml.disabled_reason == "memory" && by_default.coreml.disabled_reason.empty(),
                "a budget only the larger margin degrades: '" + assessed.coreml.disabled_reason + "' with it, '" +
                    by_default.coreml.disabled_reason + "' with the default");
        EngineOptions low = opts;
        low.coreml.wait = CoremlOptions::Wait::construct;
        low.coreml.memory_budget_bytes = wide.engine.coreml.memory_budget_bytes;
        low.fit_text = text;
        low.fit_margin_bytes = wide.margin_bytes;
        Engine engine_low(low);
        const Json lst = status_of(engine_low);
        const auto low_result = engine_low.synthesize(text);
        std::printf("[degrade] margin %.1f MiB: Engine FlowLM %s, Mimi %s\n", mib(wide.margin_bytes),
                    low_result.flow_lm_backend.c_str(), low_result.mimi_backend.c_str());
        require(stage_status(lst, "step").at("state") == "disabled" &&
                    stage_status(lst, "mimi").at("state") == "disabled" && low_result.flow_lm_backend == "ggml" &&
                    low_result.mimi_backend == "ggml",
                "Engine with the fit's nondefault margin: memory-degrade as that fit (" + low_result.flow_lm_backend +
                    " / " + low_result.mimi_backend + "), " + lst.dump());
    }
    std::puts("test-pocket-coreml --generated: ok");
    return 0;
}
#endif

// Budgets below the ggml projection, between it and the Core ML-inclusive
// one, and above both, through the public fit (each budget source, and both
// with the smaller winning) and, with `cli`, through pocket-cli --fit. One
// ceiling holds every verdict: below refuses, between degrades, above keeps
// Core ML (when the build generates).
int fit_budgets(const fs::path & bundle, const std::string & cli) {
    EngineOptions e = bundle_options(bundle);
    const fs::path cache = fs::temp_directory_path() / ("pocket-fit-budgets-" + test_process_tag());
    e.coreml.cache_dir = cache.string();
    tts_cpp::pocket::FitOptions f;
    f.engine = e;
    f.text = "The quick brown fox jumps over the lazy dog.";
    const auto probe = tts_cpp::pocket::fit_params(f);
    require(probe.status != tts_cpp::FitStatus::Error, "fit of the bundle: " + probe.report);
    if (!probe.device_shares_host_memory) {
        std::puts("SKIP: the selected device does not share host memory");
        return SKIP;
    }
    const uint64_t MIB = 1ull << 20;
    const uint64_t required = probe.device.total_bytes + probe.host_bytes + f.margin_bytes;
    const uint64_t envelope = probe.coreml.route == "generate" ? probe.coreml.bytes : 0;
    std::printf("[budgets] ggml %.1f MiB, Core ML envelope %.1f MiB (route %s), free %.1f MiB\n",
                double(required) / MIB, double(envelope) / MIB, probe.coreml.route.c_str(),
                double(probe.device_free_bytes) / MIB);
    const uint64_t below = required - 16 * MIB, between = required + envelope / 2,
                   above = required + envelope + 64 * MIB;
    if (probe.device_free_bytes < above + 256 * MIB) {
        std::puts("SKIP: not enough free memory for the budget above both projections");
        return SKIP;
    }
    struct Case {
        const char * name;
        uint64_t budget;
        bool fits;
        bool degraded;
    };
    std::vector<Case> cases = {{"below", below, false, false}, {"above", above, true, false}};
    if (envelope) cases.push_back({"between", between, true, true});
    for (const Case & c : cases) {
        for (int source = 0; source < 4; ++source) {
            tts_cpp::pocket::FitOptions g = f;
            const char * label = "";
            if (source == 0) { g.memory_budget_bytes = c.budget; label = "fit budget"; }
            if (source == 1) { g.engine.coreml.memory_budget_bytes = int64_t(c.budget); label = "Core ML budget"; }
            if (source == 2) {
                g.memory_budget_bytes = c.budget;
                g.engine.coreml.memory_budget_bytes = int64_t(c.budget);
                label = "both";
            }
            if (source == 3) {
                g.memory_budget_bytes = c.budget;
                g.engine.coreml.memory_budget_bytes = int64_t(c.budget + envelope + 1024 * MIB);
                label = "both, fit's smaller";
            }
            const auto r = tts_cpp::pocket::fit_params(g);
            const bool degraded = r.coreml.disabled_reason == "memory";
            std::printf("[budgets] %s (%s): %s%s\n", c.name, label, r.reason.c_str(), degraded ? ", memory-degrade" : "");
            require(r.fits == c.fits && degraded == c.degraded && r.device_free_bytes <= c.budget,
                    std::string("budget ") + c.name + " via " + label + ": " + r.report);
        }
    }
#ifndef _WIN32
    if (!cli.empty()) {
        const fs::path report = cache.string() + "-report.json";
        for (const Case & c : cases) {
            const uint64_t budget_mib = c.name == std::string("below") ? required / MIB - 16
                                        : c.name == std::string("between") ? (required + envelope / 2) / MIB
                                                                           : (required + envelope) / MIB + 64;
            const std::string cmd = "'" + cli + "' --fit --model-dir '" + bundle.string() + "' --text '" + f.text +
                                    "' --coreml-cache-dir '" + cache.string() + "' --fit-budget-mib " +
                                    std::to_string(budget_mib) + " --report '" + report.string() + "' > /dev/null";
            const int status = std::system(cmd.c_str());
            const int code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
            const nlohmann::json j = nlohmann::json::parse(std::ifstream(report));
            const bool degraded = j.at("coreml").at("disabled_reason") == "memory";
            std::printf("[budgets] pocket-cli %s (%llu MiB): exit %d%s\n", c.name, (unsigned long long) budget_mib, code,
                        degraded ? ", memory-degrade" : "");
            require(code == (c.fits ? 0 : 1) && j.at("fits") == c.fits && degraded == c.degraded,
                    std::string("pocket-cli --fit budget ") + c.name + ": " + j.dump());
        }
        fs::remove(report);
    }
#else
    (void) cli;
#endif
    std::error_code ec;
    fs::remove_all(cache, ec);
    std::puts("test-pocket-coreml --fit-budgets: ok");
    return 0;
}

#ifndef _WIN32
// Runs `cmd` through the shell; its exit code, or -1.
int run_command(const std::string & cmd) {
    const int status = std::system(cmd.c_str());
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

std::string file_text(const fs::path & p) {
    std::ifstream f(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}
#endif

// pocket-cli's paths, on Core ML off (POCKET_COREML_DISABLE, so only ggml
// runs): --coreml-status-json is refused before synthesis when it names the
// output, a model input or (through a symlink) an existing output, and after
// synthesis when only the filesystem equates it with the output (a
// case-insensitive volume), the audio then kept. The cache directory: the
// explicit option, else QVAC_COREML_CACHE_DIR, else Application Support under
// HOME.
int cli_paths(const fs::path & bundle, const std::string & cli) {
#ifdef _WIN32
    (void) bundle;
    (void) cli;
    std::puts("SKIP: POSIX shell needed");
    return SKIP;
#else
    const fs::path dir = fs::temp_directory_path() / ("pocket-cli-paths-" + test_process_tag());
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir / "bundle");
    for (const char * f : {"flow-lm.gguf", "mimi.gguf", "frontend.json"})
        fs::create_symlink(fs::absolute(bundle / f), dir / "bundle" / f);
    fs::copy_file(bundle / "voice.gguf", dir / "bundle" / "voice.gguf");
    const std::string voice_before = file_text(dir / "bundle" / "voice.gguf");
    // `env_set`: extra NAME=VALUE assignments (QVAC_COREML_CACHE_DIR is
    // unset first, so only an assignment here sets it).
    auto command = [&](const std::string & env_set) {
        return "env -u QVAC_COREML_CACHE_DIR " + env_set + " POCKET_COREML_DISABLE=1 HOME='" +
               (dir / "home").string() + "' '" + cli + "' --model-dir '" + (dir / "bundle").string() +
               "' --text 'Hi.'";
    };
    const std::string base = command("");
    const fs::path out = dir / "out.wav", log = dir / "log.txt";
    auto run = [&](const std::string & extra) {
        const int code = run_command(base + " " + extra + " > '" + log.string() + "' 2>&1");
        return std::make_pair(code, file_text(log));
    };
    // The same path as the output: refused before synthesis, nothing written.
    auto r = run("--output '" + out.string() + "' --coreml-status-json '" + out.string() + "'");
    require(r.first == 1 && r.second.find("same file as --output") != std::string::npos && !fs::exists(out),
            "status = output is refused before synthesis: " + r.second);
    // A model input: refused, the input untouched.
    r = run("--output '" + out.string() + "' --coreml-status-json '" + (dir / "bundle" / "voice.gguf").string() + "'");
    require(r.first == 1 && r.second.find("same file as --voice") != std::string::npos &&
                file_text(dir / "bundle" / "voice.gguf") == voice_before,
            "status = a model input is refused: " + r.second);
    // Through a symlink to an existing output: refused, the output untouched.
    std::ofstream(out, std::ios::binary) << "keep";
    fs::create_symlink(out, dir / "status-link.json");
    r = run("--output '" + out.string() + "' --coreml-status-json '" + (dir / "status-link.json").string() + "'");
    require(r.first == 1 && r.second.find("same file as --output") != std::string::npos && file_text(out) == "keep",
            "status through a symlink to the output is refused: " + r.second);
    fs::remove(out);
    // Another spelling of the same file on a case-insensitive volume: caught
    // once the WAV exists; the audio is kept and the run fails.
    std::ofstream(dir / "case-probe", std::ios::binary) << "x";
    if (fs::exists(dir / "CASE-PROBE")) {
        r = run("--output '" + out.string() + "' --coreml-status-json '" + (dir / "OUT.WAV").string() + "'");
        require(r.first == 1 && r.second.find("same file as --output") != std::string::npos &&
                    file_text(out).compare(0, 4, "RIFF") == 0,
                "status in another case of the output is refused after synthesis, the WAV kept: " + r.second);
        std::puts("[cli] case-insensitive alias: refused, audio kept");
    } else {
        std::puts("[cli] case-sensitive volume: no case alias to check");
    }
#ifdef TTS_CPP_USE_COREML
    // The cache directory each way.
    const fs::path status = dir / "status.json";
    auto cache_dir = [&](const std::string & env, const std::string & extra) {
        const int code = run_command(command(env) + " " + extra + " --output '" + out.string() +
                                     "' --coreml-status-json '" + status.string() + "' > '" + log.string() + "' 2>&1");
        require(code == 0, "pocket-cli run: " + file_text(log));
        return nlohmann::json::parse(file_text(status)).at("cache_dir").get<std::string>();
    };
    const std::string flag = (dir / "flag-cache").string(), shared = (dir / "shared-cache").string();
    const std::string by_flag = cache_dir("QVAC_COREML_CACHE_DIR='" + shared + "'", "--coreml-cache-dir '" + flag + "'");
    const std::string by_env = cache_dir("QVAC_COREML_CACHE_DIR='" + shared + "'", "");
    const std::string by_default = cache_dir("", "");
    std::printf("[cli] cache dir: flag %s; env %s; default %s\n", by_flag.c_str(), by_env.c_str(), by_default.c_str());
    require(by_flag == flag && by_env == shared &&
                by_default == (dir / "home" / "Library" / "Application Support" / "qvac-speech" / "coreml").string(),
            "pocket-cli cache dir: the option, then QVAC_COREML_CACHE_DIR, then Application Support");
#endif
    // The library's default stays "no cache".
    EngineOptions plain = bundle_options(bundle);
    env_guard shared_env("QVAC_COREML_CACHE_DIR");
    unsetenv("QVAC_COREML_CACHE_DIR");
    require(pd::coreml_plan_route(pd::CoremlStage::step, pd::coreml_engine_options(plain)).route != "generate",
            "the library does not generate without a cache directory");
    fs::remove_all(dir, ec);
    std::puts("test-pocket-coreml --cli-paths: ok");
    return 0;
#endif
}

int engine_fixture() {
#ifndef TTS_CPP_USE_COREML
    std::puts("SKIP: built without Core ML (TTS_CPP_COREML=OFF)");
    return SKIP;
#else
    const char * dir = std::getenv("POCKET_COREML_TEST_BUNDLE");
    if (dir == nullptr || !*dir) {
        std::puts("SKIP: set POCKET_COREML_TEST_BUNDLE to a Pocket bundle with flow-lm-step.mlmodelc");
        return SKIP;
    }
    const auto opts = bundle_options(dir);
    require(tts_cpp::detail::coreml_sidecar_exists(pd::coreml_step_path(opts.flow_lm_path)), "bundle has no step sidecar");
    require(tts_cpp::detail::coreml_sidecar_exists(pd::coreml_decoder_path(opts.mimi_path)), "bundle has no Mimi decoder sidecar");
    check_stage_parity(opts);
    check_decoder_parity(opts);
    check_replay(opts);
    check_engine(opts);
    std::puts("test-pocket-coreml --engine-fixture: ok");
    return 0;
#endif
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc > 1 && std::string(argv[1]) == "--shape-fixture") return shape_fixture();
        if (argc > 1 && std::string(argv[1]) == "--engine-fixture") return engine_fixture();
        if (argc > 4 && std::string(argv[1]) == "--cli-paths" && std::string(argv[3]) == "--cli")
            return cli_paths(argv[2], argv[4]);
        if (argc > 2 && std::string(argv[1]) == "--fit-budgets")
            return fit_budgets(argv[2], argc > 4 && std::string(argv[3]) == "--cli" ? argv[4] : "");
        if (argc > 1 && std::string(argv[1]) == "--generated") {
#ifndef TTS_CPP_USE_COREML
            std::puts("SKIP: built without Core ML (TTS_CPP_COREML=OFF)");
            return SKIP;
#else
            std::string bundle, cache, memory;
            for (int i = 2; i < argc; ++i) {
                const std::string a = argv[i];
                if (a == "--cache-dir" && i + 1 < argc) cache = argv[++i];
                else if (a == "--memory-schedule" && i + 1 < argc) memory = argv[++i];
                else if (bundle.empty()) bundle = a;
                else throw std::runtime_error("unknown argument " + a);
            }
            if (bundle.empty() || (cache.empty() && memory.empty()))
                throw std::runtime_error("--generated BUNDLE --cache-dir DIR | --memory-schedule DIR");
            return generated_fixture(bundle, cache, memory);
#endif
        }
        check_policy();
        check_coreml_options();
        check_naming();
        check_rope();
        check_detection_and_fit();
        check_memory_schedule_attribution();
        std::puts("test-pocket-coreml: ok");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "test-pocket-coreml: FAIL: %s\n", e.what());
        return 1;
    }
}
