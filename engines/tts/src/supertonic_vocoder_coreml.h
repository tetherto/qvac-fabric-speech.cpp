#pragma once

// The Supertonic vocoder on the shared Core ML runtime.
//
// The Engine loads the vocoder sidecar through the runtime instead of at GGUF
// load: a prebuilt `<gguf stem minus quant tag>-vocoder.mlmodelc` next to the
// GGUF loads in the background; without one, and with a cache directory, the
// sidecar is generated from the GGUF's own vocoder tensors on macOS 15 or
// later. Two templates cover the vocoder of Supertonic 1, 2 and 3:
// supertonic-vocoder-w64 for GGUFs that store the pointwise convolutions as
// [OC, IC, 1] (the f32 and f16 tiers), and supertonic-vocoder-pw2d-w64 for
// the requantised tiers, which store them squeezed to [OC, IC]. Both export
// a 64-latent-frame window, the prebuilt sidecar's default.
//
// Until the sidecar is ready every vocoder call runs on ggml. Each
// supertonic_vocoder_forward_ggml call takes its backend once, at entry, so a
// call never mixes backends; a failed prediction retires the sidecar (for
// every engine on it) and the call is redone on ggml, or fails under
// use = require.
//
// Device class. A generated sidecar is skipped (unsupported(device)) on the
// device classes the template lists in prefer_ggml_on, where the measured
// Neural Engine sidecar is slower than the GPU (today: Max and Ultra parts).
// The class comes from machdep.cpu.brand_string. use = require generates
// anyway. Prebuilt sidecars are not gated: placing one is the deployment's
// choice.
//
// Weight floor. A prebuilt sidecar carries reference-precision weights, so it
// is skipped for a GGUF whose vocoder weights are stored below 8 bits
// (SUPERTONIC_COREML_ALLOW_LOW_BIT overrides). A generated sidecar carries
// the loaded tier's own weights and has no floor.
//
// Environment (resolved once, at construction): SUPERTONIC_COREML_DISABLE
// (use = off), SUPERTONIC_COREML_STRICT (use = require),
// SUPERTONIC_COREML_COMPUTE_UNITS, QVAC_COREML_CACHE_DIR.

#include "tts-cpp/coreml_options.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tts_cpp {
namespace supertonic {
namespace detail {

struct supertonic_model;

constexpr const char * SUPERTONIC_VOCODER_TEMPLATE_W64      = "supertonic-vocoder-w64";
constexpr const char * SUPERTONIC_VOCODER_TEMPLATE_PW2D_W64 = "supertonic-vocoder-pw2d-w64";
constexpr const char * SUPERTONIC_VOCODER_GGML_BACKEND      = "ggml";
constexpr const char * SUPERTONIC_VOCODER_MIXED_BACKEND     = "mixed";

// ---- device class ---------------------------------------------------------------

// The device class of an Apple silicon brand string ("Apple M3 Ultra" ->
// "ultra", "Apple M4 Max" -> "max", "Apple M2 Pro" -> "pro", "Apple M2" ->
// "base"); a suffix such as " (Virtual)" is ignored. "" for anything that is
// not an Apple M-series brand.
std::string supertonic_device_class(const std::string & cpu_brand);
// machdep.cpu.brand_string ("" where it cannot be read).
std::string supertonic_cpu_brand();
// Whether `prefer_ggml_on` (a template's device classes) names the class of
// `cpu_brand`.
bool supertonic_prefers_ggml(const std::vector<std::string> & prefer_ggml_on, const std::string & cpu_brand);

// ---- one vocoder call ---------------------------------------------------------

// A fixed-window vocoder sidecar. Implementations: the Core ML model, and
// test doubles.
class supertonic_vocoder_sidecar {
public:
    virtual ~supertonic_vocoder_sidecar() = default;
    // Latent frames per prediction.
    virtual int window() const = 0;
    // "coreml-all", "coreml-ane", ...: reported as the vocoder backend.
    virtual const char * label() const = 0;
    // supertonic_coreml_vocoder_synthesize's contract: frames [begin, begin +
    // filled) of the channel-major [latent_channels, latent_len] latent fill
    // the window, and frames [skip, skip + keep) of its output are written to
    // wav. False on any failure.
    virtual bool synthesize(const float * latent, int64_t latent_len, int64_t begin, int64_t filled, int64_t skip,
                            int64_t keep, float * wav) = 0;
};

// What one vocoder call runs on, acquired once at its entry.
struct supertonic_vocoder_call {
    std::shared_ptr<supertonic_vocoder_sidecar> sidecar;  // null: this call runs on ggml
    bool pending = false;  // on ggml only because the sidecar was not ready yet
    // A prediction failed: retire the sidecar (for every engine on it).
    std::function<void(const std::string & reason)> retire;
};

// The seam supertonic_vocoder_forward_ggml acquires from (model.vocoder_pending).
// The Engine installs supertonic_vocoder_coreml; tests install fakes.
class supertonic_vocoder_pending {
public:
    virtual ~supertonic_vocoder_pending() = default;
    // Once per vocoder call of `latent_len` frames. A wait for the sidecar
    // returns early once `cancel` (optional) is set. Throws
    // std::runtime_error when use = require and the sidecar cannot serve the
    // call (failed, retired, timed out).
    virtual supertonic_vocoder_call acquire(int latent_len, const std::atomic<bool> * cancel) = 0;
    virtual bool strict() const = 0;
};

// ---- the stage's route --------------------------------------------------------

// Decided from options, OS, build, device class, template match and the
// presence of a prebuilt sidecar, never from the cache, so it is the same
// before and after the first generation. For the generated route it carries
// the Core ML memory envelope fit charges.
struct supertonic_vocoder_route {
    std::string route = "none";  // prebuilt | generate | none
    std::string state;           // when none: unsupported | disabled | absent | no_template
    std::string reason;
    std::string template_id;     // generate: the matched template
    std::string compute_units;   // resolved
    bool strict = false;
    bool validated = false;       // generate: the envelope is validated for compute_units
    uint64_t envelope_bytes = 0;  // generate: (resident + one transient) x margin
    // generate: the envelope's parts before the margin.
    uint64_t resident_bytes = 0;
    uint64_t workspace_bytes = 0;
    uint64_t prepare_peak_bytes = 0;
    uint64_t load_peak_bytes = 0;
    // generate: the free memory a cold Neural Engine load waits for (runtime
    // admission; fit does not charge it). 0 when compute_units has no ANE.
    uint64_t ane_compile_reserve_bytes = 0;
};

// `cpu_brand`: the brand string the device class is read from; empty reads
// this machine's (tests pass a fake one). `prebuilt_below_floor`: the GGUF's
// vocoder weights are stored below the prebuilt sidecar's 8-bit floor, so an
// adjacent prebuilt does not count (supertonic_first_low_bit_vocoder_weight on
// a loaded model). Throws std::invalid_argument for invalid explicit options.
supertonic_vocoder_route supertonic_vocoder_plan_route(const std::string & gguf_path, const CoremlOptions & options,
                                                       const std::string & cpu_brand = {},
                                                       bool prebuilt_below_floor = false);
// Whether a loaded (or metadata-only loaded) model's vocoder weights are
// stored below the prebuilt sidecar's floor, unless
// SUPERTONIC_COREML_ALLOW_LOW_BIT lifts it.
bool supertonic_vocoder_prebuilt_below_floor(const supertonic_model & model);

// The vocoder stage of one Engine on the shared Core ML runtime.
class supertonic_vocoder_coreml final : public supertonic_vocoder_pending {
public:
    // Resolves `options` (with the SUPERTONIC_* environment), plans the route,
    // creates the slot and issues the construction group (prewarm none:
    // nothing until the first call), waiting for it with wait = construct.
    // `memory_degrade` resolves use = off (fit's memory-degrade). Returns null
    // in builds without TTS_CPP_COREML. Throws std::invalid_argument for
    // invalid options and std::runtime_error when use = require cannot be
    // met.
    // `cpu_brand` as in supertonic_vocoder_plan_route.
    static std::shared_ptr<supertonic_vocoder_coreml> create(supertonic_model & model, const std::string & gguf_path,
                                                             const CoremlOptions & options,
                                                             bool memory_degrade = false,
                                                             const std::string & cpu_brand = {});
    ~supertonic_vocoder_coreml() override;

    supertonic_vocoder_call acquire(int latent_len, const std::atomic<bool> * cancel) override;
    bool strict() const override;

    const supertonic_vocoder_route & route() const;
    // The slot's state ("loading", "ready", ...; "" without a slot): cheap
    // enough to poll every millisecond.
    std::string state() const;
    // The status JSON (schema 1: engine, cache_dir, generation, stages).
    std::string status_json() const;
    // Blocks until the vocoder sidecar is ready or terminal, `timeout_ms`
    // passes (< 0: no limit) or `cancel` (optional) is set. True when it is
    // ready. Returns false at once while nothing has requested the sidecar
    // (prewarm none before the first call) or it waits for memory.
    bool wait(int64_t timeout_ms, const std::atomic<bool> * cancel = nullptr) const;

    struct impl;

private:
    explicit supertonic_vocoder_coreml(std::unique_ptr<impl> p);
    std::unique_ptr<impl> p_;
};

// Label merge across the calls of one synthesis: equal labels stay, anything
// else is "mixed" (an empty accumulator takes the next label).
std::string supertonic_merge_vocoder_backend(const std::string & acc, const std::string & next);

}  // namespace detail
}  // namespace supertonic
}  // namespace tts_cpp
