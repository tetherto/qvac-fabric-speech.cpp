#pragma once

// Parakeet's adoption of the shared Core ML runtime (engines/coreml, PLAN
// §3.2.0, §2.9-§2.12, §2.15).
//
// Every background-loaded sidecar of one model gets a slot: the encoder, the
// Sortformer bypass encoder, and one per diarizer capacity. Slots load on the
// runtime's load lane, in priority order, with memory admission for cold ANE
// compiles; until a slot is ready its stage runs on ggml. The TDT decoder
// bundle and an encoder that owns the encoder (encoder-less GGUF or
// PARAKEET_COREML_ONLY) stay synchronous (§2.15) and are not slots.
//
// Portable C++ against the runtime's namespace (QVAC_COREML_NS): no Core ML or
// Objective-C here, so the routing glue builds and is tested on every
// platform with fake loaders.

#include "encoder_route.h"
#include "parakeet/coreml_options.h"
#include "qvac_coreml/manager.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace parakeet {

namespace pcm = QVAC_COREML_NS;

// B.6 public options -> the runtime's internal mirror (field by field).
pcm::Options to_runtime_options(const CoremlOptions & opts);

// Resolves once (§2.12) with Parakeet's environment prefix (PARAKEET_COREML_*).
pcm::Resolved resolve_parakeet_coreml(const CoremlOptions & opts);

pcm::Prewarm to_runtime_prewarm(CoremlOptions::Prewarm scope);

// The resolved compute units as the argument the synchronous loaders take
// (the TDT decoder bundle and the encoder owner, §2.15): their B.5 name, or
// null when nothing was set, so the loader keeps its own stage default.
// Loaders never read the environment themselves (§2.12).
inline const char * loader_units(const pcm::Resolved & resolved) {
    return resolved.compute_units ? pcm::to_string(*resolved.compute_units) : nullptr;
}

// Operations on a loaded encoder model (the SharedModel payload). Plain
// function pointers, so tests substitute fakes for the Core ML calls.
struct EncoderModelOps {
    parakeet_coreml_context * (*context)(void * model) = nullptr;
    EncoderRouteGeometry (*describe)(parakeet_coreml_context * ctx, int n_mels) = nullptr;
    const char * (*label)(parakeet_coreml_context * ctx) = nullptr;
    EncoderRoute::EncodeFn encode = nullptr;
};

// The unit's route from one acquisition: a handle gives a Core ML route whose
// failure retires the model process-wide; no handle gives the ggml route,
// marked pending when the slot was not ready yet.
EncoderRoute route_from_acquired(pcm::Acquired && acquired, const EncoderModelOps & ops, int n_mels);

// Bytes ANECompilerService needs for a cold compile of `mlmodelc` (B.1
// default: 1.6 x weights/weight.bin), or 0 when `units` cannot use the ANE.
uint64_t ane_compile_reserve_bytes(const std::string & mlmodelc, pcm::ComputeUnits units);

// One sidecar to register (route from plan(): prebuilt when present).
struct SidecarRequest {
    std::string stage;          // status "stage": encoder, bypass-encoder, diarizer-<frames>
    std::string path;           // adjacent .mlmodelc
    bool present = false;       // the directory exists (today's path rules)
    pcm::ComputeUnits stage_default = pcm::ComputeUnits::all;
    pcm::LoadFn load_fn = nullptr;
    std::vector<int64_t> geometry;  // validated by the loader (diarizer capacity)
    bool minimal = false;       // part of the minimal (construction) set (§2.11)
};

// The sidecar slots of one loaded model.
class CoremlSidecars {
public:
    // `manager` null uses the library image's manager. `memory_degraded`:
    // the Core ML-inclusive fit failed where the ggml-only one passed, so
    // the instance resolves use = off with the reason "memory" (§2.13).
    explicit CoremlSidecars(const CoremlOptions & opts,
                            std::shared_ptr<pcm::Manager> manager = nullptr,
                            bool memory_degraded = false);
    CoremlSidecars(const CoremlSidecars &) = delete;
    CoremlSidecars & operator=(const CoremlSidecars &) = delete;
    // Drops the slots; never waits (§2.9).
    ~CoremlSidecars();

    const pcm::Resolved & resolved() const { return resolved_; }
    pcm::Manager & manager() const { return *manager_; }

    // Registers a slot. Its route comes from plan(): `prebuilt` when the
    // bundle is present and Core ML is enabled, otherwise `none` (disabled,
    // absent), reported in status and listed as skipped in groups.
    std::shared_ptr<pcm::Slot> add(const SidecarRequest & request);

    // Issues the construction group for the resolved prewarm scope; with
    // wait = construct, blocks until it finished. Call once, after every add().
    void start();

    // Another prewarm group (Engine::coreml_prewarm).
    pcm::PrewarmResult prewarm(CoremlOptions::Prewarm scope, bool wait,
                               const std::atomic<bool> * cancel = nullptr);

    // Waits for the construction group (timeout < 0: no limit). True when
    // every slot in it reached ready.
    bool wait_construction(int64_t timeout_ms, const std::atomic<bool> * cancel = nullptr);
    std::string construction_json() const;
    std::string status_json() const;

    std::vector<std::shared_ptr<pcm::Slot>> slots() const;

private:
    struct Entry {
        std::shared_ptr<pcm::Slot> slot;
        bool minimal = false;
    };
    std::vector<pcm::GroupRequest> requests(pcm::Prewarm scope, pcm::GroupKind kind,
                                            std::vector<pcm::SlotStatus> & skipped) const;

    pcm::Resolved resolved_;
    std::shared_ptr<pcm::Manager> manager_;
    std::vector<Entry> entries_;
    std::shared_ptr<pcm::Group> construction_;
};

}  // namespace parakeet
