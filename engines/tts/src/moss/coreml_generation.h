#pragma once

#include "moss/coreml_sidecar.h"
#include "tts-cpp/coreml_options.h"
#include "tts-cpp/fit.h"

#include <atomic>
#include <optional>

namespace tts_cpp::moss::detail {

enum class GeneratedStage { speech_tokenizer, sfx_vae };

struct GeneratedModel {
    virtual ~GeneratedModel() = default;
    virtual bool predict(const std::vector<float> & input,
                         std::vector<float> & output) = 0;
    virtual const char * label() const = 0;
    virtual void retire() = 0;
};

// Legacy bundles still open synchronously. This object owns only the
// generated slot and never gives a worker a pointer to engine state.
class MossCoremlGeneration {
public:
    MossCoremlGeneration(GeneratedStage stage, const std::string & gguf,
                         const CoremlOptions & options);
    ~MossCoremlGeneration();

    CoremlPolicy prebuilt_policy() const;
    bool can_generate() const;
    void start(const std::vector<int64_t> & input,
               const std::vector<int64_t> & output,
               std::optional<uint64_t> ggml_bytes = std::nullopt);
    std::unique_ptr<GeneratedModel>
    acquire(const std::atomic<bool> * cancel = nullptr);
    bool ready() const;
    // The generated slot's state ("idle" before start or without a slot).
    std::string state() const;
    std::string status_json(bool prebuilt_ready) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// The generated route of one stage and the template charges behind its Core
// ML envelope, before any slot exists. Reads the options, the build, the OS,
// the template match and an adjacent bundle's presence; never the cache, so it
// is the same before and after a generation. Charges are 0 unless `route` is
// "generate"; `envelope_bytes` includes the template margin.
struct GeneratedRoute {
    std::string template_id;
    std::string route;          // prebuilt | generate | none
    std::string state;          // the status label of a `none` route
    std::string compute_units;  // resolved units
    bool validated = false;     // the template's memory is validated for the units
    uint64_t envelope_bytes = 0;
    uint64_t resident_bytes = 0;
    uint64_t workspace_bytes = 0;
    uint64_t prepare_peak_bytes = 0;
    uint64_t load_peak_bytes = 0;
    uint64_t ane_compile_reserve_bytes = 0;
};

GeneratedRoute plan_generated_route(GeneratedStage stage,
                                    const std::string & gguf,
                                    const CoremlOptions & options);

// Pure route and budget calculation. No cache or live slot state is read.
FitCoreml sfx_vae_coreml_fit(const std::string & gguf,
                             const CoremlOptions & options, uint64_t ggml_bytes,
                             uint64_t budget, bool ggml_fits);

} // namespace tts_cpp::moss::detail
