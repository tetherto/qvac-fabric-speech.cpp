#pragma once
#include "parakeet/coreml_options.h"
#include "parakeet/export.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace parakeet::moss {

struct TranscribeOptions {
    std::string model_path;
    int n_threads = 4;
    bool use_gpu = false;
    std::string backends_dir;
    // "auto" follows use_gpu. An explicit backend (e.g. "cuda", "vulkan",
    // "hexagon" or "CUDA0") overrides use_gpu and fails if unavailable.
    std::string backend = "auto";
    // Apple Core ML encoder sidecar (PARAKEET_COREML builds; ignored
    // elsewhere). The audio encoder and adaptor run on Core ML for every 30 s
    // window once the sidecar is ready, and on ggml until then. The sidecar is
    // an adjacent `<stem>-encoder.mlmodelc`, or, without one, generated from
    // the GGUF into `coreml.cache_dir`. It loads in the background; each
    // window takes the sidecar if it is ready when the window starts. The
    // options resolve with the MOSS_COREML_* environment variables
    // (MOSS_COREML_DISABLE, MOSS_COREML_STRICT -> use = require,
    // MOSS_COREML_COMPUTE_UNITS) and QVAC_COREML_CACHE_DIR. See
    // coreml_options.h.
    CoremlOptions coreml;
    // The fit the host assessed this model with (parakeet::moss::fit_params
    // totals). With coreml.memory_budget_bytes set, construction applies
    // memory-degrade from these totals; unset, it projects fit itself for
    // 300 s of audio, the default request and a 256 MiB margin.
    std::optional<CoremlFitTotals> coreml_fit;
};

struct TranscribeRequest {
    std::string prompt;
    std::vector<std::string> hotwords;
    int max_new_tokens = 0;
};

struct TranscriptSegment {
    double start_s = 0;
    double end_s = 0;
    std::string speaker;
    std::string text;
};

struct TranscribeResult {
    std::string text;
    std::vector<TranscriptSegment> segments;
    bool cancelled = false;
    int audio_tokens = 0;
    int prompt_tokens = 0;
    int generated_tokens = 0;
    double encode_ms = 0;
    double prefill_ms = 0;
    double decode_ms = 0;
    // Where this run's windows went: "ggml", the Core ML label
    // ("coreml-all", "coreml-cpu-gpu", "coreml-ane", "coreml-cpu") or
    // "mixed"; empty when no window ran.
    std::string encoder_backend;
    // At least one window ran on ggml because the Core ML sidecar was still
    // being generated or loaded.
    bool coreml_pending = false;
};

using TranscribeProgress = std::function<bool(int generated_tokens, int max_new_tokens)>;

class PARAKEET_API TranscribeEngine {
public:
    explicit TranscribeEngine(const TranscribeOptions & options);
    ~TranscribeEngine();
    TranscribeEngine(const TranscribeEngine &) = delete;
    TranscribeEngine & operator=(const TranscribeEngine &) = delete;
    TranscribeResult transcribe(const float * pcm, size_t samples, int sample_rate,
                                const TranscribeRequest & request = {}, const TranscribeProgress & progress = {});
    void cancel() noexcept;
    int sample_rate() const noexcept;
    const char * backend_name() const noexcept;
    // True when the Core ML encoder sidecar is loaded and ready now. It loads
    // in the background, so this is false while it is generated or loaded.
    bool encoder_on_coreml() const noexcept;
    // Core ML sidecar status as JSON (state, route, compute units, label,
    // timings); "{}" on builds without PARAKEET_COREML.
    std::string coreml_status_json() const;
    // The construction group's result as JSON.
    std::string coreml_construction_json() const;
    // Waits until the construction group has finished, up to `timeout_ms`
    // (< 0: no limit). True when the sidecar reached ready, and at once when
    // there is nothing to prepare.
    bool coreml_wait(int64_t timeout_ms = -1);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace parakeet::moss
