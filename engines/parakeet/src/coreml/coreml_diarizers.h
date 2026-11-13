#pragma once

#include "coreml/parakeet-diarizer.h"
#include "parakeet_coreml_path.h"

#include <memory>
#include <string>
#include <vector>

namespace parakeet {

class CoremlSidecars;

constexpr const char * kCoremlDiarizerUnitsEnv = "PARAKEET_COREML_COMPUTE_UNITS";
constexpr const char * kCoremlDiarizerDisableEnv = "PARAKEET_COREML_DIARIZER_DISABLE";
constexpr const char * kCoremlDisableEnv = "PARAKEET_COREML_DISABLE";

// Per-call tally of diarizer passes, for the result labels.
struct CoremlDiarizerCallStats {
    int coreml = 0;      // passes that ran on a diarizer sidecar
    int not_ready = 0;   // passes that ran on ggml because the capacity was loading
    std::string label;   // Core ML label of the passes that ran on a sidecar
};

// The diarizer sidecars of one model, one runtime slot per capacity.
// Each pass acquires the smallest covering capacity that has
// not failed; while that capacity is still loading, run() returns kNoSidecar
// and the pass runs on ggml. A load or prediction failure retires the
// capacity for the process (quarantine), and later passes use the next
// covering capacity or ggml. Under use = require a pass that a covering
// capacity cannot serve (every covering capacity failed, a timeout, or a
// failed prediction, which still retires the capacity) throws
// pcm::CoremlError instead; a pass longer than every capacity runs on ggml.
class CoremlDiarizers {
public:
    // Registers one slot per capacity with `sidecars`; the smallest capacity
    // is in the minimal (construction) set.
    CoremlDiarizers(std::string gguf_path, std::vector<int> capacities, CoremlSidecars & sidecars);
    ~CoremlDiarizers();
    CoremlDiarizers(const CoremlDiarizers &) = delete;
    CoremlDiarizers & operator=(const CoremlDiarizers &) = delete;

    // A capacity that has not failed covers `frames` (independent of
    // readiness). Under use = require a failed capacity still covers, so
    // run() refuses the pass rather than the caller falling back to ggml.
    bool covers(int frames);
    // Requests the covering capacity now; true when it is ready.
    bool prepare(int frames);
    int run(int valid_frames,
            const parakeet_coreml_rows * inputs, int n_inputs,
            const parakeet_coreml_rows_out * outputs, int n_outputs);
    std::string label();
    std::vector<int> capacities() const;

    // Engine calls are not concurrent (engine.h), so one tally per model.
    void reset_call_stats();
    CoremlDiarizerCallStats call_stats();

    struct State;

private:
    std::unique_ptr<State> state_;
};

bool coreml_diarizers_disabled();

std::vector<CoremlDiarizerBundle> list_coreml_diarizer_bundles(const std::string & gguf_path);

std::unique_ptr<CoremlDiarizers> discover_coreml_diarizers(const std::string & gguf_path,
                                                           CoremlSidecars & sidecars);

}
