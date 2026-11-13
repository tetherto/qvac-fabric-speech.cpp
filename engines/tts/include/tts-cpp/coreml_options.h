#pragma once

// Core ML options for tts-cpp engines that load sidecars through the shared
// Core ML runtime (today: the Parler DAC decoder).
//
// Every field is unset by default. An engine resolves them once, at
// construction, in this order per field: the explicit field, then the model
// family's environment variable (<FAMILY>_COREML_STRICT -> use = require;
// <FAMILY>_COREML_COMPUTE_UNITS), then QVAC_COREML_CACHE_DIR (cache directory
// only), then the default. <FAMILY>_COREML_DISABLE forces use = off and
// overrides everything.
//
// Defaults: use automatic, generate background, wait none on macOS and call on
// iOS, prewarm predicted, no cache directory (no generation and no warm
// hints), the stage's own compute units, strict waits unbounded, no memory
// budget.
//
// Builds without TTS_CPP_COREML accept these options; an explicit invalid
// compute_units still throws there, and everything else is ignored.

#include <cstdint>
#include <optional>
#include <string>

namespace tts_cpp {

struct CoremlOptions {
    enum class Use      { unset, automatic, off, require };
    enum class Generate { unset, never, background };
    enum class Wait     { unset, none, construct, call };
    enum class Prewarm  { unset, none, minimal, predicted, all };

    // automatic: a stage runs on its Core ML sidecar once the sidecar is
    // loaded, on ggml until then. off: ggml only. require: like wait = call,
    // but a sidecar that cannot serve a call (failed to generate or load,
    // failed a prediction) makes the call fail instead of falling back.
    Use use = Use::unset;
    // background: with a cache directory and no prebuilt sidecar next to the
    // GGUF, build the sidecar from the GGUF in the background (on a supported
    // OS) and run the stage on ggml until it is ready. never: no generation;
    // prebuilt sidecars still load.
    Generate generate = Generate::unset;
    // none: never block, apart from a short bounded wait for a load this host
    //       finished quickly before (a warm hint).
    // construct: the constructor waits until its sidecars are ready or failed.
    // call: each call waits until the sidecar it needs is ready or failed.
    Wait wait = Wait::unset;
    // Which sidecars are requested at construction: none (only on first use),
    // or minimal / predicted / all (every engine's minimal set; see the
    // engine's documentation).
    Prewarm prewarm = Prewarm::unset;
    // Where generated sidecars and warm hints are kept. Unset ->
    // QVAC_COREML_CACHE_DIR -> "" (no generation, no hints).
    std::optional<std::string> cache_dir;
    // all | cpu_and_ane | cpu_and_gpu | cpu_only
    std::optional<std::string> compute_units;
    // Longest wait of a strict (use = require) acquisition; unset -> unbounded.
    std::optional<int64_t> strict_wait_ms;
    // Bytes fit was assessed against; unset -> no memory-degrade.
    std::optional<int64_t> memory_budget_bytes;
};

}  // namespace tts_cpp
