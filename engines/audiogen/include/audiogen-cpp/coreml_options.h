#pragma once

// Core ML options of the ACE-Step VAE decoder, which loads its sidecar through
// the shared Core ML runtime (EngineOptions::coreml, VaeOptions::coreml).
//
// Every field is unset by default. They are resolved once, when the Engine
// (or a standalone Vae) is created, in this order per field: the explicit
// field, then the ACESTEP_* environment (ACESTEP_COREML_STRICT -> use =
// require; ACESTEP_COREML_COMPUTE_UNITS), then QVAC_COREML_CACHE_DIR (cache
// directory only), then the default. ACESTEP_COREML_DISABLE forces use = off
// and overrides everything.
//
// Defaults: use automatic, generate background, wait none on macOS and call on
// iOS, prewarm predicted (which requests nothing at construction for ACE-Step:
// the decoder is requested by the first decode), no cache directory (no
// generation and no warm hints), the template's compute units (all), strict
// waits unbounded, no memory budget.
//
// Builds without AUDIOGEN_COREML accept these options; an explicit invalid
// compute_units still throws there, and everything else is ignored.

#include <cstdint>
#include <optional>
#include <string>

namespace tts_cpp::acestep {

struct CoremlOptions {
    enum class Use      { unset, automatic, off, require };
    enum class Generate { unset, never, background };
    enum class Wait     { unset, none, construct, call };
    enum class Prewarm  { unset, none, minimal, predicted, all };

    // automatic: the VAE decode runs on the Core ML sidecar once it is loaded,
    // on ggml until then. off: ggml only. require: like wait = call, but a
    // sidecar that cannot serve a decode (failed to generate or load, failed a
    // prediction, latent shorter than one window) fails the decode instead of
    // falling back.
    Use use = Use::unset;
    // background: with a cache directory and no prebuilt sidecar next to the
    // VAE GGUF, build the sidecar from the GGUF in the background (macOS 15 or
    // later) and decode on ggml until it is ready. never: no generation;
    // prebuilt sidecars still load.
    Generate generate = Generate::unset;
    // none: never block, apart from a short bounded wait for a load this host
    //       finished quickly before (a warm hint), never longer than the ggml
    //       decode it would replace.
    // construct: the constructor waits until the construction group (prewarm
    //       all) is ready or failed.
    // call: each decode waits until the sidecar is ready or failed.
    Wait wait = Wait::unset;
    // none, minimal and predicted request nothing at construction: the
    // decoder is requested by decode demand only. all loads it once at
    // construction to warm the Neural Engine plan; in the default low-memory
    // mode it is released again after a waited construction (wait =
    // construct), so install-time preparation retains no VAE.
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

}  // namespace tts_cpp::acestep
