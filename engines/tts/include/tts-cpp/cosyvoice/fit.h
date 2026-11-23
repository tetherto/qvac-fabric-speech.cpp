#pragma once

// CosyVoice3 memory-fit preflight: project whether the CosyVoice3 GGUF set
// (LM + flow + HiFT + voice) fits the device memory available right now,
// without reading any weight data.
//
// The engine's pipeline is STAGED: the LM, flow, and HiFT GGUFs are loaded,
// run, and freed one at a time inside each synthesize() (the voice GGUF is
// read once at construction and immediately freed).  The resident peak is
// therefore the LARGEST SINGLE PHASE, not the sum -- fit_params projects each
// phase and takes the verdict from the peak one.  FitBreakdown describes the
// peak phase (so its fields still sum to total_bytes == the projected peak);
// the report lists all three phases.
//
// What each phase prices, metadata-only and BY CONSTRUCTION (the same
// loaders run measure-mode, the same graph builders are priced through
// ggml's size-only APIs):
//
//   LM    -- the LM GGUF's weights (on GPU loads the two host-resident
//            embedding tables are charged to host instead, exactly as the
//            loader splits them); the F32 KV cache at the capacity the
//            runtime allocates UP FRONT (L0 + 50*(text_tokens+1) + 1
//            positions -- the dominant state term, linear in the text
//            length); and the decode arena (max of the prefill graph and the
//            deepest decode step -- one gallocr serves the whole decode)
//   flow  -- the flow GGUF's weights and the larger of the front-end and
//            DiT-estimator (CFG batch B=2) graphs, at the projected frame
//            count TM = 2*(prompt tokens + speech tokens)
//   hift  -- the HiFT GGUF's weights and the largest of the f0 / STFT /
//            decode graphs at the projected mel length
//
// Host extras cover the phase's workload-scaled slabs (LM input row, causal
// mask, logits and sampler vectors; the mu/cond/noise/CFG family and the
// baked-noise mirror; the SineGen excitation, f0/source/STFT/waveform
// buffers) plus the persistent baked-voice vectors.
//
// Scope / exclusions (documented, strict where approximate):
//   * Enabled prebuilt Core ML sidecars (LM, flow DiT, or a prebuilt
//     front-end, F0 or HiFT sidecar) return Error with reason
//     "coreml-memory-estimate-unavailable" because their resident models and
//     prediction buffers cannot be projected by the native fitter.
//   * Generated front-end, F0 and HiFT sidecars (coreml.cache_dir set, no
//     prebuilt sidecar, macOS 15) are projected (FitResult::coreml): the
//     verdict adds their templates' Core ML envelope (each generated stage's
//     resident part plus the largest stage's load or generation transient)
//     to the peak phase, since the sidecars stay loaded across phases. With
//     coreml.memory_budget_bytes, a budget only the ggml projection fits
//     reports coreml.disabled_reason "memory" and the ggml verdict (the
//     Engine given the same budget then runs those stages on ggml).
//   * The resident Qwen tokenizer (vocab.json + merges.txt live OUTSIDE the
//     GGUF set and are parsed into hash maps, typically tens of MB of host
//     RAM) is NOT projected -- the fitter never sees those files.
//   * The construction-time voice-cloning bake (reference_audio: S3Tokenizer
//     + CAM++ GGUFs) is NOT projected; the projection covers the baked-voice
//     synthesis pipeline.  Follow-up under QVAC-24283.
//   * speech_tokens defaults to the runtime's own generation cap
//     (50 * (text_tokens + 1)), so the default verdict bounds every
//     utterance the runtime could produce for that text length; pass a
//     smaller value to price a representative utterance instead.  The KV
//     cache is always priced at the cap -- that is what the runtime
//     allocates regardless of where EOS lands.
//
// See tts-cpp/fit.h for the status contract and result layout.

#include "tts-cpp/coreml_options.h"
#include "tts-cpp/export.h"
#include "tts-cpp/fit.h"

#include <cstdint>
#include <string>

namespace tts_cpp {
namespace cosyvoice {

struct FitOptions {
    // Same roles as the EngineOptions paths: all four are required.
    std::string llm_gguf_path;
    std::string flow_gguf_path;
    std::string hift_gguf_path;
    std::string voice_gguf_path;

    // Same semantics as the EngineOptions fields of the same name.
    int         n_gpu_layers  = 0;
    int         vulkan_device = 0;
    std::string backend;
    std::string backends_dir;

    // ── Workload ──────────────────────────────────────────────────────────
    // Total LM text ids of one synthesize() (prompt template + reference
    // transcript + the text to speak).  Drives the prefill width, the KV
    // capacity (50x multiplier!), and the generation cap.  The runtime's
    // length basis excludes the template/transcript ids, so using the total
    // here bounds it from above -- the strict direction.
    int text_tokens = 30;

    // Speech tokens of one generation (25/s of audio).  0 = the runtime cap
    // 50 * (text_tokens + 1); the flow and HiFT phases follow this, the KV
    // cache always follows the cap.
    int speech_tokens = 0;

    // Free-memory headroom that must remain on the device for the projection
    // to count as fitting.
    uint64_t margin_bytes = 256ull * 1024 * 1024;

    // Mirror EngineOptions::coreml and coreml_ignore_prebuilt: the front-end,
    // F0 and HiFT routes and their Core ML envelope.
    CoremlOptions coreml;
    bool coreml_ignore_prebuilt = false;
};

// Project the model set + workload in `opts` against the device memory
// available right now.  Reads only GGUF metadata, never weight data; builds
// and measures graphs but never allocates or executes them.  Never throws for
// a "does not fit" outcome (a valid Failure result) or an unreadable model
// (Error); see FitStatus in tts-cpp/fit.h.
TTS_CPP_API FitResult fit_params(const FitOptions & opts);

}  // namespace cosyvoice
}  // namespace tts_cpp

// CLI front-end over cosyvoice::fit_params (the cosyvoice-fit-params tool);
// lives in the library so hosts can link it directly.  Exit code == fit
// status: 0 fits, 1 does not fit, 2 error.
extern "C" TTS_CPP_API int cosyvoice_fit_cli_main(int argc, char ** argv);
