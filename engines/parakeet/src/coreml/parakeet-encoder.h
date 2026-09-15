// Core ML FastConformer encoder sidecar for parakeet-cpp.
//
// Mirrors the whisper.cpp `whisper_coreml_*` C shim: an opaque context wraps a
// compiled `.mlmodelc` encoder that runs on the Apple Neural Engine, and the
// caller hands log-mel features in / reads encoder hidden states out. The rest
// of the validated pipeline (mel preprocessing, TDT/EOU decode, tokenizer) stays on ggml.
//
// This header is Apple-only; it is compiled and referenced solely when the
// PARAKEET_USE_COREML build definition is set. All entry points return failure
// (nullptr / non-zero) rather than aborting so the caller can fall back to the
// ggml encoder on any error.
//
// Export contract (what the mobius `.mlmodelc` must expose so the wrapper binds):
//   - Exactly one MLMultiArray input  = log-mel features, dims {n_mels, n_mel_frames}
//     (either order; Float16 or Float32; the wrapper adapts). This is the offline FastConformer encoder
//     input, i.e. everything from the subsampling stack through the last Conformer
//     block; it must NOT include a decoder joint projection. The mel time axis may be
//     fixed or flexible. For a fixed shape, an input no longer than the declared
//     capacity is zero-padded to that capacity. Longer inputs must be divided into
//     windows by the caller. Flexible inputs are allocated at the requested length.
//   - Exactly one MLMultiArray output = encoder hidden states, dims {n_enc_frames,
//     d_model} (either order; Float32 or Float16). n_enc_frames is n_mel_frames put
//     through the three stride-2 subsampling convs (matching run_encoder's sizing).
//   - TDT exports are full-context and may pad shorter inputs. EOU exports bake
//     causal subsampling/convolution and chunk-limited attention into one exact
//     fixed shape; the caller rejects nonmatching EOU shapes before this wrapper.

#pragma once

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

struct parakeet_coreml_context;

// Load a compiled Core ML encoder from a `.mlmodelc` directory.
// Returns nullptr on any failure (missing directory, load error, unsupported
// OS, unexpected model interface) so the caller falls back to ggml.
struct parakeet_coreml_context * parakeet_coreml_init(const char * path_mlmodelc);

// Release a context created by parakeet_coreml_init. Safe to call with nullptr.
void parakeet_coreml_free(struct parakeet_coreml_context * ctx);

// Return the fixed mel-frame capacity of the model input for `n_mels`.
// Returns 0 for flexible, enumerated-multi-shape, or unrecognised inputs.
// The value describes model capacity, not the current utterance length.
int64_t parakeet_coreml_fixed_mel_frames(
        const struct parakeet_coreml_context * ctx,
        int64_t n_mels);

// Run the encoder on the Apple Neural Engine.
//
//   mel           log-mel features, row-major (n_mel_frames, n_mels),
//                 n_mels contiguous (the layout `compute_log_mel` emits).
//   encoder_out   destination for the encoder hidden states, row-major
//                 (n_enc_frames, d_model), d_model contiguous. The caller
//                 sizes it to n_enc_frames * d_model floats.
//
// The wrapper adapts the mel/output tensor orientation to whatever the exported
// model declares (features-major or time-major), so the mobius export only has
// to agree on the dimension sizes, not their order.
//
// Returns 0 on success; non-zero on a shape mismatch or prediction failure, in
// which case the caller falls back to the ggml encoder.
int parakeet_coreml_encode(struct parakeet_coreml_context * ctx,
                           int64_t       n_mel_frames,
                           int64_t       n_mels,
                           const float * mel,
                           int64_t       n_enc_frames,
                           int64_t       d_model,
                           float       * encoder_out);

// Human-readable requested-compute label for stats/logging, e.g. "coreml-ane".
// Core ML may still fall back to CPU for individual unsupported operations.
const char * parakeet_coreml_backend_label(const struct parakeet_coreml_context * ctx);

#if defined(__cplusplus)
}
#endif
