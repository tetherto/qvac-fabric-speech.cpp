// Core ML Oobleck VAE decoder sidecar for audiogen-cpp (ACE-Step).
//
// Mirrors the parakeet encoder sidecar shim: an opaque context wraps a
// compiled `.mlmodelc` decoder that runs on the Apple Neural Engine; the
// caller hands one fixed-size latent window in and reads interleaved stereo
// PCM out. Every entry point fails soft (nullptr / non-zero) so the caller
// falls back to the ggml decode path on any error.
//
// Export contract (scripts/export-vae-coreml.py):
//   - one MLMultiArray input, dims {1, 64, T_latent} (Float16 or Float32),
//     fixed T_latent
//   - one MLMultiArray output, dims {1, 2, T_latent * 1920}
//
// This header is Apple-only; it is compiled and referenced solely when the
// AUDIOGEN_USE_COREML build definition is set.

#pragma once

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

struct acestep_coreml_vae_context;

// Load a compiled Core ML decoder from a `.mlmodelc` directory. Returns
// nullptr on any failure (missing directory, load error, unexpected model
// interface, non-fixed input shape).
struct acestep_coreml_vae_context * acestep_coreml_vae_init(const char * path_mlmodelc);

// Release a context created by acestep_coreml_vae_init. Safe on nullptr.
void acestep_coreml_vae_free(struct acestep_coreml_vae_context * ctx);

// The model's fixed latent window length (frames). The caller must hand
// acestep_coreml_vae_decode exactly this many frames.
int64_t acestep_coreml_vae_window_frames(const struct acestep_coreml_vae_context * ctx);

// Decode one latent window on the Apple Neural Engine.
//
//   latent  time-major window, latent[t * 64 + c], window_frames frames
//   pcm     destination, interleaved stereo pcm[t * 2 + ch], sized by the
//           caller to window_frames * 1920 * 2 floats
//
// Returns 0 on success; non-zero on a prediction failure, in which case the
// caller falls back to the ggml decode.
int acestep_coreml_vae_decode(struct acestep_coreml_vae_context * ctx,
                              const float * latent,
                              float       * pcm);

// Requested-compute label for logging, e.g. "coreml-all". Core ML may still
// place individual operations on GPU or CPU.
const char * acestep_coreml_vae_backend_label(const struct acestep_coreml_vae_context * ctx);

#if defined(__cplusplus)
}
#endif
