// Core ML sidecar for the Audio8 codec's synthesis stack (Apple-only, compiled
// under TTS_CPP_USE_COREML).
//
// Wraps a compiled `.mlmodelc` produced by scripts/export-audio8-codec-coreml.py:
// the two upsampling stages and the DAC decoder stack, from the windowed post
// transformer's output to the waveform, with one fixed-shape MLMultiArray input
// {1, latent_dim, W} and one output {1, 1, W * frame_size} (Float16 or Float32;
// the wrapper adapts). The quantizer banks and the post transformer stay on
// ggml, as does everything the sidecar cannot serve: every entry point fails
// soft (nullptr / non-zero) so the caller falls back to the ggml synthesis
// blocks.

#pragma once

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

struct audio8_coreml_codec_context;

// Loads the compiled model and checks its interface against the codec's
// latent width and samples per frame. nullptr on any failure.
struct audio8_coreml_codec_context * audio8_coreml_codec_init(const char * path_mlmodelc,
                                                              int64_t      latent_dim,
                                                              int64_t      frame_size);

void audio8_coreml_codec_free(struct audio8_coreml_codec_context * ctx);

// The fixed window the model was exported at, in post frames.
int64_t audio8_coreml_codec_window_frames(const struct audio8_coreml_codec_context * ctx);

// post: exactly window_frames frames of the post latent, channels inner-most
//       (post[t * latent_dim + c], the engine's own layout).
// pcm:  destination for window_frames * frame_size mono samples.
// 0 on success; non-zero on a prediction failure.
int audio8_coreml_codec_synthesize(struct audio8_coreml_codec_context * ctx,
                                   const float * post,
                                   float       * pcm);

// Requested-compute label for logs and stats, e.g. "coreml-all". Core ML may
// still place individual operations elsewhere.
const char * audio8_coreml_codec_backend_label(const struct audio8_coreml_codec_context * ctx);

#if defined(__cplusplus)
}
#endif
