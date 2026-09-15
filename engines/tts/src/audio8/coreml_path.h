#pragma once
// Where the Audio8 codec's Core ML synthesis sidecar lives, given the codec
// decoder GGUF: the file extension and any trailing quantisation tag go, and
// `.mlmodelc` comes on, so one exported sidecar serves every storage tier of
// the same decoder:
//   models/audio8-codec-decoder-q8_0.gguf -> models/audio8-codec-decoder.mlmodelc
//   models/audio8-codec-decoder-f32.gguf  -> models/audio8-codec-decoder.mlmodelc
//   models/audio8-codec-decoder.gguf      -> models/audio8-codec-decoder.mlmodelc
// The tag is dropped because the synthesis stack is convolution kernels and
// per-channel parameters, which every tier stores at f16 or better, so the
// export is the same whichever GGUF it was taken from.
//
// Pure string logic, compiled on every platform so the naming contract can be
// unit-tested without Apple frameworks or a model.

#include <string>

namespace tts_cpp {
namespace audio8 {
namespace detail {

std::string coreml_codec_sidecar_path(const std::string & gguf_path);

}  // namespace detail
}  // namespace audio8
}  // namespace tts_cpp
