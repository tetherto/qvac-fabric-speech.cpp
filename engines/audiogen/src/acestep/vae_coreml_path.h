#pragma once

#include <string>

namespace tts_cpp::acestep {

// Sidecar path for a VAE GGUF: extension and trailing quantization tag are
// stripped, "-decoder.mlmodelc" is appended. "vae-BF16.gguf" and
// "vae.f16.gguf" both resolve to "vae-decoder.mlmodelc" next to the GGUF.
std::string coreml_vae_sidecar_path(const std::string & gguf_path);

}  // namespace tts_cpp::acestep
