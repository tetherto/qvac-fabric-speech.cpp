#pragma once

// Library-internal access to Vae: an Engine's VAE shares the Engine's Core ML
// stage (one slot per Engine, kept across low-memory reloads of the ggml VAE)
// instead of creating one of its own.

#include "audiogen-cpp/acestep/vae.h"

#include <memory>
#include <string>

namespace tts_cpp::acestep::detail {

class vae_coreml_stage;

struct VaeAccess {
    // Vae::load with `stage` (may be null: ggml only) as its Core ML stage.
    static std::unique_ptr<Vae> load(const std::string & gguf_path, const VaeOptions & opts,
                                     std::shared_ptr<vae_coreml_stage> stage);
    // Whether `v`'s weights came from the file its Core ML decoder is
    // generated from (decodes acquire the decoder only then). Tests.
    static bool coreml_source_ok(const Vae & v);
};

}  // namespace tts_cpp::acestep::detail
