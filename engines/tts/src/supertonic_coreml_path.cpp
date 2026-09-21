#include "supertonic_coreml_path.h"

#include "coreml_sidecar_path.h"

namespace tts_cpp {
namespace supertonic {
namespace detail {

std::string coreml_vocoder_sidecar_path(const std::string & gguf_path) {
    return ::tts_cpp::detail::coreml_sidecar_path(gguf_path, "-vocoder");
}

}  // namespace detail
}  // namespace supertonic
}  // namespace tts_cpp
