// The identity capture parler_load_gguf calls around its reads. Built with
// gguf.cpp wherever it is compiled; without Core ML both calls do nothing.

#include "dac_coreml.h"
#include "internal.h"

#ifdef TTS_CPP_USE_COREML
#include "dac_source.h"

#include "qvac_coreml/generation.h"

#include <memory>
#endif

namespace tts_cpp {
namespace parler {
namespace detail {

#ifdef TTS_CPP_USE_COREML
void parler_dac_source_before_load(parler_model & model, const std::string & path) {
    auto s = std::make_shared<parler_dac_source>();
    s->ok_before = QVAC_COREML_NS::generation::source_stat(path, s->before);
    model.dac_source = std::move(s);
}

void parler_dac_source_after_load(parler_model & model, const std::string & path) {
    if (model.dac_source) {
        model.dac_source->ok_after = QVAC_COREML_NS::generation::source_stat(path, model.dac_source->after);
    }
}
#else
void parler_dac_source_before_load(parler_model &, const std::string &) {}
void parler_dac_source_after_load(parler_model &, const std::string &) {}
#endif

} // namespace detail
} // namespace parler
} // namespace tts_cpp
