#pragma once

// The identity of the GGUF parler_load_gguf read the weights from (see
// parler_dac_source_before_load in dac_coreml.h). Core ML builds only: the
// stat tuples are the shared runtime's.

#ifdef TTS_CPP_USE_COREML
#include "qvac_coreml/slot.h"

namespace tts_cpp {
namespace parler {
namespace detail {

struct parler_dac_source {
    QVAC_COREML_NS::SourceStat before, after;
    bool ok_before = false, ok_after = false;
};

} // namespace detail
} // namespace parler
} // namespace tts_cpp
#endif
