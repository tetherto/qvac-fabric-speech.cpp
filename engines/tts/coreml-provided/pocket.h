#pragma once

// provided() implementations of the Pocket TTS Core ML templates. Compiled
// into the tts library's Core ML generator and into qvac-coreml-tool, each
// under its own QVAC_COREML_NS.

#ifndef QVAC_COREML_NS
#error "QVAC_COREML_NS must be defined"
#endif

namespace QVAC_COREML_NS {
namespace pocket_provided {

// Registers every Pocket provided() key (idempotent). The tts library calls
// it before it probes a Pocket template; the host tool registers on load.
void register_all();

}  // namespace pocket_provided
}  // namespace QVAC_COREML_NS
