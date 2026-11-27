#pragma once

#include "parakeet/fit.h"

namespace parakeet::moss::detail {

bool charge_coreml_encoder(FitResult & result, uint64_t bytes);
void finish_transcribe_fit(FitResult &result, uint64_t margin);

}
