#pragma once

struct MM3Model;

namespace tts_cpp::minimax::detail {
void attach_coreml_blocks(MM3Model & model);
void report_coreml_predictions(const MM3Model & model);
}
