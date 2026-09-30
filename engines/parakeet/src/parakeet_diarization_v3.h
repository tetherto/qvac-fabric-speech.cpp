#pragma once

#include "parakeet_ctc.h"

#include <vector>

namespace parakeet {

struct NemotronDiarizationChunk {
    std::vector<float> probabilities;
    std::vector<float> embeddings;
    int state_frames = 0;
    int chunk_frames = 0;
};

int run_nemotron_diarization_chunk(
    const ParakeetCtcModel & model,
    const float * mel,
    int mel_frames,
    const float * state,
    int state_frames,
    NemotronDiarizationChunk & output);

int run_nemotron_diarization(
    const ParakeetCtcModel & model,
    const float * mel,
    int mel_frames,
    std::vector<float> & probabilities);

}
