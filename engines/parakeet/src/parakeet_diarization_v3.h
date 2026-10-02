#pragma once

#include "parakeet_ctc.h"

#include <cstddef>
#include <vector>

namespace parakeet {

constexpr int kNemotronChunkMs = 1040;
constexpr int kNemotronSpeakerCacheFrames = 264;
constexpr int kNemotronFifoFrames = 80;
constexpr int kNemotronRightContextMs = 80;
constexpr int kNemotronUpdateFrames = 40;

struct NemotronDiarizationChunk {
    std::vector<float> probabilities;
    std::vector<float> embeddings;
    int state_frames = 0;
    int chunk_frames = 0;
};

struct NemotronDiarizationFitMeasure {
    size_t device_compute_bytes = 0;
    size_t host_input_bytes = 0;
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

int prewarm_nemotron_diarization(
    const ParakeetCtcModel & model,
    float audio_seconds);

int nemotron_diarization_encoder_frames(
    const ParakeetCtcModel & model,
    long long mel_frames);

int nemotron_diarization_stream_mel_frames(const ParakeetCtcModel & model);

int nemotron_diarization_stream_state_frames();

int measure_nemotron_diarization(
    const ParakeetCtcModel & model,
    int mel_frames,
    int state_frames,
    NemotronDiarizationFitMeasure & output);

}
