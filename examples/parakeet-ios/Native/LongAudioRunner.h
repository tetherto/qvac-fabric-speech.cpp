#pragma once

#include <parakeet/engine.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>

struct LongAudioResult {
    std::string text;
    double inference_ms = 0.0;
    int64_t audio_samples = 0;
    int batch_count = 0;
    int segment_count = 0;
    bool encoder_used_coreml = true;
};

using LongAudioSegmentCallback = std::function<void(
    const parakeet::StreamingSegment & segment,
    double absolute_start_s,
    double absolute_end_s)>;

LongAudioResult run_long_audio_wav(
    parakeet::Engine & engine,
    const std::string & wav_path,
    std::atomic<bool> & cancel_requested,
    bool require_coreml,
    LongAudioSegmentCallback on_segment,
    int64_t maximum_samples = -1);
