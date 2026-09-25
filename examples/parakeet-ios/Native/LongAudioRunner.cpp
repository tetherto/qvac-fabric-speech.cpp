#include "LongAudioRunner.h"

#include "dr_wav.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace {

constexpr int kSampleRate = 16000;
constexpr int kBatchSamples = 30 * kSampleRate;

}  // namespace

LongAudioResult run_long_audio_wav(
    parakeet::Engine & engine,
    const std::string & wav_path,
    std::atomic<bool> & cancel_requested,
    bool require_coreml,
    LongAudioSegmentCallback on_segment,
    int64_t maximum_samples) {
    drwav wav;
    if (!drwav_init_file(&wav, wav_path.c_str(), nullptr)) {
        throw std::runtime_error("Could not open prepared WAV file");
    }

    struct WavGuard {
        drwav * wav;
        ~WavGuard() { drwav_uninit(wav); }
    } guard{&wav};

    if (wav.channels != 1 || wav.sampleRate != kSampleRate) {
        throw std::runtime_error("Prepared audio must be mono 16 kHz PCM WAV");
    }

    parakeet::StreamingOptions options;
    options.chunk_ms = 8000;
    options.emit_partials = false;

    LongAudioResult output;
    output.encoder_used_coreml = require_coreml;
    std::vector<float> samples(kBatchSamples, 0.0f);

    while (!cancel_requested.load()) {
        int64_t requested = kBatchSamples;
        if (maximum_samples >= 0) {
            const int64_t remaining = maximum_samples - output.audio_samples;
            if (remaining <= 0) break;
            requested = std::min(requested, remaining);
        }

        const drwav_uint64 read = drwav_read_pcm_frames_f32(
            &wav, static_cast<drwav_uint64>(requested), samples.data());
        if (read == 0) break;

        const int actual_samples = static_cast<int>(read);
        const double time_offset = static_cast<double>(output.audio_samples) / kSampleRate;
        const double valid_batch_seconds = static_cast<double>(actual_samples) / kSampleRate;

        // A fixed inference shape prevents ggml from replacing a live graph
        // allocation for the final short tail. dr_wav writes only `read`
        // frames, so explicitly clear the padding retained from prior batches.
        if (actual_samples < kBatchSamples) {
            std::fill(samples.begin() + actual_samples, samples.end(), 0.0f);
        }

        const auto batch = engine.transcribe_samples_stream(
            samples.data(),
            kBatchSamples,
            kSampleRate,
            options,
            [&](const parakeet::StreamingSegment & segment) {
                if (segment.text.empty() || segment.start_s >= valid_batch_seconds) return;
                ++output.segment_count;
                if (on_segment) {
                    on_segment(
                        segment,
                        time_offset + segment.start_s,
                        time_offset + std::min(segment.end_s, valid_batch_seconds));
                }
            });

        if (!batch.text.empty()) {
            if (!output.text.empty() && output.text.back() != ' ') output.text.push_back(' ');
            output.text.append(batch.text);
        }
        output.audio_samples += actual_samples;
        output.inference_ms += batch.total_ms;
        output.encoder_used_coreml =
            output.encoder_used_coreml && batch.encoder_used_coreml;
        ++output.batch_count;

        if (read < static_cast<drwav_uint64>(requested)) break;
    }

    if (cancel_requested.load()) return output;
    if (output.audio_samples == 0) throw std::runtime_error("Audio file is empty");
    if (require_coreml && !output.encoder_used_coreml) {
        throw std::runtime_error("At least one audio batch fell back from Core ML");
    }
    return output;
}
