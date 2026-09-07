#pragma once

// Log-mel preprocessing: STFT, mel filterbank from GGUF, optional per-bin CMVN (NeMo-style).
//
// Filterbank weights come from the checkpoint so mel matches training without recomputing bins in C++.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace parakeet {

// Default log-zero guard used when the GGUF metadata key
// `parakeet.preproc.log_zero_guard_value` is absent. This is exactly
// 2**-24 (~5.96e-08) and matches NeMo's default and the value baked
// into the converter (scripts/convert-nemo-to-gguf.py). Use
// this constant from any code path that needs the same fallback.
inline constexpr float kDefaultLogZeroGuard = 5.960464477539063e-08f;

// Mirrors NeMo's AudioToMelSpectrogramPreprocessor.normalize:
//   PerFeature -> apply per-bin CMVN over the valid frames (CTC, TDT,
//                  Sortformer in this repo). Default in NeMo.
//   None       -> skip CMVN entirely; emit raw log-mel. EOU
//                  (`nvidia/parakeet_realtime_eou_120m-v1`) uses this.
enum class MelNormalize {
    PerFeature,
    None,
};

struct MelConfig {
    int sample_rate = 16000;
    int n_fft       = 512;
    int win_length  = 400;
    int hop_length  = 160;
    int n_mels      = 80;

    float preemph              = 0.97f;
    float log_zero_guard_value = kDefaultLogZeroGuard;
    MelNormalize normalize     = MelNormalize::PerFeature;

    std::vector<float> filterbank;
    std::vector<float> window;

    // Worker threads for the per-frame FFT and filterbank loops; 0 picks
    // from the frame count and the hardware, capped internally.
    int n_threads = 0;
};

int load_wav_mono_f32(const std::string & wav_path,
                      std::vector<float>   & out_samples,
                      int                  & out_sample_rate);

// Reusable working buffers for `compute_log_mel`. Holds every per-call
// scratch allocation so streaming workloads (Mode 2 / Mode 3 chunked
// transcribe, live diarization) don't re-allocate 6 std::vectors per
// chunk. Default-initialise once on the engine / stream session and
// pass the same `MelState&` to every `compute_log_mel` call. The
// stateful overload is byte-identical to the stateless form -- it
// just reuses buffers when their capacity is large enough.
//
// The first call grows each buffer as needed; subsequent calls with
// the same `(n_samples, n_fft, n_mels)` shape never reallocate.
//
// `window_padded_n_fft` is the cache key for the precomputed
// zero-padded analysis window: re-derives only when `cfg.n_fft`
// changes (which is once per engine lifetime in practice).
// Column range [lo, hi) of one mel filter's non-zero weights.
struct MelFilterBand {
    int lo = 0;
    int hi = 0;
};

std::vector<MelFilterBand> make_filter_bands(const std::vector<float> & filterbank, int n_mels, int n_bins);

// One log-mel frame from a power spectrum: banded filterbank dot products, then the log.
void log_mel_frame(const float * frame_power, int n_bins, int n_mels,
                   const float * filterbank, const MelFilterBand * bands,
                   float log_zero_guard, float * mel_out);

struct MelState {
    std::vector<float>               x;
    std::vector<float>               x_padded;
    std::vector<float>               window_padded;
    std::vector<float>               power;
    std::vector<MelFilterBand>       filter_bands;

    int                              window_padded_n_fft = 0;
    const std::vector<float> *       window_padded_src   = nullptr;
    const std::vector<float> *       filter_bands_src    = nullptr;
};

struct IncrementalMelState {
    struct Impl;
    std::unique_ptr<Impl> impl;

    IncrementalMelState();
    ~IncrementalMelState();
    IncrementalMelState(IncrementalMelState &&) noexcept;
    IncrementalMelState & operator=(IncrementalMelState &&) noexcept;
    IncrementalMelState(const IncrementalMelState &) = delete;
    IncrementalMelState & operator=(const IncrementalMelState &) = delete;
};

int compute_log_mel(const float        * samples,
                    int                  n_samples,
                    const MelConfig    & cfg,
                    std::vector<float> & out_mel,
                    int                & out_n_frames);

// Stateful overload: reuses scratch buffers from `state`. Functionally
// identical to the stateless form. Engines / stream sessions that call
// `compute_log_mel` repeatedly (every Mode 2/3 chunk, every live
// diarize step) should prefer this overload to amortise the per-call
// allocator pressure to zero after the first call.
int compute_log_mel(const float        * samples,
                    int                  n_samples,
                    const MelConfig    & cfg,
                    MelState           & state,
                    std::vector<float> & out_mel,
                    int                & out_n_frames);

int append_log_mel(
    const float * samples,
    int n_samples,
    bool finalize,
    const MelConfig & cfg,
    IncrementalMelState & state,
    std::vector<float> & out_mel,
    int & out_n_frames);

void reset_incremental_mel(IncrementalMelState & state);

void apply_per_feature_cmvn(std::vector<float> & mel,
                            int                  n_frames,
                            int                  n_mels);

}
