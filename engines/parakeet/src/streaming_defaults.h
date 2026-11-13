#pragma once

namespace parakeet {

inline int resolve_transcription_chunk_ms(int requested_ms, bool full_eu) {
    constexpr int kAutomaticChunkMs = 0;
    constexpr int kDefaultChunkMs = 1000;
    constexpr int kFullEuChunkMs = 2000;
    if (requested_ms != kAutomaticChunkMs) return requested_ms;
    return full_eu ? kFullEuChunkMs : kDefaultChunkMs;
}

}  // namespace parakeet
