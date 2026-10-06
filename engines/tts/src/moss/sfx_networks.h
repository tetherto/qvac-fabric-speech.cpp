#pragma once

#include "moss/sfx_model.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace tts_cpp::moss::detail {

void validate_text_encoder(const SfxModel & model);
void validate_dit(const SfxModel & model);
void validate_vae(const SfxModel & model);

std::vector<float> encode_text(SfxModel & model, const std::vector<int32_t> & ids);
SfxMemory measure_text(SfxModel & model, const std::vector<int32_t> & ids);

class SfxDitSession {
public:
    SfxDitSession(SfxModel & model, int frames, bool measure_only = false);
    ~SfxDitSession();
    SfxDitSession(const SfxDitSession &) = delete;
    SfxDitSession & operator=(const SfxDitSession &) = delete;

    std::vector<float> velocity(const std::vector<float> & latents, float timestep,
                                const std::vector<float> & context);
    SfxMemory measure();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

using SfxDecodeProgress = std::function<bool(int done, int total)>;

std::vector<float> decode_latents(SfxModel & model, const std::vector<float> & latents, int frames,
                                  int keep_frames, int window_frames, const SfxDecodeProgress & progress);
SfxMemory measure_decode(SfxModel & model, int frames, int keep_frames, int window_frames);

} // namespace tts_cpp::moss::detail
