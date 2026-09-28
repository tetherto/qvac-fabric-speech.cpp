#include "moss/speech_generation.h"

#include "moss/generation.h"

#include <algorithm>
#include <limits>

namespace tts_cpp::moss::detail {
namespace {

void divide(std::vector<float> & logits, float temperature) {
    for (float & value : logits) {
        value /= temperature;
    }
}

bool is_reply_code(int32_t audio, const SpeechTokens & tokens) {
    return audio != tokens.speech_end;
}

} // namespace

SpeechGenerationState::SpeechGenerationState(const SpeechTokens & tokens, const SpeechRow & last_prompt_row,
                                             const SpeechLimits & limits)
    : tokens_(tokens), limits_(limits), last_(last_prompt_row) {
    if (last_prompt_row.text == tokens.text_placeholder) {
        channel_ = SpeechChannel::Audio;
    }
    if (last_prompt_row.audio == tokens.audio_pad) {
        channel_ = SpeechChannel::Text;
    }
    stopping_ = limits.max_new_tokens < 1;
}

bool SpeechGenerationState::stopping() const { return stopping_; }
bool SpeechGenerationState::truncated() const { return truncated_; }
SpeechChannel SpeechGenerationState::channel() const { return channel_; }
const std::vector<SpeechRow> & SpeechGenerationState::generated() const { return generated_; }

void SpeechGenerationState::update_channel(const SpeechRow & last) {
    if (channel_ == SpeechChannel::Text && last.text == tokens_.speech_start) {
        channel_ = SpeechChannel::Audio;
    } else if (channel_ == SpeechChannel::Audio && last.audio == tokens_.speech_end) {
        channel_ = SpeechChannel::Text;
    }
}

void SpeechGenerationState::constrain_audio(std::vector<float> & audio) const {
    const float blocked = -std::numeric_limits<float>::infinity();
    const size_t first_invalid = std::min(audio.size(), (size_t) tokens_.speech_end + 1);
    std::fill(audio.begin() + (std::ptrdiff_t) first_invalid, audio.end(), blocked);
    if ((int) generated_.size() + 1 < limits_.min_new_tokens && (size_t) tokens_.speech_end < audio.size()) {
        audio[(size_t) tokens_.speech_end] = blocked;
    }
}

int32_t SpeechGenerationState::sample(std::vector<float> logits, const SpeechSampling & sampling,
                                      std::mt19937 & rng) const {
    if (sampling.do_sample && sampling.temperature > 0.0f) {
        divide(logits, sampling.temperature);
    }
    return sample_row(std::move(logits), sampling.top_p, sampling.top_k, sampling.do_sample, rng);
}

int32_t SpeechGenerationState::limit_reply(int32_t audio) {
    if (channel_ != SpeechChannel::Audio || !is_reply_code(audio, tokens_)) {
        return audio;
    }
    if (limits_.max_reply_codes > 0 && reply_codes_ >= limits_.max_reply_codes) {
        truncated_ = true;
        return tokens_.speech_end;
    }
    reply_codes_++;
    return audio;
}

void SpeechGenerationState::record(const SpeechRow & row) {
    generated_.push_back(row);
    last_ = row;
    stopping_ = row.text == tokens_.pad || row.text == tokens_.im_end ||
                (int) generated_.size() >= limits_.max_new_tokens;
}

SpeechRow SpeechGenerationState::next(SpeechLogits logits, const SpeechSampling & sampling, std::mt19937 & rng) {
    update_channel(last_);
    constrain_audio(logits.audio);
    SpeechRow row{sample(std::move(logits.text), sampling, rng), sample(std::move(logits.audio), sampling, rng)};
    if (channel_ == SpeechChannel::Audio) {
        row.text = tokens_.text_placeholder;
    }
    row.audio = limit_reply(row.audio);
    record(row);
    return row;
}

std::vector<int32_t> speech_reply_codes(const std::vector<SpeechRow> & generated, const SpeechTokens & tokens) {
    std::vector<int32_t> codes;
    const size_t kept = generated.empty() ? 0 : generated.size() - 1;
    for (size_t i = 0; i < kept && is_reply_code(generated[i].audio, tokens); ++i) {
        codes.push_back(generated[i].audio);
    }
    return codes;
}

} // namespace tts_cpp::moss::detail
