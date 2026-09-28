#pragma once

#include "moss/speech_lm.h"

#include <cstdint>
#include <random>
#include <vector>

namespace tts_cpp::moss::detail {

struct SpeechSampling {
    bool do_sample = true;
    float temperature = 0.7f;
    float top_p = 0.95f;
    int top_k = 20;
};

struct SpeechLimits {
    int max_new_tokens = 0;
    int min_new_tokens = 0;
    int max_reply_codes = 0;
};

enum class SpeechChannel { Text, Audio };

class SpeechGenerationState {
public:
    SpeechGenerationState(const SpeechTokens & tokens, const SpeechRow & last_prompt_row, const SpeechLimits & limits);

    bool stopping() const;
    bool truncated() const;
    SpeechChannel channel() const;
    const std::vector<SpeechRow> & generated() const;
    SpeechRow next(SpeechLogits logits, const SpeechSampling & sampling, std::mt19937 & rng);

private:
    void update_channel(const SpeechRow & last);
    void constrain_audio(std::vector<float> & audio) const;
    int32_t sample(std::vector<float> logits, const SpeechSampling & sampling, std::mt19937 & rng) const;
    int32_t limit_reply(int32_t audio);
    void record(const SpeechRow & row);

    SpeechTokens tokens_;
    SpeechLimits limits_;
    SpeechChannel channel_ = SpeechChannel::Text;
    SpeechRow last_;
    std::vector<SpeechRow> generated_;
    int reply_codes_ = 0;
    bool stopping_ = false;
    bool truncated_ = false;
};

std::vector<int32_t> speech_reply_codes(const std::vector<SpeechRow> & generated, const SpeechTokens & tokens);

} // namespace tts_cpp::moss::detail
