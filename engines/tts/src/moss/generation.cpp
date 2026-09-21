#include "moss/generation.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

namespace tts_cpp::moss::detail {
namespace {

constexpr int64_t DRAIN_DISARMED = std::numeric_limits<int64_t>::max();
constexpr float MASKED = -std::numeric_limits<float>::infinity();

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("moss generation: " + message);
}

std::vector<float> softmax(const std::vector<float> & logits) {
    float best = MASKED;
    for (float logit : logits) {
        if (std::isfinite(logit) && logit > best) {
            best = logit;
        }
    }
    std::vector<float> probs(logits.size(), 0.0f);
    if (!std::isfinite(best)) {
        probs[0] = 1.0f;
        return probs;
    }
    float total = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        if (std::isfinite(logits[i])) {
            probs[i] = std::exp(logits[i] - best);
            total += probs[i];
        }
    }
    for (float & p : probs) {
        p /= total;
    }
    return probs;
}

size_t multinomial(const std::vector<float> & probs, std::mt19937 & rng) {
    const float draw = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng);
    float cumulative = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        cumulative += probs[i];
        if (!(cumulative < draw)) {
            return i;
        }
    }
    return probs.size() - 1;
}

void mask_beyond_top_p(std::vector<float> & probs, std::vector<size_t> & order, float top_p) {
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return probs[a] > probs[b]; });
    float cumulative = 0.0f;
    for (size_t rank = 0; rank < order.size(); ++rank) {
        // HF keeps the first token whose cumulative crosses the threshold.
        if (rank > 0 && cumulative > top_p) {
            probs[order[rank]] = 0.0f;
        }
        cumulative += probs[order[rank]];
    }
}

int32_t argmax(const std::vector<float> & logits) {
    return (int32_t) (std::max_element(logits.begin(), logits.end()) - logits.begin());
}

int32_t sample_top_k(std::vector<float> & logits, float top_p, int top_k, std::mt19937 & rng) {
    const int k = std::min<int>(top_k, (int) logits.size());
    std::vector<size_t> indices(logits.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::nth_element(indices.begin(), indices.begin() + (k - 1), indices.end(),
            [&](size_t a, size_t b) { return logits[a] > logits[b]; });
    indices.resize(k);
    std::vector<float> subset(k);
    for (int i = 0; i < k; ++i) {
        subset[i] = logits[indices[i]];
    }
    std::vector<float> probs = softmax(subset);
    if (top_p > 0.0f && top_p < 1.0f) {
        std::vector<size_t> order(probs.size());
        std::iota(order.begin(), order.end(), 0);
        mask_beyond_top_p(probs, order, top_p);
        const float total = std::accumulate(probs.begin(), probs.end(), 0.0f);
        for (float & p : probs) {
            p /= total;
        }
    }
    return (int32_t) indices[multinomial(probs, rng)];
}

int32_t sample_top_p(std::vector<float> & logits, float top_p, std::mt19937 & rng) {
    std::vector<float> probs = softmax(logits);
    if (top_p > 0.0f && top_p < 1.0f) {
        std::vector<size_t> order(probs.size());
        std::iota(order.begin(), order.end(), 0);
        mask_beyond_top_p(probs, order, top_p);
        const float total = std::accumulate(probs.begin(), probs.end(), 0.0f);
        for (float & p : probs) {
            p /= total;
        }
    }
    return (int32_t) multinomial(probs, rng);
}

void scale_by_temperature(std::vector<float> & logits, float temperature) {
    if (temperature <= 0.0f) {
        return;
    }
    for (float & logit : logits) {
        logit /= temperature;
    }
}

} // namespace

std::vector<int32_t> apply_delay_pattern(const std::vector<int32_t> & codes, int n_frames,
                                         int n_vq, int pad_code) {
    const int delayed_frames = n_frames + n_vq - 1;
    std::vector<int32_t> delayed((size_t) delayed_frames * n_vq, pad_code);
    for (int t = 0; t < n_frames; ++t) {
        for (int c = 0; c < n_vq; ++c) {
            delayed[(size_t) (c + t) * n_vq + c] = codes[(size_t) t * n_vq + c];
        }
    }
    return delayed;
}

std::vector<int32_t> apply_de_delay_pattern(const std::vector<int32_t> & delayed, int delayed_frames,
                                            int n_vq, int pad_code) {
    if (delayed_frames + 1 <= n_vq) {
        return {};
    }
    const int n_frames = delayed_frames - n_vq + 1;
    std::vector<int32_t> codes((size_t) n_frames * n_vq, pad_code);
    for (int t = 0; t < n_frames; ++t) {
        for (int c = 0; c < n_vq; ++c) {
            codes[(size_t) t * n_vq + c] = delayed[(size_t) (c + t) * n_vq + c];
        }
    }
    return codes;
}

void apply_repetition_penalty(std::vector<float> & logits, const std::vector<int32_t> & history,
                              float penalty) {
    if (penalty == 1.0f || history.empty()) {
        return;
    }
    std::unordered_set<int32_t> seen(history.begin(), history.end());
    for (int32_t token : seen) {
        if (token < 0 || (size_t) token >= logits.size()) {
            continue;
        }
        float & logit = logits[(size_t) token];
        logit = logit > 0.0f ? logit / penalty : logit * penalty;
    }
}

int32_t sample_row(std::vector<float> logits, float top_p, int top_k, bool do_sample,
                   std::mt19937 & rng) {
    if (!do_sample) {
        return argmax(logits);
    }
    if (top_k > 0) {
        return sample_top_k(logits, top_p, top_k, rng);
    }
    return sample_top_p(logits, top_p, rng);
}

std::vector<int32_t> extract_audio_segments(const std::vector<int32_t> & codes, int n_frames,
                                            int n_vq, int pad_code) {
    std::vector<int32_t> merged;
    int segment_start = -1;
    for (int t = 0; t <= n_frames; ++t) {
        bool all_pad = true;
        if (t < n_frames) {
            for (int c = 0; c < n_vq; ++c) {
                if (codes[(size_t) t * n_vq + c] != pad_code) {
                    all_pad = false;
                    break;
                }
            }
        }
        if (!all_pad && segment_start < 0) {
            segment_start = t;
        }
        if (all_pad && segment_start >= 0) {
            merged.insert(merged.end(),
                    codes.begin() + (size_t) segment_start * n_vq,
                    codes.begin() + (size_t) t * n_vq);
            segment_start = -1;
        }
    }
    return merged;
}

DelayState::DelayState(const DelayConfig & config, const std::vector<DelayRow> & prompt_rows,
                       int32_t pad_token_id, int32_t im_end_token_id)
    : config_(config), pad_token_id_(pad_token_id), im_end_token_id_(im_end_token_id),
      delayed_length_(DRAIN_DISARMED) {
    if (prompt_rows.empty()) {
        fail("empty prompt");
    }
    text_history_.reserve(prompt_rows.size());
    audio_history_.reserve(prompt_rows.size() * (size_t) config_.n_vq);
    for (const DelayRow & row : prompt_rows) {
        text_history_.push_back(row.text);
        audio_history_.insert(audio_history_.end(), row.audio.begin(), row.audio.end());
    }
    const int32_t last = text_history_.back();
    if (last == config_.audio_start_token_id || last == config_.audio_assistant_gen_slot_token_id) {
        int64_t start_index = -1;
        for (int64_t i = (int64_t) text_history_.size() - 1; i >= 0; --i) {
            if (text_history_[i] == config_.audio_start_token_id) {
                start_index = i;
                break;
            }
        }
        if (start_index >= 0) {
            audio_length_ = (int32_t) ((int64_t) text_history_.size() - start_index);
            is_audio_ = true;
        }
    }
}

bool DelayState::channel_is_sampled(int channel) const {
    const bool pre_audio = channel < std::max<int32_t>(audio_length_, 0);
    const bool post_audio = delayed_length_ == DRAIN_DISARMED ||
            (int64_t) channel > std::max<int64_t>(delayed_length_ - 1, -1);
    return pre_audio && post_audio;
}

int32_t DelayState::next_text_token(const DelayLogits & logits, const SamplingConfig & sampling,
                                    std::mt19937 & rng) const {
    if (delayed_length_ < config_.n_vq) {
        return config_.audio_assistant_delay_slot_token_id;
    }
    if (delayed_length_ == config_.n_vq) {
        return config_.audio_end_token_id;
    }
    std::vector<float> row = logits.text;
    scale_by_temperature(row, sampling.text_temperature);
    if (!is_audio_) {
        row[(size_t) pad_token_id_] = MASKED;
        row[(size_t) config_.audio_assistant_gen_slot_token_id] = MASKED;
        row[(size_t) config_.audio_assistant_delay_slot_token_id] = MASKED;
        row[(size_t) config_.audio_end_token_id] = MASKED;
    } else {
        std::vector<float> masked(row.size(), MASKED);
        masked[(size_t) config_.audio_assistant_gen_slot_token_id] =
                row[(size_t) config_.audio_assistant_gen_slot_token_id];
        masked[(size_t) config_.audio_assistant_delay_slot_token_id] =
                row[(size_t) config_.audio_assistant_delay_slot_token_id];
        row = std::move(masked);
    }
    if (time_step_ == 0) {
        row[(size_t) config_.audio_assistant_delay_slot_token_id] = MASKED;
    }
    if (time_step_ <= config_.n_vq) {
        row[(size_t) im_end_token_id_] = MASKED;
    }
    return sample_row(std::move(row), sampling.text_top_p, sampling.text_top_k,
            sampling.text_temperature > 0.0f, rng);
}

std::vector<int32_t> DelayState::next_audio_codes(const DelayLogits & logits,
                                                  const SamplingConfig & sampling,
                                                  std::mt19937 & rng) const {
    std::vector<int32_t> codes((size_t) config_.n_vq, config_.audio_pad_code);
    const bool do_sample = sampling.audio_temperature > 0.0f;
    for (int channel = 0; channel < config_.n_vq; ++channel) {
        if (!channel_is_sampled(channel)) {
            continue;
        }
        std::vector<float> row = logits.audio[(size_t) channel];
        scale_by_temperature(row, sampling.audio_temperature);
        if (config_.audio_pad_code >= 0 && (size_t) config_.audio_pad_code < row.size()) {
            row[(size_t) config_.audio_pad_code] = MASKED;
        }
        apply_repetition_penalty(row, audio_history_, sampling.audio_repetition_penalty);
        codes[(size_t) channel] = sample_row(std::move(row), sampling.audio_top_p,
                sampling.audio_top_k, do_sample, rng);
    }
    return codes;
}

void DelayState::advance_counters(int32_t text_token) {
    if (text_token == config_.audio_start_token_id ||
        text_token == config_.audio_assistant_gen_slot_token_id ||
        text_token == config_.audio_assistant_delay_slot_token_id) {
        audio_length_ += 1;
    }
    if (text_token == config_.audio_end_token_id) {
        audio_length_ = 0;
    }
    if (delayed_length_ == DRAIN_DISARMED &&
        text_token == config_.audio_assistant_delay_slot_token_id) {
        delayed_length_ = 0;
    }
    if (delayed_length_ != DRAIN_DISARMED) {
        delayed_length_ += 1;
    }
    if (delayed_length_ != DRAIN_DISARMED && delayed_length_ > config_.n_vq) {
        delayed_length_ = DRAIN_DISARMED;
    }
    time_step_ += 1;
}

DelayRow DelayState::step(const DelayLogits & logits, const SamplingConfig & sampling,
                          std::mt19937 & rng) {
    DelayRow row;
    if (stopping_) {
        row.text = pad_token_id_;
        row.audio.assign((size_t) config_.n_vq, config_.audio_pad_code);
        return row;
    }
    row.text = next_text_token(logits, sampling, rng);
    if (row.text == config_.audio_start_token_id) {
        is_audio_ = true;
    }
    if (row.text == im_end_token_id_) {
        stopping_ = true;
    }
    row.audio = next_audio_codes(logits, sampling, rng);
    if (row.text == config_.audio_end_token_id) {
        is_audio_ = false;
    }
    advance_counters(row.text);
    text_history_.push_back(row.text);
    audio_history_.insert(audio_history_.end(), row.audio.begin(), row.audio.end());
    return row;
}

std::vector<int32_t> DelayState::generated_audio(int prompt_frames) const {
    const size_t begin = (size_t) prompt_frames * config_.n_vq;
    if (begin >= audio_history_.size()) {
        return {};
    }
    const std::vector<int32_t> delayed(audio_history_.begin() + begin, audio_history_.end());
    const int delayed_frames = (int) (delayed.size() / (size_t) config_.n_vq);
    const std::vector<int32_t> codes = apply_de_delay_pattern(delayed, delayed_frames,
            config_.n_vq, config_.audio_pad_code);
    return extract_audio_segments(codes, (int) (codes.size() / (size_t) config_.n_vq),
            config_.n_vq, config_.audio_pad_code);
}

} // namespace tts_cpp::moss::detail
