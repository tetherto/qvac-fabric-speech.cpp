// Conformance suite for the MOSS Delay generation logic, mirroring the
// upstream reference self-test: delay/de-delay round trip, the in-loop delay
// state machine, sampling primitives, segment extraction, and prompt packing.
// Pure CPU logic; no model fixtures required.

#include "moss/frontend.h"
#include "moss/generation.h"

#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace tts_cpp::moss::detail;

namespace {

int failures = 0;

void check(bool condition, const std::string & label) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", label.c_str());
        failures++;
    }
}

DelayConfig test_config() {
    DelayConfig config;
    config.n_vq = 3;
    config.text_vocab = 200;
    config.audio_vocab = 11;
    config.audio_pad_code = 10;
    config.audio_start_token_id = 100;
    config.audio_end_token_id = 101;
    config.audio_user_slot_token_id = 102;
    config.audio_assistant_gen_slot_token_id = 103;
    config.audio_assistant_delay_slot_token_id = 104;
    return config;
}

constexpr int32_t PAD_TOKEN = 0;
constexpr int32_t IM_END_TOKEN = 1;

DelayLogits uniform_logits(const DelayConfig & config) {
    DelayLogits logits;
    logits.text.assign((size_t) config.text_vocab, 0.0f);
    logits.audio.assign((size_t) config.n_vq,
            std::vector<float>((size_t) config.audio_vocab, 0.0f));
    return logits;
}

DelayLogits logits_preferring(const DelayConfig & config, int32_t text_token, int32_t audio_code) {
    DelayLogits logits = uniform_logits(config);
    logits.text[(size_t) text_token] = 50.0f;
    for (auto & channel : logits.audio) {
        channel[(size_t) audio_code] = 50.0f;
    }
    return logits;
}

SamplingConfig greedy_sampling() {
    SamplingConfig sampling;
    sampling.text_temperature = 0.0f;
    sampling.audio_temperature = 0.0f;
    return sampling;
}

std::vector<DelayRow> seeded_prompt(const DelayConfig & config) {
    std::vector<DelayRow> rows(2);
    rows[0].text = 42;
    rows[0].audio.assign((size_t) config.n_vq, config.audio_pad_code);
    rows[1].text = config.audio_start_token_id;
    rows[1].audio.assign((size_t) config.n_vq, config.audio_pad_code);
    return rows;
}

void test_delay_round_trip() {
    const int n_vq = 3;
    const int pad = 10;
    const std::vector<int32_t> codes = {0, 1, 2, 3, 4, 5};
    const std::vector<int32_t> delayed = apply_delay_pattern(codes, 2, n_vq, pad);
    check(delayed.size() == 4 * 3, "delayed frame count");
    const std::vector<int32_t> expected = {
        0, pad, pad,
        3, 1, pad,
        pad, 4, 2,
        pad, pad, 5,
    };
    check(delayed == expected, "delay pattern layout");
    check(apply_de_delay_pattern(delayed, 4, n_vq, pad) == codes, "de-delay round trip");
    check(apply_de_delay_pattern({pad, pad, pad}, 1, n_vq, pad).empty(), "de-delay short input");
}

void test_repetition_penalty() {
    std::vector<float> logits = {2.0f, -2.0f, 1.0f};
    apply_repetition_penalty(logits, {0, 1, 1}, 2.0f);
    check(logits[0] == 1.0f, "positive logit divided");
    check(logits[1] == -4.0f, "negative logit multiplied");
    check(logits[2] == 1.0f, "unseen logit untouched");
}

void test_sampling_determinism() {
    std::vector<float> logits = {0.1f, 5.0f, 0.2f, 4.9f, 0.3f};
    std::mt19937 rng_a(123);
    std::mt19937 rng_b(123);
    const int32_t a = sample_row(logits, 0.8f, 2, true, rng_a);
    const int32_t b = sample_row(logits, 0.8f, 2, true, rng_b);
    check(a == b, "top-k sampling is deterministic under one seed");
    check(a == 1 || a == 3, "top-k keeps only the two best tokens");
    std::mt19937 rng_c(7);
    check(sample_row({0.0f, 100.0f, 0.0f}, 1.0f, 0, true, rng_c) == 1,
            "dominant logit wins the multinomial");
}

void test_argmax_path() {
    std::mt19937 rng(1);
    check(sample_row({0.5f, 3.0f, 1.0f}, 1.0f, 50, false, rng) == 1, "argmax when not sampling");
}

void test_state_machine_drain() {
    const DelayConfig config = test_config();
    const SamplingConfig sampling = greedy_sampling();
    std::mt19937 rng(1234);
    DelayState state(config, seeded_prompt(config), PAD_TOKEN, IM_END_TOKEN);

    DelayRow row = state.step(logits_preferring(config, config.audio_assistant_gen_slot_token_id, 4),
            sampling, rng);
    check(row.text == config.audio_assistant_gen_slot_token_id, "step 0 keeps generating");
    check(row.audio[0] == 4, "channel 0 opens on step 0");
    check(row.audio[1] == config.audio_pad_code, "channel 1 still closed on step 0");

    row = state.step(logits_preferring(config, config.audio_assistant_gen_slot_token_id, 5),
            sampling, rng);
    check(row.audio[0] == 5 && row.audio[1] == 5, "channels ramp in one per step");
    check(row.audio[2] == config.audio_pad_code, "channel 2 closed until step 2");

    row = state.step(logits_preferring(config, config.audio_assistant_delay_slot_token_id, 6),
            sampling, rng);
    check(row.text == config.audio_assistant_delay_slot_token_id, "model arms the drain");
    check(row.audio[0] == 6 && row.audio[1] == 6 && row.audio[2] == 6, "all channels open");

    row = state.step(logits_preferring(config, config.audio_assistant_gen_slot_token_id, 7),
            sampling, rng);
    check(row.text == config.audio_assistant_delay_slot_token_id, "drain forces delay slots");
    check(row.audio[0] == config.audio_pad_code, "channel 0 drains first");
    check(row.audio[1] == 7 && row.audio[2] == 7, "later channels still open while draining");

    row = state.step(logits_preferring(config, config.audio_assistant_gen_slot_token_id, 8),
            sampling, rng);
    check(row.text == config.audio_assistant_delay_slot_token_id, "second drain step");
    check(row.audio[1] == config.audio_pad_code && row.audio[2] == 8, "drain advances by channel");

    row = state.step(logits_preferring(config, config.audio_assistant_gen_slot_token_id, 9),
            sampling, rng);
    check(row.text == config.audio_end_token_id, "audio end is forced after n_vq drain steps");

    row = state.step(logits_preferring(config, IM_END_TOKEN, 9), sampling, rng);
    check(row.text == IM_END_TOKEN, "im_end is reachable after the drain");
    check(state.stopping(), "im_end stops the generation");

    const std::vector<int32_t> audio = state.generated_audio(2);
    check(!audio.empty(), "generated audio survives de-delay");
    check(audio.size() % config.n_vq == 0, "generated audio is frame aligned");
    for (int32_t code : audio) {
        check(code != config.audio_pad_code, "no pad codes leak into decoded segments");
    }
}

void test_early_stop_masks() {
    const DelayConfig config = test_config();
    const SamplingConfig sampling = greedy_sampling();
    std::mt19937 rng(1);
    DelayState state(config, seeded_prompt(config), PAD_TOKEN, IM_END_TOKEN);
    const DelayRow row = state.step(logits_preferring(config, IM_END_TOKEN, 2), sampling, rng);
    check(row.text != IM_END_TOKEN, "im_end is masked during the ramp-in");
}

void test_segment_extraction() {
    const int n_vq = 2;
    const int pad = 9;
    const std::vector<int32_t> codes = {
        1, 2,
        pad, pad,
        3, 4,
        5, 6,
        pad, pad,
    };
    const std::vector<int32_t> merged = extract_audio_segments(codes, 5, n_vq, pad);
    check(merged == std::vector<int32_t>({1, 2, 3, 4, 5, 6}), "segments merged without pads");
    check(extract_audio_segments({pad, pad}, 1, n_vq, pad).empty(), "all-pad input yields nothing");
}

std::vector<int32_t> fake_encode(const std::string & span) {
    std::vector<int32_t> ids;
    for (char c : span) {
        ids.push_back((int32_t) (unsigned char) c % 90);
    }
    return ids;
}

void test_prompt_rows() {
    const DelayConfig config = test_config();
    PromptTokens tokens;
    tokens.pad = PAD_TOKEN;
    tokens.im_start = 2;
    tokens.im_end = 3;

    const std::vector<DelayRow> plain = build_prompt_rows(config, tokens, fake_encode,
            "hola", "es", {}, 0);
    check(plain.size() > 4, "plain prompt has rows");
    check(plain.front().text == tokens.im_start, "prompt opens with im_start");
    check(plain.back().text == config.audio_start_token_id, "prompt ends with the audio seed row");
    for (const DelayRow & row : plain) {
        check((int) row.audio.size() == config.n_vq, "every row carries n_vq channels");
    }
    for (size_t i = 0; i + 1 < plain.size(); ++i) {
        for (int32_t code : plain[i].audio) {
            check(code == config.audio_pad_code, "plain prompt rows carry only pad codes");
        }
    }

    const int reference_frames = 4;
    std::vector<int32_t> reference_codes((size_t) reference_frames * config.n_vq);
    for (size_t i = 0; i < reference_codes.size(); ++i) {
        reference_codes[i] = (int32_t) (i % 7);
    }
    const std::vector<DelayRow> cloned = build_prompt_rows(config, tokens, fake_encode,
            "hola", "es", reference_codes, reference_frames);
    int slot_rows = 0;
    int rows_with_codes = 0;
    for (const DelayRow & row : cloned) {
        if (row.text == config.audio_user_slot_token_id) {
            slot_rows++;
        }
        for (int32_t code : row.audio) {
            if (code != config.audio_pad_code) {
                rows_with_codes++;
                break;
            }
        }
    }
    check(slot_rows == reference_frames + config.n_vq - 1,
            "placeholder expands to frames + n_vq - 1 slots");
    check(rows_with_codes == reference_frames + config.n_vq - 1,
            "delayed reference codes land on the slot rows");
}

} // namespace

int main() {
    test_delay_round_trip();
    test_repetition_penalty();
    test_sampling_determinism();
    test_argmax_path();
    test_state_machine_drain();
    test_early_stop_masks();
    test_segment_extraction();
    test_prompt_rows();
    if (failures == 0) {
        std::printf("moss generation conformance: OK\n");
        return 0;
    }
    std::fprintf(stderr, "moss generation conformance: %d failures\n", failures);
    return 1;
}
