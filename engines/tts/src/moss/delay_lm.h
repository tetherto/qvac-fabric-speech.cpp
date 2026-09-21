#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tts_cpp::moss::detail {

struct DelayConfig {
    int n_layers      = 0;
    int n_embd        = 0;
    int n_heads       = 0;
    int n_kv_heads    = 0;
    int head_dim      = 0;
    int n_ff          = 0;
    int n_ctx_train   = 0;
    float rope_base   = 10000.0f;
    float rms_eps     = 1e-6f;
    int text_vocab    = 0;
    int audio_vocab   = 0;
    int n_vq          = 0;
    int audio_pad_code = 0;
    int audio_start_token_id = 0;
    int audio_end_token_id   = 0;
    int audio_user_slot_token_id           = 0;
    int audio_assistant_gen_slot_token_id  = 0;
    int audio_assistant_delay_slot_token_id = 0;
    int sampling_rate = 0;
};

struct DelayRow {
    int32_t text = 0;
    std::vector<int32_t> audio;
};

struct DelayLogits {
    std::vector<float> text;
    std::vector<std::vector<float>> audio;
};

struct DelayMemory {
    size_t weights = 0;
    size_t kv_cache = 0;
    size_t compute = 0;
};

// The MOSS Delay backbone: a Qwen3-style decoder whose input embedding is the
// text-token embedding plus one embedding per audio codebook channel, and
// whose output is one text head plus one head per channel. One instance owns
// the weights, the backend, and a bounded KV cache; calls must be serialized
// by the caller. prefill() consumes the prompt and step() advances one row,
// both returning the last row's logits for every head.
class DelayLM {
public:
    DelayLM(const std::string & path, bool use_gpu, int n_threads, int n_ctx);
    ~DelayLM();
    DelayLM(const DelayLM &) = delete;
    DelayLM & operator=(const DelayLM &) = delete;

    static DelayMemory measure(const std::string & path, int n_threads, int n_ctx, int prefill_rows);

    const DelayConfig & config() const;
    const char * backend_name() const;
    int position() const;
    int context() const;
    void reset();

    // Tokenizer vocabulary exported alongside the weights.
    std::vector<std::string> tokenizer_tokens() const;
    std::vector<std::string> tokenizer_merges() const;
    int32_t token_id(const std::string & token) const;

    DelayLogits prefill(const std::vector<DelayRow> & rows);
    DelayLogits step(const DelayRow & row);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tts_cpp::moss::detail
