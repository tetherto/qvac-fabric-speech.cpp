#include "moss/transcribe_networks.h"
#include "moss/transcribe_tensors.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace tts_cpp::moss::detail {
namespace {

constexpr int DECODER_GRAPH_NODES = 8192;
constexpr float FFN_DOWN_ACCUMULATION_SCALE = 64.0f;
constexpr size_t CACHE_TENSOR_SLACK = 8;
constexpr int CACHE_ALIGNMENT = 256;

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("moss transcribe: " + message);
}

int aligned_context(int requested) {
    return (requested + CACHE_ALIGNMENT - 1) / CACHE_ALIGNMENT * CACHE_ALIGNMENT;
}

std::string text_name(int layer, const char * suffix) {
    return "text.blk." + std::to_string(layer) + "." + suffix;
}

void validate_text_layer(const TranscribeModel & model, const TranscribeTextConfig & config, int layer) {
    const int64_t q_dim = (int64_t) config.n_heads * config.head_dim;
    const int64_t kv_dim = (int64_t) config.n_kv_heads * config.head_dim;
    require_transcribe_f32(model.tensor(text_name(layer, "attn_norm.weight")), config.n_embd);
    require_transcribe_weight(model.tensor(text_name(layer, "attn_q.weight")), config.n_embd, q_dim);
    require_transcribe_weight(model.tensor(text_name(layer, "attn_k.weight")), config.n_embd, kv_dim);
    require_transcribe_weight(model.tensor(text_name(layer, "attn_v.weight")), config.n_embd, kv_dim);
    require_transcribe_weight(model.tensor(text_name(layer, "attn_output.weight")), q_dim, config.n_embd);
    require_transcribe_f32(model.tensor(text_name(layer, "attn_q_norm.weight")), config.head_dim);
    require_transcribe_f32(model.tensor(text_name(layer, "attn_k_norm.weight")), config.head_dim);
    require_transcribe_f32(model.tensor(text_name(layer, "ffn_norm.weight")), config.n_embd);
    require_transcribe_weight(model.tensor(text_name(layer, "ffn_gate.weight")), config.n_embd, config.n_ff);
    require_transcribe_weight(model.tensor(text_name(layer, "ffn_up.weight")), config.n_embd, config.n_ff);
    require_transcribe_weight(model.tensor(text_name(layer, "ffn_down.weight")), config.n_ff, config.n_embd);
}

void validate_text_layers(const TranscribeModel & model, const TranscribeTextConfig & config) {
    for (int layer = 0; layer < config.n_layers; ++layer) {
        validate_text_layer(model, config, layer);
    }
}

void require_token_in_vocabulary(int32_t id, int vocab, const char * name) {
    if (id < 0 || id >= vocab) {
        fail(std::string(name) + " is outside the text vocabulary");
    }
}

void validate_prompt_tokens(const TranscribeConfig & config) {
    const int vocab = config.text.vocab;
    require_token_in_vocabulary(config.tokens.audio_start, vocab, "audio_start");
    require_token_in_vocabulary(config.tokens.audio_end, vocab, "audio_end");
    require_token_in_vocabulary(config.tokens.audio_pad, vocab, "audio_pad");
    require_token_in_vocabulary(config.tokens.im_start, vocab, "im_start");
    require_token_in_vocabulary(config.tokens.im_end, vocab, "im_end");
    for (int32_t id : config.default_prompt_ids) {
        require_token_in_vocabulary(id, vocab, "default prompt token");
    }
}

template <typename T>
void set_input(ggml_tensor * tensor, const std::vector<T> & values) {
    if (tensor != nullptr && tensor->buffer != nullptr) {
        ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(T));
    }
}

struct BatchPlan {
    std::vector<int32_t> ids;
    std::vector<float> audio;
    std::vector<float> audio_mask;
    std::vector<float> text_mask;
    bool has_audio = false;
};

} // namespace

void validate_transcribe_decoder(const TranscribeModel & model) {
    const TranscribeConfig & config = model.config();
    require_transcribe_weight(model.tensor("text.token_embd.weight"), config.text.n_embd, config.text.vocab);
    require_transcribe_f32(model.tensor("text.output_norm.weight"), config.text.n_embd);
    validate_text_layers(model, config.text);
    validate_prompt_tokens(config);
}

struct TranscribeDecoder::Impl {
    TranscribeModel & model;
    const TranscribeTextConfig & config;
    int32_t audio_pad = 0;
    int n_ctx = 0;
    int pos = 0;
    int audio_cursor = 0;
    ggml_context * state = nullptr;
    ggml_backend_buffer_t state_buffer = nullptr;
    std::vector<ggml_tensor *> cache_k;
    std::vector<ggml_tensor *> cache_v;

    Impl(TranscribeModel & owner, int context)
        : model(owner), config(owner.config().text), audio_pad(owner.config().tokens.audio_pad) {
        if (context < 1 || context > config.n_ctx_train) {
            fail("decoder context must be 1.." + std::to_string(config.n_ctx_train));
        }
        n_ctx = aligned_context(context);
        allocate_cache();
    }

    ~Impl() {
        if (state_buffer) ggml_backend_buffer_free(state_buffer);
        if (state) ggml_free(state);
    }

    int64_t kv_dim() const {
        return (int64_t) config.head_dim * config.n_kv_heads;
    }

    void create_cache_tensors() {
        cache_k.resize((size_t) config.n_layers);
        cache_v.resize((size_t) config.n_layers);
        for (int layer = 0; layer < config.n_layers; ++layer) {
            cache_k[(size_t) layer] = ggml_new_tensor_2d(state, GGML_TYPE_F16, kv_dim(), n_ctx);
            cache_v[(size_t) layer] = ggml_new_tensor_2d(state, GGML_TYPE_F16, n_ctx, kv_dim());
        }
    }

    void allocate_cache() {
        state = ggml_init({(2 * (size_t) config.n_layers + CACHE_TENSOR_SLACK) * ggml_tensor_overhead(), nullptr, true});
        if (!state) {
            fail("KV cache context allocation failed");
        }
        create_cache_tensors();
        state_buffer = ggml_backend_alloc_ctx_tensors(state, model.backend());
        if (!state_buffer) {
            fail("KV cache allocation failed; the audio is too long for this device");
        }
        ggml_backend_buffer_clear(state_buffer, 0);
    }

    ggml_tensor * weight(int layer, const char * suffix) const {
        return model.tensor(text_name(layer, suffix));
    }

    ggml_tensor * rms_norm(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * scale) const {
        return ggml_mul(ctx, ggml_rms_norm(ctx, cur, config.rms_eps), scale);
    }

    ggml_tensor * rope(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * positions) const {
        return ggml_rope_ext(ctx, cur, positions, nullptr, config.head_dim, GGML_ROPE_TYPE_NEOX,
                0, config.rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    }

    ggml_tensor * key_rows(ggml_context * ctx, int layer, int64_t first, int64_t count) const {
        ggml_tensor * cache = cache_k[(size_t) layer];
        return ggml_view_2d(ctx, cache, kv_dim(), count, cache->nb[1], first * cache->nb[1]);
    }

    ggml_tensor * value_columns(ggml_context * ctx, int layer, int64_t first, int64_t count) const {
        ggml_tensor * cache = cache_v[(size_t) layer];
        return ggml_view_2d(ctx, cache, count, kv_dim(), cache->nb[1], first * ggml_element_size(cache));
    }

    ggml_tensor * cached_values(ggml_context * ctx, int layer, int64_t total) const {
        ggml_tensor * cache = cache_v[(size_t) layer];
        return ggml_view_3d(ctx, cache, total, config.head_dim, config.n_kv_heads, cache->nb[1],
                cache->nb[1] * config.head_dim, 0);
    }

    void store_cache(SfxGraph & graph, int layer, ggml_tensor * k, ggml_tensor * v, int64_t tokens) const {
        ggml_context * ctx = graph.ctx();
        ggml_tensor * k_rows = ggml_reshape_2d(ctx, ggml_cont(ctx, k), kv_dim(), tokens);
        ggml_tensor * v_rows = ggml_reshape_2d(ctx, ggml_cont(ctx, v), kv_dim(), tokens);
        ggml_build_forward_expand(graph.graph(), ggml_cpy(ctx, k_rows, key_rows(ctx, layer, pos, tokens)));
        ggml_build_forward_expand(graph.graph(), ggml_cpy(ctx, ggml_transpose(ctx, v_rows),
                value_columns(ctx, layer, pos, tokens)));
    }

    ggml_tensor * attention(SfxGraph & graph, int layer, ggml_tensor * cur, int64_t tokens, ggml_tensor * positions,
                            ggml_tensor * mask) const {
        ggml_context * ctx = graph.ctx();
        const int64_t total = pos + tokens;
        ggml_tensor * q = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, weight(layer, "attn_q.weight"), cur),
                config.head_dim, config.n_heads, tokens);
        ggml_tensor * k = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, weight(layer, "attn_k.weight"), cur),
                config.head_dim, config.n_kv_heads, tokens);
        ggml_tensor * v = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, weight(layer, "attn_v.weight"), cur),
                config.head_dim, config.n_kv_heads, tokens);
        q = rope(ctx, rms_norm(ctx, q, weight(layer, "attn_q_norm.weight")), positions);
        k = rope(ctx, rms_norm(ctx, k, weight(layer, "attn_k_norm.weight")), positions);
        store_cache(graph, layer, k, v, tokens);

        ggml_tensor * keys = ggml_reshape_3d(ctx, key_rows(ctx, layer, 0, total), config.head_dim,
                config.n_kv_heads, total);
        q = ggml_permute(ctx, q, 0, 2, 1, 3);
        keys = ggml_permute(ctx, keys, 0, 2, 1, 3);
        ggml_tensor * scores = ggml_soft_max_ext(ctx, ggml_mul_mat(ctx, keys, q), mask,
                1.0f / std::sqrt((float) config.head_dim), 0.0f);
        ggml_tensor * attended = ggml_mul_mat(ctx, cached_values(ctx, layer, total), scores);
        attended = ggml_cont_2d(ctx, ggml_permute(ctx, attended, 0, 2, 1, 3),
                (int64_t) config.head_dim * config.n_heads, tokens);
        return ggml_mul_mat(ctx, weight(layer, "attn_output.weight"), attended);
    }

    ggml_tensor * feed_forward(ggml_context * ctx, int layer, ggml_tensor * cur) const {
        ggml_tensor * gate = ggml_silu(ctx, ggml_mul_mat(ctx, weight(layer, "ffn_gate.weight"), cur));
        ggml_tensor * up = ggml_mul_mat(ctx, weight(layer, "ffn_up.weight"), cur);
        ggml_tensor * shrunk = ggml_scale(ctx, ggml_mul(ctx, gate, up), 1.0f / FFN_DOWN_ACCUMULATION_SCALE);
        return ggml_scale(ctx, ggml_mul_mat(ctx, weight(layer, "ffn_down.weight"), shrunk),
                FFN_DOWN_ACCUMULATION_SCALE);
    }

    ggml_tensor * block(SfxGraph & graph, int layer, ggml_tensor * cur, int64_t tokens, ggml_tensor * positions,
                        ggml_tensor * mask) const {
        ggml_context * ctx = graph.ctx();
        ggml_tensor * attended = attention(graph, layer, rms_norm(ctx, cur, weight(layer, "attn_norm.weight")),
                tokens, positions, mask);
        cur = ggml_add(ctx, cur, attended);
        ggml_tensor * fed = feed_forward(ctx, layer, rms_norm(ctx, cur, weight(layer, "ffn_norm.weight")));
        return ggml_add(ctx, cur, fed);
    }

    ggml_tensor * blocks(SfxGraph & graph, ggml_tensor * cur, int64_t tokens, ggml_tensor * positions,
                         ggml_tensor * mask) const {
        for (int layer = 0; layer < config.n_layers; ++layer) {
            cur = block(graph, layer, cur, tokens, positions, mask);
        }
        return cur;
    }

    std::vector<int32_t> positions(int64_t tokens) const {
        std::vector<int32_t> values((size_t) tokens);
        for (int64_t i = 0; i < tokens; ++i) {
            values[(size_t) i] = (int32_t) (pos + i);
        }
        return values;
    }

    std::vector<float> causal_mask(int64_t tokens) const {
        const int64_t total = pos + tokens;
        std::vector<float> mask((size_t) (tokens * total), -std::numeric_limits<float>::infinity());
        for (int64_t query = 0; query < tokens; ++query) {
            const auto row = mask.begin() + (std::ptrdiff_t) (query * total);
            std::fill(row, row + (std::ptrdiff_t) (pos + query + 1), 0.0f);
        }
        return mask;
    }

    ggml_tensor * logits(ggml_context * ctx, ggml_tensor * cur, int64_t tokens) const {
        ggml_tensor * last = ggml_view_2d(ctx, cur, config.n_embd, 1, cur->nb[1], (size_t) (tokens - 1) * cur->nb[1]);
        last = rms_norm(ctx, last, model.tensor("text.output_norm.weight"));
        return ggml_mul_mat(ctx, model.tensor("text.token_embd.weight"), last);
    }

    ggml_tensor * embed(SfxGraph & graph, const BatchPlan & plan, ggml_tensor * ids, ggml_tensor * audio,
                        ggml_tensor * audio_mask, ggml_tensor * text_mask) const {
        ggml_context * ctx = graph.ctx();
        ggml_tensor * text = ggml_get_rows(ctx, model.tensor("text.token_embd.weight"), ids);
        if (!plan.has_audio) {
            return text;
        }
        return ggml_add(ctx, ggml_mul(ctx, text, text_mask), ggml_mul(ctx, audio, audio_mask));
    }

    void copy_audio_row(BatchPlan & plan, size_t slot, const std::vector<float> & audio_embeddings) {
        const size_t width = (size_t) config.n_embd;
        const size_t offset = (size_t) audio_cursor * width;
        if (offset + width > audio_embeddings.size()) {
            fail("prompt has more audio placeholders than audio embeddings");
        }
        std::copy(audio_embeddings.begin() + (std::ptrdiff_t) offset,
                  audio_embeddings.begin() + (std::ptrdiff_t) (offset + width),
                  plan.audio.begin() + (std::ptrdiff_t) (slot * width));
        plan.audio_mask[slot] = 1.0f;
        plan.text_mask[slot] = 0.0f;
        plan.has_audio = true;
        audio_cursor++;
    }

    void place_token(BatchPlan & plan, size_t slot, int32_t id, const std::vector<float> & audio_embeddings,
                     bool inject_audio) {
        if (id < 0 || id >= config.vocab) {
            fail("token id outside the text vocabulary");
        }
        plan.ids[slot] = id;
        if (inject_audio && id == audio_pad) {
            copy_audio_row(plan, slot, audio_embeddings);
        }
    }

    void place_tokens(BatchPlan & plan, const int32_t * ids, const std::vector<float> & audio_embeddings,
                      bool inject_audio) {
        for (size_t slot = 0; slot < plan.ids.size(); ++slot) {
            place_token(plan, slot, ids[slot], audio_embeddings, inject_audio);
        }
    }

    BatchPlan plan_batch(const int32_t * ids, size_t count, const std::vector<float> & audio_embeddings,
                         bool inject_audio) {
        BatchPlan plan;
        plan.ids.resize(count);
        plan.audio.assign(count * (size_t) config.n_embd, 0.0f);
        plan.audio_mask.assign(count, 0.0f);
        plan.text_mask.assign(count, 1.0f);
        place_tokens(plan, ids, audio_embeddings, inject_audio);
        return plan;
    }

    std::vector<float> forward(const BatchPlan & plan, bool want_logits) {
        const int64_t tokens = (int64_t) plan.ids.size();
        if (pos + tokens > n_ctx) {
            fail("decoder context overflow");
        }
        SfxGraph graph(DECODER_GRAPH_NODES);
        ggml_tensor * ids = graph.input_i32(tokens);
        ggml_tensor * positions_input = graph.input_i32(tokens);
        ggml_tensor * mask = graph.input_f32(pos + tokens, tokens);
        ggml_tensor * audio = plan.has_audio ? graph.input_f32(config.n_embd, tokens) : nullptr;
        ggml_tensor * audio_mask = plan.has_audio ? graph.input_f32(1, tokens) : nullptr;
        ggml_tensor * text_mask = plan.has_audio ? graph.input_f32(1, tokens) : nullptr;
        ggml_tensor * cur = embed(graph, plan, ids, audio, audio_mask, text_mask);
        cur = blocks(graph, cur, tokens, positions_input, mask);
        ggml_tensor * output = want_logits ? logits(graph.ctx(), cur, tokens) : nullptr;
        if (output) {
            ggml_set_output(output);
            ggml_build_forward_expand(graph.graph(), output);
        }

        model.allocate(graph);
        const std::vector<int32_t> position_data = positions(tokens);
        const std::vector<float> mask_data = causal_mask(tokens);
        set_input(ids, plan.ids);
        set_input(positions_input, position_data);
        set_input(mask, mask_data);
        set_input(audio, plan.audio);
        set_input(audio_mask, plan.audio_mask);
        set_input(text_mask, plan.text_mask);
        model.compute(graph);
        pos += (int) tokens;
        std::vector<float> values;
        if (output) {
            values.resize((size_t) config.vocab);
            ggml_backend_tensor_get(output, values.data(), 0, values.size() * sizeof(float));
        }
        return values;
    }

    std::vector<float> prefill_batches(const std::vector<int32_t> & ids, const std::vector<float> & audio_embeddings,
                                       size_t batch) {
        std::vector<float> values;
        for (size_t first = 0; first < ids.size(); first += batch) {
            const size_t count = std::min(batch, ids.size() - first);
            const bool last = first + count == ids.size();
            values = forward(plan_batch(ids.data() + first, count, audio_embeddings, true), last);
        }
        return values;
    }

    std::vector<float> prefill(const std::vector<int32_t> & ids, const std::vector<float> & audio_embeddings,
                               int batch_tokens) {
        if (ids.empty() || batch_tokens < 1) {
            fail("prefill needs at least one token and a positive batch size");
        }
        if (audio_embeddings.size() % (size_t) config.n_embd != 0) {
            fail("audio embeddings do not match the decoder width");
        }
        audio_cursor = 0;
        std::vector<float> values = prefill_batches(ids, audio_embeddings, (size_t) batch_tokens);
        if ((size_t) audio_cursor * config.n_embd != audio_embeddings.size()) {
            fail("prompt audio placeholders do not match the audio embeddings");
        }
        return values;
    }

    std::vector<float> step(int32_t id) {
        return forward(plan_batch(&id, 1, {}, false), true);
    }
};

TranscribeDecoder::TranscribeDecoder(TranscribeModel & model, int n_ctx) : impl_(new Impl(model, n_ctx)) {}

TranscribeDecoder::~TranscribeDecoder() = default;

std::vector<float> TranscribeDecoder::prefill(const std::vector<int32_t> & ids,
                                              const std::vector<float> & audio_embeddings, int batch_tokens) {
    return impl_->prefill(ids, audio_embeddings, batch_tokens);
}

std::vector<float> TranscribeDecoder::step(int32_t id) {
    return impl_->step(id);
}

int TranscribeDecoder::position() const { return impl_->pos; }
int TranscribeDecoder::context() const { return impl_->n_ctx; }

} // namespace tts_cpp::moss::detail
