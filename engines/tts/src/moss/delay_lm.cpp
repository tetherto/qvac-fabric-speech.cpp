#include "moss/delay_lm.h"

#include "backend_selection.h"
#include "backend_util.h"
#include "gguf_stream.h"
#include "sched_dispatch.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace tts_cpp::moss::detail {
namespace {

constexpr const char * ARCH = "moss-tts-delay";
constexpr int GRAPH_NODES = 8192;
constexpr int MIN_CONTEXT = 32;
constexpr int MAX_CONTEXT = 32768;

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("moss delay LM: " + message);
}

struct Layer {
    ggml_tensor * attn_norm   = nullptr;
    ggml_tensor * wq          = nullptr;
    ggml_tensor * wk          = nullptr;
    ggml_tensor * wv          = nullptr;
    ggml_tensor * wo          = nullptr;
    ggml_tensor * attn_q_norm = nullptr;
    ggml_tensor * attn_k_norm = nullptr;
    ggml_tensor * ffn_norm    = nullptr;
    ggml_tensor * ffn_gate    = nullptr;
    ggml_tensor * ffn_up      = nullptr;
    ggml_tensor * ffn_down    = nullptr;
};

struct GraphInputI32 {
    ggml_tensor * tensor;
    std::vector<int32_t> data;
};

struct GraphInputF32 {
    ggml_tensor * tensor;
    std::vector<float> data;
};

} // namespace

struct DelayLM::Impl {
    gguf_context * file = nullptr;
    ggml_context * metadata = nullptr;
    ggml_context * weights = nullptr;
    ggml_context * state = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t weight_buffer = nullptr;
    ggml_backend_buffer_t state_buffer = nullptr;
    ::tts_cpp::detail::sched_fallback sched;

    DelayConfig config;
    int n_threads = 1;
    int n_ctx = 0;
    int pos = 0;

    ggml_tensor * tok_embd = nullptr;
    ggml_tensor * output_norm = nullptr;
    ggml_tensor * output_head = nullptr;
    std::vector<ggml_tensor *> tok_embd_audio;
    std::vector<ggml_tensor *> output_audio;
    std::vector<Layer> layers;
    std::vector<ggml_tensor *> cache_k;
    std::vector<ggml_tensor *> cache_v;

    ggml_context * graph_ctx = nullptr;
    std::vector<GraphInputI32> inputs_i32;
    std::vector<GraphInputF32> inputs_f32;

    ~Impl() {
        ::tts_cpp::detail::sched_fallback_free(sched);
        if (graph_ctx) ggml_free(graph_ctx);
        if (state_buffer) ggml_backend_buffer_free(state_buffer);
        if (weight_buffer) ggml_backend_buffer_free(weight_buffer);
        if (state) ggml_free(state);
        if (weights) ggml_free(weights);
        if (metadata) ggml_free(metadata);
        if (file) gguf_free(file);
        if (backend) ggml_backend_free(backend);
    }

    int64_t find_key(const std::string & name) const {
        return gguf_find_key(file, name.c_str());
    }

    uint32_t meta_u32(const std::string & name) const {
        const int64_t key = find_key(name);
        if (key < 0) {
            fail("missing metadata: " + name);
        }
        switch (gguf_get_kv_type(file, key)) {
            case GGUF_TYPE_UINT32: return gguf_get_val_u32(file, key);
            case GGUF_TYPE_INT32:  return (uint32_t) gguf_get_val_i32(file, key);
            default:               fail("unsupported metadata type: " + name);
        }
    }

    uint32_t meta_u32_or(const std::string & name, uint32_t fallback) const {
        return find_key(name) >= 0 ? meta_u32(name) : fallback;
    }

    float meta_f32_or(const std::string & name, float fallback) const {
        const int64_t key = find_key(name);
        if (key < 0) {
            return fallback;
        }
        if (gguf_get_kv_type(file, key) != GGUF_TYPE_FLOAT32) {
            fail("unsupported metadata type: " + name);
        }
        return gguf_get_val_f32(file, key);
    }

    std::string meta_str(const std::string & name) const {
        const int64_t key = find_key(name);
        if (key < 0 || gguf_get_kv_type(file, key) != GGUF_TYPE_STRING) {
            fail("missing string metadata: " + name);
        }
        return gguf_get_val_str(file, key);
    }

    std::vector<std::string> meta_str_array(const std::string & name) const {
        const int64_t key = find_key(name);
        if (key < 0 || gguf_get_kv_type(file, key) != GGUF_TYPE_ARRAY ||
            gguf_get_arr_type(file, key) != GGUF_TYPE_STRING) {
            fail("missing string array metadata: " + name);
        }
        const size_t count = gguf_get_arr_n(file, key);
        std::vector<std::string> values(count);
        for (size_t i = 0; i < count; ++i) {
            values[i] = gguf_get_arr_str(file, key, i);
        }
        return values;
    }

    ggml_tensor * find_tensor(const std::string & name) const {
        return ggml_get_tensor(weights ? weights : metadata, name.c_str());
    }

    ggml_tensor * require_tensor(const std::string & name) const {
        ggml_tensor * tensor = find_tensor(name);
        if (tensor == nullptr) {
            fail("missing tensor: " + name);
        }
        return tensor;
    }

    void read_config() {
        const std::string arch(ARCH);
        config.n_layers    = (int) meta_u32(arch + ".block_count");
        config.n_embd      = (int) meta_u32(arch + ".embedding_length");
        config.n_ff        = (int) meta_u32(arch + ".feed_forward_length");
        config.n_heads     = (int) meta_u32(arch + ".attention.head_count");
        config.n_kv_heads  = (int) meta_u32(arch + ".attention.head_count_kv");
        config.head_dim    = (int) meta_u32_or(arch + ".attention.key_length",
                (uint32_t) (config.n_embd / config.n_heads));
        config.n_ctx_train = (int) meta_u32(arch + ".context_length");
        config.rope_base   = meta_f32_or(arch + ".rope.freq_base", 10000.0f);
        config.rms_eps     = meta_f32_or(arch + ".attention.layer_norm_rms_epsilon", 1e-6f);
        config.n_vq        = (int) meta_u32(arch + ".n_vq");
        config.audio_vocab = (int) meta_u32(arch + ".audio_vocab_size");
        config.audio_pad_code = (int) meta_u32(arch + ".audio_pad_code");
        config.audio_start_token_id = (int) meta_u32(arch + ".audio_start_token_id");
        config.audio_end_token_id   = (int) meta_u32(arch + ".audio_end_token_id");
        config.audio_user_slot_token_id = (int) meta_u32(arch + ".audio_user_slot_token_id");
        config.audio_assistant_gen_slot_token_id = (int) meta_u32(arch + ".audio_assistant_gen_slot_token_id");
        config.audio_assistant_delay_slot_token_id = (int) meta_u32(arch + ".audio_assistant_delay_slot_token_id");
        config.sampling_rate = (int) meta_u32_or(arch + ".sampling_rate", 24000);
        if (config.n_layers <= 0 || config.n_embd <= 0 || config.n_heads <= 0 ||
            config.n_kv_heads <= 0 || config.n_vq <= 0 || config.head_dim <= 0) {
            fail("invalid model geometry");
        }
    }

    std::string layer_name(int il, const char * suffix) const {
        return "blk." + std::to_string(il) + "." + suffix;
    }

    void map_tensors() {
        tok_embd    = require_tensor("token_embd.weight");
        output_norm = require_tensor("output_norm.weight");
        output_head = require_tensor("output.weight");
        config.text_vocab = (int) output_head->ne[1];
        tok_embd_audio.resize(config.n_vq);
        output_audio.resize(config.n_vq);
        for (int i = 0; i < config.n_vq; ++i) {
            tok_embd_audio[i] = require_tensor("token_embd_audio." + std::to_string(i) + ".weight");
            output_audio[i]   = require_tensor("output_audio." + std::to_string(i) + ".weight");
        }
        config.audio_vocab = (int) output_audio[0]->ne[1];
        layers.resize(config.n_layers);
        for (int il = 0; il < config.n_layers; ++il) {
            Layer & layer = layers[il];
            layer.attn_norm   = require_tensor(layer_name(il, "attn_norm.weight"));
            layer.wq          = require_tensor(layer_name(il, "attn_q.weight"));
            layer.wk          = require_tensor(layer_name(il, "attn_k.weight"));
            layer.wv          = require_tensor(layer_name(il, "attn_v.weight"));
            layer.wo          = require_tensor(layer_name(il, "attn_output.weight"));
            layer.attn_q_norm = require_tensor(layer_name(il, "attn_q_norm.weight"));
            layer.attn_k_norm = require_tensor(layer_name(il, "attn_k_norm.weight"));
            layer.ffn_norm    = require_tensor(layer_name(il, "ffn_norm.weight"));
            layer.ffn_gate    = require_tensor(layer_name(il, "ffn_gate.weight"));
            layer.ffn_up      = require_tensor(layer_name(il, "ffn_up.weight"));
            layer.ffn_down    = require_tensor(layer_name(il, "ffn_down.weight"));
        }
    }

    void require_text_token_id(int id, const char * name) const {
        if (id < 0 || id >= config.text_vocab || (int64_t) id >= tok_embd->ne[1]) {
            fail(std::string(name) + " is outside the text vocabulary");
        }
    }

    void validate_token_ids() const {
        require_text_token_id(config.audio_start_token_id, "audio_start_token_id");
        require_text_token_id(config.audio_end_token_id, "audio_end_token_id");
        require_text_token_id(config.audio_user_slot_token_id, "audio_user_slot_token_id");
        require_text_token_id(config.audio_assistant_gen_slot_token_id,
                "audio_assistant_gen_slot_token_id");
        require_text_token_id(config.audio_assistant_delay_slot_token_id,
                "audio_assistant_delay_slot_token_id");
        if (config.audio_pad_code < 0 || config.audio_pad_code >= config.audio_vocab) {
            fail("audio_pad_code is outside the audio vocabulary");
        }
    }

    void allocate_cache() {
        state = ggml_init({(size_t) (2 * config.n_layers + 8) * ggml_tensor_overhead(), nullptr, true});
        if (!state) {
            fail("state context allocation failed");
        }
        const int64_t kv_dim = (int64_t) config.head_dim * config.n_kv_heads;
        cache_k.resize(config.n_layers);
        cache_v.resize(config.n_layers);
        for (int il = 0; il < config.n_layers; ++il) {
            cache_k[il] = ggml_new_tensor_2d(state, GGML_TYPE_F16, kv_dim, n_ctx);
            cache_v[il] = ggml_new_tensor_2d(state, GGML_TYPE_F16, kv_dim, n_ctx);
        }
        state_buffer = ggml_backend_alloc_ctx_tensors(state, backend);
        if (!state_buffer) {
            fail("KV cache allocation failed");
        }
        ggml_backend_buffer_clear(state_buffer, 0);
    }

    void duplicate_metadata_tensors() {
        size_t count = 0;
        for (auto * src = ggml_get_first_tensor(metadata); src; src = ggml_get_next_tensor(metadata, src)) {
            count++;
        }
        weights = ggml_init({(count + 8) * ggml_tensor_overhead(), nullptr, true});
        if (!weights) {
            fail("weight context allocation failed");
        }
        for (auto * src = ggml_get_first_tensor(metadata); src; src = ggml_get_next_tensor(metadata, src)) {
            auto * dst = ggml_new_tensor(weights, src->type, GGML_MAX_DIMS, src->ne);
            ggml_set_name(dst, ggml_get_name(src));
        }
    }

    void upload_weights(const std::string & path) {
        weight_buffer = ggml_backend_alloc_ctx_tensors(weights, backend);
        if (!weight_buffer) {
            fail("weight allocation failed");
        }
        ::tts_cpp::detail::gguf_stream_reader reader(file, path);
        if (!reader.ok()) {
            fail("cannot reopen GGUF for streaming: " + path);
        }
        for (auto * dst = ggml_get_first_tensor(weights); dst; dst = ggml_get_next_tensor(weights, dst)) {
            if (!reader.to_backend(ggml_get_name(dst), dst)) {
                fail(std::string("failed to load tensor: ") + ggml_get_name(dst));
            }
        }
    }

    void init_backend(bool use_gpu) {
        ::tts_cpp::detail::ensure_backends_loaded();
        if (use_gpu) {
            backend = ::tts_cpp::detail::init_gpu_backend(1, false, "moss-delay");
        }
        if (!backend) {
            backend = ::tts_cpp::detail::init_cpu_backend();
        }
        if (!backend) {
            fail("no compute backend available");
        }
    }

    void load(const std::string & path, bool use_gpu, int threads, int context) {
        n_threads = threads;
        n_ctx = context;
        if (threads < 1 || threads > 1024) {
            fail("threads must be 1..1024");
        }
        if (n_ctx < MIN_CONTEXT || n_ctx > MAX_CONTEXT) {
            fail("context must be " + std::to_string(MIN_CONTEXT) + ".." + std::to_string(MAX_CONTEXT));
        }
        file = gguf_init_from_file(path.c_str(), {true, &metadata});
        if (!file || !metadata) {
            fail("cannot read GGUF: " + path);
        }
        if (meta_str("general.architecture") != ARCH) {
            fail("unsupported architecture");
        }
        read_config();
        init_backend(use_gpu);
        duplicate_metadata_tensors();
        map_tensors();
        validate_token_ids();
        allocate_cache();
        upload_weights(path);
    }

    void begin_graph() {
        if (graph_ctx) {
            ggml_free(graph_ctx);
            graph_ctx = nullptr;
        }
        inputs_i32.clear();
        inputs_f32.clear();
        const size_t buffer = (size_t) GRAPH_NODES * ggml_tensor_overhead() + ggml_graph_overhead_custom(GRAPH_NODES, false);
        graph_ctx = ggml_init({buffer, nullptr, true});
        if (!graph_ctx) {
            fail("graph context allocation failed");
        }
    }

    ggml_tensor * input_i32(std::vector<int32_t> data) {
        ggml_tensor * tensor = ggml_new_tensor_1d(graph_ctx, GGML_TYPE_I32, (int64_t) data.size());
        ggml_set_input(tensor);
        inputs_i32.push_back({tensor, std::move(data)});
        return tensor;
    }

    ggml_tensor * input_f32(std::vector<float> data, int64_t ne0, int64_t ne1) {
        ggml_tensor * tensor = ggml_new_tensor_2d(graph_ctx, GGML_TYPE_F32, ne0, ne1);
        ggml_set_input(tensor);
        inputs_f32.push_back({tensor, std::move(data)});
        return tensor;
    }

    void set_graph_inputs() {
        for (auto & input : inputs_i32) {
            ggml_backend_tensor_set(input.tensor, input.data.data(), 0, input.data.size() * sizeof(int32_t));
        }
        for (auto & input : inputs_f32) {
            ggml_backend_tensor_set(input.tensor, input.data.data(), 0, input.data.size() * sizeof(float));
        }
    }

    ggml_tensor * rms_norm(ggml_tensor * cur, ggml_tensor * weight) {
        return ggml_mul(graph_ctx, ggml_rms_norm(graph_ctx, cur, config.rms_eps), weight);
    }

    ggml_tensor * rope(ggml_tensor * cur, ggml_tensor * positions) {
        return ggml_rope_ext(graph_ctx, cur, positions, nullptr, config.head_dim,
                GGML_ROPE_TYPE_NORMAL, config.n_ctx_train, config.rope_base,
                1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    }

    ggml_tensor * cache_view(ggml_tensor * cache, int64_t count) {
        const int64_t kv_dim = (int64_t) config.head_dim * config.n_kv_heads;
        return ggml_view_2d(graph_ctx, cache, kv_dim, count, cache->nb[1], 0);
    }

    ggml_tensor * cache_write_view(ggml_tensor * cache, int64_t first, int64_t count) {
        const int64_t kv_dim = (int64_t) config.head_dim * config.n_kv_heads;
        return ggml_view_2d(graph_ctx, cache, kv_dim, count, cache->nb[1], first * cache->nb[1]);
    }

    ggml_tensor * attention(ggml_cgraph * graph, const Layer & layer, int il, ggml_tensor * cur,
                            int64_t n_tokens, int64_t total, ggml_tensor * positions, ggml_tensor * mask) {
        ggml_tensor * q = ggml_mul_mat(graph_ctx, layer.wq, cur);
        ggml_tensor * k = ggml_mul_mat(graph_ctx, layer.wk, cur);
        ggml_tensor * v = ggml_mul_mat(graph_ctx, layer.wv, cur);
        q = ggml_reshape_3d(graph_ctx, q, config.head_dim, config.n_heads, n_tokens);
        k = ggml_reshape_3d(graph_ctx, k, config.head_dim, config.n_kv_heads, n_tokens);
        v = ggml_reshape_3d(graph_ctx, v, config.head_dim, config.n_kv_heads, n_tokens);
        q = rope(rms_norm(q, layer.attn_q_norm), positions);
        k = rope(rms_norm(k, layer.attn_k_norm), positions);

        const int64_t kv_dim = (int64_t) config.head_dim * config.n_kv_heads;
        ggml_tensor * k_rows = ggml_reshape_2d(graph_ctx, ggml_cont(graph_ctx, k), kv_dim, n_tokens);
        ggml_tensor * v_rows = ggml_reshape_2d(graph_ctx, ggml_cont(graph_ctx, v), kv_dim, n_tokens);
        ggml_build_forward_expand(graph, ggml_cpy(graph_ctx, k_rows, cache_write_view(cache_k[il], pos, n_tokens)));
        ggml_build_forward_expand(graph, ggml_cpy(graph_ctx, v_rows, cache_write_view(cache_v[il], pos, n_tokens)));

        ggml_tensor * keys = ggml_reshape_3d(graph_ctx,
                ggml_cast(graph_ctx, cache_view(cache_k[il], total), GGML_TYPE_F32),
                config.head_dim, config.n_kv_heads, total);
        ggml_tensor * values = ggml_reshape_3d(graph_ctx,
                ggml_cast(graph_ctx, cache_view(cache_v[il], total), GGML_TYPE_F32),
                config.head_dim, config.n_kv_heads, total);

        q = ggml_permute(graph_ctx, q, 0, 2, 1, 3);
        keys = ggml_permute(graph_ctx, keys, 0, 2, 1, 3);
        values = ggml_cont(graph_ctx, ggml_permute(graph_ctx, values, 1, 2, 0, 3));
        ggml_tensor * scores = ggml_mul_mat(graph_ctx, keys, q);
        scores = ggml_soft_max_ext(graph_ctx, scores, mask, 1.0f / std::sqrt((float) config.head_dim), 0.0f);
        ggml_tensor * attended = ggml_mul_mat(graph_ctx, values, scores);
        attended = ggml_permute(graph_ctx, attended, 0, 2, 1, 3);
        attended = ggml_cont_2d(graph_ctx, attended, (int64_t) config.head_dim * config.n_heads, n_tokens);
        return ggml_mul_mat(graph_ctx, layer.wo, attended);
    }

    ggml_tensor * feed_forward(const Layer & layer, ggml_tensor * cur) {
        ggml_tensor * gate = ggml_silu(graph_ctx, ggml_mul_mat(graph_ctx, layer.ffn_gate, cur));
        ggml_tensor * up = ggml_mul_mat(graph_ctx, layer.ffn_up, cur);
        return ggml_mul_mat(graph_ctx, layer.ffn_down, ggml_mul(graph_ctx, gate, up));
    }

    std::vector<float> attention_mask(int64_t n_tokens, int64_t total) const {
        std::vector<float> mask(n_tokens * total, -std::numeric_limits<float>::infinity());
        for (int64_t iq = 0; iq < n_tokens; ++iq) {
            for (int64_t ik = 0; ik <= pos + iq; ++ik) {
                mask[iq * total + ik] = 0.0f;
            }
        }
        return mask;
    }

    ggml_tensor * embed_rows(const std::vector<DelayRow> & rows) {
        const int64_t n_tokens = (int64_t) rows.size();
        std::vector<int32_t> text_ids(n_tokens);
        for (int64_t i = 0; i < n_tokens; ++i) {
            text_ids[i] = rows[i].text;
        }
        ggml_tensor * cur = ggml_get_rows(graph_ctx, tok_embd, input_i32(std::move(text_ids)));
        for (int channel = 0; channel < config.n_vq; ++channel) {
            std::vector<int32_t> channel_ids(n_tokens);
            for (int64_t i = 0; i < n_tokens; ++i) {
                channel_ids[i] = rows[i].audio[channel];
            }
            ggml_tensor * embedded = ggml_get_rows(graph_ctx, tok_embd_audio[channel],
                    input_i32(std::move(channel_ids)));
            cur = ggml_add(graph_ctx, cur, embedded);
        }
        return cur;
    }

    void validate_rows(const std::vector<DelayRow> & rows) const {
        if (rows.empty()) {
            fail("empty batch");
        }
        if (pos + (int64_t) rows.size() > n_ctx) {
            fail("context overflow; reset the session or raise the context");
        }
        for (const DelayRow & row : rows) {
            if (row.text < 0 || row.text >= config.text_vocab) {
                fail("text token out of range");
            }
            if ((int) row.audio.size() != config.n_vq) {
                fail("each row needs one code per audio channel");
            }
            for (int32_t code : row.audio) {
                if (code < 0 || code >= config.audio_vocab) {
                    fail("audio code out of range");
                }
            }
        }
    }

    DelayLogits forward(const std::vector<DelayRow> & rows) {
        validate_rows(rows);
        begin_graph();
        const int64_t n_tokens = (int64_t) rows.size();
        const int64_t total = pos + n_tokens;
        ggml_cgraph * graph = ggml_new_graph_custom(graph_ctx, GRAPH_NODES, false);

        std::vector<int32_t> positions((size_t) n_tokens);
        for (int64_t i = 0; i < n_tokens; ++i) {
            positions[i] = (int32_t) (pos + i);
        }
        ggml_tensor * inp_pos = input_i32(std::move(positions));
        ggml_tensor * mask = input_f32(attention_mask(n_tokens, total), total, n_tokens);

        ggml_tensor * cur = embed_rows(rows);
        for (int il = 0; il < config.n_layers; ++il) {
            const Layer & layer = layers[il];
            ggml_tensor * residual = cur;
            cur = attention(graph, layer, il, rms_norm(cur, layer.attn_norm), n_tokens, total, inp_pos, mask);
            cur = ggml_add(graph_ctx, cur, residual);
            residual = cur;
            cur = feed_forward(layer, rms_norm(cur, layer.ffn_norm));
            cur = ggml_add(graph_ctx, cur, residual);
        }
        cur = rms_norm(cur, output_norm);
        ggml_tensor * last = ggml_get_rows(graph_ctx, cur, input_i32({(int32_t) (n_tokens - 1)}));

        ggml_tensor * text_logits = ggml_mul_mat(graph_ctx, output_head, last);
        ggml_set_output(text_logits);
        ggml_build_forward_expand(graph, text_logits);
        std::vector<ggml_tensor *> audio_logits(config.n_vq);
        for (int i = 0; i < config.n_vq; ++i) {
            audio_logits[i] = ggml_mul_mat(graph_ctx, output_audio[i], last);
            ggml_set_output(audio_logits[i]);
            ggml_build_forward_expand(graph, audio_logits[i]);
        }

        if (!::tts_cpp::detail::sched_fallback_ensure(sched, backend, GRAPH_NODES, {weight_buffer, state_buffer})) {
            fail("scheduler initialization failed");
        }
        if (!::tts_cpp::detail::sched_fallback_alloc(sched, graph)) {
            fail("graph allocation failed");
        }
        set_graph_inputs();
        const ggml_status status = ::tts_cpp::detail::sched_fallback_compute(sched, backend, graph, n_threads);
        if (status != GGML_STATUS_SUCCESS) {
            fail("graph compute failed");
        }

        DelayLogits logits;
        logits.text.resize(config.text_vocab);
        ggml_backend_tensor_get(text_logits, logits.text.data(), 0, logits.text.size() * sizeof(float));
        logits.audio.resize(config.n_vq);
        for (int i = 0; i < config.n_vq; ++i) {
            logits.audio[i].resize(config.audio_vocab);
            ggml_backend_tensor_get(audio_logits[i], logits.audio[i].data(), 0,
                    logits.audio[i].size() * sizeof(float));
        }
        pos += (int) n_tokens;
        return logits;
    }
};

DelayLM::DelayLM(const std::string & path, bool use_gpu, int n_threads, int n_ctx) : impl_(new Impl) {
    impl_->load(path, use_gpu, n_threads, n_ctx);
}

DelayLM::~DelayLM() = default;

const DelayConfig & DelayLM::config() const { return impl_->config; }
const char * DelayLM::backend_name() const { return ggml_backend_name(impl_->backend); }
int DelayLM::position() const { return impl_->pos; }
int DelayLM::context() const { return impl_->n_ctx; }

void DelayLM::reset() {
    impl_->pos = 0;
}

std::vector<std::string> DelayLM::tokenizer_tokens() const {
    return impl_->meta_str_array("tokenizer.ggml.tokens");
}

std::vector<std::string> DelayLM::tokenizer_merges() const {
    return impl_->meta_str_array("tokenizer.ggml.merges");
}

int32_t DelayLM::token_id(const std::string & token) const {
    const std::vector<std::string> tokens = tokenizer_tokens();
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i] == token) {
            return (int32_t) i;
        }
    }
    fail("unknown token: " + token);
}

DelayLogits DelayLM::prefill(const std::vector<DelayRow> & rows) {
    return impl_->forward(rows);
}

DelayLogits DelayLM::step(const DelayRow & row) {
    return impl_->forward({row});
}

} // namespace tts_cpp::moss::detail
