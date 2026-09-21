#include "moss/codec.h"

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
#include <vector>

namespace tts_cpp::moss::detail {
namespace {

constexpr float LAYER_NORM_EPS = 1e-5f;
constexpr float L2_NORM_EPS    = 3.4526698e-4f;
constexpr float ROPE_FREQ_SCALE_NEUTRAL = 1.0f;
constexpr int   GRAPH_NODES    = 16384;
constexpr int   DEFAULT_SAMPLE_RATE = 24000;
constexpr int   MAX_CLIP_SECONDS    = 60;

constexpr const char * ENCODER_ARCH = "moss-tts-audio-encoder";
constexpr const char * DECODER_ARCH = "moss-tts-audio-decoder";

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("moss codec: " + message);
}

struct TransformerLayer {
    ggml_tensor * attn_qkv  = nullptr;
    ggml_tensor * attn_out  = nullptr;
    ggml_tensor * ffn_up    = nullptr;
    ggml_tensor * ffn_down  = nullptr;
    ggml_tensor * norm1_w   = nullptr;
    ggml_tensor * norm1_b   = nullptr;
    ggml_tensor * norm2_w   = nullptr;
    ggml_tensor * norm2_b   = nullptr;
    ggml_tensor * scale1    = nullptr;
    ggml_tensor * scale2    = nullptr;
};

struct TransformerBlock {
    int input_dimension  = 0;
    int output_dimension = 0;
    int d_model          = 0;
    int num_heads        = 0;
    int num_layers       = 0;
    int context          = 0;
    float max_period     = 10000.0f;
    ggml_tensor * input_proj  = nullptr;
    ggml_tensor * output_proj = nullptr;
    std::vector<TransformerLayer> layers;
};

struct Module {
    bool is_transformer = false;
    int patch_size = 1;
    TransformerBlock transformer;
};

struct QuantizerMeta {
    int input_dim      = 0;
    int rvq_dim        = 0;
    int output_dim     = 0;
    int num_quantizers = 0;
    int codebook_size  = 0;
    int codebook_dim   = 0;
};

struct GraphInputI32 {
    ggml_tensor * tensor;
    std::vector<int32_t> data;
};

struct GraphInputF32 {
    ggml_tensor * tensor;
    std::vector<float> data;
};

std::vector<int32_t> sequential_positions(int64_t n) {
    std::vector<int32_t> positions(n);
    for (int64_t i = 0; i < n; ++i) {
        positions[i] = (int32_t) i;
    }
    return positions;
}

std::vector<float> causal_mask(int64_t n, int context) {
    std::vector<float> mask(n * n, -std::numeric_limits<float>::infinity());
    for (int64_t iq = 0; iq < n; ++iq) {
        for (int64_t ik = 0; ik <= iq; ++ik) {
            if (context > 0 && iq - ik >= context) {
                continue;
            }
            mask[iq * n + ik] = 0.0f;
        }
    }
    return mask;
}

int64_t align_up(int64_t value, int64_t multiple) {
    if (multiple <= 1) {
        return value;
    }
    return ((value + multiple - 1) / multiple) * multiple;
}

} // namespace

struct Codec::Impl {
    gguf_context * file = nullptr;
    ggml_context * metadata = nullptr;
    ggml_context * weights = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t weight_buffer = nullptr;
    ::tts_cpp::detail::sched_fallback sched;

    std::string arch;
    bool encoder = false;
    int n_threads = 1;
    int rate = DEFAULT_SAMPLE_RATE;
    int64_t downsample = 0;
    QuantizerMeta quantizer;
    std::vector<Module> modules;

    ggml_context * graph_ctx = nullptr;
    std::vector<GraphInputI32> inputs_i32;
    std::vector<GraphInputF32> inputs_f32;

    ~Impl() {
        ::tts_cpp::detail::sched_fallback_free(sched);
        if (graph_ctx) ggml_free(graph_ctx);
        if (weight_buffer) ggml_backend_buffer_free(weight_buffer);
        if (weights) ggml_free(weights);
        if (metadata) ggml_free(metadata);
        if (file) gguf_free(file);
        if (backend) ggml_backend_free(backend);
    }

    int64_t find_key(const std::string & name) const {
        return gguf_find_key(file, name.c_str());
    }

    bool has_key(const std::string & name) const {
        return find_key(name) >= 0;
    }

    std::string meta_str(const std::string & name) const {
        const int64_t key = find_key(name);
        if (key < 0 || gguf_get_kv_type(file, key) != GGUF_TYPE_STRING) {
            fail("missing string metadata: " + name);
        }
        return gguf_get_val_str(file, key);
    }

    uint32_t meta_u32(const std::string & name) const {
        const int64_t key = find_key(name);
        if (key < 0) {
            fail("missing metadata: " + name);
        }
        switch (gguf_get_kv_type(file, key)) {
            case GGUF_TYPE_UINT32:  return gguf_get_val_u32(file, key);
            case GGUF_TYPE_INT32:   return (uint32_t) gguf_get_val_i32(file, key);
            case GGUF_TYPE_STRING:  return (uint32_t) std::stoul(gguf_get_val_str(file, key));
            default:                fail("unsupported metadata type: " + name);
        }
    }

    float meta_f32(const std::string & name, float fallback) const {
        const int64_t key = find_key(name);
        if (key < 0) {
            return fallback;
        }
        switch (gguf_get_kv_type(file, key)) {
            case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(file, key);
            case GGUF_TYPE_UINT32:  return (float) gguf_get_val_u32(file, key);
            case GGUF_TYPE_STRING:  return std::stof(gguf_get_val_str(file, key));
            default:                fail("unsupported metadata type: " + name);
        }
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

    void read_quantizer_meta() {
        quantizer.input_dim      = (int) meta_u32(arch + ".quantizer.input_dim");
        quantizer.rvq_dim        = (int) meta_u32(arch + ".quantizer.rvq_dim");
        quantizer.output_dim     = (int) meta_u32(arch + ".quantizer.output_dim");
        quantizer.num_quantizers = (int) meta_u32(arch + ".quantizer.num_quantizers");
        quantizer.codebook_size  = (int) meta_u32(arch + ".quantizer.codebook_size");
        quantizer.codebook_dim   = (int) meta_u32(arch + ".quantizer.codebook_dim");
        if (quantizer.num_quantizers <= 0 || quantizer.codebook_size <= 0) {
            fail("invalid quantizer geometry");
        }
    }

    void validate_codebooks() {
        for (int iq = 0; iq < quantizer.num_quantizers; ++iq) {
            const ggml_tensor * codebook =
                    require_tensor(quantizer_tensor_name(iq, "codebook.weight"));
            if (codebook->ne[1] < (int64_t) quantizer.codebook_size) {
                fail("codebook " + std::to_string(iq) +
                     " holds fewer rows than the declared codebook_size");
            }
        }
    }

    std::string block_tensor_name(int block, int layer, const char * suffix) const {
        return "blk." + std::to_string(block) + ".layer." + std::to_string(layer) + "." + suffix;
    }

    void read_transformer_meta(const std::string & prefix, int tensor_block, TransformerBlock & block) {
        block.input_dimension  = (int) meta_u32(prefix + ".input_dimension");
        block.output_dimension = (int) meta_u32(prefix + ".output_dimension");
        block.d_model          = (int) meta_u32(prefix + ".d_model");
        block.num_heads        = (int) meta_u32(prefix + ".num_heads");
        block.num_layers       = (int) meta_u32(prefix + ".num_layers");
        block.context          = (int) meta_u32(prefix + ".context");
        block.max_period       = meta_f32(prefix + ".max_period", 10000.0f);
        if (block.d_model <= 0 || block.num_heads <= 0 || block.d_model % block.num_heads != 0) {
            fail("invalid transformer geometry in " + prefix);
        }
        block.input_proj  = find_tensor("blk." + std::to_string(tensor_block) + ".input_proj.weight");
        block.output_proj = find_tensor("blk." + std::to_string(tensor_block) + ".output_proj.weight");
        block.layers.resize(block.num_layers);
    }

    void read_transformer_layers(int tensor_block, TransformerBlock & block) {
        for (int il = 0; il < block.num_layers; ++il) {
            TransformerLayer & layer = block.layers[il];
            layer.attn_qkv = require_tensor(block_tensor_name(tensor_block, il, "attn_qkv.weight"));
            layer.attn_out = require_tensor(block_tensor_name(tensor_block, il, "attn_output.weight"));
            layer.ffn_up   = require_tensor(block_tensor_name(tensor_block, il, "ffn_up.weight"));
            layer.ffn_down = require_tensor(block_tensor_name(tensor_block, il, "ffn_down.weight"));
            layer.norm1_w  = require_tensor(block_tensor_name(tensor_block, il, "attn_norm.weight"));
            layer.norm1_b  = require_tensor(block_tensor_name(tensor_block, il, "attn_norm.bias"));
            layer.norm2_w  = require_tensor(block_tensor_name(tensor_block, il, "ffn_norm.weight"));
            layer.norm2_b  = require_tensor(block_tensor_name(tensor_block, il, "ffn_norm.bias"));
            layer.scale1   = find_tensor(block_tensor_name(tensor_block, il, "attn_scale.scale"));
            layer.scale2   = find_tensor(block_tensor_name(tensor_block, il, "ffn_scale.scale"));
        }
    }

    void read_modules() {
        const std::string section = encoder ? "encoder" : "decoder";
        const uint32_t block_count = meta_u32(arch + "." + section + ".block_count");
        modules.resize(block_count);
        int tensor_block = 0;
        for (uint32_t ib = 0; ib < block_count; ++ib) {
            const std::string prefix = arch + "." + section + "." + std::to_string(ib);
            const std::string type = meta_str(prefix + ".module_type");
            Module & module = modules[ib];
            if (type == "PatchedPretransform") {
                module.is_transformer = false;
                module.patch_size = (int) meta_u32(prefix + ".patch_size");
                continue;
            }
            if (type != "Transformer") {
                fail("unsupported module type: " + type);
            }
            module.is_transformer = true;
            read_transformer_meta(prefix, tensor_block, module.transformer);
            read_transformer_layers(tensor_block, module.transformer);
            tensor_block++;
        }
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
            backend = ::tts_cpp::detail::init_gpu_backend(1, false, "moss-codec");
        }
        if (!backend) {
            backend = ::tts_cpp::detail::init_cpu_backend();
        }
        if (!backend) {
            fail("no compute backend available");
        }
    }

    void load(const std::string & path, bool use_gpu, int threads) {
        n_threads = threads;
        if (threads < 1 || threads > 1024) {
            fail("threads must be 1..1024");
        }
        file = gguf_init_from_file(path.c_str(), {true, &metadata});
        if (!file || !metadata) {
            fail("cannot read GGUF: " + path);
        }
        arch = meta_str("general.architecture");
        if (arch == ENCODER_ARCH) {
            encoder = true;
        } else if (arch == DECODER_ARCH) {
            encoder = false;
        } else {
            fail("unsupported architecture: " + arch);
        }
        rate = has_key(arch + ".sampling_rate") ? (int) meta_u32(arch + ".sampling_rate") : DEFAULT_SAMPLE_RATE;
        if (encoder) {
            downsample = meta_u32(arch + ".downsample_rate");
        }
        read_quantizer_meta();
        init_backend(use_gpu);
        duplicate_metadata_tensors();
        read_modules();
        validate_codebooks();
        if (!encoder) {
            downsample = decoder_upsample_factor();
        }
        if (downsample <= 0) {
            fail("invalid frame geometry");
        }
        upload_weights(path);
    }

    int64_t decoder_upsample_factor() const {
        int64_t factor = 1;
        for (const Module & module : modules) {
            if (!module.is_transformer) {
                factor *= module.patch_size;
            }
        }
        return factor;
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

    ggml_tensor * as_matrix(ggml_tensor * tensor) {
        if (tensor == nullptr) {
            return nullptr;
        }
        const int n_dims = ggml_n_dims(tensor);
        if (n_dims == 2) {
            return tensor;
        }
        if (n_dims == 3 && tensor->ne[0] == 1) {
            return ggml_reshape_2d(graph_ctx, tensor, tensor->ne[1], tensor->ne[2]);
        }
        if (n_dims == 4 && tensor->ne[0] == 1 && tensor->ne[1] == 1) {
            return ggml_reshape_2d(graph_ctx, tensor, tensor->ne[2], tensor->ne[3]);
        }
        fail(std::string("unsupported projection rank: ") + ggml_get_name(tensor));
    }

    ggml_tensor * as_f32(ggml_tensor * tensor) {
        if (tensor == nullptr) {
            return nullptr;
        }
        return tensor->type == GGML_TYPE_F32 ? tensor : ggml_cast(graph_ctx, tensor, GGML_TYPE_F32);
    }

    ggml_tensor * linear(ggml_tensor * input, ggml_tensor * weight, ggml_tensor * bias) {
        ggml_tensor * cur = as_f32(input);
        if (weight != nullptr) {
            cur = ggml_mul_mat(graph_ctx, as_f32(as_matrix(weight)), cur);
        }
        if (bias != nullptr) {
            cur = ggml_add(graph_ctx, cur, as_f32(bias));
        }
        return cur;
    }

    ggml_tensor * layer_norm(ggml_tensor * cur, ggml_tensor * weight, ggml_tensor * bias) {
        cur = ggml_norm(graph_ctx, cur, LAYER_NORM_EPS);
        cur = ggml_mul(graph_ctx, cur, as_f32(weight));
        return ggml_add(graph_ctx, cur, as_f32(bias));
    }

    ggml_tensor * l2_normalize(ggml_tensor * cur) {
        ggml_tensor * norm = ggml_sqrt(graph_ctx, ggml_sum_rows(graph_ctx, ggml_sqr(graph_ctx, cur)));
        norm = ggml_clamp(graph_ctx, norm, L2_NORM_EPS, INFINITY);
        return ggml_div(graph_ctx, cur, ggml_repeat(graph_ctx, norm, cur));
    }

    ggml_tensor * attention(const TransformerLayer & layer, ggml_tensor * cur,
                            const TransformerBlock & block, int64_t frames,
                            ggml_tensor * positions, ggml_tensor * mask) {
        const int d_head = block.d_model / block.num_heads;
        ggml_tensor * qkv = ggml_mul_mat(graph_ctx, as_f32(as_matrix(layer.attn_qkv)), cur);
        const size_t head_stride = ggml_row_size(qkv->type, d_head);
        ggml_tensor * q = ggml_view_3d(graph_ctx, qkv, d_head, block.num_heads, frames,
                head_stride, qkv->nb[1], 0);
        ggml_tensor * k = ggml_view_3d(graph_ctx, qkv, d_head, block.num_heads, frames,
                head_stride, qkv->nb[1], ggml_row_size(qkv->type, block.d_model));
        ggml_tensor * v = ggml_view_3d(graph_ctx, qkv, d_head, block.num_heads, frames,
                head_stride, qkv->nb[1], ggml_row_size(qkv->type, 2 * block.d_model));
        q = ggml_rope_ext(graph_ctx, q, positions, nullptr, d_head, 0, 0,
                block.max_period, ROPE_FREQ_SCALE_NEUTRAL, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(graph_ctx, k, positions, nullptr, d_head, 0, 0,
                block.max_period, ROPE_FREQ_SCALE_NEUTRAL, 0.0f, 1.0f, 0.0f, 0.0f);
        q = ggml_permute(graph_ctx, q, 0, 2, 1, 3);
        k = ggml_permute(graph_ctx, k, 0, 2, 1, 3);
        v = ggml_cont(graph_ctx, ggml_permute(graph_ctx, v, 1, 2, 0, 3));
        ggml_tensor * scores = ggml_mul_mat(graph_ctx, k, q);
        scores = ggml_soft_max_ext(graph_ctx, scores, mask, 1.0f / std::sqrt((float) d_head), 0.0f);
        ggml_tensor * attended = ggml_mul_mat(graph_ctx, v, scores);
        attended = ggml_permute(graph_ctx, attended, 0, 2, 1, 3);
        attended = ggml_cont_2d(graph_ctx, attended,
                attended->ne[0] * attended->ne[1], attended->ne[2] * attended->ne[3]);
        return ggml_mul_mat(graph_ctx, as_f32(as_matrix(layer.attn_out)), attended);
    }

    ggml_tensor * transformer_layer(const TransformerLayer & layer, ggml_tensor * cur,
                                    const TransformerBlock & block, int64_t frames,
                                    ggml_tensor * positions, ggml_tensor * mask) {
        ggml_tensor * attended = attention(layer, layer_norm(cur, layer.norm1_w, layer.norm1_b),
                block, frames, positions, mask);
        if (layer.scale1 != nullptr) {
            attended = ggml_mul(graph_ctx, attended, as_f32(layer.scale1));
        }
        cur = ggml_add(graph_ctx, cur, attended);
        ggml_tensor * ff = layer_norm(cur, layer.norm2_w, layer.norm2_b);
        ff = ggml_mul_mat(graph_ctx, as_f32(as_matrix(layer.ffn_up)), ff);
        ff = ggml_gelu(graph_ctx, ff);
        ff = ggml_mul_mat(graph_ctx, as_f32(as_matrix(layer.ffn_down)), ff);
        if (layer.scale2 != nullptr) {
            ff = ggml_mul(graph_ctx, ff, as_f32(layer.scale2));
        }
        return ggml_add(graph_ctx, cur, ff);
    }

    ggml_tensor * transformer_block(const TransformerBlock & block, ggml_tensor * cur, int64_t frames) {
        ggml_tensor * positions = input_i32(sequential_positions(frames));
        ggml_tensor * mask = input_f32(causal_mask(frames, block.context), frames, frames);
        if (block.input_proj != nullptr) {
            cur = ggml_mul_mat(graph_ctx, as_f32(as_matrix(block.input_proj)), cur);
        }
        for (const TransformerLayer & layer : block.layers) {
            cur = transformer_layer(layer, cur, block, frames, positions, mask);
        }
        if (block.output_proj != nullptr) {
            cur = ggml_mul_mat(graph_ctx, as_f32(as_matrix(block.output_proj)), cur);
        }
        return cur;
    }

    ggml_tensor * patch_encode(ggml_tensor * cur, int channels, int64_t frames, int patch) {
        cur = ggml_reshape_3d(graph_ctx, cur, channels, patch, frames / patch);
        cur = ggml_cont(graph_ctx, ggml_permute(graph_ctx, cur, 1, 0, 2, 3));
        return ggml_reshape_2d(graph_ctx, cur, (int64_t) channels * patch, frames / patch);
    }

    ggml_tensor * patch_decode(ggml_tensor * cur, int channels, int64_t frames, int patch) {
        const int out_channels = channels / patch;
        cur = ggml_reshape_3d(graph_ctx, cur, patch, out_channels, frames);
        cur = ggml_cont(graph_ctx, ggml_permute(graph_ctx, cur, 1, 0, 2, 3));
        return ggml_reshape_2d(graph_ctx, cur, out_channels, frames * patch);
    }

    std::string quantizer_tensor_name(int index, const char * suffix) const {
        return "quantizer.quantizers." + std::to_string(index) + "." + suffix;
    }

    ggml_tensor * decoder_quantizer(const std::vector<int32_t> & codes, int64_t frames) {
        ggml_tensor * cur = nullptr;
        for (int iq = 0; iq < quantizer.num_quantizers; ++iq) {
            std::vector<int32_t> channel(frames);
            for (int64_t i = 0; i < frames; ++i) {
                channel[i] = codes[i * quantizer.num_quantizers + iq];
            }
            ggml_tensor * indices = input_i32(std::move(channel));
            ggml_tensor * codebook = as_f32(require_tensor(quantizer_tensor_name(iq, "codebook.weight")));
            ggml_tensor * embedded = ggml_get_rows(graph_ctx, codebook, indices);
            embedded = linear(embedded,
                    find_tensor(quantizer_tensor_name(iq, "out_proj.weight")),
                    find_tensor(quantizer_tensor_name(iq, "out_proj.bias")));
            cur = cur ? ggml_add(graph_ctx, cur, embedded) : embedded;
        }
        return linear(cur,
                find_tensor("quantizer.output_proj.weight"),
                find_tensor("quantizer.output_proj.bias"));
    }

    ggml_tensor * encoder_quantizer(ggml_tensor * cur) {
        cur = linear(cur,
                find_tensor("quantizer.input_proj.weight"),
                find_tensor("quantizer.input_proj.bias"));
        ggml_tensor * codes = nullptr;
        ggml_tensor * residual = cur;
        for (int iq = 0; iq < quantizer.num_quantizers; ++iq) {
            ggml_tensor * latent = linear(residual,
                    find_tensor(quantizer_tensor_name(iq, "in_proj.weight")),
                    find_tensor(quantizer_tensor_name(iq, "in_proj.bias")));
            ggml_tensor * codebook = as_f32(require_tensor(quantizer_tensor_name(iq, "codebook.weight")));
            ggml_tensor * scores = ggml_mul_mat(graph_ctx, l2_normalize(codebook), l2_normalize(latent));
            ggml_tensor * code = ggml_argmax(graph_ctx, scores);
            ggml_tensor * row = ggml_reshape_2d(graph_ctx, code, 1, code->ne[0]);
            codes = codes ? ggml_concat(graph_ctx, codes, row, 0) : row;
            ggml_tensor * decoded = linear(ggml_get_rows(graph_ctx, codebook, code),
                    find_tensor(quantizer_tensor_name(iq, "out_proj.weight")),
                    find_tensor(quantizer_tensor_name(iq, "out_proj.bias")));
            residual = ggml_sub(graph_ctx, residual, decoded);
        }
        return codes;
    }

    ggml_tensor * build_encode(const std::vector<float> & pcm, int64_t padded_samples) {
        std::vector<float> padded(padded_samples, 0.0f);
        std::copy(pcm.begin(), pcm.end(), padded.begin());
        ggml_tensor * cur = input_f32(std::move(padded), 1, padded_samples);
        int channels = 1;
        int64_t frames = padded_samples;
        int tensor_block = 0;
        for (const Module & module : modules) {
            if (!module.is_transformer) {
                cur = patch_encode(cur, channels, frames, module.patch_size);
                channels *= module.patch_size;
                frames /= module.patch_size;
                continue;
            }
            cur = transformer_block(module.transformer, cur, frames);
            channels = module.transformer.output_dimension;
            tensor_block++;
        }
        if (channels != quantizer.input_dim) {
            fail("encoder output does not match quantizer input");
        }
        return encoder_quantizer(cur);
    }

    ggml_tensor * build_decode(const std::vector<int32_t> & codes, int64_t frames) {
        ggml_tensor * cur = decoder_quantizer(codes, frames);
        int channels = quantizer.output_dim;
        for (const Module & module : modules) {
            if (!module.is_transformer) {
                cur = patch_decode(cur, channels, frames, module.patch_size);
                channels /= module.patch_size;
                frames *= module.patch_size;
                continue;
            }
            cur = transformer_block(module.transformer, cur, frames);
            channels = module.transformer.output_dimension;
        }
        if (channels != 1) {
            fail("decoder did not reduce to mono output");
        }
        return cur;
    }

    std::vector<float> read_f32(ggml_tensor * tensor) {
        std::vector<float> out(ggml_nelements(tensor));
        ggml_backend_tensor_get(tensor, out.data(), 0, out.size() * sizeof(float));
        return out;
    }

    std::vector<int32_t> read_i32(ggml_tensor * tensor) {
        std::vector<int32_t> out(ggml_nelements(tensor));
        ggml_backend_tensor_get(tensor, out.data(), 0, out.size() * sizeof(int32_t));
        return out;
    }

    ggml_cgraph * finish_graph(ggml_tensor * output) {
        ggml_set_output(output);
        ggml_cgraph * graph = ggml_new_graph_custom(graph_ctx, GRAPH_NODES, false);
        ggml_build_forward_expand(graph, output);
        if (!::tts_cpp::detail::sched_fallback_ensure(sched, backend, GRAPH_NODES, {weight_buffer})) {
            fail("scheduler initialization failed");
        }
        if (!::tts_cpp::detail::sched_fallback_alloc(sched, graph)) {
            fail("graph allocation failed");
        }
        return graph;
    }

    void compute(ggml_cgraph * graph) {
        set_graph_inputs();
        const ggml_status status = ::tts_cpp::detail::sched_fallback_compute(sched, backend, graph, n_threads);
        if (status != GGML_STATUS_SUCCESS) {
            fail("graph compute failed");
        }
    }

};

Codec::Codec(const std::string & path, bool use_gpu, int n_threads) : impl_(new Impl) {
    impl_->load(path, use_gpu, n_threads);
}

Codec::~Codec() = default;

bool Codec::is_encoder() const { return impl_->encoder; }
int Codec::sample_rate() const { return impl_->rate; }
int Codec::num_quantizers() const { return impl_->quantizer.num_quantizers; }
int Codec::samples_per_frame() const { return (int) impl_->downsample; }
const char * Codec::backend_name() const { return ggml_backend_name(impl_->backend); }

std::vector<int32_t> Codec::encode(const std::vector<float> & pcm) {
    if (!impl_->encoder) {
        fail("encode called on a decoder checkpoint");
    }
    if (pcm.empty() || pcm.size() > (size_t) impl_->rate * MAX_CLIP_SECONDS) {
        fail("reference audio must contain up to " + std::to_string(MAX_CLIP_SECONDS) + " seconds of mono samples");
    }
    for (float sample : pcm) {
        if (!std::isfinite(sample)) {
            fail("reference audio contains non-finite samples");
        }
    }
    impl_->begin_graph();
    const int64_t padded = align_up((int64_t) pcm.size(), impl_->downsample);
    ggml_tensor * output = impl_->build_encode(pcm, padded);
    ggml_cgraph * graph = impl_->finish_graph(output);
    impl_->compute(graph);
    std::vector<int32_t> codes = impl_->read_i32(output);
    // The trailing zero-padded partial frame carries no signal; the reference
    // pipeline keeps only floor(samples / downsample) frames.
    const size_t valid_frames = pcm.size() / (size_t) impl_->downsample;
    codes.resize(valid_frames * (size_t) impl_->quantizer.num_quantizers);
    return codes;
}

std::vector<float> Codec::decode(const std::vector<int32_t> & codes) {
    if (impl_->encoder) {
        fail("decode called on an encoder checkpoint");
    }
    const int n_q = impl_->quantizer.num_quantizers;
    if (codes.empty() || codes.size() % n_q != 0) {
        fail("decode expects row-major [frames, num_quantizers] codes");
    }
    for (int32_t code : codes) {
        if (code < 0 || code >= impl_->quantizer.codebook_size) {
            fail("audio code out of range");
        }
    }
    impl_->begin_graph();
    const int64_t frames = (int64_t) codes.size() / n_q;
    ggml_tensor * output = impl_->build_decode(codes, frames);
    ggml_cgraph * graph = impl_->finish_graph(output);
    impl_->compute(graph);
    return impl_->read_f32(output);
}

} // namespace tts_cpp::moss::detail
