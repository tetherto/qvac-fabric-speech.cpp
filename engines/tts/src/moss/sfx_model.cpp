#include "moss/sfx_model.h"

#include "backend_selection.h"
#include "backend_util.h"
#include "gguf_stream.h"
#include "sched_dispatch.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tts_cpp::moss::detail {
namespace {

constexpr const char * ARCH = "moss-sfx";
constexpr int SCHED_NODES = 32768;
constexpr int MAX_LAYERS = 128;
constexpr int MAX_WIDTH = 1 << 16;
constexpr int MAX_HEADS = 1024;
constexpr int MAX_THREADS = 1024;
constexpr size_t TENSOR_SLACK = 8;
constexpr int MAX_TEXT_TOKENS = 4096;
constexpr int MAX_RATES = 16;
constexpr int MAX_RATE = 64;
constexpr int MIN_SAMPLE_RATE = 8000;
constexpr int MAX_SAMPLE_RATE = 192000;
constexpr float MAX_SECONDS_LIMIT = 600.0f;
constexpr int64_t MAX_HOP = 1 << 16;
constexpr int64_t MAX_LATENT_FRAMES = 1 << 14;

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("moss sfx: " + message);
}

bool within(int value, int low, int high) {
    return value >= low && value <= high;
}

bool is_matrix_type(ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16 || type == GGML_TYPE_Q8_0;
}

void require_f32(const ggml_tensor * tensor) {
    if (tensor->type != GGML_TYPE_F32) {
        fail(std::string(ggml_get_name(tensor)) + " must be f32");
    }
}

int64_t rate_product(const std::vector<int> & rates) {
    int64_t product = 1;
    for (int rate : rates) {
        product = std::min<int64_t>(product * rate, MAX_HOP + 1);
    }
    return product;
}

} // namespace

void require_matrix(const ggml_tensor * tensor, int64_t ne0, int64_t ne1) {
    require_shape(tensor, ne0, ne1);
    if (!is_matrix_type(tensor->type)) {
        fail(std::string(ggml_get_name(tensor)) + " has an unsupported type");
    }
}

void require_vector(const ggml_tensor * tensor, int64_t ne0, int64_t ne1) {
    require_shape(tensor, ne0, ne1);
    require_f32(tensor);
}

void require_shape(const ggml_tensor * tensor, int64_t ne0, int64_t ne1, int64_t ne2) {
    if (tensor->ne[0] != ne0 || tensor->ne[1] != ne1 || tensor->ne[2] != ne2 || tensor->ne[3] != 1) {
        fail(std::string(ggml_get_name(tensor)) + " has unexpected dimensions");
    }
}

int SfxVaeConfig::hop_length() const {
    return (int) rate_product(rates);
}

int SfxConfig::latent_frames() const {
    const int64_t samples = std::llround((double) vae.sample_rate * max_seconds);
    return (int) std::min<int64_t>(samples / rate_product(vae.rates), MAX_LATENT_FRAMES + 1);
}

SfxGraph::SfxGraph(int max_nodes) {
    const size_t bytes = (size_t) max_nodes * ggml_tensor_overhead() +
            ggml_graph_overhead_custom(max_nodes, false);
    ctx_ = ggml_init({bytes, nullptr, true});
    if (!ctx_) {
        fail("graph context allocation failed");
    }
    graph_ = ggml_new_graph_custom(ctx_, max_nodes, false);
}

SfxGraph::~SfxGraph() {
    ggml_free(ctx_);
}

ggml_context * SfxGraph::ctx() const { return ctx_; }
ggml_cgraph * SfxGraph::graph() const { return graph_; }

ggml_tensor * SfxGraph::input_f32(int64_t ne0, int64_t ne1) {
    ggml_tensor * tensor = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, ne0, ne1);
    ggml_set_input(tensor);
    return tensor;
}

ggml_tensor * SfxGraph::input_i32(int64_t ne0) {
    ggml_tensor * tensor = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, ne0);
    ggml_set_input(tensor);
    return tensor;
}

struct SfxModel::Impl {
    gguf_context * file = nullptr;
    ggml_context * metadata = nullptr;
    ggml_context * weights = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t weight_buffer = nullptr;
    ::tts_cpp::detail::sched_fallback sched;
    SfxConfig config;
    int n_threads = 1;
    const SfxGraph * allocated_graph = nullptr;

    ~Impl() {
        ::tts_cpp::detail::sched_fallback_free(sched);
        if (weight_buffer) ggml_backend_buffer_free(weight_buffer);
        if (weights) ggml_free(weights);
        if (metadata) ggml_free(metadata);
        if (file) gguf_free(file);
        if (backend) ggml_backend_free(backend);
    }

    int64_t key(const std::string & name) const {
        return gguf_find_key(file, name.c_str());
    }

    uint32_t meta_u32(const std::string & name) const {
        const int64_t id = key(name);
        if (id < 0) {
            fail("missing metadata: " + name);
        }
        switch (gguf_get_kv_type(file, id)) {
            case GGUF_TYPE_UINT32: return gguf_get_val_u32(file, id);
            case GGUF_TYPE_INT32:  return (uint32_t) gguf_get_val_i32(file, id);
            default:               fail("unsupported metadata type: " + name);
        }
    }

    float meta_f32(const std::string & name) const {
        const int64_t id = key(name);
        if (id < 0 || gguf_get_kv_type(file, id) != GGUF_TYPE_FLOAT32) {
            fail("missing float metadata: " + name);
        }
        return gguf_get_val_f32(file, id);
    }

    std::string meta_str(const std::string & name) const {
        const int64_t id = key(name);
        if (id < 0 || gguf_get_kv_type(file, id) != GGUF_TYPE_STRING) {
            fail("missing string metadata: " + name);
        }
        return gguf_get_val_str(file, id);
    }

    std::vector<std::string> meta_str_array(const std::string & name) const {
        const int64_t id = key(name);
        if (id < 0 || gguf_get_kv_type(file, id) != GGUF_TYPE_ARRAY ||
            gguf_get_arr_type(file, id) != GGUF_TYPE_STRING) {
            fail("missing string array metadata: " + name);
        }
        return read_strings(id);
    }

    std::vector<std::string> read_strings(int64_t id) const {
        std::vector<std::string> values(gguf_get_arr_n(file, id));
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = gguf_get_arr_str(file, id, i);
        }
        return values;
    }

    std::vector<int> meta_int_array(const std::string & name) const {
        const int64_t id = key(name);
        if (id < 0 || gguf_get_kv_type(file, id) != GGUF_TYPE_ARRAY) {
            fail("missing array metadata: " + name);
        }
        const gguf_type type = gguf_get_arr_type(file, id);
        if (type != GGUF_TYPE_INT32 && type != GGUF_TYPE_UINT32) {
            fail("unsupported array type: " + name);
        }
        const size_t count = gguf_get_arr_n(file, id);
        if (count > (size_t) MAX_RATES) {
            fail("array is too long: " + name);
        }
        const auto * data = static_cast<const int32_t *>(gguf_get_arr_data(file, id));
        return std::vector<int>(data, data + count);
    }

    void read_text_config() {
        const std::string prefix = std::string(ARCH) + ".text";
        SfxTextConfig & text = config.text;
        text.n_layers   = (int) meta_u32(prefix + ".block_count");
        text.n_embd     = (int) meta_u32(prefix + ".embedding_length");
        text.n_ff       = (int) meta_u32(prefix + ".feed_forward_length");
        text.n_heads    = (int) meta_u32(prefix + ".attention.head_count");
        text.n_kv_heads = (int) meta_u32(prefix + ".attention.head_count_kv");
        text.head_dim   = (int) meta_u32(prefix + ".attention.key_length");
        text.max_tokens = (int) meta_u32(prefix + ".context_length");
        text.rope_base  = meta_f32(prefix + ".rope.freq_base");
        text.rms_eps    = meta_f32(prefix + ".attention.layer_norm_rms_epsilon");
    }

    void read_dit_config() {
        const std::string prefix = std::string(ARCH) + ".dit";
        SfxDitConfig & dit = config.dit;
        dit.n_layers     = (int) meta_u32(prefix + ".block_count");
        dit.n_embd       = (int) meta_u32(prefix + ".embedding_length");
        dit.n_ff         = (int) meta_u32(prefix + ".feed_forward_length");
        dit.n_heads      = (int) meta_u32(prefix + ".attention.head_count");
        dit.in_channels  = (int) meta_u32(prefix + ".in_channels");
        dit.out_channels = (int) meta_u32(prefix + ".out_channels");
        dit.text_dim     = (int) meta_u32(prefix + ".text_dim");
        dit.freq_dim     = (int) meta_u32(prefix + ".freq_dim");
        dit.eps          = meta_f32(prefix + ".epsilon");
    }

    void read_vae_config() {
        const std::string prefix = std::string(ARCH) + ".vae";
        SfxVaeConfig & vae = config.vae;
        vae.latent_dim  = (int) meta_u32(prefix + ".latent_dim");
        vae.decoder_dim = (int) meta_u32(prefix + ".decoder_dim");
        vae.sample_rate = (int) meta_u32(prefix + ".sample_rate");
        vae.rates       = meta_int_array(prefix + ".decoder_rates");
    }

    void read_generation_config() {
        const std::string prefix(ARCH);
        config.max_seconds      = meta_f32(prefix + ".max_seconds");
        config.sigma_shift      = meta_f32(prefix + ".sigma_shift");
        config.train_timesteps  = (int) meta_u32(prefix + ".num_train_timesteps");
        config.default_steps    = (int) meta_u32(prefix + ".default_steps");
        config.default_guidance = meta_f32(prefix + ".default_guidance");
    }

    void validate_text_config() const {
        const SfxTextConfig & text = config.text;
        if (!within(text.n_layers, 1, MAX_LAYERS) || !within(text.n_embd, 1, MAX_WIDTH) ||
            !within(text.n_ff, 1, MAX_WIDTH * 4) || !within(text.n_heads, 1, MAX_HEADS) ||
            !within(text.n_kv_heads, 1, MAX_HEADS) || !within(text.head_dim, 1, MAX_WIDTH) ||
            text.n_heads % text.n_kv_heads != 0 || text.head_dim % 2 != 0 ||
            !within(text.max_tokens, 1, MAX_TEXT_TOKENS)) {
            fail("invalid text encoder geometry");
        }
    }

    void validate_dit_config() const {
        const SfxDitConfig & dit = config.dit;
        if (!within(dit.n_layers, 1, MAX_LAYERS) || !within(dit.n_embd, 1, MAX_WIDTH) ||
            !within(dit.n_ff, 1, MAX_WIDTH * 4) || !within(dit.n_heads, 1, MAX_HEADS) ||
            dit.n_embd % dit.n_heads != 0 || ((dit.n_embd / dit.n_heads) % 2) != 0 ||
            !within(dit.in_channels, 1, MAX_WIDTH) || dit.out_channels != dit.in_channels ||
            dit.text_dim != config.text.n_embd || !within(dit.freq_dim, 2, MAX_WIDTH) ||
            dit.freq_dim % 2 != 0) {
            fail("invalid DiT geometry");
        }
    }

    void validate_vae_rates() const {
        const SfxVaeConfig & vae = config.vae;
        for (size_t i = 0; i < vae.rates.size(); ++i) {
            if (!within(vae.rates[i], 1, MAX_RATE) || (vae.decoder_dim >> (i + 1)) < 1) {
                fail("invalid VAE decoder rates");
            }
        }
    }

    void validate_vae_config() const {
        const SfxVaeConfig & vae = config.vae;
        if (vae.latent_dim != config.dit.out_channels || vae.rates.empty() ||
            !within(vae.decoder_dim, 1, MAX_WIDTH) ||
            !within(vae.sample_rate, MIN_SAMPLE_RATE, MAX_SAMPLE_RATE)) {
            fail("invalid VAE geometry");
        }
        validate_vae_rates();
        if (rate_product(vae.rates) > MAX_HOP) {
            fail("VAE decoder rates multiply past the supported hop length");
        }
    }

    void validate_generation_config() const {
        if (!(config.max_seconds > 0.0f && config.max_seconds <= MAX_SECONDS_LIMIT) ||
            config.latent_frames() < 1 || config.latent_frames() > MAX_LATENT_FRAMES ||
            config.train_timesteps < 1 ||
            config.default_steps < 1 || !(config.sigma_shift > 0.0f)) {
            fail("invalid generation metadata");
        }
    }

    void read_config() {
        read_text_config();
        read_dit_config();
        read_vae_config();
        read_generation_config();
        validate_text_config();
        validate_dit_config();
        validate_vae_config();
        validate_generation_config();
    }

    size_t count_metadata_tensors() const {
        size_t count = 0;
        for (auto * src = ggml_get_first_tensor(metadata); src; src = ggml_get_next_tensor(metadata, src)) {
            count++;
        }
        return count;
    }

    void mirror_metadata_tensors() {
        for (auto * src = ggml_get_first_tensor(metadata); src; src = ggml_get_next_tensor(metadata, src)) {
            ggml_tensor * dst = ggml_new_tensor(weights, src->type, GGML_MAX_DIMS, src->ne);
            ggml_set_name(dst, ggml_get_name(src));
        }
    }

    void duplicate_metadata_tensors() {
        weights = ggml_init({(count_metadata_tensors() + TENSOR_SLACK) * ggml_tensor_overhead(), nullptr, true});
        if (!weights) {
            fail("weight context allocation failed");
        }
        mirror_metadata_tensors();
    }

    void stream_tensors(::tts_cpp::detail::gguf_stream_reader & reader) {
        for (auto * dst = ggml_get_first_tensor(weights); dst; dst = ggml_get_next_tensor(weights, dst)) {
            if (!reader.to_backend(ggml_get_name(dst), dst)) {
                fail(std::string("failed to load tensor: ") + ggml_get_name(dst));
            }
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
        stream_tensors(reader);
    }

    void init_backend(bool use_gpu) {
        ::tts_cpp::detail::ensure_backends_loaded();
        if (use_gpu) {
            backend = ::tts_cpp::detail::init_gpu_backend(1, false, "moss-sfx");
        }
        if (!backend) {
            backend = ::tts_cpp::detail::init_cpu_backend();
        }
        if (!backend) {
            fail("no compute backend available");
        }
    }

    void load(const std::string & path, bool use_gpu, int threads) {
        if (threads < 1 || threads > MAX_THREADS) {
            fail("threads must be 1.." + std::to_string(MAX_THREADS));
        }
        n_threads = threads;
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
        upload_weights(path);
    }

    ggml_tensor * find(const std::string & name) const {
        ggml_tensor * tensor = ggml_get_tensor(weights, name.c_str());
        if (tensor == nullptr) {
            fail("missing tensor: " + name);
        }
        return tensor;
    }

    void allocate(SfxGraph & graph) {
        if (!::tts_cpp::detail::sched_fallback_ensure(sched, backend, SCHED_NODES, {weight_buffer})) {
            fail("scheduler initialization failed");
        }
        allocated_graph = nullptr;
        if (!::tts_cpp::detail::sched_fallback_alloc(sched, graph.graph())) {
            fail("graph allocation failed");
        }
        allocated_graph = &graph;
    }

    void compute(SfxGraph & graph) {
        if (allocated_graph != &graph) {
            fail("graph is no longer allocated; another graph ran on this model since");
        }
        const ggml_status status = ::tts_cpp::detail::sched_fallback_compute(sched, backend,
                graph.graph(), n_threads);
        if (status != GGML_STATUS_SUCCESS) {
            fail("graph compute failed");
        }
    }
};

SfxModel::SfxModel(const std::string & path, bool use_gpu, int n_threads) : impl_(new Impl) {
    impl_->load(path, use_gpu, n_threads);
}

SfxModel::~SfxModel() = default;

const SfxConfig & SfxModel::config() const { return impl_->config; }
const char * SfxModel::backend_name() const { return ggml_backend_name(impl_->backend); }
ggml_tensor * SfxModel::tensor(const std::string & name) const { return impl_->find(name); }

std::vector<std::string> SfxModel::tokenizer_tokens() const {
    return impl_->meta_str_array("tokenizer.ggml.tokens");
}

std::vector<std::string> SfxModel::tokenizer_merges() const {
    return impl_->meta_str_array("tokenizer.ggml.merges");
}

void SfxModel::allocate(SfxGraph & graph) { impl_->allocate(graph); }
void SfxModel::compute(SfxGraph & graph) { impl_->compute(graph); }

} // namespace tts_cpp::moss::detail
