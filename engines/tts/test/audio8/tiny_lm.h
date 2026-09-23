#pragma once
// A tiny but complete audio8-lm GGUF, synthesized on the fly for tests that
// need a loadable language model without a fixture.

#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <string>
#include <vector>

namespace audio8_test {

// ── Tiny synthetic audio8-lm GGUF ───────────────────────────────────────────
// The smallest hparam set the loader, the graph builders, and the RoPE tables
// accept: every dimension real models scale up, none of the structure changed.

struct tiny_lm {
    int hidden = 32, depth = 2, n_head = 4, n_kv = 2, head_dim = 8, inter = 48;
    int vocab = 96;
    int fast_depth = 1, fast_n_head = 4, fast_n_kv = 2, fast_head_dim = 8, fast_inter = 48;
    int num_codebooks = 4, codebook_size = 24;
    int semantic_begin = 8, semantic_end = 31, eos = 32, pad = 0;
    int max_seq_len = 48, ras_window = 6;
};

inline void add_f32(gguf_context * g, ggml_context * ctx, const char * name,
             std::initializer_list<int64_t> ne) {
    ggml_tensor * t = ggml_new_tensor(ctx, GGML_TYPE_F32, (int) ne.size(),
                                      std::vector<int64_t>(ne).data());
    ggml_set_name(t, name);
    float * d = (float *) t->data;
    for (int64_t i = 0; i < ggml_nelements(t); ++i) d[i] = 0.01f;
    gguf_add_tensor(g, t);
}

inline std::string write_tiny_lm_gguf(const tiny_lm & p, const std::string & path,
                                      bool with_vocab = true, bool only_meta = false) {
    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "audio8-lm");
    auto u32 = [&](const char * k, int v) {
        gguf_set_val_u32(g, (std::string("audio8.lm.") + k).c_str(), (uint32_t) v);
    };
    auto f32 = [&](const char * k, float v) {
        gguf_set_val_f32(g, (std::string("audio8.lm.") + k).c_str(), v);
    };
    auto b = [&](const char * k, bool v) {
        gguf_set_val_bool(g, (std::string("audio8.lm.") + k).c_str(), v);
    };
    u32("depth", p.depth);            u32("hidden", p.hidden);
    u32("n_head", p.n_head);          u32("n_kv", p.n_kv);
    u32("head_dim", p.head_dim);      u32("inter", p.inter);
    u32("vocab", p.vocab);
    u32("fast_depth", p.fast_depth);  u32("fast_hidden", p.hidden);
    u32("fast_n_head", p.fast_n_head); u32("fast_n_kv", p.fast_n_kv);
    u32("fast_head_dim", p.fast_head_dim); u32("fast_inter", p.fast_inter);
    u32("num_codebooks", p.num_codebooks); u32("codebook_size", p.codebook_size);
    u32("semantic_begin", p.semantic_begin); u32("semantic_end", p.semantic_end);
    u32("eos", p.eos);                u32("pad", p.pad);
    u32("max_seq_len", p.max_seq_len); u32("ras_window", p.ras_window);
    f32("rope_theta", 10000.0f);      f32("rms_eps", 1e-5f);
    f32("ras_top_p", 0.9f);           f32("ras_temperature", 0.7f);
    b("norm_fast_input", true);       b("qkv_bias", true);
    b("fast_qkv_bias", false);

    if (with_vocab) {
        const char * toks[] = {"<pad>", "a", "b", "c"};
        gguf_set_arr_str(g, "tokenizer.ggml.tokens", toks, 4);
        const char * merges[] = {"a b"};
        gguf_set_arr_str(g, "tokenizer.ggml.merges", merges, 1);
        const int32_t added[] = {0};
        gguf_set_arr_data(g, "tokenizer.ggml.added_token_ids", GGUF_TYPE_INT32, added, 1);
    }

    ggml_init_params ip = { 16u * 1024 * 1024, nullptr, /*no_alloc=*/false };
    ggml_context * ctx = ggml_init(ip);

    add_f32(g, ctx, "lm/tok_emb",      {p.hidden, p.vocab});
    add_f32(g, ctx, "lm/codebook_emb", {p.hidden, (int64_t) p.num_codebooks * p.codebook_size});
    add_f32(g, ctx, "lm/norm",         {p.hidden});
    add_f32(g, ctx, "lm/sem_head",     {p.hidden, p.codebook_size + 1});
    add_f32(g, ctx, "lm/rope_cos",     {p.head_dim / 2, p.max_seq_len});
    add_f32(g, ctx, "lm/rope_sin",     {p.head_dim / 2, p.max_seq_len});
    for (int i = 0; i < p.depth; ++i) {
        const std::string pre = "lm/blk/" + std::to_string(i) + "/";
        add_f32(g, ctx, (pre + "wq").c_str(),        {p.hidden, p.n_head * p.head_dim});
        add_f32(g, ctx, (pre + "wk").c_str(),        {p.hidden, p.n_kv * p.head_dim});
        add_f32(g, ctx, (pre + "wv").c_str(),        {p.hidden, p.n_kv * p.head_dim});
        add_f32(g, ctx, (pre + "wo").c_str(),        {p.n_head * p.head_dim, p.hidden});
        add_f32(g, ctx, (pre + "wq_b").c_str(),      {p.n_head * p.head_dim});
        add_f32(g, ctx, (pre + "wk_b").c_str(),      {p.n_kv * p.head_dim});
        add_f32(g, ctx, (pre + "wv_b").c_str(),      {p.n_kv * p.head_dim});
        add_f32(g, ctx, (pre + "attn_norm").c_str(), {p.hidden});
        add_f32(g, ctx, (pre + "w1").c_str(),        {p.hidden, p.inter});
        add_f32(g, ctx, (pre + "w2").c_str(),        {p.inter, p.hidden});
        add_f32(g, ctx, (pre + "w3").c_str(),        {p.hidden, p.inter});
        add_f32(g, ctx, (pre + "ffn_norm").c_str(),  {p.hidden});
    }
    add_f32(g, ctx, "fast/emb",      {p.hidden, p.codebook_size});
    add_f32(g, ctx, "fast/norm",     {p.hidden});
    add_f32(g, ctx, "fast/out",      {p.hidden, p.codebook_size});
    add_f32(g, ctx, "fast/rope_cos", {p.fast_head_dim / 2, p.num_codebooks});
    add_f32(g, ctx, "fast/rope_sin", {p.fast_head_dim / 2, p.num_codebooks});
    for (int i = 0; i < p.fast_depth; ++i) {
        const std::string pre = "fast/blk/" + std::to_string(i) + "/";
        add_f32(g, ctx, (pre + "wq").c_str(),        {p.hidden, p.fast_n_head * p.fast_head_dim});
        add_f32(g, ctx, (pre + "wk").c_str(),        {p.hidden, p.fast_n_kv * p.fast_head_dim});
        add_f32(g, ctx, (pre + "wv").c_str(),        {p.hidden, p.fast_n_kv * p.fast_head_dim});
        add_f32(g, ctx, (pre + "wo").c_str(),        {p.fast_n_head * p.fast_head_dim, p.hidden});
        add_f32(g, ctx, (pre + "attn_norm").c_str(), {p.hidden});
        add_f32(g, ctx, (pre + "w1").c_str(),        {p.hidden, p.fast_inter});
        add_f32(g, ctx, (pre + "w2").c_str(),        {p.fast_inter, p.hidden});
        add_f32(g, ctx, (pre + "w3").c_str(),        {p.hidden, p.fast_inter});
        add_f32(g, ctx, (pre + "ffn_norm").c_str(),  {p.hidden});
    }

    if (!gguf_write_to_file(g, path.c_str(), only_meta)) {
        std::fprintf(stderr, "FATAL: cannot write %s\n", path.c_str());
        std::exit(2);
    }
    ggml_free(ctx);
    gguf_free(g);
    return path;
}

}  // namespace audio8_test
