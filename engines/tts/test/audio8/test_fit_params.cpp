// Fit-projection parity tests (include/tts-cpp/audio8/fit.h): assert that
// the metadata-only memory projection matches what a REAL load and REAL runs
// actually allocate, byte for byte where the projection is exact by
// construction.
//
// Two modes:
//
//   * No arguments (the always-on CI form): a tiny but complete audio8-lm
//     GGUF is synthesized on the fly (same trick as test_gguf_load.cpp), so
//     the LM-side parity gates run with no model fixture at all:
//       1. projected LM weight bytes == the buffer a real load_lm allocates;
//       2. projected KV-slab bytes == the slow+fast cache buffers;
//       3. projected slow arena (max of prompt prefill and deepest decode
//          step) == the gallocr buffer real slow_step calls reserve;
//       4. projected fast arena == the gallocr buffer real fast_step
//          reserves;
//       5. a missing / wrong-architecture model is Error, never Success.
//
//   * With arguments <lm.gguf> <decoder.gguf> [encoder.gguf] [n_gpu_layers]:
//     the full-pipeline gates on real fixtures -- everything above via
//     fit_params, plus codec parity: projected decode arenas (latent +
//     planned synthesis block) == what a real decode_codes leaves in
//     model.allocr / model.block_allocr; and near-INT_MAX workloads
//     (prompt_tokens, reference_seconds) are Error/"workload-too-large",
//     never a sign-overflowed graph shape.
//
// Exit 0 on success; non-zero with a FAIL line per broken invariant.

#include "tts-cpp/audio8/fit.h"

#include "audio8/graph.h"
#include "audio8/internal.h"
#include "fit_price.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace tts_cpp::audio8::detail;

namespace {

int g_failures = 0;

void fail(const std::string & what) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
}

void expect(bool cond, const std::string & what) {
    if (!cond) fail(what);
}

void expect_eq(uint64_t projected, uint64_t real, const std::string & what) {
    if (projected != real) {
        fail(what + ": projected " + std::to_string(projected) +
             " != allocated " + std::to_string(real));
    }
}

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

void add_f32(gguf_context * g, ggml_context * ctx, const char * name,
             std::initializer_list<int64_t> ne) {
    ggml_tensor * t = ggml_new_tensor(ctx, GGML_TYPE_F32, (int) ne.size(),
                                      std::vector<int64_t>(ne).data());
    ggml_set_name(t, name);
    float * d = (float *) t->data;
    for (int64_t i = 0; i < ggml_nelements(t); ++i) d[i] = 0.01f;
    gguf_add_tensor(g, t);
}

std::string write_tiny_lm_gguf(const tiny_lm & p) {
    const std::string path =
        (fs::temp_directory_path() / "test-audio8-fit-tiny-lm.gguf").string();
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

    const char * toks[] = {"<pad>", "a", "b", "c"};
    gguf_set_arr_str(g, "tokenizer.ggml.tokens", toks, 4);
    const char * merges[] = {"a b"};
    gguf_set_arr_str(g, "tokenizer.ggml.merges", merges, 1);
    const int32_t added[] = {0};
    gguf_set_arr_data(g, "tokenizer.ggml.added_token_ids", GGUF_TYPE_INT32, added, 1);

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

    if (!gguf_write_to_file(g, path.c_str(), /*only_meta=*/false)) {
        std::fprintf(stderr, "FATAL: cannot write %s\n", path.c_str());
        std::exit(2);
    }
    ggml_free(ctx);
    gguf_free(g);
    return path;
}

// Price a freshly built LM graph the way the projector does (fit_price.h
// mirrors prepare_graph's dispatch).
bool price(lm_model & lm, scratch & build, ::tts_cpp::detail::fit_graph_price & out) {
    return build.ok() &&
           ::tts_cpp::detail::fit_price_graph(lm.backend, build.graph,
                                              2 * AUDIO8_MAX_NODES, out);
}

// What a real fast_step leaves allocated: one arena per position it ran.
uint64_t real_fast_arena_bytes(const lm_model & model) {
    uint64_t total = 0;
    for (const lm_model::fast_graph & cached : model.fast_graphs) {
        if (cached.allocr) total += ggml_gallocr_get_buffer_size(cached.allocr, 0);
    }
    return total;
}

// And in host RAM, one graph context per position. The projector multiplies
// scratch_arena_bytes by this count, so the count is what has to be pinned --
// the size itself comes from the same function the arenas are built with.
uint64_t retained_fast_contexts(const lm_model & model) {
    uint64_t kept = 0;
    for (const lm_model::fast_graph & cached : model.fast_graphs) {
        if (cached.ctx) ++kept;
    }
    return kept;
}

void run_synthetic_lm_gates() {
    const tiny_lm p;
    const std::string path = write_tiny_lm_gguf(p);

    // Metadata-only vs real load on the same (CPU) backend.
    lm_model mm, real;
    fit_load_measure lmm;
    std::string error;
    if (!load_lm_metadata_only(path, /*n_gpu_layers=*/0, mm, lmm, &error)) {
        fail("load_lm_metadata_only failed: " + error);
        return;
    }
    if (!load_lm(path, /*n_gpu_layers=*/0, real, &error)) {
        fail("real load_lm failed: " + error);
        free_lm(mm);
        return;
    }

    // 1 + 2. Weight and KV parity, byte for byte.
    expect_eq(lmm.weights_bytes, ggml_backend_buffer_get_size(real.buffer_w),
              "LM weights parity");
    expect_eq(lmm.kv_bytes,
              (uint64_t) ggml_backend_buffer_get_size(real.slow_kv.buffer) +
                  ggml_backend_buffer_get_size(real.fast_kv.buffer),
              "LM KV parity");

    // 3. Slow arena parity: real prefill + deepest decode step vs the
    //    projection of the same two shapes (one gallocr, so max).
    const int width = p.num_codebooks + 1;
    const int prompt_w = 6;
    std::vector<int32_t> frames((size_t) width * prompt_w, 0);
    for (int c = 0; c < prompt_w; ++c) frames[(size_t) c * width] = 1;
    std::vector<float> sem_logits, fast_input;
    if (!slow_step(real, frames.data(), prompt_w, /*n_past=*/0, /*n_threads=*/2,
                   sem_logits, fast_input, &error) ||
        !slow_step(real, frames.data(), /*width=*/1, /*n_past=*/prompt_w, 2,
                   sem_logits, fast_input, &error)) {
        fail("real slow_step failed: " + error);
    } else {
        ::tts_cpp::detail::fit_graph_price prefill, step;
        {
            scratch build(AUDIO8_MAX_NODES);
            slow_graph_outputs outs;
            build_slow_graph(mm, build, prompt_w, 0, outs);
            if (!price(mm, build, prefill)) fail("pricing the prefill graph failed");
        }
        {
            scratch build(AUDIO8_MAX_NODES);
            slow_graph_outputs outs;
            build_slow_graph(mm, build, 1, prompt_w, outs);
            if (!price(mm, build, step)) fail("pricing the step graph failed");
        }
        if (prefill.host_bytes == 0 && step.host_bytes == 0) {
            expect_eq(std::max(prefill.device_bytes, step.device_bytes),
                      ggml_gallocr_get_buffer_size(real.slow_allocr, 0),
                      "LM slow arena parity");
        }
    }

    // 4. Fast arena parity: a real whole-frame fast_step vs the projected sum
    //    over the positions. Each position keeps its own graph and its own
    //    arena for the life of the model, so what a real frame leaves behind is
    //    every one of them, not the widest.
    {
        std::vector<int32_t> codes;
        std::vector<float> prime_in((size_t) p.hidden, 0.0f);
        const code_picker pick = [](const std::vector<float> &, int) { return 0; };
        if (!fast_step(real, prime_in, p.semantic_begin, 2, pick, codes, &error)) {
            fail("real fast_step failed: " + error);
        } else {
            ::tts_cpp::detail::fit_graph_price projected;
            bool priced_on_device = true;
            for (int position = 0; position < p.num_codebooks; ++position) {
                scratch build(AUDIO8_FAST_MAX_NODES);
                build_fast_fit_graph(mm, build, position, /*prime=*/position == 0);
                ::tts_cpp::detail::fit_graph_price one;
                if (!price(mm, build, one)) fail("pricing a fast position graph failed");
                projected.device_bytes += one.device_bytes;
                priced_on_device = priced_on_device && one.host_bytes == 0;
            }
            if (priced_on_device) {
                expect_eq(projected.device_bytes, real_fast_arena_bytes(real),
                          "LM fast arena parity");
                expect_eq((uint64_t) p.num_codebooks, retained_fast_contexts(real),
                          "LM retained fast graph contexts");
            }
        }
    }

    free_lm(real);
    free_lm(mm);

    // 5. Errors surface as Error, never Success.
    {
        tts_cpp::audio8::FitOptions bad;
        bad.lm_gguf_path            = path + ".does-not-exist";
        bad.codec_decoder_gguf_path = path;
        tts_cpp::FitResult fr = tts_cpp::audio8::fit_params(bad);
        expect(fr.status == tts_cpp::FitStatus::Error, "missing LM was not Error");
        expect(fr.reason == "model-unreadable",
               "missing LM reason was '" + fr.reason + "'");
        // The LM GGUF is not a codec GGUF: wrong architecture must also be
        // Error, not a projection of garbage.
        tts_cpp::audio8::FitOptions wrong;
        wrong.lm_gguf_path            = path;
        wrong.codec_decoder_gguf_path = path;
        fr = tts_cpp::audio8::fit_params(wrong);
        expect(fr.status == tts_cpp::FitStatus::Error,
               "wrong-architecture decoder was not Error");
    }

    fs::remove(path);
}

void run_fixture_gates(const std::string & lm_path, const std::string & dec_path,
                       const std::string & enc_path, int n_gpu_layers) {
    tts_cpp::audio8::FitOptions fopts;
    fopts.lm_gguf_path            = lm_path;
    fopts.codec_decoder_gguf_path = dec_path;
    fopts.codec_encoder_gguf_path = enc_path;
    fopts.n_gpu_layers            = n_gpu_layers;
    fopts.prompt_tokens           = 16;
    fopts.max_frames              = 64;

    const tts_cpp::FitResult fit = tts_cpp::audio8::fit_params(fopts);
    expect(fit.status != tts_cpp::FitStatus::Error,
           "fit_params returned Error (" + fit.reason + ") for readable fixtures");
    expect(fit.device.weights_bytes > 0,       "projected weights_bytes == 0");
    expect(fit.device.state_bytes > 0,         "projected state_bytes == 0");
    expect(fit.device.lm_compute_bytes > 0,    "projected lm_compute_bytes == 0");
    expect(fit.device.codec_compute_bytes > 0, "projected codec_compute_bytes == 0");
    expect(!fit.report.empty(),                "empty report");
    if (g_failures) return;
    std::printf("%s", fit.report.c_str());

    // Codec decode parity: a real decode leaves the latent arena in
    // model.allocr and the block arena in model.block_allocr; the projection
    // of the same frame count must match both (their sum, since both stay
    // resident), byte for byte on the direct-dispatch path.
    {
        codec_model real;
        std::string error;
        if (!load_codec(dec_path, n_gpu_layers, real, &error)) {
            fail("real load_codec failed: " + error);
            return;
        }
        const int n_frames = 16;
        std::vector<int32_t> codes((size_t) real.hp.num_codebooks * n_frames, 0);
        std::vector<float> pcm;
        if (!decode_codes(real, codes.data(), n_frames, /*n_threads=*/2,
                          cancel_hook(), pcm, &error)) {
            fail("real decode_codes failed: " + error);
        } else {
            codec_model mm;
            fit_load_measure dm;
            if (!load_codec_metadata_only(dec_path, n_gpu_layers, mm, dm, &error)) {
                fail("load_codec_metadata_only failed: " + error);
            } else {
                expect_eq(dm.weights_bytes, ggml_backend_buffer_get_size(real.buffer_w),
                          "codec weights parity");
                codec_fit_measure cm;
                if (!measure_decode_memory(mm, n_frames, cm)) {
                    fail("measure_decode_memory failed");
                } else if (cm.host_bytes == 0) {
                    const uint64_t real_arenas =
                        (uint64_t) ggml_gallocr_get_buffer_size(real.allocr, 0) +
                        ggml_gallocr_get_buffer_size(real.block_allocr, 0);
                    expect_eq(cm.device_bytes, real_arenas, "codec decode arena parity");
                }
                free_codec(mm);
            }
        }
        free_codec(real);
    }

    // Near-INT_MAX workloads are rejected strictly (Error /
    // "workload-too-large"), never priced through wrapped-int graph shapes --
    // a preflight that crashes on the inputs it exists to reject violates its
    // own contract.
    {
        tts_cpp::audio8::FitOptions huge = fopts;
        huge.prompt_tokens = std::numeric_limits<int>::max();
        const tts_cpp::FitResult fr = tts_cpp::audio8::fit_params(huge);
        expect(fr.status == tts_cpp::FitStatus::Error,
               "near-INT_MAX prompt_tokens was not Error");
        expect(fr.reason == "workload-too-large",
               "near-INT_MAX prompt_tokens reason was '" + fr.reason + "'");
    }
    if (!enc_path.empty()) {
        // A reference whose sample count exceeds int must trip the widened
        // encode_positions product guard (or the RoPE-table check, whichever
        // bites first), never sign-overflow inside encoder_positions.
        tts_cpp::audio8::FitOptions huge = fopts;
        huge.reference_seconds = 2.0e5f;  // >= 2^31 samples at any real rate
        const tts_cpp::FitResult fr = tts_cpp::audio8::fit_params(huge);
        expect(fr.status == tts_cpp::FitStatus::Error,
               "over-int reference_seconds was not Error");
        expect(fr.reason == "workload-too-large",
               "over-int reference_seconds reason was '" + fr.reason + "'");
    }
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc >= 3) {
        run_fixture_gates(argv[1], argv[2], argc > 3 ? argv[3] : "",
                          argc > 4 ? std::atoi(argv[4]) : 0);
    } else {
        run_synthetic_lm_gates();
    }
    if (g_failures == 0) {
        std::printf("test-audio8-fit-params: all checks passed\n");
    }
    return g_failures;
}
