// Encoder backend parity: locate the first encoder stage/layer where the
// Hexagon (HTP0) backend diverges from the CPU reference.
//
// QVAC-25495 diagnostic. Per-op test-backend-ops passes 152/152 on HTP0 but
// end-to-end Parakeet inference produces garbage. This harness loads the same
// GGUF twice (backend A = "cpu" by default, backend B = "hexagon" by default),
// runs the same mel through run_encoder() on both, and prints per-stage
// cosine similarity, max_abs difference, and relative L2 for every captured
// tensor. It also sweeps max_layers to pinpoint the first Conformer block
// that diverges.
//
// Usage:
//   test-encoder-backend-parity --model <gguf> --wav <wav>
//                               [--backend-a cpu] [--backend-b hexagon]
//                               [--min-cos 0.99] [--layer-sweep]
//
// Exit 0 on success (all stages meet --min-cos), non-zero otherwise. The
// per-stage table is always printed so a failure is actionable.

#include "parakeet_ctc.h"
#include "mel_preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --model <gguf> --wav <wav>\n"
        "           [--backend-a <name>] [--backend-b <name>]\n"
        "           [--min-cos <float>] [--layer-sweep]\n"
        "\n"
        "Loads <gguf> under two backends and compares per-stage encoder outputs\n"
        "for the same mel input. Defaults: backend-a=cpu, backend-b=hexagon,\n"
        "min-cos=0.99. With --layer-sweep, additionally sweeps max_layers over\n"
        "{0, 1, 2, 4, 8, and every 4th layer} to identify the first diverging\n"
        "Conformer block.\n",
        argv0);
}

struct Metrics {
    double cosine  = 0.0;
    double max_abs = 0.0;
    double rel_l2  = 0.0;
    bool   nonfinite = false;
};

Metrics compute_metrics(const std::vector<float> & a, const std::vector<float> & b) {
    Metrics m;
    const size_t n = std::min(a.size(), b.size());
    double dot = 0.0, na = 0.0, nb = 0.0, diff_sq = 0.0, ref_sq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double av = a[i];
        const double bv = b[i];
        if (!std::isfinite(av) || !std::isfinite(bv)) {
            m.nonfinite = true;
        }
        const double d = av - bv;
        if (std::fabs(d) > m.max_abs) m.max_abs = std::fabs(d);
        dot     += av * bv;
        na      += av * av;
        nb      += bv * bv;
        diff_sq += d * d;
        ref_sq  += av * av;
    }
    const double denom = std::sqrt(na) * std::sqrt(nb);
    m.cosine = denom > 0.0 ? dot / denom : (na == 0.0 && nb == 0.0 ? 1.0 : 0.0);
    m.rel_l2 = ref_sq > 0.0 ? std::sqrt(diff_sq / ref_sq) : std::sqrt(diff_sq);
    return m;
}

struct StageRef {
    const char * name;
    const std::vector<float> parakeet::EncoderOutputs::* ptr;
};

const StageRef kStages[] = {
    {"subsampling_out",      &parakeet::EncoderOutputs::subsampling_out},
    {"block_0_post_ff1",     &parakeet::EncoderOutputs::block_0_post_ff1},
    {"block_0_attn_qkv",     &parakeet::EncoderOutputs::block_0_attn_qkv},
    {"block_0_attn_k_raw",   &parakeet::EncoderOutputs::block_0_attn_k_raw},
    {"block_0_attn_v_raw",   &parakeet::EncoderOutputs::block_0_attn_v_raw},
    {"block_0_attn_q_perm",  &parakeet::EncoderOutputs::block_0_attn_q_perm},
    {"block_0_attn_k_perm",  &parakeet::EncoderOutputs::block_0_attn_k_perm},
    {"block_0_attn_q_u",     &parakeet::EncoderOutputs::block_0_attn_q_u},
    {"block_0_attn_ac",      &parakeet::EncoderOutputs::block_0_attn_ac},
    {"block_0_attn_bd",      &parakeet::EncoderOutputs::block_0_attn_bd},
    {"block_0_attn_scores",  &parakeet::EncoderOutputs::block_0_attn_scores},
    {"block_0_attn_softmax", &parakeet::EncoderOutputs::block_0_attn_softmax},
    {"block_0_attn_out",     &parakeet::EncoderOutputs::block_0_attn_out},
    {"block_0_post_attn",    &parakeet::EncoderOutputs::block_0_post_attn},
    {"block_0_post_conv",    &parakeet::EncoderOutputs::block_0_post_conv},
    {"block_0_post_ff2",     &parakeet::EncoderOutputs::block_0_post_ff2},
    {"block_0_out",          &parakeet::EncoderOutputs::block_0_out},
    {"block_last_out",       &parakeet::EncoderOutputs::block_last_out},
    {"encoder_out",          &parakeet::EncoderOutputs::encoder_out},
};

int load_model(const std::string & gguf, const std::string & backend,
               parakeet::ParakeetCtcModel & out) {
    // Explicit-backend overload: pass n_gpu_layers=999 so backends that pay
    // attention to the flag (opencl / hexagon) actually get the encoder.
    int rc = parakeet::load_from_gguf(gguf, out,
                                      /*n_threads=*/0,
                                      /*n_gpu_layers=*/999,
                                      /*verbose=*/false,
                                      backend);
    return rc;
}

}  // namespace

int main(int argc, char ** argv) {
    std::string model_path;
    std::string wav_path;
    std::string backend_a = "cpu";
    std::string backend_b = "hexagon";
    double min_cos = 0.99;
    bool layer_sweep = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if      (a == "--model"     && i + 1 < argc) model_path = argv[++i];
        else if (a == "--wav"       && i + 1 < argc) wav_path   = argv[++i];
        else if (a == "--backend-a" && i + 1 < argc) backend_a  = argv[++i];
        else if (a == "--backend-b" && i + 1 < argc) backend_b  = argv[++i];
        else if (a == "--min-cos"   && i + 1 < argc) min_cos    = std::atof(argv[++i]);
        else if (a == "--layer-sweep")               layer_sweep = true;
        else if (a == "-h" || a == "--help")         { usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); usage(argv[0]); return 2; }
    }
    if (model_path.empty() || wav_path.empty()) { usage(argv[0]); return 2; }

    using namespace parakeet;

    std::fprintf(stderr, "[backend-parity] loading %s on backend A=%s\n",
                 model_path.c_str(), backend_a.c_str());
    ParakeetCtcModel model_a;
    if (int rc = load_model(model_path, backend_a, model_a); rc != 0) {
        std::fprintf(stderr, "[backend-parity] load(A=%s) rc=%d\n", backend_a.c_str(), rc);
        return 3;
    }
    std::fprintf(stderr, "[backend-parity]   active backend A: %s\n",
                 model_active_backend_name(model_a).c_str());

    std::fprintf(stderr, "[backend-parity] loading %s on backend B=%s\n",
                 model_path.c_str(), backend_b.c_str());
    ParakeetCtcModel model_b;
    if (int rc = load_model(model_path, backend_b, model_b); rc != 0) {
        std::fprintf(stderr, "[backend-parity] load(B=%s) rc=%d\n", backend_b.c_str(), rc);
        return 3;
    }
    std::fprintf(stderr, "[backend-parity]   active backend B: %s\n",
                 model_active_backend_name(model_b).c_str());

    // Mel is computed once from the wav on host; both encoder runs consume
    // the exact same float32 buffer to keep the comparison decoupled from
    // the mel pipeline.
    std::vector<float> samples;
    int sr = 0;
    if (int rc = load_wav_mono_f32(wav_path, samples, sr); rc != 0) {
        std::fprintf(stderr, "[backend-parity] load_wav rc=%d\n", rc);
        return 4;
    }
    if (sr != model_a.mel_cfg.sample_rate) {
        std::fprintf(stderr, "[backend-parity] FAIL: wav sr %d != model sr %d\n",
                     sr, model_a.mel_cfg.sample_rate);
        return 4;
    }
    std::vector<float> mel;
    int n_mel_frames = 0;
    if (int rc = compute_log_mel(samples.data(), (int) samples.size(),
                                 model_a.mel_cfg, mel, n_mel_frames); rc != 0) {
        std::fprintf(stderr, "[backend-parity] compute_log_mel rc=%d\n", rc);
        return 4;
    }
    std::fprintf(stderr, "[backend-parity] wav=%zu samples @ %d Hz, mel=%d frames x %d mels\n",
                 samples.size(), sr, n_mel_frames, model_a.mel_cfg.n_mels);

    EncoderOutputs out_a;
    if (int rc = run_encoder(model_a, mel.data(), n_mel_frames, model_a.mel_cfg.n_mels,
                             out_a, /*max_layers=*/-1, /*capture_intermediates=*/true); rc != 0) {
        std::fprintf(stderr, "[backend-parity] run_encoder(A) rc=%d\n", rc);
        return 5;
    }
    EncoderOutputs out_b;
    if (int rc = run_encoder(model_b, mel.data(), n_mel_frames, model_b.mel_cfg.n_mels,
                             out_b, /*max_layers=*/-1, /*capture_intermediates=*/true); rc != 0) {
        std::fprintf(stderr, "[backend-parity] run_encoder(B) rc=%d\n", rc);
        return 5;
    }
    if (out_a.n_enc_frames != out_b.n_enc_frames || out_a.d_model != out_b.d_model) {
        std::fprintf(stderr,
            "[backend-parity] FAIL: shape metadata drift  n_enc=%d/%d  d_model=%d/%d\n",
            out_a.n_enc_frames, out_b.n_enc_frames, out_a.d_model, out_b.d_model);
        return 6;
    }

    std::fprintf(stderr,
        "\n[backend-parity] per-stage comparison  A=%s  B=%s\n"
        "  %-20s %10s %14s %14s %14s\n",
        backend_a.c_str(), backend_b.c_str(),
        "stage", "floats", "cosine", "max_abs", "rel_l2");

    bool first_fail_reported = false;
    int failures = 0;
    for (const auto & stage : kStages) {
        const auto & va = out_a.*(stage.ptr);
        const auto & vb = out_b.*(stage.ptr);
        if (va.empty() || vb.empty()) {
            std::fprintf(stderr, "  %-20s %10s %14s %14s %14s\n",
                         stage.name, "(empty)", "-", "-", "-");
            continue;
        }
        Metrics m = compute_metrics(va, vb);
        const bool ok = !m.nonfinite && m.cosine >= min_cos;
        std::fprintf(stderr, "  %-20s %10zu %14.6f %14.3e %14.3e  %s%s\n",
                     stage.name, va.size(), m.cosine, m.max_abs, m.rel_l2,
                     ok ? "OK" : "FAIL",
                     m.nonfinite ? " (non-finite!)" : "");
        if (!ok) {
            failures++;
            if (!first_fail_reported) {
                std::fprintf(stderr,
                    "\n  >>> first divergence at stage `%s` (cos=%.6f < %.6f) <<<\n\n",
                    stage.name, m.cosine, min_cos);
                first_fail_reported = true;
            }
        }
    }

    if (layer_sweep) {
        const int n_layers = model_a.encoder_cfg.n_layers;
        std::vector<int> plan;
        for (int L = 0; L <= n_layers; L = (L == 0 ? 1 : (L < 8 ? L * 2 : L + 4))) {
            plan.push_back(std::min(L, n_layers));
            if (L >= n_layers) break;
        }
        // Ensure the final layer is compared.
        if (plan.empty() || plan.back() != n_layers) plan.push_back(n_layers);

        std::fprintf(stderr,
            "\n[backend-parity] layer sweep  (encoder has %d Conformer blocks)\n"
            "  %-12s %10s %14s %14s %14s\n",
            n_layers, "max_layers", "floats", "cosine", "max_abs", "rel_l2");

        for (int L : plan) {
            EncoderOutputs pa, pb;
            if (run_encoder(model_a, mel.data(), n_mel_frames, model_a.mel_cfg.n_mels,
                            pa, L, /*capture_intermediates=*/true) != 0) continue;
            if (run_encoder(model_b, mel.data(), n_mel_frames, model_b.mel_cfg.n_mels,
                            pb, L, /*capture_intermediates=*/true) != 0) continue;
            if (pa.encoder_out.empty() || pb.encoder_out.empty()) continue;
            Metrics m = compute_metrics(pa.encoder_out, pb.encoder_out);
            std::fprintf(stderr, "  %-12d %10zu %14.6f %14.3e %14.3e  %s\n",
                         L, pa.encoder_out.size(), m.cosine, m.max_abs, m.rel_l2,
                         m.cosine >= min_cos ? "OK" : "FAIL");
        }
    }

    if (failures > 0) {
        std::fprintf(stderr,
            "\n[backend-parity] FAIL: %d stage(s) below cosine %.4f\n", failures, min_cos);
        return 1;
    }
    std::fprintf(stderr, "\n[backend-parity] PASS: all stages >= cosine %.4f\n", min_cos);
    return 0;
}
