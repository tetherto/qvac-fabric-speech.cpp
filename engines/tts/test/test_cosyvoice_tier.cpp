// Cross-tier parity test — CosyVoice3 flow / HiFT weight tiers.
//
// Runs the same stage twice over identical deterministic synthetic inputs:
// once on the f32 reference GGUF and once on a reduced-precision tier
// (f16 / bf16 / q8_0 / q4_0), then gates the tier's output against the f32
// output at cosine >= --min-cosine and max absolute difference <=
// --max-abs.  Unlike the fixture tests this needs no PyTorch reference
// dumps, only the two GGUFs, so it runs wherever the tier files are staged
// and pins the one thing a new tier can break: that reduced-precision
// weights still reproduce the f32 stage output.
//
// The HiFT leg pins f0 (cosyvoice_hift_synth's f0_override) so both runs
// integrate identical sine phases: without the pin, sub-Hz f0 differences
// decorrelate the raw waveform with no audible effect and the gate would
// measure phase noise instead of weight-precision loss.
//
// Usage:
//   test-cosyvoice-tier --stage flow --ref-gguf F32.gguf --tier-gguf TIER.gguf
//                       [--min-cosine 0.999] [--max-abs 1.0]
//   test-cosyvoice-tier --stage hift --ref-gguf F32.gguf --tier-gguf TIER.gguf
//                       [--min-cosine 0.999] [--max-abs 0.1]

#include "cosyvoice_pipeline.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

bool parse_bounded_arg(const char * s, double lo, double hi, double & out) {
    char * end = nullptr;
    const double v = std::strtod(s, &end);
    if (end == s || !end || *end != '\0' || !(v >= lo && v <= hi)) return false;
    out = v;
    return true;
}

void compare(const std::vector<float> & ref, const std::vector<float> & tier,
             double & cosine, double & max_abs) {
    double dot = 0, nr = 0, nt = 0, mx = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double r = ref[i], t = tier[i];
        dot += r * t; nr += r * r; nt += t * t;
        mx = std::max(mx, std::fabs(r - t));
    }
    cosine  = dot / (std::sqrt(nr) * std::sqrt(nt));
    max_abs = mx;
}

// Deterministic pseudo-inputs shaped like real conditioning: speech tokens
// spread over the codebook, a prompt mel around log-mel statistics, and a
// unit-scale speaker embedding.  Both models receive byte-identical inputs,
// so any output difference is weight precision alone.
struct flow_inputs {
    std::vector<int>   prompt_token, speech_tokens;
    std::vector<float> prompt_feat, embedding;
    int mel_len1 = 0;
};

flow_inputs make_flow_inputs() {
    flow_inputs in;
    const int T_ptok = 48, T_stok = 96, MEL = 80, SPK = 192;
    for (int i = 0; i < T_ptok; ++i) in.prompt_token.push_back((i * 37 + 11) % 6561);
    for (int i = 0; i < T_stok; ++i) in.speech_tokens.push_back((i * 91 + 5) % 6561);
    in.mel_len1 = 2 * T_ptok;
    std::mt19937 rng(7);
    std::normal_distribution<float> mel_dist(-4.0f, 2.0f);
    std::normal_distribution<float> emb_dist(0.0f, 1.0f);
    in.prompt_feat.resize((size_t)in.mel_len1 * MEL);
    for (float & v : in.prompt_feat) v = mel_dist(rng);
    in.embedding.resize(SPK);
    for (float & v : in.embedding) v = emb_dist(rng);
    return in;
}

std::vector<float> run_flow(const std::string & gguf, const flow_inputs & in, int & mel_len) {
    model_ctx m = cosyvoice_load_gguf(gguf);
    std::vector<float> mel = cosyvoice_flow_run(m, in.prompt_token, in.speech_tokens,
                                                in.prompt_feat, in.mel_len1,
                                                in.embedding, mel_len);
    cosyvoice_free(m);
    return mel;
}

struct hift_inputs {
    std::vector<float> mel, f0;
    int mel_len = 0;
};

hift_inputs make_hift_inputs() {
    hift_inputs in;
    const int MEL = 80;
    in.mel_len = 120;
    std::mt19937 rng(11);
    std::normal_distribution<float> mel_dist(-4.0f, 2.0f);
    in.mel.resize((size_t)MEL * in.mel_len);
    for (float & v : in.mel) v = mel_dist(rng);
    in.f0.resize(in.mel_len);
    for (int t = 0; t < in.mel_len; ++t) {
        const bool voiced = (t / 20) % 2 == 0;
        in.f0[t] = voiced ? 150.0f + 30.0f * std::sin(0.2f * (float)t) : 0.0f;
    }
    return in;
}

std::vector<float> run_hift(const std::string & gguf, const hift_inputs & in) {
    model_ctx m = cosyvoice_load_gguf(gguf);
    std::vector<float> wav = cosyvoice_hift_synth(m, in.mel, in.mel_len, /*seed=*/42,
                                                  /*tmg=*/nullptr, &in.f0);
    cosyvoice_free(m);
    return wav;
}

}  // namespace

int main(int argc, char ** argv) {
    std::string stage, ref_gguf, tier_gguf;
    double min_cosine = 0.999, max_abs = -1.0;
    bool args_ok = true;
    for (int i = 1; i < argc && args_ok; ++i) {
        std::string a = argv[i];
        if (a == "--stage" && i + 1 < argc) stage = argv[++i];
        else if (a == "--ref-gguf" && i + 1 < argc) ref_gguf = argv[++i];
        else if (a == "--tier-gguf" && i + 1 < argc) tier_gguf = argv[++i];
        else if (a == "--min-cosine" && i + 1 < argc) args_ok = parse_bounded_arg(argv[++i], 0.0, 1.0, min_cosine);
        else if (a == "--max-abs" && i + 1 < argc) args_ok = parse_bounded_arg(argv[++i], 0.0, 1e9, max_abs);
        else args_ok = false;
    }
    if (!args_ok || ref_gguf.empty() || tier_gguf.empty() ||
        (stage != "flow" && stage != "hift")) {
        fprintf(stderr, "usage: %s --stage {flow|hift} --ref-gguf F32.gguf --tier-gguf TIER.gguf\n"
                        "          [--min-cosine 0.999] [--max-abs BOUND]\n", argv[0]);
        return 2;
    }
    if (max_abs < 0) max_abs = stage == "flow" ? 1.0 : 0.1;

    std::vector<float> ref, tier;
    if (stage == "flow") {
        const flow_inputs in = make_flow_inputs();
        int ref_len = 0, tier_len = 0;
        ref  = run_flow(ref_gguf, in, ref_len);
        tier = run_flow(tier_gguf, in, tier_len);
        if (ref_len != tier_len || ref.size() != tier.size()) {
            fprintf(stderr, "FAIL: tier mel shape differs (ref %d frames, tier %d frames)\n",
                    ref_len, tier_len);
            return 1;
        }
    } else {
        const hift_inputs in = make_hift_inputs();
        ref  = run_hift(ref_gguf, in);
        tier = run_hift(tier_gguf, in);
        if (ref.size() != tier.size()) {
            fprintf(stderr, "FAIL: tier waveform length differs (ref %zu, tier %zu)\n",
                    ref.size(), tier.size());
            return 1;
        }
    }

    double cos = 0, mx = 0;
    compare(ref, tier, cos, mx);
    fprintf(stderr, "%s tier-vs-f32: cosine = %.6f (threshold %.4f)  max|diff| = %.4f (bound %.4f)\n",
            stage.c_str(), cos, min_cosine, mx, max_abs);
    if (!(cos >= min_cosine) || !(mx <= max_abs)) {
        fprintf(stderr, "FAIL: tier output diverges from f32\n");
        return 1;
    }
    printf("OK\n");
    return 0;
}
