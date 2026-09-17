// Cross-backend parity test — CosyVoice3 stages, CPU vs the selected GPU.
//
// Runs each stage twice over identical deterministic synthetic inputs — once
// on the CPU backend, once on the GPU the engine's selection picks — and
// gates the GPU output against the CPU output.  Needs no PyTorch reference
// dumps, only the stage GGUFs, so it runs wherever the models are staged and
// pins what a backend-specific graph path can break:
//
//   llm   engine-level greedy speech-token trajectory, EXACT equality.
//         Covers the Metal flash-attention decode step, the fused-qkv
//         matvec, and the maskless single-token softmax against the naive
//         masked chain.  Engine-level because only real conditioning (the
//         baked voice prompt + tokenized text) keeps the LM's distribution
//         peaked; pseudo-random ids push it near-uniform and a backend's
//         legitimate reduction-order noise then flips greedy argmax ties,
//         so a pipeline-level gate would measure tie luck, not parity.
//         (Measured on M3 Ultra: bit-identical over 1680 consecutive greedy
//         steps with real conditioning; diverges within 64 steps on
//         pseudo-random ids.)
//   flow  mel cosine + max absolute difference.  Covers the DiT's f16-K/V
//         flash attention and the batched grouped conv_pos_embed.
//   hift  waveform cosine with pinned f0.  Covers the fused snake kernel.
//
// Usage:
//   test-cosyvoice-xb --model-dir DIR --flow-gguf FLOW.gguf --hift-gguf HIFT.gguf
//                     [--flow-min-cosine 0.995] [--flow-max-abs 4.0]
//                     [--hift-min-cosine 0.999]
//
// Stages whose GGUF argument is omitted are skipped.  Always requires a GPU
// backend (fails when none initializes) — the CPU-only half of each stage is
// already covered by the fixture and tier tests.

#include "backend_selection.h"
#include "cosyvoice_pipeline.h"
#include "tts-cpp/cosyvoice/engine.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int   kSpeechVocabSize      = 6561;
constexpr int   kMelChannels          = 80;
constexpr int   kTokenMelRatio        = 2;
constexpr int   kSpeakerEmbeddingSize = 192;
constexpr int   kPromptTokenCount     = 48;
constexpr int   kPromptTokenStride    = 37;
constexpr int   kPromptTokenOffset    = 11;
constexpr int   kSpeechTokenCount     = 96;
constexpr int   kSpeechTokenStride    = 91;
constexpr int   kSpeechTokenOffset    = 5;
constexpr float kLogMelMean           = -4.0f;
constexpr float kLogMelStddev         = 2.0f;
constexpr unsigned kPromptFeatSeed    = 7;
constexpr unsigned kEmbeddingSeed     = 13;
constexpr int   kHiftMelFrames        = 120;
constexpr unsigned kHiftMelSeed       = 11;
constexpr int   kHiftSynthSeed        = 42;
constexpr float kVoicedF0BaseHz       = 180.0f;
constexpr float kVoicedF0SwingHz      = 40.0f;
constexpr float kF0PhasePerFrame      = 0.21f;
constexpr int   kVoicingPeriodFrames  = 24;
// One fixed sentence for the engine-level LM leg; the engine tokenizes it
// and prepends the baked voice prompt, which is what keeps the greedy
// trajectory deterministic across backends (see the llm note above).
const char * kLlmText = "The quick brown fox jumps over the lazy dog near the quiet river bank.";

bool parse_bounded_arg(const char * s, double lo, double hi, double & out) {
    char * end = nullptr;
    const double v = std::strtod(s, &end);
    if (end == s || !end || *end != '\0' || !(v >= lo && v <= hi)) return false;
    out = v;
    return true;
}

void compare(const std::vector<float> & ref, const std::vector<float> & got,
             double & cosine, double & max_abs) {
    double dot = 0, nr = 0, ng = 0, mx = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double r = ref[i], g = got[i];
        dot += r * g; nr += r * r; ng += g * g;
        mx = std::max(mx, std::fabs(r - g));
    }
    cosine  = dot / (std::sqrt(nr) * std::sqrt(ng));
    max_abs = mx;
}

std::vector<int> make_token_ids(int count, int stride, int offset, int vocab) {
    std::vector<int> ids(count);
    for (int i = 0; i < count; ++i) ids[i] = (i * stride + offset) % vocab;
    return ids;
}

std::vector<float> make_normal_values(size_t count, unsigned seed, float mean, float stddev) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(mean, stddev);
    std::vector<float> values(count);
    for (float & v : values) v = dist(rng);
    return values;
}

std::vector<float> make_f0_pattern(int frames) {
    std::vector<float> f0(frames);
    for (int t = 0; t < frames; ++t) {
        const bool voiced = (t / kVoicingPeriodFrames) % 2 == 0;
        f0[t] = voiced
            ? kVoicedF0BaseHz + kVoicedF0SwingHz * std::sin(kF0PhasePerFrame * (float)t)
            : 0.0f;
    }
    return f0;
}

std::vector<float> run_flow(const std::string & gguf, ggml_backend_t backend, int & mel_len) {
    const std::vector<int> prompt_token  = make_token_ids(kPromptTokenCount, kPromptTokenStride,
                                                          kPromptTokenOffset, kSpeechVocabSize);
    const std::vector<int> speech_tokens = make_token_ids(kSpeechTokenCount, kSpeechTokenStride,
                                                          kSpeechTokenOffset, kSpeechVocabSize);
    const int mel_len1 = kTokenMelRatio * kPromptTokenCount;
    const std::vector<float> prompt_feat = make_normal_values((size_t)mel_len1 * kMelChannels,
                                                              kPromptFeatSeed, kLogMelMean, kLogMelStddev);
    const std::vector<float> embedding = make_normal_values(kSpeakerEmbeddingSize, kEmbeddingSeed, 0.0f, 1.0f);
    model_ctx m = cosyvoice_load_gguf(gguf, backend);
    std::vector<float> mel = cosyvoice_flow_run(m, prompt_token, speech_tokens,
                                                prompt_feat, mel_len1, embedding, mel_len);
    cosyvoice_free(m);
    return mel;
}

std::vector<float> run_hift(const std::string & gguf, ggml_backend_t backend) {
    const std::vector<float> mel = make_normal_values((size_t)kMelChannels * kHiftMelFrames,
                                                      kHiftMelSeed, kLogMelMean, kLogMelStddev);
    const std::vector<float> f0 = make_f0_pattern(kHiftMelFrames);
    model_ctx m = cosyvoice_load_gguf(gguf, backend);
    std::vector<float> wav = cosyvoice_hift_synth(m, mel, kHiftMelFrames, kHiftSynthSeed,
                                                  nullptr, &f0);
    cosyvoice_free(m);
    return wav;
}

std::vector<int> run_llm(const std::string & model_dir, int n_gpu_layers) {
    tts_cpp::cosyvoice::EngineOptions opts;
    opts.model_dir    = model_dir;
    opts.n_gpu_layers = n_gpu_layers;
    opts.greedy       = true;
    // The engine leaves the backend default (4 threads) at 0, which makes the
    // CPU reference leg -- a full synthesis, not just the LM -- run for tens
    // of minutes on a many-core host and turns this gate into something
    // nobody runs.
    opts.n_threads    = (int) std::thread::hardware_concurrency();
    tts_cpp::cosyvoice::Engine engine(opts);
    tts_cpp::cosyvoice::SynthesisResult res = engine.synthesize(kLlmText);
    return res.speech_tokens;
}

} // namespace

int main(int argc, char ** argv) {
    std::string model_dir, flow_gguf, hift_gguf;
    // Flow defaults calibrated on synthetic inputs, which condition the DiT
    // off-distribution and amplify absolute deviation well past the fixture
    // tests' real-input gates (measured CPU-vs-Metal: cosine 0.9986, max_abs
    // 2.0 -- and the same to 3 digits with f32 attention operands, so the
    // deviation is the backend's f16-staged GEMMs, not any one graph path).
    // The gate exists to catch a broken backend path, where cosine collapses
    // outright, not ulp noise.
    double flow_min_cosine = 0.995, flow_max_abs = 4.0, hift_min_cosine = 0.999;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        bool args_ok = true;
        if      (a == "--model-dir" && i + 1 < argc) model_dir = argv[++i];
        else if (a == "--flow-gguf" && i + 1 < argc) flow_gguf = argv[++i];
        else if (a == "--hift-gguf" && i + 1 < argc) hift_gguf = argv[++i];
        else if (a == "--flow-min-cosine" && i + 1 < argc) args_ok = parse_bounded_arg(argv[++i], 0.0, 1.0, flow_min_cosine);
        else if (a == "--flow-max-abs"    && i + 1 < argc) args_ok = parse_bounded_arg(argv[++i], 0.0, 1e9, flow_max_abs);
        else if (a == "--hift-min-cosine" && i + 1 < argc) args_ok = parse_bounded_arg(argv[++i], 0.0, 1.0, hift_min_cosine);
        else { fprintf(stderr, "usage: %s --model-dir DIR --flow-gguf F --hift-gguf F [bounds]\n", argv[0]); return 1; }
        if (!args_ok) { fprintf(stderr, "FAIL: invalid value for %s\n", a.c_str()); return 1; }
    }
    if (model_dir.empty() && flow_gguf.empty() && hift_gguf.empty()) {
        fprintf(stderr, "usage: %s --model-dir DIR --flow-gguf F --hift-gguf F [bounds]\n", argv[0]);
        return 1;
    }

    ggml_backend_t cpu = ::tts_cpp::detail::init_cpu_backend();
    if (!cpu) { fprintf(stderr, "FAIL: no CPU backend\n"); return 1; }
    ggml_backend_t gpu = ::tts_cpp::detail::init_gpu_backend(
        99, /*verbose=*/false, "test-cosyvoice-xb", /*vulkan_device=*/0,
        /*allow_arm_mali=*/false, /*out_gpu_present_but_unused=*/nullptr,
        cosyvoice_gpu_requirement());
    if (!gpu) {
        fprintf(stderr, "FAIL: no GPU backend initialized\n");
        ggml_backend_free(cpu);
        return 1;
    }
    fprintf(stderr, "cross-backend: CPU vs %s\n", ggml_backend_name(gpu));

    int rc = 0;
    if (!model_dir.empty()) {
        const std::vector<int> ref = run_llm(model_dir, /*n_gpu_layers=*/0);
        const std::vector<int> got = run_llm(model_dir, /*n_gpu_layers=*/99);
        fprintf(stderr, "llm : cpu %zu tokens, gpu %zu tokens\n", ref.size(), got.size());
        if (ref.empty() || ref != got) {
            fprintf(stderr, "FAIL: llm greedy trajectory differs between backends\n");
            rc = 1;
        }
    }
    if (!flow_gguf.empty()) {
        int ref_len = 0, got_len = 0;
        const std::vector<float> ref = run_flow(flow_gguf, cpu, ref_len);
        const std::vector<float> got = run_flow(flow_gguf, gpu, got_len);
        double cosine = 0, max_abs = 0;
        if (ref_len != got_len || ref.size() != got.size()) {
            fprintf(stderr, "FAIL: flow mel shape differs (%d vs %d frames)\n", ref_len, got_len);
            rc = 1;
        } else {
            compare(ref, got, cosine, max_abs);
            fprintf(stderr, "flow: cosine %.6f (min %.6f), max_abs %.4f (max %.4f)\n",
                    cosine, flow_min_cosine, max_abs, flow_max_abs);
            if (!(cosine >= flow_min_cosine) || !(max_abs <= flow_max_abs)) {
                fprintf(stderr, "FAIL: flow mel parity out of bounds\n");
                rc = 1;
            }
        }
    }
    if (!hift_gguf.empty()) {
        const std::vector<float> ref = run_hift(hift_gguf, cpu);
        const std::vector<float> got = run_hift(hift_gguf, gpu);
        double cosine = 0, max_abs = 0;
        if (ref.size() != got.size()) {
            fprintf(stderr, "FAIL: hift wav length differs (%zu vs %zu)\n", ref.size(), got.size());
            rc = 1;
        } else {
            compare(ref, got, cosine, max_abs);
            fprintf(stderr, "hift: cosine %.6f (min %.6f), max_abs %.4f\n",
                    cosine, hift_min_cosine, max_abs);
            if (!(cosine >= hift_min_cosine)) {
                fprintf(stderr, "FAIL: hift waveform parity out of bounds\n");
                rc = 1;
            }
        }
    }

    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    if (rc == 0) fprintf(stderr, "PASS\n");
    return rc;
}
