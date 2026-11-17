// provided("pocket.time_bias"): the FlowLM step's folded time condition.
//
// scripts/pocket_coreml_models.py folds the LSD flow's time embeddings at
// s = 0 and t = 1 into a constant when it builds the step (FlowStep:
// time_bias = time_condition(0, 1)), and coremltools adds that constant to
// flow_net.cond_embed.bias, so the step's weight.bin holds
// f16(cond_embed.bias + time_bias). The recipe computes the sum; this port
// computes time_bias:
//
//   time_condition(s, t) = 0.5 * (time_embedding(0, s) + time_embedding(1, t))
//   time_embedding(i, x) = rms(mlp.2(silu(mlp.0([cos(x * freqs), sin(x * freqs)])))) * mlp.3.alpha
//
// where rms(h) = h / sqrt(var(h) + 1e-5) with the unbiased variance
// (n / (n - 1) * mean((h - mean(h))^2)), Kyutai's "RMSNorm". The exporter
// evaluates it in float32 PyTorch; this port accumulates in double in a fixed
// order and rounds once to float32, so a value may differ by one float16 ulp
// after the final cast (the recipe language marks provided() blobs `ulp`).

#include "pocket.h"

#include "qvac_coreml/recipe.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace QVAC_COREML_NS {
namespace pocket_provided {
namespace {

constexpr double NORM_EPS = 1e-5;
constexpr int N_EMBEDDINGS = 2;
constexpr char TIME_BIAS_KEY[] = "pocket.time_bias";

using recipe::ProvidedCall;
using recipe::TensorSource;

Error read(TensorSource & src, const std::string & name, std::vector<float> & out, int64_t want) {
    const recipe::TensorInfo * ti = src.tensor(name);
    if (ti == nullptr) return Error::deterministic(std::string(TIME_BIAS_KEY) + ": tensor " + name + " is absent");
    Error e = src.read_f32(name, out, nullptr);
    if (!e.ok()) return e;
    if (static_cast<int64_t>(out.size()) != want) {
        return Error::deterministic(std::string(TIME_BIAS_KEY) + ": tensor " + name + " has " +
                                    std::to_string(out.size()) + " values, expected " + std::to_string(want));
    }
    return {};
}

// One time embedding at `time`, in double.
Error time_embedding(TensorSource & src, int index, double time, int64_t width, std::vector<double> & out) {
    const std::string p = "flow_net.time_embed." + std::to_string(index);
    const recipe::TensorInfo * fi = src.tensor(p + ".freqs");
    if (fi == nullptr) return Error::deterministic(std::string(TIME_BIAS_KEY) + ": tensor " + p + ".freqs is absent");
    const int64_t n_freqs = fi->ne[0];
    std::vector<float> freqs, w0, b0, w2, b2, alpha;
    Error e = read(src, p + ".freqs", freqs, n_freqs);
    if (e.ok()) e = read(src, p + ".mlp.0.weight", w0, width * 2 * n_freqs);
    if (e.ok()) e = read(src, p + ".mlp.0.bias", b0, width);
    if (e.ok()) e = read(src, p + ".mlp.2.weight", w2, width * width);
    if (e.ok()) e = read(src, p + ".mlp.2.bias", b2, width);
    if (e.ok()) e = read(src, p + ".mlp.3.alpha", alpha, width);
    if (!e.ok()) return e;

    // The exporter forms time * freqs in float32 and takes cos and sin of it.
    std::vector<double> x(static_cast<size_t>(2 * n_freqs));
    for (int64_t k = 0; k < n_freqs; ++k) {
        const double arg = static_cast<double>(static_cast<float>(time) * freqs[static_cast<size_t>(k)]);
        x[static_cast<size_t>(k)] = std::cos(arg);
        x[static_cast<size_t>(n_freqs + k)] = std::sin(arg);
    }
    // mlp.0 (Linear, weight [width, 2 * n_freqs]) and SiLU.
    std::vector<double> h1(static_cast<size_t>(width));
    for (int64_t o = 0; o < width; ++o) {
        double acc = 0.0;
        const float * row = w0.data() + o * 2 * n_freqs;
        for (int64_t k = 0; k < 2 * n_freqs; ++k) acc += static_cast<double>(row[k]) * x[static_cast<size_t>(k)];
        const double v = acc + static_cast<double>(b0[static_cast<size_t>(o)]);
        h1[static_cast<size_t>(o)] = v / (1.0 + std::exp(-v));
    }
    // mlp.2 (Linear, weight [width, width]).
    std::vector<double> h(static_cast<size_t>(width));
    double sum = 0.0;
    for (int64_t o = 0; o < width; ++o) {
        double acc = 0.0;
        const float * row = w2.data() + o * width;
        for (int64_t k = 0; k < width; ++k) acc += static_cast<double>(row[k]) * h1[static_cast<size_t>(k)];
        h[static_cast<size_t>(o)] = acc + static_cast<double>(b2[static_cast<size_t>(o)]);
        sum += h[static_cast<size_t>(o)];
    }
    // mlp.3: divide by the unbiased standard deviation, scale by alpha.
    const double mean = sum / static_cast<double>(width);
    double sq = 0.0;
    for (double v : h) sq += (v - mean) * (v - mean);
    const double var = sq / static_cast<double>(width) * (static_cast<double>(width) / static_cast<double>(width - 1)) +
                       NORM_EPS;
    const double inv = 1.0 / std::sqrt(var);
    out.resize(static_cast<size_t>(width));
    for (int64_t o = 0; o < width; ++o) {
        out[static_cast<size_t>(o)] = h[static_cast<size_t>(o)] * inv * static_cast<double>(alpha[static_cast<size_t>(o)]);
    }
    return {};
}

Error time_bias(TensorSource & src, const ProvidedCall & call, std::vector<float> & out) {
    int64_t width = 1;
    for (int64_t d : call.shape) width *= d;
    if (width < 2) return Error::deterministic(std::string(TIME_BIAS_KEY) + ": the declared shape needs 2+ values");
    // The LSD step from s = 0 to t = 1.
    const double times[N_EMBEDDINGS] = {0.0, 1.0};
    std::vector<double> acc(static_cast<size_t>(width), 0.0), emb;
    for (int i = 0; i < N_EMBEDDINGS; ++i) {
        Error e = time_embedding(src, i, times[i], width, emb);
        if (!e.ok()) return e;
        for (int64_t o = 0; o < width; ++o) acc[static_cast<size_t>(o)] += emb[static_cast<size_t>(o)];
    }
    out.resize(static_cast<size_t>(width));
    for (int64_t o = 0; o < width; ++o) out[static_cast<size_t>(o)] = static_cast<float>(0.5 * acc[static_cast<size_t>(o)]);
    return {};
}

// The host tool links this file directly; registering on load gives it the
// key without a call. The tts library calls register_all() itself.
[[maybe_unused]] const bool registered_on_load = (register_all(), true);

}  // namespace

void register_all() { recipe::register_provided(TIME_BIAS_KEY, &time_bias); }

}  // namespace pocket_provided
}  // namespace QVAC_COREML_NS
