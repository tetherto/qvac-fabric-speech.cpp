// Core ML VAE decoder benchmark: times the ggml GPU decode and the Core ML
// sidecar decode of the same deterministic latents and prints a markdown
// table to stdout (detailed lines go to stderr), so a CI step can append the
// numbers to its job summary. The first sidecar decode is a discarded warm-up
// because a freshly compiled .mlmodelc pays a one-time on-device ANE
// compilation. The two decodes must also agree at cosine 0.999, so a run
// that produces fast but wrong audio fails instead of reporting a win.
//
// Skips (exit 77) under the same conditions as test-vae-coreml-parity.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace {
constexpr int SKIP_EXIT_CODE = 77;
}

#ifndef AUDIOGEN_USE_COREML

int main() {
    std::fprintf(stderr, "[bench-vae-coreml] built without AUDIOGEN_COREML; skipping\n");
    return SKIP_EXIT_CODE;
}

#else

#include "audiogen-cpp/acestep/vae.h"
#include "vae_coreml_path.h"

#include <sys/stat.h>

namespace {

constexpr int    LATENT_CHANNELS  = 64;
constexpr int    BENCH_T_LATENTS[] = {750, 1500};  // 30 s and 60 s at 5 Hz
constexpr int    BENCH_REPS       = 3;
constexpr double MIN_COSINE       = 0.999;

bool path_exists(const std::string & path) {
    struct stat st{};
    return stat(path.c_str(), &st) == 0;
}

std::string find_vae_gguf(const std::string & dir) {
    const char * candidates[] = {"vae-BF16.gguf", "vae-F16.gguf", "vae.gguf"};
    for (const char * name : candidates) {
        const std::string path = dir + "/" + name;
        if (path_exists(path)) return path;
    }
    return {};
}

// Deterministic latent in the amplitude range real DiT latents occupy.
std::vector<float> make_latent(int T_latent) {
    std::vector<float> latent((size_t) T_latent * LATENT_CHANNELS);
    uint32_t state = 0x2468ace1u;
    for (float & v : latent) {
        state = state * 1664525u + 1013904223u;
        v = ((float) (state >> 8) / (float) (1u << 24)) * 4.0f - 2.0f;
    }
    return latent;
}

double cosine(const std::vector<float> & a, const std::vector<float> & b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += (double) a[i] * (double) b[i];
        na  += (double) a[i] * (double) a[i];
        nb  += (double) b[i] * (double) b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
}

std::unique_ptr<tts_cpp::acestep::Vae> load_vae(const std::string & gguf) {
    tts_cpp::acestep::VaeOptions opts;
    opts.verbose      = true;  // logs the sidecar window and compute-unit label
    opts.with_encoder = false;
    opts.n_gpu_layers = 1;
    return tts_cpp::acestep::Vae::load(gguf, opts);
}

double time_decode(const tts_cpp::acestep::Vae & vae, const std::vector<float> & latent,
                   int T_latent, std::vector<float> * pcm_out) {
    std::vector<double> times;
    for (int rep = 0; rep < BENCH_REPS; ++rep) {
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<float> pcm = vae.decode(latent, T_latent);
        const auto t1 = std::chrono::steady_clock::now();
        if (pcm.empty()) return -1.0;
        times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        if (pcm_out != nullptr) *pcm_out = std::move(pcm);
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

struct BenchRow {
    int    T_latent;
    double ggml_ms;
    double coreml_ms;
    double cos;
};

bool bench_one(const std::string & gguf, int T_latent, BenchRow * row) {
    const std::vector<float> latent = make_latent(T_latent);

    setenv("ACESTEP_COREML_DISABLE", "1", 1);
    const auto vae_ggml = load_vae(gguf);
    unsetenv("ACESTEP_COREML_DISABLE");
    std::vector<float> pcm_ggml;
    const double ggml_ms = time_decode(*vae_ggml, latent, T_latent, &pcm_ggml);

    const auto vae_coreml = load_vae(gguf);
    std::vector<float> warmup = vae_coreml->decode(latent, T_latent);
    if (warmup.empty()) {
        std::fprintf(stderr, "[bench-vae-coreml] FAIL: warm-up decode empty at T=%d\n", T_latent);
        return false;
    }
    std::vector<float> pcm_coreml;
    const double coreml_ms = time_decode(*vae_coreml, latent, T_latent, &pcm_coreml);

    if (ggml_ms < 0.0 || coreml_ms < 0.0 || pcm_ggml.size() != pcm_coreml.size()) {
        std::fprintf(stderr, "[bench-vae-coreml] FAIL: decode sizes ggml=%zu coreml=%zu at T=%d\n",
                     pcm_ggml.size(), pcm_coreml.size(), T_latent);
        return false;
    }

    *row = {T_latent, ggml_ms, coreml_ms, cosine(pcm_ggml, pcm_coreml)};
    std::fprintf(stderr,
                 "[bench-vae-coreml] T=%d ggml=%.0f ms coreml=%.0f ms (median of %d) cosine=%.7f\n",
                 T_latent, ggml_ms, coreml_ms, BENCH_REPS, row->cos);
    return row->cos >= MIN_COSINE;
}

void print_markdown(const std::vector<BenchRow> & rows) {
    std::printf("### ACE-Step VAE decode: ggml GPU vs Core ML sidecar\n\n");
    std::printf("| T_latent | audio | ggml GPU | Core ML | speedup | cosine |\n");
    std::printf("|---|---|---|---|---|---|\n");
    for (const BenchRow & r : rows) {
        std::printf("| %d | %.0f s | %.0f ms | %.0f ms | %.2fx | %.5f |\n",
                    r.T_latent, r.T_latent / 25.0, r.ggml_ms, r.coreml_ms,
                    r.ggml_ms / r.coreml_ms, r.cos);
    }
}

}  // namespace

int main() {
    const char * models_dir = std::getenv("AUDIOGEN_TEST_MODELS_DIR");
    if (models_dir == nullptr || *models_dir == '\0') {
        std::fprintf(stderr, "[bench-vae-coreml] AUDIOGEN_TEST_MODELS_DIR not set; skipping\n");
        return SKIP_EXIT_CODE;
    }
    const std::string gguf = find_vae_gguf(models_dir);
    if (gguf.empty()) {
        std::fprintf(stderr, "[bench-vae-coreml] no VAE GGUF under %s; skipping\n", models_dir);
        return SKIP_EXIT_CODE;
    }
    const std::string sidecar = tts_cpp::acestep::coreml_vae_sidecar_path(gguf);
    if (!path_exists(sidecar)) {
        std::fprintf(stderr, "[bench-vae-coreml] no sidecar at %s; skipping\n", sidecar.c_str());
        return SKIP_EXIT_CODE;
    }

    std::vector<BenchRow> rows;
    for (int T_latent : BENCH_T_LATENTS) {
        BenchRow row{};
        if (!bench_one(gguf, T_latent, &row)) return 1;
        rows.push_back(row);
    }

    print_markdown(rows);
    return 0;
}

#endif  // AUDIOGEN_USE_COREML
