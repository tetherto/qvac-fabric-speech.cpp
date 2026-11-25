// Core ML VAE decoder parity: decodes one deterministic latent through the
// Core ML sidecar and through the ggml decoder (forced via
// ACESTEP_COREML_DISABLE) and compares the audio by cosine similarity, so the
// production sidecar path is exercised end to end against the ggml reference.
//
//   test-vae-coreml-parity                    the prebuilt vae-decoder.mlmodelc
//                                             next to the VAE GGUF
//   test-vae-coreml-parity --cache-dir DIR    the sidecar generated from the
//                                             GGUF into DIR (no prebuilt may
//                                             sit next to the GGUF); a second
//                                             Vae on the same DIR must find
//                                             the published artifact (route
//                                             cached) and match it too
//
// Every Core ML leg runs with use = require, so a sidecar that cannot serve
// fails the run instead of silently measuring ggml, and must report a
// coreml-* label.
//
// Skips (exit 77) when the build has no AUDIOGEN_COREML, when
// AUDIOGEN_TEST_MODELS_DIR is not staged, or, without --cache-dir, when no
// compiled `vae-decoder.mlmodelc` sits next to the VAE GGUF.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "parity_checks.h"

namespace {
constexpr int SKIP_EXIT_CODE = 77;
}

#ifndef AUDIOGEN_USE_COREML

int main() {
    std::fprintf(stderr, "[coreml-parity] built without AUDIOGEN_COREML; skipping\n");
    return SKIP_EXIT_CODE;
}

#else

#include "audiogen-cpp/acestep/vae.h"
#include "vae_coreml_path.h"

#include <sys/stat.h>

#include <stdexcept>

namespace {

using tts_cpp::acestep::CoremlOptions;
using tts_cpp::acestep::Vae;
using tts_cpp::acestep::VaeOptions;

constexpr int    LATENT_CHANNELS = 64;
constexpr int    T_LATENT        = 1024;
constexpr double MIN_COSINE      = 0.999;

int g_failures = 0;

void check(bool ok, const std::string & what) {
    std::fprintf(stderr, "[coreml-parity] [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_failures;
}

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
std::vector<float> make_latent() {
    std::vector<float> latent((size_t) T_LATENT * LATENT_CHANNELS);
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

std::string json_string(const std::string & json, const std::string & key) {
    const std::string k = "\"" + key + "\":\"";
    const size_t at = json.find(k);
    if (at == std::string::npos) return {};
    const size_t end = json.find('"', at + k.size());
    return end == std::string::npos ? std::string() : json.substr(at + k.size(), end - at - k.size());
}

std::unique_ptr<Vae> load(const std::string & gguf, const std::string & cache_dir, bool require) {
    VaeOptions opts;
    opts.with_encoder = false;
    if (require) opts.coreml.use = CoremlOptions::Use::require;
    if (!cache_dir.empty()) opts.coreml.cache_dir = cache_dir;
    return Vae::load(gguf, opts);
}

// One Core ML leg: a strict decode on a fresh Vae, its label and its parity
// against the ggml decode.
std::vector<float> coreml_leg(const std::string & gguf, const std::string & cache_dir, const std::string & name,
                              const std::vector<float> & latent, const std::vector<float> & pcm_ggml,
                              std::string * route) {
    std::unique_ptr<Vae> vae;
    try {
        vae = load(gguf, cache_dir, true);
    } catch (const std::exception & e) {
        check(false, name + ": strict load: " + e.what());
        return {};
    }
    std::vector<float> pcm = vae->decode(latent, T_LATENT);
    const std::string label = vae->last_decode_backend();
    const std::string status = vae->coreml_status();
    if (route) *route = json_string(status, "route");
    check(!pcm.empty() && label.rfind("coreml-", 0) == 0 && !vae->last_decode_coreml_pending(),
          name + ": strict decode ran on " + label + " (route " + json_string(status, "route") + ")");
    if (pcm.empty() || pcm.size() != pcm_ggml.size()) {
        check(false, name + ": decode sizes ggml=" + std::to_string(pcm_ggml.size()) + " coreml=" +
                         std::to_string(pcm.size()));
        return {};
    }
    const double cos = cosine(pcm_ggml, pcm);
    char line[160];
    std::snprintf(line, sizeof(line), "%s: T_latent=%d samples=%zu cosine=%.7f (min %.4f)", name.c_str(), T_LATENT,
                  pcm.size(), cos, MIN_COSINE);
    check(audiogen_test::cosine_passes(cos, MIN_COSINE), line);
    return pcm;
}

// A callback returning false must cancel and yield an empty result, never
// fall back to the ggml decode.
void check_coreml_cancellation(const std::string & gguf, const std::string & cache_dir,
                               const std::vector<float> & latent) {
    std::unique_ptr<Vae> vae = load(gguf, cache_dir, true);
    int calls = 0;
    const std::vector<float> pcm = vae->decode(latent, T_LATENT, [&calls](int, int) {
        ++calls;
        return false;
    });
    check(pcm.empty() && calls == 1, "cancellation: " + std::to_string(pcm.size()) + " samples after " +
                                         std::to_string(calls) + " callback(s)");
}

}  // namespace

int main(int argc, char ** argv) {
    std::string cache_dir;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--cache-dir" && i + 1 < argc) {
            cache_dir = argv[++i];
        } else {
            std::fprintf(stderr, "usage: test-vae-coreml-parity [--cache-dir DIR]\n");
            return 2;
        }
    }
    const char * models_dir = std::getenv("AUDIOGEN_TEST_MODELS_DIR");
    if (models_dir == nullptr || *models_dir == '\0') {
        std::fprintf(stderr, "[coreml-parity] AUDIOGEN_TEST_MODELS_DIR not set; skipping\n");
        return SKIP_EXIT_CODE;
    }
    const std::string gguf = find_vae_gguf(models_dir);
    if (gguf.empty()) {
        std::fprintf(stderr, "[coreml-parity] no VAE GGUF under %s; skipping\n", models_dir);
        return SKIP_EXIT_CODE;
    }
    const std::string sidecar = tts_cpp::acestep::coreml_vae_sidecar_path(gguf);
    if (cache_dir.empty() && !path_exists(sidecar)) {
        std::fprintf(stderr, "[coreml-parity] no sidecar at %s; skipping\n", sidecar.c_str());
        return SKIP_EXIT_CODE;
    }
    if (!cache_dir.empty() && path_exists(sidecar)) {
        std::fprintf(stderr, "[coreml-parity] FAIL: --cache-dir tests the generated sidecar, but a prebuilt one "
                             "sits at %s\n", sidecar.c_str());
        return 1;
    }

    const std::vector<float> latent = make_latent();

    // The kill switch wins over strict: the decode runs on ggml and says so.
    setenv("ACESTEP_COREML_DISABLE", "1", 1);
    std::vector<float> pcm_ggml;
    {
        std::unique_ptr<Vae> vae = load(gguf, cache_dir, true);
        pcm_ggml = vae->decode(latent, T_LATENT);
        check(!pcm_ggml.empty() && vae->last_decode_backend() == "ggml",
              "ACESTEP_COREML_DISABLE: the reference decode ran on " + vae->last_decode_backend());
    }
    unsetenv("ACESTEP_COREML_DISABLE");

    // Strict with no usable sidecar (no cache directory, nothing prebuilt)
    // fails at load instead of decoding on ggml.
    if (!cache_dir.empty()) {
        bool threw = false;
        try {
            load(gguf, "", true);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        check(threw, "strict without a sidecar or a cache directory throws at load");
    }

    std::string route;
    const std::vector<float> pcm_coreml =
        coreml_leg(gguf, cache_dir, cache_dir.empty() ? "prebuilt" : "generated", latent, pcm_ggml, &route);
    if (!cache_dir.empty()) {
        check(route == "generated" || route == "cached", "first generated leg route " + route);
        // A second owner finds the published artifact.
        std::string route2;
        const std::vector<float> pcm2 = coreml_leg(gguf, cache_dir, "cached", latent, pcm_ggml, &route2);
        check(route2 == "cached", "second leg route " + route2 + " (want cached)");
        check(!pcm2.empty() && pcm2 == pcm_coreml, "the cached sidecar decodes bit-identically to the first leg");
    }
    if (!pcm_coreml.empty()) check_coreml_cancellation(gguf, cache_dir, latent);

    std::fprintf(stderr, "[coreml-parity] %s (%d failure(s))\n", g_failures ? "FAIL" : "OK", g_failures);
    return g_failures ? 1 : 0;
}

#endif  // AUDIOGEN_USE_COREML
