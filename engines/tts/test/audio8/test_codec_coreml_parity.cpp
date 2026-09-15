// Core ML codec sidecar parity: synthesises deterministic code sequences
// through the sidecar and through the ggml synthesis stack (forced via
// AUDIO8_COREML_DISABLE) and compares the waveforms, so the production
// dispatch in decode_codes is exercised end to end against the ggml
// reference: a short utterance that fits one zero-padded window, a long one
// the plan has to stitch from several, the strict-mode refusals, and a
// cancel between windows.
//
// Skips (exit 77) when the build has no TTS_CPP_COREML, when
// AUDIO8_COREML_TEST_MODELS_DIR is not staged, or when no compiled sidecar
// sits next to the decoder GGUF there.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
constexpr int SKIP_EXIT_CODE = 77;
}

#ifndef TTS_CPP_USE_COREML

int main() {
    std::fprintf(stderr, "[coreml-parity] built without TTS_CPP_COREML; skipping\n");
    return SKIP_EXIT_CODE;
}

#else

#include "audio8/coreml/codec-synth.h"
#include "audio8/coreml_path.h"
#include "audio8/internal.h"

#include <sys/stat.h>

using namespace tts_cpp::audio8::detail;

namespace {

bool path_exists(const std::string & path) {
    struct stat st {};
    return stat(path.c_str(), &st) == 0;
}

std::string find_decoder_gguf(const std::string & dir) {
    const char * candidates[] = {"audio8-codec-decoder-f32.gguf", "audio8-codec-decoder-f16.gguf",
                                 "audio8-codec-decoder-q8_0.gguf", "audio8-codec-decoder.gguf"};
    for (const char * name : candidates) {
        const std::string path = dir + "/" + name;
        if (path_exists(path)) return path;
    }
    return {};
}

// Deterministic codes inside every codebook, [num_codebooks, n_frames]
// row-major as decode_codes reads them.
std::vector<int32_t> make_codes(const codec_hparams & hp, int n_frames, uint32_t seed) {
    std::vector<int32_t> codes(static_cast<size_t>(hp.num_codebooks) * n_frames);
    uint32_t state = seed;
    for (int book = 0; book < hp.num_codebooks; ++book) {
        const int limit = book == 0 ? hp.semantic_codebook_size : hp.residual_codebook_size;
        for (int frame = 0; frame < n_frames; ++frame) {
            state = state * 1664525u + 1013904223u;
            codes[static_cast<size_t>(book) * n_frames + frame] =
                static_cast<int32_t>((state >> 8) % static_cast<uint32_t>(limit));
        }
    }
    return codes;
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

double max_abs_err(const std::vector<float> & a, const std::vector<float> & b) {
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        worst = std::max(worst, (double) std::fabs(a[i] - b[i]));
    }
    return worst;
}

struct owned_codec {
    codec_model model;
    ~owned_codec() { free_codec(model); }
};

}  // namespace

namespace {

// fp16 storage and Neural Engine arithmetic against the f32 ggml stack, over
// an expansive convolution stack: the ACE-Step VAE sidecar measures 0.99999 at
// this gate, and the Audio8 stack is held to the same bar.
constexpr double MIN_COSINE = 0.999;
constexpr int    SHORT_FRAMES = 10;
constexpr int    LONG_FRAMES  = 200;  // ~9.3 s, several windows at any width up to 128
constexpr int    N_THREADS    = 4;

bool decode(codec_model & model, const std::vector<int32_t> & codes, int n_frames,
            const cancel_hook & cancel, std::vector<float> & pcm, decode_timing & timing,
            std::string & error) {
    return decode_codes(model, codes.data(), n_frames, N_THREADS, cancel, pcm, &error, nullptr,
                        &timing);
}

bool load(const std::string & gguf, owned_codec & codec) {
    std::string error;
    if (load_codec(gguf, /*n_gpu_layers=*/0, codec.model, &error)) return true;
    std::fprintf(stderr, "[coreml-parity] load: %s\n", error.c_str());
    return false;
}

bool check_parity(const char * tag, codec_model & ggml, codec_model & coreml, int n_frames) {
    const std::vector<int32_t> codes = make_codes(ggml.hp, n_frames, 0x2468ace1u);
    std::vector<float> pcm_ggml, pcm_coreml;
    decode_timing t_ggml, t_coreml;
    std::string error;

    setenv("AUDIO8_COREML_DISABLE", "1", 1);
    const bool ran_ggml = decode(ggml, codes, n_frames, nullptr, pcm_ggml, t_ggml, error);
    unsetenv("AUDIO8_COREML_DISABLE");
    if (!ran_ggml) {
        std::fprintf(stderr, "[coreml-parity] %s: ggml decode failed: %s\n", tag, error.c_str());
        return false;
    }
    setenv("AUDIO8_COREML_STRICT", "1", 1);
    const bool ran_coreml = decode(coreml, codes, n_frames, nullptr, pcm_coreml, t_coreml, error);
    unsetenv("AUDIO8_COREML_STRICT");
    if (!ran_coreml) {
        std::fprintf(stderr, "[coreml-parity] %s: Core ML decode failed: %s\n", tag, error.c_str());
        return false;
    }
    if (t_ggml.synthesis_backend != "ggml" || t_coreml.synthesis_backend.rfind("coreml", 0) != 0) {
        std::fprintf(stderr, "[coreml-parity] %s: FAIL backends were '%s' and '%s'\n", tag,
                     t_ggml.synthesis_backend.c_str(), t_coreml.synthesis_backend.c_str());
        return false;
    }
    const size_t want = static_cast<size_t>(n_frames) * ggml.hp.frame_size;
    if (pcm_ggml.size() != want || pcm_coreml.size() != want) {
        std::fprintf(stderr, "[coreml-parity] %s: FAIL sizes ggml=%zu coreml=%zu want=%zu\n", tag,
                     pcm_ggml.size(), pcm_coreml.size(), want);
        return false;
    }
    const double cos = cosine(pcm_ggml, pcm_coreml);
    std::fprintf(stderr,
                 "[coreml-parity] %s: frames=%d window=%d samples=%zu cosine=%.7f max_abs=%.3e "
                 "(min cosine %.4f, %s)\n",
                 tag, n_frames, t_coreml.block_frames, want, cos, max_abs_err(pcm_ggml, pcm_coreml),
                 MIN_COSINE, t_coreml.synthesis_backend.c_str());
    return cos >= MIN_COSINE;
}

// Strict mode with the sidecar disabled must fail the decode instead of
// producing the ggml audio, or the parity above could compare ggml against
// itself and pass vacuously.
bool check_strict_refusal(const std::string & gguf) {
    setenv("AUDIO8_COREML_DISABLE", "1", 1);
    setenv("AUDIO8_COREML_STRICT", "1", 1);
    owned_codec codec;
    bool ok = load(gguf, codec);
    if (ok) {
        const std::vector<int32_t> codes = make_codes(codec.model.hp, SHORT_FRAMES, 7u);
        std::vector<float> pcm;
        decode_timing timing;
        std::string error;
        if (decode(codec.model, codes, SHORT_FRAMES, nullptr, pcm, timing, error)) {
            std::fprintf(stderr, "[coreml-parity] FAIL: strict decode without a sidecar returned audio\n");
            ok = false;
        }
        if (codec.model.synthesis_on_coreml) {
            std::fprintf(stderr, "[coreml-parity] FAIL: a disabled sidecar was still attached\n");
            ok = false;
        }
    }
    unsetenv("AUDIO8_COREML_STRICT");
    unsetenv("AUDIO8_COREML_DISABLE");
    return ok;
}

// A cancel between windows must stop with CANCELLED, keep exactly the frames
// of the completed windows, and never fall back to the ggml stack.
bool check_cancellation(codec_model & coreml) {
    const std::vector<int32_t> codes = make_codes(coreml.hp, LONG_FRAMES, 99u);
    int asked = 0;
    const cancel_hook after_one = [&asked] { return asked++ >= 1; };
    std::vector<float> pcm;
    decode_timing timing;
    std::string error;
    setenv("AUDIO8_COREML_STRICT", "1", 1);
    const bool ran = decode(coreml, codes, LONG_FRAMES, after_one, pcm, timing, error);
    unsetenv("AUDIO8_COREML_STRICT");
    if (ran || error != CANCELLED) {
        std::fprintf(stderr, "[coreml-parity] FAIL: cancel %s ('%s')\n",
                     ran ? "was ignored" : "stopped with the wrong error", error.c_str());
        return false;
    }
    const int window = static_cast<int>(audio8_coreml_codec_window_frames(coreml.coreml));
    const size_t one_window = static_cast<size_t>(std::min(window, LONG_FRAMES)) * coreml.hp.frame_size;
    if (pcm.size() != one_window) {
        std::fprintf(stderr, "[coreml-parity] FAIL: cancel kept %zu samples, expected one window of %zu\n",
                     pcm.size(), one_window);
        return false;
    }
    std::fprintf(stderr, "[coreml-parity] cancel: stopped after one window, %zu samples kept\n", pcm.size());
    return true;
}

}  // namespace

int main() {
    const char * models_dir = std::getenv("AUDIO8_COREML_TEST_MODELS_DIR");
    if (models_dir == nullptr || *models_dir == '\0') {
        std::fprintf(stderr, "[coreml-parity] AUDIO8_COREML_TEST_MODELS_DIR not set; skipping\n");
        return SKIP_EXIT_CODE;
    }
    const std::string gguf = find_decoder_gguf(models_dir);
    if (gguf.empty()) {
        std::fprintf(stderr, "[coreml-parity] no codec decoder GGUF under %s; skipping\n", models_dir);
        return SKIP_EXIT_CODE;
    }
    const std::string sidecar = coreml_codec_sidecar_path(gguf);
    if (!path_exists(sidecar)) {
        std::fprintf(stderr, "[coreml-parity] no sidecar at %s; skipping\n", sidecar.c_str());
        return SKIP_EXIT_CODE;
    }

    if (!check_strict_refusal(gguf)) return 1;

    // Two loads of one GGUF: the reference keeps the ggml stack, the other
    // attaches the sidecar. Both on the CPU backend, so the ggml leg is the
    // f32 reference rather than another accelerator's rounding.
    setenv("AUDIO8_COREML_DISABLE", "1", 1);
    owned_codec ggml;
    const bool loaded_ggml = load(gguf, ggml);
    unsetenv("AUDIO8_COREML_DISABLE");
    if (!loaded_ggml) return 1;

    setenv("AUDIO8_COREML_STRICT", "1", 1);
    owned_codec coreml;
    const bool loaded_coreml = load(gguf, coreml);
    unsetenv("AUDIO8_COREML_STRICT");
    if (!loaded_coreml) return 1;
    if (!coreml.model.synthesis_on_coreml || !coreml.model.coreml) {
        std::fprintf(stderr, "[coreml-parity] FAIL: sidecar at %s did not initialise\n", sidecar.c_str());
        return 1;
    }
    const int window = static_cast<int>(audio8_coreml_codec_window_frames(coreml.model.coreml));
    const int context = synthesis_context_frames(coreml.model);
    std::fprintf(stderr, "[coreml-parity] sidecar %s: window %d frames, causal context %d frames\n",
                 sidecar.c_str(), window, context);
    if (context >= window) {
        std::fprintf(stderr, "[coreml-parity] FAIL: the window cannot carry the causal context\n");
        return 1;
    }

    bool ok = check_parity("short", ggml.model, coreml.model, SHORT_FRAMES);
    ok &= check_parity("long", ggml.model, coreml.model, LONG_FRAMES);
    ok &= check_cancellation(coreml.model);
    std::printf("\n%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

#endif  // TTS_CPP_USE_COREML
