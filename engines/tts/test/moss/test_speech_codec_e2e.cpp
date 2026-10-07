// Real-checkpoint decoder regression, without loading the Speech LM. CPU and
// Vulkan receive identical speech codes, short voice conditioning and CFM noise.
#include "moss/speech_codec.h"
#include "backend_selection.h"
#include "tts-cpp/chatterbox/s3gen_pipeline.h"
#include "npy.h"
#include "dr_wav.h"
#include "../test_env_portable.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace tts_cpp::moss::detail;
namespace fs = std::filesystem;

namespace {

// Whisper-VQ codes for a CPU-rendered "The capital of France is Paris."
// Keeping these fixed isolates the decoder from tokenizer/LM differences.
const std::vector<int32_t> CODES{
    1734, 13838, 2233, 16127, 11358, 8577, 14729, 1732, 13871, 15901,
    11369, 1186, 2902, 4850, 429, 4430, 9911, 79, 15829, 2517, 9403,
    10332, 15763, 3926, 10309, 2408, 520, 56, 16226, 10691, 14335,
};

void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

void write_wav(const fs::path & path, const std::vector<float> & pcm) {
    drwav_data_format format{drwav_container_riff, DR_WAVE_FORMAT_PCM, 1, 24000, 16};
    drwav wav{};
    require(drwav_init_file_write(&wav, path.string().c_str(), &format, nullptr), "cannot write WAV");
    std::vector<drwav_int16> samples(pcm.size());
    drwav_f32_to_s16(samples.data(), pcm.data(), pcm.size());
    const auto written = drwav_write_pcm_frames(&wav, samples.size(), samples.data());
    drwav_uninit(&wav);
    require(written == samples.size(), "short WAV write");
}

struct Result {
    std::vector<float> pcm;
    npy_array mel;
};

std::vector<Result> decode(const std::string & model, const SpeechVoice & voice,
                          const std::vector<float> & noise, bool gpu, const fs::path & output) {
    require(s3gen_preload(model, gpu ? 1 : 0) == 0, "decoder preload failed");
    struct Lease { ~Lease() { s3gen_unload(); } } lease;
    std::vector<Result> results;
    // On GPU, cfg=0 exercises basic_tfm (B=1); cfg=0.7 exercises basic_tfm_b
    // (B=2). CPU implements CFG with two B=1 calls, independently of GPU batching.
    for (float cfg : {0.0f, 0.7f}) {
        const std::string label = std::string("codec-") + (gpu ? "vulkan" : "cpu") +
                                  (cfg == 0.0f ? "-single" : "-cfg");
        s3gen_synthesize_opts opts;
        Result result;
        opts.s3gen_gguf_path = model;
        opts.pcm_out = &result.pcm;
        opts.prompt_token_view_data = voice.tokens.data();
        opts.prompt_token_view_size = voice.tokens.size();
        opts.prompt_feat_view_data = voice.feat.data();
        opts.prompt_feat_view_size = voice.feat.size();
        opts.prompt_feat_view_rows = voice.feat_frames;
        opts.embedding_view_data = voice.embedding.data();
        opts.embedding_view_size = voice.embedding.size();
        opts.cfm_z0_override = noise;
        opts.cfg_rate = cfg;
        opts.n_threads = 2;
        opts.n_gpu_layers = gpu ? 1 : 0;
        opts.finalize = true;
        opts.append_lookahead_silence = false;
        opts.apply_trim_fade = false;
        opts.dump_mel_path = (output / (label + "-mel.npy")).string();
        require(s3gen_synthesize_to_wav(CODES, opts) == 0, "codec synthesis failed");
        require(!result.pcm.empty() && std::all_of(result.pcm.begin(), result.pcm.end(),
                [](float x) { return std::isfinite(x); }), "non-finite/empty PCM");
        result.mel = npy_load(opts.dump_mel_path);
        write_wav(output / (label + ".wav"), result.pcm);
        results.push_back(std::move(result));
    }
    return results;
}

double rms(const std::vector<float> & pcm) {
    double sum = 0;
    for (float x : pcm) sum += (double) x * x;
    return std::sqrt(sum / pcm.size());
}

void compare(const Result & cpu, const Result & gpu) {
    require(cpu.mel.n_elements() > 0 && cpu.mel.shape == gpu.mel.shape, "mel shape mismatch");
    const auto stats = compare_f32(npy_as_f32(gpu.mel), npy_as_f32(cpu.mel), cpu.mel.n_elements());
    print_compare("codec mel CPU/Vulkan", stats);
    require(compare_within(stats, 0.02), "non-finite/divergent decoder mel");
    require(cpu.pcm.size() == gpu.pcm.size(), "PCM length mismatch");
    // HiFT integrates f0 into phase: small mel differences can shift waveform
    // phase. Compare signal level here, with the strict parity gate on mel.
    const double reference = rms(cpu.pcm);
    require(reference > 1e-5 && std::fabs(rms(gpu.pcm) / reference - 1) < 0.15,
            "silent/divergent waveform level");
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s CODEC.gguf OUTPUT_DIR\n", argv[0]);
        return 77;
    }
    try {
        // The fallback used during diagnosis must not mask a regression.
        unsetenv("GGML_VK_DISABLE_F16");
        unsetenv("CHATTERBOX_CFG_RATE");
        setenv("TTS_CPP_GPU_BACKEND", "vulkan", 1);
        auto backend = tts_cpp::detail::init_gpu_backend(
            1, true, "moss-speech-codec-e2e", 0, false, nullptr,
            tts_cpp::detail::GpuBackendRequirement::Vulkan);
        require(backend != nullptr, "Vulkan backend required");
        const std::string name = ggml_backend_name(backend);
        ggml_backend_free(backend);
        require(name.find("Vulkan") == 0, "Vulkan backend required");
        std::fprintf(stderr, "[moss-speech-codec-e2e] backend: %s\n", name.c_str());
        const fs::path output(argv[2]);
        fs::create_directories(output);
        SpeechVoice voice;
        int ratio = 0;
        {
            SpeechCodec codec(argv[1], false, 2);
            voice = codec.default_voice();
            ratio = codec.token_mel_ratio();
        } // Release the tokenizer/CAM++ before loading the decoder.
        voice.tokens.resize(std::min<size_t>(32, voice.tokens.size()));
        voice.feat_frames = (int) voice.tokens.size() * ratio;
        voice.feat.resize((size_t) voice.feat_frames * 80);
        require(!voice.tokens.empty(), "missing voice conditioning");
        std::vector<float> noise(80 * ratio * (voice.tokens.size() + CODES.size()));
        std::mt19937 rng(0);
        std::normal_distribution<float> normal(0, 1);
        for (float & x : noise) x = normal(rng);
        const auto cpu = decode(argv[1], voice, noise, false, output);
        const auto gpu = decode(argv[1], voice, noise, true, output);
        for (size_t i = 0; i < cpu.size(); ++i) compare(cpu[i], gpu[i]);
        std::puts("MOSS Speech codec CPU/Vulkan agreement: OK");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "moss speech codec e2e: %s\n", error.what());
        return 1;
    }
}
