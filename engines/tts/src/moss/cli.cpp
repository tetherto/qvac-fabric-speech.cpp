#include "tts-cpp/moss/engine.h"

#include "dr_wav.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void print_usage() {
    std::fprintf(stderr,
        "usage: moss-cli --backbone model.gguf --decoder decoder.gguf --text \"...\" --out out.wav\n"
        "       [--encoder encoder.gguf --ref-audio ref.wav]   voice cloning\n"
        "       [--language zh] [--max-new-tokens 2048] [--context 4096]\n"
        "       [--seed 1234] [--threads 4] [--gpu]\n"
        "       [--text-temperature 1.5] [--text-top-p 1.0] [--text-top-k 50]\n"
        "       [--audio-temperature 1.7] [--audio-top-p 0.8] [--audio-top-k 25]\n"
        "       [--audio-repetition-penalty 1.0]\n");
}

bool save_wav(const std::string & path, const std::vector<float> & pcm, int sample_rate) {
    drwav_data_format format{};
    format.container = drwav_container_riff;
    format.format = DR_WAVE_FORMAT_PCM;
    format.channels = 1;
    format.sampleRate = (drwav_uint32) sample_rate;
    format.bitsPerSample = 16;
    drwav wav{};
    if (!drwav_init_file_write(&wav, path.c_str(), &format, nullptr)) {
        return false;
    }
    std::vector<drwav_int16> samples(pcm.size());
    drwav_f32_to_s16(samples.data(), pcm.data(), pcm.size());
    const drwav_uint64 written = drwav_write_pcm_frames(&wav, samples.size(), samples.data());
    drwav_uninit(&wav);
    return written == samples.size();
}

struct CliArgs {
    tts_cpp::moss::EngineOptions options;
    std::string text;
    std::string out_path = "moss-out.wav";
};

bool parse_args(int argc, char ** argv, CliArgs & args) {
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + flag);
            }
            return argv[++i];
        };
        if (flag == "--backbone")           args.options.backbone_path = next();
        else if (flag == "--decoder")       args.options.decoder_path = next();
        else if (flag == "--encoder")       args.options.encoder_path = next();
        else if (flag == "--ref-audio")     args.options.reference_audio_path = next();
        else if (flag == "--language")      args.options.language = next();
        else if (flag == "--text")          args.text = next();
        else if (flag == "--out")           args.out_path = next();
        else if (flag == "--max-new-tokens") args.options.max_new_tokens = std::atoi(next());
        else if (flag == "--context")       args.options.context = std::atoi(next());
        else if (flag == "--seed")          args.options.seed = (uint32_t) std::strtoul(next(), nullptr, 10);
        else if (flag == "--threads")       args.options.n_threads = std::atoi(next());
        else if (flag == "--gpu")           args.options.use_gpu = true;
        else if (flag == "--text-temperature")  args.options.text_temperature = std::strtof(next(), nullptr);
        else if (flag == "--text-top-p")        args.options.text_top_p = std::strtof(next(), nullptr);
        else if (flag == "--text-top-k")        args.options.text_top_k = std::atoi(next());
        else if (flag == "--audio-temperature") args.options.audio_temperature = std::strtof(next(), nullptr);
        else if (flag == "--audio-top-p")       args.options.audio_top_p = std::strtof(next(), nullptr);
        else if (flag == "--audio-top-k")       args.options.audio_top_k = std::atoi(next());
        else if (flag == "--audio-repetition-penalty")
            args.options.audio_repetition_penalty = std::strtof(next(), nullptr);
        else {
            std::fprintf(stderr, "unknown flag: %s\n", flag.c_str());
            return false;
        }
    }
    if (args.options.backbone_path.empty() || args.options.decoder_path.empty() || args.text.empty()) {
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    CliArgs args;
    try {
        if (!parse_args(argc, argv, args)) {
            print_usage();
            return 1;
        }
        tts_cpp::moss::Engine engine(args.options);
        std::fprintf(stderr, "[moss-cli] backend: %s\n", engine.backend_name());
        const tts_cpp::moss::SynthesisResult result = engine.synthesize(args.text);
        if (result.cancelled) {
            std::fprintf(stderr, "[moss-cli] synthesis cancelled\n");
            return 1;
        }
        if (!save_wav(args.out_path, result.pcm, result.sample_rate)) {
            std::fprintf(stderr, "[moss-cli] cannot write %s\n", args.out_path.c_str());
            return 1;
        }
        std::fprintf(stderr,
            "[moss-cli] wrote %s: %.2fs @ %d Hz (%d frames, generate %.0f ms, decode %.0f ms)\n",
            args.out_path.c_str(), (double) result.pcm.size() / result.sample_rate,
            result.sample_rate, result.generated_frames, result.generation_ms, result.decode_ms);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "[moss-cli] error: %s\n", error.what());
        return 1;
    }
}
