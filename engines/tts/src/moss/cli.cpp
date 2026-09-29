#include "moss/cli.h"

#include "dr_wav.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace tts_cpp::moss::cli {

void print_usage() {
    std::fprintf(stderr,
        "usage: moss-cli --backbone model.gguf --decoder decoder.gguf --text \"...\" --out out.wav\n"
        "       --stream --out -   writes raw s16le PCM chunks to stdout\n"
        "       [--encoder encoder.gguf --ref-audio ref.wav]   voice cloning\n"
        "       [--encoder encoder.gguf --dialogue-ref s1.wav --dialogue-ref s2.wav]\n"
        "           two-speaker dialogue: text carries [S1]/[S2] turns, references first\n"
        "       [--language zh] [--duration-tokens 0]   target length, 12.5 tokens per second\n"
        "       [--max-new-tokens 2048] [--context 4096]\n"
        "       [--stream] [--stream-chunk-frames 25]\n"
        "       [--seed 1234] [--threads 4] [--gpu] [--backends-dir dir]\n"
        "       [--text-temperature 1.5] [--text-top-p 1.0] [--text-top-k 50]\n"
        "       [--audio-temperature 1.7] [--audio-top-p 0.8] [--audio-top-k 25]\n"
        "       [--audio-repetition-penalty 1.0]\n"
        "       moss-cli --mode sfx --model sfx.gguf --text \"rain on a tin roof\" --out out.wav\n"
        "       [--seconds 10] [--negative-prompt \"...\"] [--seed 0] [--threads 4] [--gpu] [--backends-dir dir]\n"
        "       [--steps N] [--guidance G] [--shift S]   defaults come from the model file\n");
}

namespace {

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

bool parse_mode(const std::string & value, Mode & mode) {
    if (value == "tts") {
        mode = Mode::Speech;
        return true;
    }
    if (value == "sfx") {
        mode = Mode::SoundEffect;
        return true;
    }
    std::fprintf(stderr, "unknown mode: %s (expected tts or sfx)\n", value.c_str());
    return false;
}

void share_runtime_flags(CliArgs & args) {
    args.sound_options.n_threads = args.options.n_threads;
    args.sound_options.use_gpu = args.options.use_gpu;
    args.sound_options.backends_dir = args.options.backends_dir;
    args.sound_request.prompt = args.text;
}

bool has_required_flags(const CliArgs & args) {
    if (args.text.empty()) {
        return false;
    }
    if (args.mode == Mode::SoundEffect && args.stream) {
        std::fprintf(stderr, "--stream is not available in sfx mode\n");
        return false;
    }
    if (args.mode == Mode::SoundEffect) {
        return !args.sound_options.model_path.empty();
    }
    return !args.options.backbone_path.empty() && !args.options.decoder_path.empty();
}

int run_sound_effect(const CliArgs & args) {
    tts_cpp::moss::SoundEffectEngine engine(args.sound_options);
    std::fprintf(stderr, "[moss-cli] backend: %s\n", engine.backend_name());
    const tts_cpp::moss::SoundEffectResult result = engine.generate(args.sound_request,
            [](int step, int total) {
                std::fprintf(stderr, "\r[moss-cli] step %d/%d", step, total);
                if (step == total) {
                    std::fprintf(stderr, "\n");
                }
                return true;
            });
    if (result.cancelled) {
        std::fprintf(stderr, "[moss-cli] generation cancelled\n");
        return 1;
    }
    if (!save_wav(args.out_path, result.pcm, result.sample_rate)) {
        std::fprintf(stderr, "[moss-cli] cannot write %s\n", args.out_path.c_str());
        return 1;
    }
    std::fprintf(stderr,
        "[moss-cli] wrote %s: %.2fs @ %d Hz (text %.0f ms, diffusion %.0f ms, decode %.0f ms)\n",
        args.out_path.c_str(), (double) result.pcm.size() / result.sample_rate, result.sample_rate,
        result.text_ms, result.diffusion_ms, result.decode_ms);
    return 0;
}

} // namespace

bool parse_args(int argc, const char * const * argv, CliArgs & args) {
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + flag);
            }
            return argv[++i];
        };
        if (flag == "--mode") {
            if (!parse_mode(next(), args.mode)) {
                return false;
            }
        }
        else if (flag == "--backbone")      args.options.backbone_path = next();
        else if (flag == "--decoder")       args.options.decoder_path = next();
        else if (flag == "--encoder")       args.options.encoder_path = next();
        else if (flag == "--ref-audio")     args.options.reference_audio_path = next();
        else if (flag == "--dialogue-ref")  args.options.dialogue_reference_paths.push_back(next());
        else if (flag == "--language")      args.options.language = next();
        else if (flag == "--duration-tokens") args.options.duration_tokens = std::atoi(next());
        else if (flag == "--text")          args.text = next();
        else if (flag == "--out")           args.out_path = next();
        else if (flag == "--max-new-tokens") args.options.max_new_tokens = std::atoi(next());
        else if (flag == "--context")       args.options.context = std::atoi(next());
        else if (flag == "--seed") {
            args.options.seed = (uint32_t) std::strtoul(next(), nullptr, 10);
            args.sound_request.seed = args.options.seed;
        }
        else if (flag == "--model")         args.sound_options.model_path = next();
        else if (flag == "--negative-prompt") args.sound_request.negative_prompt = next();
        else if (flag == "--seconds")       args.sound_request.seconds = std::strtod(next(), nullptr);
        else if (flag == "--steps")         args.sound_request.steps = std::atoi(next());
        else if (flag == "--guidance")      args.sound_request.guidance = std::strtof(next(), nullptr);
        else if (flag == "--shift")         args.sound_request.shift = std::strtof(next(), nullptr);
        else if (flag == "--threads")       args.options.n_threads = std::atoi(next());
        else if (flag == "--gpu")           args.options.use_gpu = true;
        else if (flag == "--stream")        args.stream = true;
        else if (flag == "--stream-chunk-frames")   args.options.stream_chunk_frames = std::atoi(next());
        else if (flag == "--backends-dir")  args.options.backends_dir = next();
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
    share_runtime_flags(args);
    return has_required_flags(args);
}

int run(const CliArgs & args) {
    try {
        if (args.mode == Mode::SoundEffect) {
            return run_sound_effect(args);
        }
        tts_cpp::moss::Engine engine(args.options);
        std::fprintf(stderr, "[moss-cli] backend: %s\n", engine.backend_name());
        std::vector<float> pcm;
        tts_cpp::moss::SynthesisResult result;
        if (args.stream) {
            const bool to_stdout = args.out_path == "-";
            size_t chunks = 0;
            result = engine.synthesize_stream(args.text,
                    [&](const float * samples, size_t count, int rate) {
                        if (to_stdout) {
                            std::vector<drwav_int16> s16(count);
                            drwav_f32_to_s16(s16.data(), samples, count);
                            std::fwrite(s16.data(), sizeof(drwav_int16), count, stdout);
                            std::fflush(stdout);
                        } else {
                            pcm.insert(pcm.end(), samples, samples + count);
                        }
                        chunks++;
                        std::fprintf(stderr, "[moss-cli] chunk %zu: %zu samples @ %d Hz\n",
                                chunks, count, rate);
                        return true;
                    });
            std::fprintf(stderr, "[moss-cli] first audio after %.0f ms in %zu chunks\n",
                    result.first_audio_ms, chunks);
            if (to_stdout) {
                return result.cancelled ? 1 : 0;
            }
        } else {
            result = engine.synthesize(args.text);
            pcm = result.pcm;
        }
        if (result.cancelled) {
            std::fprintf(stderr, "[moss-cli] synthesis cancelled\n");
            return 1;
        }
        if (!save_wav(args.out_path, pcm, result.sample_rate)) {
            std::fprintf(stderr, "[moss-cli] cannot write %s\n", args.out_path.c_str());
            return 1;
        }
        std::fprintf(stderr,
            "[moss-cli] wrote %s: %.2fs @ %d Hz (%d frames, generate %.0f ms, decode %.0f ms)\n",
            args.out_path.c_str(), (double) pcm.size() / result.sample_rate,
            result.sample_rate, result.generated_frames, result.generation_ms, result.decode_ms);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "[moss-cli] error: %s\n", error.what());
        return 1;
    }
}

} // namespace tts_cpp::moss::cli
