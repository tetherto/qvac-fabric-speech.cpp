#include "moss/cli.h"

#include "dr_wav.h"
#include "json.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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
        "       [--steps N] [--guidance G] [--shift S]   defaults come from the model file\n"
        "       moss-cli --mode transcribe --model transcribe.gguf --audio speech.wav [--out transcript.json]\n"
        "       [--prompt \"...\"] [--max-new-tokens N] [--threads 4] [--gpu] [--backends-dir dir]\n"
        "           16 kHz WAV in; prints one [start-end] Sxx: text line per segment\n");
}

namespace {

constexpr const char * DEFAULT_WAV_OUT = "moss-out.wav";

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
    if (value == "transcribe") {
        mode = Mode::Transcribe;
        return true;
    }
    std::fprintf(stderr, "unknown mode: %s (expected tts, sfx or transcribe)\n", value.c_str());
    return false;
}

void share_runtime_flags(CliArgs & args) {
    args.sound_options.n_threads = args.options.n_threads;
    args.sound_options.use_gpu = args.options.use_gpu;
    args.sound_options.backends_dir = args.options.backends_dir;
    args.sound_request.prompt = args.text;
    args.transcribe_options.model_path = args.sound_options.model_path;
    args.transcribe_options.n_threads = args.options.n_threads;
    args.transcribe_options.use_gpu = args.options.use_gpu;
    args.transcribe_options.backends_dir = args.options.backends_dir;
    args.transcribe_request.max_new_tokens = args.mode == Mode::Transcribe ? args.options.max_new_tokens : 0;
    if (args.out_path.empty() && args.mode != Mode::Transcribe) {
        args.out_path = DEFAULT_WAV_OUT;
    }
}

bool has_transcribe_flags(const CliArgs & args) {
    if (args.stream) {
        std::fprintf(stderr, "--stream is not available in transcribe mode\n");
        return false;
    }
    return !args.transcribe_options.model_path.empty() && !args.audio_path.empty();
}

bool has_required_flags(const CliArgs & args) {
    if (args.mode == Mode::Transcribe) {
        return has_transcribe_flags(args);
    }
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

std::vector<float> mono_mix(const std::vector<float> & interleaved, unsigned channels) {
    std::vector<float> mono(interleaved.size() / channels, 0.0f);
    for (size_t i = 0; i < interleaved.size(); ++i) {
        mono[i / channels] += interleaved[i] / (float) channels;
    }
    return mono;
}

std::vector<float> load_wav(const std::string & path, int & sample_rate) {
    unsigned channels = 0;
    unsigned rate = 0;
    drwav_uint64 frames = 0;
    float * data = drwav_open_file_and_read_pcm_frames_f32(path.c_str(), &channels, &rate, &frames, nullptr);
    if (data == nullptr || channels == 0) {
        throw std::runtime_error("cannot read WAV: " + path);
    }
    const std::vector<float> interleaved(data, data + frames * channels);
    drwav_free(data, nullptr);
    sample_rate = (int) rate;
    return channels == 1 ? interleaved : mono_mix(interleaved, channels);
}

void print_segment_lines(const std::vector<tts_cpp::moss::TranscriptSegment> & segments) {
    for (const auto & segment : segments) {
        std::printf("[%.2f-%.2f] %s: %s\n", segment.start_s, segment.end_s, segment.speaker.c_str(),
                segment.text.c_str());
    }
}

void print_segments(const tts_cpp::moss::TranscribeResult & result) {
    print_segment_lines(result.segments);
    if (result.segments.empty()) {
        std::printf("%s\n", result.text.c_str());
    }
}

bool save_transcript(const std::string & path, const tts_cpp::moss::TranscribeResult & result) {
    std::ofstream output(path, std::ios::binary);
    output << transcript_json(result) << "\n";
    return (bool) output;
}

int run_transcribe(const CliArgs & args) {
    int sample_rate = 0;
    const std::vector<float> pcm = load_wav(args.audio_path, sample_rate);
    tts_cpp::moss::TranscribeEngine engine(args.transcribe_options);
    std::fprintf(stderr, "[moss-cli] backend: %s\n", engine.backend_name());
    const tts_cpp::moss::TranscribeResult result = engine.transcribe(pcm.data(), pcm.size(), sample_rate,
            args.transcribe_request);
    if (result.cancelled) {
        std::fprintf(stderr, "[moss-cli] transcription cancelled\n");
        return 1;
    }
    print_segments(result);
    if (!args.out_path.empty() && !save_transcript(args.out_path, result)) {
        std::fprintf(stderr, "[moss-cli] cannot write %s\n", args.out_path.c_str());
        return 1;
    }
    std::fprintf(stderr,
        "[moss-cli] %.1fs audio, %d audio tokens, %d generated (encode %.0f ms, prefill %.0f ms, decode %.0f ms)\n",
        (double) pcm.size() / sample_rate, result.audio_tokens, result.generated_tokens,
        result.encode_ms, result.prefill_ms, result.decode_ms);
    return 0;
}

} // namespace

namespace {

nlohmann::ordered_json segments_json(const std::vector<TranscriptSegment> & segments) {
    nlohmann::ordered_json values = nlohmann::ordered_json::array();
    for (const auto & segment : segments) {
        values.push_back({{"start", segment.start_s}, {"end", segment.end_s},
                          {"speaker", segment.speaker}, {"text", segment.text}});
    }
    return values;
}

} // namespace

std::string transcript_json(const TranscribeResult & result) {
    nlohmann::ordered_json document = {
        {"text", result.text},
        {"segments", segments_json(result.segments)},
        {"audio_tokens", result.audio_tokens},
        {"prompt_tokens", result.prompt_tokens},
        {"generated_tokens", result.generated_tokens},
        {"encode_ms", result.encode_ms},
        {"prefill_ms", result.prefill_ms},
        {"decode_ms", result.decode_ms},
    };
    return document.dump(2);
}

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
        else if (flag == "--audio")         args.audio_path = next();
        else if (flag == "--prompt")        args.transcribe_request.prompt = next();
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
        if (args.mode == Mode::Transcribe) {
            return run_transcribe(args);
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
