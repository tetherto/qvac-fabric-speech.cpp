#include "transcribe_fixtures.h"

#include "moss/cli.h"
#include "moss/transcribe_audio.h"
#include "moss/transcribe_model.h"
#include "moss/transcribe_networks.h"
#include "moss/transcribe_text.h"
#include "tts-cpp/moss/transcribe.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

using namespace moss_transcribe_fixtures;
using namespace tts_cpp::moss::detail;

namespace {

constexpr uint32_t WEIGHT_SEED = 20260925;
constexpr uint32_t AUDIO_SEED = 7;
constexpr float FFT_TOLERANCE = 1e-3f;
constexpr float LOGIT_TOLERANCE = 1e-4f;
constexpr double PI = 3.14159265358979323846;
constexpr int WHISPER_SAMPLE_RATE = 16000;
constexpr int WHISPER_CHUNK = 480000;
constexpr int WHISPER_HOP = 160;
constexpr int WHISPER_MERGE = 4;
constexpr int REFERENCE_SAMPLES = 1980480;
constexpr int REFERENCE_AUDIO_TOKENS = 1548;
constexpr int REFERENCE_SPAN = 1600;
constexpr int CACHE_ALIGNMENT = 256;

int failures = 0;

void check(bool condition, const std::string & label) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", label.c_str());
        failures++;
    }
}

template <typename Fn>
void expect_failure(Fn && fn, const std::string & needle, const std::string & label) {
    try {
        fn();
        check(false, label + ": accepted");
    } catch (const std::runtime_error & e) {
        check(std::string(e.what()).find(needle) != std::string::npos, label + ": wrong failure: " + e.what());
    }
}

float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        return INFINITY;
    }
    float worst = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        worst = std::max(worst, std::fabs(a[i] - b[i]));
    }
    return worst;
}

std::vector<float> random_signal(size_t count, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> draw(-1.0f, 1.0f);
    std::vector<float> values(count);
    for (float & value : values) {
        value = draw(rng);
    }
    return values;
}

std::vector<std::complex<float>> naive_dft(const std::vector<float> & input) {
    const size_t n = input.size();
    std::vector<std::complex<float>> output(n);
    for (size_t k = 0; k < n; ++k) {
        std::complex<double> sum = 0.0;
        for (size_t j = 0; j < n; ++j) {
            sum += (double) input[j] * std::polar(1.0, -2.0 * PI * (double) (j * k) / (double) n);
        }
        output[k] = std::complex<float>((float) sum.real(), (float) sum.imag());
    }
    return output;
}

float spectrum_error(const std::vector<std::complex<float>> & a, const std::vector<std::complex<float>> & b) {
    float worst = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        worst = std::max(worst, std::abs(a[i] - b[i]));
    }
    return worst;
}

void test_fft_matches_dft() {
    for (int size : {12, 25, 400}) {
        const std::vector<float> signal = random_signal((size_t) size, AUDIO_SEED);
        check(spectrum_error(TranscribeFft(size).transform(signal), naive_dft(signal)) < FFT_TOLERANCE,
              "mixed-radix FFT matches the direct DFT for size " + std::to_string(size));
    }
}

void test_log_mel_normalization() {
    std::vector<float> mel = {-12.0f, -3.0f, 0.0f, 1.0f};
    normalize_log_mel(mel);
    check(std::fabs(mel[0] - ((1.0f - 8.0f + 4.0f) / 4.0f)) < 1e-6f, "values below max-8 clamp to the floor");
    check(std::fabs(mel[3] - (5.0f / 4.0f)) < 1e-6f, "the maximum maps to (max + 4) / 4");
}

TranscribeConfig whisper_like_config() {
    TranscribeConfig config;
    config.audio.sample_rate = WHISPER_SAMPLE_RATE;
    config.audio.hop_length = WHISPER_HOP;
    config.audio.chunk_samples = WHISPER_CHUNK;
    config.audio.chunk_frames = WHISPER_CHUNK / WHISPER_HOP;
    config.merge_size = WHISPER_MERGE;
    config.audio_tokens_per_second = 12.5f;
    config.time_marker_every_seconds = 5;
    config.time_markers = true;
    config.tokens.audio_pad = TOKEN_AUDIO_PAD;
    return config;
}

void test_chunk_arithmetic() {
    const TranscribeConfig config = whisper_like_config();
    check(transcribe_chunk_count(config.audio, 0) == 0, "no audio has no chunks");
    check(transcribe_chunk_count(config.audio, WHISPER_CHUNK) == 1, "a full window is one chunk");
    check(transcribe_chunk_count(config.audio, WHISPER_CHUNK + 1) == 2, "one extra sample opens a chunk");
    check(transcribe_chunk_tokens(config, WHISPER_CHUNK) == 375, "a 30 s chunk yields 375 tokens");
    check(transcribe_chunk_tokens(config, 1) == 1, "a single sample yields one token");
    check(transcribe_audio_tokens(config, REFERENCE_SAMPLES) == REFERENCE_AUDIO_TOKENS,
          "audio token count matches the Hugging Face processor");
}

TranscribeTokenizer fixture_tokenizer() {
    return TranscribeTokenizer(transcribe_vocab(), {"a b"}, transcribe_token_types());
}

size_t count_pads(const std::vector<int32_t> & span) {
    return (size_t) std::count(span.begin(), span.end(), TOKEN_AUDIO_PAD);
}

void test_audio_span_markers() {
    const TranscribeConfig config = whisper_like_config();
    const TranscribeTokenizer tokenizer = fixture_tokenizer();
    const std::vector<int32_t> span = transcribe_audio_span(config, tokenizer, REFERENCE_AUDIO_TOKENS);
    check(span.size() == REFERENCE_SPAN, "span length matches the Hugging Face processor");
    check(count_pads(span) == REFERENCE_AUDIO_TOKENS, "every audio token keeps its placeholder");
    check(span[62] == byte_token('5'), "the 5 s marker follows 62 placeholders");
    check(span[125] == byte_token('1') && span[126] == byte_token('0'), "the 10 s marker is spelled digit by digit");
    TranscribeConfig plain = config;
    plain.time_markers = false;
    check(transcribe_audio_span(plain, tokenizer, 10) == std::vector<int32_t>(10, TOKEN_AUDIO_PAD),
          "disabled markers leave only placeholders");
    check(transcribe_audio_span(config, tokenizer, 0).empty(), "no audio tokens give an empty span");
}

std::vector<int32_t> bytes_of(const std::string & text) {
    std::vector<int32_t> ids;
    for (unsigned char ch : text) {
        ids.push_back(byte_token(ch));
    }
    return ids;
}

std::vector<int32_t> expected_prompt(const std::vector<int32_t> & span, const std::vector<int32_t> & body) {
    std::vector<int32_t> ids{TOKEN_IM_START};
    for (const auto & part : {bytes_of("system\nsys"), std::vector<int32_t>{TOKEN_IM_END}, bytes_of("\n"),
                              std::vector<int32_t>{TOKEN_IM_START}, bytes_of("user\n"),
                              std::vector<int32_t>{TOKEN_AUDIO_START}, span,
                              std::vector<int32_t>{TOKEN_AUDIO_END}, body, std::vector<int32_t>{TOKEN_IM_END},
                              bytes_of("\n"), std::vector<int32_t>{TOKEN_IM_START}, bytes_of("assistant\n")}) {
        ids.insert(ids.end(), part.begin(), part.end());
    }
    return ids;
}

void test_prompt_layout() {
    TranscribeConfig config = whisper_like_config();
    config.tokens = {TOKEN_AUDIO_START, TOKEN_AUDIO_END, TOKEN_AUDIO_PAD, TOKEN_IM_START, TOKEN_IM_END, 0};
    config.system_prompt = SYSTEM_PROMPT;
    config.default_prompt_ids = {byte_token('h'), byte_token('i')};
    config.time_markers = false;
    const TranscribeTokenizer tokenizer = fixture_tokenizer();
    const std::vector<int32_t> span(3, TOKEN_AUDIO_PAD);
    check(transcribe_prompt(config, tokenizer, 3, "") == expected_prompt(span, bytes_of("\nhi")),
          "empty prompt uses the stored default prompt ids");
    check(transcribe_prompt(config, tokenizer, 3, "ok") == expected_prompt(span, bytes_of("\nok")),
          "a custom prompt replaces the default");
}

void test_detokenizer() {
    const TranscribeTokenizer tokenizer = fixture_tokenizer();
    std::vector<int32_t> ids = bytes_of("h\xc3\xa9");
    ids.insert(ids.begin(), TOKEN_IM_START);
    ids.push_back(TOKEN_USER_DEFINED);
    ids.push_back(TOKEN_IM_END);
    check(tokenizer.decode(ids) == "h\xc3\xa9<think>", "decode restores bytes, keeps user tokens, skips control");
    check(tokenizer.decode({-1, 100000}) == "", "out-of-range ids decode to nothing");
    check(tokenizer.encode("h\xc3\xa9") == bytes_of("h\xc3\xa9"), "byte-level encode round-trips");
}

void test_parser_basic() {
    const auto segments = parse_transcript("[0.50][S01] hello there [1.25][1.25][S02]world[2]");
    check(segments.size() == 2, "two segments parse");
    check(segments.size() == 2 && segments[0].start_s == 0.5 && segments[0].end_s == 1.25 &&
          segments[0].speaker == "S01" && segments[0].text == "hello there", "first segment fields");
    check(segments.size() == 2 && segments[1].speaker == "S02" && segments[1].end_s == 2.0,
          "the last segment is emitted on close");
}

void test_parser_edge_cases() {
    check(parse_transcript("[1.0][X1]nope[2.0]").empty(), "an invalid speaker drops the segment");
    check(parse_transcript("[1.0][S1]   [2.0]").empty(), "empty text is skipped");
    const auto backwards = parse_transcript("[2.0][S1]a[1.0]b[3.0]");
    check(backwards.size() == 1 && backwards[0].text == "a[1.0]b", "an end before the start stays in the text");
    const auto spaced = parse_transcript("[0][S1]a[1] b[2]");
    check(spaced.size() == 1 && spaced[0].text == "a[1] b" && spaced[0].end_s == 2.0,
          "text after an end timestamp reopens the segment");
    check(parse_transcript("[0][S1]dangling").empty(), "an unterminated segment is not emitted");
    check(parse_transcript("noise [0][S1]ok[1]").size() == 1, "leading noise is ignored");
    check(parse_transcript("[1.2.3][S1]x[4]").empty(), "a timestamp with two dots is rejected");
}

void test_parser_streaming() {
    const std::string text = "[0.50][S01] hello [1.25][1.25][S02]world[2.00]";
    TranscriptParser parser;
    std::vector<tts_cpp::moss::TranscriptSegment> streamed;
    for (char ch : text) {
        const auto emitted = parser.feed(std::string(1, ch));
        streamed.insert(streamed.end(), emitted.begin(), emitted.end());
    }
    const auto tail = parser.close();
    streamed.insert(streamed.end(), tail.begin(), tail.end());
    check(streamed.size() == parse_transcript(text).size(), "character streaming matches a whole parse");
}

void test_model_loads() {
    const auto path = write_transcribe_model("transcribe-load", WEIGHT_SEED);
    TranscribeModel model(path.string(), false, 1);
    const TranscribeConfig & config = model.config();
    check(config.text.vocab == moss_fixtures::TEXT_VOCAB, "vocabulary size comes from the embedding table");
    check(config.samples_per_token() == HOP * 2 * MERGE, "samples per token follow hop, stride and merge");
    check(config.default_prompt_ids.size() == 2, "default prompt ids load");
    validate_transcribe_encoder(model);
    validate_transcribe_decoder(model);
    std::filesystem::remove(path);
}

void expect_model_rejects(const std::filesystem::path & path, const std::string & needle, const std::string & label) {
    expect_failure([&] { TranscribeModel model(path.string(), false, 1); }, needle, label);
    std::filesystem::remove(path);
}

void test_model_rejections() {
    expect_model_rejects(write_transcribe_model("transcribe-arch", WEIGHT_SEED, [](gguf_context * f) {
        gguf_set_val_str(f, "general.architecture", "moss-sfx");
    }), "unsupported architecture", "wrong architecture");
    expect_model_rejects(write_transcribe_model("transcribe-window", WEIGHT_SEED, [](gguf_context * f) {
        gguf_set_val_u32(f, "moss-transcribe.encoder.context_length", ENC_CTX + 1);
    }), "encoder geometry", "encoder window must be half the mel frames");
    expect_model_rejects(write_transcribe_model("transcribe-frames", WEIGHT_SEED, [](gguf_context * f) {
        gguf_set_val_u32(f, "moss-transcribe.audio.chunk_frames", CHUNK_FRAMES + 1);
    }), "audio front-end", "chunk frames must match samples / hop");
    expect_model_rejects(write_transcribe_model("transcribe-gqa", WEIGHT_SEED, [](gguf_context * f) {
        gguf_set_val_u32(f, "moss-transcribe.text.attention.head_count_kv", 3);
    }), "decoder geometry", "query heads must be a multiple of key heads");
    expect_model_rejects(write_transcribe_model("transcribe-merge", WEIGHT_SEED, [](gguf_context * f) {
        gguf_set_val_u32(f, "moss-transcribe.adaptor.merge_size", 3);
    }), "prompt metadata", "merge size must divide the encoder window");
    expect_model_rejects(write_transcribe_model("transcribe-prompt", WEIGHT_SEED, [](gguf_context * f) {
        gguf_remove_key(f, "moss-transcribe.default_prompt_ids");
    }), "default_prompt_ids", "default prompt ids are required");
    const auto healthy = write_transcribe_model("transcribe-threads", WEIGHT_SEED);
    expect_failure([&] { TranscribeModel model(healthy.string(), false, 0); }, "threads", "zero threads rejected");
    std::filesystem::remove(healthy);
}

void test_network_rejections() {
    const auto shapes = moss_fixtures::TensorShapes{{"adaptor.fc1.weight", {ENC_EMBD, TEXT_EMBD}}};
    const auto path = write_transcribe_model("transcribe-adaptor", WEIGHT_SEED, [](gguf_context *) {}, shapes);
    TranscribeModel model(path.string(), false, 1);
    expect_failure([&] { validate_transcribe_encoder(model); }, "adaptor.fc1.weight", "adaptor width is validated");
    std::filesystem::remove(path);
}

std::vector<float> tiny_mel(const TranscribeModel & model, const std::vector<float> & audio) {
    const TranscribeMel mel(model.config().audio, read_mel_filters(model));
    return mel.chunk(audio.data(), audio.size());
}

void test_encoder_chunk() {
    const auto path = write_transcribe_model("transcribe-encoder", WEIGHT_SEED);
    TranscribeModel model(path.string(), false, 1);
    const std::vector<float> mel = tiny_mel(model, random_signal(CHUNK_SAMPLES, AUDIO_SEED));
    check(mel.size() == (size_t) N_MELS * CHUNK_FRAMES, "mel chunk has n_mels x frames values");
    const TranscribeChunkEncoding full = encode_audio_chunk(model, mel, CHUNK_TOKENS, true);
    check(full.embeddings.size() == (size_t) CHUNK_TOKENS * TEXT_EMBD, "a full chunk yields one row per token");
    check(full.encoder_states.size() == (size_t) ENC_CTX * ENC_EMBD, "encoder states cover the window");
    const TranscribeChunkEncoding partial = encode_audio_chunk(model, mel, 2, false);
    check(partial.encoder_states.empty(), "encoder states are dropped unless requested");
    const std::vector<float> head(full.embeddings.begin(), full.embeddings.begin() + 2 * TEXT_EMBD);
    check(max_abs_diff(partial.embeddings, head) < LOGIT_TOLERANCE, "a short chunk keeps the leading tokens");
    expect_failure([&] { encode_audio_chunk(model, mel, CHUNK_TOKENS + 1, false); }, "encoder window",
                   "too many tokens rejected");
    expect_failure([&] { encode_audio_chunk(model, {1.0f}, 1, false); }, "wrong size", "bad mel size rejected");
    std::filesystem::remove(path);
}

std::vector<int32_t> tiny_prompt(int audio_tokens) {
    std::vector<int32_t> ids{TOKEN_IM_START, byte_token('a'), TOKEN_AUDIO_START};
    ids.insert(ids.end(), (size_t) audio_tokens, TOKEN_AUDIO_PAD);
    ids.push_back(TOKEN_AUDIO_END);
    ids.push_back(byte_token('b'));
    return ids;
}

std::vector<float> prefill_logits(TranscribeModel & model, const std::vector<int32_t> & prompt,
                                  const std::vector<float> & audio, int batch) {
    TranscribeDecoder decoder(model, (int) prompt.size() + 1);
    return decoder.prefill(prompt, audio, batch);
}

void test_decoder_batches() {
    const auto path = write_transcribe_model("transcribe-decoder", WEIGHT_SEED);
    TranscribeModel model(path.string(), false, 1);
    const std::vector<int32_t> prompt = tiny_prompt(5);
    const std::vector<float> audio = random_signal((size_t) 5 * TEXT_EMBD, AUDIO_SEED);
    const std::vector<float> whole = prefill_logits(model, prompt, audio, (int) prompt.size());
    check(whole.size() == (size_t) moss_fixtures::TEXT_VOCAB, "prefill returns one logit per vocabulary entry");
    check(max_abs_diff(prefill_logits(model, prompt, audio, 1), whole) < LOGIT_TOLERANCE,
          "token-by-token prefill matches a single batch");
    check(max_abs_diff(prefill_logits(model, prompt, audio, 3), whole) < LOGIT_TOLERANCE,
          "uneven batches match a single batch");
    const std::vector<float> silent(audio.size(), 0.0f);
    check(max_abs_diff(prefill_logits(model, prompt, silent, 3), whole) > LOGIT_TOLERANCE,
          "audio embeddings replace the placeholder rows");
    TranscribeDecoder decoder(model, 3);
    check(decoder.context() % CACHE_ALIGNMENT == 0 && decoder.context() >= 3, "context is padded for alignment");
    expect_failure([&] { prefill_logits(model, prompt, std::vector<float>(TEXT_EMBD), 4); }, "audio",
                   "missing audio rows rejected");
    expect_failure([&] { prefill_logits(model, prompt, std::vector<float>(7), 4); }, "decoder width",
                   "ragged audio rows rejected");
    expect_failure([&] { TranscribeDecoder oversized(model, TEXT_CTX + 1); }, "context", "context past training");
    std::filesystem::remove(path);
}

tts_cpp::moss::TranscribeOptions tiny_options(const std::filesystem::path & path) {
    tts_cpp::moss::TranscribeOptions options;
    options.model_path = path.string();
    options.n_threads = 1;
    return options;
}

void test_engine_end_to_end() {
    const auto path = write_transcribe_model("transcribe-engine", WEIGHT_SEED);
    tts_cpp::moss::TranscribeEngine engine(tiny_options(path));
    const std::vector<float> audio = random_signal(CHUNK_SAMPLES * 2 + 40, AUDIO_SEED);
    const auto result = engine.transcribe(audio.data(), audio.size(), SAMPLE_RATE);
    check(!result.cancelled, "a tiny transcription completes");
    check(result.audio_tokens == 2 * CHUNK_TOKENS + 2, "audio tokens cover full and partial chunks");
    check(result.generated_tokens <= DEFAULT_MAX_NEW_TOKENS, "the model default caps generation");
    check(result.prompt_tokens > result.audio_tokens, "the prompt wraps the audio span");
    tts_cpp::moss::TranscribeRequest request;
    request.max_new_tokens = 2;
    check(engine.transcribe(audio.data(), audio.size(), SAMPLE_RATE, request).generated_tokens <= 2,
          "max_new_tokens overrides the default");
    check(engine.sample_rate() == SAMPLE_RATE, "sample rate comes from the model");
    std::filesystem::remove(path);
}

void test_engine_rejections() {
    const auto path = write_transcribe_model("transcribe-reject", WEIGHT_SEED);
    tts_cpp::moss::TranscribeEngine engine(tiny_options(path));
    const std::vector<float> audio = random_signal(CHUNK_SAMPLES, AUDIO_SEED);
    expect_failure([&] { engine.transcribe(audio.data(), audio.size(), SAMPLE_RATE * 2); }, "sampled at",
                   "wrong sample rate rejected");
    expect_failure([&] { engine.transcribe(audio.data(), 0, SAMPLE_RATE); }, "empty", "empty audio rejected");
    tts_cpp::moss::TranscribeRequest negative;
    negative.max_new_tokens = -1;
    expect_failure([&] { engine.transcribe(audio.data(), audio.size(), SAMPLE_RATE, negative); }, "max_new_tokens",
                   "negative max_new_tokens rejected");
    tts_cpp::moss::TranscribeRequest long_prompt;
    long_prompt.prompt = std::string(10000, 'x');
    expect_failure([&] { engine.transcribe(audio.data(), audio.size(), SAMPLE_RATE, long_prompt); }, "prompt",
                   "oversized prompt rejected");
    expect_failure([] { tts_cpp::moss::TranscribeEngine missing(tts_cpp::moss::TranscribeOptions{}); }, "model_path",
                   "model path required");
    std::filesystem::remove(path);
}

void test_engine_cancel() {
    const auto path = write_transcribe_model("transcribe-cancel", WEIGHT_SEED, [](gguf_context * f) {
        gguf_set_val_u32(f, "moss-transcribe.token.im_end", moss_fixtures::TEXT_VOCAB - 1);
    });
    tts_cpp::moss::TranscribeEngine engine(tiny_options(path));
    const std::vector<float> audio = random_signal(CHUNK_SAMPLES, AUDIO_SEED);
    int calls = 0;
    const auto result = engine.transcribe(audio.data(), audio.size(), SAMPLE_RATE, {},
            [&](int, int) { return ++calls < 2; });
    check(result.cancelled && result.generated_tokens == 2, "the progress callback stops generation");
    std::filesystem::remove(path);
}

void test_cli_flags() {
    using tts_cpp::moss::cli::CliArgs;
    using tts_cpp::moss::cli::parse_args;
    const char * full[] = {"moss-cli", "--mode", "transcribe", "--model", "t.gguf", "--audio", "a.wav",
                           "--prompt", "p", "--max-new-tokens", "9", "--gpu"};
    CliArgs args;
    check(parse_args(12, full, args), "transcribe flags parse");
    check(args.transcribe_options.model_path == "t.gguf" && args.audio_path == "a.wav" &&
          args.transcribe_request.prompt == "p" && args.transcribe_request.max_new_tokens == 9 &&
          args.transcribe_options.use_gpu && args.out_path.empty(), "transcribe values land in the request");
    const char * no_audio[] = {"moss-cli", "--mode", "transcribe", "--model", "t.gguf"};
    CliArgs missing;
    check(!parse_args(5, no_audio, missing), "--audio is required");
    const char * streaming[] = {"moss-cli", "--mode", "transcribe", "--model", "t.gguf", "--audio", "a.wav",
                                "--stream"};
    CliArgs stream;
    check(!parse_args(8, streaming, stream), "--stream is rejected in transcribe mode");
}

void test_transcript_json() {
    tts_cpp::moss::TranscribeResult result;
    result.text = "[0][S01]hi[1]";
    result.segments = {{0.0, 1.0, "S01", "hi"}};
    const std::string json = tts_cpp::moss::cli::transcript_json(result);
    check(json.find("\"speaker\": \"S01\"") != std::string::npos && json.find("\"text\": \"hi\"") != std::string::npos,
          "transcript JSON carries the segments");
}

} // namespace

int main() {
    try {
        test_fft_matches_dft();
        test_log_mel_normalization();
        test_chunk_arithmetic();
        test_audio_span_markers();
        test_prompt_layout();
        test_detokenizer();
        test_parser_basic();
        test_parser_edge_cases();
        test_parser_streaming();
        test_model_loads();
        test_model_rejections();
        test_network_rejections();
        test_encoder_chunk();
        test_decoder_batches();
        test_engine_end_to_end();
        test_engine_rejections();
        test_engine_cancel();
        test_cli_flags();
        test_transcript_json();
    } catch (const std::exception & e) {
        std::fprintf(stderr, "unexpected: %s\n", e.what());
        return 1;
    }
    if (failures == 0) {
        std::printf("moss transcribe: OK\n");
        return 0;
    }
    std::fprintf(stderr, "moss transcribe: %d failures\n", failures);
    return 1;
}
