// Codec reconstruction only: no text LM, generation, gain normalization or lag search.
#include "audio8/codec_bench_cli.h"
#include "audio8/internal.h"
#include "voice_features.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <vector>

namespace codec = tts_cpp::audio8::detail;
namespace {
void require(bool ok, const std::string & message) {
    if (!ok) throw std::runtime_error(message);
}
struct owned_codec {
    codec::codec_model model;
    ~owned_codec() { codec::free_codec(model); }
};

void check_headers(const codec::codec_header & enc, const codec::codec_header & dec) {
    require(enc.part == "encoder" && dec.part == "decoder", "wrong codec half supplied");
    const auto & a = enc.hp;
    const auto & b = dec.hp;
    require(a.num_codebooks == a.residual_codebooks + 1 &&
            b.num_codebooks == b.residual_codebooks + 1, "invalid codec codebook count");
    require(a.sample_rate == b.sample_rate && a.frame_size == b.frame_size &&
            a.num_codebooks == b.num_codebooks && a.codebook_dim == b.codebook_dim &&
            a.latent_dim == b.latent_dim && a.semantic_codebook_size == b.semantic_codebook_size &&
            a.residual_codebook_size == b.residual_codebook_size &&
            a.residual_codebooks == b.residual_codebooks, "incompatible codec encoder and decoder");
    require(a.sample_rate > 0 && a.frame_size > 0, "invalid codec rate or frame size");
}

void write_wav(const std::string & path, const std::vector<float> & pcm, int rate) {
    require(pcm.size() <= (UINT32_MAX - 36) / 2, "output exceeds WAV size limit");
    std::vector<unsigned char> bytes;
    bytes.reserve(44 + pcm.size() * 2);
    auto tag = [&](const char * s) { bytes.insert(bytes.end(), s, s + 4); };
    auto le = [&](uint32_t n, int width) {
        for (int i = 0; i < width; ++i) bytes.push_back(static_cast<unsigned char>(n >> (8 * i)));
    };
    tag("RIFF"); le(36 + static_cast<uint32_t>(pcm.size() * 2), 4); tag("WAVE");
    tag("fmt "); le(16, 4); le(1, 2); le(1, 2); le(rate, 4); le(rate * 2, 4);
    le(2, 2); le(16, 2); tag("data"); le(static_cast<uint32_t>(pcm.size() * 2), 4);
    for (float sample : pcm) {
        require(std::isfinite(sample), "codec produced nonfinite audio");
        const auto value = static_cast<int16_t>(std::lrintf(std::max(-1.0f, std::min(1.0f, sample)) * 32767.0f));
        le(static_cast<uint16_t>(value), 2);
    }
    std::FILE * f = std::fopen(path.c_str(), "wb");
    require(f != nullptr, "cannot open output WAV: " + path);
    const bool written = std::fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size();
    const bool closed = std::fclose(f) == 0;
    require(written && closed, "failed writing output WAV: " + path);
}
}

int main(int argc, char ** argv) {
    try {
        const auto opts = audio8_codec_bench::parse(argc, argv);
        if (opts.help) {
            std::puts("audio8-codec-bench --codec-encoder FILE --codec-decoder FILE --in WAV --out WAV [--threads N] [--n-gpu-layers N]\n"
                      "Writes mono PCM16 at the input rate and duration; removes only codec end padding.\n"
                      "Input channels are averaged to mono; sinc resampling and PCM16 clipping are included.");
            return 0;
        }
        for (const auto & input : {opts.input, opts.encoder, opts.decoder}) {
            std::error_code ec;
            require(!std::filesystem::equivalent(input, opts.output, ec), "output aliases an input or model file");
        }
        std::vector<float> pcm;
        int rate = 0;
        require(wav_load(opts.input, pcm, rate) && !pcm.empty(), "cannot read nonempty input WAV");
        require(rate >= 8000 && rate <= 192000, "input rate must be in [8000, 192000]");
        for (float v : pcm) require(std::isfinite(v), "input WAV contains nonfinite samples");
        const size_t input_samples = pcm.size();
        codec::codec_header enc_header, dec_header;
        std::string error;
        require(codec::peek_codec_header(opts.encoder, enc_header, &error), error);
        require(codec::peek_codec_header(opts.decoder, dec_header, &error), error);
        check_headers(enc_header, dec_header);
        const int native_rate = enc_header.hp.sample_rate;
        // resample_sinc rounds down. Retain the full input duration by adding
        // at most one zero sample at the end before the codec's frame padding.
        const size_t native_samples = static_cast<size_t>(std::ceil(
            static_cast<double>(input_samples) * native_rate / rate));
        require(native_samples <= static_cast<size_t>(INT_MAX - enc_header.hp.frame_size), "input too long for codec");
        pcm = resample_sinc(pcm, rate, native_rate);
        const size_t resample_padding = native_samples - pcm.size();
        require(resample_padding <= 1, "unexpected input resampler length");
        pcm.resize(native_samples, 0.0f);
        owned_codec encoder, decoder;
        require(codec::load_codec(opts.encoder, opts.gpu_layers, encoder.model, &error), error);
        require(codec::load_codec(opts.decoder, opts.gpu_layers, decoder.model, &error), error);
        std::printf("encoder backend: %s\ndecoder backend: %s\n", ggml_backend_name(encoder.model.backend), ggml_backend_name(decoder.model.backend));
        std::vector<int32_t> codes;
        int frames = 0;
        require(codec::encode_audio(encoder.model, pcm.data(), static_cast<int>(pcm.size()),
                                   opts.threads, {}, codes, frames, &error), error);
        std::vector<float> decoded;
        require(codec::decode_codes(decoder.model, codes.data(), frames, opts.threads, {}, decoded, &error), error);
        const size_t expected = static_cast<size_t>(frames) * dec_header.hp.frame_size;
        require(decoded.size() == expected && expected >= native_samples &&
                expected - native_samples < static_cast<size_t>(dec_header.hp.frame_size), "unexpected codec output length");
        // Keep the known zero-padded tail through resampling to avoid inserting
        // a new filter boundary into retained speech, then remove only that tail.
        decoded = resample_sinc(decoded, native_rate, rate);
        require(decoded.size() >= input_samples, "resampled output is shorter than input");
        decoded.resize(input_samples);
        write_wav(opts.output, decoded, rate);
        std::printf("codec roundtrip: input_samples=%zu sample_rate=%d native_samples=%zu native_rate=%d frames=%d codec_end_padding=%zu resample_end_padding=%zu output_samples=%zu\n",
                    input_samples, rate, native_samples, native_rate, frames, expected - native_samples, resample_padding, decoded.size());
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "audio8-codec-bench: %s\n", e.what());
        return 1;
    }
}
