#include "parakeet/engine.h"

#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s model.gguf english.wav french.wav\n", argv[0]);
        return 2;
    }
    try {
        parakeet::EngineOptions options;
        options.model_gguf_path = argv[1];
        options.n_threads = 4;
        parakeet::Engine engine(options);
        for (int i = 2; i < argc; ++i) {
            const std::string wav = argv[i];
            const auto direct = engine.transcribe(wav);
            const auto via_mel = engine.transcribe_mel(engine.compute_mel_wav(wav));
            if (direct.text.empty() || direct.text != via_mel.text ||
                direct.token_ids != via_mel.token_ids) {
                std::fprintf(stderr, "TranscribePsy EU API mismatch for %s\n", argv[i]);
                return 1;
            }
        }
        auto malformed = engine.compute_mel_wav(argv[2]);
        malformed.eu_gate_frames = 2001;
        malformed.eu_gate_mel = {0.0f};
        try {
            engine.transcribe_mel(std::move(malformed));
            std::fprintf(stderr, "TranscribePsy EU accepted an undersized gate mel\n");
            return 1;
        } catch (const std::runtime_error & error) {
            if (std::string(error.what()).find("invalid EU gate mel buffer") == std::string::npos)
                throw;
        }
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
