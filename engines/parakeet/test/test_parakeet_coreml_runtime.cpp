// Parakeet Core ML runtime adoption on the device (PLAN §3.2.0, §2.15).
// Needs a TDT GGUF with its adjacent `-encoder.mlmodelc` and a wav; the
// diarizer and encoder-less cases run when their fixtures are given.
//
//   ggml-unchanged        a call whose sidecar is still loading runs on ggml
//                         (coreml_pending) and its transcript equals use = off
//   prebuilt-slot-loads   the prebuilt encoder loads in the background through
//                         its slot; the next call runs on Core ML
//   diarizer-pending      a diarizer capacity that is not ready -> kNoSidecar
//                         -> ggml; ready after prewarm all -> Core ML
//   coreml-only-full      PARAKEET_COREML_ONLY with a full GGUF keeps the ggml
//                         encoder when Core ML is off or the takeover fails
//   encoderless-use-off   an encoder-less GGUF with use = off fails with
//                         kGgufLoadNoEncoderWeights, as today
//
// Every case runs or fails: there is no skip path (exit 0 = pass, 1 = fail,
// 2 = bad arguments).

#include "parakeet/engine.h"
#include "parakeet_ctc.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>

#include <unistd.h>

namespace fs = std::filesystem;
using parakeet::CoremlOptions;

namespace {

int g_failures = 0;

void expect(bool ok, const char * what) {
    std::printf("[%s] %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++g_failures;
}

constexpr const char * kCoremlOnlyEnv = "PARAKEET_COREML_ONLY";

fs::path fresh_dir(const char * tag) {
    const fs::path dir = fs::temp_directory_path() /
                         ("parakeet-coreml-runtime-" + std::string(tag) + "-" + std::to_string(::getpid()));
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir);
    return dir;
}

parakeet::EngineOptions engine_options(const std::string & gguf, CoremlOptions coreml) {
    parakeet::EngineOptions o;
    o.model_gguf_path = gguf;
    o.n_gpu_layers = 99;
    o.coreml = std::move(coreml);
    return o;
}

CoremlOptions off() {
    CoremlOptions o;
    o.use = CoremlOptions::Use::off;
    return o;
}

// Nothing requested at construction and a cache with no hints: the first
// call's acquisition requests the load and cannot wait for it (no warm hint,
// no bounded wait), so it runs on ggml.
CoremlOptions lazy(const fs::path & cache) {
    CoremlOptions o;
    o.prewarm = CoremlOptions::Prewarm::none;
    o.wait = CoremlOptions::Wait::none;
    o.cache_dir = cache.string();
    return o;
}

bool has(const std::string & text, const char * needle) { return text.find(needle) != std::string::npos; }

void case_transcription(const std::string & gguf, const std::string & wav,
                        parakeet::EngineResult & reference) {
    parakeet::Engine ggml_engine(engine_options(gguf, off()));
    reference = ggml_engine.transcribe(wav);
    expect(reference.encoder_backend == "ggml" && !reference.coreml_pending,
           "use = off: encoder_backend ggml, not pending");
    expect(!reference.text.empty(), "use = off produced a transcript");

    const fs::path cache = fresh_dir("cache");
    parakeet::Engine engine(engine_options(gguf, lazy(cache)));
    const parakeet::EngineResult first = engine.transcribe(wav);
    std::printf("       pending call: backend=%s pending=%d\n", first.encoder_backend.c_str(),
                first.coreml_pending ? 1 : 0);
    expect(first.encoder_backend == "ggml" && first.coreml_pending && !first.encoder_used_coreml,
           "ggml-unchanged: the first call ran on ggml because the sidecar was loading");
    expect(first.text == reference.text && first.token_ids == reference.token_ids,
           "ggml-unchanged: the pending call's transcript equals use = off (no template is embedded "
           "in this build, so this is the without-template side)");

    const std::string prewarm = engine.coreml_prewarm(CoremlOptions::Prewarm::minimal, /*wait=*/true);
    expect(has(prewarm, "\"success\":true") || has(prewarm, "\"success\": true"),
           "prebuilt-slot-loads: the minimal group reached ready");
    const std::string status = engine.coreml_status_json();
    std::printf("       status: %s\n", status.c_str());
    expect(has(status, "\"route\":\"prebuilt\"") || has(status, "\"route\": \"prebuilt\""),
           "prebuilt-slot-loads: status reports the prebuilt route");
    const parakeet::EngineResult second = engine.transcribe(wav);
    std::printf("       ready call: backend=%s pending=%d\n", second.encoder_backend.c_str(),
                second.coreml_pending ? 1 : 0);
    expect(second.encoder_used_coreml && second.encoder_backend.rfind("coreml-", 0) == 0 &&
               !second.coreml_pending,
           "prebuilt-slot-loads: the next call runs on Core ML");
    std::error_code ec;
    fs::remove_all(cache, ec);
}

void case_diarizer(const std::string & gguf, const std::string & wav) {
    const fs::path cache = fresh_dir("diar-cache");
    parakeet::Engine engine(engine_options(gguf, lazy(cache)));
    const parakeet::DiarizationResult first = engine.diarize(wav);
    std::printf("       pending diarize: backend=%s pending=%d\n", first.encoder_backend.c_str(),
                first.coreml_pending ? 1 : 0);
    expect(first.encoder_backend == "ggml" && first.coreml_pending && !first.encoder_used_coreml,
           "diarizer-pending: a capacity that is not ready gives kNoSidecar -> ggml");

    engine.coreml_prewarm(CoremlOptions::Prewarm::all, /*wait=*/true);
    const parakeet::DiarizationResult second = engine.diarize(wav);
    std::printf("       ready diarize: backend=%s pending=%d\n", second.encoder_backend.c_str(),
                second.coreml_pending ? 1 : 0);
    expect(second.encoder_used_coreml && second.encoder_backend != "ggml",
           "diarizer-pending: after prewarm all the diarizer runs on Core ML");
    std::error_code ec;
    fs::remove_all(cache, ec);
}

void case_coreml_only_full(const std::string & gguf, const std::string & wav,
                           const parakeet::EngineResult & reference) {
    ::setenv(kCoremlOnlyEnv, "1", 1);
    {
        parakeet::ParakeetCtcModel model;
        const int rc = parakeet::load_from_gguf(gguf, model, 0, 99, false, "auto", off());
        expect(rc == 0 && !parakeet::model_encoder_owned_by_coreml(model),
               "coreml-only-full: use = off keeps the GGUF's ggml encoder");
    }
    {
        // The takeover attempt fails (no sidecar next to this copy): the full
        // GGUF keeps its ggml encoder and transcribes as before.
        const fs::path dir = fresh_dir("no-sidecar");
        const fs::path staged = dir / fs::path(gguf).filename();
        fs::create_symlink(fs::absolute(gguf), staged);
        parakeet::ParakeetCtcModel model;
        const int rc = parakeet::load_from_gguf(staged.string(), model, 0, 99, false, "auto", CoremlOptions{});
        expect(rc == 0 && !parakeet::model_encoder_owned_by_coreml(model),
               "coreml-only-full: a failed takeover keeps the ggml encoder");
        parakeet::Engine engine(engine_options(staged.string(), CoremlOptions{}));
        const parakeet::EngineResult r = engine.transcribe(wav);
        expect(r.text == reference.text && r.encoder_backend == "ggml",
               "coreml-only-full: the kept ggml encoder transcribes as use = off");
        std::error_code ec;
        fs::remove_all(dir, ec);
    }
    ::unsetenv(kCoremlOnlyEnv);
}

void case_encoderless(const std::string & gguf) {
    parakeet::ParakeetCtcModel model;
    const int rc = parakeet::load_from_gguf(gguf, model, 0, 0, false, "auto", off());
    std::printf("       encoder-less load rc=%d\n", rc);
    expect(rc == parakeet::kGgufLoadNoEncoderWeights,
           "encoderless-use-off: kGgufLoadNoEncoderWeights, as today");
}

}  // namespace

int main(int argc, char ** argv) {
    std::string gguf, wav, diar_gguf, diar_wav, encoderless;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--gguf") gguf = argv[i + 1];
        else if (a == "--wav") wav = argv[i + 1];
        else if (a == "--diar-gguf") diar_gguf = argv[i + 1];
        else if (a == "--diar-wav") diar_wav = argv[i + 1];
        else if (a == "--encoderless-gguf") encoderless = argv[i + 1];
        else {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    if (gguf.empty() || wav.empty()) {
        std::fprintf(stderr, "usage: %s --gguf TDT.gguf --wav W.wav [--diar-gguf D.gguf --diar-wav W.wav] "
                             "[--encoderless-gguf E.gguf]\n", argv[0]);
        return 2;
    }
    try {
        parakeet::EngineResult reference;
        case_transcription(gguf, wav, reference);
        case_coreml_only_full(gguf, wav, reference);
        if (!diar_gguf.empty()) case_diarizer(diar_gguf, diar_wav.empty() ? wav : diar_wav);
        if (!encoderless.empty()) case_encoderless(encoderless);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: exception: %s\n", e.what());
        return 1;
    }
    std::printf("test-parakeet-coreml-runtime: %d failure(s)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
