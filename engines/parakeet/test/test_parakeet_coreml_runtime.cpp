// Parakeet Core ML runtime adoption on the device. `--case` picks one group,
// each registered as its own CTest test so a missing fixture shows as a test
// that did not run, never as a pass:
//   transcription  (default) --gguf TDT.gguf (with its adjacent
//                  `-encoder.mlmodelc`) --wav W.wav: every case below except
//                  the two that follow
//   diarizer       --diar-gguf D.gguf (with its `-diarizer-<frames>.mlmodelc`
//                  sidecars) --wav W.wav: diarizer-pending
//   encoderless    --encoderless-gguf E.gguf: encoderless-use-off
//
//   ggml-unchanged        a call whose sidecar is still loading runs on ggml
//                         (coreml_pending) and its transcript equals use = off
//   prebuilt-slot-loads   the prebuilt encoder loads in the background through
//                         its slot; the next call runs on Core ML
//   capacity-miss         with long-form windowing disabled, an input longer
//                         than the fixed encoder's capacity runs that call on
//                         ggml and leaves the slot ready; the next call that
//                         fits runs on Core ML (a capacity miss never retires)
//   strict-over-capacity  the same over-capacity call with use = require
//                         throws instead of running on ggml, and
//                         the slot stays ready; with default windowing too,
//                         since a standard TDT sidecar is never split into
//                         Core ML windows
//   load-error-class      a corrupt encoder bundle fails its slot as transient,
//                         naming the Core ML error domain; a diarizer bundle
//                         whose capacity is not the one its name says fails
//                         as deterministic, a corrupt one as transient
//   strict-push-stream    with use = require, stream_start refuses a TDT push
//                         stream whose windows are longer than the sidecar,
//                         before any audio; use = automatic still starts it
//   diarizer-pending      a diarizer capacity that is not ready -> kNoSidecar
//                         -> ggml; ready after prewarm all -> Core ML
//   coreml-only-full      PARAKEET_COREML_ONLY with a full GGUF keeps the ggml
//                         encoder when Core ML is off or the takeover fails:
//                         the engine is not on Core ML and transcribes as
//                         use = off
//   encoderless-use-off   an encoder-less GGUF with use = off fails to load
//                         with kGgufLoadNoEncoderWeights (rc=17), as today
//
// Public Engine API only, so it links against a shared qvac-parakeet too; the
// loader-level ownership checks live in test-coreml-encoder-owner. Every case
// runs or fails: there is no skip path (exit 0 = pass, 1 = fail, 2 = bad
// arguments).

#include "coreml_test_options.h"
#include "parakeet/engine.h"
#include "test_utils.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

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

// `"key":"value"` in the status JSON, compact or pretty-printed.
bool has_kv(const std::string & json, const char * key, const char * value) {
    const std::string k = std::string("\"") + key + "\":";
    const std::string v = std::string("\"") + value + "\"";
    return json.find(k + v) != std::string::npos || json.find(k + " " + v) != std::string::npos;
}

// The status object of one stage ("" when absent): the innermost object
// around its "stage" key, found by matching braces (keys may come in any
// order, and a slot object holds nested objects).
std::string stage_status(const std::string & status, const std::string & stage) {
    for (const char * sep : {"\"stage\":\"", "\"stage\": \""}) {
        const size_t at = status.find(sep + stage + "\"");
        if (at == std::string::npos) continue;
        size_t begin = std::string::npos;
        for (size_t i = at, depth = 0; i-- > 0;) {
            if (status[i] == '}') ++depth;
            else if (status[i] == '{') {
                if (depth == 0) { begin = i; break; }
                --depth;
            }
        }
        if (begin == std::string::npos) continue;
        for (size_t i = begin, depth = 0; i < status.size(); ++i) {
            if (status[i] == '{') ++depth;
            else if (status[i] == '}' && --depth == 0) return status.substr(begin, i - begin + 1);
        }
    }
    return {};
}

// A directory Core ML cannot load as a model.
void write_corrupt_bundle(const fs::path & bundle) {
    fs::create_directories(bundle / "weights");
    std::ofstream(bundle / "coremldata.bin") << "not a Core ML model";
    std::ofstream(bundle / "model.mil") << "not MIL";
    std::ofstream(bundle / "weights" / "weight.bin") << std::string(64, 'x');
}

void case_encoder_load_error(const std::string & gguf) {
    const fs::path dir = fresh_dir("corrupt-encoder");
    const fs::path staged = dir / "model.gguf";
    fs::create_symlink(fs::absolute(gguf), staged);
    write_corrupt_bundle(dir / "model-encoder.mlmodelc");
    parakeet::Engine engine(engine_options(staged.string(), coreml_ready_options()));
    const std::string encoder = stage_status(engine.coreml_status_json(), "encoder");
    std::printf("       corrupt encoder: %s\n", encoder.c_str());
    expect(has_kv(encoder, "state", "failed") && has_kv(encoder, "error_class", "transient") &&
               encoder.find("com.apple.CoreML") != std::string::npos,
           "load-error-class: a corrupt encoder bundle fails as transient with the Core ML error domain");
    std::error_code ec;
    fs::remove_all(dir, ec);
}

void case_diarizer_load_errors(const std::string & gguf) {
    // The smallest capacity bundle, staged under a larger capacity's name.
    const fs::path source_dir = fs::absolute(gguf).parent_path();
    const std::string stem = fs::path(gguf).stem().string();
    const std::string base = stem.substr(0, stem.find('.'));
    fs::path smallest;
    int smallest_frames = 0;
    for (const auto & entry : fs::directory_iterator(source_dir)) {
        const std::string name = entry.path().filename().string();
        const std::string prefix = base + "-diarizer-";
        if (name.rfind(prefix, 0) != 0 || entry.path().extension() != ".mlmodelc") continue;
        const int frames = std::atoi(name.c_str() + prefix.size());
        if (frames > 0 && (smallest_frames == 0 || frames < smallest_frames)) {
            smallest_frames = frames;
            smallest = entry.path();
        }
    }
    expect(smallest_frames > 0, "load-error-class: a diarizer bundle is next to the GGUF");
    if (smallest_frames == 0) return;
    const fs::path dir = fresh_dir("diarizer-errors");
    const fs::path staged = dir / (base + ".gguf");
    fs::create_symlink(fs::absolute(gguf), staged);
    const int renamed = smallest_frames * 2;
    const int corrupt = smallest_frames * 3;
    fs::create_directory_symlink(fs::canonical(smallest),
                                 dir / (base + "-diarizer-" + std::to_string(renamed) + ".mlmodelc"));
    write_corrupt_bundle(dir / (base + "-diarizer-" + std::to_string(corrupt) + ".mlmodelc"));
    parakeet::Engine engine(engine_options(staged.string(), coreml_ready_options()));
    const std::string status = engine.coreml_status_json();
    const std::string wrong = stage_status(status, "diarizer-" + std::to_string(renamed));
    const std::string broken = stage_status(status, "diarizer-" + std::to_string(corrupt));
    std::printf("       renamed diarizer: %s\n       corrupt diarizer: %s\n", wrong.c_str(), broken.c_str());
    expect(has_kv(wrong, "state", "failed") && has_kv(wrong, "error_class", "deterministic"),
           "load-error-class: a diarizer whose capacity is not its name's fails as deterministic");
    expect(has_kv(broken, "state", "failed") && has_kv(broken, "error_class", "transient") &&
               broken.find("com.apple.CoreML") != std::string::npos,
           "load-error-class: a corrupt diarizer bundle fails as transient with the Core ML error domain");
    std::error_code ec;
    fs::remove_all(dir, ec);
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
        parakeet::Engine engine(engine_options(gguf, off()));
        const parakeet::EngineResult r = engine.transcribe(wav);
        expect(!engine.encoder_on_coreml() && r.encoder_backend == "ggml" && r.text == reference.text,
               "coreml-only-full: use = off keeps the GGUF's ggml encoder");
    }
    {
        // The takeover attempt fails (no sidecar next to this copy): the full
        // GGUF keeps its ggml encoder and transcribes as before.
        const fs::path dir = fresh_dir("no-sidecar");
        const fs::path staged = dir / fs::path(gguf).filename();
        fs::create_symlink(fs::absolute(gguf), staged);
        parakeet::Engine engine(engine_options(staged.string(), coreml_ready_options()));
        const parakeet::EngineResult r = engine.transcribe(wav);
        expect(!engine.encoder_on_coreml() && r.text == reference.text && r.encoder_backend == "ggml",
               "coreml-only-full: a failed takeover keeps the ggml encoder, which transcribes as use = off");
        std::error_code ec;
        fs::remove_all(dir, ec);
    }
    ::unsetenv(kCoremlOnlyEnv);
}

void case_encoderless(const std::string & gguf) {
    std::string error;
    try {
        parakeet::Engine engine(engine_options(gguf, off()));
    } catch (const std::runtime_error & e) {
        error = e.what();
    }
    std::printf("       encoder-less load: %s\n", error.empty() ? "(loaded)" : error.c_str());
    expect(has(error, "(rc=17)"), "encoderless-use-off: kGgufLoadNoEncoderWeights (rc=17), as today");
}

}  // namespace

// The TDT fixture's encoder is a fixed 1501-frame export (15 s). 30 s is
// above it, so with windowing disabled the encoder cannot serve the call.
void case_capacity_miss(const std::string & gguf, const std::string & wav) {
    std::vector<float> source;
    int rate = 0;
    expect(parakeet_test::load_wav_pcm16le_mono(wav, source, rate) && !source.empty(),
           "capacity-miss: the wav loads");
    if (source.empty()) return;
    std::vector<float> long_input;
    const size_t target = (size_t) rate * 30;
    while (long_input.size() < target) {
        const size_t n = std::min(source.size(), target - long_input.size());
        long_input.insert(long_input.end(), source.begin(), source.begin() + (long) n);
    }

    parakeet::EngineOptions o = engine_options(gguf, coreml_ready_options());
    o.long_form_window_frames = -1;
    parakeet::Engine engine(o);
    expect(engine.encoder_on_coreml(), "capacity-miss: the encoder is ready after construction");
    const parakeet::EngineResult over =
        engine.transcribe_samples(long_input.data(), (int) long_input.size(), rate);
    std::printf("       over-capacity call: mel=%d backend=%s\n", over.mel_frames,
                over.encoder_backend.c_str());
    expect(over.encoder_backend == "ggml" && !over.encoder_used_coreml && !over.text.empty(),
           "capacity-miss: the over-capacity single-pass call ran on ggml");
    const std::string status = engine.coreml_status_json();
    expect(!has(status, "\"retired\"") && engine.encoder_on_coreml(),
           "capacity-miss: the encoder slot is still ready (not retired)");
    const parakeet::EngineResult next = engine.transcribe(wav);
    std::printf("       next call: backend=%s\n", next.encoder_backend.c_str());
    expect(next.encoder_used_coreml && next.encoder_backend.rfind("coreml-", 0) == 0,
           "capacity-miss: the next call that fits runs on Core ML");

    // use = require: the call the sidecar cannot serve is an error.
    CoremlOptions strict = coreml_ready_options();
    strict.use = CoremlOptions::Use::require;
    parakeet::EngineOptions so = engine_options(gguf, strict);
    so.long_form_window_frames = -1;
    parakeet::Engine strict_engine(so);
    expect(strict_engine.encoder_on_coreml(), "strict-over-capacity: the encoder is ready");
    bool threw = false;
    try {
        (void) strict_engine.transcribe_samples(long_input.data(), (int) long_input.size(), rate);
    } catch (const std::runtime_error & e) {
        threw = has(e.what(), "Core ML is required");
        std::printf("       strict over-capacity call threw: %s\n", e.what());
    }
    expect(threw, "strict-over-capacity: the call throws instead of running on ggml");
    expect(!has(strict_engine.coreml_status_json(), "\"retired\"") && strict_engine.encoder_on_coreml(),
           "strict-over-capacity: the encoder slot is still ready");

    // Default windowing: a standard TDT sidecar (no validated seam context)
    // is never split into Core ML windows, so the same input is still one
    // dispatch the sidecar cannot serve.
    parakeet::Engine strict_default(engine_options(gguf, strict));
    expect(strict_default.encoder_on_coreml(),
           "strict-over-capacity: default windowing, the standard TDT sidecar is ready");
    threw = false;
    try {
        (void) strict_default.transcribe_samples(long_input.data(), (int) long_input.size(), rate);
    } catch (const std::runtime_error & e) {
        threw = has(e.what(), "Core ML is required");
        std::printf("       strict default-windowing call threw: %s\n", e.what());
    }
    expect(threw, "strict-over-capacity: with default windowing the standard TDT call throws too");
    expect(!has(strict_default.coreml_status_json(), "\"retired\"") && strict_default.encoder_on_coreml(),
           "strict-over-capacity: default windowing leaves the encoder slot ready");
    const parakeet::EngineResult fits = strict_default.transcribe(wav);
    expect(fits.encoder_used_coreml && fits.encoder_backend.rfind("coreml-", 0) == 0,
           "strict-over-capacity: a call within capacity still runs on Core ML");

    // Push streams: the offline TDT stream's 30 s chunk plus context is
    // longer than the 15 s sidecar, so under require every full window would
    // throw; stream_start refuses the stream before any audio.
    threw = false;
    try {
        (void) strict_default.stream_start({}, [](const parakeet::StreamingSegment &) {});
    } catch (const std::runtime_error & e) {
        threw = has(e.what(), "push-stream windows") && has(e.what(), "Core ML is required");
        std::printf("       strict stream_start threw: %s\n", e.what());
    }
    expect(threw, "strict-push-stream: stream_start refuses the stream, naming the window and capacity");
    bool started = false;
    try {
        started = engine.stream_start({}, [](const parakeet::StreamingSegment &) {}) != nullptr;
    } catch (const std::exception & e) {
        std::printf("       automatic stream_start threw: %s\n", e.what());
    }
    expect(started, "strict-push-stream: use = automatic still starts the same stream");
}

int main(int argc, char ** argv) {
    std::string which = "transcription", gguf, wav, diar_gguf, encoderless;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--case") which = argv[i + 1];
        else if (a == "--gguf") gguf = argv[i + 1];
        else if (a == "--wav") wav = argv[i + 1];
        else if (a == "--diar-gguf") diar_gguf = argv[i + 1];
        else if (a == "--encoderless-gguf") encoderless = argv[i + 1];
        else {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    const bool ok_args = (which == "transcription" && !gguf.empty() && !wav.empty()) ||
                         (which == "diarizer" && !diar_gguf.empty() && !wav.empty()) ||
                         (which == "encoderless" && !encoderless.empty());
    if (!ok_args) {
        std::fprintf(stderr, "usage: %s [--case transcription] --gguf TDT.gguf --wav W.wav\n"
                             "       %s --case diarizer --diar-gguf D.gguf --wav W.wav\n"
                             "       %s --case encoderless --encoderless-gguf E.gguf\n",
                     argv[0], argv[0], argv[0]);
        return 2;
    }
    try {
        if (which == "transcription") {
            parakeet::EngineResult reference;
            case_transcription(gguf, wav, reference);
            case_capacity_miss(gguf, wav);
            case_coreml_only_full(gguf, wav, reference);
            case_encoder_load_error(gguf);
        } else if (which == "diarizer") {
            case_diarizer(diar_gguf, wav);
            case_diarizer_load_errors(diar_gguf);
        } else {
            case_encoderless(encoderless);
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: exception: %s\n", e.what());
        return 1;
    }
    std::printf("test-parakeet-coreml-runtime (%s): %d failure(s)\n", which.c_str(), g_failures);
    return g_failures == 0 ? 0 : 1;
}
