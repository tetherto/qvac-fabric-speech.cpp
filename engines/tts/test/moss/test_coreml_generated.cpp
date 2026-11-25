// The MOSS speech tokenizer and SoundEffect VAE generated from their GGUFs on
// the device:
//
//   test-moss-coreml-generated [--tokenizer GGUF] [--vae GGUF] [--wav WAV]
//       --cache-dir DIR [--require-coreml] [--cpu]
//   test-moss-coreml-generated [--tokenizer GGUF] [--vae GGUF]
//       --memory-schedule DIR [--require-coreml] [--cpu]
//
// --tokenizer takes a GGUF with the speech tokenizer (the codec GGUF, or the
// tokenizer-only one convert-moss-speech-codec-to-gguf.py --tokenizer-only
// writes), --vae one with the SoundEffect VAE (the SoundEffect GGUF, or the
// VAE-only one convert-moss-sfx-to-gguf.py --vae-only writes). --cache-dir
// must be empty or absent: each stage is generated from the GGUF alone into
// its own subdirectory, compiled and loaded on the GPU (cpu_and_gpu, the
// stages' default), then checked against ggml on the same weights:
//   tokenizer  one full 30 s segment of --wav (default: jfk.wav looped) gives
//              codes that agree with ggml at >= 0.95 (the prebuilt sidecar's
//              gate); a shorter input never demands the generated model
//   VAE        a 500-frame decode on the generated model has cosine >= 0.999
//              with the ggml decode, and fit charges the template envelope,
//              the same before and after the generation, with a stale hint,
//              and drops Core ML near the budget (memory-degrade)
// plus each stage's route and status rules (validated units only, strict
// partial segments, the memory budget).
//
// --memory-schedule runs the overlapped memory schedule of each stage into a
// fresh cache under DIR and holds every phase to the template constant that
// charges it (engines/coreml/test/memory_schedule.h).
//
// Exits 77 (skip) without TTS_CPP_COREML or when no stage is given;
// --require-coreml turns every Core ML route that is not `generate` into a
// failure instead.

#include "../test_env_portable.h"

#include "moss/coreml_generation.h"
#include "moss/coreml_sidecar.h"
#include "moss/sfx_coreml.h"
#include "moss/sfx_model.h"
#include "moss/sfx_networks.h"
#include "moss/sfx_request.h"
#include "moss/sfx_sampler.h"
#include "moss/speech_tokenizer.h"

#include "memory_probe.h"
#include "memory_schedule.h"
#include "voice_features.h"
#include "../../../coreml/third_party/nlohmann/json.hpp"

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace tts_cpp::moss::detail;
using tts_cpp::CoremlOptions;
namespace fs = std::filesystem;

namespace {

constexpr int SKIP = 77;
constexpr int THREADS = 8;
constexpr double CODE_AGREEMENT = 0.95;
constexpr double VAE_COSINE = 0.999;
constexpr int VAE_KEEP_FRAMES = 500;
constexpr uint32_t NOISE_SEED = 7;
constexpr const char * GPU_LABEL = "coreml-gpu";
constexpr int SCHEDULE_COREML_CALLS = 5;
constexpr auto SCHEDULE_LIMIT = std::chrono::minutes(10);

int failures = 0;
bool require_coreml = false;
bool on_gpu = true;

void check(bool condition, const std::string & label) {
    std::printf("%s: %s\n", condition ? "ok  " : "FAIL", label.c_str());
    std::fflush(stdout);
    if (!condition) ++failures;
}

double mib(uint64_t bytes) { return (double) bytes / (1024.0 * 1024.0); }

std::string cpu_brand() {
#if defined(__APPLE__)
    char brand[256] = {};
    size_t size = sizeof(brand);
    if (sysctlbyname("machdep.cpu.brand_string", brand, &size, nullptr, 0) == 0) return brand;
#endif
    return "unknown";
}

BackendRequest backend() { return {"", on_gpu}; }

CoremlOptions generated_options(const fs::path & cache) {
    CoremlOptions options;
    options.cache_dir = cache.string();
    options.compute_units = "cpu_and_gpu";
    return options;
}

std::vector<float> looped(const std::vector<float> & pcm, size_t samples) {
    std::vector<float> out;
    out.reserve(samples);
    while (out.size() < samples) {
        const size_t take = std::min(pcm.size(), samples - out.size());
        out.insert(out.end(), pcm.begin(), pcm.begin() + (std::ptrdiff_t) take);
    }
    return out;
}

double cosine(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size() || a.empty()) return -2.0;
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += (double) a[i] * b[i];
        na += (double) a[i] * a[i];
        nb += (double) b[i] * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
}

double code_agreement(const std::vector<int32_t> & a, const std::vector<int32_t> & b) {
    if (a.size() != b.size() || a.empty()) return 0.0;
    size_t same = 0;
    for (size_t i = 0; i < a.size(); ++i) same += a[i] == b[i] ? 1 : 0;
    return (double) same / (double) a.size();
}

nlohmann::json status_of(const MossCoremlGeneration & generation) {
    return nlohmann::json::parse(generation.status_json(false));
}

// Whether `stage` generates from `gguf` under cpu_and_gpu. A route that does
// not is a failure under --require-coreml and a reason to skip the stage
// otherwise.
bool generates(GeneratedStage stage, const std::string & gguf, const char * name) {
    const auto route = plan_generated_route(stage, gguf, generated_options(fs::temp_directory_path()));
    if (route.route == "generate") {
        check(route.validated && route.compute_units == "cpu_and_gpu" && route.envelope_bytes > 0,
              std::string(name) + ": route generate, validated for cpu_and_gpu, envelope " +
                  std::to_string(route.envelope_bytes) + " bytes");
        return true;
    }
    const std::string why = std::string(name) + ": route " + route.route + " (" + route.state + ")";
    if (require_coreml) {
        check(false, why + "; expected generate");
    } else {
        std::printf("%s: not run (%s)\n", name, why.c_str());
    }
    return false;
}

// Only the validated units generate: cpu_and_ane has no measured envelope.
void check_unvalidated_units(GeneratedStage stage, const std::string & gguf, const fs::path & cache,
                             const char * name) {
    auto options = generated_options(cache);
    options.compute_units = "cpu_and_ane";
    const auto route = plan_generated_route(stage, gguf, options);
    MossCoremlGeneration generation(stage, gguf, options);
    check(route.route == "none" && route.state == "unsupported(unvalidated_units)" && !generation.can_generate() &&
              status_of(generation)["state"] == "unsupported(unvalidated_units)",
          std::string(name) + ": cpu_and_ane is not validated, so it stays on ggml (" + route.state + ")");
}

void check_ready(const MossCoremlGeneration & generation, const char * route, const std::string & name) {
    const auto json = status_of(generation);
    check(json["state"] == "ready" && json["route"] == route && json["label"] == GPU_LABEL,
          name + ": slot ready via " + route + " (" + json.dump() + ")");
    check(json["timings_ms"]["warm"].get<double>() > 0.0, name + ": the GPU slot ran its warm-up prediction");
}

// ---- speech tokenizer ------------------------------------------------------

void run_tokenizer(const std::string & gguf, const std::vector<float> & audio, const fs::path & cache) {
    check_unvalidated_units(GeneratedStage::speech_tokenizer, gguf, cache / "unvalidated", "tokenizer");
    auto options = generated_options(cache / "generated");
    options.prewarm = CoremlOptions::Prewarm::none;
    options.wait = CoremlOptions::Wait::call;
    SpeechTokenizer generated(gguf, backend(), THREADS, AttentionLayout::Automatic, options);
    auto off = options;
    off.use = CoremlOptions::Use::off;
    SpeechTokenizer ggml(gguf, backend(), THREADS, AttentionLayout::Automatic, off);
    check(!generated.on_coreml() && nlohmann::json::parse(generated.coreml_status_json())["state"] == "idle",
          "tokenizer: an unrequested generated tokenizer stays idle");

    // A partial segment never demands the generated model (prewarm none:
    // nothing else would start it).
    const size_t segment = (size_t) generated.config().chunk_samples;
    const auto partial = looped(audio, segment - 1);
    generated.begin_run();
    check(generated.encode(partial) == ggml.encode(partial) && generated.run_backend() == GGML_STAGE_BACKEND &&
              nlohmann::json::parse(generated.coreml_status_json())["state"] == "idle",
          "tokenizer: a partial segment is bit-equal to ggml and leaves generation idle");

    // One full segment waits for the generated model (wait=call) and runs on it.
    const auto full = looped(audio, segment);
    generated.begin_run();
    const auto codes = generated.encode(full);
    const double agreement = code_agreement(codes, ggml.encode(full));
    std::printf("tokenizer: %zu codes, agreement %.4f with ggml on %s\n", codes.size(), agreement,
                generated.run_backend().c_str());
    check(generated.run_backend() == GPU_LABEL, "tokenizer: the full segment ran on the generated model");
    check(agreement >= CODE_AGREEMENT, "tokenizer: generated codes agree with ggml (>= 0.95)");
    const auto status = nlohmann::json::parse(generated.coreml_status_json());
    check(status["state"] == "ready" && status["route"] == "generated" && status["label"] == GPU_LABEL &&
              status["timings_ms"]["warm"].get<double>() > 0.0,
          "tokenizer: slot ready via generated after its warm-up prediction (" + status.dump() + ")");

    // A second tokenizer on the same cache adopts the artifact.
    SpeechTokenizer cached(gguf, backend(), THREADS, AttentionLayout::Automatic, options);
    cached.begin_run();
    cached.encode(full);
    check(cached.run_backend() == GPU_LABEL &&
              nlohmann::json::parse(cached.coreml_status_json())["route"] == "cached",
          "tokenizer: a second tokenizer reuses the cached artifact");

    // Strict: a partial segment is not a failure while the generated route
    // exists; it never runs on that model.
    auto strict = options;
    strict.use = CoremlOptions::Use::require;
    bool threw = false;
    try {
        SpeechTokenizer required(gguf, backend(), THREADS, AttentionLayout::Automatic, strict);
        required.begin_run();
        required.encode(partial);
        threw = required.run_backend() != GGML_STAGE_BACKEND;
    } catch (const std::exception &) {
        threw = true;
    }
    check(!threw, "tokenizer: strict mode runs a partial segment on ggml");

    // SpeechEngine has no tokenizer fit estimate, so a memory budget
    // suppresses generation (and its strictness) rather than charge an
    // envelope nothing assessed.
    auto budget = strict;
    budget.memory_budget_bytes = int64_t(1) << 40;
    SpeechTokenizer limited(gguf, backend(), THREADS, AttentionLayout::Automatic, budget);
    limited.begin_run();
    bool limited_ok = true;
    try {
        limited.encode(full);
    } catch (const std::exception &) {
        limited_ok = false;
    }
    check(limited_ok && limited.run_backend() == GGML_STAGE_BACKEND &&
              nlohmann::json::parse(limited.coreml_status_json())["state"] == "disabled",
          "tokenizer: a memory budget keeps the tokenizer on ggml, strict included (" +
              std::string(limited_ok ? "" : "threw, ") + limited.run_backend() + ", " + limited.coreml_status_json() +
              ")");
}

// ---- SoundEffect VAE -------------------------------------------------------

std::vector<int64_t> vae_input(const SfxConfig & config) { return {1, config.vae.latent_dim, GENERATED_VAE_WINDOW}; }

std::vector<int64_t> vae_output(const SfxConfig & config) {
    return {1, 1, (int64_t) config.vae.hop_length() * GENERATED_VAE_WINDOW};
}

void stale_hints(const fs::path & cache) {
    for (const auto & entry : fs::directory_iterator(cache / "qvac-coreml-v1" / "hints")) {
        if (entry.path().extension() != ".json") continue;
        auto hint = nlohmann::json::parse(std::ifstream(entry.path()));
        hint["stale_until"] = "2099-01-01T00:00:00Z";
        std::ofstream(entry.path()) << hint;
    }
}

void run_vae(const std::string & gguf, const fs::path & cache_root) {
    check_unvalidated_units(GeneratedStage::sfx_vae, gguf, cache_root / "unvalidated", "VAE");
    const fs::path cache = cache_root / "generated";
    auto options = generated_options(cache);
    options.prewarm = CoremlOptions::Prewarm::none;
    options.wait = CoremlOptions::Wait::call;
    constexpr uint64_t GGML_BYTES = 64ull << 20;
    constexpr uint64_t FIT_BUDGET = 4ull << 30;
    const auto before = sfx_vae_coreml_fit(gguf, options, GGML_BYTES, FIT_BUDGET, true);
    const auto route = plan_generated_route(GeneratedStage::sfx_vae, gguf, options);
    check(before.route == "generate" && before.bytes == route.envelope_bytes && before.validated && before.fits,
          "VAE: fit charges the template envelope (" + std::to_string(before.bytes) + " bytes)");

    const SfxLayout layout{TablePlacement::Device, AttentionLayout::Automatic, WeightPacking::Plain};
    SfxModel model(gguf, backend(), THREADS, false, layout);
    const auto & config = model.config();
    MossCoremlGeneration generation(GeneratedStage::sfx_vae, gguf, options);
    generation.start(vae_input(config), vae_output(config));
    auto sidecar = acquire_sfx_vae_sidecar(generation, nullptr);
    check(sidecar != nullptr, "VAE: a request acquires its generated decoder");
    check_ready(generation, "generated", "VAE");
    if (sidecar) {
        const int frames = config.latent_frames();
        const auto latents = gaussian_noise((size_t) frames * config.vae.latent_dim, NOISE_SEED);
        const auto reference =
            decode_latents(model, latents, frames, VAE_KEEP_FRAMES, SFX_DECODE_WINDOW_FRAMES, {});
        StageBackends backends;
        const auto pcm = decode_sfx_latents(model, sidecar, true, backends, latents, frames, VAE_KEEP_FRAMES,
                                            SFX_DECODE_WINDOW_FRAMES, {});
        check(backends.summary() == GPU_LABEL, "VAE: the whole decode ran on the generated model");
        const double c = cosine(pcm, reference);
        std::printf("VAE: waveform cosine %.6f with ggml\n", c);
        check(c >= VAE_COSINE, "VAE: the generated waveform matches ggml (>= 0.999)");
    }

    const auto after = sfx_vae_coreml_fit(gguf, options, GGML_BYTES, FIT_BUDGET, true);
    check(after.route == before.route && after.bytes == before.bytes && after.fits == before.fits,
          "VAE: fit is the same with the populated cache and its hint");
    MossCoremlGeneration cached(GeneratedStage::sfx_vae, gguf, options);
    cached.start(vae_input(config), vae_output(config));
    check(cached.acquire() != nullptr && status_of(cached)["route"] == "cached",
          "VAE: a second slot reuses the cached artifact");
    stale_hints(cache);
    const auto stale = sfx_vae_coreml_fit(gguf, options, GGML_BYTES, FIT_BUDGET, true);
    check(stale.route == before.route && stale.bytes == before.bytes && stale.fits == before.fits,
          "VAE: fit is the same with a stale hint");

    // Near the budget the Core ML-inclusive verdict fails while ggml alone
    // fits: fit keeps the ggml verdict and construction resolves use=off,
    // even with the artifact cached.
    auto near = options;
    near.memory_budget_bytes = (int64_t) (GGML_BYTES + before.bytes / 2);
    near.use = CoremlOptions::Use::require;
    const auto degraded = sfx_vae_coreml_fit(gguf, near, GGML_BYTES, FIT_BUDGET, true);
    check(degraded.disabled_reason == "memory" && degraded.ggml_only_fits && degraded.route == "generate",
          "VAE: near the budget fit reports memory-degrade with the ggml verdict");
    MossCoremlGeneration limited(GeneratedStage::sfx_vae, gguf, near);
    limited.start(vae_input(config), vae_output(config), GGML_BYTES);
    check(!limited.can_generate() && limited.acquire() == nullptr && status_of(limited)["state"] == "disabled" &&
              !limited.prebuilt_policy().strict,
          "VAE: near the budget construction drops Core ML (and strict) even with the artifact cached");
}

// ---- memory schedule -------------------------------------------------------

struct ScheduleCall {
    bool ok = false;
    std::string backend;
};

// The overlapped memory schedule of one stage: ggml calls on this thread
// while the stage's sidecar is generated, compiled, loaded and warmed in the
// background into a fresh cache, until SCHEDULE_COREML_CALLS calls have run on
// Core ML. One sampler thread reads the footprint and the slot state every
// millisecond. The units (cpu_and_gpu) do not include the Neural Engine, so
// there is no compiler reserve to check. 1.25 x every phase's measurement
// must stay within the template constant that charges it, and a failed probe
// or a call that failed or ran on another backend fails the run.
// `coreml_call` acquires the stage's model from the slot and runs one call on
// it, or returns no call while the slot is not ready.
template <class GgmlCall, class CoremlCall>
void memory_schedule(const char * name, GeneratedStage stage, const std::string & gguf, const fs::path & cache,
                     const std::vector<int64_t> & input, const std::vector<int64_t> & output, GgmlCall ggml_call,
                     CoremlCall coreml_call) {
    using namespace qvac_coreml_memory;
    const std::string chip = cpu_brand();
    std::printf("%s: memory schedule (cache %s, host %s)\n", name, cache.c_str(), chip.c_str());
    auto options = generated_options(cache);
    // Create the slot without queueing it. The first acquire starts work
    // after the sampler can safely read the slot.
    options.prewarm = CoremlOptions::Prewarm::none;
    options.wait = CoremlOptions::Wait::none;
    const auto route = plan_generated_route(stage, gguf, options);
    if (route.route != "generate") {
        check(false, std::string(name) + ": memory schedule: route " + route.route + " (" + route.state + ")");
        return;
    }
    for (int i = 0; i < 2; ++i) {  // warm the ggml arena
        const ScheduleCall call = ggml_call();
        if (classify_call(call.ok, call.backend, GGML_STAGE_BACKEND, "") != schedule_call::ggml) {
            check(false, std::string(name) + ": memory schedule: the ggml warm-up call succeeds (" + call.backend +
                             ")");
            return;
        }
    }
    MossCoremlGeneration generation(stage, gguf, options);
    generation.start(input, output);

    uint64_t base = 0;
    const bool base_ok = phys_footprint(base);
    size_t footprint_failures = 0;
    std::vector<mem_sample> samples;
    samples.reserve(1 << 20);
    int ggml_calls = 0, coreml_calls = 0, bad_calls = 0;
    const auto t0 = std::chrono::steady_clock::now();
    {
        std::atomic<bool> stop{false};
        std::thread sampler([&] {
            while (!stop.load()) {
                const mem_phase phase = mem_phase_of(generation.state());
                uint64_t fp = 0;
                if (phys_footprint(fp)) {
                    samples.push_back({fp, phase});
                } else {
                    ++footprint_failures;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
        struct StopSampler {
            std::atomic<bool> & stop;
            std::thread & sampler;
            ~StopSampler() {
                stop.store(true);
                sampler.join();
            }
        } stop_sampler{stop, sampler};
        while (std::chrono::steady_clock::now() - t0 < SCHEDULE_LIMIT) {
            const std::optional<ScheduleCall> on_coreml = coreml_call(generation);
            const ScheduleCall call = on_coreml ? *on_coreml : ggml_call();
            const schedule_call k = classify_call(call.ok, call.backend, GGML_STAGE_BACKEND, GPU_LABEL);
            if (k == schedule_call::bad) {
                ++bad_calls;
                check(false, std::string(name) + ": memory schedule: every overlapped call succeeds on ggml or " +
                                 GPU_LABEL + " (" + (call.ok ? call.backend : "failed on " + call.backend) + ")");
                break;
            }
            if (k == schedule_call::ggml) {
                ++ggml_calls;
            } else if (++coreml_calls >= SCHEDULE_COREML_CALLS) {
                break;
            }
        }
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("  status: %s\n", generation.status_json(false).c_str());

    const mem_breakdown b = mem_breakdown_of(samples, base);
    const mem_charges c{route.prepare_peak_bytes, route.load_peak_bytes, route.resident_bytes,
                        route.ane_compile_reserve_bytes, route.envelope_bytes};
    const uint64_t other_resident = route.resident_bytes - route.workspace_bytes;
    const uint64_t workspace = b.steady > other_resident ? b.steady - other_resident : 0;
    std::printf("  memory (MiB; each phase over the footprint just before it began, steady over the footprint just "
                "before the load, whole run over the warm ggml baseline %.1f; a phase needs x%.2f its "
                "measurement):\n"
                "    prepare    %7.1f  needs %7.1f  charged %7.1f  (generate %.1f, compile %.1f)\n"
                "    load+warm  %7.1f  needs %7.1f  charged %7.1f  (load %.1f, warm-up %.1f)\n"
                "    steady     %7.1f  needs %7.1f  charged %7.1f  (workspace %.1f, charged %.1f; over the baseline "
                "%.1f)\n"
                "    whole run  %7.1f  needs %7.1f  charged %7.1f  (the envelope: margined phases)\n",
                mib(base), MEASURED_MARGIN, mib(b.prepare), MEASURED_MARGIN * mib(b.prepare), mib(c.prepare),
                mib(b.generate), mib(b.compile), mib(b.load_warm), MEASURED_MARGIN * mib(b.load_warm), mib(c.load),
                mib(b.load), mib(b.warm), mib(b.steady), MEASURED_MARGIN * mib(b.steady), mib(c.resident),
                mib(workspace), mib(route.workspace_bytes), mib(b.steady_over_base), mib(b.total), mib(b.total),
                mib(c.envelope));
    std::printf("  memory report: {\"chip\":\"%s\",\"template\":\"%s\",\"units\":\"%s\",\"base\":%llu,"
                "\"prepare\":%llu,\"generate\":%llu,\"compile\":%llu,\"load_warm\":%llu,\"load\":%llu,\"warm\":%llu,"
                "\"steady\":%llu,\"steady_over_base\":%llu,\"workspace\":%llu,\"total\":%llu,\"samples\":%zu}\n",
                chip.c_str(), route.template_id.c_str(), route.compute_units.c_str(), (unsigned long long) base,
                (unsigned long long) b.prepare, (unsigned long long) b.generate, (unsigned long long) b.compile,
                (unsigned long long) b.load_warm, (unsigned long long) b.load, (unsigned long long) b.warm,
                (unsigned long long) b.steady, (unsigned long long) b.steady_over_base,
                (unsigned long long) workspace, (unsigned long long) b.total, samples.size());
    std::vector<std::string> problems = probe_failures(base_ok, footprint_failures, false, AneGrowth{});
    for (const std::string & f : mem_failures(b, 0, c)) problems.push_back(f);
    for (const std::string & f : problems) check(false, std::string(name) + ": memory schedule: " + f);
    std::printf("%s: overlapped schedule: %d ggml + %d Core ML calls in %.1f s; footprint base %.1f MiB, peak +%.1f "
                "MiB (envelope %.1f MiB, units %s)\n",
                name, ggml_calls, coreml_calls, secs, mib(base), mib(b.total), mib(route.envelope_bytes),
                route.compute_units.c_str());
    check(coreml_calls >= SCHEDULE_COREML_CALLS && ggml_calls >= 1 && bad_calls == 0,
          std::string(name) + ": memory schedule: " + std::to_string(ggml_calls) + " ggml and " +
              std::to_string(coreml_calls) + " Core ML calls");
}

void schedule_tokenizer(const std::string & gguf, const fs::path & cache) {
    auto off = generated_options(cache);
    off.use = CoremlOptions::Use::off;
    SpeechTokenizer ggml(gguf, backend(), THREADS, AttentionLayout::Automatic, off);
    const auto & config = ggml.config();
    const SpeechSegmentCapacity capacity = speech_segment_capacity(config);
    std::vector<float> audio((size_t) config.chunk_samples);
    for (size_t i = 0; i < audio.size(); ++i) audio[i] = 0.1f * std::sin(0.05f * (float) i);
    std::vector<float> mel((size_t) config.n_mels * capacity.mel_frames, 0.0f);
    std::vector<float> states;
    memory_schedule(
        "tokenizer", GeneratedStage::speech_tokenizer, gguf, cache, {1, config.n_mels, capacity.mel_frames},
        {1, capacity.tokens, config.n_embd},
        [&] {
            ggml.begin_run();
            const bool ok = !ggml.encode(audio).empty();
            return ScheduleCall{ok, ggml.run_backend()};
        },
        [&](MossCoremlGeneration & generation) -> std::optional<ScheduleCall> {
            auto model = generation.acquire();
            if (!model) return std::nullopt;
            return ScheduleCall{model->predict(mel, states), model->label()};
        });
}

void schedule_vae(const std::string & gguf, const fs::path & cache) {
    const SfxLayout layout{TablePlacement::Device, AttentionLayout::Automatic, WeightPacking::Plain};
    SfxModel model(gguf, backend(), THREADS, false, layout);
    const auto & config = model.config();
    const int frames = config.latent_frames();
    const auto latents = gaussian_noise((size_t) frames * config.vae.latent_dim, NOISE_SEED);
    const auto decode = [&](std::unique_ptr<SfxVaeSidecar> & sidecar) {
        StageBackends backends;
        const auto pcm = decode_sfx_latents(model, sidecar, false, backends, latents, frames, VAE_KEEP_FRAMES,
                                            SFX_DECODE_WINDOW_FRAMES, {});
        return ScheduleCall{!pcm.empty(), backends.summary()};
    };
    std::unique_ptr<SfxVaeSidecar> none;
    memory_schedule(
        "VAE", GeneratedStage::sfx_vae, gguf, cache, vae_input(config), vae_output(config),
        [&] { return decode(none); },
        [&](MossCoremlGeneration & generation) -> std::optional<ScheduleCall> {
            // One request holds one model for its whole decode, as the engine does.
            auto sidecar = acquire_sfx_vae_sidecar(generation, nullptr);
            if (!sidecar) return std::nullopt;
            return decode(sidecar);
        });
}

}  // namespace

int main(int argc, char ** argv) {
    std::string tokenizer, vae, cache_dir, memory_dir;
    std::string wav = TTS_CPP_MOSS_TEST_WAV;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--tokenizer" && i + 1 < argc) tokenizer = argv[++i];
        else if (a == "--vae" && i + 1 < argc) vae = argv[++i];
        else if (a == "--wav" && i + 1 < argc) wav = argv[++i];
        else if (a == "--cache-dir" && i + 1 < argc) cache_dir = argv[++i];
        else if (a == "--memory-schedule" && i + 1 < argc) memory_dir = argv[++i];
        else if (a == "--require-coreml") require_coreml = true;
        else if (a == "--cpu") on_gpu = false;
        else {
            std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
            return 2;
        }
    }
#ifndef TTS_CPP_USE_COREML
    std::printf("%s: built without TTS_CPP_COREML\n", require_coreml ? "FAIL" : "SKIP");
    return require_coreml ? 1 : SKIP;
#endif
    if (tokenizer.empty() && vae.empty()) {
        std::printf("%s: pass --tokenizer and/or --vae\n", require_coreml ? "FAIL" : "SKIP");
        return require_coreml ? 1 : SKIP;
    }
    if (cache_dir.empty() == memory_dir.empty()) {
        std::fprintf(stderr, "pass exactly one of --cache-dir and --memory-schedule\n");
        return 2;
    }
    unsetenv("MOSS_COREML_DISABLE");
    unsetenv("MOSS_COREML_STRICT");
    unsetenv("MOSS_COREML_COMPUTE_UNITS");
    unsetenv("QVAC_COREML_CACHE_DIR");
    const fs::path root = cache_dir.empty() ? fs::path(memory_dir) : fs::path(cache_dir);
    std::error_code ec;
    if (fs::exists(root) && !fs::is_empty(root, ec)) {
        std::fprintf(stderr, "%s must be empty: every stage is generated into a fresh cache\n", root.c_str());
        return 2;
    }
    bool ran = false;
    try {
        if (!tokenizer.empty() && generates(GeneratedStage::speech_tokenizer, tokenizer, "tokenizer")) {
            ran = true;
            if (memory_dir.empty()) {
                std::vector<float> audio;
                int rate = 0;
                if (!wav_load(wav, audio, rate)) throw std::runtime_error("cannot read " + wav);
                run_tokenizer(tokenizer, audio, root / "tokenizer");
            } else {
                schedule_tokenizer(tokenizer, root / "tokenizer");
            }
        }
        if (!vae.empty() && generates(GeneratedStage::sfx_vae, vae, "VAE")) {
            ran = true;
            if (memory_dir.empty()) {
                run_vae(vae, root / "vae");
            } else {
                schedule_vae(vae, root / "vae");
            }
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    if (!ran && failures == 0) {
        std::printf("SKIP: no stage generates on this host\n");
        return SKIP;
    }
    std::printf(failures ? "moss generated Core ML: %d failure(s)\n" : "moss generated Core ML: OK\n", failures);
    return failures ? 1 : 0;
}
