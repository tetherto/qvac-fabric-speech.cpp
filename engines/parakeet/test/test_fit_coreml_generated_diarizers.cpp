// fit_params on a Nemotron 3 Diarization or Sortformer v2.1 GGUF with no
// adjacent diarizer bundle charges the generated capacities' Core ML
// envelope, and the charge
// never depends on the cache: an empty cache and a cache holding every
// generated capacity give the same Core ML bytes, verdict and validation.
// An Engine whose memory budget lies between the ggml-only total and the
// Core ML-inclusive total resolves Core ML off for memory; one with room for
// both keeps generation on. An Engine on compute units the templates were not
// validated for generates nothing, and every capacity keeps a status entry
// that says so; so does an Engine with generation off (generate = never).
// With prewarm = all the charge covers every generated capacity, and so does
// the default 300 s workload: an instance projected for it also runs shorter
// calls as one pass up to the long-form window (1125 frames for Nemotron),
// which keep their capacities loaded. A short workload charges less when the
// model has several capacities, the same when it has one (construction loads
// the only capacity either way).
//
// Usage: test-fit-coreml-generated-diarizers <model.gguf> <empty cache dir> [capacities]
// `capacities` is the number of the model's diarizer templates (default 3,
// Nemotron's; Sortformer has 1). The GGUF's directory must hold no diarizer
// bundle. Never skips: every check needs Core ML generation (macOS 15, the
// generator built in).

#include "parakeet/engine.h"
#include "parakeet/fit.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>

namespace {

constexpr int kGpuLayers = 1;
constexpr uint64_t kRoomy = 64ull * 1024 * 1024 * 1024;
int g_capacities = 3;  // the model's diarizer templates (Nemotron: 358, 745 and 1125 frames)

bool g_passed = true;

void check(bool condition, const char * label) {
    std::printf("[%s] %s\n", condition ? "ok  " : "FAIL", label);
    g_passed = g_passed && condition;
}

bool contains(const std::string & text, const char * needle) { return text.find(needle) != std::string::npos; }

int count(const std::string & text, const std::string & needle) {
    int n = 0;
    for (size_t at = text.find(needle); at != std::string::npos; at = text.find(needle, at + needle.size())) ++n;
    return n;
}

// Occurrences of `needle` in the diarizer stage objects of a status JSON
// (other stages, such as an absent Sortformer encoder sidecar, are left out).
// Stage objects have sorted keys, so each starts at its "artifact_key".
int count_in_diarizers(const std::string & status, const std::string & needle) {
    const std::string start = "{\"artifact_key\"";
    int n = 0;
    for (size_t at = status.find(start); at != std::string::npos;) {
        const size_t next = status.find(start, at + 1);
        const std::string stage = status.substr(at, next == std::string::npos ? std::string::npos : next - at);
        at = next;
        if (stage.find("\"stage\":\"diarizer-") != std::string::npos) n += count(stage, needle);
    }
    return n;
}

parakeet::CoremlOptions coreml_options(const std::string & cache_dir) {
    parakeet::CoremlOptions o;
    o.cache_dir = cache_dir;
    return o;
}

// The same FitOptions the Engine's memory-degrade check builds (the default
// workload), or a workload of `audio_seconds`.
parakeet::FitResult project(const std::string & gguf, const parakeet::CoremlOptions & coreml,
                            float audio_seconds = 0) {
    parakeet::FitOptions opts;
    opts.model_gguf_path = gguf;
    opts.n_gpu_layers = kGpuLayers;
    opts.coreml = coreml;
    if (audio_seconds > 0) opts.audio_seconds = audio_seconds;
    return parakeet::fit_params(opts);
}

void print(const char * what, const parakeet::FitResult & r) {
    std::printf("%s: ggml %llu, coreml %llu (fits %d, validated %d), free %llu\n", what,
                (unsigned long long) r.ggml_required_bytes, (unsigned long long) r.coreml.bytes, r.coreml.fits,
                r.coreml.validated, (unsigned long long) r.device_free_bytes);
}

// The Engine's Core ML status, and its construction group in `construction`.
std::string engine_status(const std::string & gguf, parakeet::CoremlOptions coreml,
                          std::string * construction = nullptr) {
    parakeet::EngineOptions opts;
    opts.model_gguf_path = gguf;
    opts.n_gpu_layers = kGpuLayers;
    opts.coreml = std::move(coreml);
    parakeet::Engine engine(opts);
    if (opts.coreml.wait == parakeet::CoremlOptions::Wait::construct) engine.coreml_wait();
    if (construction != nullptr) *construction = engine.coreml_construction_json();
    return engine.coreml_status_json();
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 3 && argc != 4) {
        std::fprintf(stderr, "usage: %s <model.gguf> <empty cache dir> [capacities]\n", argv[0]);
        return 2;
    }
    if (argc == 4) g_capacities = std::atoi(argv[3]);
    if (g_capacities < 1) {
        std::fprintf(stderr, "capacities must be at least 1\n");
        return 2;
    }
    const std::string gguf = argv[1];
    const std::string cache = argv[2];
    std::error_code ec;
    std::filesystem::remove_all(cache, ec);

    const parakeet::FitResult before = project(gguf, coreml_options(cache));
    print("empty cache", before);
    check(before.status != parakeet::FitStatus::Error, "fit_params projects the GGUF");
    check(before.coreml.disabled_reason.empty(), "Core ML is enabled for the projection");
    check(before.coreml.bytes > 0 && before.coreml.validated,
          "the generated capacities are charged, validated for the resolved units");

    // prewarm = all loads every capacity at construction: the charge covers
    // all of them. The default workload (300 s, long form) reaches every
    // Nemotron capacity too: its long-form chunks and stream steps load 358
    // and 745, and a shorter call of up to 1125 frames (about 90 s) runs as
    // one pass on 1125. A model with one capacity loads it at construction.
    parakeet::CoremlOptions every = coreml_options(cache);
    every.prewarm = parakeet::CoremlOptions::Prewarm::all;
    const parakeet::FitResult all = project(gguf, every);
    print("prewarm all", all);
    check(all.coreml.bytes > 0 && all.coreml.validated, "prewarm all charges every generated capacity");
    check(before.coreml.bytes == all.coreml.bytes,
          "the default workload charges every capacity a shorter single pass can load");
    const parakeet::FitResult brief = project(gguf, coreml_options(cache), 5.0f);
    print("5 s workload", brief);
    if (g_capacities > 1) {
        check(brief.coreml.bytes > 0 && brief.coreml.bytes < all.coreml.bytes,
              "a 5 s workload charges fewer capacities");
    } else {
        check(brief.coreml.bytes == all.coreml.bytes,
              "one capacity: a 5 s workload charges the capacity construction loads");
    }

    // Generate every capacity (wait for construction, prewarm all).
    parakeet::CoremlOptions generate = coreml_options(cache);
    generate.wait = parakeet::CoremlOptions::Wait::construct;
    generate.prewarm = parakeet::CoremlOptions::Prewarm::all;
    const std::string generated = engine_status(gguf, generate);
    check(count_in_diarizers(generated, "\"route\":\"generated\"") == g_capacities &&
              count_in_diarizers(generated, "\"state\":\"ready\"") == g_capacities,
          "every capacity was generated into the cache and loaded");

    const parakeet::FitResult after = project(gguf, coreml_options(cache));
    print("after generation", after);
    check(after.coreml.bytes == before.coreml.bytes, "fit-stable: the same Core ML bytes after generation");
    check(after.coreml.validated == before.coreml.validated, "fit-stable: the same validation");
    check(after.ggml_required_bytes == before.ggml_required_bytes, "fit-stable: the same ggml-only total");
    if (after.device_free_bytes == before.device_free_bytes) {
        check(after.coreml.fits == before.coreml.fits && after.fits == before.fits,
              "fit-stable: the same verdicts at the same free memory");
    } else {
        check(after.coreml.fits ==
                  (after.ggml_required_bytes + after.coreml.bytes <= after.device_free_bytes),
              "fit-stable: the Core ML verdict follows the unchanged charge (free memory moved)");
    }

    // Units the templates were not validated for: nothing is generated or
    // loaded from the cache, and every capacity keeps an entry that says why.
    parakeet::CoremlOptions gpu = coreml_options(cache);
    gpu.compute_units = "cpu_and_gpu";
    std::string gpu_group;
    const std::string unvalidated = engine_status(gguf, gpu, &gpu_group);
    check(contains(unvalidated, "\"generation\":{\"reason\":\"unvalidated_units\",\"supported\":false}"),
          "unvalidated compute units: generation off, reason unvalidated_units");
    check(count_in_diarizers(unvalidated, "\"state\":\"unsupported(unvalidated_units)\"") == g_capacities &&
              count_in_diarizers(unvalidated, "\"route\":\"none\"") == g_capacities,
          "every capacity reports unsupported(unvalidated_units), route none");
    check(contains(gpu_group, "\"slots\":[]") && count(gpu_group, "\"stage\":\"diarizer-") == g_capacities &&
              count(gpu_group, "unsupported(unvalidated_units)") == g_capacities,
          "the construction group loads nothing and lists every capacity as skipped");

    // Generation off for the instance: every capacity is still listed, with
    // the reason.
    parakeet::CoremlOptions never = coreml_options(cache);
    never.generate = parakeet::CoremlOptions::Generate::never;
    std::string never_group;
    const std::string off = engine_status(gguf, never, &never_group);
    check(contains(off, "\"generation\":{\"reason\":\"generate=never\",\"supported\":false}") &&
              count(off, "\"stage\":\"diarizer-") == g_capacities,
          "generate = never: generation off, every capacity keeps a status entry");
    check(contains(never_group, "\"slots\":[]") && count(never_group, "\"stage\":\"diarizer-") == g_capacities,
          "generate = never: the construction group lists every capacity as skipped");

    // A budget between the two totals: Core ML off for memory, before any load.
    parakeet::CoremlOptions tight = coreml_options(cache);
    tight.memory_budget_bytes = static_cast<int64_t>(after.ggml_required_bytes + after.coreml.bytes / 2);
    const std::string degraded = engine_status(gguf, tight);
    check(contains(degraded, "\"generation\":{\"reason\":\"memory\",\"supported\":false}"),
          "a budget between the ggml-only and Core ML-inclusive totals: generation off, reason memory");
    check(!contains(degraded, "\"route\":\"generated\"") && !contains(degraded, "\"route\":\"cached\""),
          "and no generated route");

    // Just under the full charge of the default workload (above what its
    // long-form and stream calls alone load): Core ML off for memory.
    parakeet::CoremlOptions near = coreml_options(cache);
    near.memory_budget_bytes =
        static_cast<int64_t>(after.ggml_required_bytes + all.coreml.bytes - 1024 * 1024);
    check(contains(engine_status(gguf, near), "\"generation\":{\"reason\":\"memory\",\"supported\":false}"),
          "a budget 1 MiB under the three-capacity charge: generation off, reason memory");

    // Room for both: generation stays on (the cached minimal capacity loads
    // before the engine goes away).
    parakeet::CoremlOptions roomy = coreml_options(cache);
    roomy.wait = parakeet::CoremlOptions::Wait::construct;
    roomy.memory_budget_bytes = static_cast<int64_t>(after.ggml_required_bytes + after.coreml.bytes + kRoomy);
    const std::string kept = engine_status(gguf, roomy);
    check(contains(kept, "\"generation\":{\"reason\":null,\"supported\":true}"),
          "a budget with room for Core ML keeps generation on");

    std::filesystem::remove_all(cache, ec);
    std::printf("%s\n", g_passed ? "PASS" : "FAIL");
    return g_passed ? 0 : 1;
}
