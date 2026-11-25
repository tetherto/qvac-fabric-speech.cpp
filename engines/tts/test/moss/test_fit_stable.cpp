// The MOSS generated routes and the SoundEffect VAE's Core ML fit do not
// depend on the cache.
//
//   test-moss-fit-stable [--vae GGUF] [--tokenizer GGUF] [--cache-dir DIR]
//
// Each stage's route (prebuilt | generate | none) is decided from the
// options, OS, build, template match and an adjacent bundle's presence, never
// from the cache, so it is the same before and after the first generation.
// This checks plan_generated_route for each stage given and the VAE's Core ML
// fit (sfx_vae_coreml_fit, the part of fit_params a generated VAE adds) on an
// empty cache directory against a cache holding what a generation leaves
// behind, then the same cache with non-stale warm hints and with stale ones,
// written where the runtime looks for them. It also pins the template
// charges of each generated route under cpu_and_gpu.
//
// Without --cache-dir the generation is simulated in a scratch cache: an
// artifact per template with its memo and used marker. With --cache-dir DIR
// (a cache test-moss-coreml-generated populated, whose stages sit in
// tokenizer/generated and vae/generated) DIR is compared as it is, then
// copied to scratch for the hint cases.
//
// The GGUFs may be twins of the templates' reference manifests or the real
// stage GGUFs.

#include "../test_env_portable.h"

#include "moss/coreml_generation.h"

#ifdef TTS_CPP_USE_COREML
#include "hints.h"
#include "qvac_coreml/generation.h"
#include "qvac_coreml/manager.h"
#endif
#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

using namespace tts_cpp::moss::detail;
namespace fs = std::filesystem;

namespace {

int failures = 0;

void check(bool ok, const std::string & what) {
    std::fprintf(stderr, "  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++failures;
}

struct Stage {
    GeneratedStage stage;
    const char * name;
    const char * template_id;
    const char * cache_subdir;  // where test-moss-coreml-generated keeps it
    std::string gguf;
};

std::string describe(const GeneratedRoute & r) {
    return r.route + "|" + r.state + "|" + r.compute_units + "|" + std::to_string(r.envelope_bytes) + "|" +
           (r.validated ? "validated" : "unvalidated");
}

// The VAE's Core ML fit with a fixed ggml projection, a roomy budget and a
// tight memory budget, so a change of route, envelope or degrade shows.
std::string describe_fit(const std::string & gguf, tts_cpp::CoremlOptions options) {
    constexpr uint64_t GGML_BYTES = 64ull << 20;
    constexpr uint64_t FREE_BYTES = 4ull << 30;
    std::string out;
    for (const bool tight : {false, true}) {
        if (tight) options.memory_budget_bytes = (int64_t) GGML_BYTES;
        const tts_cpp::FitCoreml f = sfx_vae_coreml_fit(gguf, options, GGML_BYTES, FREE_BYTES, true);
        out += f.route + "|" + std::to_string(f.bytes) + "|" + (f.fits ? "fits" : "no") + "|" +
               (f.ggml_only_fits ? "ggml-fits" : "ggml-no") + "|" + (f.validated ? "validated" : "unvalidated") +
               "|" + f.disabled_reason + ";";
    }
    return out;
}

void write_file(const fs::path & p, const std::string & text) {
    fs::create_directories(p.parent_path());
    std::ofstream(p) << text;
}

std::vector<fs::path> artifact_dirs(const fs::path & cache) {
    std::vector<fs::path> out;
    std::error_code ec;
    for (const auto & e : fs::directory_iterator(cache / "qvac-coreml-v1" / "artifacts", ec)) {
        if (e.is_directory()) out.push_back(e.path());
    }
    return out;
}

#ifdef TTS_CPP_USE_COREML
namespace pcm = QVAC_COREML_NS;

std::string runtime_hint_path(const fs::path & cache, const fs::path & artifact_dir, const std::string & units,
                              const std::string & host_id = {}) {
    const auto m = pcm::Manager::instance();
    return pcm::detail::hint_path_for(cache.string(), host_id.empty() ? m->host_id() : host_id,
                                      (artifact_dir / "model.mlmodelc").string(), "main", units, m->os_build());
}

// Rewrites every hint record under `cache` with `stale` and writes one at the
// runtime's id for every artifact; returns how many read back.
int write_hints(const fs::path & cache, const std::string & units, bool stale) {
    const std::optional<int64_t> until =
        stale ? std::optional<int64_t>(pcm::detail::unix_now_s() + 365LL * 24 * 3600) : std::nullopt;
    std::error_code ec;
    for (const auto & e : fs::directory_iterator(cache / "qvac-coreml-v1" / "hints", ec)) {
        if (e.path().extension() != ".json") continue;
        std::optional<pcm::detail::HintRecord> rec = pcm::detail::read_hint(e.path().string());
        if (!rec) continue;
        rec->stale_until_s = until;
        pcm::detail::write_hint(e.path().string(), *rec);
    }
    int written = 0;
    for (const fs::path & art : artifact_dirs(cache)) {
        const std::string path = runtime_hint_path(cache, art, units);
        pcm::detail::HintRecord rec;
        rec.warm_load_ms = 120;
        rec.cold_load_ms = 8000;
        rec.last_ok = pcm::detail::iso8601_utc(pcm::detail::unix_now_s());
        rec.stale_until_s = until;
        rec.artifact = (art / "model.mlmodelc").string();
        fs::create_directories(fs::path(path).parent_path());
        const std::optional<pcm::detail::HintRecord> back =
            pcm::detail::write_hint(path, rec) ? pcm::detail::read_hint(path) : std::nullopt;
        if (back && back->stale_until_s.has_value() == stale) ++written;
    }
    return written;
}
#endif

// These pins are the cpu_and_gpu charges from the two templates. The
// envelope is resident cost plus the larger of prepare and load/warm-up.
struct Charges {
    uint64_t prepare, load, resident, envelope;
};
constexpr Charges TOKENIZER_CHARGES{122683392, 1011875840, 999348800, 999348800ULL + 1011875840};
constexpr Charges VAE_CHARGES{542113792, 1321205760, 1369792256, 1369792256ULL + 1321205760};

Charges expected_charges(GeneratedStage stage) {
    return stage == GeneratedStage::speech_tokenizer ? TOKENIZER_CHARGES : VAE_CHARGES;
}

void template_charges(const Stage & s, const GeneratedRoute & r) {
#if defined(TTS_CPP_USE_COREML) && defined(__APPLE__) && TARGET_OS_OSX
    // A prebuilt-only Core ML build or macOS before 15 generates nothing.
    const bool generator = pcm::generation::built();
    const bool os_supported = pcm::os_at_least(pcm::Manager::instance()->os_version(), 15, 0);
    if (!generator || !os_supported) {
        check(r.route == "none", std::string(s.name) + ": no generator or macOS before 15, so route none (" +
                                     r.route + " " + r.state + ")");
        return;
    }
    if (r.route != "generate" || r.compute_units != "cpu_and_gpu") {
        check(false, std::string(s.name) + ": route " + r.route + " (" + r.state + "), units " + r.compute_units +
                         "; expected generate under cpu_and_gpu");
        return;
    }
#else
    if (r.route != "generate") {
        std::fprintf(stderr, "%s: template charges not checked (route %s %s)\n", s.name, r.route.c_str(),
                     r.state.c_str());
        return;
    }
#endif
    const Charges c = expected_charges(s.stage);
    check(r.prepare_peak_bytes == c.prepare && r.load_peak_bytes == c.load && r.resident_bytes == c.resident &&
              r.ane_compile_reserve_bytes == 0,
          std::string(s.name) + ": template charges: prepare " + std::to_string(r.prepare_peak_bytes) + ", load " +
              std::to_string(r.load_peak_bytes) + ", resident " + std::to_string(r.resident_bytes) +
              ", no compiler reserve on the GPU");
    check(r.envelope_bytes == c.envelope && r.validated,
          std::string(s.name) + ": envelope " + std::to_string(r.envelope_bytes) + " bytes (expected " +
              std::to_string(c.envelope) + ", margin 1.0), validated for cpu_and_gpu");
}

}  // namespace

int main(int argc, char ** argv) {
    std::vector<Stage> stages;
    std::string given_cache;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--vae" && i + 1 < argc) {
            stages.push_back({GeneratedStage::sfx_vae, "VAE", "moss-sfx-vae-w320", "vae", argv[++i]});
        } else if (a == "--tokenizer" && i + 1 < argc) {
            stages.push_back(
                {GeneratedStage::speech_tokenizer, "tokenizer", "moss-speech-tokenizer-30s", "tokenizer", argv[++i]});
        } else if (a == "--cache-dir" && i + 1 < argc) {
            given_cache = argv[++i];
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
            return 2;
        }
    }
    if (stages.empty()) {
        std::fprintf(stderr, "SKIP: pass --vae and/or --tokenizer\n");
        return 77;
    }
    for (const Stage & s : stages) {
        if (!std::ifstream(s.gguf).good()) {
            std::fprintf(stderr, "cannot read %s\n", s.gguf.c_str());
            return 1;
        }
    }
    unsetenv("MOSS_COREML_DISABLE");
    unsetenv("MOSS_COREML_STRICT");
    unsetenv("MOSS_COREML_COMPUTE_UNITS");
    unsetenv("QVAC_COREML_CACHE_DIR");

    const fs::path scratch =
        fs::temp_directory_path() /
        ("moss-fit-stable-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    for (const Stage & s : stages) {
        const auto options_at = [](const fs::path & cache) {
            tts_cpp::CoremlOptions o;
            o.cache_dir = cache.string();
            return o;
        };
        const auto state_at = [&](const fs::path & cache) {
            std::string out = describe(plan_generated_route(s.stage, s.gguf, options_at(cache)));
            if (s.stage == GeneratedStage::sfx_vae) out += " fit " + describe_fit(s.gguf, options_at(cache));
            return out;
        };
        const fs::path empty = scratch / s.cache_subdir / "empty";
        fs::create_directories(empty);
        const GeneratedRoute first = plan_generated_route(s.stage, s.gguf, options_at(empty));
        template_charges(s, first);
        const std::string state0 = state_at(empty);
        std::fprintf(stderr, "%s on an empty cache: %s\n", s.name, state0.c_str());
        const auto same = [&](const fs::path & cache, const std::string & what) {
            const std::string now = state_at(cache);
            check(now == state0, std::string(s.name) + ": " + what + ": " + now);
        };

        fs::path cache;
        if (given_cache.empty()) {
            cache = scratch / s.cache_subdir / "cache";
            const std::string key = "0123456789abcdef0123456789abcdef";
            const fs::path root = cache / "qvac-coreml-v1";
            const fs::path art = root / "artifacts" / (std::string(s.template_id) + "." + key);
            write_file(art / "model.mlmodelc" / "model.mil", "program(1.0)\n");
            write_file(art / "artifact.json", "{\"schema\":1,\"complete\":true,\"artifact_key\":\"" + key +
                                                  "\",\"inventory\":[{\"path\":\"model.mlmodelc/model.mil\","
                                                  "\"bytes\":13}]}\n");
            write_file(root / "sources" / ("fedcba9876543210fedcba9876543210." + key + ".json"), "{\"schema\":1}\n");
            write_file(root / "used" / key, "");
            same(cache, "after a generation (artifact, memo)");
        } else {
            const fs::path populated = fs::path(given_cache) / s.cache_subdir / "generated";
            check(!artifact_dirs(populated).empty(), std::string(s.name) + ": " + populated.string() +
                                                         " holds the generated artifact");
            same(populated, "the populated cache " + populated.string());
            cache = scratch / s.cache_subdir / "copy";
            std::error_code cec;
            fs::copy(populated, cache, fs::copy_options::recursive | fs::copy_options::copy_symlinks, cec);
            check(!cec, std::string(s.name) + ": copied the populated cache" + (cec ? ": " + cec.message() : ""));
            same(cache, "a copy of the populated cache");
        }
#ifdef TTS_CPP_USE_COREML
        const size_t n = artifact_dirs(cache).size();
        int written = write_hints(cache, "cpu_and_gpu", false);
        check(n > 0 && written == (int) n,
              std::string(s.name) + ": non-stale hints at the runtime's id for " + std::to_string(written) + " of " +
                  std::to_string(n) + " artifact(s)");
        same(cache, "with non-stale hints");
        written = write_hints(cache, "cpu_and_gpu", true);
        check(n > 0 && written == (int) n, std::string(s.name) + ": stale hints at the runtime's id for " +
                                               std::to_string(written) + " of " + std::to_string(n) +
                                               " artifact(s)");
        same(cache, "with stale hints");
#else
        std::fprintf(stderr, "built without TTS_CPP_COREML: hint cases skipped\n");
#endif
    }
    std::error_code ec;
    fs::remove_all(scratch, ec);
    std::fprintf(stderr, failures ? "FAILED (%d)\n" : "OK\n", failures);
    return failures ? 1 : 0;
}
