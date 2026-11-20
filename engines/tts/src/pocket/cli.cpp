#include "tts-cpp/pocket/engine.h"
#include "tts-cpp/pocket/fit.h"
#include "fit_util.h"
#define DR_WAV_IMPLEMENTATION
#define DRWAV_API static
#define DRWAV_PRIVATE static
#include "dr_wav.h"
#include "json.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
using namespace tts_cpp::pocket;
using Clock = std::chrono::steady_clock;
using Json = nlohmann::json;
struct Arguments {
    std::map<std::string, std::string> values;
    bool has(const std::string & key) const { return values.count(key) != 0; }
    std::string get(const std::string & key, const std::string & fallback) const {
        const auto it = values.find(key); return it == values.end() ? fallback : it->second;
    }
    int integer(const std::string & key, int fallback) const {
        const auto s = get(key, std::to_string(fallback)); size_t end;
        const int n = std::stoi(s, &end);
        if (end != s.size()) throw std::runtime_error("invalid integer: "+key);
        return n;
    }
    uint64_t mib(const std::string & key, int fallback) const {
        const int value = integer(key, fallback);
        if (value < 0) throw std::runtime_error("memory size must be nonnegative: "+key);
        return tts_cpp::fitutil::margin_mib_to_bytes(uint64_t(value));
    }
};
Arguments parse_arguments(int argc, char ** argv) {
    Arguments args;
    const std::set<std::string> accepted = {"--model-dir","--flow-lm","--mimi","--frontend","--voice","--reference-audio",
        "--text","--output","--threads","--steps","--temperature","--seed","--runs","--report","--context","--max-tokens",
        "--frames-after-eos","--fit-budget-mib","--fit-margin-mib","--backends-dir","--backend",
        "--coreml-cache-dir","--coreml-wait","--coreml-prewarm","--coreml-step-units","--coreml-decoder-units",
        "--coreml-status-json"};
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help") { args.values[key] = "true"; return args; }
        if (key == "--fit" || key == "--gpu") {
            if (args.has(key)) throw std::runtime_error("duplicate option: "+key);
            args.values[key] = "true"; continue;
        }
        if (!accepted.count(key) || i+1 == argc || args.has(key)) throw std::runtime_error("invalid or duplicate option: "+key);
        args.values[key] = argv[++i];
    }
    return args;
}
void print_help() {
    std::puts("pocket-cli --model-dir DIR --text TEXT --output FILE.wav [--voice FILE.gguf | --reference-audio FILE.wav]\n"
              "           [--threads 1 --steps 1 --temperature 0.3 --seed 1234 --runs 1 --report FILE.json]\n"
              "           [--gpu] [--backends-dir DIR] [--frames-after-eos -1]\n"
              "--gpu selects Metal, CUDA, Vulkan or OpenCL for FlowLM and Mimi; CPU is the default.\n"
              "Core ML sidecars next to the GGUFs (flow-lm-step.mlmodelc, mimi-decoder.mlmodelc) run the generation\n"
              "steps and Mimi decoding; POCKET_COREML_DISABLE, POCKET_COREML_STRICT and POCKET_COREML_COMPUTE_UNITS\n"
              "set their policy.\n"
              "--backend cpu|opencl|webgpu|hexagon|HTP0 explicitly selects a backend; unavailable backends are errors.\n"
              "--frames-after-eos -1 uses the checkpoint/text default; explicit range 0..100.\n"
              "DIR contains flow-lm.gguf, mimi.gguf, frontend.json, voice.gguf. Individual asset paths can override DIR.\n"
              "--runs greater than 1 includes one untimed warmup and records each warm run.\n"
              "--fit estimates memory on the selected backend without loading weights; optional --fit-budget-mib and --fit-margin-mib.\n"
              "  Synthesis uses the same budget, text and margin to decide whether Core ML fits (memory-degrade).\n"
              "--coreml-cache-dir DIR generates the Core ML sidecars from the GGUFs into DIR when none is prebuilt (macOS 15+);\n"
              "  default: QVAC_COREML_CACHE_DIR, else ~/Library/Application Support/qvac-speech/coreml;\n"
              "  --coreml-wait none|construct|call, --coreml-prewarm none|minimal|predicted|all,\n"
              "  --coreml-step-units / --coreml-decoder-units all|cpu_and_ane|cpu_and_gpu|cpu_only,\n"
              "  --coreml-status-json FILE writes the Core ML status after the runs.\n"
              "Fit exit status: 0 fits, 1 does not fit, 2 measurement error.");
}
// The CLI keeps generated sidecars and warm-load hints under Application
// Support when neither --coreml-cache-dir nor QVAC_COREML_CACHE_DIR names a
// directory (Library/Caches can be purged, and a purge means a new path and a
// cold compile). The library's own default stays "no cache".
void apply_cli_cache_default(tts_cpp::CoremlOptions & coreml) {
#if defined(__APPLE__)
    const char * shared = std::getenv("QVAC_COREML_CACHE_DIR");
    const char * home = std::getenv("HOME");
    if (!coreml.cache_dir && (shared == nullptr || *shared == '\0') && home != nullptr && *home != '\0')
        coreml.cache_dir = std::string(home) + "/Library/Application Support/qvac-speech/coreml";
#else
    (void) coreml;
#endif
}
EngineOptions engine_options(const Arguments & args, const std::string & text) {
    const auto dir = args.get("--model-dir", ".");
    EngineOptions opts;
    // The fit that assesses this run (--fit) and the Engine's own
    // memory-degrade decision use one budget, text and margin.
    const uint64_t budget = args.mib("--fit-budget-mib", 0);
    if (budget) opts.coreml.memory_budget_bytes = int64_t(budget);
    opts.fit_text = text;
    opts.fit_margin_bytes = args.mib("--fit-margin-mib", 256);
    opts.use_gpu = args.has("--gpu");
    opts.backends_dir = args.get("--backends-dir", "");
    opts.backend = args.get("--backend", "");
    opts.flow_lm_path = args.get("--flow-lm", dir+"/flow-lm.gguf");
    opts.mimi_path = args.get("--mimi", dir+"/mimi.gguf");
    opts.frontend_path = args.get("--frontend", dir+"/frontend.json");
    opts.reference_audio_path = args.get("--reference-audio", "");
    opts.voice_path = args.get("--voice", opts.reference_audio_path.empty() ? dir+"/voice.gguf" : "");
    opts.n_threads = args.integer("--threads", opts.n_threads); opts.steps = args.integer("--steps", opts.steps);
    opts.context = args.integer("--context", opts.context); opts.max_tokens = args.integer("--max-tokens", opts.max_tokens);
    opts.frames_after_eos = args.integer("--frames-after-eos", opts.frames_after_eos);
    if (args.has("--coreml-cache-dir")) opts.coreml.cache_dir = args.get("--coreml-cache-dir", "");
    apply_cli_cache_default(opts.coreml);
    if (args.has("--coreml-wait")) {
        const auto w = args.get("--coreml-wait", "");
        if (w == "none") opts.coreml.wait = tts_cpp::CoremlOptions::Wait::none;
        else if (w == "construct") opts.coreml.wait = tts_cpp::CoremlOptions::Wait::construct;
        else if (w == "call") opts.coreml.wait = tts_cpp::CoremlOptions::Wait::call;
        else throw std::runtime_error("--coreml-wait must be none, construct or call");
    }
    if (args.has("--coreml-prewarm")) {
        const auto w = args.get("--coreml-prewarm", "");
        if (w == "none") opts.coreml.prewarm = tts_cpp::CoremlOptions::Prewarm::none;
        else if (w == "minimal") opts.coreml.prewarm = tts_cpp::CoremlOptions::Prewarm::minimal;
        else if (w == "predicted") opts.coreml.prewarm = tts_cpp::CoremlOptions::Prewarm::predicted;
        else if (w == "all") opts.coreml.prewarm = tts_cpp::CoremlOptions::Prewarm::all;
        else throw std::runtime_error("--coreml-prewarm must be none, minimal, predicted or all");
    }
    opts.coreml_step_units = args.get("--coreml-step-units", "");
    opts.coreml_decoder_units = args.get("--coreml-decoder-units", "");
    const int seed = args.integer("--seed", opts.seed);
    if (seed < 0) throw std::runtime_error("seed must be nonnegative");
    opts.seed = seed;
    size_t end; const auto temperature = args.get("--temperature", std::to_string(opts.temperature));
    opts.temperature = std::stof(temperature, &end);
    if (end != temperature.size()) throw std::runtime_error("invalid temperature");
    return opts;
}
void write_report(const std::string & path, const Json & report) {
    if (path.empty()) return;
    std::ofstream file(path); file << report.dump(2) << '\n';
    if (!file) throw std::runtime_error("cannot write report");
}
int run_fit(const Arguments & args, const EngineOptions & opts, const std::string & text) {
    FitOptions fit; fit.engine = opts; fit.text = text;
    fit.margin_bytes = opts.fit_margin_bytes;
    fit.memory_budget_bytes = args.mib("--fit-budget-mib", 0);
    const auto result = fit_params(fit);
    std::puts(result.report.c_str());
    write_report(args.get("--report", ""), {
        {"status", int(result.status)}, {"fits", result.fits}, {"reason", result.reason},
        {"backend", result.device_name}, {"device_is_cpu", result.device_is_cpu},
        {"device_shares_host_memory", result.device_shares_host_memory},
        {"weights_bytes", result.device.weights_bytes}, {"state_bytes", result.device.state_bytes},
        {"lm_compute_bytes", result.device.lm_compute_bytes}, {"codec_compute_bytes", result.device.codec_compute_bytes},
        {"device_bytes", result.device.total_bytes}, {"host_bytes", result.host_bytes},
        {"available_bytes", result.device_free_bytes}, {"total_memory_bytes", result.device_total_bytes},
        {"margin_bytes", fit.margin_bytes}, {"budget_bytes", fit.memory_budget_bytes},
        {"coreml", {{"route", result.coreml.route}, {"bytes", result.coreml.bytes}, {"fits", result.coreml.fits},
                    {"validated", result.coreml.validated}, {"disabled_reason", result.coreml.disabled_reason},
                    {"ggml_only_fits", result.coreml.ggml_only_fits}}},
        {"report", result.report}});
    return int(result.status);
}
void write_wav(const std::string & path, const SynthesisResult & result) {
    drwav_data_format format{drwav_container_riff, DR_WAVE_FORMAT_IEEE_FLOAT, 1, drwav_uint32(result.sample_rate), 32};
    drwav wav;
    if (!drwav_init_file_write(&wav, path.c_str(), &format, nullptr)) throw std::runtime_error("cannot open output WAV");
    const auto written = drwav_write_pcm_frames(&wav, result.pcm.size(), result.pcm.data());
    drwav_uninit(&wav);
    if (written != result.pcm.size()) throw std::runtime_error("incomplete WAV write");
}
// Whether two paths name the same file: the same absolute path once
// normalized and symlinks resolved, or (both existing) the same file, which
// also catches spellings only the filesystem equates (a case-insensitive
// volume, a hard link).
bool same_file(const std::string & a, const std::string & b) {
    namespace fs = std::filesystem;
    if (a.empty() || b.empty()) return false;
    std::error_code ec;
    const fs::path pa = fs::weakly_canonical(fs::absolute(a, ec), ec);
    if (!ec) {
        const fs::path pb = fs::weakly_canonical(fs::absolute(b, ec), ec);
        if (!ec && pa == pb) return true;
    }
    ec.clear();
    return fs::exists(a, ec) && fs::exists(b, ec) && fs::equivalent(a, b, ec);
}
// The files --coreml-status-json must never replace: the model inputs, the
// audio and the report.
std::vector<std::pair<std::string, std::string>> protected_files(const Arguments & args, const EngineOptions & opts) {
    return {{"--output", args.get("--output", "pocket-output.wav")}, {"--report", args.get("--report", "")},
            {"--flow-lm", opts.flow_lm_path}, {"--mimi", opts.mimi_path}, {"--frontend", opts.frontend_path},
            {"--voice", opts.voice_path}, {"--reference-audio", opts.reference_audio_path}};
}
std::filesystem::path resolved(const std::string & p) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path r = fs::weakly_canonical(fs::absolute(p, ec), ec);
    return ec ? fs::path() : r;
}
bool is_within(const std::filesystem::path & path, const std::filesystem::path & dir) {
    if (path.empty() || dir.empty()) return false;
    auto p = path.begin();
    for (auto d = dir.begin(); d != dir.end(); ++d, ++p) {
        if (d->empty()) continue;  // a trailing separator
        if (p == path.end() || *p != *d) return false;
    }
    return true;
}
bool is_compiled_bundle(const std::filesystem::path & p) {
    const auto ext = p.extension().string();
    return ext == ".mlmodelc" || ext == ".mlpackage";
}
// The Core ML files --coreml-status-json must never replace, as directories:
// every compiled sidecar next to the FlowLM and Mimi GGUFs (prebuilt, used or
// not), the Core ML cache directory (generated sidecars and warm hints) and,
// with an engine, each stage's resolved sidecar path from its status.
std::vector<std::pair<std::string, std::filesystem::path>> protected_coreml_dirs(const EngineOptions & opts,
                                                                                 const Engine * engine) {
    namespace fs = std::filesystem;
    std::vector<std::pair<std::string, fs::path>> dirs;
    std::error_code ec;
    for (const std::string & gguf : {opts.flow_lm_path, opts.mimi_path}) {
        // The directory the GGUF is named in (the Engine looks for sidecars
        // there, even when the GGUF itself is a symlink).
        const fs::path parent = gguf.empty() ? fs::path() : resolved(fs::absolute(gguf, ec).parent_path().string());
        ec.clear();
        if (parent.empty()) continue;
        for (fs::directory_iterator it(parent, ec), end; !ec && it != end; it.increment(ec)) {
            if (is_compiled_bundle(it->path())) dirs.push_back({"a Core ML sidecar", resolved(it->path().string())});
        }
        ec.clear();
    }
    std::string cache = opts.coreml.cache_dir.value_or("");
    if (cache.empty()) {
        const char * shared = std::getenv("QVAC_COREML_CACHE_DIR");
        if (shared != nullptr) cache = shared;
    }
    if (engine != nullptr) {
        const Json status = Json::parse(engine->coreml_status(), nullptr, false);
        if (status.is_object()) {
            if (status.contains("cache_dir") && status["cache_dir"].is_string() && cache.empty())
                cache = status["cache_dir"].get<std::string>();
            if (status.contains("stages") && status["stages"].is_array()) {
                for (const Json & stage : status["stages"]) {
                    if (stage.contains("path") && stage["path"].is_string())
                        dirs.push_back({"a Core ML sidecar", resolved(stage["path"].get<std::string>())});
                }
            }
        }
    }
    if (!cache.empty()) dirs.push_back({"the Core ML cache", resolved(cache)});
    return dirs;
}
void check_status_path(const Arguments & args, const EngineOptions & opts, const std::string & status_path,
                       const Engine * engine = nullptr) {
    namespace fs = std::filesystem;
    for (const auto & [name, path] : protected_files(args, opts)) {
        if (same_file(status_path, path))
            throw std::runtime_error("--coreml-status-json names the same file as " + name + " (" + path + ")");
    }
    // Inside a compiled bundle anywhere, or inside a protected Core ML
    // directory (after symlinks resolve), or another name of one of its files
    // (a hard link): refused, since opening the status truncates it.
    const fs::path target = resolved(status_path);
    for (fs::path p = target; !p.empty() && p != p.parent_path(); p = p.parent_path()) {
        if (is_compiled_bundle(p))
            throw std::runtime_error("--coreml-status-json is inside a compiled Core ML model (" + p.string() + ")");
    }
    std::error_code ec;
    const bool exists = fs::exists(status_path, ec);
    for (const auto & [what, dir] : protected_coreml_dirs(opts, engine)) {
        if (dir.empty()) continue;
        if (is_within(target, dir))
            throw std::runtime_error("--coreml-status-json is inside " + what + " (" + dir.string() + ")");
        if (!exists) continue;
        ec.clear();
        if (fs::is_regular_file(dir, ec) && fs::equivalent(status_path, dir, ec))
            throw std::runtime_error("--coreml-status-json names a file of " + what + " (" + dir.string() + ")");
        ec.clear();
        for (fs::recursive_directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
            std::error_code fe;
            if (it->is_regular_file(fe) && fs::equivalent(status_path, it->path(), fe))
                throw std::runtime_error("--coreml-status-json names a file of " + what + " (" + it->path().string() +
                                         ")");
        }
    }
}
double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now()-start).count();
}
Json benchmark_once(Engine & engine, const std::string & text, const std::string & output) {
    const auto start = Clock::now();
    std::vector<float> pcm; double first_audio = 0;
    auto result = engine.synthesize_stream(text, [&](const float * chunk, size_t n, int) {
        if (pcm.empty()) first_audio = seconds_since(start);
        pcm.insert(pcm.end(), chunk, chunk+n); return true;
    });
    result.pcm = std::move(pcm);
    const double elapsed = seconds_since(start);
    const double duration = double(result.pcm.size())/result.sample_rate;
    Json measurement = {{"seconds",elapsed},{"first_audio_seconds",first_audio},
                        {"audio_seconds",duration},{"real_time_factor",elapsed/duration},
                        {"flow_lm_backend",result.flow_lm_backend},{"mimi_backend",result.mimi_backend},
                        {"coreml_pending",result.coreml_pending}};
    std::puts(measurement.dump().c_str());
    if (!output.empty()) write_wav(output, result);
    return measurement;
}
Json benchmark_runs(Engine & engine, const std::string & text, const std::string & output, int runs) {
    Json measurements = Json::array();
    for (int i = 0; i < runs; ++i) measurements.push_back(benchmark_once(engine, text, i == 0 ? output : ""));
    return measurements;
}
int run_synthesis(const Arguments & args, const EngineOptions & opts, const std::string & text) {
    const int runs = args.integer("--runs", 1);
    if (runs < 1 || runs > 100) throw std::runtime_error("runs must be 1..100");
    // Refused before any synthesis, so no audio, report or input is ever
    // overwritten by the status.
    const auto status_path = args.get("--coreml-status-json", "");
    if (!status_path.empty()) check_status_path(args, opts, status_path);
    const auto start = Clock::now(); Engine engine(opts);
    const double load = seconds_since(start);
    // Again with the stages' resolved sidecars (generated ones in the cache).
    if (!status_path.empty()) check_status_path(args, opts, status_path, &engine);
    std::fprintf(stderr, "Pocket FlowLM: %s%s; Mimi: %s%s; device: %s\n", engine.flow_lm_backend_name(),
                 engine.flow_lm_on_coreml() ? " (generation steps on the Core ML sidecar)" : "", engine.mimi_backend_name(),
                 engine.mimi_on_coreml() ? " (decoding on the Core ML sidecar)" : "", engine.device_name());
    if (runs > 1) engine.synthesize("A short warm up.");
    const auto measurements = benchmark_runs(engine, text, args.get("--output", "pocket-output.wav"), runs);
    write_report(args.get("--report", ""), {
        {"implementation","Fabric native ggml"},{"backend",engine.backend_name()},
        {"flow_lm_backend",engine.flow_lm_backend_name()},{"mimi_backend",engine.mimi_backend_name()},
        {"device",engine.device_name()},
        {"threads_per_worker",opts.n_threads},{"worker_threads",2},{"steps",opts.steps},{"temperature",opts.temperature},
        {"frames_after_eos",opts.frames_after_eos},
        {"seed",opts.seed},{"text",text},{"load_seconds",load},{"warmups",runs>1?1:0},{"runs",measurements},
        {"source_sha256",engine.model_source_sha256()},{"sample_rate",engine.sample_rate()},
        {"compute_precision",engine.flow_lm_on_coreml() || engine.mimi_on_coreml() ?
            "float32 (Core ML sidecars: float16)" : "float32"},{"rng","Fabric MT19937 Box-Muller (not PyTorch seed-equivalent)"},
        {"assets",{{"flow_lm",opts.flow_lm_path},{"mimi",opts.mimi_path},{"frontend",opts.frontend_path},{"voice",opts.voice_path}}},
        {"coreml_status",Json::parse(engine.coreml_status())}});
    if (!status_path.empty()) {
        // Checked again now that the audio and report exist: a spelling only
        // the filesystem equates with one of them is caught here, before the
        // status file is opened (and truncated).
        check_status_path(args, opts, status_path, &engine);
        std::ofstream file(status_path); file << engine.coreml_status() << '\n';
        if (!file) throw std::runtime_error("cannot write the Core ML status");
    }
    return 0;
}
} // namespace
int main(int argc, char ** argv) {
    try {
        const auto args = parse_arguments(argc, argv);
        if (args.has("--help")) { print_help(); return 0; }
        const auto text = args.get("--text", "");
        if (text.empty()) throw std::runtime_error("--text is required");
        const auto opts = engine_options(args, text);
        return args.has("--fit") ? run_fit(args, opts, text) : run_synthesis(args, opts, text);
    } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
