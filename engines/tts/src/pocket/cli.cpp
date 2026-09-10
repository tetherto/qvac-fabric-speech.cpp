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
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>

int main(int argc, char ** argv) {
    try {
        std::map<std::string,std::string> args;
        const std::set<std::string> accepted = {"--model-dir","--flow-lm","--mimi","--frontend","--voice","--reference-audio",
            "--text","--output","--threads","--steps","--temperature","--seed","--runs","--report","--context","--max-tokens","--fit-budget-mib","--fit-margin-mib"};
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--fit") {
                if (args.count(key)) throw std::runtime_error("duplicate option: --fit");
                args[key] = "true"; continue;
            }
            if (key == "--help") {
                std::puts("pocket-cli --model-dir DIR --text TEXT --output FILE.wav [--voice FILE.gguf | --reference-audio FILE.wav]\n"
                          "           [--threads 1 --steps 1 --temperature 0.3 --seed 1234 --runs 1 --report FILE.json]\n"
                          "DIR contains flow-lm.gguf, mimi.gguf, frontend.json, voice.gguf. Individual asset paths can override DIR.\n"
                          "--runs greater than 1 includes one untimed warmup and records each warm run.\n"
                          "--fit estimates CPU memory without loading weights; optional --fit-budget-mib and --fit-margin-mib.\n"
                          "Fit exit status: 0 fits, 1 does not fit, 2 measurement error."); return 0;
            }
            if (!accepted.count(key) || i+1 == argc || args.count(key)) throw std::runtime_error("invalid or duplicate option: "+key);
            args[key] = argv[++i];
        }
        auto get = [&](const std::string & k, const std::string & fallback) { const auto it = args.find(k); return it == args.end() ? fallback : it->second; };
        auto integer = [&](const std::string & key, int fallback) {
            const auto s = get(key, std::to_string(fallback)); size_t end; int n = std::stoi(s, &end);
            if (end != s.size()) throw std::runtime_error("invalid integer: "+key); return n;
        };
        const auto dir = get("--model-dir", ".");
        tts_cpp::pocket::EngineOptions opts;
        opts.flow_lm_path = get("--flow-lm", dir+"/flow-lm.gguf"); opts.mimi_path = get("--mimi", dir+"/mimi.gguf");
        opts.frontend_path = get("--frontend", dir+"/frontend.json");
        opts.reference_audio_path = get("--reference-audio", "");
        opts.voice_path = get("--voice", opts.reference_audio_path.empty() ? dir+"/voice.gguf" : "");
        opts.n_threads = integer("--threads", 1); opts.steps = integer("--steps", 1);
        opts.context = integer("--context", 2048); opts.max_tokens = integer("--max-tokens", 50);
        const int seed = integer("--seed", 1234); if (seed < 0) throw std::runtime_error("seed must be nonnegative"); opts.seed = seed;
        size_t end; const auto temperature = get("--temperature", "0.3"); opts.temperature = std::stof(temperature, &end);
        if (end != temperature.size()) throw std::runtime_error("invalid temperature");
        const auto text = get("--text", ""); if (text.empty()) throw std::runtime_error("--text is required");
        if (args.count("--fit")) {
            tts_cpp::pocket::FitOptions fit; fit.engine = opts; fit.text = text;
            auto mib = [&](const std::string & key, uint64_t fallback) {
                const auto value = integer(key, int(fallback));
                if (value < 0) throw std::runtime_error("memory size must be nonnegative: "+key);
                return tts_cpp::fitutil::margin_mib_to_bytes(uint64_t(value));
            };
            fit.margin_bytes = mib("--fit-margin-mib", 256);
            fit.memory_budget_bytes = mib("--fit-budget-mib", 0);
            const auto result = tts_cpp::pocket::fit_params(fit);
            std::puts(result.report.c_str());
            const auto report = get("--report", "");
            if (!report.empty()) {
                nlohmann::json j = {{"status", int(result.status)}, {"fits", result.fits}, {"reason", result.reason},
                    {"weights_bytes", result.device.weights_bytes}, {"state_bytes", result.device.state_bytes},
                    {"lm_compute_bytes", result.device.lm_compute_bytes}, {"codec_compute_bytes", result.device.codec_compute_bytes},
                    {"device_bytes", result.device.total_bytes}, {"host_bytes", result.host_bytes},
                    {"available_bytes", result.device_free_bytes}, {"total_memory_bytes", result.device_total_bytes},
                    {"margin_bytes", fit.margin_bytes}, {"report", result.report}};
                std::ofstream f(report); f << j.dump(2) << '\n';
                if (!f) throw std::runtime_error("cannot write fit report");
            }
            return int(result.status);
        }
        const auto output = get("--output", "pocket-output.wav");
        const int runs = integer("--runs", 1); if (runs < 1 || runs > 100) throw std::runtime_error("runs must be 1..100");
        auto start = std::chrono::steady_clock::now(); tts_cpp::pocket::Engine engine(opts);
        const double load = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        if (runs > 1) engine.synthesize("A short warm up.");
        nlohmann::json measurements = nlohmann::json::array();
        for (int i = 0; i < runs; ++i) {
            start = std::chrono::steady_clock::now();
            std::vector<float> pcm; double first_audio = 0;
            auto result = engine.synthesize_stream(text, [&](const float * chunk, size_t n, int) {
                if (pcm.empty()) first_audio = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
                pcm.insert(pcm.end(), chunk, chunk+n); return true;
            });
            result.pcm = std::move(pcm);
            const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            const double duration = double(result.pcm.size())/result.sample_rate;
            measurements.push_back({{"seconds",elapsed},{"first_audio_seconds",first_audio},
                                    {"audio_seconds",duration},{"real_time_factor",elapsed/duration}});
            std::puts(measurements.back().dump().c_str());
            if (i == 0) {
                drwav_data_format format{drwav_container_riff, DR_WAVE_FORMAT_IEEE_FLOAT, 1, 24000, 32}; drwav wav;
                if (!drwav_init_file_write(&wav, output.c_str(), &format, nullptr)) throw std::runtime_error("cannot open output WAV");
                const auto written = drwav_write_pcm_frames(&wav, result.pcm.size(), result.pcm.data()); drwav_uninit(&wav);
                if (written != result.pcm.size()) throw std::runtime_error("incomplete WAV write");
            }
        }
        const auto report = get("--report", "");
        if (!report.empty()) {
            nlohmann::json j = {{"implementation","Fabric native ggml"},{"backend",engine.backend_name()},
                {"threads_per_worker",opts.n_threads},{"worker_threads",2},{"steps",opts.steps},{"temperature",opts.temperature},
                {"seed",opts.seed},{"text",text},{"load_seconds",load},{"warmups",runs>1?1:0},{"runs",measurements}};
            j["source_sha256"] = engine.model_source_sha256();
            j["sample_rate"] = 24000;
            j["compute_precision"] = "float32";
            j["rng"] = "Fabric MT19937 Box-Muller (not PyTorch seed-equivalent)";
            j["assets"] = {{"flow_lm",opts.flow_lm_path},{"mimi",opts.mimi_path},{"frontend",opts.frontend_path},{"voice",opts.voice_path}};
            std::ofstream f(report); f << j.dump(2) << '\n'; if (!f) throw std::runtime_error("cannot write report");
        }
        return 0;
    } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
