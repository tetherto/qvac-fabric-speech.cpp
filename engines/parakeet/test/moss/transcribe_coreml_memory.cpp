// Memory validation of the MOSS-Transcribe Core ML encoder under an
// overlapped schedule: transcriptions of one 30 s window run in a loop while
// the runtime prepares (generates and compiles), loads and warms the encoder
// in the background (wait = none). A sampler thread reads this process's
// phys_footprint every 2 ms and the slot's state every 10 ms; a second thread
// reads ANECompilerService's resident size every 100 ms. The report
// attributes the peaks to phases:
//   - prepare (digest, generate, compile) of the encoder while windows run on
//     ggml;
//   - load and warm of the encoder while windows run on ggml;
// and records the resident growth the encoder leaves behind: the settled
// footprint (the median sample) from the moment it is ready to the end of the
// run, less the ggml-only run's peak. It also records ANECompilerService's
// growth during the load (its peak less its size when the load starts), and
// compares the run's peak footprint with the Core ML bytes fit charges for
// the same options, as an increment over a ggml-only run of the same
// transcriptions (--baseline-bytes, from --ggml-only). A sample that could not
// be read (task_info, or ps failing) is dropped and counted in the report's
// sampler_errors, so a report with missing measurements is never taken for a
// quiet one. The report has the schema
// scripts/check-coreml-memory-schedule.py reads.
//
//   moss-transcribe-coreml-memory --gguf M.gguf --wav A.wav --cache-dir DIR
//       [--ggml-only] [--baseline-bytes N] [--passes-after-ready N]
//       [--ggml-passes N] [--max-new-tokens N] [--json OUT]
//
// The generated schedule needs a GGUF with no adjacent encoder bundle and an
// empty DIR. A fresh process identity and fresh paths make the ANE load cold.
// Exit status: 0 when the encoder became ready, served --passes-after-ready
// Core ML windows after that, and the peak increment stays within the charged
// envelope (always 0 with --ggml-only); 1 otherwise; 2 on usage or setup
// errors.

#include "mel_preprocess.h"
#include "parakeet/fit.h"
#include "parakeet/moss_transcribe.h"
#include "parakeet/moss_transcribe_fit.h"

#include <mach/mach.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kThreads = 4;
constexpr const char * kStage = "encoder";
constexpr double kMiB = 1048576.0;
constexpr double kTraceSeconds = 0.1;  // the bucket of the report's footprint trace
constexpr double kWindowSeconds = 30.0;
constexpr uint64_t kAneCompilerCleanBytes = 32ull << 20;

uint64_t footprint_now() {
    task_vm_info_data_t info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t) &info, &count) != KERN_SUCCESS) return 0;
    return info.phys_footprint;
}

// ANECompilerService's resident size (0 when it is not running); false when
// ps could not be run or failed.
bool ane_compiler_rss(uint64_t & total) {
    total = 0;
    FILE * f = popen("ps -axo rss=,comm= 2>/dev/null", "r");
    if (!f) return false;
    char line[1024];
    bool any = false;
    while (fgets(line, sizeof(line), f)) {
        any = true;
        if (strstr(line, "ANECompilerService")) total += (uint64_t) std::strtoull(line, nullptr, 10) * 1024;
    }
    return pclose(f) == 0 && any;
}

// The encoder stage's state, from the status JSON ("" when it has none).
std::string encoder_state(const std::string & status) {
    const std::string start = "{\"artifact_key\"";
    auto value = [](const std::string & s, const char * key) {
        const std::string k = std::string("\"") + key + "\":\"";
        const size_t at = s.find(k);
        if (at == std::string::npos) return std::string();
        const size_t from = at + k.size();
        return s.substr(from, s.find('"', from) - from);
    };
    for (size_t at = status.find(start); at != std::string::npos;) {
        const size_t next = status.find(start, at + 1);
        const std::string obj = status.substr(at, next == std::string::npos ? std::string::npos : next - at);
        at = next;
        if (value(obj, "stage") == kStage) return value(obj, "state");
    }
    return std::string();
}

const char * phase_of(const std::string & state) {
    if (state == "digesting" || state == "generating") return "generate";
    if (state == "compiling") return "compile";
    if (state == "loading" || state == "warming") return "load";
    return nullptr;
}

struct Sample {
    double t = 0;
    uint64_t footprint = 0;
};
struct StateSample {
    double t = 0;
    std::string state;
};
struct AneSample {
    double t = 0;
    uint64_t rss = 0;
};

struct PassInterval {
    double t0;
    double t1;
    bool coreml;
};

// One phase of the encoder stage: [t0, t1).
struct Segment {
    std::string phase;
    double t0 = 0, t1 = 0;
};

std::vector<Segment> segments_of(const std::vector<StateSample> & states, double t_end) {
    std::vector<Segment> out;
    bool open = false;
    Segment current;
    for (const StateSample & s : states) {
        const char * p = phase_of(s.state);
        if (open && (p == nullptr || current.phase != p)) {
            current.t1 = s.t;
            out.push_back(current);
            open = false;
        }
        if (p != nullptr && !open) {
            current = Segment{p, s.t, 0};
            open = true;
        }
    }
    if (open) {
        current.t1 = t_end;
        out.push_back(current);
    }
    return out;
}

}  // namespace

int main(int argc, char ** argv) {
    std::string gguf, wav, cache_dir, json_out;
    bool ggml_only = false;
    uint64_t baseline = 0;
    int passes_after_ready = 4;
    int ggml_passes = 20;
    int max_new_tokens = 8;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--gguf") gguf = next();
        else if (a == "--wav") wav = next();
        else if (a == "--cache-dir") cache_dir = next();
        else if (a == "--json") json_out = next();
        else if (a == "--ggml-only") ggml_only = true;
        else if (a == "--baseline-bytes") baseline = std::strtoull(next().c_str(), nullptr, 10);
        else if (a == "--passes-after-ready") passes_after_ready = std::atoi(next().c_str());
        else if (a == "--ggml-passes") ggml_passes = std::atoi(next().c_str());
        else if (a == "--max-new-tokens") max_new_tokens = std::atoi(next().c_str());
        else {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    if (gguf.empty() || wav.empty() || (!ggml_only && cache_dir.empty())) {
        std::fprintf(stderr, "usage: %s --gguf M.gguf --wav A.wav --cache-dir DIR [--ggml-only] "
                             "[--baseline-bytes N] [--passes-after-ready N] [--ggml-passes N] "
                             "[--max-new-tokens N] [--json OUT]\n",
                     argv[0]);
        return 2;
    }

    parakeet::moss::TranscribeOptions options;
    options.model_path = gguf;
    options.n_threads = kThreads;
    options.use_gpu = true;
    options.coreml.cache_dir = cache_dir;
    options.coreml.wait = parakeet::CoremlOptions::Wait::none;
    if (ggml_only) options.coreml.use = parakeet::CoremlOptions::Use::off;
    parakeet::moss::TranscribeRequest request;
    request.max_new_tokens = max_new_tokens;

    // The charge fit makes for these options (the Core ML-enabled ones, also
    // for the ggml-only run, so both reports carry the same charge).
    auto fit_options = options;
    fit_options.coreml.use = parakeet::CoremlOptions::Use::unset;
    const parakeet::FitResult fr = parakeet::moss::fit_params(fit_options, request, kWindowSeconds, 0);

    std::vector<float> pcm;
    int sample_rate = 0;
    if (parakeet::load_wav_mono_f32(wav, pcm, sample_rate) != 0) {
        std::fprintf(stderr, "cannot read %s\n", wav.c_str());
        return 2;
    }
    pcm.resize(std::min(pcm.size(), (size_t) (kWindowSeconds * sample_rate)));

    const auto t0 = Clock::now();
    auto now_s = [&] { return std::chrono::duration<double>(Clock::now() - t0).count(); };
    std::vector<Sample> samples;
    std::vector<StateSample> states;
    std::vector<AneSample> ane;
    std::mutex mu;
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> sampler_errors{0};
    std::unique_ptr<parakeet::moss::TranscribeEngine> engine;
    std::atomic<parakeet::moss::TranscribeEngine *> loaded{nullptr};

    std::thread sampler([&] {
        int tick = 0;
        while (!stop.load()) {
            const Sample s{now_s(), footprint_now()};
            if (s.footprint == 0) {
                ++sampler_errors;
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }
            parakeet::moss::TranscribeEngine * e = loaded.load();
            const bool poll = !ggml_only && e != nullptr && tick % 5 == 0;
            std::string st;
            if (poll) st = encoder_state(e->coreml_status_json());
            {
                std::lock_guard<std::mutex> lock(mu);
                samples.push_back(s);
                if (poll) states.push_back(StateSample{s.t, std::move(st)});
            }
            ++tick;
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    });
    std::thread ane_sampler([&] {
        while (!stop.load()) {
            AneSample a{now_s(), 0};
            if (!ane_compiler_rss(a.rss)) {
                ++sampler_errors;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            {
                std::lock_guard<std::mutex> lock(mu);
                ane.push_back(a);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });
    auto finish = [&] {
        stop.store(true);
        sampler.join();
        ane_sampler.join();
    };

    try {
        engine = std::make_unique<parakeet::moss::TranscribeEngine>(options);
    } catch (const std::exception & e) {
        finish();
        std::fprintf(stderr, "cannot load %s: %s\n", gguf.c_str(), e.what());
        return 2;
    }
    loaded.store(engine.get());
    const double t_loaded = now_s();

    int passes = 0, coreml_passes = 0, ggml_after_ready = 0, coreml_after_ready = 0;
    double t_all_ready = -1;
    std::vector<PassInterval> intervals;
    const auto deadline = Clock::now() + std::chrono::minutes(20);
    for (;;) {
        parakeet::moss::TranscribeResult result;
        const double pass_start = now_s();
        try {
            result = engine->transcribe(pcm.data(), pcm.size(), sample_rate, request);
        } catch (const std::exception & e) {
            finish();
            std::fprintf(stderr, "a transcription failed: %s\n", e.what());
            return 2;
        }
        const double pass_end = now_s();
        ++passes;
        const bool on_coreml = result.encoder_backend.rfind("coreml", 0) == 0;
        intervals.push_back({pass_start, pass_end, on_coreml});
        if (on_coreml) ++coreml_passes;
        if (ggml_only) {
            if (passes >= ggml_passes) break;
            continue;
        }
        const bool ready = encoder_state(engine->coreml_status_json()) == "ready";
        if (ready && t_all_ready < 0) t_all_ready = now_s();
        if (ready && !on_coreml) ++ggml_after_ready;
        if (ready && on_coreml) ++coreml_after_ready;
        if ((ready && coreml_after_ready >= passes_after_ready) || Clock::now() > deadline) break;
    }
    const double t_end = now_s();
    const std::string status = ggml_only ? std::string("{}") : engine->coreml_status_json();
    finish();

    // ---- report
    uint64_t peak = 0, first_pass = 0;
    for (const Sample & s : samples) peak = std::max(peak, s.footprint);
    for (const Sample & s : samples) {
        if (s.t >= t_loaded) {
            first_pass = s.footprint;
            break;
        }
    }
    auto peak_in = [&](double a, double b) {
        uint64_t p = 0;
        for (const Sample & s : samples) {
            if (s.t >= a && s.t < b) p = std::max(p, s.footprint);
        }
        return p;
    };
    auto at = [&](double t) {
        uint64_t f = 0;
        for (const Sample & s : samples) {
            if (s.t > t) break;
            f = s.footprint;
        }
        return f;
    };
    auto ane_in = [&](double a, double b) {
        uint64_t p = 0;
        for (const AneSample & s : ane) {
            if (s.t >= a && s.t < b + 0.1) p = std::max(p, s.rss);
        }
        return p;
    };
    // The median footprint sample in [a, b), or the sample at `a` when the
    // window holds none.
    auto settled = [&](double a, double b) {
        std::vector<uint64_t> v;
        for (const Sample & s : samples) {
            if (s.t >= a && s.t < b) v.push_back(s.footprint);
        }
        if (v.empty()) return at(a);
        std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
        return v[v.size() / 2];
    };
    uint64_t ane_peak = 0, ane_min = UINT64_MAX;
    for (const AneSample & s : ane) {
        ane_peak = std::max(ane_peak, s.rss);
        ane_min = std::min(ane_min, s.rss);
    }
    if (ane.empty()) ane_min = 0;
    const std::vector<Segment> segments = segments_of(states, t_end);
    double first_load = -1;
    for (const Segment & seg : segments) {
        if (seg.phase == "load" && (first_load < 0 || seg.t0 < first_load)) first_load = seg.t0;
    }
    uint64_t ane_before_loads = 0;
    bool ane_sampled_before = false;
    for (const AneSample & s : ane) {
        if (first_load < 0 || s.t > first_load) break;
        ane_before_loads = s.rss;
        ane_sampled_before = true;
    }
    // ANECompilerService before the load: clean when it held at most 32 MiB
    // (its growth is then a cold compile's whole size), dirty when it already
    // held memory from earlier compiles, unknown without a sample.
    const char * ane_baseline = !ane_sampled_before ? "unknown"
                                : ane_before_loads <= kAneCompilerCleanBytes ? "clean" : "dirty";
    // The resident step: the settled level once the last load ended.
    double t_ready = -1;
    for (const Segment & seg : segments) {
        if (seg.phase == "load") t_ready = std::max(t_ready, seg.t1);
    }
    const uint64_t level = t_ready >= 0 ? settled(t_ready, t_end) : 0;
    const uint64_t steady_increment = baseline > 0 && t_ready >= 0 && level > baseline ? level - baseline : 0;

    std::string json = "{";
    char buf[1024];
    std::snprintf(buf, sizeof(buf),
                  "\"mode\":\"%s\",\"passes\":%d,\"coreml_passes\":%d,\"ggml_passes_after_ready\":%d,"
                  "\"seconds\":%.2f,\"all_ready_s\":%.2f,\"peak_footprint\":%llu,\"footprint_after_load\":%llu,"
                  "\"baseline_bytes\":%llu,\"fit_coreml_bytes\":%llu,\"fit_coreml_validated\":%d,"
                  "\"fit_ggml_required\":%llu,\"ane_compiler_peak\":%llu,\"ane_compiler_min\":%llu,"
                  "\"samples\":%zu,\"state_polls\":%zu,\"sampler_errors\":%llu,\"segments\":[",
                  ggml_only ? "ggml-only" : "coreml", passes, coreml_passes, ggml_after_ready, t_end, t_all_ready,
                  (unsigned long long) peak, (unsigned long long) first_pass, (unsigned long long) baseline,
                  (unsigned long long) fr.coreml.bytes, fr.coreml.validated ? 1 : 0,
                  (unsigned long long) fr.ggml_required_bytes, (unsigned long long) ane_peak,
                  (unsigned long long) ane_min, samples.size(), states.size(),
                  (unsigned long long) sampler_errors.load());
    json += buf;
    bool first = true;
    uint64_t compile_inc = 0, load_inc = 0, generate_inc = 0, ane_load_peak = 0, ane_load_growth = 0;
    for (const Segment & seg : segments) {
        const uint64_t before = at(seg.t0);
        const uint64_t p = std::max(before, peak_in(seg.t0, seg.t1));
        const uint64_t a = ane_in(seg.t0, seg.t1);
        const uint64_t inc = p - before;
        if (seg.phase == "compile") compile_inc = std::max(compile_inc, inc);
        if (seg.phase == "generate") generate_inc = std::max(generate_inc, inc);
        std::string load_fields;
        if (seg.phase == "load") {
            load_inc = std::max(load_inc, inc);
            ane_load_peak = std::max(ane_load_peak, a);
            uint64_t ane_before = 0;
            for (const AneSample & s : ane) {
                if (s.t > seg.t0) break;
                ane_before = s.rss;
            }
            ane_load_growth = std::max(ane_load_growth, a > ane_before ? a - ane_before : 0);
            const uint64_t step = seg.t1 == t_ready ? steady_increment : 0;
            char extra[320];
            std::snprintf(extra, sizeof(extra),
                          ",\"settled_after_ready\":%llu,\"resident_increment\":%llu,\"ane_compiler_before\":%llu",
                          (unsigned long long) (seg.t1 == t_ready ? level : 0), (unsigned long long) step,
                          (unsigned long long) ane_before);
            load_fields = extra;
        }
        std::snprintf(buf, sizeof(buf),
                      "%s{\"stage\":\"%s\",\"phase\":\"%s\",\"t0\":%.3f,\"t1\":%.3f,\"before\":%llu,\"peak\":%llu,"
                      "\"increment\":%llu,\"ane_compiler_peak\":%llu%s}",
                      first ? "" : ",", kStage, seg.phase.c_str(), seg.t0, seg.t1, (unsigned long long) before,
                      (unsigned long long) p, (unsigned long long) inc, (unsigned long long) a,
                      load_fields.c_str());
        json += buf;
        first = false;
    }
    const uint64_t increment = baseline > 0 && peak > baseline ? peak - baseline : 0;
    const uint64_t ane_growth = ane_load_peak > ane_before_loads ? ane_load_peak - ane_before_loads : 0;
    json += "],\"pass_intervals\":[";
    first = true;
    for (const PassInterval & pass : intervals) {
        std::snprintf(buf, sizeof(buf), "%s[%.4f,%.4f,%d]", first ? "" : ",", pass.t0, pass.t1, pass.coreml ? 1 : 0);
        json += buf;
        first = false;
    }
    // The checker keys Core ML passes by the stage name's last '-' field,
    // which for "encoder" is the whole name.
    std::snprintf(buf, sizeof(buf),
                  "],\"max_generate_increment\":%llu,\"max_compile_increment\":%llu,\"max_load_increment\":%llu,"
                  "\"max_resident_increment\":%llu,\"steady_increment_over_baseline\":%llu,"
                  "\"ane_compiler_peak_during_load\":%llu,\"ane_compiler_before_loads\":%llu,"
                  "\"ane_compiler_growth\":%llu,\"ane_compiler_max_load_growth\":%llu,"
                  "\"peak_increment_over_baseline\":%llu,\"ane_compiler_baseline\":\"%s\","
                  "\"coreml_passes_by_capacity\":{\"%s\":%d},\"trace\":[",
                  (unsigned long long) generate_inc, (unsigned long long) compile_inc,
                  (unsigned long long) load_inc, (unsigned long long) steady_increment,
                  (unsigned long long) steady_increment, (unsigned long long) ane_load_peak,
                  (unsigned long long) ane_before_loads, (unsigned long long) ane_growth,
                  (unsigned long long) ane_load_growth, (unsigned long long) increment, ane_baseline, kStage,
                  coreml_after_ready);
    json += buf;
    first = true;
    for (size_t i = 0; i < samples.size();) {
        const double b0 = std::floor(samples[i].t / kTraceSeconds) * kTraceSeconds;
        uint64_t lo = UINT64_MAX, hi = 0;
        for (; i < samples.size() && samples[i].t < b0 + kTraceSeconds; ++i) {
            lo = std::min(lo, samples[i].footprint);
            hi = std::max(hi, samples[i].footprint);
        }
        std::snprintf(buf, sizeof(buf), "%s[%.1f,%llu,%llu]", first ? "" : ",", b0, (unsigned long long) lo,
                      (unsigned long long) hi);
        json += buf;
        first = false;
    }
    json += "],\"status\":";
    json += status + "}";
    if (!json_out.empty()) {
        if (FILE * f = std::fopen(json_out.c_str(), "w")) {
            std::fputs(json.c_str(), f);
            std::fputc('\n', f);
            std::fclose(f);
        }
    }
    std::printf("%s: %d passes (%d Core ML) in %.1f s, ready at %.1f s; peak footprint %.1f MiB "
                "(after load %.1f); fit Core ML %.1f MiB (validated %d)\n",
                ggml_only ? "ggml-only" : "coreml", passes, coreml_passes, t_end, t_all_ready, peak / kMiB,
                first_pass / kMiB, fr.coreml.bytes / kMiB, fr.coreml.validated ? 1 : 0);
    if (!ggml_only) {
        std::printf("increments: generate %.1f MiB, compile %.1f MiB, load+warm %.1f MiB, resident step %.1f MiB "
                    "over ggml; ANECompilerService peak %.1f MiB (min %.1f), %.1f MiB before the load (%s), "
                    "growth %.1f MiB\n",
                    generate_inc / kMiB, compile_inc / kMiB, load_inc / kMiB, steady_increment / kMiB,
                    ane_peak / kMiB, ane_min / kMiB, ane_before_loads / kMiB, ane_baseline, ane_growth / kMiB);
        std::printf("peak increment over the ggml-only baseline %.1f MiB, envelope %.1f MiB: %s\n",
                    increment / kMiB, fr.coreml.bytes / kMiB,
                    baseline == 0 ? "no baseline" : (increment <= fr.coreml.bytes ? "WITHIN" : "OVER"));
        if (baseline > 0 && increment > fr.coreml.bytes) return 1;
        if (t_all_ready < 0) {
            std::printf("the encoder did not become ready before the deadline\n");
            return 1;
        }
        if (coreml_after_ready < passes_after_ready) {
            std::printf("the encoder served %d Core ML windows after readiness (%d required)\n", coreml_after_ready,
                        passes_after_ready);
            return 1;
        }
    }
    return 0;
}
