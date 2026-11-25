#include "cosyvoice_coreml.h"
#include "cosyvoice_coreml_stages.h"
#include "cosyvoice_pipeline.h"
#include "coreml_sidecar_path.h"
#include "tts-cpp/cosyvoice/engine.h"
#include "tts-cpp/cosyvoice/fit.h"
#include "memory_schedule.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include <cmath>
#include <fstream>
#include <filesystem>
#include "test_env_portable.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <map>
#include <optional>
#include <random>
#include <thread>
#include <iterator>
#include <sstream>
#include <stdexcept>

static bool compare(const char *, const std::vector<float> &, const std::vector<float> &, double, double);

struct compute_units_guard {
    const char * name = "COSYVOICE_COREML_COMPUTE_UNITS";
    bool present = std::getenv(name) != nullptr;
    std::string value = present ? std::getenv(name) : "";
    ~compute_units_guard() {
        if (present) setenv(name, value.c_str(), 1);
        else unsetenv(name);
    }
};

static void require_fit_reason(const tts_cpp::cosyvoice::FitOptions & options, const std::string & reason) {
    const auto result = tts_cpp::cosyvoice::fit_params(options);
    if (result.status != tts_cpp::FitStatus::Error || result.fits || result.reason != reason ||
        result.device.total_bytes != 0)
        throw std::runtime_error("unexpected Core ML fit result: " + result.reason);
    if (reason == "coreml-memory-estimate-unavailable" && result.report.empty())
        throw std::runtime_error("missing Core ML fit diagnostic");
}

static void check_coreml_fit() {
    compute_units_guard units;
    setenv(units.name, "cpu_and_ane", 1);
    const auto directory = std::filesystem::path(test_tmpdir()) / ("cosyvoice-coreml-fit-" + test_process_tag());
    std::filesystem::create_directories(directory);
    struct cleanup {
        std::filesystem::path path;
        ~cleanup() { std::error_code error; std::filesystem::remove_all(path, error); }
    } guard{directory};
    tts_cpp::cosyvoice::FitOptions options;
    options.llm_gguf_path = (directory / "llm-f32.gguf").string();
    options.flow_gguf_path = (directory / "flow-f32.gguf").string();
    options.hift_gguf_path = (directory / "hift-f32.gguf").string();
    options.voice_gguf_path = (directory / "voice.gguf").string();
    options.backend = "cpu";
    require_fit_reason(options, "model-unreadable");
    const std::pair<std::string, std::string> stages[] = {
        {options.llm_gguf_path, ""}, {options.flow_gguf_path, ""},
        {options.flow_gguf_path, "-frontend"}, {options.hift_gguf_path, ""},
        {options.hift_gguf_path, "-f0"}};
    for (const auto & stage : stages) {
        const auto path = tts_cpp::detail::coreml_sidecar_path(stage.first, stage.second);
        std::filesystem::create_directory(path);
#ifdef TTS_CPP_USE_COREML
        require_fit_reason(options, "coreml-memory-estimate-unavailable");
#else
        require_fit_reason(options, "model-unreadable");
#endif
        setenv(units.name, "disabled", 1);
        require_fit_reason(options, "model-unreadable");
        setenv(units.name, "cpu_and_ane", 1);
        std::filesystem::remove(path);
    }
}

static std::vector<int> generate_lm(model_ctx & model, const std::vector<int> & text, int steps) {
    constexpr int seed = 42;
    return cosyvoice_llm_generate(model, cosyvoice_qwen_hp(model), text, {2}, steps,
                                  true, seed, steps, nullptr, false);
}

static std::string token_sequence(const std::vector<int> & tokens) {
    std::ostringstream output;
    std::copy(tokens.begin(), tokens.end(), std::ostream_iterator<int>(output, " "));
    return output.str();
}

static void require_tokens(const char * label, const std::vector<int> & actual,
                           const std::vector<int> & expected) {
    if (actual != expected)
        throw std::runtime_error(std::string(label) + ": expected " + token_sequence(expected) +
                                 "; actual " + token_sequence(actual));
}

static void check_lm_routing(const std::string & path) {
    auto model = cosyvoice_load_gguf(path);
    struct cleanup {
        model_ctx & model;
        ~cleanup() { cosyvoice_free(model); }
    } guard{model};
    ggml_backend_cpu_set_n_threads(model.backend, 2);
    constexpr int short_steps = 3;
    constexpr int prefix_rows = 5;
    const std::vector<int> text{1, 2};
    const auto native = generate_lm(model, text, short_steps);
    model.coreml = cosyvoice_coreml::load(path);
    if (!model.coreml) throw std::runtime_error("missing LM generation fixture");
    const auto first = generate_lm(model, text, short_steps);
    const auto first_predictions = model.coreml->predictions();
    const auto alternate = generate_lm(model, {3, 4}, short_steps);
    const auto second = generate_lm(model, text, short_steps);
    require_tokens("Core ML LM differs from native generation", first, native);
    require_tokens("Core ML LM cache reset differs", second, first);
    if (native.size() != short_steps || alternate.size() != short_steps ||
        first_predictions != prefix_rows + short_steps ||
        model.coreml->predictions() != first_predictions)
        throw std::runtime_error("LM generation routing/cache reset failed");
    const auto capacity = model.coreml->dimensions("mask").back();
    constexpr int fixture_context = 16;
    if (capacity != fixture_context) throw std::runtime_error("unexpected routing fixture context");
    model.coreml->clear_predictions();
    const auto fallback = generate_lm(model, text, fixture_context);
    if (model.coreml->predictions() != 0)
        throw std::runtime_error("undersized LM sidecar was selected");
    model.coreml.reset();
    if (fallback.size() != fixture_context || fallback != generate_lm(model, text, fixture_context))
        throw std::runtime_error("LM fallback differs from native generation");
    std::cout << "LM generation: Core ML selected, caches reset, undersized context retained ggml; tokens "
              << token_sequence(first) << '\n';
}

static void check_unavailable_stages() {
    std::vector<float> output{12.0f};
    if (compare("zero-output-regression", {1.0f}, {0.0f}, 0.998, 2.0))
        throw std::runtime_error("zero parity output was accepted");
    if (cosyvoice_coreml::load(test_tmpdir() + "/unavailable-cosyvoice.gguf"))
        throw std::runtime_error("missing Core ML model was selected");
    if (cosyvoice_coreml_frontend({}, {1}, {1}, output, output) ||
        cosyvoice_coreml_euler({}, 64, {}, {}, {}, 1, output) ||
        cosyvoice_coreml_f0({}, {}, 1, output) ||
        cosyvoice_coreml_spectrum({}, {}, 1, {}, 1, output))
        throw std::runtime_error("unavailable Core ML stage must retain ggml");
    if (output != std::vector<float>{12.0f})
        throw std::runtime_error("fallback changed its output before execution");
}

static std::vector<float> read_floats(const model_ctx & model, const char * name) {
    auto * tensor = cosyvoice_get(model, name);
    std::vector<float> data(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, data.data(), 0, data.size() * sizeof(float));
    return data;
}

static bool compare(const char * label, const std::vector<float> & reference,
                     const std::vector<float> & actual, double minimum_cosine, double maximum_error) {
    if (reference.size() != actual.size()) throw std::runtime_error("parity shape mismatch");
    double dot = 0, aa = 0, bb = 0, error = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i])) throw std::runtime_error("non-finite parity output");
        dot += double(reference[i]) * actual[i];
        aa += double(reference[i]) * reference[i];
        bb += double(actual[i]) * actual[i];
        error = std::max(error, double(std::abs(reference[i] - actual[i])));
    }
    const double cosine = dot / std::sqrt(aa * bb);
    std::cerr << label << " cosine=" << cosine << " max_abs=" << error << '\n';
    return std::isfinite(cosine) && cosine >= minimum_cosine && error <= maximum_error;
}

static void dump(const std::string & directory, const char * name, const std::vector<float> & data) {
    if (directory.empty()) return;
    std::filesystem::create_directories(directory);
    std::ofstream out(directory + "/" + name, std::ios::binary);
    out.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(float));
}

static bool hift_parity(const std::string & directory, const std::vector<float> & reference,
                         int native_frames, const std::string & output) {
    auto hift = cosyvoice_load_gguf(directory + "/cosyvoice3-hift-f32.gguf");
    ggml_backend_cpu_set_n_threads(hift.backend, 8);
    const auto f0 = cosyvoice_hift_f0(hift, reference, native_frames);
    const auto waveform = cosyvoice_hift_synth(hift, reference, native_frames, 42, nullptr, &f0);
    hift.coreml_f0 = cosyvoice_coreml::load(directory + "/cosyvoice3-hift-f32.gguf", "-f0");
    const auto coreml_f0 = cosyvoice_hift_f0(hift, reference, native_frames);
    bool passed = compare("f0", f0, coreml_f0, 0.9999, 5.0);
    dump(output, "f0-native.f32", f0);
    dump(output, "f0-coreml.f32", coreml_f0);
    hift.coreml = cosyvoice_coreml::load(directory + "/cosyvoice3-hift-f32.gguf");
    const auto coreml_waveform = cosyvoice_hift_synth(hift, reference, native_frames, 42, nullptr, &f0);
    if (!hift.coreml || !hift.coreml_f0 || !hift.coreml->predictions() || !hift.coreml_f0->predictions())
        throw std::runtime_error("Core ML HiFT did not run");
    passed = compare("hift", waveform, coreml_waveform, 0.999, 0.01) && passed;
    dump(output, "hift-native.f32", waveform);
    dump(output, "hift-coreml.f32", coreml_waveform);
    cosyvoice_free(hift);
    return passed;
 }

static void stage_parity(const std::string & directory, const std::string & output = "") {
    auto voice = cosyvoice_load_gguf(directory + "/voice.gguf");
    auto * token_tensor = cosyvoice_get(voice, "voice/prompt_token");
    std::vector<int> prompt(ggml_nelements(token_tensor));
    ggml_backend_tensor_get(token_tensor, prompt.data(), 0, prompt.size() * sizeof(int));
    const auto features = read_floats(voice, "voice/prompt_feat");
    const auto embedding = read_floats(voice, "voice/embedding");
    cosyvoice_free(voice);
    std::vector<int> tokens;
    std::ifstream stream(directory + "/parity-tokens.txt");
    for (int token; stream >> token;) tokens.push_back(token);
    if (tokens.empty()) throw std::runtime_error("missing parity-tokens.txt");
    auto flow = cosyvoice_load_gguf(directory + "/cosyvoice3-flow-f32.gguf");
    ggml_backend_cpu_set_n_threads(flow.backend, 8);
    int native_frames = 0, coreml_frames = 0;
    const auto reference = cosyvoice_flow_run(flow, prompt, tokens, features, features.size() / 80,
                                             embedding, native_frames);
    bool passed = hift_parity(directory, reference, native_frames, output);
    flow.coreml_frontend = cosyvoice_coreml::load(directory + "/cosyvoice3-flow-f32.gguf", "-frontend");
    flow.coreml = cosyvoice_coreml::load(directory + "/cosyvoice3-flow-f32.gguf");
    const auto actual = cosyvoice_flow_run(flow, prompt, tokens, features, features.size() / 80,
                                          embedding, coreml_frames);
    if (!flow.coreml || !flow.coreml_frontend || !flow.coreml->predictions() ||
        !flow.coreml_frontend->predictions()) throw std::runtime_error("Core ML flow did not run");
    passed = compare("flow", reference, actual, 0.998, 2.0) && passed;
    dump(output, "flow-native.f32", reference);
    dump(output, "flow-coreml.f32", actual);
    cosyvoice_free(flow);
    if (!passed) throw std::runtime_error("Core ML stage parity failed");
}

static std::vector<float> lm_prefix(const std::string & path) {
    auto native = cosyvoice_load_gguf(path);
    auto * speech = cosyvoice_get(native, "lm/speech_embedding/weight");
    auto * text = cosyvoice_get(native, "lm/embed_tokens/weight");
    const auto sos = cosyvoice_meta_i(native, "cosyvoice3.llm.sos", 6561);
    const size_t width = speech->ne[0];
    std::vector<float> prefix(width * 2);
    ggml_backend_tensor_get(speech, prefix.data(), sos * width * sizeof(float), width * sizeof(float));
    ggml_backend_tensor_get(text, prefix.data() + width, 0, width * sizeof(float));
    cosyvoice_free(native);
    return prefix;
}

static std::vector<float> native_lm_logits(const std::string & path, const std::vector<float> & input) {
    auto native = cosyvoice_load_gguf(path);
    const auto hp = cosyvoice_qwen_hp(native);
    constexpr size_t graph_nodes = 8192;
    constexpr int rows = 2;
    const size_t bytes = (graph_nodes + 16) * ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_nodes, false);
    auto * context = ggml_init({bytes, nullptr, true});
    auto * graph = ggml_new_graph_custom(context, graph_nodes, false);
    auto * x = ggml_new_tensor_2d(context, GGML_TYPE_F32, hp.hidden, rows);
    auto * positions = ggml_new_tensor_1d(context, GGML_TYPE_I32, rows);
    auto * mask = ggml_new_tensor_2d(context, GGML_TYPE_F32, rows, rows);
    ggml_set_input(x);
    ggml_set_input(positions);
    ggml_set_input(mask);
    auto * logits = build_qwen(context, native, hp, x, positions, mask, rows);
    ggml_build_forward_expand(graph, logits);
    auto * allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(native.backend));
    if (!ggml_gallocr_alloc_graph(allocator, graph)) throw std::runtime_error("native LM fixture allocation failed");
    const int32_t position_values[] = {0, 1};
    const float mask_values[] = {0, -INFINITY, 0, 0};
    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_set(positions, position_values, 0, sizeof(position_values));
    ggml_backend_tensor_set(mask, mask_values, 0, sizeof(mask_values));
    ggml_backend_cpu_set_n_threads(native.backend, 8);
    const auto status = ggml_backend_graph_compute(native.backend, graph);
    std::vector<float> result(ggml_nelements(logits));
    ggml_backend_tensor_get(logits, result.data(), 0, result.size() * sizeof(float));
    ggml_gallocr_free(allocator);
    ggml_free(context);
    cosyvoice_free(native);
    if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("native LM reference failed");
    return result;
}

// ---- the front-end, F0 and HiFT stages on the shared runtime -----------------

static void require(bool ok, const std::string & what) {
    if (!ok) throw std::runtime_error(what);
}

// A fake model's retire counter, as the shared runtime's binding would see it.
struct retire_probe {
    std::shared_ptr<int> count = std::make_shared<int>(0);
    std::shared_ptr<std::string> reason = std::make_shared<std::string>();
    cosyvoice_coreml::binding binding(bool strict) const {
        cosyvoice_coreml::binding b;
        auto c = count;
        auto r = reason;
        b.retire = [c, r](const std::string & why) { ++*c; *r = why; };
        b.strict = strict;
        return b;
    }
};

static int failing_predict(const char * const *, const float * const *, int, const char * const *, float * const *, int) {
    return 1;
}

// Writes `value` into every output of the declared shape.
static cosyvoice_coreml::fake_predict filling_predict(std::map<std::string, std::vector<int64_t>> dims, float value) {
    return [dims, value](const char * const *, const float * const *, int, const char * const * names, float * const * out,
                         int n) {
        for (int i = 0; i < n; ++i) {
            size_t count = 1;
            for (int64_t d : dims.at(names[i])) count *= static_cast<size_t>(d);
            std::fill(out[i], out[i] + count, value);
        }
        return 0;
    };
}

// A unit whose prediction fails on a model the shared runtime holds is
// retired once and redone on ggml (false, outputs untouched); under strict it
// fails; a legacy model still throws; a model too small for the unit runs
// ggml, or fails under strict.
static void check_stage_retire_seam() {
    const std::map<std::string, std::vector<int64_t>> f0_dims = {{"mel", {1, 80, 8}}, {"f0", {1, 8}}};
    const std::vector<float> mel(80 * 4, 0.25f);
    {
        retire_probe probe;
        auto model = cosyvoice_coreml::fake(f0_dims, failing_predict, "coreml-ane", probe.binding(false));
        std::vector<float> f0{7.0f};
        require(!cosyvoice_coreml_f0(model, mel, 4, f0) && f0 == std::vector<float>{7.0f},
                "a failed F0 prediction runs ggml with the output untouched");
        require(*probe.count == 1 && model->retired() && probe.reason->find("F0") != std::string::npos,
                "a failed F0 prediction retires the model once");
        require(!cosyvoice_coreml_f0(model, mel, 4, f0) && *probe.count == 1,
                "a retired model is retired only once");
    }
    {
        retire_probe probe;
        auto model = cosyvoice_coreml::fake(f0_dims, failing_predict, "coreml-ane", probe.binding(true));
        std::vector<float> f0;
        bool threw = false;
        try {
            cosyvoice_coreml_f0(model, mel, 4, f0);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        require(threw && *probe.count == 1, "under strict a failed prediction retires the model and fails the call");
        threw = false;
        auto wide = cosyvoice_coreml::fake(f0_dims, filling_predict(f0_dims, 1.0f), "coreml-ane", probe.binding(true));
        try {
            cosyvoice_coreml_f0(wide, std::vector<float>(80 * 9, 0.0f), 9, f0);
        } catch (const std::runtime_error & e) {
            threw = std::string(e.what()).find("cannot serve") != std::string::npos;
        }
        require(threw, "under strict a unit wider than the model fails");
    }
    {
        auto legacy = cosyvoice_coreml::fake(f0_dims, failing_predict, "coreml-ane", {});
        std::vector<float> f0;
        bool threw = false;
        try {
            cosyvoice_coreml_f0(legacy, mel, 4, f0);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        require(threw && !legacy->retirable(), "a legacy model's failure still throws");
    }
    {
        retire_probe probe;
        auto model = cosyvoice_coreml::fake(f0_dims, filling_predict(f0_dims, 3.0f), "coreml-ane", probe.binding(false));
        std::vector<float> f0{7.0f};
        require(!cosyvoice_coreml_f0(model, std::vector<float>(80 * 9, 0.0f), 9, f0) && f0 == std::vector<float>{7.0f} &&
                    *probe.count == 0,
                "a unit wider than the model runs ggml without retiring it");
        require(cosyvoice_coreml_f0(model, mel, 4, f0) && f0 == std::vector<float>(4, 3.0f) && model->predictions() == 1,
                "a unit within the model runs on it, cropped to its frames");
    }
    {
        const std::map<std::string, std::vector<int64_t>> fe_dims = {{"token_ids", {1, 6}}, {"token_mask", {1, 6, 1}},
                                                                     {"speaker", {1, 192}}, {"mu", {1, 12, 80}},
                                                                     {"spks", {1, 80}}};
        retire_probe probe;
        auto model = cosyvoice_coreml::fake(fe_dims, failing_predict, "coreml-ane", probe.binding(false));
        std::vector<float> mu{1.0f}, spks{2.0f};
        require(!cosyvoice_coreml_frontend(model, {1, 2, 3}, std::vector<float>(192, 0.5f), mu, spks) &&
                    mu == std::vector<float>{1.0f} && spks == std::vector<float>{2.0f} && *probe.count == 1,
                "a failed front-end prediction retires the model and runs ggml with the outputs untouched");
        auto ok = cosyvoice_coreml::fake(fe_dims, filling_predict(fe_dims, 0.5f), "coreml-ane", probe.binding(false));
        require(cosyvoice_coreml_frontend(ok, {1, 2, 3}, std::vector<float>(192, 0.5f), mu, spks) &&
                    mu.size() == 3 * 2 * 80 && spks.size() == 80,
                "a front-end unit within the model runs on it");
    }
    {
        const int capacity = 4, source = capacity * 120 + 1;
        const std::map<std::string, std::vector<int64_t>> hift_dims = {
            {"mel", {1, 80, capacity}}, {"source_stft", {1, 18, source}}, {"spec", {1, 18, source}}};
        retire_probe probe;
        auto model = cosyvoice_coreml::fake(hift_dims, failing_predict, "coreml-ane", probe.binding(false));
        std::vector<float> spectrum{5.0f};
        require(!cosyvoice_coreml_spectrum(model, std::vector<float>(80 * 3, 0.0f), 3,
                                           std::vector<float>(18 * 361, 0.0f), 361, spectrum) &&
                    spectrum == std::vector<float>{5.0f} && *probe.count == 1,
                "a failed HiFT prediction retires the model and runs ggml with the output untouched");
    }
    require(cosyvoice_merge_label("", "coreml-ane") == "coreml-ane" &&
                cosyvoice_merge_label("coreml-ane", "coreml-ane") == "coreml-ane" &&
                cosyvoice_merge_label("coreml-ane", "ggml") == "mixed",
            "stage labels merge to one label or mixed");
    std::cout << "stage retire seam: failure retires and redoes on ggml, strict fails, legacy throws\n";
}

// The F0 unit redone on ggml after a failed prediction equals the unit run on
// ggml alone, bit for bit (a GGUF with the HiFT F0 predictor's tensors, e.g.
// a twin of the cosyvoice3-hift-f0-m512 template's reference manifest).
static void check_f0_redo(const std::string & gguf) {
    auto hift = cosyvoice_load_gguf(gguf);
    struct cleanup {
        model_ctx & m;
        ~cleanup() { cosyvoice_free(m); }
    } guard{hift};
    const int frames = 24;
    std::mt19937 rng(7);
    std::normal_distribution<float> normal(-4.0f, 2.0f);
    std::vector<float> mel(80 * frames);
    for (float & x : mel) x = normal(rng);
    const auto reference = cosyvoice_hift_f0(hift, mel, frames);
    retire_probe probe;
    hift.coreml_f0 = cosyvoice_coreml::fake({{"mel", {1, 80, 32}}, {"f0", {1, 32}}}, failing_predict, "coreml-ane",
                                            probe.binding(false));
    const auto redone = cosyvoice_hift_f0(hift, mel, frames);
    require(*probe.count == 1, "the failed F0 prediction retired the model");
    require(redone.size() == reference.size() &&
                std::memcmp(redone.data(), reference.data(), redone.size() * sizeof(float)) == 0,
            "the F0 unit redone on ggml equals the ggml unit bit for bit");
    std::cout << "F0 redo: a failed prediction is retired and the unit redone on ggml, bit-equal\n";
}

// ---- memory schedule attribution (per stage) ---------------------------------------------

namespace qm = qvac_coreml_memory;

// One footprint sample of the CosyVoice memory schedule, with each stage's
// slot state when it was taken.
struct cosy_mem_sample {
    uint64_t footprint = 0;
    qm::mem_phase stage[COSYVOICE_STAGE_COUNT] = {qm::mem_phase::ggml, qm::mem_phase::ggml, qm::mem_phase::ggml};
};

// The schedule's attribution per stage and for the three together. One Core
// ML job runs at a time, so at most one stage prepares, loads or warms at any
// sample:
//   per stage  that stage's prepare and load+warm, each over the footprint just
//              before that phase began (the other stages' work counts as ggml
//              for it);
//   all        ready only once every stage is, otherwise the active stage's
//              phase: its steady state is over the footprint just before the
//              first load of any stage (against the resident parts summed) and
//              its whole run over the baseline (the envelope).
struct cosy_mem {
    qm::mem_breakdown stage[COSYVOICE_STAGE_COUNT];
    qm::mem_breakdown all;
};

static bool mem_active(qm::mem_phase p) { return p != qm::mem_phase::ggml && p != qm::mem_phase::ready; }

static cosy_mem cosy_mem_of(const std::vector<cosy_mem_sample> & samples, uint64_t base) {
    std::vector<qm::mem_sample> per[COSYVOICE_STAGE_COUNT], all;
    for (const cosy_mem_sample & s : samples) {
        qm::mem_phase a = qm::mem_phase::ggml;
        bool ready = true;
        for (int i = 0; i < COSYVOICE_STAGE_COUNT; ++i) {
            per[i].push_back({s.footprint, mem_active(s.stage[i]) ? s.stage[i] : qm::mem_phase::ggml});
            if (mem_active(s.stage[i])) a = s.stage[i];
            ready = ready && s.stage[i] == qm::mem_phase::ready;
        }
        all.push_back({s.footprint, a != qm::mem_phase::ggml ? a : ready ? qm::mem_phase::ready : qm::mem_phase::ggml});
    }
    cosy_mem m;
    for (int i = 0; i < COSYVOICE_STAGE_COUNT; ++i) m.stage[i] = qm::mem_breakdown_of(per[i], base);
    m.all = qm::mem_breakdown_of(all, base);
    return m;
}

struct cosy_stage_charges {
    uint64_t prepare = 0, load = 0, ane_reserve = 0;
};

// One line per charge the run exceeds, or per state it never sampled: each
// stage's prepare and load+warm and (on the Neural Engine) ANECompilerService's
// absolute size while it loaded, 1.25 x measured; the steady state with every
// stage ready, 1.25 x measured, against the resident parts summed; the whole
// run, as measured, against the envelope (qvac_coreml_memory::mem_failures).
static std::vector<std::string> cosy_mem_failures(const cosy_mem & m, const cosy_stage_charges (&c)[COSYVOICE_STAGE_COUNT],
                                                  const uint64_t (&ane_load_peak)[COSYVOICE_STAGE_COUNT],
                                                  uint64_t resident, uint64_t envelope) {
    std::vector<std::string> out;
    for (int i = 0; i < COSYVOICE_STAGE_COUNT; ++i) {
        qm::mem_breakdown b = m.stage[i];
        b.ready = true;  // the steady state is every stage's (below)
        b.steady = b.total = 0;
        // The load and the warm-up are one charged phase; a sample of either
        // is a sample of it (a front-end warm-up takes a few milliseconds).
        b.loaded = b.warmed = b.loaded || b.warmed;
        for (const std::string & f : qm::mem_failures(b, ane_load_peak[i], {c[i].prepare, c[i].load, 0, c[i].ane_reserve, 0}))
            out.push_back(std::string(cosyvoice_stage_name(static_cast<cosyvoice_stage>(i))) + " " + f);
    }
    qm::mem_breakdown b = m.all;
    b.generated = b.compiled = b.loaded = b.warmed = true;  // per stage (above)
    b.prepare = b.load_warm = 0;
    for (const std::string & f : qm::mem_failures(b, 0, {0, 0, resident, 0, envelope})) out.push_back("all " + f);
    return out;
}

static bool contains_text(const std::vector<std::string> & lines, const std::string & what) {
    for (const std::string & l : lines)
        if (l.find(what) != std::string::npos) return true;
    return false;
}

// The schedule's arithmetic on a synthetic run (no device).
static void check_memory_schedule_attribution() {
    using P = qm::mem_phase;
    auto at = [](uint64_t fp, P a, P b, P c) {
        cosy_mem_sample s;
        s.footprint = fp;
        s.stage[0] = a;
        s.stage[1] = b;
        s.stage[2] = c;
        return s;
    };
    const std::vector<cosy_mem_sample> run = {
        at(100, P::ggml, P::ggml, P::ggml),       at(102, P::generate, P::ggml, P::ggml),
        at(120, P::compile, P::ggml, P::ggml),    at(105, P::ggml, P::generate, P::ggml),
        at(125, P::ggml, P::compile, P::ggml),    at(110, P::ggml, P::ggml, P::generate),
        at(150, P::ggml, P::ggml, P::compile),    at(130, P::load, P::ggml, P::ggml),
        at(140, P::ready, P::load, P::ggml),      at(150, P::ready, P::warm, P::ggml),
        at(145, P::ready, P::ready, P::load),     at(175, P::ready, P::ready, P::warm),
        at(170, P::ready, P::ready, P::ready),    at(168, P::ready, P::ready, P::ready)};
    const cosy_mem m = cosy_mem_of(run, 100);
    // Each phase is over the footprint just before it began, whatever the
    // other stages held then (stage 1's generation starts at 120).
    require(m.stage[0].prepare == 20 && m.stage[1].prepare == 5 && m.stage[2].prepare == 25,
            "memory schedule: each stage's prepare over the footprint just before it began");
    require(m.stage[0].load_warm == 0 && m.stage[1].load_warm == 20 && m.stage[2].load_warm == 25,
            "memory schedule: each stage's load+warm over the footprint just before it began");
    require(m.all.steady == 20 && m.all.total == 75 && m.all.ready,
            "memory schedule: steady with every stage ready over the first load, whole run over the baseline");
    const cosy_stage_charges ok[COSYVOICE_STAGE_COUNT] = {{25, 1, 0}, {7, 25, 0}, {32, 32, 63}};
    const uint64_t ane[COSYVOICE_STAGE_COUNT] = {0, 0, 50};
    require(cosy_mem_failures(m, ok, ane, 25, 75).empty(), "memory schedule: a run within every charge");
    // 1.25 x 25 = 31.25: a 32 charge holds it, 31 does not.
    cosy_stage_charges low[COSYVOICE_STAGE_COUNT] = {{25, 1, 0}, {7, 25, 0}, {31, 32, 63}};
    require(contains_text(cosy_mem_failures(m, low, ane, 25, 75), "hift prepare"),
            "memory schedule: the margin applies to each stage's prepare");
    const uint64_t ane_high[COSYVOICE_STAGE_COUNT] = {0, 0, 51};
    require(contains_text(cosy_mem_failures(m, ok, ane_high, 25, 75), "ANECompilerService"),
            "memory schedule: 1.25 x the compiler's size while a stage loads must fit its reserve");
    require(contains_text(cosy_mem_failures(m, ok, ane, 24, 75), "steady state") &&
                contains_text(cosy_mem_failures(m, ok, ane, 25, 74), "whole run"),
            "memory schedule: the steady state carries the margin, the whole run is held to the envelope as measured");
    std::vector<cosy_mem_sample> partial = run;
    partial.resize(12);
    require(contains_text(cosy_mem_failures(cosy_mem_of(partial, 100), ok, ane, 25, 75), "never sampled the slot ready"),
            "memory schedule: a run without every stage ready fails");
    partial = run;
    partial[1].stage[0] = P::digest;
    require(contains_text(cosy_mem_failures(cosy_mem_of(partial, 100), ok, ane, 25, 75),
                          "frontend the schedule never sampled the slot generating"),
            "memory schedule: a digest alone does not count as the generation");
    std::cout << "memory schedule attribution and margins: ok\n";
}

#ifdef TTS_CPP_USE_COREML
// ---- device gates on the generated sidecars --------------------------------------------

static std::string status_state(const std::string & json, const char * stage) {
    const std::string key = std::string("\"stage\":\"") + stage + "\"";
    const auto at = json.find(key);
    if (at == std::string::npos) return {};
    const auto st = json.find("\"state\":\"", at);
    if (st == std::string::npos) return {};
    const auto b = st + 9;
    return json.substr(b, json.find('"', b) - b);
}

// The status writes each stage's keys in order ("route" before "stage"), so
// the route is the last one before the stage's name.
static std::string status_route(const std::string & json, const char * stage) {
    const std::string key = std::string("\"stage\":\"") + stage + "\"";
    const auto at = json.find(key);
    const auto st = at == std::string::npos ? std::string::npos : json.rfind("\"route\":\"", at);
    if (st == std::string::npos) return {};
    const auto b = st + 9;
    return json.substr(b, json.find('"', b) - b);
}

static std::vector<int> read_tokens(const std::string & path) {
    std::vector<int> tokens;
    std::ifstream stream(path);
    for (int token; stream >> token;) tokens.push_back(token);
    return tokens;
}

static double max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) return INFINITY;
    double e = 0;
    for (size_t i = 0; i < a.size(); ++i) e = std::max(e, double(std::abs(a[i] - b[i])));
    return e;
}

// The generated route end to end on a model directory (the staged GGUFs,
// voice.gguf, vocab, merges and parity-tokens.txt), with `cache` empty:
//  - a call before the sidecars are ready runs every stage on ggml and equals
//    the Core ML-off Engine's output (coreml_pending set);
//  - once ready, every stage runs on its generated sidecar, routed `generated`;
//  - the existing stage parity gates on the generated sidecars: F0 (cosine
//    0.9999, max 5 Hz), the HiFT waveform with pinned F0 (cosine 0.999, max
//    0.01) and the flow mel with the front-end on Core ML (cosine 0.998,
//    max 2.0), each against ggml;
//  - a second Engine on the same cache adopts the artifacts (`cached`).
static void generated_gates(const std::string & dir, const std::string & cache) {
    namespace fs = std::filesystem;
    require(!fs::exists(fs::path(cache) / "qvac-coreml-v1" / "artifacts") ||
                fs::is_empty(fs::path(cache) / "qvac-coreml-v1" / "artifacts"),
            "--generated needs an empty cache directory: " + cache);
    const std::vector<int> tokens = read_tokens(dir + "/parity-tokens.txt");
    require(!tokens.empty(), "missing parity-tokens.txt in " + dir);
    tts_cpp::cosyvoice::EngineOptions base;
    base.model_dir = dir;
    base.force_speech_tokens = tokens;
    // The speech tokens are pinned, so the LM is never loaded: a directory
    // without one (CI stages only the flow and HiFT GGUFs) names a path that
    // is never opened.
    bool has_llm = false;
    for (const auto & e : fs::directory_iterator(dir))
        has_llm = has_llm || e.path().filename().string().rfind("cosyvoice3-llm", 0) == 0;
    if (!has_llm) base.llm_gguf_path = dir + "/cosyvoice3-llm-not-loaded.gguf";
    base.n_threads = 4;
    base.coreml_ignore_prebuilt = true;
    tts_cpp::cosyvoice::EngineOptions off = base;
    off.coreml.use = tts_cpp::CoremlOptions::Use::off;
    const auto ggml = tts_cpp::cosyvoice::Engine(off).synthesize("parity");
    require(ggml.coreml_backend == "ggml" && !ggml.coreml_pending, "Core ML off runs every stage on ggml");

    tts_cpp::cosyvoice::EngineOptions gen = base;
    gen.coreml.cache_dir = cache;
    gen.coreml.prewarm = tts_cpp::CoremlOptions::Prewarm::none;  // nothing loads before the first call
    tts_cpp::cosyvoice::Engine engine(gen);
    const auto pending = engine.synthesize("parity");
    require(pending.coreml_backend == "ggml" && pending.coreml_pending,
            "a call before the sidecars are ready runs on ggml (coreml_pending)");
    require(max_abs_diff(pending.pcm, ggml.pcm) <= 1e-6,
            "a pending call equals the Core ML-off output (max diff " + std::to_string(max_abs_diff(pending.pcm, ggml.pcm)) + ")");
    require(engine.coreml_wait(-1), "every generated sidecar becomes ready: " + engine.coreml_status());
    const std::string status = engine.coreml_status();
    for (const char * stage : {"frontend", "f0", "hift"}) {
        require(status_state(status, stage) == "ready" && status_route(status, stage) == "generated",
                std::string(stage) + " generated and ready: " + status);
    }
    const auto ready = engine.synthesize("parity");
    require(ready.frontend_backend == "coreml-ane" && ready.f0_backend == "coreml-ane" &&
                ready.hift_backend == "coreml-ane" && ready.coreml_backend == "coreml-ane" && !ready.coreml_pending,
            "once ready every stage runs on Core ML (" + ready.frontend_backend + "/" + ready.f0_backend + "/" +
                ready.hift_backend + ")");
    require(compare("engine waveform (generated vs ggml)", ggml.pcm, ready.pcm, 0.0, 2.0),
            "the generated Engine output is finite and bounded");

    // Stage parity with the generated models, at the existing gates.
    tts_cpp::CoremlOptions opts;
    opts.cache_dir = cache;
    opts.wait = tts_cpp::CoremlOptions::Wait::construct;
    auto stages = cosyvoice_coreml_stages::create(dir + "/cosyvoice3-flow-f32.gguf", dir + "/cosyvoice3-hift-f32.gguf",
                                                  opts, true, false);
    require(stages && stages->wait(-1, nullptr, 0), "the stages adopt the cached artifacts");
    for (const char * stage : {"frontend", "f0", "hift"}) {
        require(status_route(stages->status_json(), stage) == "cached", std::string(stage) + " adopted from the cache");
    }
    auto voice = cosyvoice_load_gguf(dir + "/voice.gguf");
    auto * token_tensor = cosyvoice_get(voice, "voice/prompt_token");
    std::vector<int> prompt(ggml_nelements(token_tensor));
    ggml_backend_tensor_get(token_tensor, prompt.data(), 0, prompt.size() * sizeof(int));
    const auto features = read_floats(voice, "voice/prompt_feat");
    const auto embedding = read_floats(voice, "voice/embedding");
    cosyvoice_free(voice);
    auto flow = cosyvoice_load_gguf(dir + "/cosyvoice3-flow-f32.gguf");
    ggml_backend_cpu_set_n_threads(flow.backend, 8);
    int frames = 0, coreml_frames = 0;
    const auto mel = cosyvoice_flow_run(flow, prompt, tokens, features, features.size() / 80, embedding, frames);
    const cosyvoice_stage_call fe = stages->acquire(cosyvoice_stage::frontend, (int) (prompt.size() + tokens.size()), nullptr);
    flow.coreml_frontend = fe.model;
    const auto mel_coreml = cosyvoice_flow_run(flow, prompt, tokens, features, features.size() / 80, embedding, coreml_frames);
    require(fe.model && fe.model->predictions() == 1, "the generated front-end ran");
    bool passed = compare("flow (generated front-end)", mel, mel_coreml, 0.998, 2.0);
    flow.coreml_frontend.reset();
    cosyvoice_free(flow);

    auto hift = cosyvoice_load_gguf(dir + "/cosyvoice3-hift-f32.gguf");
    ggml_backend_cpu_set_n_threads(hift.backend, 8);
    const auto f0 = cosyvoice_hift_f0(hift, mel, frames);
    const auto wav = cosyvoice_hift_synth(hift, mel, frames, 42, nullptr, &f0);
    const cosyvoice_stage_call f0_call = stages->acquire(cosyvoice_stage::f0, frames, nullptr);
    const cosyvoice_stage_call hift_call = stages->acquire(cosyvoice_stage::hift, frames, nullptr);
    hift.coreml_f0 = f0_call.model;
    const auto f0_coreml = cosyvoice_hift_f0(hift, mel, frames);
    passed = compare("f0 (generated)", f0, f0_coreml, 0.9999, 5.0) && passed;
    hift.coreml = hift_call.model;
    const auto wav_coreml = cosyvoice_hift_synth(hift, mel, frames, 42, nullptr, &f0);
    require(f0_call.model && hift_call.model && f0_call.model->predictions() == 1 && hift_call.model->predictions() == 1,
            "the generated F0 and HiFT ran");
    passed = compare("hift (generated, pinned F0)", wav, wav_coreml, 0.999, 0.01) && passed;
    hift.coreml.reset();
    hift.coreml_f0.reset();
    cosyvoice_free(hift);
    require(passed, "generated stage parity");
    std::cout << "generated: pending call equals ggml, every stage generated and ready, parity at the stage gates, "
                 "cached adoption\n";
}

static std::string host_chip() {
    char buf[256] = {};
    size_t n = sizeof(buf);
    FILE * p = popen("sysctl -n machdep.cpu.brand_string", "r");
    if (!p) return "unknown";
    n = std::fread(buf, 1, sizeof(buf) - 1, p);
    pclose(p);
    std::string s(buf, n);
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
    return s.rfind("Apple ", 0) == 0 ? s.substr(6) : s;
}

static double mib(uint64_t bytes) { return bytes / 1048576.0; }

// The memory validation behind the three templates' memory constants: ggml
// F0 units run in a loop on this thread while the three sidecars are
// generated, compiled, loaded and warmed into `cache_dir` in the background
// (the construction group, pipeline order), then every stage runs on Core ML
// three times. (The F0 graph's arena is small: the whole HiFT decode
// allocates and frees about 100 MiB per call, which would swamp the phases
// measured against the footprint before each began.) One sampler thread
// reads the footprint every 2 ms and each slot's state every 10 ms; another
// reads ANECompilerService every 20 ms. Each phase must stay within the
// template constant that charges it (cosy_mem_failures).
static void memory_schedule(const std::string & dir, const std::string & cache_dir) {
    std::cerr << "memory schedule (cache " << cache_dir << ", host " << host_chip() << ")\n";
    const std::string flow_gguf = dir + "/cosyvoice3-flow-f32.gguf", hift_gguf = dir + "/cosyvoice3-hift-f32.gguf";
    tts_cpp::CoremlOptions opts;
    opts.cache_dir = cache_dir;
    const cosyvoice_coreml_plan plan = cosyvoice_plan_coreml(flow_gguf, hift_gguf, opts, true);
    for (const auto & s : plan.stages) require(s.route == "generate", "route generate (" + s.route + " " + s.state + " " + s.reason + ")");
    auto hift = cosyvoice_load_gguf(hift_gguf);
    struct cleanup {
        model_ctx & m;
        ~cleanup() { cosyvoice_free(m); }
    } guard{hift};
    ggml_backend_cpu_set_n_threads(hift.backend, 4);
    const int frames = 200;
    std::mt19937 rng(11);
    std::normal_distribution<float> normal(-4.0f, 2.0f);
    std::vector<float> mel(80 * frames);
    for (float & x : mel) x = normal(rng);
    const std::vector<int32_t> tokens(400, 17);
    const std::vector<float> speaker(192, 0.1f);
    for (int i = 0; i < 2; ++i) cosyvoice_hift_f0(hift, mel, frames);  // warm the ggml arena
    uint64_t base = 0;
    const bool base_ok = qm::phys_footprint(base);
    size_t footprint_failures = 0;
    std::atomic<bool> stop{false};
    std::atomic<const cosyvoice_coreml_stages *> watched{nullptr};
    std::atomic<int> phases[COSYVOICE_STAGE_COUNT];
    for (auto & p : phases) p.store((int) qm::mem_phase::ggml);
    std::vector<cosy_mem_sample> samples;
    samples.reserve(1 << 17);
    auto read_phases = [&](qm::mem_phase (&out)[COSYVOICE_STAGE_COUNT]) {
        const cosyvoice_coreml_stages * st = watched.load();
        const std::string json = st ? st->status_json() : std::string();
        for (int i = 0; i < COSYVOICE_STAGE_COUNT; ++i)
            out[i] = st ? qm::mem_phase_of(status_state(json, cosyvoice_stage_name(static_cast<cosyvoice_stage>(i))))
                        : qm::mem_phase::ggml;
    };
    std::thread sampler([&] {
        for (int tick = 0; !stop.load(); ++tick) {
            if (tick % 5 == 0) {
                qm::mem_phase p[COSYVOICE_STAGE_COUNT];
                read_phases(p);
                for (int i = 0; i < COSYVOICE_STAGE_COUNT; ++i) phases[i].store((int) p[i]);
            }
            uint64_t fp = 0;
            if (qm::phys_footprint(fp)) {
                cosy_mem_sample s;
                s.footprint = fp;
                for (int i = 0; i < COSYVOICE_STAGE_COUNT; ++i) s.stage[i] = (qm::mem_phase) phases[i].load();
                samples.push_back(s);
            } else {
                ++footprint_failures;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    });
    // Each compiler query is bracketed by two reads of the slot states: it is
    // attributed to the one stage loading or warming at both reads, counted as
    // before or after the loads when none was, and dropped otherwise.
    std::vector<qm::AneSample> ane_samples[COSYVOICE_STAGE_COUNT];
    size_t ane_crossing = 0;
    std::thread ane_sampler([&] {
        auto loading = [](qm::mem_phase p) { return p == qm::mem_phase::load || p == qm::mem_phase::warm; };
        while (!stop.load()) {
            qm::mem_phase before[COSYVOICE_STAGE_COUNT], after[COSYVOICE_STAGE_COUNT];
            read_phases(before);
            uint64_t a = 0;
            const bool ok = qm::ane_compiler_rss(a);
            read_phases(after);
            int stage = -1;
            bool crossing = false;
            for (int i = 0; i < COSYVOICE_STAGE_COUNT; ++i) {
                const std::optional<bool> l = qm::ane_query_loading(before[i], after[i]);
                if (!l) crossing = true;
                else if (*l) stage = i;
            }
            if (crossing) {
                ++ane_crossing;
            } else {
                // Not loading: a sample before or after every stage's load.
                for (int i = 0; i < COSYVOICE_STAGE_COUNT; ++i) {
                    if (stage < 0 || stage == i) ane_samples[i].push_back({a, stage == i, ok});
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    });
    const auto t0 = std::chrono::steady_clock::now();
    auto stages = cosyvoice_coreml_stages::create(flow_gguf, hift_gguf, opts, true, false);
    watched.store(stages.get());
    int ggml_calls = 0, coreml_calls[COSYVOICE_STAGE_COUNT] = {0, 0, 0};
    // ggml alone while the construction group prepares, loads and warms the
    // three stages in pipeline order, so each stage's phases overlap inference
    // and nothing else of Core ML's; then every unit on Core ML, so the steady
    // state holds each stage's prediction workspace.
    while (std::chrono::steady_clock::now() - t0 < std::chrono::minutes(5) && !stages->wait(0, nullptr, 0)) {
        cosyvoice_hift_f0(hift, mel, frames);
        ++ggml_calls;
    }
    while (std::chrono::steady_clock::now() - t0 < std::chrono::minutes(5)) {
        const cosyvoice_stage_call fe = stages->acquire(cosyvoice_stage::frontend, (int) tokens.size(), nullptr);
        std::vector<float> mu, spks;
        if (fe.model && cosyvoice_coreml_frontend(fe.model, tokens, speaker, mu, spks)) ++coreml_calls[0];
        const cosyvoice_stage_call f0 = stages->acquire(cosyvoice_stage::f0, frames, nullptr);
        const cosyvoice_stage_call hc = stages->acquire(cosyvoice_stage::hift, frames, nullptr);
        hift.coreml_f0 = f0.model;
        hift.coreml = hc.model;
        cosyvoice_hift_synth(hift, mel, frames, 42);
        if (f0.model && f0.model->predictions()) ++coreml_calls[1];
        if (hc.model && hc.model->predictions()) ++coreml_calls[2];
        if (!f0.model || !hc.model) ++ggml_calls;
        hift.coreml_f0.reset();
        hift.coreml.reset();
        if (coreml_calls[0] >= 3 && coreml_calls[1] >= 3 && coreml_calls[2] >= 3) break;
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    stop.store(true);
    sampler.join();
    ane_sampler.join();
    std::cerr << "  status: " << stages->status_json() << "\n";

    const cosy_mem m = cosy_mem_of(samples, base);
    cosy_stage_charges charges[COSYVOICE_STAGE_COUNT];
    uint64_t ane_peak[COSYVOICE_STAGE_COUNT] = {0, 0, 0}, resident = 0;
    qm::AneGrowth growth[COSYVOICE_STAGE_COUNT];
    std::vector<std::string> failures;
    if (!base_ok || footprint_failures) failures.push_back("footprint probe failed");
    for (int i = 0; i < COSYVOICE_STAGE_COUNT; ++i) {
        const cosyvoice_stage_route & r = plan.stages[i];
        charges[i] = {(uint64_t) std::ceil(r.prepare_peak_bytes * r.margin), (uint64_t) std::ceil(r.load_peak_bytes * r.margin),
                      r.ane_compile_reserve_bytes};
        resident += (uint64_t) std::ceil(r.resident_bytes * r.margin);
        growth[i] = qm::ane_compiler_growth(ane_samples[i]);
        ane_peak[i] = growth[i].peak;
        for (const std::string & f : qm::probe_failures(true, 0, true, growth[i]))
            failures.push_back(std::string(cosyvoice_stage_name(static_cast<cosyvoice_stage>(i))) + " " + f);
    }
    for (const std::string & f : cosy_mem_failures(m, charges, ane_peak, resident, plan.envelope_bytes)) failures.push_back(f);
    std::fprintf(stderr, "  memory (MiB; each phase over the footprint just before it began, steady over the footprint "
                         "just before the first load, whole run over the warm ggml baseline %.1f; a phase needs x1.25):\n",
                 mib(base));
    std::string report = "{\"chip\":\"" + host_chip() + "\",\"units\":\"cpu_and_ane\",\"base\":" + std::to_string(base);
    for (int i = 0; i < COSYVOICE_STAGE_COUNT; ++i) {
        const char * name = cosyvoice_stage_name(static_cast<cosyvoice_stage>(i));
        const qm::mem_breakdown & b = m.stage[i];
        std::fprintf(stderr,
                     "    %-8s prepare %7.1f (charged %7.1f; generate %.1f, compile %.1f)  load+warm %7.1f (charged %7.1f)"
                     "  ANECompilerService while loading %.1f (reserve %.1f, growth %.1f, %s baseline)\n",
                     name, mib(b.prepare), mib(charges[i].prepare), mib(b.generate), mib(b.compile), mib(b.load_warm),
                     mib(charges[i].load), mib(ane_peak[i]), mib(charges[i].ane_reserve), mib(growth[i].growth),
                     qm::ane_baseline_name(qm::ane_compiler_baseline(growth[i])));
        report += std::string(",\"") + name + "\":{\"prepare\":" + std::to_string(b.prepare) + ",\"generate\":" +
                  std::to_string(b.generate) + ",\"compile\":" + std::to_string(b.compile) + ",\"load_warm\":" +
                  std::to_string(b.load_warm) + ",\"ane_load_peak\":" + std::to_string(ane_peak[i]) +
                  ",\"ane_before\":" + std::to_string(growth[i].before) + ",\"ane_growth\":" +
                  std::to_string(growth[i].growth) + ",\"ane_baseline\":\"" +
                  qm::ane_baseline_name(qm::ane_compiler_baseline(growth[i])) + "\"}";
    }
    std::fprintf(stderr, "    steady   %7.1f (over the baseline %.1f; charged resident %7.1f)  whole run %7.1f (envelope %7.1f)\n",
                 mib(m.all.steady), mib(m.all.steady_over_base), mib(resident), mib(m.all.total),
                 mib(plan.envelope_bytes));
    report += ",\"steady\":" + std::to_string(m.all.steady) + ",\"steady_over_base\":" +
              std::to_string(m.all.steady_over_base) + ",\"total\":" + std::to_string(m.all.total) +
              ",\"samples\":" + std::to_string(samples.size()) + ",\"ane_crossing\":" + std::to_string(ane_crossing) + "}";
    std::fprintf(stderr, "  memory report: %s\n", report.c_str());
    for (const std::string & f : failures) std::cerr << "  [FAIL] memory schedule: " << f << '\n';
    std::fprintf(stderr, "  overlapped schedule: %d ggml + %d/%d/%d Core ML units in %.1f s\n", ggml_calls,
                 coreml_calls[0], coreml_calls[1], coreml_calls[2], secs);
    require(failures.empty() && ggml_calls >= 1 && coreml_calls[0] >= 3 && coreml_calls[1] >= 3 && coreml_calls[2] >= 3,
            "memory schedule");
}
#endif

int main(int argc, char ** argv) {
    try {
#ifndef TTS_CPP_USE_COREML
        if (argc == 2 && (std::string(argv[1]) == "--stages-fixture" ||
                          std::string(argv[1]) == "--lm-fixture" ||
                          std::string(argv[1]) == "--routing-fixture")) return 77;
        if (argc == 4 && (std::string(argv[1]) == "--generated" || std::string(argv[1]) == "--memory-schedule")) {
            std::cerr << "SKIP: built without TTS_CPP_COREML\n";
            return 77;
        }
#else
        if (argc == 4 && std::string(argv[1]) == "--generated") {
            generated_gates(argv[2], argv[3]);
            return 0;
        }
        if (argc == 4 && std::string(argv[1]) == "--memory-schedule") {
            memory_schedule(argv[2], argv[3]);
            return 0;
        }
#endif
        if (argc == 3 && std::string(argv[1]) == "--f0-redo") {
            check_f0_redo(argv[2]);
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--routing-fixture") {
            const char * file = std::getenv("COSYVOICE_COREML_ROUTING_GGUF");
            if (!file || !*file) return 77;
            check_lm_routing(file);
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--stages-fixture") {
            const char * root = std::getenv("COSYVOICE_COREML_TEST_MODELS_DIR");
            if (!root || !*root) return 77;
            stage_parity(root);
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--lm-fixture") {
            const char * file = std::getenv("COSYVOICE_COREML_TEST_LM_GGUF");
            if (!file || !*file) return 77;
            argv[1] = const_cast<char *>(file);
        }
        if (argc == 4 && std::string(argv[1]) == "--hift") {
            const auto bytes = std::filesystem::file_size(argv[3]);
            if (!bytes || bytes % (80 * sizeof(float))) throw std::runtime_error("invalid mel fixture");
            std::vector<float> mel(bytes / sizeof(float));
            std::ifstream in(argv[3], std::ios::binary);
            in.read(reinterpret_cast<char *>(mel.data()), bytes);
            if (!in || !hift_parity(argv[2], mel, mel.size() / 80, ""))
                throw std::runtime_error("Core ML HiFT parity failed");
            return 0;
        }
        if ((argc == 3 || argc == 4) && std::string(argv[1]) == "--parity") { stage_parity(argv[2], argc == 4 ? argv[3] : ""); return 0; }
        if (argc == 2) {
            auto model = cosyvoice_coreml::load(argv[1]);
            if (!model) throw std::runtime_error("Core ML fixture did not load");
            model->reset();
            const auto x_shape = model->dimensions("x");
            const auto cosine_shape = model->dimensions("cosine");
            const auto mask_shape = model->dimensions("mask");
            const auto logits_shape = model->dimensions("logits");
            if (model->stateful() || x_shape.size() != 3 || cosine_shape.size() != 4 ||
                mask_shape.size() != 4 || logits_shape.size() != 2)
                throw std::runtime_error("expected explicit-cache LM fixture");
            const auto prefix = lm_prefix(argv[1]);
            std::vector<float> x(prefix.begin(), prefix.begin() + x_shape.back());
            std::vector<float> cosine(cosine_shape.back(), 1), sine(cosine.size());
            std::vector<float> mask(mask_shape.back(), -1e4f), write(mask.size()), first(logits_shape.back()), second(first.size());
            const auto cache_shape = model->dimensions("keys");
            if (cache_shape.size() != 5 || !model->matches("keys_out", cache_shape) ||
                !model->matches("values", cache_shape) || !model->matches("values_out", cache_shape))
                throw std::runtime_error("unexpected LM cache schema");
            size_t count = 1;
            for (int64_t dimension : cache_shape) count *= dimension;
            std::vector<float> keys(count), values(count);
            mask[0] = 0;
            write[0] = 1;
            const auto predict = [&](std::vector<float> & output) {
                model->predict({{"x", x.data(), x_shape}, {"cosine", cosine.data(), cosine_shape},
                                {"sine", sine.data(), cosine_shape}, {"mask", mask.data(), mask_shape},
                                {"write_mask", write.data(), {1, 1, static_cast<int64_t>(write.size()), 1}},
                                {"keys", keys.data(), cache_shape}, {"values", values.data(), cache_shape}},
                               {{"logits", output.data(), logits_shape}, {"keys_out", keys.data(), cache_shape},
                                {"values_out", values.data(), cache_shape}});
            };
            predict(first);
            model->clear_predictions();
            std::vector<float> unchanged{12};
            if (model->predictions() != 0 || cosyvoice_coreml_f0(model, {}, 1, unchanged) ||
                model->predictions() != 0 || unchanged != std::vector<float>{12})
                throw std::runtime_error("Core ML activity survived fallback");
            std::fill(keys.begin(), keys.end(), 0);
            std::fill(values.begin(), values.end(), 0);
            model->reset();
            predict(second);
            if (first != second || model->predictions() != 1)
                throw std::runtime_error("Core ML reset/prediction contract failed");
            std::copy(prefix.begin() + x.size(), prefix.end(), x.begin());
            mask[1] = 0;
            write[0] = 0;
            write[1] = 1;
            auto native = cosyvoice_load_gguf(argv[1]);
            const auto hp = cosyvoice_qwen_hp(native);
            cosyvoice_free(native);
            for (size_t i = 0; i < cosine.size(); ++i) {
                const float angle = std::pow(hp.theta, -2.0f * i / hp.head_dim);
                cosine[i] = std::cos(angle);
                sine[i] = std::sin(angle);
            }
            std::vector<float> next(first.size());
            predict(next);
            const auto expected = native_lm_logits(argv[1], prefix);
            const std::vector<float> first_expected(expected.begin(), expected.begin() + first.size());
            const std::vector<float> next_expected(expected.begin() + first.size(), expected.end());
            const char * output = std::getenv("COSYVOICE_COREML_PARITY_OUTPUT");
            dump(output ? output : "", "lm-native.f32", expected);
            bool lm_passed = compare("lm-position-0", first_expected, first, 0.999, 0.5);
            lm_passed = compare("lm-position-1", next_expected, next, 0.999, 0.5) && lm_passed;
            if (!lm_passed)
                throw std::runtime_error("Core ML LM native parity failed");
            std::cout << model->label() << " stateful=" << model->stateful() << '\n';
            return 0;
        }
        check_coreml_fit();
        check_unavailable_stages();
        check_stage_retire_seam();
        check_memory_schedule_attribution();
        setenv("COSYVOICE_COREML_COMPUTE_UNITS", "disabled", 1);
        if (cosyvoice_coreml::load(test_tmpdir() + "/unavailable-cosyvoice.gguf"))
            throw std::runtime_error("disabled Core ML model was selected");
        unsetenv("COSYVOICE_COREML_COMPUTE_UNITS");
        std::cout << "CosyVoice Core ML unavailable-stage and disable checks passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
