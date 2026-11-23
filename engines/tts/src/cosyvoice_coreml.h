#pragma once

// CosyVoice3 Core ML sidecars.
//
// cosyvoice_coreml wraps one Core ML model and its feature shapes. The LM and
// the flow DiT sidecars are opened synchronously next to their GGUFs
// (load()). The flow front-end, the HiFT F0 predictor and the HiFT decoder
// run on the shared Core ML runtime instead (cosyvoice_coreml_stages.h): a
// prebuilt sidecar next to the GGUF loads in the background, and without one,
// with a cache directory on macOS 15 or later, the sidecar is generated from
// the GGUF (templates cosyvoice3-frontend-f1024, cosyvoice3-hift-f0-m512 and
// cosyvoice3-hift-m512). Each of those stages takes its model once, at the
// start of its unit (one front-end call, one F0 call, one HiFT decode), and
// runs on ggml with exactly today's plan when there is none. A prediction
// that fails retires the stage's model and the unit is redone on ggml from
// its inputs.

#include <atomic>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct cosyvoice_coreml_input {
    const char * name;
    const float * data;
    std::vector<int64_t> shape;
    bool finite_already_checked = false;
};

struct cosyvoice_coreml_output {
    const char * name;
    float * data;
    std::vector<int64_t> shape;
};

class cosyvoice_coreml {
public:
    // A model the shared Core ML runtime holds for one unit of work: every
    // prediction takes the model's lock, and a failure retires it (for every
    // engine on it) so the unit is redone on ggml, or fails under strict.
    struct binding {
        std::function<std::unique_lock<std::mutex>()> lock;
        std::function<void(const std::string &)> retire;
        bool strict = false;
    };
    // A stand-in for a Core ML model (tests): declared feature shapes and a
    // prediction function returning 0 on success, as moss_coreml_predict does.
    using fake_predict = std::function<int(const char * const * input_names, const float * const * inputs,
                                           int n_inputs, const char * const * output_names,
                                           float * const * outputs, int n_outputs)>;

    static bool has_sidecar(const std::string & gguf, const std::string & suffix = "");
    static std::shared_ptr<cosyvoice_coreml> load(const std::string & gguf, const std::string & suffix = "");
    // Wraps `moss_model` (a moss_coreml_model kept alive by `owner`) for one
    // unit of work on the shared runtime.
    static std::shared_ptr<cosyvoice_coreml> borrow(std::shared_ptr<void> owner, void * moss_model,
                                                    std::string label, binding b);
    static std::shared_ptr<cosyvoice_coreml> fake(std::map<std::string, std::vector<int64_t>> dims,
                                                  fake_predict predict, std::string label, binding b);
    std::vector<int64_t> dimensions(const char * name) const;
    bool matches(const char * name, const std::vector<int64_t> & shape) const;
    bool stateful() const;
    void reset();
    void clear_predictions() { predictions_.store(0); }
    void predict(std::initializer_list<cosyvoice_coreml_input> inputs,
                 std::initializer_list<cosyvoice_coreml_output> outputs);
    const std::string & label() const { return label_; }
    uint64_t predictions() const { return predictions_.load(); }

    // Held by the shared runtime: a failure retires it instead of throwing.
    bool retirable() const { return static_cast<bool>(binding_.retire); }
    bool strict() const { return binding_.strict; }
    void retire(const std::string & reason);
    bool retired() const { return retired_.load(); }

private:
    std::shared_ptr<void> model_;
    std::string label_;
    std::atomic<uint64_t> predictions_{0};
    binding binding_;
    std::atomic<bool> retired_{false};
    std::map<std::string, std::vector<int64_t>> fake_dims_;
    fake_predict fake_predict_;
};

// Each runs one stage on `model` and returns true, or returns false with the
// outputs untouched so the caller runs the stage on ggml: no model, a model
// that cannot serve the input (too many frames, other features), or a
// prediction that failed on a model the shared runtime holds (it is retired).
// A legacy model's failure throws; under strict (use = require) a model that
// cannot serve the input or fails throws as well.
bool cosyvoice_coreml_frontend(const std::shared_ptr<cosyvoice_coreml> & model,
                               const std::vector<int32_t> & tokens, const std::vector<float> & speaker,
                               std::vector<float> & mu, std::vector<float> & spks);
bool cosyvoice_coreml_euler(const std::shared_ptr<cosyvoice_coreml> & model, int head_dim,
                           const std::vector<float> & mu, const std::vector<float> & cond,
                           const std::vector<float> & spks, int frames, std::vector<float> & x);
bool cosyvoice_coreml_f0(const std::shared_ptr<cosyvoice_coreml> & model,
                        const std::vector<float> & mel, int frames, std::vector<float> & f0);
bool cosyvoice_coreml_spectrum(const std::shared_ptr<cosyvoice_coreml> & model,
                              const std::vector<float> & mel, int frames,
                              const std::vector<float> & source, int source_frames,
                              std::vector<float> & spectrum);
