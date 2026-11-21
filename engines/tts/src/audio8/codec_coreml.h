#pragma once

#include "tts-cpp/coreml_options.h"
#include "tts-cpp/fit.h"

#include <atomic>
#include <memory>

#ifdef QVAC_COREML_NS
#include "qvac_coreml/slot.h"
#endif

namespace tts_cpp::audio8::detail {

struct codec_model;
struct codec_source;

#ifdef QVAC_COREML_NS
struct codec_source {
    QVAC_COREML_NS::SourceStat stat;
};
#endif

// A single acquired model, held by the caller and its stream worker until
// the call finishes. The worker uses the handle to predict and retire a
// failed model. The caller owns acquisition and replay.
class codec_coreml_model {
  public:
    virtual ~codec_coreml_model() = default;
    virtual int window() const = 0;
    virtual const char * label() const = 0;
    virtual bool synthesize(const float * post, float * pcm) const = 0;
    virtual void retire(const char * reason) const = 0;
};

struct codec_coreml_call {
    std::shared_ptr<const codec_coreml_model> model;
    bool pending = false;
    bool strict = false;
};

enum class codec_budget { unprojected, projected, degrade };

class codec_coreml {
  public:
    static std::shared_ptr<codec_coreml> create(const codec_model & codec,
                                                const std::string & path,
                                                const codec_source & source,
                                                const CoremlOptions & options,
                                                codec_budget budget);
    ~codec_coreml();

    codec_coreml_call acquire(const std::atomic<bool> * cancel) const;
    std::string status() const;
    std::string construction_json() const;
    std::string prewarm(CoremlOptions::Prewarm scope, bool wait,
                        const std::atomic<uint64_t> & cancels) const;
    bool wait(int64_t timeout_ms,
              const std::atomic<uint64_t> * cancels = nullptr) const;

  private:
    struct Impl;
    explicit codec_coreml(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

void validate_codec_coreml_options(const CoremlOptions & options);

#ifdef TTS_CPP_USE_COREML
bool codec_needs_budget_projection(const std::string & path,
                                   const CoremlOptions & options);
uint64_t prebuilt_codec_ane_reserve(const std::string & path);
#endif

// Fit plans only from a template match and adjacent sidecar presence. No
// model load, generation, cache or hint lookup takes place here.
FitCoreml fit_codec_coreml(const std::string & path,
                           const CoremlOptions & options, uint64_t ggml_bytes,
                           uint64_t free_bytes, int frames,
                           const codec_model & model);

} // namespace tts_cpp::audio8::detail
