#pragma once

// EncoderRoute: the Core ML encoder handle one unit of work acquires once, at
// its start, and holds until it ends (PLAN §2.10, §3.2.0).
//
// A unit is one top-level call: an offline transcription, a diarization, one
// streaming window. Every plan that depends on the sidecar (the long-form
// window plan, the independent-segment cap, the Nemotron cache-aware switch)
// is chosen from the route:
//   - a route that holds a handle runs the unit on Core ML;
//   - a null route runs it on ggml with exactly today's ggml-only plan, so a
//     sidecar that is still loading never changes a ggml transcript.
// A Core ML failure at window k retires the sidecar process-wide (later
// acquisitions return null) and windows k.. of the same unit run on ggml with
// the plan already chosen; the call then reports "mixed". The next call
// acquires a null route and reverts to the ggml plan.
//
// Portable on purpose (no Core ML or runtime types), so the routing logic and
// its tests build on every platform. Copies share one state: a failure seen
// through any copy is seen through all of them.

#include <memory>
#include <mutex>
#include <string>

struct parakeet_coreml_context;

namespace parakeet {

// What the long-form plan reads from the acquired handle: the encoder's I/O
// after load (§3.2.0 "Capacity"), so no pre-load descriptor is needed.
struct EncoderRouteGeometry {
    int  fixed_mel_frames    = 0;      // 0: flexible or unknown capacity
    int  seam_context_frames = 0;      // 0: the engine default
    bool masks_mel_padding   = false;  // fixed TDT exports with mel_valid_mask
};

class EncoderRoute {
public:
    // Plain function pointers: they cannot capture an engine or a ggml tensor.
    using RetireFn = void (*)(void * hold, const char * reason);
    using LockFn   = std::unique_lock<std::mutex> (*)(void * hold);
    using EncodeFn = int (*)(void * hold, parakeet_coreml_context * ctx,
                             int n_mel_frames, int mel_valid, int n_mels, const float * mel,
                             int n_enc_frames, int d_model, float * encoder_out);

    struct Ops {
        RetireFn retire = nullptr;  // called once, on the unit's first failure
        LockFn   lock   = nullptr;  // one in-flight prediction per model instance
        EncodeFn encode = nullptr;  // the sidecar forward pass
    };

    // The ggml route: today's ggml-only plan. `pending` records that the
    // unit runs on ggml only because the sidecar was not ready yet.
    EncoderRoute() = default;
    static EncoderRoute ggml(bool pending = false) {
        EncoderRoute r;
        r.pending_ = pending;
        return r;
    }

    // A route that holds a Core ML encoder for one unit. `hold` keeps the
    // model alive (a runtime handle, or the owner context's lifetime token).
    // `owner`: the sidecar owns the encoder (no ggml weights), so a failure is
    // an error instead of a fallback.
    static EncoderRoute coreml(parakeet_coreml_context * ctx, EncoderRouteGeometry geometry,
                               std::string label, std::shared_ptr<void> hold, Ops ops,
                               bool owner) {
        EncoderRoute r;
        r.s_ = std::make_shared<State>();
        r.s_->ctx = ctx;
        r.s_->geometry = geometry;
        r.s_->label = std::move(label);
        r.s_->hold = std::move(hold);
        r.s_->ops = ops;
        r.s_->owner = owner;
        return r;
    }

    // A handle was acquired for this unit; the plan is the Core ML plan even
    // after a failure (windows after the failure run on ggml).
    bool holds() const { return s_ != nullptr; }
    // The handle can still serve the next window of this unit.
    bool usable() const { return s_ != nullptr && !s_->failed; }
    bool failed() const { return s_ != nullptr && s_->failed; }
    bool owner() const { return s_ != nullptr && s_->owner; }
    bool pending() const { return pending_; }

    parakeet_coreml_context * context() const { return s_ ? s_->ctx : nullptr; }
    const EncoderRouteGeometry & geometry() const {
        static const EncoderRouteGeometry none;
        return s_ ? s_->geometry : none;
    }
    // The Core ML label of the held handle ("" for the ggml route).
    const std::string & coreml_label() const {
        static const std::string none;
        return s_ ? s_->label : none;
    }

    // Runs the sidecar under the model's prediction lock.
    int encode(int n_mel_frames, int mel_valid, int n_mels, const float * mel,
               int n_enc_frames, int d_model, float * encoder_out) const {
        if (!usable() || s_->ops.encode == nullptr) return -1;
        std::unique_lock<std::mutex> lock;
        if (s_->ops.lock != nullptr) lock = s_->ops.lock(s_->hold.get());
        return s_->ops.encode(s_->hold.get(), s_->ctx, n_mel_frames, mel_valid, n_mels, mel,
                              n_enc_frames, d_model, encoder_out);
    }

    std::unique_lock<std::mutex> lock_for_prediction() const {
        if (s_ == nullptr || s_->ops.lock == nullptr) return {};
        return s_->ops.lock(s_->hold.get());
    }

    // Window k failed: retire the sidecar process-wide and run the rest of
    // this unit on ggml. The owner route never retires (nothing to fall back to).
    void fail(const char * reason) const {
        if (s_ == nullptr || s_->failed) return;
        s_->failed = true;
        if (!s_->owner && s_->ops.retire != nullptr) s_->ops.retire(s_->hold.get(), reason);
    }

private:
    struct State {
        parakeet_coreml_context * ctx = nullptr;
        EncoderRouteGeometry geometry;
        std::string label;
        std::shared_ptr<void> hold;
        Ops ops;
        bool owner = false;
        bool failed = false;
    };
    std::shared_ptr<State> s_;
    bool pending_ = false;
};

// One encoder dispatch on a route: Core ML when the route can serve it and
// the call shape is eligible, otherwise ggml. A Core ML failure retires the
// sidecar and runs this dispatch on ggml (an owner route returns the error).
// `used_coreml` reports where the dispatch ran.
template <class CoremlFn, class GgmlFn>
int dispatch_on_route(const EncoderRoute & route, bool eligible, CoremlFn && run_coreml,
                      GgmlFn && run_ggml, bool & used_coreml, int * coreml_rc = nullptr) {
    used_coreml = false;
    if (eligible && route.usable()) {
        const int rc = run_coreml();
        if (coreml_rc != nullptr) *coreml_rc = rc;
        if (rc == 0) {
            used_coreml = true;
            return 0;
        }
        if (route.owner()) return rc;
        route.fail("prediction failed");
    }
    return run_ggml();
}

// Per-call backend label: "ggml" when nothing ran on Core ML, the Core ML
// label when everything did, "mixed" otherwise (§2.10 labels).
struct EncoderBackendTally {
    int coreml = 0;
    int ggml = 0;
    bool pending = false;
    std::string coreml_label;

    void add(bool used_coreml, const std::string & label) {
        if (used_coreml) {
            ++coreml;
            if (coreml_label.empty()) coreml_label = label;
            else if (coreml_label != label) coreml_label = "mixed";
        } else {
            ++ggml;
        }
    }
    void add_route(const EncoderRoute & route) { pending = pending || route.pending(); }

    std::string label() const {
        if (coreml == 0) return "ggml";
        if (ggml > 0) return "mixed";
        return coreml_label.empty() ? std::string("coreml") : coreml_label;
    }
    bool all_coreml() const { return coreml > 0 && ggml == 0; }
};

}  // namespace parakeet
