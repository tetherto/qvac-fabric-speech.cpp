#pragma once

// A cancellation is consumed by the generation that observes it, never erased
// at startup: cancel() arriving between a caller's own precheck and generate()
// entry must cancel that run instead of being discarded. Clearing the flag on
// scope exit keeps the next run clean on every return path.
//
// Mirrors the MiniMax engine's CancellationScope; kept in a header so the
// contract is unit-testable without model weights.

#include <atomic>
#include <mutex>
#include <set>

namespace tts_cpp::acestep {

class CancellationScope {
public:
    explicit CancellationScope(std::atomic<bool> & cancelled) : cancelled_(cancelled) {}

    CancellationScope(const CancellationScope &) = delete;
    CancellationScope & operator=(const CancellationScope &) = delete;

    ~CancellationScope() {
        cancelled_.store(false);
    }

private:
    std::atomic<bool> & cancelled_;
};

// Engine::cancel() for one Engine: its generations (generate(), understand())
// and its blocking Core ML waits (coreml_wait(), a waited coreml_prewarm()).
//
// Every cancel() ends every wait in flight, each through its own flag, so no wait clears a cancel meant for another wait or for a
// generation. The generation flag (`flag`, consumed by the generation that
// observes it) is set when a generation is in flight or when nothing is (a
// cancel armed before generate() still cancels that run), but not by a cancel
// that only waits observed: cancelling a wait leaves the next generate() to
// run.
class EngineCancel {
public:
    std::atomic<bool> flag{ false };

    void cancel() {
        std::lock_guard<std::mutex> lock(mu_);
        for (std::atomic<bool> * w : waits_) w->store(true);
        if (calls_ > 0 || waits_.empty()) flag.store(true);
    }

    // A generation in flight; the flag is cleared when the last one ends.
    class Call {
    public:
        explicit Call(EngineCancel & c) : c_(c) {
            std::lock_guard<std::mutex> lock(c_.mu_);
            ++c_.calls_;
        }
        ~Call() {
            std::lock_guard<std::mutex> lock(c_.mu_);
            if (--c_.calls_ == 0) c_.flag.store(false);
        }
        Call(const Call &) = delete;
        Call & operator=(const Call &) = delete;

    private:
        EngineCancel & c_;
    };

    // A blocking wait in flight: `stop()` is set by any cancel() made while it
    // exists, and by nothing else.
    class Wait {
    public:
        explicit Wait(EngineCancel & c) : c_(c) {
            std::lock_guard<std::mutex> lock(c_.mu_);
            c_.waits_.insert(&stop_);
        }
        ~Wait() {
            std::lock_guard<std::mutex> lock(c_.mu_);
            c_.waits_.erase(&stop_);
        }
        Wait(const Wait &) = delete;
        Wait & operator=(const Wait &) = delete;
        const std::atomic<bool> * stop() const { return &stop_; }

    private:
        EngineCancel & c_;
        std::atomic<bool> stop_{ false };
    };

private:
    std::mutex mu_;
    int calls_ = 0;
    std::set<std::atomic<bool> *> waits_;
};

}
