// The host log sink is never called after the host replaced or cleared it.
// Background work (a Core ML sidecar load finishing after its engine was
// destroyed) logs from another thread; parakeet_log_set must return only
// after every callback still running on the old sink has returned, so the
// host may free that sink's user data at once.
//
//   drain        a callback in flight when the sink is cleared finishes
//                before parakeet_log_set returns
//   reentrant    a callback that logs through parakeet, or clears the sink
//                itself, does not deadlock
//
// Hermetic; exit 0 = pass, 1 = fail.

#include "parakeet/log.h"
#include "parakeet_log.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

namespace {

int g_failures = 0;

void expect(bool ok, const char * what) {
    std::printf("[%s] %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++g_failures;
}

struct Sink {
    std::atomic<bool> entered{false};
    std::atomic<bool> freed{false};        // the host released its user data
    std::atomic<bool> used_after_free{false};
};

void slow_callback(ggml_log_level, const char *, void * user) {
    auto * sink = static_cast<Sink *>(user);
    sink->entered = true;
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    if (sink->freed) sink->used_after_free = true;
}

void case_drain() {
    Sink sink;
    parakeet_log_set(&slow_callback, &sink);
    std::thread worker([] { PARAKEET_LOG_INFO("a background load finished\n"); });
    while (!sink.entered) std::this_thread::yield();
    parakeet_log_set(nullptr, nullptr);
    sink.freed = true;  // what a host does right after clearing its sink
    worker.join();
    expect(!sink.used_after_free, "drain: the sink's user data is not used after parakeet_log_set returns");
}

std::atomic<int> g_nested{0};

void reentrant_callback(ggml_log_level, const char * text, void *) {
    if (g_nested++ == 0) {
        PARAKEET_LOG_INFO("nested: %s", text);
        parakeet_log_set(nullptr, nullptr);
    }
}

void case_reentrant() {
    parakeet_log_set(&reentrant_callback, nullptr);
    PARAKEET_LOG_INFO("outer\n");
    expect(g_nested.load() == 2, "reentrant: a callback may log and clear the sink without deadlock");
    parakeet_log_set(nullptr, nullptr);
}

}  // namespace

int main() {
    case_drain();
    case_reentrant();
    std::printf("test-log-unregister: %d failure(s)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
