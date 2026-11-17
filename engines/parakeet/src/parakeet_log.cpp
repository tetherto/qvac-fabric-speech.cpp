// log_impl and log_set_callback; installs hook and forwards to ggml_log_set.

#include "parakeet_log.h"

#include "parakeet/log.h"

#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

namespace parakeet {

namespace {

// The sink and the calls in flight on it. Background work (Core ML sidecar
// loads on the runtime's load lane) can log after the engine that started it
// is gone, so log_set_callback waits for every callback still running before
// it returns: once a host has replaced or cleared its sink, its user data is
// never used again. A callback that logs through parakeet or replaces the sink
// itself does not wait for its own call.
std::mutex g_mutex;
std::condition_variable g_idle;
ggml_log_callback g_callback = nullptr;
void * g_user_data = nullptr;
int g_in_flight = 0;
thread_local int t_in_flight = 0;

}

void log_set_callback(ggml_log_callback cb, void * user_data) {
    std::unique_lock<std::mutex> lock(g_mutex);
    g_callback = cb;
    g_user_data = user_data;
    g_idle.wait(lock, [] { return g_in_flight == t_in_flight; });
    ggml_log_set(cb, user_data);
}

void log_impl(ggml_log_level level, const char * fmt, ...) {
    char stack_buf[1024];

    va_list args;
    va_start(args, fmt);
    va_list args_copy;
    va_copy(args_copy, args);
    const int n = std::vsnprintf(stack_buf, sizeof(stack_buf), fmt, args);
    va_end(args);

    const char * text = stack_buf;
    std::vector<char> heap_buf;
    if (n >= (int) sizeof(stack_buf)) {
        heap_buf.resize((size_t) n + 1);
        std::vsnprintf(heap_buf.data(), heap_buf.size(), fmt, args_copy);
        text = heap_buf.data();
    }
    va_end(args_copy);

    ggml_log_callback cb = nullptr;
    void * user_data = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        cb = g_callback;
        user_data = g_user_data;
        if (cb) {
            ++g_in_flight;
            ++t_in_flight;
        }
    }
    if (!cb) {
        std::fputs(text, stderr);
        return;
    }
    cb(level, text, user_data);
    std::lock_guard<std::mutex> lock(g_mutex);
    --g_in_flight;
    --t_in_flight;
    g_idle.notify_all();
}

}

extern "C" PARAKEET_API void parakeet_log_set(ggml_log_callback cb,
                                                        void *            user_data) {
    parakeet::log_set_callback(cb, user_data);
}
