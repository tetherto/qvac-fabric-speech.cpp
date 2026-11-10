// Image pin (PLAN §2.9 "Library unload"). Pure POSIX: dladdr and
// RTLD_NOLOAD | RTLD_NODELETE exist in the macOS and iOS SDKs (dlfcn.h) and in
// glibc, so the dlclose test runs on the Linux lanes as well.

#include "device.h"

#include <atomic>

#include <dlfcn.h>

namespace QVAC_COREML_NS {
namespace detail {

namespace {
// 0 = not tried, 1 = pinned, 2 = not pinned (e.g. the main executable).
std::atomic<int> g_pin_state{0};
}  // namespace

bool pin_image() {
    const int state = g_pin_state.load();
    if (state != 0) return state == 1;
    Dl_info info{};
    void * self = reinterpret_cast<void *>(&pin_image);
    bool pinned = false;
    if (dladdr(self, &info) != 0 && info.dli_fname != nullptr) {
        // The returned handle is leaked on purpose: the image stays referenced
        // and, with RTLD_NODELETE, mapped for the life of the process.
        pinned = dlopen(info.dli_fname, RTLD_LAZY | RTLD_NOLOAD | RTLD_NODELETE) != nullptr;
    }
    int expected = 0;
    g_pin_state.compare_exchange_strong(expected, pinned ? 1 : 2);
    return g_pin_state.load() == 1;
}

}  // namespace detail
}  // namespace QVAC_COREML_NS
