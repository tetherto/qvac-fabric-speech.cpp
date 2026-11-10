// A dlopen-able module carrying its own copy of the runtime (namespace
// qvac_coreml_plugin). test-coreml-scheduler starts a load whose LoadFn lives
// in this module, dlcloses the module while the load is in flight, then lets
// the load finish: the image pin (dladdr + RTLD_NOLOAD | RTLD_NODELETE) must
// keep the code mapped for the scheduler thread.

#include "qvac_coreml/manager.h"

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>

using namespace QVAC_COREML_NS;

namespace {

std::mutex * g_mu = new std::mutex();
std::condition_variable * g_cv = new std::condition_variable();
bool g_released = false;
std::atomic<int> g_started{0};
std::shared_ptr<Slot> * g_slot = nullptr;  // leaked: the "engine" never drops it

LoadResult plugin_load(const LoadRequest &) {
    g_started.store(1);
    std::unique_lock<std::mutex> lk(*g_mu);
    g_cv->wait(lk, [] { return g_released; });
    LoadResult r;
    r.model = std::make_shared<int>(7);
    return r;
}

}  // namespace

#define QVAC_PLUGIN_EXPORT extern "C" __attribute__((visibility("default")))

QVAC_PLUGIN_EXPORT int qvac_coreml_plugin_start(const char * dir) {
    SlotSpec spec;
    spec.engine = "plugin";
    spec.stage = "plugin";
    spec.function = "main";
    spec.units = ComputeUnits::cpu_only;
    spec.route = Route::prebuilt;
    spec.artifact_path = std::string(dir) + "/plugin.mlmodelc";
    spec.load_fn = &plugin_load;
    g_slot = new std::shared_ptr<Slot>(Manager::instance()->make_slot(spec, SlotPolicy{}));
    (*g_slot)->request(Priority::minimal);
    return 0;
}

QVAC_PLUGIN_EXPORT int qvac_coreml_plugin_started() { return g_started.load(); }

QVAC_PLUGIN_EXPORT void qvac_coreml_plugin_release() {
    std::lock_guard<std::mutex> lk(*g_mu);
    g_released = true;
    g_cv->notify_all();
}

QVAC_PLUGIN_EXPORT int qvac_coreml_plugin_ready() {
    return g_slot && (*g_slot)->state() == SlotState::ready ? 1 : 0;
}
