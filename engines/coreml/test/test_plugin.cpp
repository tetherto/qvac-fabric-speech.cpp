// A dlopen-able module carrying its own copy of the runtime (namespace
// qvac_coreml_plugin). test-coreml-scheduler (one child process per case, so
// each starts with an unpinned image) requests a load whose LoadFn lives in
// this module and dlcloses the module: right after the request returns, while
// the first job is deferred by low power, or while the load is in flight. The
// image pin (dladdr + RTLD_NOLOAD | RTLD_NODELETE) must keep the code mapped
// for the scheduler threads.

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
std::shared_ptr<Manager> * g_manager = nullptr;  // leaked: the deferred case's own manager
std::atomic<bool> g_low_power{true};

bool plugin_low_power() { return g_low_power.load(); }

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

namespace {
SlotSpec plugin_spec(const char * dir) {
    SlotSpec spec;
    spec.engine = "plugin";
    spec.stage = "plugin";
    spec.function = "main";
    spec.units = ComputeUnits::cpu_only;
    spec.route = Route::prebuilt;
    spec.artifact_path = std::string(dir) + "/plugin.mlmodelc";
    spec.load_fn = &plugin_load;
    return spec;
}
}  // namespace

// The first job of this image, at minimal priority (it starts at once).
QVAC_PLUGIN_EXPORT int qvac_coreml_plugin_start(const char * dir) {
    g_slot = new std::shared_ptr<Slot>(Manager::instance()->make_slot(plugin_spec(dir), SlotPolicy{}));
    (*g_slot)->request(Priority::minimal);
    return 0;
}

// The first job of this image, at prewarm priority in low power: deferred, so
// no worker ever reaches LoadFn while the host unloads the module.
QVAC_PLUGIN_EXPORT int qvac_coreml_plugin_start_deferred(const char * dir) {
    Config cfg;
    cfg.low_power = &plugin_low_power;
    g_manager = new std::shared_ptr<Manager>(Manager::create(cfg));
    g_slot = new std::shared_ptr<Slot>((*g_manager)->make_slot(plugin_spec(dir), SlotPolicy{}));
    (*g_slot)->request(Priority::prewarm);
    return 0;
}

QVAC_PLUGIN_EXPORT void qvac_coreml_plugin_power_restored() {
    g_low_power = false;
    if (g_manager) (*g_manager)->notify();
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
