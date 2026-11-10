#pragma once

// The per-library-image Core ML manager: scheduler, model registry, request
// groups and route planning (PLAN §2.9, §2.10, §2.11, §2.13).
//
// Scheduler (one per manager):
//  - a prepare queue (1 thread, QOS_CLASS_UTILITY) and a load lane (1 thread),
//    both priority-ordered (now > current > minimal > prewarm, then FIFO);
//  - an admission gate: one Core ML job (prepare, or load + warm) at a time;
//  - a job whose key lock is busy (would_block) releases the gate and re-queues
//    with backoff (1 s doubling to 30 s) while other jobs run;
//  - memory admission: a load whose units include the ANE and that has no
//    non-stale hint waits, reported as deferred(memory), until the available
//    memory reading reaches the slot's ane_compile_reserve_bytes; deferred jobs
//    hold neither the gate nor the lane;
//  - low power: `prewarm` work is deferred unless a waited group needs it.
//
// "Process-wide" means per engine library image: each engine library compiles
// its own copy of this runtime in its own namespace. instance() is leaked on
// purpose; on its first job it pins its own image (dladdr + dlopen
// RTLD_NOLOAD | RTLD_NODELETE) so code and callbacks outlive any dlclose, and an
// atexit flag stops new work.

#include "qvac_coreml/options.h"
#include "qvac_coreml/slot.h"
#include "qvac_coreml/status.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace QVAC_COREML_NS {

namespace detail {
struct Graveyard;
}  // namespace detail

struct Config {
    // Device facts; null/empty reads the device. Tests inject fakes.
    uint64_t (*available_memory)() = nullptr;
    bool (*low_power)() = nullptr;
    std::string host_id;
    std::string os_build;

    std::chrono::milliseconds memory_recheck{5000};  // deferred(memory) re-check period
    std::chrono::milliseconds backoff_initial{1000}; // would_block re-queue backoff
    std::chrono::milliseconds backoff_max{30000};
    std::chrono::milliseconds wait_poll{50};         // cancellation polling of waits
    bool pin_image = true;
};

struct Stats {
    uint64_t loads_started = 0;
    uint64_t loads_completed = 0;
    uint64_t load_failures = 0;
    uint64_t prepares_started = 0;
    uint64_t prepares_completed = 0;
    uint64_t requeues = 0;
    uint64_t cancelled = 0;
    uint64_t evictions = 0;
    uint64_t hint_writes = 0;
};

// ---- Route planning (§2.13, §2.14, §2.15) ---------------------------------

struct PlanInput {
    bool coreml_built = false;        // <E>_COREML compiled in
    bool generation_built = false;    // the build embeds templates (<E>_COREML_GENERATE)
    bool prebuilt_present = false;    // adjacent <stem><suffix>.mlmodelc by today's path rules
    bool template_available = false;  // the library embeds a template for this stage
    bool template_matched = false;    // template.json.match accepted the GGUF
    bool multifunction_ane = false;   // template needs macOS 15.7 (F11)
    bool device_excluded = false;     // template prefer_ggml_on matched this device
    std::vector<ComputeUnits> validated_units;  // template.json memory.validated_for units
    ComputeUnits units = ComputeUnits::all;     // resolved units for this stage
    Platform platform = Platform::other;
    OsVersion os;
};

struct Plan {
    Route route = Route::none;
    SlotState none_state = SlotState::unsupported;  // when route == none
    std::string reason;
};

// Decides from policy, OS, build, device class, template match and adjacent
// prebuilt presence; never reads the cache, so fit is identical before and
// after the first load. A generated route whose resolved units are not in
// validated_units is suppressed as unsupported(unvalidated_units); prebuilt
// routes are never suppressed by validation.
Plan plan(const PlanInput & in, const Resolved & resolved);

// ---- Request groups (§2.10) -----------------------------------------------

enum class GroupKind { construction, prewarm };

const char * to_string(GroupKind k);

struct GroupRequest {
    std::shared_ptr<Slot> slot;
    Priority priority = Priority::minimal;
};

class Group {
public:
    // Every slot is terminal, deferred(memory), idle (cancelled) or reached
    // `ready` since the group was issued.
    bool finished() const;
    // Blocks until finished, `timeout` or `cancel` (polled every 50 ms).
    PrewarmResult wait(std::optional<Clock::duration> timeout = std::nullopt,
                       const std::atomic<bool> * cancel = nullptr);
    PrewarmResult result() const;

private:
    friend class Manager;
    struct Entry {
        std::weak_ptr<detail::SlotCore> core;
        SlotStatus last;       // last observed status (ready snapshot once it reached ready)
        uint64_t baseline = 0; // ready events at issue
        bool reached_ready = false;
    };
    // Under the runtime mutex; strong references go to the caller's graveyard.
    PrewarmResult collect(bool & finished, detail::Graveyard & gy) const;

    GroupKind kind_ = GroupKind::construction;
    std::shared_ptr<detail::Runtime> rt_;
    mutable std::vector<Entry> entries_;
    std::vector<SlotStatus> skipped_;
};

class Manager {
public:
    Manager(const Manager &) = delete;
    Manager & operator=(const Manager &) = delete;
    // Stops this manager's threads. The per-image instance() is never destroyed.
    ~Manager();

    // The per-library-image manager (leaked on purpose).
    static std::shared_ptr<Manager> instance();
    // An independent manager (tests; "the next process").
    static std::shared_ptr<Manager> create(Config cfg = {});

    std::shared_ptr<Slot> make_slot(SlotSpec spec, SlotPolicy policy);
    std::shared_ptr<ResidentPool> make_pool(size_t cap);

    // Issues a group: requests every slot at its priority. `waited` marks the
    // group as explicitly waited on (wait:'construct' or coreml_prewarm(...,
    // wait=true)), which makes its prewarm work non-deferrable in low power.
    // `skipped_route_none` lists slots whose plan() route is none.
    std::shared_ptr<Group> issue_group(GroupKind kind, const std::vector<GroupRequest> & requests, bool waited,
                                       std::vector<SlotStatus> skipped_route_none = {});

    // Re-run admission now (a memory reading or the low-power state changed).
    void notify();

    EngineStatus status(const std::string & engine, const std::string & cache_dir,
                        const std::vector<std::shared_ptr<Slot>> & slots, GenerationStatus generation) const;

    const std::string & host_id() const;
    const std::string & os_build() const;
    OsVersion os_version() const;
    Platform platform() const;

    // Test introspection.
    Stats stats() const;
    std::vector<std::string> events() const;

private:
    explicit Manager(std::shared_ptr<detail::Runtime> rt) : rt_(std::move(rt)) {}
    std::shared_ptr<detail::Runtime> rt_;
};

}  // namespace QVAC_COREML_NS
