#pragma once

// Internal scheduler state shared by scheduler.cpp and manager.cpp.
//
// Locking rule: Runtime::mu guards every field marked (mu) here and in SlotCore,
// Job and PoolCore. SharedModel and SlotCore destructors take Runtime::mu, so a
// strong reference to either must never be released while mu is held: code
// that drops or temporarily locks one under mu moves it into a Graveyard that
// is declared before the lock and therefore destroyed after it is released.

#include "qvac_coreml/manager.h"
#include "qvac_coreml/slot.h"
#include "qvac_coreml/status.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace QVAC_COREML_NS {
namespace detail {

struct Graveyard {
    std::vector<std::shared_ptr<void>> items;
    template <class T> void keep(const std::shared_ptr<T> & p) {
        if (p) items.push_back(p);
    }
};

// The atexit flag (§2.9 "Process exit"): stops new work in this image.
bool exiting();
void register_atexit_once();

struct SlotCore;
struct PoolCore;

enum class JobKind { prepare, load };

struct Sub {
    std::weak_ptr<SlotCore> core;
    uint64_t epoch = 0;
    Priority prio = Priority::prewarm;
    bool non_deferrable = false;
};

struct Job {
    JobKind kind = JobKind::load;
    uint64_t seq = 0;
    std::string dedup_key;  // key in Runtime::inflight
    std::string model_key;  // (artifact, function, units): registry and retirement key
    LoadRequest load;
    LoadFn load_fn = nullptr;
    WarmFn warm_fn = nullptr;
    PrepareRequest prepare;
    PrepareFn prepare_fn = nullptr;
    bool shareable = true;
    bool ane = false;
    uint64_t ane_reserve = 0;
    std::string hint_path;
    std::optional<int64_t> expected_warm_ms;

    // (mu)
    std::vector<Sub> subs;
    bool started = false;
    bool done = false;
    bool completing = false;  // load and warm finished; publishing
    SlotState phase = SlotState::queued;
    Clock::time_point enqueued_at{};
    Clock::time_point started_at{};
    Clock::time_point phase_at{};
    Clock::time_point retry_at{};
    int64_t backoff_ms = 0;
    bool deferred_memory = false;
    Clock::time_point deferred_since{};
    // The last worker screening found the resident cap denying this load. Only
    // for the event and the wake-up: acquisitions decide the denial when they
    // ask (Runtime::cap_blocked_now_locked), never from this cached verdict.
    bool cap_blocked = false;
    std::vector<std::weak_ptr<PoolCore>> pools;  // pools served while running
    Timings timings;

    std::atomic<bool> hinted_at_start{false};
    std::atomic<bool> wait_expired{false};  // a bounded wait expired on this slow load
};

struct SlotCore {
    SlotSpec spec;
    SlotPolicy policy;
    std::shared_ptr<Runtime> rt;
    // Runtime-local identity, set once by Runtime::add_slot: the dedup key of
    // its non-shareable jobs. Kept across evictions; a new slot never reuses
    // it, even at the same address while a destroyed slot's job still runs.
    uint64_t id = 0;

    // (mu)
    SlotState state = SlotState::idle;
    std::string reason;
    Error error;
    uint64_t epoch = 0;
    std::shared_ptr<Job> job;
    std::shared_ptr<SharedModel> model;
    std::string artifact_path;
    std::string artifact_key;
    std::string hint_path;
    StatusRoute status_route = StatusRoute::none;
    uint64_t ready_events = 0;
    SlotStatus ready_snapshot;
    Clock::time_point last_used{};
    bool queued_behind = false;
    Timings timings;
    std::weak_ptr<PoolCore> pool;
};

// An instance a pool counts because units retain it: the Handles acquired
// through the pool's members. It counts while `holds` > 0, whatever happened to
// those slots meanwhile (evicted, reloaded or destroyed).
struct PoolRef {
    const SharedModel * model = nullptr;  // identity only; alive while holds > 0
    std::atomic<int> holds{0};
};

// §2.9 resident cap. A pool counts the distinct instances it references: those
// its live members hold, plus those units retain through it (PoolRef), plus the
// load in flight for it, plus the instances its members dropped (evicted,
// released, retired or replaced) whose strong reference is still on its way
// out (`releasing`). Each is counted once, by identity, however many of these
// reference it. Another owner's reference to the same instance never counts
// here: a dropped instance stops counting once this pool's references to it are
// released, whoever else still holds it.
struct PoolCore {
    size_t cap = 0;
    // (mu)
    std::vector<std::weak_ptr<SlotCore>> members;
    std::vector<std::weak_ptr<SlotCore>> predicted;
    std::vector<std::shared_ptr<PoolRef>> refs;
    // One entry per pending release (Runtime::drop_model_locked), by instance
    // identity: an instance two members dropped is listed twice, counted once.
    std::vector<const SharedModel *> releasing;
};

class Runtime : public std::enable_shared_from_this<Runtime> {
public:
    explicit Runtime(Config cfg);
    ~Runtime();
    Runtime(const Runtime &) = delete;
    Runtime & operator=(const Runtime &) = delete;

    const Config cfg;
    std::string host_id;
    std::string os_build;
    OsVersion os;
    Platform platform = Platform::other;

    // ---- API used by the facades (each takes mu itself) ----
    void add_slot(const std::shared_ptr<SlotCore> & c);
    void request(const std::shared_ptr<SlotCore> & c, Priority p, bool non_deferrable);
    Acquired acquire(const std::shared_ptr<SlotCore> & c, Demand d, std::optional<Clock::time_point> deadline,
                     const std::atomic<bool> * cancel);
    void evict(const std::shared_ptr<SlotCore> & c);
    void release_slot(const std::shared_ptr<SlotCore> & c);
    void retire_slot(const std::shared_ptr<SlotCore> & c, const std::string & reason);
    void retire_model(SharedModel & m, const std::string & reason);
    SlotState state(const std::shared_ptr<SlotCore> & c);
    SlotStatus status(const std::shared_ptr<SlotCore> & c);
    std::string hint_path(const std::shared_ptr<SlotCore> & c);
    void pool_add(PoolCore & p, const std::shared_ptr<SlotCore> & c);
    void pool_set_predicted(PoolCore & p, const std::vector<std::shared_ptr<SlotCore>> & cores);
    size_t pool_live(const PoolCore & p);
    void on_model_released(const std::string & key);
    // A pending release of instance `m` from pool `p` (drop_model_locked)
    // completed outside mu: its reference is gone.
    void on_pool_released(const std::weak_ptr<PoolCore> & p, const SharedModel * m);
    void notify();
    void stop();
    // Pins this image (§2.9) before work is requested; never under mu. Every
    // entry point that reaches request_locked() calls it first.
    void pin_for_work();
    Stats stats_snapshot();
    std::vector<std::string> events_snapshot();

    // Groups (manager.cpp) use these under mu.
    std::mutex mu;
    std::condition_variable state_cv;
    void request_locked(const std::shared_ptr<SlotCore> & c, Priority p, bool nd, Graveyard & gy);
    SlotStatus status_locked(SlotCore & c, Clock::time_point now) const;
    // The resident cap denies slot `c` the instance its load job brings, right
    // now (§2.9): acquisitions and groups do not wait on it. Decided at the
    // call, from the current holds, predictions, residents and reservations.
    bool cap_blocked_now_locked(const SlotCore & c, Graveyard & gy) const;

private:
    struct WaitPlan {
        bool wait = false;
        int64_t w_ms = 0;
    };
    struct JobView {
        bool live = false;
        Priority prio = Priority::prewarm;
        bool deferrable = false;
    };
    enum class PoolVerdict { admit, wait_release, blocked };

    void worker_main(JobKind kind);
    void ensure_threads_locked();
    void bump_locked();
    void event_locked(std::string e);

    std::shared_ptr<SlotCore> live_core(const Sub & s, const Job & j, Graveyard & gy) const;
    JobView view_locked(const Job & j, Graveyard & gy) const;
    void set_subs_state_locked(Job & j, SlotState st, Graveyard & gy);
    void cancel_locked(const std::shared_ptr<Job> & j);
    void unsubscribe_locked(const std::shared_ptr<SlotCore> & c, Graveyard & gy);
    void evict_locked(const std::shared_ptr<SlotCore> & c, Graveyard & gy);
    // Drops slot `c`'s model into `gy`. A pooled slot's reference always leaves
    // as a pending release: the pool counts the instance until that reference
    // is released outside mu (§2.9: a replacement is admitted only then),
    // whether or not another member or a Handle also counts it meanwhile.
    void drop_model_locked(SlotCore & c, Graveyard & gy);
    void retire_model_locked(SharedModel & m, const std::string & reason, Graveyard & gy);
    void subscribe_load_locked(const std::shared_ptr<SlotCore> & c, Priority p, bool nd, Graveyard & gy);
    void subscribe_prepare_locked(const std::shared_ptr<SlotCore> & c, Priority p, bool nd);
    bool low_power_locked() const;
    bool hinted(const std::string & hint_path) const;
    int64_t job_w(const Job & j) const;
    std::shared_ptr<Job> pick_locked(Clock::time_point now, Clock::time_point & wake, Graveyard & gy);
    PoolVerdict pool_admits_locked(Job & j, Graveyard & gy);
    void set_cap_blocked_locked(Job & j, bool blocked);
    size_t pool_live_locked(const PoolCore & p, Graveyard & gy) const;
    bool pool_counts_locked(const PoolCore & p, const SharedModel * m, Graveyard & gy) const;
    bool pool_takes_locked(const SlotCore & c, const SharedModel * m, Graveyard & gy) const;
    bool running_serves_locked(const PoolCore & p) const;
    void reserve_locked(Job & j, const SlotCore & c, Graveyard & gy);
    bool sub_denied_locked(const Job & j, const SlotCore & c, const std::vector<std::shared_ptr<SlotCore>> & own,
                           Graveyard & gy) const;
    const SharedModel * existing_locked(const Job & j, Graveyard & gy) const;
    void job_pools_locked(const Job & j, std::vector<std::shared_ptr<SlotCore>> & own,
                          std::vector<std::shared_ptr<PoolCore>> & pools, Graveyard & gy) const;
    std::vector<std::shared_ptr<SlotCore>> pool_victim_locked(const PoolCore & p,
                                                              const std::vector<std::shared_ptr<SlotCore>> & own,
                                                              Graveyard & gy) const;
    bool cap_denied_locked(const Job & j, Graveyard & gy) const;
    void screen_locked(Clock::time_point now, Clock::time_point & wake, Graveyard & gy);
    std::shared_ptr<PoolRef> pool_ref_locked(const SlotCore & c);
    void start_locked(const std::shared_ptr<Job> & j, Clock::time_point now, Graveyard & gy);
    void set_phase(const std::shared_ptr<Job> & j, SlotState st);
    void run_load(const std::shared_ptr<Job> & j);
    void run_prepare(const std::shared_ptr<Job> & j);
    void publish(const std::shared_ptr<Job> & j, const std::shared_ptr<SharedModel> & m, bool loaded,
                 const std::optional<HintView> & hint);
    void fail(const std::shared_ptr<Job> & j, const Error & e, SlotState st = SlotState::failed);
    void requeue(const std::shared_ptr<Job> & j);
    std::optional<HintView> record_hint(const Job & j, int64_t ms, bool warm);
    void write_stale(const std::string & path);
    WaitPlan plan_bounded_wait_locked(SlotCore & c, std::optional<Clock::time_point> deadline,
                                      Clock::time_point now, Graveyard & gy);

    // (mu)
    std::condition_variable work_cv;
    std::vector<std::shared_ptr<Job>> queue_;
    std::shared_ptr<Job> running_;
    std::unordered_map<std::string, std::shared_ptr<Job>> inflight_;
    std::unordered_map<std::string, std::weak_ptr<SharedModel>> registry_;
    std::unordered_map<std::string, std::string> retired_;
    std::vector<std::weak_ptr<SlotCore>> slots_;
    uint64_t seq_ = 0;
    uint64_t slot_ids_ = 0;
    uint64_t gen_ = 0;
    int handoff_ = -1;  // JobKind the other worker must take
    bool stopping_ = false;
    bool threads_started_ = false;
    std::thread prepare_thread_;
    std::thread load_thread_;
    Stats stats_;
    std::vector<std::string> events_;

    std::mutex hint_mu_;  // serialises hint read-modify-write within this image
};

std::string model_key(const std::string & artifact, const std::string & function, ComputeUnits units);

}  // namespace detail
}  // namespace QVAC_COREML_NS
