// test-coreml-scheduler (PLAN §2.16, §2.9, §2.12): the shared Core ML runtime
// with fake LoadFn / WarmFn / PrepareFn only. Portable: Linux and macOS.
//
//   test-coreml-scheduler --plugin <path-to-qvac-coreml-test-plugin> [--only <case>]
//
// Every case prints "ok <name>" or FAIL lines; any failure exits 1. There is no
// skip path.

#include "qvac_coreml/manager.h"

#include "hints.h"
#include "sha256.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <dlfcn.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char ** environ;

using namespace QVAC_COREML_NS;
namespace fs = std::filesystem;
using namespace std::chrono_literals;
using std::chrono::milliseconds;

namespace {

// ---- harness ---------------------------------------------------------------

int g_checks = 0;
int g_failures = 0;
std::string g_case;
std::string g_plugin;
std::string g_self;

void check(bool ok, const char * what, const char * file, int line) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::fprintf(stderr, "FAIL [%s] %s:%d: %s\n", g_case.c_str(), file, line, what);
    }
}
#define CHECK(c) check(static_cast<bool>(c), #c, __FILE__, __LINE__)

int64_t ms_since(Clock::time_point t) {
    return std::chrono::duration_cast<milliseconds>(Clock::now() - t).count();
}

// Runs `fn` on a detached thread; if it has not returned within `timeout_ms`
// the runtime is deadlocked, so fail and exit at once (joining would hang).
void must_complete(const char * what, std::function<void()> fn, int timeout_ms = 3000) {
    auto done = std::make_shared<std::atomic<bool>>(false);
    std::thread([fn = std::move(fn), done] {
        fn();
        done->store(true);
    }).detach();
    const auto end = Clock::now() + milliseconds(timeout_ms);
    while (!done->load() && Clock::now() < end) std::this_thread::sleep_for(1ms);
    if (!done->load()) {
        std::fprintf(stderr, "FAIL [%s] %s did not complete within %d ms (deadlock)\n", g_case.c_str(), what,
                     timeout_ms);
        std::fflush(stderr);
        _exit(1);
    }
}

bool wait_for(const std::function<bool()> & pred, int timeout_ms = 5000) {
    const auto end = Clock::now() + milliseconds(timeout_ms);
    while (Clock::now() < end) {
        if (pred()) return true;
        std::this_thread::sleep_for(1ms);
    }
    return pred();
}

// ---- fake world --------------------------------------------------------------

struct Behaviour {
    bool block = false;       // wait for release(<key>)
    int delay_ms = 0;
    bool fail = false;
    ErrorClass cls = ErrorClass::transient;
    int would_block = 0;      // first N calls report a busy key lock
    bool warm_block = false;  // WarmFn waits for release("warm:<fn>")
    bool warm_fail = false;
};

struct World {
    std::mutex mu;
    std::condition_variable cv;
    std::map<std::string, Behaviour> beh;
    std::set<std::string> released;
    bool release_all = false;
    std::map<std::string, int> calls;
    std::vector<std::string> log;
    std::set<std::string> in_progress;
    int active = 0;
    int max_active = 0;
};

World * W = new World();
std::atomic<int> g_live_models{0};
std::atomic<uint64_t> g_mem{uint64_t(64) << 30};
std::atomic<bool> g_low_power{false};

struct FakeModel {
    explicit FakeModel(std::string n) : name(std::move(n)) { ++g_live_models; }
    ~FakeModel() { --g_live_models; }
    std::string name;
};

void reset_world() {
    std::lock_guard<std::mutex> lk(W->mu);
    W->beh.clear();
    W->released.clear();
    W->release_all = false;
    W->calls.clear();
    W->log.clear();
    W->in_progress.clear();
    W->active = 0;
    W->max_active = 0;
    g_mem = uint64_t(64) << 30;
    g_low_power = false;
}

void set_beh(const std::string & k, Behaviour b) {
    std::lock_guard<std::mutex> lk(W->mu);
    W->beh[k] = b;
}

void release(const std::string & k) {
    std::lock_guard<std::mutex> lk(W->mu);
    W->released.insert(k);
    W->cv.notify_all();
}

void unrelease(const std::string & k) {
    std::lock_guard<std::mutex> lk(W->mu);
    W->released.erase(k);
}

void release_everything() {
    std::lock_guard<std::mutex> lk(W->mu);
    W->release_all = true;
    W->cv.notify_all();
}

int calls(const std::string & k) {
    std::lock_guard<std::mutex> lk(W->mu);
    auto it = W->calls.find(k);
    return it == W->calls.end() ? 0 : it->second;
}

bool in_progress(const std::string & k) {
    std::lock_guard<std::mutex> lk(W->mu);
    return W->in_progress.count(k) > 0;
}

int max_active() {
    std::lock_guard<std::mutex> lk(W->mu);
    return W->max_active;
}

std::vector<std::string> log_snapshot() {
    std::lock_guard<std::mutex> lk(W->mu);
    return W->log;
}

int log_index(const std::vector<std::string> & log, const std::string & e, int nth = 0) {
    for (size_t i = 0; i < log.size(); ++i) {
        if (log[i] == e && nth-- == 0) return static_cast<int>(i);
    }
    return -1;
}

// Shared by the fakes: enter, honour the behaviour, leave.
Behaviour enter(const std::string & key, const std::string & log_entry, int & n) {
    std::lock_guard<std::mutex> lk(W->mu);
    n = ++W->calls[key];
    W->log.push_back(log_entry);
    W->in_progress.insert(key);
    ++W->active;
    W->max_active = std::max(W->max_active, W->active);
    W->cv.notify_all();
    return W->beh[key];
}

void block_on(const std::string & key) {
    std::unique_lock<std::mutex> lk(W->mu);
    W->cv.wait(lk, [&] { return W->release_all || W->released.count(key) > 0; });
}

void leave(const std::string & key) {
    std::lock_guard<std::mutex> lk(W->mu);
    W->in_progress.erase(key);
    --W->active;
    W->cv.notify_all();
}

LoadResult fake_load(const LoadRequest & r) {
    const std::string key = r.function;
    int n = 0;
    const Behaviour b = enter(key, "load:" + key, n);
    LoadResult out;
    if (n <= b.would_block) {
        leave(key);
        out.would_block = true;
        return out;
    }
    if (b.block) block_on(key);
    if (b.delay_ms > 0) std::this_thread::sleep_for(milliseconds(b.delay_ms));
    leave(key);
    if (b.fail) {
        out.error = Error{b.cls, "test", 7, "fake load failure"};
        return out;
    }
    out.model = std::make_shared<FakeModel>(key);
    return out;
}

Error fake_warm(void * model, const LoadRequest & r) {
    const std::string key = "warm:" + r.function;
    int n = 0;
    const Behaviour b = enter(key, key, n);
    if (b.warm_block) block_on(key);
    leave(key);
    if (static_cast<FakeModel *>(model)->name != r.function) return Error::deterministic("wrong model");
    if (b.warm_fail) return Error::transient("fake warm failure");
    return Error{};
}

std::mutex g_tid_mu;
std::thread::id g_prepare_tid;  // the prepare worker that ran fake_prepare last

PrepareResult fake_prepare(const PrepareRequest & req, PrepareContext & ctx) {
    {
        std::lock_guard<std::mutex> lk(g_tid_mu);
        g_prepare_tid = std::this_thread::get_id();
    }
    const std::string key = "prepare:" + req.key;
    int n = 0;
    const Behaviour b = enter(key, key, n);
    PrepareResult out;
    if (n <= b.would_block) {
        leave(key);
        out.would_block = true;
        return out;
    }
    ctx.phase(PreparePhase::digesting);
    ctx.phase(PreparePhase::generating);
    if (b.block) block_on(key);
    ctx.phase(PreparePhase::compiling);
    leave(key);
    if (b.fail) {
        out.error = Error::deterministic("fake prepare failure");
        return out;
    }
    out.artifact_path = req.cache_dir + "/qvac-coreml-v1/artifacts/" + req.key + ".mlmodelc";
    out.artifact_key = req.key;
    return out;
}

uint64_t fake_mem() { return g_mem.load(); }
bool fake_low_power() { return g_low_power.load(); }

Config fast_config() {
    Config c;
    c.available_memory = &fake_mem;
    c.low_power = &fake_low_power;
    c.host_id = "test-host";
    c.os_build = "TEST1";
    c.memory_recheck = milliseconds(20);
    c.backoff_initial = milliseconds(10);
    c.backoff_max = milliseconds(80);
    c.wait_poll = milliseconds(5);
    c.pin_image = false;
    return c;
}

std::string make_temp_dir() {
    const char * base = std::getenv("TMPDIR");
    std::string tmpl = std::string(base && *base ? base : "/tmp") + "/qvac-coreml-test.XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    if (!mkdtemp(buf.data())) {
        std::perror("mkdtemp");
        std::exit(2);
    }
    return buf.data();
}

// One test's manager and temp dir. Declared first in each case so it is
// destroyed last: blocked fakes are released before the threads are joined.
struct Env {
    explicit Env(Config c = fast_config()) : dir(make_temp_dir()), cfg(c), m(Manager::create(c)) {}
    ~Env() {
        release_everything();
        m.reset();
        std::error_code ec;
        fs::remove_all(dir, ec);
    }
    std::string cache() const { return dir + "/cache"; }

    SlotSpec spec(const std::string & fn, ComputeUnits u = ComputeUnits::cpu_and_gpu) const {
        SlotSpec s;
        s.engine = "test";
        s.stage = "stage";
        s.function = fn;
        s.units = u;
        s.route = Route::prebuilt;
        s.artifact_path = dir + "/art.mlmodelc";
        s.load_fn = &fake_load;
        return s;
    }
    SlotPolicy policy(Wait w = Wait::none) const {
        SlotPolicy p;
        p.wait = w;
        p.cache_dir = cache();
        return p;
    }
    std::shared_ptr<Slot> slot(const std::string & fn, ComputeUnits u = ComputeUnits::cpu_and_gpu,
                               Wait w = Wait::none) const {
        return m->make_slot(spec(fn, u), policy(w));
    }
    std::shared_ptr<Slot> generated(const std::string & fn, const std::string & key) const {
        SlotSpec s = spec(fn);
        s.route = Route::generate;
        s.artifact_path.clear();
        s.prepare.key = key;
        s.prepare.cache_dir = cache();
        s.prepare_fn = &fake_prepare;
        return m->make_slot(s, policy());
    }

    std::string dir;
    Config cfg;
    std::shared_ptr<Manager> m;
};

void seed_hint(const std::shared_ptr<Slot> & s, int64_t warm_ms, std::optional<int64_t> stale_until = std::nullopt) {
    detail::HintRecord rec;
    rec.warm_load_ms = warm_ms;
    rec.last_ok = detail::iso8601_utc(detail::unix_now_s());
    rec.stale_until_s = stale_until;
    CHECK(detail::write_hint(s->hint_path(), rec));
}

std::optional<detail::HintRecord> hint_of(const std::shared_ptr<Slot> & s) { return detail::read_hint(s->hint_path()); }

bool is_ready(const std::shared_ptr<Slot> & s) { return s->state() == SlotState::ready; }

// ---- scheduler cases (§2.16) --------------------------------------------------

void case_priority_order() {
    Env e;
    set_beh("blocker", {true});
    auto blocker = e.slot("blocker");
    blocker->request(Priority::minimal);
    CHECK(wait_for([] { return in_progress("blocker"); }));
    auto pre = e.slot("pre");
    auto min = e.slot("min");
    auto cur = e.slot("cur");
    auto now = e.slot("now");
    pre->request(Priority::prewarm);
    min->request(Priority::minimal);
    cur->request(Priority::current);
    now->request(Priority::now);
    release("blocker");
    CHECK(wait_for([&] { return is_ready(pre) && is_ready(min) && is_ready(cur) && is_ready(now); }));
    const auto log = log_snapshot();
    CHECK(log_index(log, "load:now") < log_index(log, "load:cur"));
    CHECK(log_index(log, "load:cur") < log_index(log, "load:min"));
    CHECK(log_index(log, "load:min") < log_index(log, "load:pre"));
    CHECK(max_active() == 1);  // one load in flight
}

void case_dedup_two_owners() {
    // Two owners of one artifact (§2.9): one load, one shared model; either owner
    // dropping leaves the other's model intact.
    Env e;
    const int base = g_live_models.load();
    set_beh("shared", {true});
    auto a = e.slot("shared");
    auto b = e.slot("shared");
    a->request(Priority::minimal);
    b->request(Priority::minimal);
    release("shared");
    CHECK(wait_for([&] { return is_ready(a) && is_ready(b); }));
    CHECK(calls("shared") == 1);
    CHECK(e.m->stats().loads_started == 1);  // one deduplicated job served both owners
    Acquired ha = a->acquire(Demand::if_ready);
    Acquired hb = b->acquire(Demand::if_ready);
    CHECK(ha && hb && ha.model.get() == hb.model.get());
    CHECK(g_live_models.load() == base + 1);
    ha = Acquired{};
    a.reset();
    CHECK(is_ready(b));
    CHECK(hb && hb.model->as<FakeModel>()->name == "shared");
    {
        auto lock = hb.model->lock_for_prediction();
        CHECK(lock.owns_lock());
    }
    hb = Acquired{};
    b.reset();
    CHECK(wait_for([&] { return g_live_models.load() == base; }));
}

void case_cancel_when_owners_drop() {
    Env e;
    set_beh("blocker", {true});
    auto blocker = e.slot("blocker");
    blocker->request(Priority::minimal);
    CHECK(wait_for([] { return in_progress("blocker"); }));
    auto victim = e.slot("victim");
    victim->request(Priority::now);
    victim.reset();  // the engine drops its slot before the job started
    release("blocker");
    CHECK(wait_for([&] { return is_ready(blocker); }));
    std::this_thread::sleep_for(50ms);
    CHECK(calls("victim") == 0);
    CHECK(e.m->stats().cancelled >= 1);
}

void case_admission_gate() {
    // A prepare job and a load never run at the same time in one image.
    Env e;
    set_beh("prepare:k1", {true});
    auto gen = e.generated("gen", "k1");
    gen->request(Priority::now);
    CHECK(wait_for([] { return in_progress("prepare:k1"); }));
    auto l1 = e.slot("l1");
    auto l2 = e.slot("l2");
    l1->request(Priority::now);
    l2->request(Priority::now);
    std::this_thread::sleep_for(40ms);
    CHECK(calls("l1") == 0 && calls("l2") == 0);  // gate held by the prepare job
    CHECK(gen->state() == SlotState::generating);
    release("prepare:k1");
    CHECK(wait_for([&] { return is_ready(gen) && is_ready(l1) && is_ready(l2); }));
    CHECK(max_active() == 1);
    const SlotStatus st = gen->status();
    CHECK(st.route == StatusRoute::generated);
    CHECK(st.artifact_key == "k1");
}

void case_current_outranks_minimal() {
    Env e;
    set_beh("blocker", {true});
    auto blocker = e.slot("blocker");
    blocker->request(Priority::minimal);
    CHECK(wait_for([] { return in_progress("blocker"); }));
    auto m1 = e.slot("m1");
    auto m2 = e.slot("m2");
    auto m3 = e.slot("m3");
    m1->request(Priority::minimal);
    m2->request(Priority::minimal);
    m3->request(Priority::minimal);
    auto x = e.slot("x");  // a call starts and predicts width x
    x->request(Priority::current);
    release("blocker");
    CHECK(wait_for([&] { return is_ready(x) && is_ready(m1) && is_ready(m2) && is_ready(m3); }));
    const auto log = log_snapshot();
    CHECK(log_index(log, "load:blocker") == 0);  // only the load in flight was ahead
    CHECK(log_index(log, "load:x") == 1);
}

void case_bounded_wait_queued() {
    // A hinted slot queued behind a hinted load waits; two acquisitions share
    // one aggregate deadline.
    Env e;
    auto ahead = e.slot("ahead");
    auto mine = e.slot("mine");
    auto other = e.slot("other");
    seed_hint(ahead, 20);
    seed_hint(mine, 20);
    seed_hint(other, 20);
    set_beh("ahead", {true});
    ahead->request(Priority::minimal);
    CHECK(wait_for([] { return in_progress("ahead"); }));
    std::thread releaser([] {
        std::this_thread::sleep_for(60ms);
        release("ahead");
    });
    const auto t0 = Clock::now();
    const auto deadline = t0 + 3000ms;
    Acquired a1 = mine->acquire(Demand::now, deadline);
    Acquired a2 = other->acquire(Demand::now, deadline);
    releaser.join();
    CHECK(a1 && a2);
    CHECK(!a1.pending && !a2.pending);
    CHECK(Clock::now() < deadline);
    std::printf("    bounded wait (queued): %lld ms for two acquisitions\n", (long long) ms_since(t0));
}

void case_bounded_wait_loading() {
    Env e;
    auto s = e.slot("slow");
    seed_hint(s, 40);
    set_beh("slow", {false, 80});
    s->request(Priority::minimal);
    CHECK(wait_for([] { return in_progress("slow"); }));
    Acquired a = s->acquire(Demand::now, Clock::now() + 3000ms);
    CHECK(a && !a.pending);
}

void case_stale_hint_cooldown() {
    Env e;
    auto s = e.slot("cold");
    seed_hint(s, 20);
    set_beh("cold", {true});
    s->request(Priority::minimal);
    CHECK(wait_for([] { return in_progress("cold"); }));
    // The hint says 20 ms, but this load is cold: the wait expires.
    const auto t0 = Clock::now();
    Acquired a = s->acquire(Demand::now, t0 + 250ms);
    const int64_t penalty = ms_since(t0);
    std::printf("    first-call penalty with a stale hint: %lld ms (deadline 250 ms)\n", (long long) penalty);
    CHECK(!a && a.pending);
    CHECK(Clock::now() >= t0 + 250ms);  // it waited up to the deadline (the reported penalty)
    auto h = hint_of(s);
    CHECK(h && h->stale_until_s && *h->stale_until_s >= detail::unix_now_s() + 23 * 3600);
    CHECK(h && h->warm_load_ms && *h->warm_load_ms == 20);
    CHECK(s->status().hint && s->status().hint->stale);

    // The late completion neither trains warm_load_ms nor clears stale_until.
    release("cold");
    CHECK(wait_for([&] { return is_ready(s); }));
    h = hint_of(s);
    CHECK(h && h->warm_load_ms && *h->warm_load_ms == 20);
    CHECK(h && h->cold_load_ms && *h->cold_load_ms >= 200);
    CHECK(h && h->stale_until_s.has_value());

    // The next acquisition sees the cooldown: no wait.
    s->evict();
    unrelease("cold");
    s->request(Priority::minimal);
    CHECK(wait_for([] { return in_progress("cold"); }));
    // "No wait" = it returns before a deadline that a wait would run into.
    const auto d1 = Clock::now() + 10s;
    Acquired a2 = s->acquire(Demand::now, d1);
    CHECK(!a2 && Clock::now() < d1);

    // So does the next process.
    Env next(e.cfg);
    auto s2 = next.m->make_slot(e.spec("cold"), e.policy());
    CHECK(s2->hint_path() == s->hint_path());
    const auto d2 = Clock::now() + 10s;
    Acquired a3 = s2->acquire(Demand::now, d2);
    CHECK(!a3 && Clock::now() < d2);
    release("cold");
}

void case_warm_queued_behind_cold() {
    Env e;
    auto cold = e.slot("coldjob");  // no hint: possibly a cold compile
    auto warm = e.slot("warm");
    seed_hint(warm, 20);
    set_beh("coldjob", {true});
    cold->request(Priority::minimal);
    CHECK(wait_for([] { return in_progress("coldjob"); }));
    const auto deadline = Clock::now() + 10s;
    Acquired a = warm->acquire(Demand::now, deadline);
    CHECK(!a && a.pending);
    CHECK(Clock::now() < deadline);  // no wait behind an unhinted load
    auto h = hint_of(warm);
    CHECK(h && !h->stale_until_s.has_value());  // and not staled
    CHECK(warm->status().wait.empty());
    release("coldjob");
}

void case_queued_behind_expiry() {
    Env e;
    auto ahead = e.slot("ahead");
    auto mine = e.slot("mine");
    seed_hint(ahead, 20);
    seed_hint(mine, 20);
    set_beh("ahead", {true});
    ahead->request(Priority::minimal);
    CHECK(wait_for([] { return in_progress("ahead"); }));
    const auto deadline = Clock::now() + 200ms;
    Acquired a = mine->acquire(Demand::now, deadline);
    CHECK(!a && a.pending);
    CHECK(Clock::now() >= deadline);  // it did wait, then expired while still queued
    CHECK(mine->status().wait == "queued_behind");
    auto h = hint_of(mine);
    CHECK(h && !h->stale_until_s.has_value());  // hint untouched
    CHECK(nlohmann::json::parse(to_json(mine->status())).value("wait", "") == "queued_behind");
    release("ahead");
}

void case_prepare_ahead_no_wait() {
    Env e;
    set_beh("prepare:p", {true});
    auto gen = e.generated("genfn", "p");
    gen->request(Priority::minimal);
    CHECK(wait_for([] { return in_progress("prepare:p"); }));
    auto s = e.slot("hinted");
    seed_hint(s, 10);
    const auto deadline = Clock::now() + 10s;
    Acquired a = s->acquire(Demand::now, deadline);
    CHECK(!a && a.pending && Clock::now() < deadline);
    release("prepare:p");
}

void case_groups_construct() {
    Env e;
    set_beh("g1", {false, 30});
    set_beh("g2", {false, 30});
    Behaviour bad;
    bad.fail = true;
    bad.cls = ErrorClass::deterministic;
    set_beh("g3", bad);
    auto g1 = e.slot("g1");
    auto g2 = e.slot("g2");
    auto g3 = e.slot("g3");
    auto group = e.m->issue_group(GroupKind::construction,
                                  {{g1, Priority::minimal}, {g2, Priority::prewarm}, {g3, Priority::minimal}}, true);
    const PrewarmResult r = group->wait(5000ms);
    CHECK(group->finished());
    CHECK(!r.success && !r.nothing_to_prepare);
    CHECK(r.group == "construction");
    CHECK(r.slots.size() == 3);
    if (r.slots.size() == 3) {
        CHECK(r.slots[0].state == SlotState::ready && r.slots[1].state == SlotState::ready);
        CHECK(r.slots[2].state == SlotState::failed);
        CHECK(r.slots[2].error_class == ErrorClass::deterministic);
    }
    const auto j = nlohmann::json::parse(to_json(r));
    CHECK(j["success"] == false && j["slots"].size() == 3 && j["slots"][2]["error_class"] == "deterministic");
}

void case_nothing_to_prepare() {
    Env e;
    SlotSpec none = e.spec("none");
    none.route = Route::none;
    none.none_state = SlotState::unsupported;
    none.none_reason = "os";
    auto s = e.m->make_slot(none, e.policy());
    // An all-route=none group, and prewarm:'none' with wait:'construct'
    // (an empty group): both return at once (wait() has no timeout here).
    PrewarmResult r;
    must_complete("empty construct group", [&] {
        r = e.m->issue_group(GroupKind::construction, {}, true, {s->status()})->wait();
    });
    CHECK(r.nothing_to_prepare && r.success);
    CHECK(r.skipped_route_none.size() == 1 && r.skipped_route_none[0].state == SlotState::unsupported);
    const auto j = nlohmann::json::parse(to_json(r));
    CHECK(j["nothing_to_prepare"] == true && j["skipped_route_none"][0]["state"] == "unsupported(os)");
}

void case_low_power() {
    Env e;
    g_low_power = true;
    auto deferred = e.slot("lp-deferred");
    auto waited = e.slot("lp-waited");
    auto g_free = e.m->issue_group(GroupKind::prewarm, {{deferred, Priority::prewarm}}, false);
    std::this_thread::sleep_for(60ms);
    CHECK(calls("lp-deferred") == 0);  // prewarm work deferred in low power
    // {prewarm:'all', wait:'construct'} still completes.
    auto g_waited = e.m->issue_group(GroupKind::construction, {{waited, Priority::prewarm}}, true);
    const PrewarmResult r = g_waited->wait(3000ms);
    CHECK(r.success);
    CHECK(calls("lp-deferred") == 0);
    g_low_power = false;
    e.m->notify();
    CHECK(wait_for([&] { return is_ready(deferred); }));
    CHECK(g_free->wait(2000ms).success);
}

// The prepare worker picks the published load and hands it to the load worker;
// low power turns on before the load worker selects it, so the handed-off job
// is no longer eligible. The handoff must not keep the load worker spinning:
// once the job is cancelled, a spinning worker would hold the runtime lock.
std::atomic<bool> g_handoff_armed{false};
std::atomic<bool> g_handoff_prep_checked{false};
std::atomic<int> g_handoff_after{0};  // low-power checks by the load worker after the handoff
bool handoff_low_power() {
    if (!g_handoff_armed.load()) return false;
    {
        std::lock_guard<std::mutex> lk(g_tid_mu);
        if (std::this_thread::get_id() == g_prepare_tid) {
            g_handoff_prep_checked = true;
            return false;  // the prepare worker still sees normal power and hands the load off
        }
    }
    if (g_handoff_prep_checked.load()) ++g_handoff_after;
    return true;
}

void case_handoff_ineligible() {
    g_handoff_armed = false;
    g_handoff_prep_checked = false;
    g_handoff_after = 0;
    Config cfg = fast_config();
    cfg.low_power = &handoff_low_power;
    cfg.memory_recheck = std::chrono::seconds(30);  // no periodic wake inside the window
    Env e(cfg);
    set_beh("prepare:handoff", {true});
    auto slot = e.generated("handoff-load", "handoff");
    slot->request(Priority::prewarm);
    CHECK(wait_for([] { return in_progress("prepare:handoff"); }));
    g_handoff_armed = true;
    release("prepare:handoff");
    CHECK(wait_for([] { return g_handoff_after.load() >= 1; }));  // handed off, then deferred
    CHECK(wait_for([&] { return slot->state() == SlotState::queued; }));
    // Deferred, not spinning: the load worker sleeps until something changes.
    std::this_thread::sleep_for(50ms);
    const int n0 = g_handoff_after.load();
    std::this_thread::sleep_for(100ms);
    CHECK(g_handoff_after.load() - n0 < 10);
    CHECK(calls("handoff-load") == 0);
    // Cancel the handed-off job; later runtime calls must still complete.
    must_complete("slot destruction", [&] { slot.reset(); });
    must_complete("Manager::notify", [&] { e.m->notify(); });
    auto other = e.slot("handoff-other");
    must_complete("request", [&] { other->request(Priority::now); });
    g_handoff_armed = false;
    must_complete("Manager::notify", [&] { e.m->notify(); });
    CHECK(wait_for([&] { return is_ready(other); }));
}

void case_eviction() {
    Env e;
    const int base = g_live_models.load();
    // (a) An unstarted job is cancelled.
    set_beh("blocker", {true});
    auto blocker = e.slot("blocker");
    blocker->request(Priority::minimal);
    CHECK(wait_for([] { return in_progress("blocker"); }));
    auto queued = e.slot("ev-queued");
    queued->request(Priority::minimal);
    queued->evict();
    CHECK(queued->state() == SlotState::idle);
    release("blocker");
    CHECK(wait_for([&] { return is_ready(blocker); }));
    std::this_thread::sleep_for(30ms);
    CHECK(calls("ev-queued") == 0);

    // (b) A started load is not cancelled: it completes, writes its hint and
    // does not repopulate the slot; the model is released at once.
    set_beh("ev-started", {true});
    auto started = e.slot("ev-started");
    started->request(Priority::minimal);
    CHECK(wait_for([] { return in_progress("ev-started"); }));
    const uint64_t done_before = e.m->stats().loads_completed;
    started->evict();
    release("ev-started");
    CHECK(wait_for([&] { return e.m->stats().loads_completed == done_before + 1; }));
    CHECK(started->state() == SlotState::idle);
    CHECK(wait_for([&] { return g_live_models.load() == base + 1; }));  // only the blocker's model
    auto h = hint_of(started);
    CHECK(h && h->cold_load_ms.has_value());

    // (c) A request after the eviction subscribes to the running job: one load.
    set_beh("ev-again", {true});
    auto again = e.slot("ev-again");
    again->request(Priority::minimal);
    CHECK(wait_for([] { return in_progress("ev-again"); }));
    again->evict();
    again->request(Priority::minimal);
    release("ev-again");
    CHECK(wait_for([&] { return is_ready(again); }));
    CHECK(calls("ev-again") == 1);
}

void case_resident_cap() {
    Env e;
    const int base = g_live_models.load();
    auto pool = e.m->make_pool(3);
    std::vector<std::shared_ptr<Slot>> w;
    for (int i = 1; i <= 5; ++i) {
        SlotSpec s = e.spec("w" + std::to_string(i));
        s.pool = pool;
        w.push_back(e.m->make_slot(s, e.policy()));
    }
    std::atomic<bool> stop{false};
    std::atomic<size_t> max_live{0};
    std::thread sampler([&] {
        while (!stop.load()) {
            const size_t n = pool->live();
            size_t m = max_live.load();
            while (n > m && !max_live.compare_exchange_weak(m, n)) {}
            std::this_thread::sleep_for(1ms);
        }
    });
    for (int i = 0; i < 3; ++i) {
        w[i]->request(Priority::minimal);
        CHECK(wait_for([&] { return is_ready(w[i]); }));
    }
    Acquired h1 = w[0]->acquire(Demand::now);  // a unit holds w1
    CHECK(h1);
    pool->set_predicted({w[1]});              // the current call predicts w2
    w[3]->request(Priority::current);         // a fourth width
    CHECK(wait_for([&] { return is_ready(w[3]); }));
    CHECK(w[2]->state() == SlotState::idle);  // w3 was the only evictable one
    CHECK(is_ready(w[0]) && is_ready(w[1]));
    CHECK(h1 && h1.model->as<FakeModel>()->name == "w1");

    // Nothing can be released: the fifth width is not loaded meanwhile.
    pool->clear_predicted();
    Acquired h2 = w[1]->acquire(Demand::now);
    Acquired h4 = w[3]->acquire(Demand::now);
    w[4]->request(Priority::current);
    std::this_thread::sleep_for(80ms);
    CHECK(calls("w5") == 0);
    CHECK(w[4]->state() == SlotState::queued);
    CHECK(pool->live() == 3);
    CHECK(g_live_models.load() == base + 3);
    // The unit releases w4: it becomes evictable and w5 loads.
    h4 = Acquired{};
    CHECK(wait_for([&] { return is_ready(w[4]); }));
    CHECK(w[3]->state() == SlotState::idle);
    CHECK(is_ready(w[0]) && is_ready(w[1]));
    stop = true;
    sampler.join();
    CHECK(max_live.load() <= 3);
    std::printf("    resident cap 3: max live instances observed %zu\n", max_live.load());
}

// §2.9 counts handles a unit still retains as live instances, whatever happened
// to the slot that published them.
void case_resident_cap_retained_handle() {
    Env e;
    const int base = g_live_models.load();
    auto pool = e.m->make_pool(1);
    auto mk = [&](const std::string & fn) {
        SlotSpec s = e.spec(fn);
        s.pool = pool;
        return e.m->make_slot(s, e.policy());
    };
    auto a = mk("ra");
    a->request(Priority::minimal);
    CHECK(wait_for([&] { return is_ready(a); }));
    Acquired held = a->acquire(Demand::if_ready);
    CHECK(held);
    a.reset();  // the engine destroys its slot; the unit still holds the model
    CHECK(g_live_models.load() == base + 1);
    CHECK(pool->live() == 1);
    // A reload of the same function shares the retained instance.
    auto a2 = mk("ra");
    a2->request(Priority::minimal);
    CHECK(wait_for([&] { return is_ready(a2); }));
    CHECK(calls("ra") == 1);
    CHECK(g_live_models.load() == base + 1 && pool->live() == 1);
    a2.reset();
    CHECK(pool->live() == 1);
    // A replacement is not admitted over the cap while the handle lives.
    auto b = mk("rb");
    b->request(Priority::current);
    std::this_thread::sleep_for(80ms);
    CHECK(calls("rb") == 0);
    CHECK(b->state() == SlotState::queued);
    CHECK(g_live_models.load() == base + 1 && pool->live() == 1);
    held = Acquired{};  // the unit finishes: the instance dies and rb loads
    CHECK(wait_for([&] { return is_ready(b); }));
    CHECK(g_live_models.load() == base + 1);
    CHECK(pool->live() == 1);
}

// The caller itself holds every instance the cap permits: nothing can be
// released, so call, strict, the bounded wait and construct groups never wait
// on the replacement (§2.9, §2.10); it loads in the background after a release.
void case_resident_cap_acquire() {
    Env e;
    auto pool = e.m->make_pool(1);
    auto mk = [&](const std::string & fn, Wait w = Wait::none, bool strict = false) {
        SlotSpec s = e.spec(fn);
        s.pool = pool;
        SlotPolicy p = e.policy(w);
        p.strict = strict;
        return e.m->make_slot(s, p);
    };
    auto a = mk("ca");
    a->request(Priority::minimal);
    CHECK(wait_for([&] { return is_ready(a); }));
    Acquired held = a->acquire(Demand::now);
    CHECK(held);

    auto b = mk("cb", Wait::call);
    Acquired r;
    must_complete("call acquisition under the cap", [&] { r = b->acquire(Demand::now); });
    CHECK(!r && r.pending);
    CHECK(b->state() == SlotState::queued);  // still queued for a later release

    auto st = mk("cs", Wait::call, true);  // strict, default unbounded strict_wait_ms
    std::optional<SlotState> thrown;
    must_complete("strict acquisition under the cap", [&] {
        try {
            st->acquire(Demand::now);
        } catch (const CoremlError & err) {
            thrown = err.state;
        }
    });
    CHECK(thrown == SlotState::queued);
    st.reset();

    auto bw = mk("cb");  // bounded wait with a non-stale hint: no wait either
    seed_hint(bw, 20);
    Acquired rw;
    must_complete("bounded wait under the cap", [&] { rw = bw->acquire(Demand::now, Clock::now() + 10s); });
    CHECK(!rw && rw.pending);
    bw.reset();

    auto g = e.m->issue_group(GroupKind::construction, {{b, Priority::minimal}}, true);
    PrewarmResult gr;
    must_complete("construct group under the cap", [&] { gr = g->wait(10s); });
    CHECK(g->finished() && !gr.success);

    // Released but predicted for the current call: still never evicted.
    pool->set_predicted({a});
    held = Acquired{};
    must_complete("call acquisition with a predicted instance", [&] { r = b->acquire(Demand::now); });
    CHECK(!r && r.pending);
    CHECK(is_ready(a));
    CHECK(calls("cb") == 0 && calls("cs") == 0);

    // Nothing protects it any more: the queued replacement loads.
    pool->clear_predicted();
    CHECK(wait_for([&] { return is_ready(b); }));
    CHECK(a->state() == SlotState::idle);
    CHECK(calls("cb") == 1);
    CHECK(pool->live() == 1);

    // The denial is decided when the caller asks, not by the last worker
    // screening: an acquisition right after the caller lifts it (releasing its
    // handle, or narrowing the prediction) waits for the replacement (§2.10).
    // Each round first acquires under the denial, so a cached verdict exists.
    auto denied_now = [](const std::shared_ptr<Slot> & s, bool strict) {
        bool denied = false;
        must_complete("acquisition under the cap", [&] {
            try {
                denied = s->acquire(Demand::now).pending;
            } catch (const CoremlError &) {
                denied = strict;
            }
        });
        return denied;
    };
    for (int round = 0; round < 8; ++round) {
        for (const bool strict : {false, true}) {
            for (const bool via_predicted : {false, true}) {
                const std::string tag = std::to_string(round) + (strict ? "s" : "c") + (via_predicted ? "p" : "r");
                auto rp = e.m->make_pool(1);
                auto mk2 = [&](const std::string & fn, bool s) {
                    SlotSpec spec = e.spec(fn);
                    spec.pool = rp;
                    SlotPolicy p = e.policy(Wait::call);
                    p.strict = s;
                    return e.m->make_slot(spec, p);
                };
                auto ra = mk2("la" + tag, false);
                ra->request(Priority::minimal);
                CHECK(wait_for([&] { return is_ready(ra); }));
                Acquired rheld = ra->acquire(Demand::now);
                CHECK(rheld);
                auto rb = mk2("lb" + tag, strict);
                CHECK(denied_now(rb, strict));
                if (via_predicted) {
                    rp->set_predicted({ra});
                    rheld = Acquired{};
                    CHECK(denied_now(rb, strict));
                }
                Acquired got;
                std::string threw;
                must_complete("acquisition right after the denial was lifted", [&] {
                    // Lift and acquire on one thread, before a worker re-screens.
                    if (via_predicted) {
                        rp->set_predicted({rb});  // narrowed: ra is evictable now
                    } else {
                        rheld = Acquired{};  // released: ra is evictable now
                    }
                    try {
                        got = rb->acquire(Demand::now);
                    } catch (const CoremlError & err) {
                        threw = err.what();
                    }
                });
                if (!got || !threw.empty()) {
                    std::printf("    round %s: pending=%d threw='%s'\n", tag.c_str(), int(got.pending), threw.c_str());
                }
                CHECK(got && !got.pending && threw.empty());
                CHECK(got && got.model->as<FakeModel>()->name == "lb" + tag);
                CHECK(rp->live() == 1);
            }
        }
    }
}

// Two engines, one resident pool each, sharing one instance (§2.9): each pool
// counts only what it references, so either engine can move to another
// function, and the instance lives on for the other engine.
void case_resident_cap_two_pools() {
    for (const bool evict_first : {false, true}) {
        Env e;
        const int base = g_live_models.load();
        const int loads0 = calls("w1");
        auto p1 = e.m->make_pool(1);
        auto p2 = e.m->make_pool(1);
        auto mk = [&](const std::shared_ptr<ResidentPool> & pool, const std::string & fn) {
            SlotSpec s = e.spec(fn);
            s.pool = pool;
            return e.m->make_slot(s, e.policy(Wait::call));
        };
        auto e1_w1 = mk(p1, "w1");
        auto e2_w1 = mk(p2, "w1");
        e1_w1->request(Priority::minimal);
        e2_w1->request(Priority::minimal);
        CHECK(wait_for([&] { return is_ready(e1_w1) && is_ready(e2_w1); }));
        CHECK(calls("w1") == loads0 + 1);  // one shared instance
        CHECK(g_live_models.load() == base + 1);
        CHECK(p1->live() == 1 && p2->live() == 1);
        Acquired h2 = e2_w1->acquire(Demand::now);  // engine 2 runs a unit on it
        CHECK(h2);
        if (evict_first) {
            e1_w1->evict();
            CHECK(p1->live() == 0);  // engine 2's references count only in p2
        }
        auto e1_w2 = mk(p1, "w2");
        Acquired a;
        must_complete("engine 1 moves to w2", [&] { a = e1_w2->acquire(Demand::now); });
        CHECK(a && a.model->as<FakeModel>()->name == "w2");
        CHECK(e1_w1->state() == SlotState::idle);
        CHECK(is_ready(e2_w1));
        CHECK(h2 && h2.model->as<FakeModel>()->name == "w1");
        CHECK(g_live_models.load() == base + 2);  // w1 lives on for engine 2
        CHECK(p1->live() == 1 && p2->live() == 1);
    }
}

// Another owner's instance enters a capped pool only through that pool's
// admission (§2.9): a ready registry hit, a late subscriber to a running load
// and a subscriber joining the other owner's queued job all stay pending while
// the pool is full with nothing evictable, without delaying the other owner;
// after a release the slot gets the instance (the same one while the other
// owner keeps it, a fresh load after it dropped it). With room, both share at
// once.
void case_resident_cap_cross_owner() {
    enum class Join { registry, running, queued };
    int round = 0;
    for (const Join join : {Join::registry, Join::running, Join::queued}) {
        for (const bool drop_owner : {false, true}) {
            reset_world();  // the previous round's Env released everything
            Env e;
            const std::string tag = std::to_string(round++);
            const std::string shared = "xs" + tag;
            const std::string gate = "xg" + tag;
            const int base = g_live_models.load();
            auto pool = e.m->make_pool(1);
            auto mk = [&](const std::string & fn, Wait w) {
                SlotSpec s = e.spec(fn);
                s.pool = pool;
                return e.m->make_slot(s, e.policy(w));
            };
            auto a = mk("xh" + tag, Wait::none);
            a->request(Priority::minimal);
            CHECK(wait_for([&] { return is_ready(a); }));
            Acquired held = a->acquire(Demand::if_ready);
            CHECK(held);
            std::shared_ptr<Slot> gate_slot;
            if (join == Join::queued) {  // the other owner's job waits behind the gate
                set_beh(gate, {true});
                gate_slot = e.slot(gate);
                gate_slot->request(Priority::minimal);
                CHECK(wait_for([&] { return in_progress(gate); }));
            }
            if (join == Join::running) set_beh(shared, {true});
            auto external = e.slot(shared);  // another owner, no pool
            external->request(Priority::minimal);
            if (join == Join::running) CHECK(wait_for([&] { return in_progress(shared); }));
            if (join == Join::registry) CHECK(wait_for([&] { return is_ready(external); }));
            auto b = mk(shared, Wait::call);
            b->request(Priority::current);
            Acquired r;
            if (join == Join::running) {
                // Denied while the other owner's load runs: no place was reserved.
                must_complete("call acquisition during the load", [&] { r = b->acquire(Demand::now); });
                CHECK(!r && r.pending);
                release(shared);
            }
            if (join == Join::queued) release(gate);
            CHECK(wait_for([&] { return is_ready(external); }));  // the other owner is served
            std::this_thread::sleep_for(50ms);
            CHECK(!is_ready(b));
            CHECK(pool->live() == 1);
            CHECK(calls(shared) == 1);
            CHECK(g_live_models.load() == base + (join == Join::queued ? 3 : 2));  // held + external (+ gate)
            must_complete("call acquisition under the cap", [&] { r = b->acquire(Demand::now); });
            CHECK(!r && r.pending);
            {  // strict cannot be served either (§2.10), and gives no place
                SlotSpec ss = e.spec(shared);
                ss.pool = pool;
                SlotPolicy sp = e.policy(Wait::call);
                sp.strict = true;
                auto strict = e.m->make_slot(ss, sp);
                bool threw = false;
                must_complete("strict acquisition under the cap", [&] {
                    try {
                        (void) strict->acquire(Demand::now);
                    } catch (const CoremlError &) {
                        threw = true;
                    }
                });
                CHECK(threw);
                CHECK(pool->live() == 1);
            }
            if (drop_owner) {
                external.reset();
                CHECK(g_live_models.load() == base + (join == Join::queued ? 2 : 1));  // the capped owner never took it
                CHECK(pool->live() == 1);
            }
            held = Acquired{};
            must_complete("call acquisition after the release", [&] { r = b->acquire(Demand::now); });
            CHECK(r && r.model->as<FakeModel>()->name == shared);
            CHECK(a->state() == SlotState::idle);
            CHECK(pool->live() == 1);
            CHECK(calls(shared) == (drop_owner ? 2 : 1));
            // a's instance died; b's is the external one or its own reload.
            CHECK(g_live_models.load() == base + 1 + (join == Join::queued ? 1 : 0));
            r = Acquired{};
        }
    }
    // With room the instance is shared at once: no second load, one place.
    for (const bool running : {false, true}) {
        reset_world();
        Env e;
        const std::string shared = std::string("xr") + (running ? "1" : "0");
        const int base = g_live_models.load();
        auto pool = e.m->make_pool(2);
        SlotSpec hs = e.spec("xrh" + shared);
        hs.pool = pool;
        auto a = e.m->make_slot(hs, e.policy());
        a->request(Priority::minimal);
        CHECK(wait_for([&] { return is_ready(a); }));
        Acquired held = a->acquire(Demand::if_ready);
        if (running) set_beh(shared, {true});
        auto external = e.slot(shared);
        external->request(Priority::minimal);
        if (running) {
            CHECK(wait_for([&] { return in_progress(shared); }));
        } else {
            CHECK(wait_for([&] { return is_ready(external); }));
        }
        SlotSpec bs = e.spec(shared);
        bs.pool = pool;
        auto b = e.m->make_slot(bs, e.policy());
        b->request(Priority::current);
        if (running) {
            CHECK(pool->live() == 2);  // the running load holds the reserved place
            release(shared);
        }
        CHECK(wait_for([&] { return is_ready(b) && is_ready(external); }));
        CHECK(calls(shared) == 1);
        CHECK(is_ready(a));
        CHECK(pool->live() == 2);
        CHECK(g_live_models.load() == base + 2);
    }
}

// A load queued under the cap becomes redundant when another member of its
// pool adopts the same registry instance meanwhile (§2.9): the pool already
// counts it, so the queued subscriber takes it without a second place, an
// eviction or a load, and call / strict acquisitions are served (§2.10).
void case_resident_cap_queued_adopt() {
    int round = 0;
    for (const bool acquire_it : {true, false}) {
        for (const bool strict : {false, true}) {
            if (!acquire_it && strict) continue;
            reset_world();
            Env e;
            const std::string tag = std::to_string(round++);
            const std::string shared = "qs" + tag;
            const int base = g_live_models.load();
            auto pool = e.m->make_pool(1);
            auto mk = [&](const std::string & fn, bool s = false) {
                SlotSpec spec = e.spec(fn);
                spec.pool = pool;
                SlotPolicy p = e.policy(Wait::call);
                p.strict = s;
                return e.m->make_slot(spec, p);
            };
            auto a = mk("qa" + tag);
            Acquired held = a->acquire(Demand::now);
            CHECK(held);
            auto external = e.slot(shared);  // another owner, no pool
            external->request(Priority::minimal);
            CHECK(wait_for([&] { return is_ready(external); }));
            g_low_power = true;  // keeps the prewarm request queued
            auto b = mk(shared, strict);
            b->request(Priority::prewarm);
            CHECK(b->state() == SlotState::queued);
            a->evict();
            held = Acquired{};
            auto c = mk(shared);
            Acquired hc = c->acquire(Demand::now);  // adopts the registry instance at once
            CHECK(hc);
            CHECK(pool->live() == 1);
            const uint64_t ev0 = e.m->stats().evictions;
            Acquired hb;
            std::string threw;
            if (acquire_it) {
                must_complete("acquisition of an instance the pool counts", [&] {
                    try {
                        hb = b->acquire(Demand::now);
                    } catch (const CoremlError & err) {
                        threw = err.what();
                    }
                });
                if (!threw.empty()) std::printf("    round %s: threw '%s'\n", tag.c_str(), threw.c_str());
                CHECK(threw.empty());
            } else {
                // Not held by c: still nothing is evicted, the worker hands the
                // prewarm subscriber the counted instance.
                hc = Acquired{};
                g_low_power = false;
                CHECK(wait_for([&] { return is_ready(b); }));
                hb = b->acquire(Demand::if_ready);
                CHECK(is_ready(c));
                hc = c->acquire(Demand::if_ready);
            }
            CHECK(hb && hc && hb.model.get() == hc.model.get());
            CHECK(hb && hb.model->as<FakeModel>()->name == shared);
            CHECK(calls(shared) == 1);
            CHECK(e.m->stats().evictions == ev0);
            CHECK(pool->live() == 1);
            CHECK(g_live_models.load() == base + 1);  // only the shared instance
        }
    }
}

// A native model whose deleter waits for the test (functions "gd*").
std::atomic<int> g_delete_entered{0};
std::atomic<bool> g_delete_open{true};

LoadResult gated_delete_load(const LoadRequest & r) {
    LoadResult out = fake_load(r);
    if (!out.model || r.function.rfind("gd", 0) != 0) return out;
    auto inner = std::static_pointer_cast<FakeModel>(out.model);
    out.model = std::shared_ptr<FakeModel>(new FakeModel(inner->name), [](FakeModel * m) {
        ++g_delete_entered;
        while (!g_delete_open.load()) std::this_thread::sleep_for(1ms);
        delete m;
    });
    return out;
}

// An instance dropped from a pool on another thread (explicit eviction, slot
// destruction, retirement) keeps its place until its release completed outside the lock
// (§2.9): no replacement load, no second eviction, and never more live native
// models than the cap while the deleter runs. A reference another owner keeps
// stops counting in the pool once the pool's own reference is released.
void case_resident_cap_pending_release() {
    enum class Drop { evict, destroy, retire };
    for (const Drop how : {Drop::evict, Drop::destroy, Drop::retire}) {
        for (const size_t cap : {size_t(1), size_t(2)}) {
            reset_world();
            Env e;
            const std::string tag = std::string(how == Drop::evict ? "e" : how == Drop::destroy ? "d" : "r") +
                                    std::to_string(cap);
            const int base = g_live_models.load();
            auto pool = e.m->make_pool(cap);
            auto mk = [&](const std::string & fn) {
                SlotSpec s = e.spec(fn);
                s.pool = pool;
                s.load_fn = &gated_delete_load;
                return e.m->make_slot(s, e.policy());
            };
            std::shared_ptr<Slot> keep;  // cap 2: a second resident that must survive
            if (cap == 2) {
                keep = mk("pk" + tag);
                keep->request(Priority::minimal);
                CHECK(wait_for([&] { return is_ready(keep); }));
            }
            auto a = mk("gd" + tag);
            a->request(Priority::minimal);
            CHECK(wait_for([&] { return is_ready(a); }));
            CHECK(pool->live() == cap);
            const uint64_t ev0 = e.m->stats().evictions;
            g_delete_entered = 0;
            g_delete_open = false;
            std::thread dropper([&] {
                switch (how) {
                    case Drop::evict: a->evict(); break;
                    case Drop::destroy: a.reset(); break;
                    case Drop::retire: a->retire("test"); break;  // no handle outstanding
                }
            });
            CHECK(wait_for([] { return g_delete_entered.load() == 1; }));
            const std::string repl = "pr" + tag;
            auto b = mk(repl);
            b->request(Priority::current);
            const auto end = Clock::now() + 200ms;
            int over = 0;
            while (Clock::now() < end) {
                if (calls(repl) != 0 || is_ready(b) || g_live_models.load() > base + int(cap) || pool->live() != cap) {
                    ++over;
                }
                std::this_thread::sleep_for(1ms);
            }
            CHECK(over == 0);
            CHECK(calls(repl) == 0);
            CHECK(!keep || is_ready(keep));  // no second victim while the release runs
            CHECK(e.m->stats().evictions - ev0 == (how == Drop::evict ? 1u : 0u));
            g_delete_open = true;
            dropper.join();
            CHECK(wait_for([&] { return is_ready(b); }));
            CHECK(!keep || is_ready(keep));  // the release made the room
            CHECK(e.m->stats().evictions - ev0 == (how == Drop::evict ? 1u : 0u));
            CHECK(pool->live() == cap);
            CHECK(g_live_models.load() == base + int(cap));
        }
    }
    // Another owner keeps the evicted instance alive: the pool's own release
    // completes at once and the replacement loads without waiting for it.
    reset_world();
    Env e;
    const int base = g_live_models.load();
    auto pool = e.m->make_pool(1);
    auto mk = [&](const std::string & fn) {
        SlotSpec s = e.spec(fn);
        s.pool = pool;
        s.load_fn = &gated_delete_load;
        return e.m->make_slot(s, e.policy());
    };
    auto a = mk("gdo");
    a->request(Priority::minimal);
    CHECK(wait_for([&] { return is_ready(a); }));
    SlotSpec os = e.spec("gdo");
    os.load_fn = &gated_delete_load;
    auto other = e.m->make_slot(os, e.policy());  // no pool: adopts a's instance
    other->request(Priority::minimal);
    CHECK(is_ready(other));
    g_delete_entered = 0;
    g_delete_open = false;
    a->evict();
    CHECK(pool->live() == 0);
    auto b = mk("pro");
    b->request(Priority::current);
    CHECK(wait_for([&] { return is_ready(b); }));
    CHECK(g_delete_entered.load() == 0);
    CHECK(pool->live() == 1);
    CHECK(g_live_models.load() == base + 2);  // the other owner's and the replacement
    g_delete_open = true;
}

// A native model whose deleter takes 3 ms (functions "sd*").
LoadResult slow_delete_load(const LoadRequest & r) {
    LoadResult out = fake_load(r);
    if (!out.model || r.function.rfind("sd", 0) != 0) return out;
    auto inner = std::static_pointer_cast<FakeModel>(out.model);
    out.model = std::shared_ptr<FakeModel>(new FakeModel(inner->name), [](FakeModel * m) {
        std::this_thread::sleep_for(3ms);
        delete m;
    });
    return out;
}

// References a pool drops while another member or a Handle still counts the
// instance are accounted too (§2.9): whichever reference is released last, the
// place comes back only after the native deleter ran.
void case_resident_cap_shared_release() {
    // Two members share the victim: the worker evicts both, and the graveyard
    // releases their references in either order.
    {
        reset_world();
        Env e;
        const int base = g_live_models.load();
        auto pool = e.m->make_pool(1);
        auto mk = [&](const std::string & fn) {
            SlotSpec s = e.spec(fn);
            s.pool = pool;
            s.load_fn = &gated_delete_load;
            return e.m->make_slot(s, e.policy());
        };
        auto b = mk("gds");
        b->request(Priority::minimal);
        CHECK(wait_for([&] { return is_ready(b); }));
        auto c = mk("gds");
        c->request(Priority::minimal);
        CHECK(is_ready(c));  // adopts b's instance: one place
        CHECK(pool->live() == 1);
        CHECK(g_live_models.load() == base + 1);
        g_delete_entered = 0;
        g_delete_open = false;
        auto d = mk("srr");
        d->request(Priority::current);  // the worker evicts b and c
        CHECK(wait_for([] { return g_delete_entered.load() == 1; }));
        const auto end = Clock::now() + 200ms;
        int over = 0;
        while (Clock::now() < end) {
            if (calls("srr") != 0 || is_ready(d) || g_live_models.load() > base + 1 || pool->live() != 1) ++over;
            std::this_thread::sleep_for(1ms);
        }
        CHECK(over == 0);
        CHECK(calls("srr") == 0);
        g_delete_open = true;
        CHECK(wait_for([&] { return is_ready(d); }));
        CHECK(pool->live() == 1);
        CHECK(g_live_models.load() == base + 1);
    }
    // An explicit eviction of a held instance races the Handle's release, both
    // started together; the native deleter is slow. Whichever reference goes
    // last, the replacement never overlaps it.
    int over_rounds = 0;
    for (int i = 0; i < 60; ++i) {
        reset_world();
        Env e;
        const std::string tag = std::to_string(i);
        const int base = g_live_models.load();
        auto pool = e.m->make_pool(1);
        auto mk = [&](const std::string & fn) {
            SlotSpec s = e.spec(fn);
            s.pool = pool;
            s.load_fn = &slow_delete_load;
            return e.m->make_slot(s, e.policy());
        };
        auto a = mk("sd" + tag);
        a->request(Priority::minimal);
        CHECK(wait_for([&] { return is_ready(a); }));
        Acquired h = a->acquire(Demand::now);
        CHECK(h);
        auto b = mk("sr" + tag);
        b->request(Priority::current);  // cap-blocked: a's instance is held
        std::atomic<bool> go{false};
        std::thread t1([&] {
            while (!go.load()) std::this_thread::yield();
            a->evict();
        });
        std::thread t2([&] {
            while (!go.load()) std::this_thread::yield();
            h = Acquired{};
        });
        go = true;
        int max_live = 0;
        const auto end = Clock::now() + 12ms;
        while (Clock::now() < end) max_live = std::max(max_live, g_live_models.load() - base);
        t1.join();
        t2.join();
        CHECK(wait_for([&] { return is_ready(b); }));
        max_live = std::max(max_live, g_live_models.load() - base);
        if (max_live > 1) ++over_rounds;
    }
    CHECK(over_rounds == 0);
}

// A pooled wait=call acquisition queued behind a long load keeps nothing alive
// while it waits: when the only owner of the registry instance it would adopt
// goes away, that instance is released at once, not when the waiter returns.
void case_call_wait_pins_nothing() {
    reset_world();
    Env e;
    const int base = g_live_models.load();
    auto pool = e.m->make_pool(1);
    auto mk = [&](const std::string & fn) {
        SlotSpec s = e.spec(fn);
        s.pool = pool;
        return e.m->make_slot(s, e.policy(Wait::call));
    };
    auto a = mk("wpa");
    a->request(Priority::minimal);
    CHECK(wait_for([&] { return is_ready(a); }));
    auto x = e.slot("wps");  // another owner of the shared instance, no pool
    x->request(Priority::minimal);
    CHECK(wait_for([&] { return is_ready(x); }));
    set_beh("wpl", {true});
    auto l = e.slot("wpl");  // a long load holding the gate
    l->request(Priority::minimal);
    CHECK(wait_for([] { return in_progress("wpl"); }));
    auto b = mk("wps");  // full pool: a queued job, not an adoption
    std::atomic<bool> done{false};
    Acquired hb;
    std::thread acq([&] {
        hb = b->acquire(Demand::now);
        done = true;
    });
    CHECK(wait_for([&] { return b->state() == SlotState::queued; }));
    std::this_thread::sleep_for(30ms);  // several polls of the wait
    CHECK(g_live_models.load() == base + 2);
    x.reset();  // the only owner of the shared instance goes away
    CHECK(wait_for([&] { return g_live_models.load() == base + 1; }, 1000));
    CHECK(!done.load());
    release("wpl");
    acq.join();
    CHECK(hb);
    CHECK(calls("wps") == 2);  // the instance died: a fresh load
    CHECK(pool->live() == 1);
}

// The pool count drops when an instance is released or evicted, not when the
// last strong reference is finally gone (§2.9): a release wakes the replacement
// at once, never at the periodic recheck, and one replacement evicts once.
void case_resident_cap_release_window() {
    Config c = fast_config();
    c.memory_recheck = 10s;  // a load that waits for the recheck shows up as a stall
    Env e(c);
    int slow = 0;
    int over_evicted = 0;
    for (int i = 0; i < 40; ++i) {
        const std::string tag = std::to_string(i);
        auto pool = e.m->make_pool(1);
        auto mk = [&](const std::string & fn) {
            SlotSpec s = e.spec(fn);
            s.pool = pool;
            return e.m->make_slot(s, e.policy());
        };
        auto a = mk("xa" + tag);
        auto b = mk("xb" + tag);
        a->request(Priority::minimal);
        CHECK(wait_for([&] { return is_ready(a); }));
        Acquired held = a->acquire(Demand::now);
        b->request(Priority::current);
        std::this_thread::sleep_for(std::chrono::microseconds(100 * (i % 5)));
        held = Acquired{};
        if (!wait_for([&] { return is_ready(b); }, 2000)) ++slow;

        auto pool3 = e.m->make_pool(3);
        std::vector<std::shared_ptr<Slot>> w;
        for (int k = 0; k < 4; ++k) {
            SlotSpec s = e.spec("xw" + tag + "_" + std::to_string(k));
            s.pool = pool3;
            w.push_back(e.m->make_slot(s, e.policy()));
        }
        for (int k = 0; k < 3; ++k) {
            w[k]->request(Priority::minimal);
            CHECK(wait_for([&] { return is_ready(w[k]); }));
        }
        const uint64_t ev0 = e.m->stats().evictions;
        w[3]->request(Priority::current);
        CHECK(wait_for([&] { return is_ready(w[3]); }));
        std::this_thread::sleep_for(1ms);
        if (e.m->stats().evictions - ev0 != 1) ++over_evicted;
        CHECK(pool3->live() == 3);
    }
    CHECK(slow == 0);
    CHECK(over_evicted == 0);
}

void case_key_lock_requeue() {
    Config cfg = fast_config();
    cfg.backoff_initial = milliseconds(60);  // wide enough that "other" runs inside it
    cfg.backoff_max = milliseconds(200);
    Env e(cfg);
    Behaviour busy;
    busy.would_block = 2;
    set_beh("prepare:busy", busy);
    auto gen = e.generated("busyfn", "busy");
    gen->request(Priority::now);
    CHECK(wait_for([] { return calls("prepare:busy") >= 1; }));
    auto other = e.slot("other");
    other->request(Priority::minimal);
    CHECK(wait_for([&] { return is_ready(gen) && is_ready(other); }));
    const auto log = log_snapshot();
    CHECK(calls("prepare:busy") == 3);
    // Another job ran while the busy key backed off (the gate was released).
    CHECK(log_index(log, "load:other") >= 0 && log_index(log, "load:other") < log_index(log, "prepare:busy", 1));
    CHECK(e.m->stats().requeues >= 2);
    CHECK(max_active() == 1);
}

void case_memory_admission() {
    Env e;
    g_mem = 1 << 20;  // 1 MiB available
    const uint64_t reserve = uint64_t(100) << 20;
    auto mk = [&](const std::string & fn, ComputeUnits u, Wait w = Wait::none, bool strict = false) {
        SlotSpec s = e.spec(fn, u);
        s.ane_compile_reserve_bytes = reserve;
        SlotPolicy p = e.policy(w);
        p.strict = strict;
        return e.m->make_slot(s, p);
    };
    auto cold = mk("ane-cold", ComputeUnits::cpu_and_ane);
    auto hinted = mk("ane-hinted", ComputeUnits::all);
    auto gpu = mk("gpu", ComputeUnits::cpu_and_gpu);
    seed_hint(hinted, 10);
    cold->request(Priority::now);
    hinted->request(Priority::minimal);
    gpu->request(Priority::minimal);
    // Hinted and non-ANE loads, requested later, run past the deferred one.
    CHECK(wait_for([&] { return is_ready(hinted) && is_ready(gpu); }));
    CHECK(cold->state() == SlotState::deferred_memory);
    CHECK(calls("ane-cold") == 0);
    std::this_thread::sleep_for(30ms);
    const SlotStatus st = cold->status();
    CHECK(st.deferred_ms >= 20);
    const auto j = nlohmann::json::parse(to_json(st));
    CHECK(j["state"] == "deferred(memory)" && j.contains("deferred_ms"));

    // call falls back at once; strict throws; construct counts it finished.
    auto as_call = mk("ane-cold", ComputeUnits::cpu_and_ane, Wait::call);
    Acquired a;
    must_complete("call acquisition on deferred(memory)", [&] { a = as_call->acquire(Demand::now); });
    CHECK(!a && a.pending);
    auto as_strict = mk("ane-cold", ComputeUnits::cpu_and_ane, Wait::call, true);
    bool threw = false;
    try {
        as_strict->acquire(Demand::now);
    } catch (const CoremlError & err) {
        threw = err.state == SlotState::deferred_memory;
    }
    CHECK(threw);
    const PrewarmResult r = e.m->issue_group(GroupKind::construction, {{cold, Priority::minimal}}, true)->wait(2000ms);
    CHECK(!r.success && r.slots.size() == 1 && r.slots[0].state == SlotState::deferred_memory);

    // Raising the reading admits it.
    g_mem = uint64_t(8) << 30;
    e.m->notify();
    CHECK(wait_for([&] { return is_ready(cold); }));
    CHECK(calls("ane-cold") == 1);
}

// Memory admission while an unrelated job holds the gate: the cold ANE load is
// marked deferred(memory) at once, waits do not sit behind the gate holder,
// and the periodic recheck still runs.
void case_memory_behind_gate() {
    Env e;
    g_mem = 1 << 20;
    const uint64_t reserve = uint64_t(100) << 20;
    auto mk = [&](Wait w = Wait::none, bool strict = false) {
        SlotSpec s = e.spec("ane-gated", ComputeUnits::cpu_and_ane);
        s.ane_compile_reserve_bytes = reserve;
        SlotPolicy p = e.policy(w);
        p.strict = strict;
        return e.m->make_slot(s, p);
    };
    set_beh("gate-holder", {true});
    auto holder = e.slot("gate-holder");
    holder->request(Priority::minimal);
    CHECK(wait_for([] { return in_progress("gate-holder"); }));

    auto cold = mk();
    cold->request(Priority::now);
    CHECK(wait_for([&] { return cold->state() == SlotState::deferred_memory; }));
    CHECK(in_progress("gate-holder"));  // decided without the gate

    auto as_call = mk(Wait::call);
    Acquired a;
    must_complete("call acquisition behind the gate", [&] { a = as_call->acquire(Demand::now); });
    CHECK(!a && a.pending);
    auto as_strict = mk(Wait::call, true);
    std::optional<SlotState> thrown;
    must_complete("strict acquisition behind the gate", [&] {
        try {
            as_strict->acquire(Demand::now);
        } catch (const CoremlError & err) {
            thrown = err.state;
        }
    });
    CHECK(thrown == SlotState::deferred_memory);
    auto g = e.m->issue_group(GroupKind::construction, {{cold, Priority::minimal}}, true);
    PrewarmResult r;
    must_complete("construct group behind the gate", [&] { r = g->wait(10s); });
    CHECK(g->finished() && !r.success);

    // The periodic recheck lifts the deferral with the gate still held (no notify).
    g_mem = uint64_t(8) << 30;
    CHECK(wait_for([&] { return cold->state() == SlotState::queued; }));
    CHECK(in_progress("gate-holder"));
    CHECK(calls("ane-gated") == 0);
    release("gate-holder");
    CHECK(wait_for([&] { return is_ready(cold); }));
    CHECK(calls("ane-gated") == 1);
}

void case_evicted_after_ready_group() {
    Env e;
    auto s = e.slot("ga");
    auto group = e.m->issue_group(GroupKind::construction, {{s, Priority::minimal}}, true);
    CHECK(group->wait(3000ms).success);
    s->evict();  // e.g. ACE-Step low-memory mode right after preparation
    CHECK(s->state() == SlotState::idle);
    const PrewarmResult r = group->result();
    CHECK(r.success && r.slots.size() == 1 && r.slots[0].state == SlotState::ready);
}

void case_strict() {
    Env e;
    SlotPolicy strict = e.policy(Wait::call);
    strict.strict = true;
    auto thrown_state = [&](const std::shared_ptr<Slot> & s) -> std::optional<SlotState> {
        try {
            s->acquire(Demand::now);
        } catch (const CoremlError & err) {
            return err.state;
        }
        return std::nullopt;
    };
    auto expect_throw = [&](const std::shared_ptr<Slot> & s, SlotState want) { return thrown_state(s) == want; };
    SlotSpec absent = e.spec("absent");
    absent.route = Route::none;
    absent.none_state = SlotState::absent;
    CHECK(expect_throw(e.m->make_slot(absent, strict), SlotState::absent));

    Behaviour bad;
    bad.fail = true;
    set_beh("strict-fail", bad);
    CHECK(expect_throw(e.m->make_slot(e.spec("strict-fail"), strict), SlotState::failed));

    set_beh("strict-slow", {true});
    SlotPolicy timed = strict;
    timed.strict_wait_ms = 50;
    const auto t0 = Clock::now();
    const std::optional<SlotState> slow = thrown_state(e.m->make_slot(e.spec("strict-slow"), timed));
    CHECK(slow == SlotState::loading || slow == SlotState::queued);
    CHECK(ms_since(t0) >= 45);
    release("strict-slow");

    auto ok = e.m->make_slot(e.spec("strict-ok"), strict);
    Acquired a = ok->acquire(Demand::now);
    CHECK(a && a.label() == "coreml-gpu");
}

void case_retire() {
    Env e;
    auto s1 = e.slot("ret");
    auto s2 = e.slot("ret");
    s1->request(Priority::minimal);
    s2->request(Priority::minimal);
    CHECK(wait_for([&] { return is_ready(s1) && is_ready(s2); }));
    Acquired held = s1->acquire(Demand::if_ready);
    CHECK(held);
    held.model->retire("prediction failed");  // process-wide for the key
    CHECK(s1->state() == SlotState::retired && s2->state() == SlotState::retired);
    CHECK(held.model->retired() && held.model->as<FakeModel>()->name == "ret");  // the unit finishes
    Acquired later = s2->acquire(Demand::now);
    CHECK(!later && !later.pending);
    auto s3 = e.slot("ret");
    s3->request(Priority::minimal);
    CHECK(s3->state() == SlotState::retired);
    CHECK(calls("ret") == 1);
    CHECK(s2->status().reason == "prediction failed");
    // Slot::retire on a slot without a model.
    auto s4 = e.slot("ret-other");
    s4->retire("cannot serve");
    CHECK(s4->state() == SlotState::retired);
}

void case_call_wait_and_cancel() {
    Env e;
    set_beh("cw", {false, 40});
    auto s = e.slot("cw", ComputeUnits::cpu_and_gpu, Wait::call);
    const auto t0 = Clock::now();
    Acquired a = s->acquire(Demand::now);  // today's per-width blocking
    CHECK(a && ms_since(t0) >= 35);

    set_beh("cw-block", {true});
    auto b = e.slot("cw-block", ComputeUnits::cpu_and_gpu, Wait::call);
    std::atomic<bool> cancel{false};
    std::thread canceller([&] {
        std::this_thread::sleep_for(40ms);
        cancel = true;
    });
    Acquired c;
    must_complete("cancelled call acquisition", [&] { c = b->acquire(Demand::now, std::nullopt, &cancel); });
    canceller.join();
    CHECK(!c && c.pending);
    release("cw-block");
}

// ---- lifetime cases (§2.9) ------------------------------------------------------

void case_destroy_during_jobs() {
    Env e;
    const int base = g_live_models.load();
    // prepare
    set_beh("prepare:dp", {true});
    auto gen = e.generated("dpfn", "dp");
    gen->request(Priority::minimal);
    CHECK(wait_for([] { return in_progress("prepare:dp"); }));
    gen.reset();
    release("prepare:dp");
    CHECK(wait_for([&] { return e.m->stats().prepares_completed == 1; }));
    std::this_thread::sleep_for(20ms);
    CHECK(calls("dpfn") == 0);  // no subscriber left to load it
    // load
    set_beh("dl", {true});
    auto dl = e.slot("dl");
    dl->request(Priority::minimal);
    CHECK(wait_for([] { return in_progress("dl"); }));
    dl.reset();
    release("dl");
    CHECK(wait_for([&] { return e.m->stats().loads_completed == 1; }));
    CHECK(wait_for([&] { return g_live_models.load() == base; }));
    // warm
    Behaviour wb;
    wb.warm_block = true;
    set_beh("warm:dw", wb);
    SlotSpec sw = e.spec("dw");
    sw.warm_fn = &fake_warm;
    auto dw = e.m->make_slot(sw, e.policy());
    dw->request(Priority::minimal);
    CHECK(wait_for([] { return in_progress("warm:dw"); }));
    CHECK(dw->state() == SlotState::warming);
    dw.reset();
    release("warm:dw");
    CHECK(wait_for([&] { return e.m->stats().loads_completed == 2; }));
    CHECK(wait_for([&] { return g_live_models.load() == base; }));
    // The manager keeps serving other owners.
    auto after = e.slot("after");
    CHECK(after->acquire(Demand::now, Clock::now() + 10ms).state != SlotState::failed);
    CHECK(wait_for([&] { return is_ready(after); }));
}

void case_reload_during_load() {
    Env e;
    set_beh("rl", {true});
    auto first = e.slot("rl");
    first->request(Priority::minimal);
    CHECK(wait_for([] { return in_progress("rl"); }));
    first.reset();  // engine destroyed mid-load ...
    auto second = e.slot("rl");  // ... and reloaded: it joins the running job
    second->request(Priority::minimal);
    release("rl");
    CHECK(wait_for([&] { return is_ready(second); }));
    CHECK(calls("rl") == 1);
}

void case_stateful_isolation() {
    Env e;
    SlotSpec s = e.spec("state");
    s.shareable = false;  // e.g. Pocket's step: one model per owner
    auto o1 = e.m->make_slot(s, e.policy());
    auto o2 = e.m->make_slot(s, e.policy());
    o1->request(Priority::minimal);
    o2->request(Priority::minimal);
    CHECK(wait_for([&] { return is_ready(o1) && is_ready(o2); }));
    CHECK(calls("state") == 2);
    Acquired a1 = o1->acquire(Demand::if_ready);
    Acquired a2 = o2->acquire(Demand::if_ready);
    CHECK(a1 && a2 && a1.model.get() != a2.model.get());

    // A new owner never joins a destroyed owner's load still in flight, even
    // when its slot lands at the destroyed slot's address: its own load runs.
    SlotSpec r = e.spec("state-reload");
    r.shareable = false;
    set_beh("state-reload", {true});
    auto first = e.m->make_slot(r, e.policy());
    const void * first_core = &first->spec();
    first->request(Priority::minimal);
    CHECK(wait_for([] { return in_progress("state-reload"); }));
    first.reset();
    std::shared_ptr<Slot> next;
    for (int i = 0; i < 1000; ++i) {  // try for the same allocation
        next = e.m->make_slot(r, e.policy());
        if (&next->spec() == first_core) break;
    }
    next->request(Priority::minimal);
    CHECK(next->state() == SlotState::queued);  // not the running job's phase
    release("state-reload");
    CHECK(wait_for([&] { return is_ready(next); }));
    CHECK(calls("state-reload") == 2);
}

// Child process: one dlclose scenario on a freshly loaded (unpinned) plugin.
//   in-flight:  unload while the plugin's LoadFn runs;
//   immediate:  unload right after request() returns;
//   deferred:   unload while the image's first job is deferred by low power.
int child_dlclose(const std::string & mode, const std::string & plugin, const std::string & dir) {
    void * h = dlopen(plugin.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        std::fprintf(stderr, "dlopen: %s\n", dlerror());
        return 3;
    }
    using StartFn = int (*)(const char *);
    using IntFn = int (*)();
    auto start = reinterpret_cast<StartFn>(
        dlsym(h, mode == "deferred" ? "qvac_coreml_plugin_start_deferred" : "qvac_coreml_plugin_start"));
    auto started = reinterpret_cast<IntFn>(dlsym(h, "qvac_coreml_plugin_started"));
    if (!start || !started || start(dir.c_str()) != 0) return 3;
    if (mode == "in-flight" && !wait_for([&] { return started() == 1; })) return 3;
    dlclose(h);  // the host unloads the library
    void * again = dlopen(plugin.c_str(), RTLD_NOW | RTLD_NOLOAD);
    if (!again) {
        // Exit before an unmapped worker runs.
        std::fprintf(stderr, "dlclose %s: image not pinned; the scheduler threads would run unmapped code\n",
                     mode.c_str());
        std::fflush(stderr);
        _exit(1);
    }
    // Still mapped: finish the job through the pinned image.
    using VoidFn = void (*)();
    auto rel = reinterpret_cast<VoidFn>(dlsym(again, "qvac_coreml_plugin_release"));
    auto power = reinterpret_cast<VoidFn>(dlsym(again, "qvac_coreml_plugin_power_restored"));
    auto ready = reinterpret_cast<IntFn>(dlsym(again, "qvac_coreml_plugin_ready"));
    if (!rel || !power || !ready) return 3;
    if (mode == "deferred" && started() != 0) return 4;  // it really was deferred
    power();
    rel();
    if (!wait_for([&] { return ready() == 1; })) return 5;
    dlclose(again);
    return 0;
}

int run_child(const std::vector<std::string> & args, int timeout_ms = 10000) {
    std::vector<char *> argv;
    argv.push_back(const_cast<char *>(g_self.c_str()));
    for (const std::string & a : args) argv.push_back(const_cast<char *>(a.c_str()));
    argv.push_back(nullptr);
    pid_t pid = 0;
    if (posix_spawn(&pid, g_self.c_str(), nullptr, nullptr, argv.data(), environ) != 0) return -1;
    int status = 0;
    const auto t0 = Clock::now();
    while (ms_since(t0) < timeout_ms) {
        if (waitpid(pid, &status, WNOHANG) == pid) return WIFEXITED(status) ? WEXITSTATUS(status) : 128;
        std::this_thread::sleep_for(5ms);
    }
    kill(pid, SIGKILL);
    waitpid(pid, &status, 0);
    return -2;  // hung
}

void case_dlclose() {
    if (g_plugin.empty()) {
        CHECK(!"--plugin was not given");
        return;
    }
    for (const char * mode : {"immediate", "deferred", "in-flight"}) {
        const std::string dir = make_temp_dir();
        const int rc = run_child({"--child", "dlclose", mode, g_plugin, dir});
        if (rc != 0) std::fprintf(stderr, "    dlclose %s: child exit %d\n", mode, rc);
        CHECK(rc == 0);
        std::error_code ec;
        fs::remove_all(dir, ec);
    }
}

int child_exit_during_load(const std::string & dir) {
    // The per-image manager, a load in flight that never returns and a second
    // job queued behind it; then the process exits.
    struct Never {
        static LoadResult load(const LoadRequest &) {
            for (;;) std::this_thread::sleep_for(1h);
        }
    };
    SlotSpec s;
    s.engine = "child";
    s.stage = "child";
    s.units = ComputeUnits::cpu_only;
    s.route = Route::prebuilt;
    s.artifact_path = dir + "/child.mlmodelc";
    s.load_fn = &fake_load;
    set_beh("in-flight", {true});
    s.function = "in-flight";
    auto a = Manager::instance()->make_slot(s, SlotPolicy{});
    s.function = "queued";
    s.load_fn = &Never::load;
    auto b = Manager::instance()->make_slot(s, SlotPolicy{});
    a->request(Priority::minimal);
    b->request(Priority::minimal);
    if (!wait_for([] { return in_progress("in-flight"); })) return 3;
    std::exit(0);
}

void case_exit_during_load() {
    const std::string dir = make_temp_dir();
    const char * argv[] = {g_self.c_str(), "--child", "exit-during-load", dir.c_str(), nullptr};
    pid_t pid = 0;
    CHECK(posix_spawn(&pid, g_self.c_str(), nullptr, nullptr, const_cast<char * const *>(argv), environ) == 0);
    int status = 0;
    bool exited = false;
    const auto t0 = Clock::now();
    while (ms_since(t0) < 10000) {
        if (waitpid(pid, &status, WNOHANG) == pid) {
            exited = true;
            break;
        }
        std::this_thread::sleep_for(5ms);
    }
    if (!exited) {
        kill(pid, SIGKILL);
        waitpid(pid, &status, 0);
    }
    CHECK(exited);
    CHECK(exited && WIFEXITED(status) && WEXITSTATUS(status) == 0);
    std::error_code ec;
    fs::remove_all(dir, ec);
}

// ---- options, plan, fit, status (§2.12, §2.13, B.5) -------------------------

std::map<std::string, std::string> * g_env = new std::map<std::string, std::string>();
const char * fake_getenv(const char * name) {
    auto it = g_env->find(name);
    return it == g_env->end() ? nullptr : it->second.c_str();
}

void case_resolve_precedence() {
    auto resolve = [](const Options & o, std::map<std::string, std::string> env, Platform p = Platform::macos) {
        *g_env = std::move(env);
        return resolve_coreml(o, "X", &fake_getenv, p);
    };
    Options none;
    Resolved d = resolve(none, {});
    CHECK(d.use == Use::automatic && d.generate == Generate::background && d.wait == Wait::none);
    CHECK(d.prewarm == Prewarm::predicted && d.cache_dir.empty() && !d.compute_units);
    CHECK(!d.strict_wait_ms && !d.memory_budget_bytes && d.use_source == Source::default_value);
    CHECK(resolve(none, {}, Platform::ios).wait == Wait::call);  // iOS keeps blocking (R8)

    // Kill switch: presence (even "0") overrides an explicit require.
    Options req;
    req.use = Use::require;
    Resolved k = resolve(req, {{"X_COREML_DISABLE", "0"}});
    CHECK(k.use == Use::off && k.disabled_reason == "X_COREML_DISABLE" && k.use_source == Source::engine_env);
    CHECK(resolve(none, {{"Y_COREML_DISABLE", "1"}}).use == Use::automatic);  // other families untouched

    // STRICT env -> require, unless the field is explicit.
    Resolved s = resolve(none, {{"X_COREML_STRICT", ""}});
    CHECK(s.use == Use::require && s.strict() && s.acquire_wait() == Wait::call);
    Options autouse;
    autouse.use = Use::automatic;
    CHECK(resolve(autouse, {{"X_COREML_STRICT", "1"}}).use == Use::automatic);
    Options off;
    off.use = Use::off;
    CHECK(resolve(off, {}).disabled_reason == "use=off");

    // Compute units: explicit > env > stage default; "" counts as unset.
    Options units;
    units.compute_units = "cpu_only";
    Resolved u1 = resolve(units, {{"X_COREML_COMPUTE_UNITS", "cpu_and_ane"}});
    CHECK(u1.compute_units == ComputeUnits::cpu_only && u1.units_source == Source::explicit_field);
    Resolved u2 = resolve(none, {{"X_COREML_COMPUTE_UNITS", "cpu_and_ane"}});
    CHECK(u2.compute_units == ComputeUnits::cpu_and_ane && u2.units_source == Source::engine_env);
    Options empty_units;
    empty_units.compute_units = "";
    CHECK(resolve(empty_units, {{"X_COREML_COMPUTE_UNITS", "all"}}).compute_units == ComputeUnits::all);
    CHECK(u2.units_or(ComputeUnits::all) == ComputeUnits::cpu_and_ane);
    CHECK(d.units_or(ComputeUnits::cpu_and_gpu) == ComputeUnits::cpu_and_gpu);
    Resolved bad_env = resolve(none, {{"X_COREML_COMPUTE_UNITS", "ane"}});
    CHECK(!bad_env.compute_units && bad_env.warnings.size() == 1);
    Options bad;
    bad.compute_units = "neural";
    bool threw = false;
    try {
        resolve(bad, {});
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw);

    // Cache dir: explicit > QVAC_COREML_CACHE_DIR > "" (an explicit "" wins).
    Options cache;
    cache.cache_dir = "/explicit";
    CHECK(resolve(cache, {{"QVAC_COREML_CACHE_DIR", "/env"}}).cache_dir == "/explicit");
    Resolved c2 = resolve(none, {{"QVAC_COREML_CACHE_DIR", "/env"}});
    CHECK(c2.cache_dir == "/env" && c2.cache_dir_source == Source::shared_env);
    Options no_cache;
    no_cache.cache_dir = "";
    CHECK(resolve(no_cache, {{"QVAC_COREML_CACHE_DIR", "/env"}}).cache_dir.empty());
    CHECK(resolve(none, {{"X_COREML_CACHE_DIR", "/nope"}}).cache_dir.empty());  // no per-engine cache var

    // Remaining fields.
    Options all;
    all.generate = Generate::never;
    all.wait = Wait::construct;
    all.prewarm = Prewarm::all;
    all.strict_wait_ms = 1500;
    all.memory_budget_bytes = 1 << 30;
    Resolved a = resolve(all, {}, Platform::ios);
    CHECK(a.generate == Generate::never && a.wait == Wait::construct && a.prewarm == Prewarm::all);
    CHECK(a.strict_wait_ms == 1500 && a.memory_budget_bytes == (1 << 30));
    CHECK(SlotPolicy::from(a).wait == Wait::none);   // construct: acquisitions use the bounded wait
    CHECK(SlotPolicy::from(s).wait == Wait::call);   // strict behaves like call
    Options neg;
    neg.strict_wait_ms = -1;
    threw = false;
    try {
        resolve(neg, {});
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw);
}

void case_plan_route() {
    Env e;
    Resolved r;
    r.cache_dir = e.cache();
    PlanInput in;
    in.coreml_built = true;
    in.generation_built = true;
    in.template_available = true;
    in.template_matched = true;
    in.platform = Platform::macos;
    in.os = {15, 7, 3};
    in.units = ComputeUnits::all;
    in.validated_units = {ComputeUnits::all};
    CHECK(plan(in, r).route == Route::generate);

    auto with = [&](auto mutate) {
        PlanInput x = in;
        Resolved y = r;
        mutate(x, y);
        return plan(x, y);
    };
    Plan p = with([](PlanInput & x, Resolved &) { x.coreml_built = false; });
    CHECK(p.route == Route::none && p.none_state == SlotState::unsupported && p.reason == "build");
    p = with([](PlanInput &, Resolved & y) {
        y.use = Use::off;
        y.disabled_reason = "memory";
    });
    CHECK(p.route == Route::none && p.none_state == SlotState::disabled && p.reason == "memory");
    // Adjacent prebuilt wins and is never suppressed by OS floor or validation.
    p = with([](PlanInput & x, Resolved &) {
        x.prebuilt_present = true;
        x.os = {14, 6, 0};
        x.units = ComputeUnits::cpu_and_ane;
    });
    CHECK(p.route == Route::prebuilt);
    p = with([](PlanInput & x, Resolved &) { x.template_available = false; });
    CHECK(p.none_state == SlotState::absent);
    p = with([](PlanInput & x, Resolved &) { x.template_matched = false; });
    CHECK(p.none_state == SlotState::no_template);
    p = with([](PlanInput &, Resolved & y) { y.generate = Generate::never; });
    CHECK(p.none_state == SlotState::disabled && p.reason == "generate=never");
    p = with([](PlanInput &, Resolved & y) { y.cache_dir.clear(); });
    CHECK(p.none_state == SlotState::unsupported && p.reason == "no_cache_dir");
    p = with([](PlanInput & x, Resolved &) { x.os = {14, 6, 0}; });
    CHECK(p.reason == "os");
    p = with([](PlanInput & x, Resolved &) {
        x.multifunction_ane = true;
        x.os = {15, 3, 0};
    });
    CHECK(p.reason == "os");
    p = with([](PlanInput & x, Resolved &) { x.multifunction_ane = true; });
    CHECK(p.route == Route::generate);
    p = with([](PlanInput & x, Resolved &) { x.platform = Platform::ios; });
    CHECK(p.reason == "os");
    p = with([](PlanInput & x, Resolved &) { x.device_excluded = true; });
    CHECK(p.reason == "device");
    p = with([](PlanInput & x, Resolved &) { x.units = ComputeUnits::cpu_and_ane; });
    CHECK(p.route == Route::none && p.reason == "unvalidated_units");
    CHECK(state_label(p.none_state, p.reason) == "unsupported(unvalidated_units)");

    // plan() never reads the cache: identical before and after artifacts and
    // hints appear (fit-stable).
    const Plan before = plan(in, r);
    fs::create_directories(e.cache() + "/qvac-coreml-v1/hints");
    detail::HintRecord rec;
    rec.warm_load_ms = 5;
    detail::write_hint(e.cache() + "/qvac-coreml-v1/hints/x.json", rec);
    const Plan after = plan(in, r);
    CHECK(before.route == after.route && before.reason == after.reason);
}

void case_memory_degrade() {
    Resolved r;
    FitVerdict v;
    v.coreml_fits = false;
    v.ggml_only_fits = true;
    CHECK(!apply_memory_degrade(r, v) && r.use == Use::automatic);  // no budget, no degrade
    r.memory_budget_bytes = int64_t(1) << 30;
    CHECK(apply_memory_degrade(r, v) && r.use == Use::off && r.disabled_reason == "memory");
    const FitReport rep = make_fit_report(r, v);
    CHECK(rep.fits);  // the working ggml verdict is kept
    const auto j = nlohmann::json::parse(to_json(rep));
    CHECK(j["fits"] == true && j["coreml"]["fits"] == false && j["coreml"]["disabled_reason"] == "memory" &&
          j["ggml_only"]["fits"] == true);
    Resolved both;
    both.memory_budget_bytes = 1;
    FitVerdict fail_both;
    fail_both.coreml_fits = false;
    fail_both.ggml_only_fits = false;
    CHECK(!apply_memory_degrade(both, fail_both) && both.use == Use::automatic);
    Resolved fine;
    fine.memory_budget_bytes = 1;
    CHECK(!apply_memory_degrade(fine, FitVerdict{}) && fine.use == Use::automatic);
}

void case_status_json() {
    Env e;
    auto s = e.slot("st", ComputeUnits::cpu_and_ane);
    seed_hint(s, 12);
    s->request(Priority::minimal);
    CHECK(wait_for([&] { return is_ready(s); }));
    GenerationStatus gen;
    gen.supported = false;
    gen.reason = "unvalidated_units";
    const EngineStatus es = e.m->status("chatterbox", e.cache(), {s}, gen);
    const auto j = nlohmann::json::parse(to_json(es));
    CHECK(j["schema"] == 1 && j["engine"] == "chatterbox" && j["cache_dir"] == e.cache());
    CHECK(j["generation"]["supported"] == false && j["generation"]["reason"] == "unvalidated_units");
    const auto & st = j["stages"][0];
    for (const char * key : {"stage", "function", "state", "route", "compute_units", "label", "artifact_key", "path",
                             "hint", "timings_ms", "bytes_on_disk", "error", "error_class"}) {
        CHECK(st.contains(key));
    }
    for (const char * key : {"queued", "digest", "write", "compile", "load", "warm"}) CHECK(st["timings_ms"].contains(key));
    CHECK(st["state"] == "ready" && st["route"] == "prebuilt" && st["compute_units"] == "cpu_and_ane");
    CHECK(st["label"] == "coreml-ane" && st["error"].is_null() && st["error_class"].is_null());
    CHECK(st["hint"]["stale"] == false && st["hint"]["warm_load_ms"].is_number());

    CHECK(std::string(label_for(ComputeUnits::all)) == "coreml-all");
    CHECK(std::string(label_for(ComputeUnits::cpu_and_gpu)) == "coreml-gpu");
    CHECK(std::string(label_for(ComputeUnits::cpu_only)) == "coreml-cpu");
    CHECK(merge_labels("", "coreml-ane") == "coreml-ane");
    CHECK(merge_labels("coreml-ane", "coreml-ane") == "coreml-ane");
    CHECK(merge_labels("coreml-ane", "ggml") == "mixed");
    Acquired none;
    CHECK(none.label() == "ggml");
}

void case_hint_keys() {
    using detail::sha256;
    using detail::to_hex;
    CHECK(to_hex(sha256("abc", 3)) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    CHECK(to_hex(sha256("", 0)) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    detail::Sha256 h;
    const std::string chunk(1000, 'a');
    for (int i = 0; i < 1000; ++i) h.update(chunk);
    CHECK(to_hex(h.finish()) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    // Appendix B.4 hint_id, checked against an independent implementation.
    CHECK(detail::hint_id("host", "/a/b.mlmodelc", 123, "main", "all", "24G419") == "beee5eda1b4787deef6745db7228c2a0");
    const int64_t t = 1767225600;  // 2026-01-01T00:00:00Z
    CHECK(detail::iso8601_utc(t) == "2026-01-01T00:00:00Z");
    CHECK(detail::parse_iso8601_utc("2026-01-01T00:00:00Z") == t);
}

struct Case {
    const char * name;
    void (*fn)();
};

const Case kCases[] = {
    {"priority-order", case_priority_order},
    {"dedup-two-owners", case_dedup_two_owners},
    {"cancel-when-owners-drop", case_cancel_when_owners_drop},
    {"admission-gate", case_admission_gate},
    {"current-outranks-minimal", case_current_outranks_minimal},
    {"bounded-wait-queued", case_bounded_wait_queued},
    {"bounded-wait-loading", case_bounded_wait_loading},
    {"stale-hint-cooldown", case_stale_hint_cooldown},
    {"warm-queued-behind-cold", case_warm_queued_behind_cold},
    {"queued-behind-expiry", case_queued_behind_expiry},
    {"prepare-ahead-no-wait", case_prepare_ahead_no_wait},
    {"groups-construct", case_groups_construct},
    {"nothing-to-prepare", case_nothing_to_prepare},
    {"low-power", case_low_power},
    {"handoff-ineligible", case_handoff_ineligible},
    {"eviction", case_eviction},
    {"resident-cap", case_resident_cap},
    {"resident-cap-retained-handle", case_resident_cap_retained_handle},
    {"resident-cap-acquire", case_resident_cap_acquire},
    {"resident-cap-two-pools", case_resident_cap_two_pools},
    {"resident-cap-cross-owner", case_resident_cap_cross_owner},
    {"resident-cap-release-window", case_resident_cap_release_window},
    {"resident-cap-queued-adopt", case_resident_cap_queued_adopt},
    {"resident-cap-pending-release", case_resident_cap_pending_release},
    {"resident-cap-shared-release", case_resident_cap_shared_release},
    {"call-wait-pins-nothing", case_call_wait_pins_nothing},
    {"key-lock-requeue", case_key_lock_requeue},
    {"memory-admission", case_memory_admission},
    {"memory-behind-gate", case_memory_behind_gate},
    {"evicted-after-ready-group", case_evicted_after_ready_group},
    {"strict", case_strict},
    {"retire", case_retire},
    {"call-wait-and-cancel", case_call_wait_and_cancel},
    {"destroy-during-prepare-load-warm", case_destroy_during_jobs},
    {"reload-during-load", case_reload_during_load},
    {"stateful-isolation", case_stateful_isolation},
    {"dlclose", case_dlclose},
    {"exit-during-load", case_exit_during_load},
    {"resolve-precedence", case_resolve_precedence},
    {"plan-route", case_plan_route},
    {"memory-degrade", case_memory_degrade},
    {"status-json", case_status_json},
    {"hint-keys", case_hint_keys},
};

}  // namespace

int main(int argc, char ** argv) {
    g_self = argv[0];
    std::string only;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--plugin" && i + 1 < argc) {
            g_plugin = argv[++i];
        } else if (a == "--only" && i + 1 < argc) {
            only = argv[++i];
        } else if (a == "--child" && i + 2 < argc && std::string(argv[i + 1]) == "exit-during-load") {
            return child_exit_during_load(argv[i + 2]);
        } else if (a == "--child" && i + 4 < argc && std::string(argv[i + 1]) == "dlclose") {
            g_case = "dlclose";
            return child_dlclose(argv[i + 2], argv[i + 3], argv[i + 4]);
        } else {
            std::fprintf(stderr, "usage: %s --plugin <module> [--only <case>]\n", argv[0]);
            return 2;
        }
    }
    int ran = 0;
    for (const Case & c : kCases) {
        if (!only.empty() && only != c.name) continue;
        g_case = c.name;
        reset_world();
        const int before = g_failures;
        const auto t0 = Clock::now();
        c.fn();
        ++ran;
        std::printf("%s %s (%lld ms)\n", g_failures == before ? "ok  " : "FAIL", c.name, (long long) ms_since(t0));
        std::fflush(stdout);
    }
    if (ran == 0) {
        std::fprintf(stderr, "no case named '%s'\n", only.c_str());
        return 2;
    }
    std::printf("%s: %d cases, %d checks, %d failures\n", g_failures == 0 ? "PASS" : "FAIL", ran, g_checks,
                g_failures);
    return g_failures == 0 ? 0 : 1;
}
