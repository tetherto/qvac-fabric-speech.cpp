// bench-ttr.h: the time-to-ready bookkeeping of bench-load-driver.mm, kept
// free of Core ML and of the runtime so test_bench_ttr.cpp can drive it with
// fakes on any host.
//
//   run_manual    the hand-built baseline: one prepare thread and one load
//                 thread, optionally sharing one mutex (the gate by hand).
//                 Every model that loads is retained until the caller drops
//                 the outcome, so both baselines measure the same resident
//                 workload as the Manager, whose slots hold their models.

#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace bench_ttr {

struct ManualStage {
    std::string stage;
    std::string path;
    std::vector<std::string> functions;
};

struct ManualResult {
    std::string slot;  // stage/function
    double ready_ms;   // -1 on failure
    bool ok;
};

struct ManualOutcome {
    std::vector<ManualResult> results;
    std::vector<std::shared_ptr<void>> retained;  // every model that loaded and warmed
};

// prepare(stage, path, out_path) -> bool, run only when `generate`;
// load(stage, path, function) -> std::shared_ptr<void>, the loaded and warmed
// model or null on failure; now() -> ms.
template <class Prepare, class Load, class Now>
ManualOutcome run_manual(const std::vector<ManualStage> & stages, bool generate, bool gated, Prepare prepare,
                         Load load, Now now, double t_issue) {
    struct Item {
        std::string stage, path, fn;
    };
    std::mutex gate, q_mu;
    std::condition_variable q_cv;
    std::deque<Item> loads;
    bool prepares_done = false;
    ManualOutcome out;

    std::thread prep([&] {
        for (const ManualStage & s : stages) {
            std::string path = s.path;
            if (generate) {
                std::unique_lock<std::mutex> g(gate, std::defer_lock);
                if (gated) g.lock();
                std::string prepared;
                if (prepare(s.stage, s.path, prepared)) path = prepared;
                else path.clear();
            }
            std::lock_guard<std::mutex> lk(q_mu);
            for (const std::string & fn : s.functions) loads.push_back({s.stage, path, fn});
            q_cv.notify_all();
        }
        std::lock_guard<std::mutex> lk(q_mu);
        prepares_done = true;
        q_cv.notify_all();
    });

    std::thread loader([&] {
        while (true) {
            Item it;
            {
                std::unique_lock<std::mutex> lk(q_mu);
                q_cv.wait(lk, [&] { return !loads.empty() || prepares_done; });
                if (loads.empty()) return;
                it = loads.front();
                loads.pop_front();
            }
            std::shared_ptr<void> model;
            if (!it.path.empty()) {
                std::unique_lock<std::mutex> g(gate, std::defer_lock);
                if (gated) g.lock();
                model = load(it.stage, it.path, it.fn);
            }
            const bool ok = model != nullptr;
            out.results.push_back({it.stage + "/" + it.fn, ok ? now() - t_issue : -1.0, ok});
            if (ok) out.retained.push_back(std::move(model));
        }
    });
    prep.join();
    loader.join();
    return out;
}

inline double manual_ready_ms(const std::vector<ManualResult> & results) {
    double m = 0;
    for (const ManualResult & r : results) {
        if (r.ready_ms < 0) return -1;
        m = std::max(m, r.ready_ms);
    }
    return m;
}

}  // namespace bench_ttr
