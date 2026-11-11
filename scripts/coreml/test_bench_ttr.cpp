// Hermetic tests for bench-ttr.h (no Core ML). Built and run by
// test_bench_load_time.py; standalone:
//   c++ -std=c++17 -pthread -I scripts/coreml scripts/coreml/test_bench_ttr.cpp -o t && ./t

#include <atomic>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "bench-ttr.h"

namespace {

int g_failures = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::fprintf(stderr, "%s:%d: CHECK(%s)\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                    \
        }                                                                    \
    } while (0)

std::atomic<int> g_live{0};

struct Counted {
    Counted() { ++g_live; }
    ~Counted() { --g_live; }
};

double fake_now() {
    static std::atomic<int> t{0};
    return ++t;
}

// Every model that loads stays alive until the outcome is dropped: the live
// count seen by each load grows by one per earlier load, and after the run it
// equals the number of successful loads.
void manual_retains_every_loaded_model(bool generate, bool gated) {
    const std::vector<bench_ttr::ManualStage> stages = {{"encoder", "enc", {"enc_a", "enc_b"}},
                                                        {"cfm", "cfm", {"cfm_a", "cfm_b", "cfm_c"}}};
    std::vector<int> live_at_load;
    auto prepare = [](const std::string & stage, const std::string &, std::string & out) {
        out = "prepared-" + stage;
        return true;
    };
    auto load = [&](const std::string &, const std::string & path, const std::string &) {
        if (generate) CHECK(path.rfind("prepared-", 0) == 0);
        live_at_load.push_back(g_live.load());
        return std::static_pointer_cast<void>(std::make_shared<Counted>());
    };
    {
        bench_ttr::ManualOutcome o = bench_ttr::run_manual(stages, generate, gated, prepare, load, fake_now, 0.0);
        CHECK(o.results.size() == 5);
        CHECK(o.retained.size() == 5);
        CHECK(g_live.load() == 5);
        CHECK((live_at_load == std::vector<int>{0, 1, 2, 3, 4}));
        CHECK(bench_ttr::manual_ready_ms(o.results) > 0);
    }
    CHECK(g_live.load() == 0);
}

// A failed load is reported not-ready and is not counted as resident.
void manual_failure_is_not_retained() {
    const std::vector<bench_ttr::ManualStage> stages = {{"hift", "hift", {"ok_1", "bad", "ok_2"}}};
    auto prepare = [](const std::string &, const std::string &, std::string &) { return false; };
    auto load = [](const std::string &, const std::string &, const std::string & fn) {
        if (fn == "bad") return std::shared_ptr<void>();
        return std::static_pointer_cast<void>(std::make_shared<Counted>());
    };
    {
        bench_ttr::ManualOutcome o = bench_ttr::run_manual(stages, false, true, prepare, load, fake_now, 0.0);
        CHECK(o.retained.size() == 2);
        CHECK(g_live.load() == 2);
        CHECK(o.results.size() == 3);
        CHECK(!o.results[1].ok && o.results[1].ready_ms < 0);
        CHECK(bench_ttr::manual_ready_ms(o.results) == -1);
    }
    CHECK(g_live.load() == 0);
}

// A failed prepare fails every load of its stage without calling load.
void manual_failed_prepare_fails_its_stage() {
    const std::vector<bench_ttr::ManualStage> stages = {{"cfm", "cfm", {"a", "b"}}};
    int loads = 0;
    auto prepare = [](const std::string &, const std::string &, std::string &) { return false; };
    auto load = [&](const std::string &, const std::string &, const std::string &) {
        ++loads;
        return std::static_pointer_cast<void>(std::make_shared<Counted>());
    };
    bench_ttr::ManualOutcome o = bench_ttr::run_manual(stages, true, false, prepare, load, fake_now, 0.0);
    CHECK(loads == 0);
    CHECK(o.retained.empty());
    CHECK(bench_ttr::manual_ready_ms(o.results) == -1);
}

}  // namespace

int main() {
    for (bool generate : {false, true})
        for (bool gated : {false, true}) manual_retains_every_loaded_model(generate, gated);
    manual_failure_is_not_retained();
    manual_failed_prepare_fails_its_stage();
    if (g_failures) {
        std::fprintf(stderr, "test_bench_ttr: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_bench_ttr: ok\n");
    return 0;
}
