// test-coreml-exit-load: a process that ends while the runtime's load worker
// is inside a real Core ML model load must still exit cleanly.
//
// Each run copies the model to a fresh path (so Core ML compiles it again
// rather than reusing a plan cached for an earlier path) and spawns a child.
// The child makes a prebuilt slot on the per-image manager, requests it, waits
// until the slot is loading, sleeps for that run's delay and returns from
// main. The delays sweep 0..--max-delay-ms in equal steps (default 60 runs
// over 0-360 ms), so some exits land in each early phase of the load. The
// fatal window is the in-process validation of the program for the Neural
// Engine: for an exported Supertonic vocoder on an M2, exits 160-200 ms after
// the load started crashed without the exit wait (3 of 40 runs over 0-400 ms;
// 1 of 20 over 0-1200 ms). Every child must exit with status
// 0: a signal (the failure this guards against is a SIGSEGV inside Apple's
// compiler while exit() runs static destructors) or any other status fails.
//
// Before it starts the runtime, the child creates a static object whose
// destructor blocks for --finalizer-ms (default 500), as ggml's Metal backend
// does when it frees its devices at exit in every engine process. It is
// destroyed after the frameworks' statics (it was constructed first), so a
// load left running at exit keeps going inside a half torn-down process
// instead of being cut short by the end of a 2 ms exit. The child also runs
// with MallocScribble=1, so memory a destructor freed is overwritten at once:
// Apple's compiler keeps lookup tables in function-local statics, built on
// first use during the compile and therefore destroyed first at exit, and a
// read through a freed table then faults every time instead of only when
// another destructor happened to reuse the memory.
//
// Every compile of a fresh path adds a bundle (~20 MB for a vocoder) to this
// program's e5rt cache in ~/Library/Caches/<program name>; the bundles a run
// added are deleted after it.
//
// A run whose load already finished before the child exited checks nothing,
// so at least half of the runs must have exited with the load in flight;
// fewer fails the test as vacuous rather than passing it.
//
//   test-coreml-exit-load --model <x.mlmodelc> [--runs N] [--max-delay-ms M]
//                         [--units all|cpu_and_ane|cpu_and_gpu|cpu_only]
//                         [--finalizer-ms F]

#include "qvac_coreml/manager.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char ** environ;

using namespace QVAC_COREML_NS;

namespace {

int64_t mono_ns() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);  // system-wide: comparable across processes
    return int64_t(ts.tv_sec) * 1000000000 + ts.tv_nsec;
}

MLComputeUnits ml_units(ComputeUnits u) {
    switch (u) {
        case ComputeUnits::cpu_only: return MLComputeUnitsCPUOnly;
        case ComputeUnits::cpu_and_gpu: return MLComputeUnitsCPUAndGPU;
        case ComputeUnits::cpu_and_ane: return MLComputeUnitsCPUAndNeuralEngine;
        case ComputeUnits::all: return MLComputeUnitsAll;
    }
    return MLComputeUnitsAll;
}

LoadResult load_mlmodel(const LoadRequest & r) {
    LoadResult out;
    @autoreleasepool {
        MLModelConfiguration * cfg = [MLModelConfiguration new];
        cfg.computeUnits = ml_units(r.units);
        NSError * err = nil;
        NSURL * url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:r.artifact_path.c_str()]];
        MLModel * m = [MLModel modelWithContentsOfURL:url configuration:cfg error:&err];
        if (m == nil) {
            out.error = Error::deterministic(err ? err.localizedDescription.UTF8String : "load failed");
            return out;
        }
        out.model = std::shared_ptr<void>((__bridge_retained void *) m, [](void * p) { CFRelease(p); });
    }
    return out;
}

struct SlowFinalizer {
    int ms;
    ~SlowFinalizer() { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }
};

// Child: request the load, wait for it to be in flight, then end the process
// `delay_ms` later by returning from main. Prints the state it exited in.
int child(const char * model, const char * units_s, int delay_ms, int finalizer_ms) {
    static SlowFinalizer slow{finalizer_ms};
    std::optional<ComputeUnits> units = parse_compute_units(units_s);
    if (!units) return 64;
    SlotSpec s;
    s.engine = "exit-load";
    s.stage = "model";
    s.units = *units;
    s.route = Route::prebuilt;
    s.artifact_path = model;
    s.load_fn = &load_mlmodel;
    std::shared_ptr<Slot> slot = Manager::instance()->make_slot(s, SlotPolicy{});
    slot->request(Priority::minimal);
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    SlotState st = slot->state();
    while (st != SlotState::loading && st != SlotState::warming && st != SlotState::ready &&
           std::chrono::steady_clock::now() < end) {
        if (st == SlotState::failed) {
            std::printf("state=failed reason=%s\n", slot->status().reason.c_str());
            return 65;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        st = slot->state();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    st = slot->state();
    std::printf("state=%s exit_ns=%lld\n", to_string(st), (long long) mono_ns());
    std::fflush(stdout);
    return 0;
}

struct RunResult {
    bool exited = false;
    int status = 0;
    std::string state;
    double exit_ms = -1;  // from the child's return from main to its reaping
};

bool copy_model(const std::string & src, const std::string & dst) {
    NSError * err = nil;
    return [[NSFileManager defaultManager] copyItemAtPath:[NSString stringWithUTF8String:src.c_str()]
                                                   toPath:[NSString stringWithUTF8String:dst.c_str()]
                                                    error:&err];
}

void remove_tree(const std::string & p) {
    [[NSFileManager defaultManager] removeItemAtPath:[NSString stringWithUTF8String:p.c_str()] error:nil];
}

// The bundle directories (<os build>/<hash>) in this program's e5rt cache.
std::string e5rt_cache() {
    const char * home = std::getenv("HOME");
    return std::string(home ? home : "") + "/Library/Caches/" + getprogname() + "/com.apple.e5rt.e5bundlecache";
}

std::set<std::string> e5rt_bundles() {
    std::set<std::string> out;
    NSFileManager * fm = [NSFileManager defaultManager];
    NSString * root = [NSString stringWithUTF8String:e5rt_cache().c_str()];
    for (NSString * os in [fm contentsOfDirectoryAtPath:root error:nil]) {
        NSString * dir = [root stringByAppendingPathComponent:os];
        for (NSString * b in [fm contentsOfDirectoryAtPath:dir error:nil]) {
            out.insert([dir stringByAppendingPathComponent:b].UTF8String);
        }
    }
    return out;
}

RunResult run_child(const std::string & self, const std::string & model, const std::string & units, int delay_ms,
                    int finalizer_ms) {
    RunResult rr;
    const char * tmp = std::getenv("TMPDIR");
    std::string tmpl = std::string(tmp && *tmp ? tmp : "/tmp") + "/coreml-exit-load-XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    if (mkdtemp(buf.data()) == nullptr) return rr;
    const std::string dir = buf.data();
    std::string name = model;
    while (!name.empty() && name.back() == '/') name.pop_back();
    name = name.substr(name.find_last_of('/') + 1);
    const std::string copy = dir + "/" + name;
    const std::string out = dir + "/child.out";
    if (!copy_model(model, copy)) {
        std::fprintf(stderr, "cannot copy %s\n", model.c_str());
        remove_tree(dir);
        return rr;
    }
    const std::set<std::string> bundles_before = e5rt_bundles();
    const std::string delay = std::to_string(delay_ms), fin = std::to_string(finalizer_ms);
    const char * argv[] = {self.c_str(), "--child", copy.c_str(), units.c_str(), delay.c_str(), fin.c_str(), nullptr};
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addopen(&fa, 1, out.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    std::vector<char *> env;
    for (char ** e = environ; *e != nullptr; ++e) env.push_back(*e);
    std::string scribble = "MallocScribble=1";
    env.push_back(scribble.data());
    env.push_back(nullptr);
    pid_t pid = 0;
    const int rc = posix_spawn(&pid, self.c_str(), &fa, nullptr, const_cast<char * const *>(argv), env.data());
    posix_spawn_file_actions_destroy(&fa);
    if (rc != 0) {
        remove_tree(dir);
        return rr;
    }
    int status = 0;
    int64_t reaped_ns = 0;
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    while (std::chrono::steady_clock::now() < end) {
        if (waitpid(pid, &status, WNOHANG) == pid) {
            reaped_ns = mono_ns();
            rr.exited = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!rr.exited) {
        kill(pid, SIGKILL);
        waitpid(pid, &status, 0);
    }
    rr.status = status;
    if (FILE * f = std::fopen(out.c_str(), "r")) {
        char line[512] = {0};
        while (std::fgets(line, sizeof line, f)) {
            char st[64] = {0};
            long long ns = 0;
            if (std::sscanf(line, "state=%63s exit_ns=%lld", st, &ns) == 2) {
                rr.state = st;
                if (reaped_ns > 0) rr.exit_ms = double(reaped_ns - ns) / 1e6;
            } else if (std::sscanf(line, "state=%63s", st) == 1) {
                rr.state = st;
            }
        }
        std::fclose(f);
    }
    remove_tree(dir);
    for (const std::string & b : e5rt_bundles()) {
        if (!bundles_before.count(b)) remove_tree(b);
    }
    return rr;
}

}  // namespace

int main(int argc, char ** argv) {
    std::string model, units = "all";
    int runs = 60, max_delay = 360, finalizer_ms = 500;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--child" && i + 4 < argc) {
            return child(argv[i + 1], argv[i + 2], std::atoi(argv[i + 3]), std::atoi(argv[i + 4]));
        }
        if (a == "--model" && i + 1 < argc) {
            model = argv[++i];
        } else if (a == "--runs" && i + 1 < argc) {
            runs = std::atoi(argv[++i]);
        } else if (a == "--max-delay-ms" && i + 1 < argc) {
            max_delay = std::atoi(argv[++i]);
        } else if (a == "--units" && i + 1 < argc) {
            units = argv[++i];
        } else if (a == "--finalizer-ms" && i + 1 < argc) {
            finalizer_ms = std::atoi(argv[++i]);
        } else {
            std::fprintf(stderr,
                         "usage: %s --model <x.mlmodelc> [--runs N] [--max-delay-ms M] [--units U] "
                         "[--finalizer-ms F]\n",
                         argv[0]);
            return 2;
        }
    }
    if (model.empty() || runs < 1 || max_delay < 0 || finalizer_ms < 0 || !parse_compute_units(units)) {
        std::fprintf(stderr, "test-coreml-exit-load: --model is required; --runs >= 1; --units is a units name\n");
        return 2;
    }
    int failures = 0, in_flight = 0;
    double max_exit_ms = 0;
    for (int i = 0; i < runs; ++i) {
        const int delay = runs == 1 ? 0 : max_delay * i / (runs - 1);
        const RunResult r = run_child(argv[0], model, units, delay, finalizer_ms);
        const bool clean = r.exited && WIFEXITED(r.status) && WEXITSTATUS(r.status) == 0;
        const bool loading = r.state == "loading" || r.state == "warming";
        if (loading) ++in_flight;
        if (r.exit_ms > max_exit_ms) max_exit_ms = r.exit_ms;
        char how[64];
        if (!r.exited) {
            std::snprintf(how, sizeof how, "timed out");
        } else if (WIFSIGNALED(r.status)) {
            std::snprintf(how, sizeof how, "signal %d", WTERMSIG(r.status));
        } else {
            std::snprintf(how, sizeof how, "exit %d", WEXITSTATUS(r.status));
        }
        std::printf("%s run %d delay %d ms: state %s, %s, exit took %.1f ms\n", clean ? "ok  " : "FAIL", i, delay,
                    r.state.empty() ? "?" : r.state.c_str(), how, r.exit_ms);
        std::fflush(stdout);
        if (!clean) ++failures;
    }
    const bool vacuous = in_flight * 2 < runs;
    if (vacuous) {
        std::printf("FAIL only %d of %d runs exited with the load in flight: the model loads too fast for this test\n",
                    in_flight, runs);
    }
    std::printf("%s: %d runs, %d exited with the load in flight, %d failures, slowest exit %.1f ms\n",
                failures == 0 && !vacuous ? "PASS" : "FAIL", runs, in_flight, failures, max_exit_ms);
    return failures == 0 && !vacuous ? 0 : 1;
}
