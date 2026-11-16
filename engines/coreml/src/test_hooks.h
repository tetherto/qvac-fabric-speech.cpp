#pragma once

// Test seams of the scheduler, compiled only into the tests' own runtime
// (QVAC_COREML_TEST_HOOKS, qvac-coreml-test-runtime). Each is called with no
// lock held and may block, so a test can hold a worker or a releasing thread
// at a point no public API can reach. Null when unset.

#include <atomic>

namespace QVAC_COREML_NS {
namespace detail {

struct TestHooks {
    // worker_main(): after a job was admitted and started, before the worker
    // runs it.
    std::atomic<void (*)()> job_started{nullptr};
    // worker_main(): after this worker handed the job it picked to the other
    // worker, before it releases what its scan collected.
    std::atomic<void (*)()> handed_off{nullptr};
    // run_load(): before the load worker publishes another owner's live
    // instance instead of loading (the job holds the gate meanwhile).
    std::atomic<void (*)()> before_adopt{nullptr};
    // ~PendingRelease: after the pool's reference is released, before the
    // pool is told (Runtime::on_pool_released).
    std::atomic<void (*)()> pending_released{nullptr};
};

TestHooks & test_hooks();

}  // namespace detail
}  // namespace QVAC_COREML_NS
