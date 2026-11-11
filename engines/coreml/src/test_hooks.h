#pragma once

// Test seams of the scheduler, compiled only into the tests' own runtime
// (QVAC_COREML_TEST_HOOKS, qvac-coreml-test-runtime). Each is called with no
// lock held and may block, so a test can hold a worker or a releasing thread
// at a point no public API can reach. Null when unset.

#include <atomic>

namespace QVAC_COREML_NS {
namespace detail {

struct TestHooks {
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
