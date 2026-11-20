#pragma once

// Process exit and Core ML (libaudiogen-cpp: ACE-Step).
//
// A Core ML compile or load cannot be cancelled, and exit() destroys
// framework state a running one still reads. This library waits for the job
// its Core ML worker is running (bounded: 15 s on macOS, 2 s on iOS, one
// deadline for the library) before any destructor runs, when exit starts on
// the main thread and the main thread is armed. The main thread is armed
// automatically when it makes a Core ML stage or requests a load; a host that
// makes its engines on another thread (an addon whose load runs on a worker)
// arms it with coreml_arm_exit_guard() on its main (JS) thread, or calls
// coreml_shutdown_for_exit() before it exits. Without either, a process that
// exits while a load runs can crash inside the frameworks.
//
// In builds without Core ML these do nothing (and return false / true).

#include "audiogen-cpp/export.h"

#include <cstdint>

namespace tts_cpp::acestep {

// Call on the main thread (the JS thread of a Bare or Node host, e.g. where
// the addon instance is created). Returns true when the guard is armed;
// false on another thread, or in a build without Core ML.
AUDIOGEN_API bool coreml_arm_exit_guard();

// Stops new Core ML work in this library for the rest of the process, then
// waits up to timeout_ms for the running job. Returns true when no job is
// running. Call before exiting, from any thread.
AUDIOGEN_API bool coreml_shutdown_for_exit(int64_t timeout_ms);

// false: the main thread is not armed automatically. For hosts whose main
// thread may end with pthread_exit() while the process goes on
// (dispatch_main()); after an armed main thread ends, later Core ML requests
// fail with "the process is exiting". Such hosts call
// coreml_shutdown_for_exit() before they exit.
AUDIOGEN_API void coreml_set_automatic_exit_guard(bool on);

}  // namespace tts_cpp::acestep
