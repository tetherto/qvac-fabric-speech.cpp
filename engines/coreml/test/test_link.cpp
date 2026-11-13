// test-coreml-link: tts-cpp, qvac-parakeet and audiogen
// in one binary, together with every engine's copy of the Core ML runtime.
// The link itself is the main assertion (a symbol that escaped its namespace,
// or an Objective-C class, is a duplicate); at run time each copy works and
// each has its own per-image manager.

#include "qvac_coreml_link_probes.h"

#include <algorithm>
#include <cstdio>
#include <vector>

#define QVAC_DECLARE_PROBE(ns)          \
    namespace ns {                      \
    int link_probe();                   \
    const void * link_probe_identity(); \
    }
QVAC_COREML_LINK_PROBES(QVAC_DECLARE_PROBE)
#undef QVAC_DECLARE_PROBE

int main() {
    int failures = 0;
    std::vector<const void *> ids;
#define QVAC_RUN_PROBE(ns)                                                  \
    {                                                                       \
        const int rc = ns::link_probe();                                    \
        std::printf("%s %s (rc=%d)\n", rc == 0 ? "ok  " : "FAIL", #ns, rc); \
        if (rc != 0) ++failures;                                            \
        ids.push_back(ns::link_probe_identity());                           \
    }
    QVAC_COREML_LINK_PROBES(QVAC_RUN_PROBE)
#undef QVAC_RUN_PROBE
    std::vector<const void *> sorted = ids;
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
        std::printf("FAIL two engines share one Core ML manager\n");
        ++failures;
    }
    std::printf("%s: %zu runtime copies in one image\n", failures == 0 ? "PASS" : "FAIL", ids.size());
    return failures == 0 ? 0 : 1;
}
