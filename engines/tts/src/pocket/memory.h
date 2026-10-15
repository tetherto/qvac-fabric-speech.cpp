#pragma once
#include <cstdint>
#include <string>

namespace tts_cpp::pocket::detail {
// Internal metadata-only measurements. Compute includes CPU work-plan scratch;
// the CPU backend retains its largest work buffer across graph executions.
struct MemoryMeasure {
    uint64_t weights = 0, state = 0, compute = 0;
    uint64_t metadata = 0, graph_metadata = 0, load_staging = 0;
    std::string source_hash;
    std::string backend_name;
    uint64_t device_free = 0, device_total = 0;
    bool shares_host_memory = true;
};
}
