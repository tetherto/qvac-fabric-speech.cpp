#pragma once

// Generation glue for engines: the template a stage generates from, its match
// against the engine's GGUF, the PrepareFn of route `generate` and the lease a
// LoadFn holds on a generated artifact.
//
// Every function is available in every build. Without <P>_COREML_GENERATE
// (built() false) probe() reports the template as unavailable, so plan()
// never chooses `generate` and nothing here touches the cache.

#include "qvac_coreml/options.h"
#include "qvac_coreml/slot.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace QVAC_COREML_NS {
namespace generation {

// Whether this library compiled the generator and its templates in.
bool built();

// template.json `memory` and the I/O of its functions: what the Core ML
// envelope of a generated stage charges. Optional values are the template's
// "use the default" nulls.
struct TemplateMemory {
    uint64_t weight_bin_size = 0;
    uint64_t runtime_weight_bytes = 0;
    uint64_t workspace_bytes = 0;
    uint64_t state_bytes = 0;
    uint64_t generation_peak_bytes = 0;
    uint64_t io_bytes = 0;  // every function's inputs and outputs, at their dtypes
    std::optional<uint64_t> compile_peak_bytes;
    std::optional<uint64_t> ane_compile_reserve_bytes;
    std::optional<uint64_t> load_peak_cpu_and_ane;
    std::optional<uint64_t> load_peak_all;
    std::optional<uint64_t> load_peak_cpu_and_gpu;
    std::optional<uint64_t> load_peak_cpu_only;
    double margin = 1.25;
    std::vector<ComputeUnits> validated_units;  // memory.validated_for[].units
};

// The Core ML envelope of one generated stage, from its template:
//   resident  = runtime_weight_bytes + 2 x io_bytes + state + workspace
//   transient = max(prepare_peak, load_peak), where
//               prepare_peak = generation_peak_bytes + 8 MiB + compile_peak_bytes
//                              (default 0.5 x weight_bin_size)
//               load_peak    = the template's value for `units`, by default
//                              0.25 x weight_bin_size for cpu_and_ane and
//                              1.0 x for the other unit classes
// Each part is returned unmargined; envelope_bytes() applies the margin.
uint64_t resident_bytes(const TemplateMemory & m);
uint64_t transient_bytes(const TemplateMemory & m, ComputeUnits units);
uint64_t envelope_bytes(const TemplateMemory & m, ComputeUnits units);
// The memory admission reserve of a cold ANE load (default 1.6 x weight_bin_size).
uint64_t ane_compile_reserve_bytes(const TemplateMemory & m, ComputeUnits units);

// One template considered for one stage of one GGUF.
struct Candidate {
    std::string template_id;
    bool available = false;  // the library has the template (embedded, or the development override)
    bool matched = false;    // template.json `match` and every recipe source accept the GGUF
    std::string reason;      // why not available or not matched
    const void * tmpl = nullptr;  // the template, alive for the process (PrepareRequest::tmpl)
    std::string sha256_hex;
    std::optional<ComputeUnits> default_units;  // template.json default_compute_units
    TemplateMemory memory;
};

// Looks the templates up (embedded, or <resolved.template_dir>/<id> where the
// development override is compiled in) and matches each against `gguf_path`,
// opened once. Reads only GGUF metadata and tensor shapes; never the cache.
std::vector<Candidate> probe(const std::vector<std::string> & template_ids, const std::string & gguf_path,
                             const Resolved & resolved);

// The stat tuple of `path` with its realpath (PrepareRequest::source).
bool source_stat(const std::string & path, SourceStat & out);

// The PrepareRequest of route `generate` for a matched candidate.
PrepareRequest prepare_request(const Candidate & c, const SourceStat & source, const Resolved & resolved);

// PrepareFn of route `generate`: the cache pipeline (memo, digest, generate,
// compile, publish, adopt) with the on-device compile. The artifact's lease
// is released when it returns; the LoadFn takes its own (begin_load).
PrepareResult prepare(const PrepareRequest & req, PrepareContext & ctx);

// For LoadFns. A generated artifact (LoadRequest::tmpl set) loads under the
// shared lease of its cache entry: begin_load takes it and returns false
// when the load must not start (`out` then carries would_block, or the
// error when the artifact is gone). end_load, after the engine's load,
// touches used/<key> on success and releases the lease on failure. Both do
// nothing for a prebuilt artifact.
bool begin_load(const LoadRequest & req, LoadResult & out);
void end_load(const LoadRequest & req, LoadResult & out);

// Tests: replaces the on-device compile of prepare(). It writes the compiled
// bundle at `mlmodelc_out` and returns "" on success or an error message.
// Null restores the Core ML compile.
using CompileOverride = std::string (*)(const std::string & mlpackage, const std::string & mlmodelc_out);
void set_compile_override(CompileOverride fn);

}  // namespace generation
}  // namespace QVAC_COREML_NS
