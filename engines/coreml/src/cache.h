#pragma once

// The content-addressed generation cache (PLAN §2.7 steps 1-5 and 7-8, §2.8,
// B.2-B.4). Portable: everything here runs on Linux; the Apple compile of
// step 6 is a CompileFn (S2b), and tests pass a fake.
//
//   <cache_dir>/qvac-coreml-v1/
//     sources/<memo_id>.<key32>.json   memo = one reference, written once under the key lock
//     artifacts/<template-id>.<key32>/{model.mlmodelc/, artifact.json}   write-once
//     locks/<key32>.lock               never deleted
//     staging/<key32>.<pid>.<rand>/    scratch, never loaded from; `template` holds the owner's template id
//     used/<key32>                     mtime = last successful load
//     hints/<hint_id>.json             (S1a)
//     failed/<key32>[.<fn_units_hash>].json
//     quarantine/<name>.<pid>/         unusable artifacts moved aside under EX
//
// Locking: every key lock is taken LOCK_NB (jobs run holding the scheduler's
// admission gate, §2.9). A busy lock is `would_block`: the job releases the
// gate and re-queues with backoff. A memo is only ever written holding the
// key's lock (shared for adoption, exclusive for the generator); prune deletes
// only holding it exclusively, so a reference cannot appear between prune's
// check and its unlink. A loaded model keeps adoption's shared lock as its
// lease until it is destroyed.

#include "qvac_coreml/recipe.h"
#include "qvac_coreml/slot.h"

#include "digest.h"
#include "gguf_source.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#ifndef QVAC_COREML_NS
#error "QVAC_COREML_NS must be defined"
#endif

namespace QVAC_COREML_NS {
namespace cache {

constexpr const char * ROOT = "qvac-coreml-v1";

// staging/<key32>.<pid>.<rand>/template: the template id that owns a staging
// directory, written when the directory is created. Prune removes only staging
// it can attribute to this library (§2.8: unknown owners are left alone).
constexpr const char * STAGING_OWNER_FILE = "template";

struct Layout {
    explicit Layout(const std::string & cache_dir);

    std::string root;
    std::string sources;
    std::string artifacts;
    std::string locks;
    std::string staging;
    std::string used;
    std::string hints;
    std::string failed;
    std::string quarantine;

    std::string artifact_dir(const std::string & template_id, const std::string & key32) const;
    std::string lock_path(const std::string & key32) const;
    std::string memo_path(const std::string & memo_id, const std::string & key32) const;
    std::string used_path(const std::string & key32) const;

    // mkdir -p of every directory (transient I/O error on failure).
    Error ensure() const;
};

// flock(2) on locks/<key32>.lock, always non-blocking.
class KeyLock {
public:
    enum class Mode { shared, exclusive };
    enum class Result { ok, would_block, error };

    KeyLock() = default;
    KeyLock(KeyLock && o) noexcept : fd_(o.fd_) { o.fd_ = -1; }
    KeyLock & operator=(KeyLock && o) noexcept;
    KeyLock(const KeyLock &) = delete;
    KeyLock & operator=(const KeyLock &) = delete;
    ~KeyLock() { release(); }

    static Result take(const Layout & l, const std::string & key32, Mode mode, KeyLock & out, Error * err = nullptr);

    bool held() const { return fd_ >= 0; }
    void release();
    // Hands the descriptor (and the lock) to the caller: a model's lease
    // (LoadResult::lease_fd, closed with the SharedModel).
    int release_fd() {
        const int fd = fd_;
        fd_ = -1;
        return fd;
    }

private:
    int fd_ = -1;
};

// ---- Memos (B.3) ---------------------------------------------------------------

struct Memo {
    std::string path;  // realpath of the GGUF
    uint64_t size = 0;
    int64_t mtime_ns = 0;
    uint64_t ino = 0;
    uint64_t dev = 0;
    std::string template_sha256;  // hex
    uint32_t generator_version = 0;
    std::string options;
    std::string template_id;
    std::string source_digest;  // hex
    std::string artifact_key;   // key32
};

std::string to_json(const Memo & m);
std::optional<Memo> memo_from_json(const std::string & text);

struct MemoFile {
    std::string file;  // sources/<memo_id>.<key32>.json
    std::string memo_id;
    std::string key32;
    std::optional<Memo> memo;  // nullopt: unreadable
    int64_t mtime_s = 0;
};

// The memo of `memo_id` (one directory scan), or nullopt.
std::optional<MemoFile> find_memo(const Layout & l, const std::string & memo_id);
// Every memo of `key32`.
std::vector<MemoFile> memos_for_key(const Layout & l, const std::string & key32);
std::vector<MemoFile> all_memos(const Layout & l);
// Atomic: sources/.tmp.<pid>.<rand>, fsync, rename. The caller holds the key lock.
// Whether sources/<memo_id>.<m.artifact_key>.json is readable and equal to `m`.
bool memo_present(const Layout & l, const std::string & memo_id, const Memo & m);
Error write_memo(const Layout & l, const std::string & memo_id, const Memo & m);

// ---- Artifacts (B.2) -----------------------------------------------------------

struct InventoryEntry {
    std::string path;  // relative to the artifact directory, e.g. model.mlmodelc/model.mil
    uint64_t bytes = 0;
};

struct ArtifactInfo {
    bool complete = true;
    std::string artifact_key;
    std::string template_id;
    std::string template_sha256;
    uint32_t generator_version = 0;
    std::string options;
    std::string source_digest;
    uint32_t source_tensors = 0;
    uint64_t source_bytes = 0;
    SourceStat provenance;
    std::string os_build;
    std::string coremlc;
    std::string export_kind;
    std::vector<std::string> functions;
    std::string default_compute_units;
    std::vector<InventoryEntry> inventory;
    int64_t digest_ms = 0;
    int64_t write_ms = 0;
    int64_t compile_ms = 0;
};

std::string to_json(const ArtifactInfo & a);
std::optional<ArtifactInfo> read_artifact_json(const std::string & artifact_dir);

enum class Validity { ok, missing, invalid };
// The hit check (§2.8 adoption step 2): artifact.json present, schema 1,
// complete, and every inventory file present at exactly its size.
Validity validate(const std::string & artifact_dir, std::string * why = nullptr);

// Every regular file under `root`, relative to `base`, sorted.
std::vector<InventoryEntry> inventory_of(const std::string & base, const std::string & root);

// ---- Adoption (§2.8) -----------------------------------------------------------

struct Adoption {
    enum class Kind { adopted, missing, invalid, would_block, error };
    Kind kind = Kind::error;
    KeyLock lease;    // adopted: the shared lease
    std::string dir;  // artifacts/<template-id>.<key32>
    Error error;
};

// Takes LOCK_SH|LOCK_NB, validates the inventory, writes `memo` under the lease
// unless this source's memo is present and equal to it, and keeps the lease. A
// failed memo write is an error (the lease is released): the memo is the
// reference that protects the artifact from prune after unload.
Adoption adopt(const Layout & l, const std::string & template_id, const std::string & key32,
               const std::string * memo_id = nullptr, const Memo * memo = nullptr);

// Moves an unusable artifact aside (quarantine/<template-id>.<key32>.<pid>) under
// LOCK_EX|LOCK_NB, i.e. only when no lease is held. `would_block` when leased.
KeyLock::Result quarantine(const Layout & l, const std::string & template_id, const std::string & key32,
                           Error * err = nullptr);

// used/<key32>: touched after every successful load (routine prune's age).
void touch_used(const Layout & l, const std::string & key32);

// ---- The prepare pipeline (§2.7) -------------------------------------------------

struct CompileResult {
    Error error;
    std::string os_build;
    std::string coremlc;
};
// Step 6: compile `mlpackage` and leave the compiled bundle at `mlmodelc_out`
// (Apple: compileModelAtURL, move with a copy fallback; S2b).
using CompileFn = std::function<CompileResult(const std::string & mlpackage, const std::string & mlmodelc_out)>;
using OpenFn = std::function<std::unique_ptr<OpenSource>(const std::string & path, Error & err)>;

struct FsFacts {
    uint64_t free_bytes = 0;
    bool local = false;
    std::string fstype;
};
// statfs of the cache volume. The default accepts local apfs/hfs on Apple
// (where flock and atomic directory rename are relied on) and the usual local
// Linux filesystems for the portable tests and tools.
using FsProbe = std::function<bool(const std::string & dir, FsFacts & out)>;
bool default_fs_probe(const std::string & dir, FsFacts & out);
bool fs_supported(const FsFacts & f);

struct PrepareInputs {
    std::string cache_dir;
    std::shared_ptr<const recipe::Template> tmpl;
    SourceStat expected;  // the stat tuple the engine captured when it loaded the GGUF (path included)
    std::string options = digest::DEFAULT_OPTIONS;
    std::string os_build;  // failure records
    CompileFn compile;
    OpenFn open;           // default: GgufSource::open (needs ggml)
    FsProbe fs_probe;      // default: default_fs_probe
    int64_t now_s = 0;     // 0: the clock (failure records)
};

struct PrepareOutcome {
    enum class Kind { ready, would_block, failed, unsupported };
    Kind kind = Kind::failed;
    bool cache_hit = false;  // adopted an existing artifact (status route: cached)
    bool memo_hit = false;   // hot path: no hashing
    std::string artifact_dir;
    std::string model_path;  // <artifact_dir>/model.mlmodelc
    std::string key32;
    std::string memo_id;
    KeyLock lease;           // ready: the shared lease (hand it to the loaded model)
    Error error;             // failed: class, domain, code
    std::string reason;      // source_changed | no_space | cache_fs | ...
    int64_t digest_ms = 0;
    int64_t write_ms = 0;
    int64_t compile_ms = 0;
    uint64_t generation_peak_bytes = 0;
};

// Steps 1-5 and 7-8, with step 6 through `compile`. Never blocks on a lock.
PrepareOutcome prepare(const PrepareInputs & in, PrepareContext * ctx = nullptr);

// Writes the .mlpackage (Manifest.json, Data/com.apple.CoreML/model.mlmodel,
// Data/com.apple.CoreML/weights/weight.bin) for `src` at `package_dir`
// (qvac-coreml-tool generate, the pipeline's step 5). `observer` sees every
// tensor read (the write-time digest re-check).
Error write_package(const recipe::Template & t, recipe::TensorSource & src, const std::string & package_dir,
                    recipe::EvalStats * stats = nullptr, const recipe::ReadObserver & observer = nullptr);

// ---- Prune (§2.8) --------------------------------------------------------------

// What this library owns: the template ids it embeds (with their current
// template_sha256) and the ids it retired. Unknown ids belong to another addon.
struct Catalog {
    uint32_t generator_version = GENERATOR_VERSION;
    std::vector<std::pair<std::string, std::string>> templates;  // (id, template_sha256 hex)
    std::vector<std::string> retired_ids;

    bool owns(const std::string & id) const;
    bool current(const std::string & id, const std::string & sha256_hex, uint32_t generator_version) const;
};

struct PrunePolicy {
    enum class Mode { routine, scoped, all };
    Mode mode = Mode::routine;
    int64_t grace_s = 30LL * 24 * 3600;  // routine
    std::vector<std::string> dead_paths;  // scoped
    std::optional<int64_t> now_s;         // tests
    std::string tmp_dir;                  // routine $TMPDIR sweep; "" -> $TMPDIR or /tmp

    static PrunePolicy routine(int64_t grace_s = 30LL * 24 * 3600);
    static PrunePolicy scoped(std::vector<std::string> dead_paths);
    static PrunePolicy all();
};

struct PruneReport {
    std::vector<std::string> deleted_artifacts;  // directory names
    std::vector<std::string> busy_artifacts;     // leased (LOCK_EX|LOCK_NB failed): skipped
    std::vector<std::string> deleted_memos;      // file names
    size_t staging = 0;
    size_t tmp = 0;
    size_t failed = 0;
    size_t quarantine = 0;
};

// The three mutually exclusive modes of §2.8. Static: needs no engine
// instance. Never deletes locks/ or the root.
PruneReport prune(const std::string & cache_dir, const PrunePolicy & policy, const Catalog & catalog);

// Whether a memo is live: its file still has the recorded identity, and its
// (template id, template_sha256, generator_version) is current.
bool memo_live(const Memo & m, const Catalog & c);

}  // namespace cache
}  // namespace QVAC_COREML_NS
