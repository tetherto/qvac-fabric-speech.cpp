#pragma once

// Recipe language, interpreter and templates of the portable Core ML generator.
//
// A recipe says, blob by blob in weight.bin order, how each weight of a
// weight-free Core ML template is computed from GGUF tensors. parse() checks
// the grammar and every static rule of the language (known ops, use before `let`, the
// dtype of the top cast, element counts, offsets against the MILBlob writer,
// the total size); evaluate() streams the blobs to a sink and
// evaluate_patches() the spec patches. The op vocabulary grows with its
// consumers; this generator version knows:
//   t f32 f64 f16 add mul div recip clip reshape permute take concat (+ let)
//   slice neg zeros ones lit
//   sub sqrt
//   meta_f meta_s provided
// and `patch` lines of dtype fp16 and fp32, which rewrite inline constants of
// the spec (model.mlmodel) in place, and `patch ... str <len>` lines, which
// rewrite a fixed-length string of the spec (meta_s(key), a GGUF metadata
// string of exactly that length).
//
// Arithmetic is IEEE in the operand dtype with no FMA contraction (the
// interpreter is compiled with -ffp-contract=off). A blob whose expression
// contains provided() is `ulp` (compare_blob()); every other blob of this
// vocabulary is `exact`.

#include "qvac_coreml/slot.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifndef QVAC_COREML_NS
#error "QVAC_COREML_NS must be defined"
#endif

namespace QVAC_COREML_NS {

// QVAC_COREML_GENERATOR_VERSION: covers the interpreter, the MILBlob
// writer and every provided() implementation. Enters every artifact key.
constexpr uint32_t GENERATOR_VERSION = 1;

namespace recipe {

// ---- AST ---------------------------------------------------------------------

// Array dtypes. `weak` is a numeric literal: a scalar that takes the dtype of
// the array it meets (numpy weak scalars).
enum class DType : uint8_t { weak, f32, f64 };
enum class BlobType : uint8_t { fp16, fp32 };

const char * to_string(DType d);
const char * to_string(BlobType b);
size_t blob_type_size(BlobType b);

struct Expr;
using ExprP = std::shared_ptr<Expr>;

struct Expr {
    enum class Kind : uint8_t { number, string, name, list, call };
    Kind kind = Kind::number;
    double number = 0.0;      // number
    std::string text;         // string literal, name, or op of a call
    std::vector<ExprP> args;  // list items or call arguments
    int line = 0;

    // Filled by the static checker for value expressions (not for the shape,
    // axis or index arguments of reshape/permute/take/concat).
    DType dtype = DType::weak;
    std::vector<int64_t> shape;  // numpy order; empty = scalar
    bool ulp = false;            // contains a reduction, transcendental or provided()
};

struct Source {
    std::string name;
    std::vector<uint32_t> types;  // allowed ggml types
    std::vector<int64_t> shape;   // numpy order = GGUF ne reversed
    int line = 0;
};

// GGUF metadata a recipe reads; enters the source digest.
struct Meta {
    std::string key;
    std::string mtype;  // u32|i32|f32|f64|bool|str|arr_u32|arr_f32
    int line = 0;
};

struct Item {
    enum class Kind : uint8_t { let, blob, patch };
    Kind kind = Kind::blob;
    std::string name;     // let
    uint64_t offset = 0;  // blob: MILBlob metadata offset; patch: byte offset in model.mlmodel
    BlobType dtype = BlobType::fp16;
    uint64_t count = 0;  // blob, patch: element count (patch str: byte length)
    bool str = false;    // patch str: `expr` is meta_s(key), written as `count` bytes
    ExprP expr;
    int line = 0;
    bool ulp = false;  // blob tolerance class (see compare_blob)
};

// A weight-independent constant read by lit(id): template.json `literals`
// entry {id, dtype, shape, offset}, the bytes in the template's literals.bin
// (little-endian fp16 or fp32, row-major, `shape` in numpy order). lit()
// yields f32 (fp16 widened exactly).
struct Literal {
    std::string id;
    BlobType dtype = BlobType::fp16;
    std::vector<int64_t> shape;
    uint64_t offset = 0;              // in literals.bin
    const uint8_t * data = nullptr;   // offset applied; owned by the template (or the caller of parse)
};

// template.json `patches` entry: the bytes of model.mlmodel at
// [spec_offset, spec_offset + len) hold the sentinel the extraction wrote, and
// the recipe's patch line at that offset replaces them.
struct PatchSite {
    uint64_t spec_offset = 0;
    uint64_t len = 0;
    std::string sentinel_hex;  // lowercase hex, 2 * len characters
};

// template.json `provided` entry: the shape (numpy order) of the f32 array
// provided(key, ...) yields. The values come from the ProvidedFn registered
// under `key` (register_provided).
struct ProvidedDecl {
    std::string key;
    std::vector<int64_t> shape;
};

// What parse() checks a recipe against besides its own text: the template's
// literals, its provided() declarations and, when known, its spec and patch
// sites. A default context declares no literal, no provided() and no patch
// site.
struct ParseContext {
    std::vector<Literal> literals;     // `data` set: lit() can be evaluated
    std::vector<ProvidedDecl> provided;
    const uint8_t * spec = nullptr;    // model.mlmodel; null: patch bounds and sentinels are not checked
    size_t spec_size = 0;
    std::vector<PatchSite> patches;    // must equal the recipe's patch lines when `spec` is set
};

struct Recipe {
    std::string template_id;
    std::string spec_sha256;  // hex of model.mlmodel
    uint64_t weight_bin_size = 0;
    uint32_t blob_count = 0;
    std::vector<Source> sources;
    std::vector<Meta> meta;
    std::vector<Item> items;  // let, blob and patch lines, in file order
    std::vector<Literal> literals;  // from the ParseContext
    std::vector<ProvidedDecl> provided;  // from the ParseContext

    const Source * find_source(std::string_view name) const;
    const Meta * find_meta(std::string_view key) const;
    const Literal * find_literal(std::string_view id) const;
    const ProvidedDecl * find_provided(std::string_view key) const;
    size_t patch_count() const;
    // The provided() keys the recipe's items use, sorted.
    std::vector<std::string> provided_keys() const;
};

class RecipeError : public std::runtime_error {
public:
    RecipeError(int line, const std::string & msg)
        : std::runtime_error(line > 0 ? "recipe:" + std::to_string(line) + ": " + msg : "recipe: " + msg),
          line(line) {}
    int line;
};

// Grammar and every static check of the language. Throws RecipeError (a deterministic
// recipe-validation failure). Patch lines come in spec offset order, do not
// overlap, hold exactly count x dtype-size bytes and, with `ctx.spec`, lie
// inside the spec on a declared patch site whose bytes equal its sentinel.
Recipe parse(std::string_view text, const ParseContext & ctx = {});

// ggml type <-> name ("f32", "f16", "bf16", "q8_0", ...): the ABI values of
// enum ggml_type, so the parser needs no ggml (gguf_source.cpp asserts them).
std::optional<uint32_t> ggml_type_from_name(std::string_view name);
const char * ggml_type_name_of(uint32_t type);

// gguf_type of a recipe `meta` MTYPE (arrays report GGUF_TYPE_ARRAY and
// their element type in `elem`).
std::optional<uint32_t> gguf_type_of_mtype(std::string_view mtype, uint32_t * elem = nullptr);

// gguf_type values (gguf.h ABI) the generator needs without ggml.
enum GgufType : uint32_t {
    GGUF_U8 = 0, GGUF_I8 = 1, GGUF_U16 = 2, GGUF_I16 = 3, GGUF_U32 = 4, GGUF_I32 = 5, GGUF_F32 = 6,
    GGUF_BOOL = 7, GGUF_STRING = 8, GGUF_ARRAY = 9, GGUF_U64 = 10, GGUF_I64 = 11, GGUF_F64 = 12,
};

// ---- Sources -----------------------------------------------------------------

struct TensorInfo {
    std::string name;
    uint32_t type = 0;  // ggml type
    int n_dims = 1;
    std::array<int64_t, 4> ne{{1, 1, 1, 1}};
    uint64_t nbytes = 0;
};

// Sees every raw payload byte a read delivers, in file order.
using RawTap = std::function<void(const uint8_t * data, size_t n)>;

// What the interpreter, the digest and template matching read. GgufSource
// (src/gguf_source.h) reads a GGUF through one open fd; tests use fakes.
class TensorSource {
public:
    virtual ~TensorSource() = default;
    virtual const TensorInfo * tensor(const std::string & name) const = 0;
    // Raw payload in file order, in chunks of at most 8 MiB. `sink` returns
    // false to abort (reported as a transient error).
    virtual Error read_raw(const std::string & name, const std::function<bool(const uint8_t *, size_t)> & sink) = 0;
    // t(): ggml to_float (ggml_get_type_traits(type)->to_float; F32 copied).
    // `tap`, when set, sees the raw bytes the conversion read.
    virtual Error read_f32(const std::string & name, std::vector<float> & out, const RawTap & tap) = 0;
    // Canonical metadata value, as the source digest hashes it: the gguf type and its little-endian bytes
    // (strings: str(); arrays: u64le(n) || elements). For arrays,
    // `array_elem_type` (when given) receives the element's gguf type. False
    // when absent.
    virtual bool meta_canonical(const std::string & key, uint32_t & gguf_type, std::vector<uint8_t> & bytes,
                                uint32_t * array_elem_type = nullptr) const = 0;
    // The value as JSON text (template.json `match.metadata`); "" when absent.
    virtual std::string meta_json(const std::string & key) const = 0;
};

// ---- provided() ----------------------------------------------------------------

// One provided(key, [deps...]) call: the declared shape and the tensors and
// metadata keys it names as dependencies. Every dependency is a declared
// source or meta line, so the source digest covers what the function reads.
struct ProvidedCall {
    std::string key;
    std::vector<int64_t> shape;        // numpy order, from template.json `provided`
    std::vector<std::string> tensors;  // t("...") dependencies, in call order
    std::vector<std::string> meta;     // meta_f("...") / meta_s("...") dependencies, in call order
    int line = 0;
};

// Computes provided(key, ...) into `out` (row-major, `call.shape` elements).
// `src` reads only the call's dependencies (any other tensor or key is
// absent), through the job's own source. A plain function pointer: it cannot
// capture an engine. A non-ok Error aborts the evaluation with its class.
using ProvidedFn = Error (*)(TensorSource & src, const ProvidedCall & call, std::vector<float> & out);

// The process-wide table of this namespace: an engine registers its
// implementations before it probes a template that uses them (registering a
// key again with the same function is a no-op; another function throws
// std::logic_error). Every implementation is part of the generator, so a
// change to one that alters its output must bump GENERATOR_VERSION.
void register_provided(const std::string & key, ProvidedFn fn);
ProvidedFn find_provided_fn(std::string_view key);

// ---- Evaluation --------------------------------------------------------------

struct BlobOut {
    uint64_t offset = 0;
    BlobType dtype = BlobType::fp16;
    uint64_t count = 0;
    const uint8_t * data = nullptr;  // little-endian fp16 / fp32
    size_t bytes = 0;
    bool ulp = false;
    int line = 0;
};

struct EvalStats {
    uint64_t peak_bytes = 0;  // generation_peak_bytes: live let and temporary buffers
    uint64_t tensors_read = 0;
};

// Called after every completed t() read with the SHA-256 of the raw bytes it
// read, so a generator can prove the bytes it wrote from are the bytes it
// digested. A non-ok result aborts the evaluation.
using ReadObserver = std::function<Error(const std::string & name, const std::array<uint8_t, 32> & raw_sha256)>;

// Evaluates every blob in file order. `let` bindings are evaluated on first use
// and freed after their last use. Errors: a non-finite fp16/fp32 output, NaN in
// clip() or a source that no longer matches its `source` line are
// deterministic; source read failures carry the source's class.
Error evaluate(const Recipe & r, TensorSource & src, const std::function<Error(const BlobOut &)> & sink,
               EvalStats * stats = nullptr, const ReadObserver & observer = nullptr);

// One evaluated patch line: `bytes` little-endian fp16 / fp32 values (or, for
// a str line, the string's bytes) to write at `offset` in model.mlmodel.
struct PatchOut {
    uint64_t offset = 0;
    BlobType dtype = BlobType::fp16;
    bool str = false;
    uint64_t count = 0;
    const uint8_t * data = nullptr;
    size_t bytes = 0;
    int line = 0;
};

// Evaluates every patch line in spec offset order, with the errors of evaluate().
// `stats` gets the patch expressions' interpreter peak.
Error evaluate_patches(const Recipe & r, TensorSource & src, const std::function<Error(const PatchOut &)> & sink,
                       const ReadObserver & observer = nullptr, EvalStats * stats = nullptr);

// The spec with every patch applied (`spec` is the template's model.mlmodel).
// `stats` gets the pass's peak: the spec copy plus the patch expressions'.
Error patched_spec(const Recipe & r, TensorSource & src, const uint8_t * spec, size_t spec_size,
                   std::vector<uint8_t> & out, const ReadObserver & observer = nullptr, EvalStats * stats = nullptr);

// ---- Number formats and tolerance ---------------------------------------------

// IEEE binary16, round to nearest even (numpy's astype(float16)); overflow
// gives infinity, NaN stays NaN.
uint16_t f32_to_f16(float x);
float f16_to_f32(uint16_t h);

// Ordered-integer distance in the blob dtype; +0 and -0 are 0 apart. Returns -1
// when either value is NaN or infinite.
int64_t ulp_distance_f16(uint16_t a, uint16_t b);
int64_t ulp_distance_f32(float a, float b);

struct BlobCompare {
    uint64_t offset = 0;
    bool ulp = false;        // tolerance class
    uint64_t differing = 0;  // elements whose bits differ
    int64_t max_ulp = 0;     // -1: a non-finite value was involved
    bool pass = false;
};

// Tolerance: `exact` blobs must be bit-equal; `ulp` blobs pass per element when
// ulp_distance <= 1 or |delta| <= 2^-10 * max|reference blob|. NaN or Inf in
// either value fails.
BlobCompare compare_blob(BlobType dt, bool ulp, const uint8_t * ours, const uint8_t * ref, uint64_t count);

struct BlobCheck {
    BlobCompare cmp;     // cmp.offset is the recipe's metadata offset
    std::string layout;  // non-empty: offset, size or dtype differs from the recipe
};

struct WeightBinCheck {
    std::vector<std::string> errors;  // file-level: index, total size, blob count, non-payload bytes
    std::vector<BlobCheck> blobs;     // one per recipe blob, in recipe order
    bool pass = false;
};

// `qvac-coreml-tool verify`: `ours` against the exporter's
// weight.bin `ref`. Both files must have the recipe's blob count, total size,
// offsets, sizes and MILBlob dtypes; every byte outside the blob payloads
// (header, metadata, padding) must be equal; each payload passes by
// compare_blob (exact: byte-equal; ulp: the tolerance rule above).
WeightBinCheck verify_weight_bin(const Recipe & r, const uint8_t * ours, size_t ours_size, const uint8_t * ref,
                                 size_t ref_size);

// ---- Templates -----------------------------------------------------------------

// One embedded file (from the build's embedded-template table) or one file read from a
// template directory (Linux tests, qvac-coreml-tool, the dev override).
struct TemplateFile {
    std::string name;
    const uint8_t * data = nullptr;
    size_t size = 0;
    std::array<uint8_t, 32> sha256{};
};

// template.json `names`: how the names a recipe reads (its `source` lines
// and t() calls) map to the GGUF's tensor names, for a GGUF that stores its
// tensors under generated names and lists the model's own names in metadata.
// Built from the GGUF's metadata, the way the engine's loader resolves them:
//   - `map` [from, to]: two string arrays of equal length; from[i] reads
//     tensor to[i] (a later duplicate wins);
//   - `aliases` [from, to], optional: from[i] reads what to[i] reads, unless
//     from[i] already resolves (both arrays absent: no aliases);
//   - `fallbacks` [[name, other], ...]: `name` reads what `other` reads when
//     `name` does not resolve otherwise (names older converters did not write).
// A name the table does not resolve is absent. Declared by the template, so
// the same table applies in the engine, qvac-coreml-tool and the extraction
// gates, and a change to it changes template_sha256.
struct NameSpec {
    std::string map_from, map_to;
    std::string aliases_from, aliases_to;
    std::vector<std::pair<std::string, std::string>> fallbacks;
    bool empty() const { return map_from.empty(); }
};

struct TemplateInfo {
    std::string id;
    std::string family;
    std::string stage;
    std::string export_kind;
    std::string default_compute_units;
    std::vector<std::string> functions;
    uint64_t weight_bin_size = 0;
    uint64_t generation_peak_bytes = 0;
    bool stateful = false;
    bool shareable = true;
    std::string match_architecture;
    std::string match_metadata_json;  // the `match.metadata` object as JSON text
    NameSpec names;                   // template.json `names`; empty: the recipe uses the GGUF's own names
};

class Template {
public:
    // The required files; literals.bin is optional. reference-gguf.json is
    // test-only and never part of a template (or of template_sha256).
    static const std::vector<std::string> & required_files();

    // Reads <dir>/{model.mlmodel, Manifest.json, recipe.txt, template.json,
    // [literals.bin]}. Throws RecipeError / std::runtime_error.
    static std::shared_ptr<const Template> from_dir(const std::string & dir);
    // Files whose bytes outlive the template (static embedded data). Each
    // file's sha256 is computed here.
    static std::shared_ptr<const Template> from_files(std::vector<TemplateFile> files);
    // Embedded files whose sha256 the build computed (the embedded-template
    // table): nothing is hashed, so the large files (model.mlmodel,
    // Manifest.json, literals.bin) are not paged in until a package is written.
    static std::shared_ptr<const Template> from_embedded(std::vector<TemplateFile> files);

    const std::string & id() const { return info_.id; }
    const TemplateInfo & info() const { return info_; }
    const Recipe & recipe() const { return recipe_; }
    // template_sha256: SHA-256 over every template file's name and digest.
    const std::array<uint8_t, 32> & sha256() const { return sha256_; }
    std::string sha256_hex() const;
    const TemplateFile * file(std::string_view name) const;
    const std::vector<TemplateFile> & files() const { return files_; }

private:
    Template() = default;
    void init(bool hash_files);
    std::vector<TemplateFile> files_;
    std::vector<std::shared_ptr<std::vector<uint8_t>>> owned_;
    TemplateInfo info_;
    Recipe recipe_;
    std::array<uint8_t, 32> sha256_{};
};

// A TensorSource that reads `base` under a NameSpec: tensor(), read_raw() and
// read_f32() take the recipe's names; metadata reads pass through. A name the
// table does not resolve is absent. When the GGUF's name arrays are absent or
// malformed, error() says why and no tensor resolves.
class RenamedSource final : public TensorSource {
public:
    RenamedSource(TensorSource & base, const NameSpec & spec);

    const std::string & error() const { return error_; }
    // The GGUF's own name for `name` ("" when it does not resolve).
    std::string stored_name(const std::string & name) const;

    const TensorInfo * tensor(const std::string & name) const override;
    Error read_raw(const std::string & name, const std::function<bool(const uint8_t *, size_t)> & sink) override;
    Error read_f32(const std::string & name, std::vector<float> & out, const RawTap & tap) override;
    bool meta_canonical(const std::string & key, uint32_t & gguf_type, std::vector<uint8_t> & bytes,
                        uint32_t * array_elem_type = nullptr) const override;
    std::string meta_json(const std::string & key) const override;

private:
    TensorSource & base_;
    std::map<std::string, std::string> stored_;  // recipe name -> GGUF name
    std::map<std::string, TensorInfo> infos_;    // recipe name -> the GGUF tensor's info under the recipe name
    std::string error_;
};

struct MatchResult {
    bool ok = false;
    std::string reason;  // no_template detail when !ok
};

// template.json `match`: general.architecture, the required metadata
// values, every recipe `source` (name, allowed type, shape), every `meta`
// declaration (present, declared type) and, for each provided() key the
// recipe uses, a registered implementation.
MatchResult match(const Template & t, const TensorSource & src);

}  // namespace recipe
}  // namespace QVAC_COREML_NS
