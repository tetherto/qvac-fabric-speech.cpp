#pragma once

// Recipe language, interpreter and templates of the portable Core ML generator
// (PLAN §2.6, Appendix A, Appendix B.1).
//
// A recipe says, blob by blob in weight.bin order, how each weight of a
// weight-free Core ML template is computed from GGUF tensors. parse() checks
// the grammar and every static rule of A.1 (known ops, use before `let`, the
// dtype of the top cast, element counts, offsets against the MILBlob writer,
// the total size); evaluate() streams the blobs to a sink. The op vocabulary
// grows with its consumers; this generator version knows Parler's DAC subset:
//   t f32 f64 f16 add mul div recip clip reshape permute take concat (+ let)
// `patch`, provided(), meta_f/meta_s and lit() arrive with later PRs (§5.1);
// the parser rejects them.
//
// Arithmetic is IEEE in the operand dtype with no FMA contraction (the
// interpreter is compiled with -ffp-contract=off). No op of this subset is a
// reduction, a transcendental or provided(), so every blob is `exact` (A.3);
// compare_blob() already implements the `ulp` rule later ops flag into.

#include "qvac_coreml/slot.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifndef QVAC_COREML_NS
#error "QVAC_COREML_NS must be defined"
#endif

namespace QVAC_COREML_NS {

// QVAC_COREML_GENERATOR_VERSION (§2.5): covers the interpreter, the MILBlob
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

// GGUF metadata a recipe reads; enters the source digest (B.4).
struct Meta {
    std::string key;
    std::string mtype;  // u32|i32|f32|f64|bool|str|arr_u32|arr_f32
    int line = 0;
};

struct Item {
    enum class Kind : uint8_t { let, blob };
    Kind kind = Kind::blob;
    std::string name;     // let
    uint64_t offset = 0;  // blob: MILBlob metadata offset
    BlobType dtype = BlobType::fp16;
    uint64_t count = 0;  // blob: element count
    ExprP expr;
    int line = 0;
    bool ulp = false;  // blob tolerance class (A.3)
};

struct Recipe {
    std::string template_id;
    std::string spec_sha256;  // hex of model.mlmodel
    uint64_t weight_bin_size = 0;
    uint32_t blob_count = 0;
    std::vector<Source> sources;
    std::vector<Meta> meta;
    std::vector<Item> items;  // let and blob lines, in file order

    const Source * find_source(std::string_view name) const;
};

class RecipeError : public std::runtime_error {
public:
    RecipeError(int line, const std::string & msg)
        : std::runtime_error(line > 0 ? "recipe:" + std::to_string(line) + ": " + msg : "recipe: " + msg),
          line(line) {}
    int line;
};

// Grammar and every static check of A.1. Throws RecipeError (a deterministic
// recipe-validation failure).
Recipe parse(std::string_view text);

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
    // B.4 canonical metadata value: the gguf type and its little-endian bytes
    // (strings: str(); arrays: u64le(n) || elements). For arrays,
    // `array_elem_type` (when given) receives the element's gguf type. False
    // when absent.
    virtual bool meta_canonical(const std::string & key, uint32_t & gguf_type, std::vector<uint8_t> & bytes,
                                uint32_t * array_elem_type = nullptr) const = 0;
    // The value as JSON text (template.json `match.metadata`); "" when absent.
    virtual std::string meta_json(const std::string & key) const = 0;
};

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
// digested (§2.7 step 5). A non-ok result aborts the evaluation.
using ReadObserver = std::function<Error(const std::string & name, const std::array<uint8_t, 32> & raw_sha256)>;

// Evaluates every blob in file order. `let` bindings are evaluated on first use
// and freed after their last use. Errors: a non-finite fp16/fp32 output, NaN in
// clip() or a source that no longer matches its `source` line are
// deterministic; source read failures carry the source's class.
Error evaluate(const Recipe & r, TensorSource & src, const std::function<Error(const BlobOut &)> & sink,
               EvalStats * stats = nullptr, const ReadObserver & observer = nullptr);

// ---- Number formats and tolerance (A.3) ---------------------------------------

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

// A.3: `exact` blobs must be bit-equal; `ulp` blobs pass per element when
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

// `qvac-coreml-tool verify` (A.3, §2.4 G1-G3): `ours` against the exporter's
// weight.bin `ref`. Both files must have the recipe's blob count, total size,
// offsets, sizes and MILBlob dtypes; every byte outside the blob payloads
// (header, metadata, padding) must be equal; each payload passes by
// compare_blob (exact: byte-equal; ulp: the A.3 rule).
WeightBinCheck verify_weight_bin(const Recipe & r, const uint8_t * ours, size_t ours_size, const uint8_t * ref,
                                 size_t ref_size);

// ---- Templates (Appendix B.1) ------------------------------------------------

// One embedded file (S2b's assembler-stub table) or one file read from a
// template directory (Linux tests, qvac-coreml-tool, the dev override).
struct TemplateFile {
    std::string name;
    const uint8_t * data = nullptr;
    size_t size = 0;
    std::array<uint8_t, 32> sha256{};
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

    const std::string & id() const { return info_.id; }
    const TemplateInfo & info() const { return info_; }
    const Recipe & recipe() const { return recipe_; }
    // template_sha256 of B.4.
    const std::array<uint8_t, 32> & sha256() const { return sha256_; }
    std::string sha256_hex() const;
    const TemplateFile * file(std::string_view name) const;
    const std::vector<TemplateFile> & files() const { return files_; }

private:
    Template() = default;
    void init();
    std::vector<TemplateFile> files_;
    std::vector<std::shared_ptr<std::vector<uint8_t>>> owned_;
    TemplateInfo info_;
    Recipe recipe_;
    std::array<uint8_t, 32> sha256_{};
};

struct MatchResult {
    bool ok = false;
    std::string reason;  // no_template detail when !ok
};

// template.json `match` (§2.5): general.architecture, the required metadata
// values, every recipe `source` (name, allowed type, shape) and every `meta`
// declaration (present, declared type).
MatchResult match(const Template & t, const TensorSource & src);

}  // namespace recipe
}  // namespace QVAC_COREML_NS
