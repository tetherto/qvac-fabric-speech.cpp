// test-coreml-recipe (PLAN §2.16, §2.6, Appendix A): grammar and static
// checks, each op of generator version 1 on hand-computed vectors, dtype
// promotion, tolerance classes and ULP distance, templates and matching, and
// (with ggml) GgufSource. Portable: Linux and macOS.

#include "generator_test_util.h"

#include "digest.h"

#include <cmath>

#if defined(QVAC_COREML_HAVE_GGML)
#include "ggml.h"
#include "gguf.h"
#endif

using namespace gtest;
using recipe::BlobType;
using recipe::DType;

std::string mutate_dir;  // case_template's rejects() mutates this template dir

namespace {

// ---- helpers ---------------------------------------------------------------------

std::string hdr(uint64_t wbs, uint32_t blobs, const std::string & id = "t") {
    return "mlrecipe 1\ntemplate " + id + "\nspec_sha256 " + zeros64() + "\nweight_bin_size " + std::to_string(wbs) +
           "\nblobs " + std::to_string(blobs) + "\n";
}

// Parses, expecting a RecipeError whose message contains `needle`.
void expect_error(const std::string & text, const std::string & needle, int line = -1) {
    try {
        recipe::parse(text);
        CHECK_MSG(false, "parsed, expected an error containing '" + needle + "'");
    } catch (const recipe::RecipeError & e) {
        CHECK_MSG(std::string(e.what()).find(needle) != std::string::npos,
                  std::string("error '") + e.what() + "' lacks '" + needle + "'");
        if (line >= 0) CHECK_MSG(e.line == line, std::string("line of '") + e.what() + "'");
    }
}

const std::string kSrcA = "source \"a\" f32,f16 [2,3]\n";

// Evaluates a recipe over `src`; returns blob payloads.
std::vector<std::vector<uint8_t>> run(const std::string & text, recipe::TensorSource & src, Error * err = nullptr,
                                      recipe::EvalStats * stats = nullptr) {
    std::vector<std::vector<uint8_t>> out;
    recipe::Recipe r;
    try {
        r = recipe::parse(text);
    } catch (const std::exception & e) {
        CHECK_MSG(false, e.what());
        if (err) *err = Error::deterministic(e.what());
        return out;
    }
    Error e = recipe::evaluate(
        r, src,
        [&](const recipe::BlobOut & b) {
            out.emplace_back(b.data, b.data + b.bytes);
            return Error{};
        },
        stats);
    if (err) *err = e;
    CHECK_MSG(err || e.ok(), e.message);
    return out;
}

std::vector<float> f32s(const std::vector<uint8_t> & b) {
    std::vector<float> v(b.size() / 4);
    std::memcpy(v.data(), b.data(), b.size());
    return v;
}
std::vector<uint16_t> f16s(const std::vector<uint8_t> & b) {
    std::vector<uint16_t> v(b.size() / 2);
    std::memcpy(v.data(), b.data(), b.size());
    return v;
}
uint32_t bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}
float from_bits(uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// ---- grammar ---------------------------------------------------------------------------

void case_grammar_valid() {
    // Comments, blank lines and continuation lines inside ( and [.
    const std::string text = hdr(192, 1) + kSrcA +
                             "# a comment\n"
                             "\n"
                             "let x = reshape(t(\"a\"),   # comment inside an open paren\n"
                             "                [3,\n"
                             "                 2])\n"
                             "blob 64 fp16 6 f16(\n"
                             "    x)\n"
                             "end\n";
    try {
        const recipe::Recipe r = recipe::parse(text);
        CHECK(r.template_id == "t");
        CHECK(r.weight_bin_size == 192);
        CHECK(r.blob_count == 1);
        CHECK(r.sources.size() == 1 && r.sources[0].types.size() == 2);
        CHECK(r.items.size() == 2);
        CHECK(r.items[0].kind == recipe::Item::Kind::let && r.items[0].line == 9);
        CHECK(r.items[1].kind == recipe::Item::Kind::blob && r.items[1].line == 12);
        CHECK(r.items[1].expr->shape == std::vector<int64_t>({3, 2}));
        CHECK(!r.items[1].ulp);  // no op of generator version 1 is ulp
    } catch (const std::exception & e) {
        CHECK_MSG(false, e.what());
    }
    // meta lines are accepted (they enter the digest) after sources.
    try {
        recipe::parse(hdr(192, 1) + kSrcA + "meta \"x.n\" u32\nblob 64 fp16 6 f16(t(\"a\"))\nend\n");
    } catch (const std::exception & e) {
        CHECK_MSG(false, e.what());
    }
}

void case_grammar_errors() {
    const std::string blob = "blob 64 fp16 6 f16(t(\"a\"))\n";
    expect_error("template t\n", "mlrecipe");
    expect_error("mlrecipe 2\n", "version");
    expect_error(hdr(192, 1, "Bad.Id") + kSrcA + blob + "end\n", "template id");
    expect_error("mlrecipe 1\ntemplate t\nspec_sha256 abc\nweight_bin_size 128\nblobs 1\n" + kSrcA + blob + "end\n",
                 "spec_sha256");
    expect_error(hdr(192, 1) + kSrcA + blob, "missing 'end'");
    expect_error(hdr(192, 1) + kSrcA + blob + "end\nblob 1\n", "after 'end'");
    expect_error(hdr(192, 1) + kSrcA + "blob 64 fp16 6 f16(t(\"a\")\nend\n", "unclosed");
    expect_error(hdr(192, 1) + kSrcA + "blob 64 fp16 6 f16(t(\"a\")))\nend\n", "unbalanced");
    expect_error(hdr(192, 1) + "source \"a f32 [2]\n" + blob + "end\n", "unterminated string");
    expect_error(hdr(192, 1) + kSrcA + "bogus 1\n" + blob + "end\n", "unknown line");
    // Unknown and not-yet-available ops.
    expect_error(hdr(192, 1) + kSrcA + "blob 64 fp16 6 f16(frob(t(\"a\")))\nend\n", "unknown op frob");
    expect_error(hdr(192, 1) + kSrcA + "blob 64 fp16 6 f16(tile(t(\"a\"), 0, 1))\nend\n", "arrives with A1");
    expect_error(hdr(192, 1) + kSrcA + "blob 64 fp16 6 f16(provided(\"k\", [t(\"a\")]))\nend\n", "arrives with S5");
    expect_error(hdr(192, 1) + kSrcA + "patch 10 fp16 1 1.0\n" + blob + "end\n", "patch lines are not");
    // Bindings.
    expect_error(hdr(192, 1) + kSrcA + "blob 64 fp16 6 f16(x)\nlet x = t(\"a\")\nend\n", "used before its let", 7);
    expect_error(hdr(192, 1) + kSrcA + "let x = t(\"a\")\nlet x = t(\"a\")\n" + blob + "end\n", "already defined");
    expect_error(hdr(192, 1) + kSrcA + "let add = t(\"a\")\n" + blob + "end\n", "op name");
    expect_error(hdr(192, 1) + kSrcA + "let y = y\n" + blob + "end\n", "used before its let");
    // Sources.
    expect_error(hdr(192, 1) + kSrcA + "blob 64 fp16 6 f16(t(\"b\"))\nend\n", "no source line");
    expect_error(hdr(192, 1) + kSrcA + kSrcA + blob + "end\n", "duplicate source");
    expect_error(hdr(192, 1) + "source \"a\" f99 [2,3]\n" + blob + "end\n", "unknown ggml type");
    expect_error(hdr(192, 1) + "source \"a\" f32 [1,1,1,1,1]\n" + blob + "end\n", "at most 4 dims");
    expect_error(hdr(192, 1) + "let x = 1\n" + kSrcA + blob + "end\n", "source lines come before");
    expect_error(hdr(192, 1) + kSrcA + "meta \"k\" u99\n" + blob + "end\n", "unknown meta type");
    // Casts and dtypes.
    expect_error(hdr(192, 1) + kSrcA + "blob 64 fp16 6 f16(f64(t(\"a\")))\nend\n", "f16() takes f32, not f64");
    expect_error(hdr(192, 1) + kSrcA + "blob 64 fp16 6 t(\"a\")\nend\n", "must end in f16");
    expect_error(hdr(192, 1) + kSrcA + "blob 64 fp32 6 f16(t(\"a\"))\nend\n", "must end in f32");
    expect_error(hdr(192, 1) + kSrcA + "blob 64 fp16 6 f16(f32(f16(t(\"a\"))))\nend\n", "only the top cast");
    expect_error(hdr(192, 1) + kSrcA + "blob 64 bf16 6 f16(t(\"a\"))\nend\n", "fp16 or fp32");
    // Counts, offsets, sizes.
    expect_error(hdr(192, 1) + kSrcA + "blob 64 fp16 7 f16(t(\"a\"))\nend\n", "declares 7 elements");
    expect_error(hdr(192, 1) + kSrcA + "blob 128 fp16 6 f16(t(\"a\"))\nend\n", "running offset 64");
    expect_error(hdr(256, 1) + kSrcA + blob + "end\n", "weight_bin_size 256");
    expect_error(hdr(192, 2) + kSrcA + blob + "end\n", "declares 2 blobs");
    // Numbers.
    expect_error(hdr(192, 1) + kSrcA + "blob 64 fp16 6 f16(mul(t(\"a\"), 0x10))\nend\n", "malformed number");
    expect_error(hdr(192, 1) + kSrcA + "blob 64 fp16 6 f16(mul(t(\"a\"), 1e))\nend\n", "exponent");
    expect_error(hdr(192, 1) + kSrcA + "blob 64 fp16 6 f16(mul(t(\"a\"), 1e999))\nend\n", "out of range");
    // Op arguments.
    expect_error(hdr(192, 1) + kSrcA + "blob 64 fp16 6 f16(reshape(t(\"a\"), [-1,-1]))\nend\n", "one -1");
    expect_error(hdr(192, 1) + kSrcA + "blob 64 fp16 6 f16(reshape(t(\"a\"), [4,2]))\nend\n", "element count");
    expect_error(hdr(192, 1) + kSrcA + "blob 64 fp16 6 f16(permute(t(\"a\"), [0,0]))\nend\n", "permutation");
    expect_error(hdr(192, 1) + kSrcA + "blob 64 fp16 6 f16(take(t(\"a\"), 1, [3]))\nend\n", "out of range");
    expect_error(hdr(192, 1) + kSrcA + "blob 64 fp16 6 f16(take(t(\"a\"), 2, [0]))\nend\n", "axis out of range");
    expect_error(hdr(192, 1) + kSrcA + "blob 64 fp16 12 f16(concat(1, t(\"a\"), reshape(t(\"a\"), [3,2])))\nend\n",
                 "differ off the axis");
    expect_error(hdr(192, 1) + kSrcA + "blob 64 fp16 6 f16(add(t(\"a\"), reshape(t(\"a\"), [3,2])))\nend\n",
                 "do not broadcast");
    expect_error(hdr(192, 1) + kSrcA + "blob 64 fp16 6 f16(clip(t(\"a\"), 2, 1))\nend\n", "lower bound above");
    expect_error(hdr(192, 1) + kSrcA + "blob 64 fp16 6 f16(clip(t(\"a\"), t(\"a\"), 1))\nend\n", "literals");
    expect_error(hdr(192, 1) + kSrcA + "blob 64 fp16 6 f16(add(t(\"a\")))\nend\n", "takes 2 argument");
    expect_error(hdr(192, 1) + kSrcA + "blob 64 fp16 6 f16(reshape(1.0, [1]))\nend\n", "needs an array");
    expect_error(hdr(192, 1) + kSrcA + "blob 64 fp16 6 f16(1.0)\nend\n", "f16() takes f32, not weak");
}

// ---- ops on hand-computed vectors ---------------------------------------------------------

MemSource src_a(const std::vector<float> & a, const std::vector<int64_t> & shape = {2, 3}, uint32_t type = T_F32) {
    return MemSource({TT{"a", type, shape, a}}, {});
}

void case_op_t_and_casts() {
    MemSource s = src_a({1.0f, 0.5f, -2.0f, 65504.0f, 1.0f / 3.0f, -0.0f});
    auto out = run(recipe_text("t", zeros64(), kSrcA,
                               {blob("fp16", 6, "f16(t(\"a\"))"), blob("fp32", 6, "f32(t(\"a\"))"),
                                blob("fp32", 6, "f32(f64(t(\"a\")))")}),
                   s);
    CHECK(out.size() == 3);
    if (out.size() != 3) return;
    const auto h = f16s(out[0]);
    CHECK(h == std::vector<uint16_t>({0x3c00, 0x3800, 0xc000, 0x7bff, 0x3555, 0x8000}));
    const auto f = f32s(out[1]);
    CHECK(f.size() == 6 && f[4] == 1.0f / 3.0f && bits(f[5]) == 0x80000000u);
    CHECK(out[2] == out[1]);  // f64 round trip is exact
    // f16 source: t() is the exact widening.
    MemSource s16 = src_a({1.0f, 0.5f, -2.0f, 65504.0f, 0.333251953125f, 0.0f}, {2, 3}, T_F16);
    auto o16 = run(recipe_text("t", zeros64(), kSrcA, {blob("fp32", 6, "f32(t(\"a\"))")}), s16);
    CHECK(!o16.empty() && f32s(o16[0])[4] == 0.333251953125f);
}

void case_op_arith_broadcast() {
    MemSource s({TT{"a", T_F32, {2, 3}, {1, 2, 3, 4, 5, 6}}, TT{"b", T_F32, {3}, {10, 20, 40}},
                 TT{"c", T_F32, {2, 1}, {2, 4}}},
                {});
    const std::string d = "source \"a\" f32 [2,3]\nsource \"b\" f32 [3]\nsource \"c\" f32 [2,1]\n";
    auto out = run(recipe_text("t", zeros64(), d,
                               {blob("fp32", 6, "f32(add(t(\"a\"), t(\"b\")))"),
                                blob("fp32", 6, "f32(mul(t(\"a\"), t(\"c\")))"),
                                blob("fp32", 6, "f32(div(t(\"b\"), t(\"c\")))"),
                                blob("fp32", 6, "f32(add(t(\"c\"), t(\"b\")))"),
                                blob("fp32", 6, "f32(mul(2, t(\"a\")))"), blob("fp32", 1, "f32(add(1, 2))")}),
                   s);
    CHECK(out.size() == 6);
    if (out.size() != 6) return;
    CHECK(f32s(out[0]) == std::vector<float>({11, 22, 43, 14, 25, 46}));
    CHECK(f32s(out[1]) == std::vector<float>({2, 4, 6, 16, 20, 24}));
    CHECK(f32s(out[2]) == std::vector<float>({5, 10, 20, 2.5f, 5, 10}));
    CHECK(f32s(out[3]) == std::vector<float>({12, 22, 42, 14, 24, 44}));
    CHECK(f32s(out[4]) == std::vector<float>({2, 4, 6, 8, 10, 12}));
    CHECK(f32s(out[5]) == std::vector<float>({3}));
}

void case_dtype_promotion() {
    // A weak literal takes the array's dtype: div(f32, pi) divides in f32 by
    // float(pi); div(f64(x), pi) divides in f64. Find values where the two
    // differ, so the test proves which one ran.
    std::vector<float> xs;
    for (int i = 1; xs.size() < 6 && i < 100000; ++i) {
        const float x = 1.0f + static_cast<float>(i) * 0.001953125f;
        const float in_f32 = x / static_cast<float>(3.141592653589793);
        const float in_f64 = static_cast<float>(static_cast<double>(x) / 3.141592653589793);
        if (bits(in_f32) != bits(in_f64)) xs.push_back(x);
    }
    CHECK(xs.size() == 6);
    MemSource s = src_a(xs);
    auto out = run(recipe_text("t", zeros64(), kSrcA,
                               {blob("fp32", 6, "f32(div(t(\"a\"), 3.141592653589793))"),
                                blob("fp32", 6, "f32(div(f64(t(\"a\")), 3.141592653589793))"),
                                blob("fp32", 6, "f32(add(t(\"a\"), f64(t(\"a\"))))")}),
                   s);
    if (out.size() != 3) return;
    const auto p32 = f32s(out[0]), p64 = f32s(out[1]), mix = f32s(out[2]);
    for (size_t i = 0; i < xs.size(); ++i) {
        CHECK(bits(p32[i]) == bits(xs[i] / static_cast<float>(3.141592653589793)));
        CHECK(bits(p64[i]) == bits(static_cast<float>(static_cast<double>(xs[i]) / 3.141592653589793)));
        CHECK(bits(p32[i]) != bits(p64[i]));
        CHECK(mix[i] == static_cast<float>(static_cast<double>(xs[i]) * 2.0));
    }
    // The static checker's dtypes.
    const recipe::Recipe r = recipe::parse(recipe_text(
        "t", zeros64(), kSrcA,
        {let_("p = div(t(\"a\"), 2)"), let_("q = add(t(\"a\"), f64(t(\"a\")))"), let_("w = add(1, 2)"),
         blob("fp32", 6, "f32(p)")}));
    CHECK(r.items[0].expr->dtype == DType::f32);
    CHECK(r.items[1].expr->dtype == DType::f64);
    CHECK(r.items[2].expr->dtype == DType::weak);
}

void case_op_recip_clip() {
    MemSource s = src_a({2.0f, -4.0f, 0.5f, 100.0f, -100.0f, 3.0f});
    auto out = run(recipe_text("t", zeros64(), kSrcA,
                               {blob("fp32", 6, "f32(recip(t(\"a\")))"),
                                blob("fp32", 6, "f32(clip(t(\"a\"), -10, 10.5))"),
                                blob("fp32", 6, "f32(clip(recip(f64(t(\"a\"))), -0.3, 0.3))")}),
                   s);
    if (out.size() != 3) return;
    CHECK(f32s(out[0]) == std::vector<float>({0.5f, -0.25f, 2.0f, 0.01f, -0.01f, 1.0f / 3.0f}));
    CHECK(f32s(out[1]) == std::vector<float>({2.0f, -4.0f, 0.5f, 10.5f, -10.0f, 3.0f}));
    CHECK(f32s(out[2]) ==
          std::vector<float>({0.3f, -0.25f, 0.3f, 0.01f, -0.01f, 0.3f}));
    // NaN in clip() is a deterministic failure.
    MemSource sn = src_a({1, 2, std::nanf(""), 4, 5, 6});
    Error e;
    run(recipe_text("t", zeros64(), kSrcA, {blob("fp32", 6, "f32(clip(t(\"a\"), 0, 1))")}), sn, &e);
    CHECK(!e.ok() && e.cls == ErrorClass::deterministic && e.message.find("NaN") != std::string::npos);
    // Non-finite output: deterministic (fp16 overflow and fp32 inf).
    MemSource big = src_a({70000, 0, 0, 0, 0, 0});
    run(recipe_text("t", zeros64(), kSrcA, {blob("fp16", 6, "f16(t(\"a\"))")}), big, &e);
    CHECK(!e.ok() && e.cls == ErrorClass::deterministic && e.message.find("non-finite") != std::string::npos);
    MemSource zero = src_a({0, 1, 1, 1, 1, 1});
    run(recipe_text("t", zeros64(), kSrcA, {blob("fp32", 6, "f32(recip(t(\"a\")))")}), zero, &e);
    CHECK(!e.ok() && e.cls == ErrorClass::deterministic);
}

void case_op_layout() {
    // a = arange(24).reshape(2,3,4)
    std::vector<float> v(24);
    for (int i = 0; i < 24; ++i) v[i] = static_cast<float>(i);
    MemSource s({TT{"a", T_F32, {2, 3, 4}, v}}, {});
    const std::string d = "source \"a\" f32 [2,3,4]\n";
    auto out = run(recipe_text("t", zeros64(), d,
                               {blob("fp32", 24, "f32(reshape(t(\"a\"), [4,-1]))"),
                                blob("fp32", 24, "f32(permute(t(\"a\"), [2,0,1]))"),
                                blob("fp32", 18, "f32(take(t(\"a\"), 2, [3,-1,0]))"),
                                blob("fp32", 48, "f32(concat(1, t(\"a\"), t(\"a\")))"),
                                blob("fp32", 16, "f32(take(t(\"a\"), 1, [2,-1]))")}),
                   s);
    if (out.size() != 5) return;
    CHECK(f32s(out[0]) == v);
    // permute [2,0,1]: out[k,i,j] = a[i,j,k]
    std::vector<float> p;
    for (int k = 0; k < 4; ++k)
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 3; ++j) p.push_back(static_cast<float>(i * 12 + j * 4 + k));
    CHECK(f32s(out[1]) == p);
    // take axis 2 [3,-1,0]: index -1 yields zeros
    std::vector<float> tk;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j) {
            tk.push_back(static_cast<float>(i * 12 + j * 4 + 3));
            tk.push_back(0.0f);
            tk.push_back(static_cast<float>(i * 12 + j * 4));
        }
    CHECK(f32s(out[2]) == tk);
    // concat axis 1
    std::vector<float> c;
    for (int i = 0; i < 2; ++i)
        for (int rep = 0; rep < 2; ++rep)
            for (int j = 0; j < 12; ++j) c.push_back(static_cast<float>(i * 12 + j));
    CHECK(f32s(out[3]) == c);
    std::vector<float> t1;
    for (int i = 0; i < 2; ++i) {
        for (int k = 0; k < 4; ++k) t1.push_back(static_cast<float>(i * 12 + 8 + k));
        for (int k = 0; k < 4; ++k) t1.push_back(0.0f);
    }
    CHECK(f32s(out[4]) == t1);
}

void case_parler_examples() {
    // A.4 phase kernel on a small layer: w [IC, OC, 2s]; out[r*OC+o, i, 0] =
    // w[i,o,r+s], out[r*OC+o, i, 1] = w[i,o,r].
    const int IC = 3, OC = 2, S = 2;
    std::vector<float> w(IC * OC * 2 * S);
    for (size_t i = 0; i < w.size(); ++i) w[i] = 0.25f * static_cast<float>(i) - 3.0f;
    std::vector<float> alpha = {0.5f, 1.25f, 2.0f};
    MemSource s({TT{"w", T_F32, {IC, OC, 2 * S}, w}, TT{"alpha", T_F32, {IC}, alpha}}, {});
    const std::string d = "source \"w\" f32 [3,2,4]\nsource \"alpha\" f32 [3]\n";
    auto out = run(recipe_text("t", zeros64(), d,
                               {let_("w0 = t(\"w\")"),
                                blob("fp16", 24,
                                     "f16(reshape(permute(concat(3,\n"
                                     "      reshape(take(w0, 2, [2,3]), [3,2,2,1]),\n"
                                     "      reshape(take(w0, 2, [0,1]), [3,2,2,1])),\n"
                                     "    [2,1,0,3]), [4,3,2]))"),
                                let_("a = reshape(t(\"alpha\"), [1,3,1])"),
                                let_("inv = clip(recip(add(f64(a), 1e-9)), -60000, 60000)"),
                                blob("fp16", 3, "f16(f32(mul(inv, 9.8672466)))"),
                                blob("fp16", 3, "f16(div(a, 3.141592653589793))")}),
                   s);
    if (out.size() != 3) return;
    const auto k = f16s(out[0]);
    auto W = [&](int i, int o, int r) { return w[static_cast<size_t>((i * OC + o) * 2 * S + r)]; };
    for (int r = 0; r < S; ++r)
        for (int o = 0; o < OC; ++o)
            for (int i = 0; i < IC; ++i) {
                const size_t base = static_cast<size_t>(((r * OC + o) * IC + i) * 2);
                CHECK(k[base] == recipe::f32_to_f16(W(i, o, r + S)));
                CHECK(k[base + 1] == recipe::f32_to_f16(W(i, o, r)));
            }
    const auto coef = f16s(out[1]);
    const auto scale = f16s(out[2]);
    for (int i = 0; i < IC; ++i) {
        double inv = 1.0 / (static_cast<double>(alpha[i]) + 1e-9);
        inv = std::min(std::max(inv, -60000.0), 60000.0);
        CHECK(coef[i] == recipe::f32_to_f16(static_cast<float>(inv * 9.8672466)));
        CHECK(scale[i] == recipe::f32_to_f16(alpha[i] / static_cast<float>(3.141592653589793)));
    }
}

void case_liveness_and_observer() {
    // x is used by blobs 0 and 1, y by blob 2: x must be freed before y is
    // read, so the peak is one tensor plus one output, not both tensors.
    const int n = 256 * 1024;
    std::vector<float> a(n, 1.0f), b(n, 2.0f);
    MemSource s({TT{"a", T_F32, {n}, a}, TT{"b", T_F32, {n}, b}}, {});
    const std::string d = "source \"a\" f32 [" + std::to_string(n) + "]\nsource \"b\" f32 [" + std::to_string(n) + "]\n";
    recipe::EvalStats st;
    std::map<std::string, std::array<uint8_t, 32>> seen;
    const recipe::Recipe r = recipe::parse(recipe_text(
        "t", zeros64(), d,
        {let_("x = t(\"a\")"), let_("y = t(\"b\")"), blob("fp16", n, "f16(x)"), blob("fp16", n, "f16(x)"),
         blob("fp16", n, "f16(y)")}));
    size_t blobs = 0;
    Error e = recipe::evaluate(
        r, s,
        [&](const recipe::BlobOut &) {
            ++blobs;
            return Error{};
        },
        &st, [&](const std::string & name, const std::array<uint8_t, 32> & sha) {
            seen[name] = sha;
            return Error{};
        });
    CHECK_MSG(e.ok(), e.message);
    CHECK(blobs == 3);
    CHECK(st.tensors_read == 2);  // x is cached across its two uses
    const uint64_t one = static_cast<uint64_t>(n) * 4;
    CHECK_MSG(st.peak_bytes == one + one / 2, "peak " + std::to_string(st.peak_bytes));
    CHECK(seen.size() == 2);
    CHECK(seen["a"] == detail::sha256(s.raw_["a"].data(), s.raw_["a"].size()));
    // An observer error aborts the evaluation with its class.
    e = recipe::evaluate(
        r, s, [](const recipe::BlobOut &) { return Error{}; }, nullptr,
        [](const std::string &, const std::array<uint8_t, 32> &) { return Error::transient("changed"); });
    CHECK(!e.ok() && e.cls == ErrorClass::transient);
}

void case_golden_weight_bin() {
    // Every op of generator version 1 over awkward values; the weight.bin
    // bytes are pinned, so Linux and macOS CI prove the same bytes (§2.6:
    // IEEE per op, no FMA contraction, our own fp16 rounding).
    std::vector<float> w(64);
    uint64_t x = 0x9E3779B97F4A7C15ull;
    for (float & v : w) {
        x = x * 6364136223846793005ull + 1442695040888963407ull;
        v = static_cast<float>(static_cast<double>(x >> 11) * (6.0 / 9007199254740992.0) - 3.0);
    }
    w[5] = 1e-12f;   // recip near zero: clipped
    w[9] = -0.0f;
    w[17] = 65519.0f / 4.0f;
    MemSource s({TT{"w", T_F32, {4, 16}, w}}, {});
    const std::string d = "source \"w\" f32 [4,16]\n";
    std::vector<uint8_t> file;
    const recipe::Recipe r = recipe::parse(recipe_text(
        "t", zeros64(), d,
        {let_("w = t(\"w\")"), blob("fp16", 64, "f16(w)"), blob("fp16", 64, "f16(div(w, 3.141592653589793))"),
         blob("fp16", 64, "f16(f32(mul(clip(recip(add(f64(w), 1e-9)), -6000, 6000), 9.8672466)))"),
         blob("fp32", 64, "f32(permute(reshape(w, [4,4,4]), [2,0,1]))"),
         blob("fp16", 76, "f16(concat(1, take(w, 1, [15,-1,0]), mul(w, recip(3))))"),
         blob("fp32", 16, "f32(add(f64(take(w, 0, [3])), div(take(w, 0, [1]), 7)))")}));
    milblob::Writer wr(r.blob_count, r.weight_bin_size, [&](const uint8_t * p, size_t n) {
        file.insert(file.end(), p, p + n);
        return Error{};
    });
    CHECK(wr.begin().ok());
    Error e = recipe::evaluate(r, s, [&](const recipe::BlobOut & b) {
        return wr.blob(b.offset, b.dtype == BlobType::fp16 ? milblob::FLOAT16 : milblob::FLOAT32, b.data, b.bytes);
    });
    CHECK_MSG(e.ok(), e.message);
    CHECK(wr.finish().ok());
    const std::string got = detail::to_hex(detail::sha256(file.data(), file.size()));
    // The same bytes as an independent numpy 1.26.4 reproduction of these
    // expressions (the exporters' arithmetic).
    CHECK_MSG(got == "12c3c0bcedc45c76cd8515703889829bf4d57fcd4bbec8e6cc7988d1ada43db6", "weight.bin sha256 " + got);
}

// ---- number formats and tolerance -----------------------------------------------------------

void case_f16_rounding() {
    CHECK(recipe::f32_to_f16(1.0f) == 0x3c00);
    CHECK(recipe::f32_to_f16(65504.0f) == 0x7bff);
    CHECK(recipe::f32_to_f16(65519.0f) == 0x7bff);
    CHECK(recipe::f32_to_f16(65520.0f) == 0x7c00);  // ties to even overflows
    CHECK(recipe::f32_to_f16(std::ldexp(1.0f, -24)) == 0x0001);
    CHECK(recipe::f32_to_f16(std::ldexp(1.0f, -25)) == 0x0000);       // tie to even (0)
    CHECK(recipe::f32_to_f16(std::ldexp(3.0f, -25)) == 0x0002);       // tie to even (2)
    CHECK(recipe::f32_to_f16(1.0f + std::ldexp(1.0f, -11)) == 0x3c00);  // tie to even
    CHECK(recipe::f32_to_f16(1.0f + std::ldexp(3.0f, -11)) == 0x3c02);
    CHECK(recipe::f32_to_f16(-0.0f) == 0x8000);
    CHECK((recipe::f32_to_f16(std::nanf("")) & 0x7c00) == 0x7c00 && (recipe::f32_to_f16(std::nanf("")) & 0x3ff) != 0);
    // Round trip of every finite half.
    int bad = 0;
    for (uint32_t h = 0; h < 0x10000; ++h) {
        if ((h & 0x7c00) == 0x7c00) continue;
        if (recipe::f32_to_f16(recipe::f16_to_f32(static_cast<uint16_t>(h))) != h) ++bad;
    }
    CHECK(bad == 0);
#if defined(__FLT16_MAX__)
    // Against the compiler's IEEE conversion over a strided sweep of floats.
    int mismatch = 0;
    for (uint64_t u = 0; u < 0x100000000ull; u += 4099) {
        const float f = from_bits(static_cast<uint32_t>(u));
        if (std::isnan(f)) continue;
        const _Float16 ref = static_cast<_Float16>(f);
        uint16_t rb;
        std::memcpy(&rb, &ref, 2);
        if (rb != recipe::f32_to_f16(f)) ++mismatch;
    }
    CHECK(mismatch == 0);
#endif
}

void case_ulp_and_compare() {
    // ULP distance across +-0.
    CHECK(recipe::ulp_distance_f16(0x0000, 0x8000) == 0);
    CHECK(recipe::ulp_distance_f16(0x0001, 0x8001) == 2);
    CHECK(recipe::ulp_distance_f16(0x3c00, 0x3c01) == 1);
    CHECK(recipe::ulp_distance_f16(0x7c00, 0x3c00) == -1);
    CHECK(recipe::ulp_distance_f32(0.0f, -0.0f) == 0);
    CHECK(recipe::ulp_distance_f32(from_bits(1), from_bits(0x80000001u)) == 2);
    CHECK(recipe::ulp_distance_f32(1.0f, std::nextafter(1.0f, 2.0f)) == 1);
    CHECK(recipe::ulp_distance_f32(std::nanf(""), 1.0f) == -1);

    auto cmp16 = [](bool ulp, std::vector<uint16_t> a, std::vector<uint16_t> b) {
        return recipe::compare_blob(BlobType::fp16, ulp, reinterpret_cast<const uint8_t *>(a.data()),
                                    reinterpret_cast<const uint8_t *>(b.data()), a.size());
    };
    // exact: any differing bit fails, including +0 vs -0 (byte-equal rule).
    auto c = cmp16(false, {0x3c00, 0x3c01}, {0x3c00, 0x3c00});
    CHECK(!c.pass && c.differing == 1 && c.max_ulp == 1);
    c = cmp16(false, {0x0000}, {0x8000});
    CHECK(!c.pass && c.differing == 1 && c.max_ulp == 0);
    // ulp: +0 vs -0 differ in bits but are 0 ulp apart, so they pass.
    c = cmp16(true, {0x0000, 0x3c00}, {0x8000, 0x3c00});
    CHECK(c.pass && c.differing == 1 && c.max_ulp == 0);
    std::vector<float> za = {0.0f}, zb = {-0.0f};
    auto cz = recipe::compare_blob(BlobType::fp32, false, reinterpret_cast<const uint8_t *>(za.data()),
                                   reinterpret_cast<const uint8_t *>(zb.data()), 1);
    CHECK(!cz.pass && cz.differing == 1);
    cz = recipe::compare_blob(BlobType::fp32, true, reinterpret_cast<const uint8_t *>(za.data()),
                              reinterpret_cast<const uint8_t *>(zb.data()), 1);
    CHECK(cz.pass && cz.max_ulp == 0);
    // ulp: 1 ulp passes; 2 ulp passes only within 2^-10 * max|ref|.
    c = cmp16(true, {0x3c01, 0x4000}, {0x3c00, 0x4000});
    CHECK(c.pass && c.differing == 1 && c.max_ulp == 1);
    c = cmp16(true, {0x3c02}, {0x3c00});
    CHECK(!c.pass && c.max_ulp == 2);
    c = cmp16(true, {0x0002, 0x5000}, {0x0000, 0x5000});  // near zero, large max: within tolerance
    CHECK(c.pass && c.max_ulp == 2);
    c = cmp16(true, {0x7c00}, {0x7c00});  // Inf in either: failure
    CHECK(!c.pass && c.max_ulp == -1);
    std::vector<float> fa = {1.0f, 2.0f}, fb = {1.0f, std::nextafter(2.0f, 3.0f)};
    auto cf = recipe::compare_blob(BlobType::fp32, false, reinterpret_cast<const uint8_t *>(fa.data()),
                                   reinterpret_cast<const uint8_t *>(fb.data()), 2);
    CHECK(!cf.pass && cf.max_ulp == 1);
    cf = recipe::compare_blob(BlobType::fp32, true, reinterpret_cast<const uint8_t *>(fa.data()),
                              reinterpret_cast<const uint8_t *>(fb.data()), 2);
    CHECK(cf.pass);
}

// ---- templates and matching ------------------------------------------------------------------

TemplateSpec toy_spec() {
    TemplateSpec s;
    s.id = "toy-dac";
    s.decls = "source \"a\" f32,f16 [2,3]\nmeta \"toy.n\" u32\n";
    s.items = {blob("fp16", 6, "f16(t(\"a\"))")};
    s.match = {{"architecture", "toy"}, {"metadata", {{"toy.n", 7}, {"toy.rates", {8, 4}}}}};
    return s;
}

void case_template() {
    TempDir td;
    auto t = make_template(td / "tpl", toy_spec());
    CHECK(t->id() == "toy-dac");
    CHECK(t->info().export_kind == "toy-v1");
    CHECK(t->info().functions == std::vector<std::string>({"main"}));
    CHECK(t->files().size() == 4);
    // template_sha256 by hand (B.4).
    detail::Sha256 h;
    static const char tag[] = "qvac-coreml-template";
    h.update(tag, sizeof(tag));
    for (const char * n : {"Manifest.json", "model.mlmodel", "recipe.txt", "template.json"}) {
        const std::string b = read_bytes(td / ("tpl/" + std::string(n)));
        h.str(n);
        const auto fsha = detail::sha256(b.data(), b.size());
        h.update(fsha.data(), 32);
    }
    CHECK(t->sha256() == h.finish());
    // Any file change changes the identity; reference-gguf.json does not count.
    write_bytes(td / "tpl/reference-gguf.json", "{}");
    CHECK(recipe::Template::from_dir(td / "tpl")->sha256() == t->sha256());
    write_bytes(td / "tpl/literals.bin", "x");
    CHECK(recipe::Template::from_dir(td / "tpl")->sha256() != t->sha256());
    fs::remove(td / "tpl/literals.bin");

    // Inconsistent templates are rejected.
    auto rejects = [&](const std::function<void()> & mutate, const std::string & needle) {
        TempDir d2;
        make_template(d2 / "x", toy_spec());
        const std::string dir = d2 / "x";
        try {
            mutate_dir = dir;
            mutate();
            recipe::Template::from_dir(dir);
            CHECK_MSG(false, "accepted, expected '" + needle + "'");
        } catch (const std::exception & e) {
            CHECK_MSG(std::string(e.what()).find(needle) != std::string::npos, e.what());
        }
    };
    rejects([] { write_bytes(mutate_dir + "/model.mlmodel", "other spec"); }, "spec_sha256");
    rejects(
        [] {
            json j = json::parse(read_bytes(mutate_dir + "/template.json"));
            j["id"] = "other";
            write_bytes(mutate_dir + "/template.json", j.dump());
        },
        "differs from recipe");
    rejects(
        [] {
            json j = json::parse(read_bytes(mutate_dir + "/template.json"));
            j["patches"] = json::array({{{"spec_offset", 1}}});
            write_bytes(mutate_dir + "/template.json", j.dump());
        },
        "patches is not supported");
    rejects([] { fs::remove(mutate_dir + "/Manifest.json"); }, "missing Manifest.json");
}

void case_match() {
    TempDir td;
    auto t = make_template(td / "tpl", toy_spec());
    const std::vector<float> six(6, 1.0f);
    auto with = [&](std::vector<TT> tt, std::vector<TM> mm) {
        MemSource s(std::move(tt), std::move(mm));
        return recipe::match(*t, s);
    };
    const std::vector<TM> meta = {{"general.architecture", "str", "toy"},
                                  {"toy.n", "u32", 7},
                                  {"toy.rates", "arr_u32", json::array({8, 4})}};
    CHECK(with({TT{"a", T_F32, {2, 3}, six}}, meta).ok);
    CHECK(with({TT{"a", T_F16, {2, 3}, six}}, meta).ok);
    auto m = with({TT{"a", 8 /* q8_0 */, {2, 3}, six}}, meta);
    CHECK(!m.ok && m.reason.find("not an allowed type") != std::string::npos);
    m = with({TT{"a", T_F32, {3, 2}, six}}, meta);
    CHECK(!m.ok && m.reason.find("ne") != std::string::npos);
    m = with({TT{"b", T_F32, {2, 3}, six}}, meta);
    CHECK(!m.ok && m.reason.find("absent") != std::string::npos);
    m = with({TT{"a", T_F32, {2, 3}, six}},
             {{"general.architecture", "str", "other"}, {"toy.n", "u32", 7}, {"toy.rates", "arr_u32", json::array({8, 4})}});
    CHECK(!m.ok && m.reason.find("architecture") != std::string::npos);
    m = with({TT{"a", T_F32, {2, 3}, six}},
             {{"general.architecture", "str", "toy"}, {"toy.n", "u32", 8}, {"toy.rates", "arr_u32", json::array({8, 4})}});
    CHECK(!m.ok && m.reason.find("toy.n") != std::string::npos);
    m = with({TT{"a", T_F32, {2, 3}, six}},
             {{"general.architecture", "str", "toy"}, {"toy.n", "f32", 7.0}, {"toy.rates", "arr_u32", json::array({8, 4})}});
    CHECK(!m.ok && m.reason.find("not u32") != std::string::npos);  // value equal, declared type differs
}

#if defined(QVAC_COREML_HAVE_GGML)
// ---- GgufSource (ggml-base only) --------------------------------------------------------------

void case_gguf_source() {
    TempDir td;
    const std::string path = td / "src.gguf";
    // A q8_0 tensor larger than the 8 MiB read chunk, so blocks straddle chunks.
    const int64_t ne0 = 4096, ne1 = 2100;  // 8.6M floats -> ~9.1 MB of q8_0
    std::vector<float> big(static_cast<size_t>(ne0 * ne1));
    for (size_t i = 0; i < big.size(); ++i) big[i] = static_cast<float>(std::sin(0.001 * static_cast<double>(i)));
    std::vector<uint8_t> q(ggml_row_size(GGML_TYPE_Q8_0, ne0) * static_cast<size_t>(ne1));
    ggml_quantize_chunk(GGML_TYPE_Q8_0, big.data(), q.data(), 0, ne1, ne0, nullptr);
    std::vector<float> small = {1.5f, -2.25f, 3.0f};
    std::vector<ggml_fp16_t> half(4);
    for (int i = 0; i < 4; ++i) half[i] = ggml_fp32_to_fp16(0.5f * static_cast<float>(i));
    {
        gguf_context * g = gguf_init_empty();
        gguf_set_val_str(g, "general.architecture", "toy");
        gguf_set_val_u32(g, "toy.n", 7);
        const uint32_t rates[2] = {8, 4};
        gguf_set_arr_data(g, "toy.rates", GGUF_TYPE_UINT32, rates, 2);
        ggml_init_params p{ggml_tensor_overhead() * 8 + q.size() + 64 * 1024, nullptr, false};
        ggml_context * ctx = ggml_init(p);
        ggml_tensor * tq = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, ne0, ne1);
        ggml_set_name(tq, "big");
        std::memcpy(tq->data, q.data(), q.size());
        ggml_tensor * ts = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);
        ggml_set_name(ts, "small");
        std::memcpy(ts->data, small.data(), 12);
        ggml_tensor * th = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 2, 2);
        ggml_set_name(th, "half");
        std::memcpy(th->data, half.data(), 8);
        gguf_add_tensor(g, tq);
        gguf_add_tensor(g, ts);
        gguf_add_tensor(g, th);
        CHECK(gguf_write_to_file(g, path.c_str(), false));
        gguf_free(g);
        ggml_free(ctx);
    }
    Error err;
    auto src = GgufSource::open(path, err);
    CHECK_MSG(src != nullptr, err.message);
    if (!src) return;
    const recipe::TensorInfo * ti = src->tensor("big");
    CHECK(ti && ti->type == GGML_TYPE_Q8_0 && ti->ne[0] == ne0 && ti->ne[1] == ne1 && ti->nbytes == q.size());
    CHECK(src->tensor("half")->ne[0] == 2 && src->tensor("half")->n_dims == 2);
    // The file is read through the open descriptor: unlinking the path changes nothing.
    fs::remove(path);
    std::vector<float> got;
    std::vector<uint8_t> tapped;
    Error e = src->read_f32("big", got, [&](const uint8_t * p, size_t n) { tapped.insert(tapped.end(), p, p + n); });
    CHECK_MSG(e.ok(), e.message);
    CHECK(tapped == q);
    std::vector<float> ref(big.size());
    ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(q.data(), ref.data(), static_cast<int64_t>(ref.size()));
    CHECK(got == ref);
    CHECK(src->read_f32("small", got, nullptr).ok() && got == small);
    CHECK(src->read_f32("half", got, nullptr).ok() && got == std::vector<float>({0.0f, 0.5f, 1.0f, 1.5f}));
    // Metadata, canonical (B.4) and as JSON.
    uint32_t type = 0, elem = 0;
    std::vector<uint8_t> bytes;
    CHECK(src->meta_canonical("toy.n", type, bytes, &elem) && type == recipe::GGUF_U32 &&
          bytes == std::vector<uint8_t>({7, 0, 0, 0}));
    CHECK(src->meta_canonical("general.architecture", type, bytes) && type == recipe::GGUF_STRING &&
          bytes == std::vector<uint8_t>({3, 0, 0, 0, 't', 'o', 'y'}));
    CHECK(src->meta_canonical("toy.rates", type, bytes, &elem) && type == recipe::GGUF_ARRAY &&
          elem == recipe::GGUF_U32 &&
          bytes == std::vector<uint8_t>({2, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 4, 0, 0, 0}));
    CHECK(!src->meta_canonical("missing", type, bytes));
    CHECK(src->meta_json("toy.rates") == "[8,4]");
    CHECK(src->meta_json("general.architecture") == "\"toy\"");
    SourceStat st;
    CHECK(src->stat_now(st) && st.size > q.size());
    // A file that is not a GGUF: deterministic.
    write_bytes(td / "bad.gguf", "not a gguf");
    CHECK(GgufSource::open(td / "bad.gguf", err) == nullptr && err.cls == ErrorClass::deterministic);
    CHECK(GgufSource::open(td / "absent.gguf", err) == nullptr && err.cls == ErrorClass::transient);
}
#endif

}  // namespace

int main(int argc, char ** argv) {
    return run_cases(argc, argv,
                     {
                         {"grammar-valid", case_grammar_valid},
                         {"grammar-errors", case_grammar_errors},
                         {"op-t-and-casts", case_op_t_and_casts},
                         {"op-arith-broadcast", case_op_arith_broadcast},
                         {"dtype-promotion", case_dtype_promotion},
                         {"op-recip-clip", case_op_recip_clip},
                         {"op-layout", case_op_layout},
                         {"parler-examples", case_parler_examples},
                         {"liveness-and-observer", case_liveness_and_observer},
                         {"golden-weight-bin", case_golden_weight_bin},
                         {"f16-rounding", case_f16_rounding},
                         {"ulp-and-compare", case_ulp_and_compare},
                         {"template", case_template},
                         {"match", case_match},
#if defined(QVAC_COREML_HAVE_GGML)
                         {"gguf-source", case_gguf_source},
#endif
                     },
                     "test-coreml-recipe");
}
