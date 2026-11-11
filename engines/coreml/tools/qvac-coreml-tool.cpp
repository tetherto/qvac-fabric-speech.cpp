// qvac-coreml-tool: host CLI of the portable Core ML generator (PLAN §2.3, §2.4,
// A.3). Built with the engines' *_BUILD_TESTS (and by the standalone
// engines/coreml build when ggml is found); Linux and macOS.
//
//   dump     <gguf> [-o reference-gguf.json]
//            Shape manifest: every metadata key/value and every tensor's name,
//            type and ne, plus per-tensor value statistics (min, max, mean, std).
//   twin     --manifest <reference-gguf.json> --seed <S> --types <t[,t...]> -o <out.gguf>
//            [--set key=value ...]
//            A production-geometry random GGUF: values per tensor inside the
//            recorded range; tensors that are quantised (or f16/bf16) in the
//            reference take the listed types in turn, the rest stay f32.
//            Quantised with ggml's own quantizers. --set overrides a string
//            metadata value (extraction sentinels).
//   f32      <in.gguf> -o <out.gguf>
//            An F32 copy dequantised by ggml (to_float): the exporter reference
//            for quantised sources.
//   generate --template <dir> --gguf <file> -o <out.mlpackage>
//            Runs the C++ interpreter and MILBlob writer (no compile).
//   digest   --template <dir> --gguf <file> [--options out=fp16]
//            template_sha256, source_digest, artifact_key and memo_id (B.4).
//   verify   --template <dir> (--gguf <file> | --ours <weight.bin>) --ref <weight.bin>
//            Per blob: offset, exact|ulp, differing elements, max ulp distance
//            (A.3). Exit 1 when any blob fails.
//
// The tool links the tests' copy of the runtime (namespace qvac_coreml_test).

#include "qvac_coreml/recipe.h"

#include "cache.h"
#include "digest.h"
#include "gguf_source.h"
#include "milblob.h"
#include "sha256.h"

#include "ggml.h"
#include "gguf.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <limits.h>
#include <stdlib.h>

using namespace QVAC_COREML_NS;
using json = nlohmann::json;

namespace {

[[noreturn]] void die(const std::string & m) {
    std::fprintf(stderr, "qvac-coreml-tool: %s\n", m.c_str());
    std::exit(2);
}

struct Args {
    std::vector<std::string> pos;
    std::map<std::string, std::vector<std::string>> opt;

    std::string get(const std::string & k, const std::string & def = {}) const {
        auto it = opt.find(k);
        return it == opt.end() || it->second.empty() ? def : it->second.back();
    }
    std::string need(const std::string & k) const {
        const std::string v = get(k);
        if (v.empty()) die("missing " + k);
        return v;
    }
};

Args parse_args(int argc, char ** argv, int from) {
    Args a;
    for (int i = from; i < argc; ++i) {
        std::string s = argv[i];
        if (s.size() > 1 && s[0] == '-') {
            if (i + 1 >= argc) die("missing value for " + s);
            a.opt[s].push_back(argv[++i]);
        } else {
            a.pos.push_back(s);
        }
    }
    return a;
}

std::unique_ptr<GgufSource> open_gguf(const std::string & path) {
    Error err;
    auto src = GgufSource::open(path, err);
    if (!src) die(err.message);
    return src;
}

void write_text(const std::string & path, const std::string & text) {
    if (path.empty() || path == "-") {
        std::fwrite(text.data(), 1, text.size(), stdout);
        std::fputc('\n', stdout);
        return;
    }
    std::ofstream out(path, std::ios::binary);
    out << text << "\n";
    if (!out) die("cannot write " + path);
}

// ---- dump ------------------------------------------------------------------------

json meta_entry(const gguf_context * g, int64_t id) {
    json e;
    e["key"] = gguf_get_key(g, id);
    const gguf_type t = gguf_get_kv_type(g, id);
    e["gguf_type"] = static_cast<int>(t);
    if (t == GGUF_TYPE_ARRAY) e["elem_type"] = static_cast<int>(gguf_get_arr_type(g, id));
    return e;
}

int cmd_dump(const Args & a) {
    if (a.pos.size() != 1) die("dump <gguf> [-o out.json]");
    auto src = open_gguf(a.pos[0]);
    const auto * g = static_cast<const gguf_context *>(src->gguf());
    json j;
    j["schema"] = 1;
    j["gguf_version"] = gguf_get_version(g);
    json meta = json::array();
    for (int64_t i = 0; i < gguf_get_n_kv(g); ++i) {
        json e = meta_entry(g, i);
        e["value"] = json::parse(src->meta_json(e["key"].get<std::string>()));
        meta.push_back(e);
    }
    j["metadata"] = meta;
    json tensors = json::array();
    for (const recipe::TensorInfo & ti : src->tensors()) {
        json t;
        t["name"] = ti.name;
        t["type"] = ggml_type_name(static_cast<ggml_type>(ti.type));
        t["ne"] = std::vector<int64_t>(ti.ne.begin(), ti.ne.end());
        t["n_dims"] = ti.n_dims;
        std::vector<float> v;
        Error e = src->read_f32(ti.name, v, nullptr);
        if (e.ok() && !v.empty()) {
            double mn = v[0], mx = v[0], sum = 0.0, sq = 0.0;
            for (float x : v) {
                mn = std::min(mn, static_cast<double>(x));
                mx = std::max(mx, static_cast<double>(x));
                sum += x;
            }
            const double mean = sum / static_cast<double>(v.size());
            for (float x : v) sq += (x - mean) * (x - mean);
            t["stats"] = {{"min", mn}, {"max", mx}, {"mean", mean}, {"std", std::sqrt(sq / static_cast<double>(v.size()))}};
        } else {
            t["stats"] = nullptr;  // not a float tensor
        }
        tensors.push_back(t);
    }
    j["tensors"] = tensors;
    write_text(a.get("-o"), j.dump(1));
    return 0;
}

// ---- GGUF writing (meta first, then each tensor's data streamed) -------------------

class StreamWriter {
public:
    StreamWriter(gguf_context * g, std::string path) : g_(g), path_(std::move(path)) {
        ggml_init_params p{};
        p.mem_size = ggml_tensor_overhead() * 32768;
        p.no_alloc = true;
        ctx_ = ggml_init(p);
    }
    ~StreamWriter() {
        if (ctx_) ggml_free(ctx_);
    }

    void add(const std::string & name, ggml_type type, int n_dims, const int64_t * ne) {
        ggml_tensor * t = ggml_new_tensor(ctx_, type, n_dims, ne);
        ggml_set_name(t, name.c_str());
        gguf_add_tensor(g_, t);
        order_.push_back(t);
    }

    // Meta, then `fill(i, bytes)` for every tensor in order.
    template <class Fill>
    void write(Fill fill) {
        if (!gguf_write_to_file(g_, path_.c_str(), /*only_meta=*/true)) die("cannot write " + path_);
        FILE * f = std::fopen(path_.c_str(), "ab");
        if (!f) die("cannot append to " + path_);
        const size_t align = gguf_get_alignment(g_);
        std::vector<uint8_t> bytes;
        for (size_t i = 0; i < order_.size(); ++i) {
            bytes.clear();
            fill(i, order_[i], bytes);
            if (bytes.size() != ggml_nbytes(order_[i])) die(std::string("size mismatch for ") + order_[i]->name);
            const size_t pad = (align - bytes.size() % align) % align;
            bytes.resize(bytes.size() + pad, 0);
            if (std::fwrite(bytes.data(), 1, bytes.size(), f) != bytes.size()) die("short write to " + path_);
        }
        if (std::fclose(f) != 0) die("cannot close " + path_);
    }

private:
    gguf_context * g_;
    std::string path_;
    ggml_context * ctx_ = nullptr;
    std::vector<ggml_tensor *> order_;
};

std::vector<uint8_t> encode(ggml_type type, const std::vector<float> & v, int64_t ne0) {
    std::vector<uint8_t> out;
    const int64_t n = static_cast<int64_t>(v.size());
    if (type == GGML_TYPE_F32) {
        out.resize(v.size() * 4);
        std::memcpy(out.data(), v.data(), out.size());
    } else if (type == GGML_TYPE_F16) {
        out.resize(v.size() * 2);
        for (size_t i = 0; i < v.size(); ++i) {
            const uint16_t h = recipe::f32_to_f16(v[i]);
            std::memcpy(out.data() + 2 * i, &h, 2);
        }
    } else {
        const int64_t rows = n / ne0;
        out.resize(ggml_row_size(type, ne0) * static_cast<size_t>(rows));
        ggml_quantize_chunk(type, v.data(), out.data(), 0, rows, ne0, nullptr);
    }
    return out;
}

// ---- twin ------------------------------------------------------------------------

// splitmix64: the same stream on every host.
struct Rng {
    uint64_t s;
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    double uniform() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); }
    // Irwin-Hall(12) - 6: approximately N(0, 1) with no libm, so twins are
    // byte-identical across hosts.
    double normal() {
        double s = 0.0;
        for (int i = 0; i < 12; ++i) s += uniform();
        return s - 6.0;
    }
};

void set_meta(gguf_context * g, const json & e) {
    const std::string k = e.at("key");
    const int t = e.at("gguf_type");
    const json & v = e.at("value");
    switch (t) {
        case GGUF_TYPE_UINT8: gguf_set_val_u8(g, k.c_str(), v.get<uint8_t>()); break;
        case GGUF_TYPE_INT8: gguf_set_val_i8(g, k.c_str(), v.get<int8_t>()); break;
        case GGUF_TYPE_UINT16: gguf_set_val_u16(g, k.c_str(), v.get<uint16_t>()); break;
        case GGUF_TYPE_INT16: gguf_set_val_i16(g, k.c_str(), v.get<int16_t>()); break;
        case GGUF_TYPE_UINT32: gguf_set_val_u32(g, k.c_str(), v.get<uint32_t>()); break;
        case GGUF_TYPE_INT32: gguf_set_val_i32(g, k.c_str(), v.get<int32_t>()); break;
        case GGUF_TYPE_FLOAT32: gguf_set_val_f32(g, k.c_str(), v.get<float>()); break;
        case GGUF_TYPE_UINT64: gguf_set_val_u64(g, k.c_str(), v.get<uint64_t>()); break;
        case GGUF_TYPE_INT64: gguf_set_val_i64(g, k.c_str(), v.get<int64_t>()); break;
        case GGUF_TYPE_FLOAT64: gguf_set_val_f64(g, k.c_str(), v.get<double>()); break;
        case GGUF_TYPE_BOOL: gguf_set_val_bool(g, k.c_str(), v.get<bool>()); break;
        case GGUF_TYPE_STRING: gguf_set_val_str(g, k.c_str(), v.get<std::string>().c_str()); break;
        case GGUF_TYPE_ARRAY: {
            const int et = e.at("elem_type");
            if (et == GGUF_TYPE_STRING) {
                std::vector<std::string> s = v.get<std::vector<std::string>>();
                std::vector<const char *> p;
                for (const std::string & x : s) p.push_back(x.c_str());
                gguf_set_arr_str(g, k.c_str(), p.data(), p.size());
                break;
            }
            std::vector<uint8_t> raw;
            for (const json & x : v) {
                auto put = [&](auto val) {
                    const auto * b = reinterpret_cast<const uint8_t *>(&val);
                    raw.insert(raw.end(), b, b + sizeof(val));
                };
                switch (et) {
                    case GGUF_TYPE_UINT8: put(x.get<uint8_t>()); break;
                    case GGUF_TYPE_INT8: put(x.get<int8_t>()); break;
                    case GGUF_TYPE_UINT16: put(x.get<uint16_t>()); break;
                    case GGUF_TYPE_INT16: put(x.get<int16_t>()); break;
                    case GGUF_TYPE_UINT32: put(x.get<uint32_t>()); break;
                    case GGUF_TYPE_INT32: put(x.get<int32_t>()); break;
                    case GGUF_TYPE_FLOAT32: put(x.get<float>()); break;
                    case GGUF_TYPE_UINT64: put(x.get<uint64_t>()); break;
                    case GGUF_TYPE_INT64: put(x.get<int64_t>()); break;
                    case GGUF_TYPE_FLOAT64: put(x.get<double>()); break;
                    case GGUF_TYPE_BOOL: put(static_cast<int8_t>(x.get<bool>())); break;
                    default: die("unsupported array element type for " + k);
                }
            }
            gguf_set_arr_data(g, k.c_str(), static_cast<gguf_type>(et), raw.data(), v.size());
            break;
        }
        default: die("unsupported metadata type for " + k);
    }
}

int cmd_twin(const Args & a) {
    std::ifstream in(a.need("--manifest"));
    if (!in) die("cannot read the manifest");
    const json m = json::parse(in, nullptr, false);
    if (!m.is_object() || m.value("schema", 0) != 1) die("not a schema-1 reference-gguf.json");
    const uint64_t seed = std::stoull(a.need("--seed"));
    std::vector<ggml_type> types;
    {
        std::stringstream ss(a.need("--types"));
        std::string t;
        while (std::getline(ss, t, ',')) {
            auto ty = recipe::ggml_type_from_name(t);
            if (!ty) die("unknown type " + t);
            types.push_back(static_cast<ggml_type>(*ty));
        }
    }
    std::map<std::string, std::string> sets;
    if (auto it = a.opt.find("--set"); it != a.opt.end()) {
        for (const std::string & kv : it->second) {
            const size_t eq = kv.find('=');
            if (eq == std::string::npos) die("--set key=value");
            sets[kv.substr(0, eq)] = kv.substr(eq + 1);
        }
    }
    gguf_context * g = gguf_init_empty();
    for (const json & e : m.at("metadata")) {
        const std::string k = e.at("key");
        if (k == GGUF_KEY_GENERAL_ALIGNMENT) continue;  // the writer's own alignment
        if (sets.count(k)) {
            gguf_set_val_str(g, k.c_str(), sets[k].c_str());
            sets.erase(k);
            continue;
        }
        set_meta(g, e);
    }
    for (const auto & [k, v] : sets) gguf_set_val_str(g, k.c_str(), v.c_str());

    struct Plan {
        std::string name;
        ggml_type type;
        int n_dims;
        int64_t ne[4];
        double lo, hi, mean, sd;
        bool has_stats;
    };
    std::vector<Plan> plan;
    size_t rr = 0;
    for (const json & t : m.at("tensors")) {
        Plan p{};
        p.name = t.at("name");
        p.n_dims = t.at("n_dims");
        const auto ne = t.at("ne").get<std::vector<int64_t>>();
        for (int d = 0; d < 4; ++d) p.ne[d] = ne.at(d);
        const auto ref = recipe::ggml_type_from_name(t.at("type").get<std::string>());
        p.has_stats = t.contains("stats") && t["stats"].is_object();
        if (!p.has_stats) die("tensor " + p.name + " has no value statistics (not a float tensor)");
        p.lo = t["stats"]["min"];
        p.hi = t["stats"]["max"];
        p.mean = t["stats"]["mean"];
        p.sd = t["stats"]["std"];
        p.type = GGML_TYPE_F32;
        const bool low_precision = ref && *ref != GGML_TYPE_F32;
        if (low_precision) {
            const ggml_type want = types[rr % types.size()];
            if (want == GGML_TYPE_F32 || p.ne[0] % ggml_blck_size(want) == 0) {
                p.type = want;
                ++rr;
            }
        }
        plan.push_back(p);
    }
    const std::string out = a.need("-o");
    StreamWriter w(g, out);
    for (const Plan & p : plan) w.add(p.name, p.type, p.n_dims, p.ne);
    w.write([&](size_t i, ggml_tensor *, std::vector<uint8_t> & bytes) {
        const Plan & p = plan[i];
        detail::Sha256 h;
        h.str(p.name);
        const auto d = h.finish();
        uint64_t s = seed;
        for (int b = 0; b < 8; ++b) s ^= static_cast<uint64_t>(d[b]) << (8 * b);
        Rng rng{s};
        const int64_t n = p.ne[0] * p.ne[1] * p.ne[2] * p.ne[3];
        std::vector<float> v(static_cast<size_t>(n));
        for (float & x : v) {
            double y = p.mean + p.sd * rng.normal();
            y = std::min(std::max(y, p.lo), p.hi);
            x = static_cast<float>(y);
        }
        bytes = encode(p.type, v, p.ne[0]);
    });
    gguf_free(g);
    return 0;
}

// ---- f32 -------------------------------------------------------------------------

int cmd_f32(const Args & a) {
    if (a.pos.size() != 1) die("f32 <in.gguf> -o <out.gguf>");
    auto src = open_gguf(a.pos[0]);
    gguf_context * g = gguf_init_empty();
    gguf_set_kv(g, static_cast<const gguf_context *>(src->gguf()));
    gguf_remove_key(g, GGUF_KEY_GENERAL_ALIGNMENT);
    StreamWriter w(g, a.need("-o"));
    std::vector<bool> raw;
    for (const recipe::TensorInfo & ti : src->tensors()) {
        const auto type = static_cast<ggml_type>(ti.type);
        const ggml_type_traits * tr = ggml_get_type_traits(type);
        const bool convert = type == GGML_TYPE_F32 || (tr && tr->to_float);
        raw.push_back(!convert);
        w.add(ti.name, convert ? GGML_TYPE_F32 : type, ti.n_dims, ti.ne.data());
    }
    w.write([&](size_t i, ggml_tensor *, std::vector<uint8_t> & bytes) {
        const recipe::TensorInfo & ti = src->tensors()[i];
        if (raw[i]) {
            Error e = src->read_raw(ti.name, [&](const uint8_t * p, size_t n) {
                bytes.insert(bytes.end(), p, p + n);
                return true;
            });
            if (!e.ok()) die(e.message);
            return;
        }
        std::vector<float> v;
        Error e = src->read_f32(ti.name, v, nullptr);
        if (!e.ok()) die(e.message);
        bytes.resize(v.size() * 4);
        std::memcpy(bytes.data(), v.data(), bytes.size());
    });
    gguf_free(g);
    return 0;
}

// ---- generate / digest / verify --------------------------------------------------------

std::shared_ptr<const recipe::Template> load_template(const Args & a) {
    try {
        return recipe::Template::from_dir(a.need("--template"));
    } catch (const std::exception & e) {
        die(e.what());
    }
}

void require_match(const recipe::Template & t, const recipe::TensorSource & src) {
    const recipe::MatchResult m = recipe::match(t, src);
    if (!m.ok) die("no_template: " + m.reason);
}

int cmd_generate(const Args & a) {
    auto t = load_template(a);
    auto src = open_gguf(a.need("--gguf"));
    require_match(*t, *src);
    digest::SourceDigest dg;
    Error e = digest::source_digest(t->recipe(), *src, dg);
    if (!e.ok()) die(e.message);
    recipe::EvalStats stats;
    const std::string pkg = a.need("-o");
    e = cache::write_package(*t, *src, pkg, &stats);
    if (!e.ok()) die(e.message);
    std::ifstream wb(pkg + "/Data/com.apple.CoreML/weights/weight.bin", std::ios::binary);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(wb)), std::istreambuf_iterator<char>());
    json j;
    j["template"] = t->id();
    j["template_sha256"] = t->sha256_hex();
    j["source_digest"] = detail::to_hex(dg.digest);
    j["artifact_key"] = digest::artifact_key(GENERATOR_VERSION, t->sha256(), dg.digest, a.get("--options", digest::DEFAULT_OPTIONS));
    j["weight_bin_sha256"] = detail::to_hex(detail::sha256(bytes.data(), bytes.size()));
    j["weight_bin_size"] = bytes.size();
    j["generation_peak_bytes"] = stats.peak_bytes;
    j["generator_version"] = GENERATOR_VERSION;
    write_text("-", j.dump(1));
    return 0;
}

int cmd_digest(const Args & a) {
    auto t = load_template(a);
    const std::string path = a.need("--gguf");
    auto src = open_gguf(path);
    require_match(*t, *src);
    digest::SourceDigest dg;
    Error e = digest::source_digest(t->recipe(), *src, dg);
    if (!e.ok()) die(e.message);
    SourceStat st;
    src->stat_now(st);
    char rp[PATH_MAX];
    const std::string real = ::realpath(path.c_str(), rp) ? rp : path;
    const std::string options = a.get("--options", digest::DEFAULT_OPTIONS);
    json j;
    j["template"] = t->id();
    j["template_sha256"] = t->sha256_hex();
    j["source_digest"] = detail::to_hex(dg.digest);
    j["source_tensors"] = dg.tensors;
    j["source_bytes"] = dg.bytes;
    j["artifact_key"] = digest::artifact_key(GENERATOR_VERSION, t->sha256(), dg.digest, options);
    j["memo_id"] = digest::memo_id(real, st.size, st.mtime_ns, st.ino, st.dev, t->sha256(), GENERATOR_VERSION, options);
    j["realpath"] = real;
    j["options"] = options;
    write_text("-", j.dump(1));
    return 0;
}

std::vector<uint8_t> read_bytes(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) die("cannot read " + path);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

int cmd_verify(const Args & a) {
    auto t = load_template(a);
    std::vector<uint8_t> ours;
    if (!a.get("--ours").empty()) {
        ours = read_bytes(a.get("--ours"));
    } else {
        auto src = open_gguf(a.need("--gguf"));
        require_match(*t, *src);
        const recipe::Recipe & r = t->recipe();
        milblob::Writer w(r.blob_count, r.weight_bin_size, [&](const uint8_t * p, size_t n) -> Error {
            ours.insert(ours.end(), p, p + n);
            return {};
        });
        Error e = w.begin();
        if (e.ok()) {
            e = recipe::evaluate(r, *src, [&](const recipe::BlobOut & b) {
                return w.blob(b.offset, b.dtype == recipe::BlobType::fp16 ? milblob::FLOAT16 : milblob::FLOAT32, b.data,
                              b.bytes);
            });
        }
        if (e.ok()) e = w.finish();
        if (!e.ok()) die(e.message);
    }
    const std::vector<uint8_t> ref = read_bytes(a.need("--ref"));
    std::string err;
    auto oi = milblob::read_index(ours.data(), ours.size(), &err);
    if (!oi) die("ours: " + err);
    auto ri = milblob::read_index(ref.data(), ref.size(), &err);
    if (!ri) die("ref: " + err);
    bool ok = ours.size() == ref.size() && oi->size() == ri->size();
    if (!ok) std::printf("layout differs: %zu vs %zu bytes, %zu vs %zu blobs\n", ours.size(), ref.size(), oi->size(), ri->size());
    std::printf("%-12s %-5s %-6s %10s %8s %s\n", "offset", "class", "dtype", "differing", "max_ulp", "result");
    size_t bi = 0;
    for (const recipe::Item & it : t->recipe().items) {
        if (it.kind != recipe::Item::Kind::blob) continue;
        const milblob::Entry * o = bi < oi->size() ? &(*oi)[bi] : nullptr;
        const milblob::Entry * f = bi < ri->size() ? &(*ri)[bi] : nullptr;
        ++bi;
        const size_t bytes = it.count * recipe::blob_type_size(it.dtype);
        if (!o || !f || o->offset != it.offset || f->offset != it.offset || o->bytes != bytes || f->bytes != bytes) {
            std::printf("%-12llu %-5s %-6s %10s %8s FAIL (layout)\n", static_cast<unsigned long long>(it.offset),
                        it.ulp ? "ulp" : "exact", recipe::to_string(it.dtype), "-", "-");
            ok = false;
            continue;
        }
        const recipe::BlobCompare c =
            recipe::compare_blob(it.dtype, it.ulp, ours.data() + o->data_offset, ref.data() + f->data_offset, it.count);
        std::printf("%-12llu %-5s %-6s %10llu %8lld %s\n", static_cast<unsigned long long>(it.offset),
                    it.ulp ? "ulp" : "exact", recipe::to_string(it.dtype), static_cast<unsigned long long>(c.differing),
                    static_cast<long long>(c.max_ulp), c.pass ? "ok" : "FAIL");
        ok = ok && c.pass;
    }
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: qvac-coreml-tool dump|twin|f32|generate|digest|verify ...\n");
        return 2;
    }
    const std::string cmd = argv[1];
    const Args a = parse_args(argc, argv, 2);
    if (cmd == "dump") return cmd_dump(a);
    if (cmd == "twin") return cmd_twin(a);
    if (cmd == "f32") return cmd_f32(a);
    if (cmd == "generate") return cmd_generate(a);
    if (cmd == "digest") return cmd_digest(a);
    if (cmd == "verify") return cmd_verify(a);
    die("unknown command " + cmd);
}
