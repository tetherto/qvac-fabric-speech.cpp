// Template-declared tensor names (recipe::NameSpec, recipe::RenamedSource) and
// the source a template reads (template_source in gguf_source.h). Portable: no ggml.

#include "gguf_source.h"
#include "qvac_coreml/recipe.h"

#include "nlohmann/json.hpp"

#include <utility>

namespace QVAC_COREML_NS {
namespace recipe {

namespace {

// A metadata string array, or nullopt with `why` set when it is absent or not
// an array of strings.
std::optional<std::vector<std::string>> string_array(const TensorSource & src, const std::string & key,
                                                     std::string & why) {
    const std::string text = src.meta_json(key);
    if (text.empty()) {
        why = "metadata " + key + " is absent";
        return std::nullopt;
    }
    const nlohmann::json j = nlohmann::json::parse(text, nullptr, false);
    if (!j.is_array()) {
        why = "metadata " + key + " is not an array";
        return std::nullopt;
    }
    std::vector<std::string> out;
    out.reserve(j.size());
    for (const nlohmann::json & v : j) {
        if (!v.is_string()) {
            why = "metadata " + key + " is not an array of strings";
            return std::nullopt;
        }
        out.push_back(v.get<std::string>());
    }
    return out;
}

}  // namespace

RenamedSource::RenamedSource(TensorSource & base, const NameSpec & spec) : base_(base) {
    std::string why;
    const auto from = string_array(base, spec.map_from, why);
    const auto to = from ? string_array(base, spec.map_to, why) : std::nullopt;
    if (!from || !to) {
        error_ = why;
        return;
    }
    if (from->size() != to->size()) {
        error_ = "metadata " + spec.map_from + " and " + spec.map_to + " differ in length";
        return;
    }
    for (size_t i = 0; i < from->size(); ++i) stored_[(*from)[i]] = (*to)[i];

    // Aliases apply only when both arrays are present, like the engine's loader.
    if (!spec.aliases_from.empty() && !base.meta_json(spec.aliases_from).empty() &&
        !base.meta_json(spec.aliases_to).empty()) {
        const auto af = string_array(base, spec.aliases_from, why);
        const auto at = af ? string_array(base, spec.aliases_to, why) : std::nullopt;
        if (!af || !at) {
            error_ = why;
            stored_.clear();
            return;
        }
        if (af->size() != at->size()) {
            error_ = "metadata " + spec.aliases_from + " and " + spec.aliases_to + " differ in length";
            stored_.clear();
            return;
        }
        for (size_t i = 0; i < af->size(); ++i) {
            auto target = stored_.find((*at)[i]);
            if (target != stored_.end()) stored_.emplace((*af)[i], target->second);  // never replaces a name
        }
    }
    for (const auto & [name, other] : spec.fallbacks) {
        if (stored_.count(name)) continue;
        auto it = stored_.find(other);
        if (it != stored_.end()) stored_[name] = it->second;
    }
    for (const auto & [name, stored] : stored_) {
        if (const TensorInfo * ti = base_.tensor(stored)) {
            TensorInfo info = *ti;
            info.name = name;
            infos_.emplace(name, std::move(info));
        }
    }
}

std::string RenamedSource::stored_name(const std::string & name) const {
    auto it = stored_.find(name);
    return it == stored_.end() ? std::string() : it->second;
}

const TensorInfo * RenamedSource::tensor(const std::string & name) const {
    auto it = infos_.find(name);
    return it == infos_.end() ? nullptr : &it->second;
}

Error RenamedSource::read_raw(const std::string & name, const std::function<bool(const uint8_t *, size_t)> & sink) {
    const std::string stored = stored_name(name);
    if (stored.empty()) return Error::deterministic("tensor " + name + " does not resolve to a GGUF tensor");
    return base_.read_raw(stored, sink);
}

Error RenamedSource::read_f32(const std::string & name, std::vector<float> & out, const RawTap & tap) {
    const std::string stored = stored_name(name);
    if (stored.empty()) return Error::deterministic("tensor " + name + " does not resolve to a GGUF tensor");
    return base_.read_f32(stored, out, tap);
}

bool RenamedSource::meta_canonical(const std::string & key, uint32_t & gguf_type, std::vector<uint8_t> & bytes,
                                   uint32_t * array_elem_type) const {
    return base_.meta_canonical(key, gguf_type, bytes, array_elem_type);
}

std::string RenamedSource::meta_json(const std::string & key) const { return base_.meta_json(key); }

}  // namespace recipe

namespace {

// An OpenSource read under a template's names: the descriptor's identity
// checks go to the opened file.
class RenamedOpenSource final : public OpenSource {
public:
    RenamedOpenSource(std::unique_ptr<OpenSource> base, const recipe::NameSpec & spec)
        : base_(std::move(base)), named_(*base_, spec) {}

    const recipe::TensorInfo * tensor(const std::string & name) const override { return named_.tensor(name); }
    Error read_raw(const std::string & name, const std::function<bool(const uint8_t *, size_t)> & sink) override {
        return named_.read_raw(name, sink);
    }
    Error read_f32(const std::string & name, std::vector<float> & out, const recipe::RawTap & tap) override {
        return named_.read_f32(name, out, tap);
    }
    bool meta_canonical(const std::string & key, uint32_t & gguf_type, std::vector<uint8_t> & bytes,
                        uint32_t * array_elem_type) const override {
        return named_.meta_canonical(key, gguf_type, bytes, array_elem_type);
    }
    std::string meta_json(const std::string & key) const override { return named_.meta_json(key); }
    bool stat_now(SourceStat & out) const override { return base_->stat_now(out); }

private:
    std::unique_ptr<OpenSource> base_;
    recipe::RenamedSource named_;
};

}  // namespace

std::unique_ptr<OpenSource> template_source(const recipe::Template & t, std::unique_ptr<OpenSource> base) {
    if (!base || t.info().names.empty()) return base;
    return std::make_unique<RenamedOpenSource>(std::move(base), t.info().names);
}

}  // namespace QVAC_COREML_NS
