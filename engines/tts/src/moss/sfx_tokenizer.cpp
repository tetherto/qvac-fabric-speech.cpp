#include "moss/sfx_tokenizer.h"

#include "qwen_tokenizer.h"

#include <algorithm>
#include <stdexcept>

namespace tts_cpp::moss::detail {
namespace {

void load_vocabulary(QwenTokenizer & tokenizer, const std::vector<std::string> & tokens) {
    tokenizer.vocab.reserve(tokens.size());
    for (size_t id = 0; id < tokens.size(); ++id) {
        tokenizer.vocab.emplace(tokens[id], (int) id);
    }
}

void load_merges(QwenTokenizer & tokenizer, const std::vector<std::string> & merges) {
    tokenizer.merge_rank.reserve(merges.size());
    for (size_t rank = 0; rank < merges.size(); ++rank) {
        tokenizer.merge_rank.emplace(merges[rank], (int) rank);
    }
}

std::vector<int32_t> truncate_ids(const std::vector<int> & ids, size_t limit) {
    const size_t count = std::min(ids.size(), limit);
    return std::vector<int32_t>(ids.begin(), ids.begin() + (std::ptrdiff_t) count);
}

} // namespace

struct SfxTokenizer::Impl {
    mutable QwenTokenizer tokenizer;
    size_t limit = 0;
};

SfxTokenizer::SfxTokenizer(const SfxModel & model) : impl_(new Impl) {
    const std::vector<std::string> tokens = model.tokenizer_tokens();
    const std::vector<std::string> merges = model.tokenizer_merges();
    if (tokens.empty() || merges.empty()) {
        throw std::runtime_error("moss sfx: GGUF is missing the tokenizer vocabulary");
    }
    impl_->tokenizer.build_byte_map();
    load_vocabulary(impl_->tokenizer, tokens);
    load_merges(impl_->tokenizer, merges);
    impl_->limit = (size_t) model.config().text.max_tokens;
}

SfxTokenizer::~SfxTokenizer() = default;

std::vector<int32_t> SfxTokenizer::encode(const std::string & text) const {
    if (text.empty()) {
        return {};
    }
    return truncate_ids(impl_->tokenizer.encode(text), impl_->limit);
}

} // namespace tts_cpp::moss::detail
