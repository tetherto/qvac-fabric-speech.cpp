#include "moss/frontend.h"

#include "moss/generation.h"
#include "qwen_tokenizer.h"

#include <stdexcept>

namespace tts_cpp::moss::detail {
namespace {

constexpr const char * PAD_TOKEN      = "<|endoftext|>";
constexpr const char * IM_START_TOKEN = "<|im_start|>";
constexpr const char * IM_END_TOKEN   = "<|im_end|>";
constexpr const char * AUDIO_PLACEHOLDER = "<|audio|>";

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("moss frontend: " + message);
}

std::string user_instruction(const std::string & reference, const std::string & language,
                             const std::string & text) {
    return "<user_inst>\n"
           "- Reference(s):\n" + reference + "\n"
           "- Instruction:\nNone\n"
           "- Tokens:\nNone\n"
           "- Quality:\nNone\n"
           "- Sound Event:\nNone\n"
           "- Ambient Sound:\nNone\n"
           "- Language:\n" + language + "\n"
           "- Text:\n" + text + "\n"
           "</user_inst>";
}

std::vector<DelayRow> rows_from_text_ids(const std::vector<int32_t> & text_ids,
                                         const DelayConfig & config) {
    std::vector<DelayRow> rows(text_ids.size());
    for (size_t i = 0; i < text_ids.size(); ++i) {
        rows[i].text = text_ids[i];
        rows[i].audio.assign((size_t) config.n_vq, config.audio_pad_code);
    }
    return rows;
}

void fill_reference_rows(std::vector<DelayRow> & rows, int64_t offset,
                         const std::vector<int32_t> & delayed, const DelayConfig & config) {
    const int64_t delayed_rows = (int64_t) delayed.size() / config.n_vq;
    for (int64_t row = 0; row < delayed_rows; ++row) {
        for (int channel = 0; channel < config.n_vq; ++channel) {
            rows[(size_t) (offset + row)].audio[(size_t) channel] =
                    delayed[(size_t) row * config.n_vq + channel];
        }
    }
}

} // namespace

std::vector<DelayRow> build_prompt_rows(const DelayConfig & config, const PromptTokens & tokens,
                                        const TextEncoder & encode, const std::string & text,
                                        const std::string & language,
                                        const std::vector<int32_t> & reference_codes,
                                        int reference_frames) {
    const bool has_reference = reference_frames > 0;
    if (has_reference &&
        reference_codes.size() != (size_t) reference_frames * (size_t) config.n_vq) {
        fail("reference codes do not match the reported frame count");
    }

    const std::string reference_field = has_reference ? "[S1]:\n" + std::string(AUDIO_PLACEHOLDER)
                                                      : "None";
    const std::string content = user_instruction(reference_field, language, text);

    std::vector<int32_t> text_ids;
    std::vector<int32_t> delayed_reference;
    int64_t audio_row_offset = -1;

    auto append = [&](const std::vector<int32_t> & ids) {
        text_ids.insert(text_ids.end(), ids.begin(), ids.end());
    };

    append({tokens.im_start});
    append(encode("user\n"));
    const size_t split = content.find(AUDIO_PLACEHOLDER);
    if (has_reference && split != std::string::npos) {
        append(encode(content.substr(0, split)));
        append({(int32_t) config.audio_start_token_id});
        audio_row_offset = (int64_t) text_ids.size();
        delayed_reference = apply_delay_pattern(reference_codes, reference_frames,
                config.n_vq, config.audio_pad_code);
        const int64_t slots = (int64_t) reference_frames + config.n_vq - 1;
        text_ids.insert(text_ids.end(), (size_t) slots,
                (int32_t) config.audio_user_slot_token_id);
        append({(int32_t) config.audio_end_token_id});
        append(encode(content.substr(split + std::string(AUDIO_PLACEHOLDER).size())));
    } else {
        append(encode(content));
    }
    append({tokens.im_end});
    append(encode("\n"));
    append({tokens.im_start});
    append(encode("assistant\n"));

    std::vector<DelayRow> rows = rows_from_text_ids(text_ids, config);
    if (audio_row_offset >= 0) {
        fill_reference_rows(rows, audio_row_offset, delayed_reference, config);
    }

    DelayRow seed;
    seed.text = config.audio_start_token_id;
    seed.audio.assign((size_t) config.n_vq, config.audio_pad_code);
    rows.push_back(seed);
    return rows;
}

struct Frontend::Impl {
    QwenTokenizer tokenizer;
    PromptTokens tokens;

    void load(const DelayLM & model) {
        const std::vector<std::string> vocab = model.tokenizer_tokens();
        const std::vector<std::string> merges = model.tokenizer_merges();
        if (vocab.empty() || merges.empty()) {
            fail("backbone GGUF is missing the tokenizer vocabulary");
        }
        tokenizer.build_byte_map();
        tokenizer.vocab.reserve(vocab.size());
        for (size_t id = 0; id < vocab.size(); ++id) {
            tokenizer.vocab.emplace(vocab[id], (int) id);
        }
        tokenizer.merge_rank.reserve(merges.size());
        for (size_t rank = 0; rank < merges.size(); ++rank) {
            tokenizer.merge_rank.emplace(merges[rank], (int) rank);
        }
        tokens.pad      = lookup(PAD_TOKEN);
        tokens.im_start = lookup(IM_START_TOKEN);
        tokens.im_end   = lookup(IM_END_TOKEN);
        const int32_t text_vocab = model.config().text_vocab;
        if (tokens.pad >= text_vocab || tokens.im_start >= text_vocab ||
            tokens.im_end >= text_vocab) {
            fail("special tokens fall outside the model's text vocabulary");
        }
    }

    int32_t lookup(const std::string & token) const {
        const auto it = tokenizer.vocab.find(token);
        if (it == tokenizer.vocab.end()) {
            fail("tokenizer is missing the special token " + token);
        }
        return it->second;
    }

    std::vector<int32_t> encode(const std::string & text) {
        std::vector<int32_t> ids;
        for (int id : tokenizer.encode(text)) {
            ids.push_back((int32_t) id);
        }
        return ids;
    }
};

Frontend::Frontend(const DelayLM & model) : impl_(new Impl) {
    impl_->load(model);
}

Frontend::~Frontend() = default;

const PromptTokens & Frontend::tokens() const { return impl_->tokens; }

std::vector<int32_t> Frontend::encode(const std::string & text) const {
    return impl_->encode(text);
}

std::vector<DelayRow> Frontend::build_prompt(const DelayConfig & config, const std::string & text,
                                             const std::string & language,
                                             const std::vector<int32_t> & reference_codes,
                                             int reference_frames) const {
    return build_prompt_rows(config, impl_->tokens,
            [this](const std::string & span) { return impl_->encode(span); },
            text, language, reference_codes, reference_frames);
}

} // namespace tts_cpp::moss::detail
