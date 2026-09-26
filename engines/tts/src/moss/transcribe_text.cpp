#include "moss/transcribe_text.h"

#include "qwen_tokenizer.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <unordered_map>

namespace tts_cpp::moss::detail {
namespace {

constexpr int32_t TOKEN_TYPE_CONTROL = 3;
constexpr size_t MAX_TIMESTAMP_CHARS = 32;
constexpr size_t MAX_SPEAKER_CHARS = 16;
constexpr const char * WHITESPACE = " \t\n\v\f\r";

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("moss transcribe: " + message);
}

bool is_space(char ch) {
    return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\v' || ch == '\f' || ch == '\r';
}

bool is_digit(char ch) {
    return ch >= '0' && ch <= '9';
}

bool is_timestamp_char(char ch) {
    return is_digit(ch) || ch == '.';
}

bool is_speaker_char(char ch) {
    return ch == 'S' || is_digit(ch);
}

size_t count_char(const std::string & text, char wanted) {
    size_t count = 0;
    for (char ch : text) {
        count += ch == wanted ? 1 : 0;
    }
    return count;
}

bool all_timestamp_chars(const std::string & text) {
    for (char ch : text) {
        if (!is_timestamp_char(ch)) {
            return false;
        }
    }
    return true;
}

bool all_digits(const std::string & text) {
    for (char ch : text) {
        if (!is_digit(ch)) {
            return false;
        }
    }
    return true;
}

bool parse_timestamp(const std::string & text, double & value) {
    if (text.empty() || !all_timestamp_chars(text) || count_char(text, '.') > 1 || count_char(text, '.') == text.size()) {
        return false;
    }
    value = std::strtod(text.c_str(), nullptr);
    return true;
}

bool is_speaker(const std::string & text) {
    return text.size() >= 2 && text[0] == 'S' && all_digits(text.substr(1));
}

size_t utf8_length(unsigned char lead) {
    if (lead < 0x80) return 1;
    if ((lead >> 5) == 0x6) return 2;
    if ((lead >> 4) == 0xE) return 3;
    if ((lead >> 3) == 0x1E) return 4;
    return 1;
}

using ByteMap = std::unordered_map<std::string, char>;

ByteMap reverse_byte_map(const QwenTokenizer & tokenizer) {
    ByteMap map;
    for (int byte = 0; byte < 256; ++byte) {
        map.emplace(tokenizer.byte2u[byte], (char) byte);
    }
    return map;
}

std::string bytes_of(const std::string & token, const ByteMap & map) {
    std::string bytes;
    for (size_t i = 0; i < token.size();) {
        const size_t length = std::min(utf8_length((unsigned char) token[i]), token.size() - i);
        const std::string symbol = token.substr(i, length);
        const auto found = map.find(symbol);
        bytes += found != map.end() ? std::string(1, found->second) : symbol;
        i += length;
    }
    return bytes;
}

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

void append(std::vector<int32_t> & ids, const std::vector<int32_t> & more) {
    ids.insert(ids.end(), more.begin(), more.end());
}

void append_pads(std::vector<int32_t> & span, int32_t pad, int count) {
    span.insert(span.end(), (size_t) std::max(count, 0), pad);
}

std::vector<int32_t> digit_ids(const TranscribeTokenizer & tokenizer) {
    std::vector<int32_t> ids;
    for (char digit = '0'; digit <= '9'; ++digit) {
        const std::vector<int32_t> encoded = tokenizer.encode(std::string(1, digit));
        if (encoded.size() != 1) {
            fail("digit is not a single token in the vocabulary");
        }
        ids.push_back(encoded[0]);
    }
    return ids;
}

void append_marker(std::vector<int32_t> & span, int seconds, const std::vector<int32_t> & digits) {
    for (char ch : std::to_string(seconds)) {
        span.push_back(digits[(size_t) (ch - '0')]);
    }
}

int append_marked_pads(std::vector<int32_t> & span, int32_t pad, int every, int per_marker, int duration,
                       const std::vector<int32_t> & digits) {
    int consumed = 0;
    for (int seconds = every; seconds <= duration; seconds += every) {
        const int position = (seconds / every) * per_marker;
        append_pads(span, pad, position - consumed);
        consumed = std::max(consumed, position);
        append_marker(span, seconds, digits);
    }
    return consumed;
}

std::vector<int32_t> prompt_body(const TranscribeConfig & config, const TranscribeTokenizer & tokenizer,
                                 const std::string & prompt) {
    if (!strip_whitespace(prompt).empty()) {
        return tokenizer.encode("\n" + prompt);
    }
    std::vector<int32_t> body = tokenizer.encode("\n");
    append(body, config.default_prompt_ids);
    return body;
}

} // namespace

struct TranscribeTokenizer::Impl {
    mutable QwenTokenizer tokenizer;
    std::vector<std::string> pieces;

    void build_pieces(const std::vector<std::string> & tokens, const std::vector<int32_t> & types) {
        const ByteMap map = reverse_byte_map(tokenizer);
        pieces.resize(tokens.size());
        for (size_t id = 0; id < tokens.size(); ++id) {
            pieces[id] = piece(tokens[id], id < types.size() ? types[id] : 0, map);
        }
    }

    static std::string piece(const std::string & token, int32_t type, const ByteMap & map) {
        if (type == TOKEN_TYPE_CONTROL) {
            return {};
        }
        return type == 1 || type == 0 ? bytes_of(token, map) : token;
    }
};

TranscribeTokenizer::TranscribeTokenizer(const std::vector<std::string> & tokens,
                                         const std::vector<std::string> & merges,
                                         const std::vector<int32_t> & types) : impl_(new Impl) {
    if (tokens.empty() || merges.empty()) {
        fail("GGUF is missing the tokenizer vocabulary");
    }
    impl_->tokenizer.build_byte_map();
    load_vocabulary(impl_->tokenizer, tokens);
    load_merges(impl_->tokenizer, merges);
    impl_->build_pieces(tokens, types);
}

TranscribeTokenizer::TranscribeTokenizer(const TranscribeModel & model)
    : TranscribeTokenizer(model.tokenizer_tokens(), model.tokenizer_merges(), model.tokenizer_types()) {}

TranscribeTokenizer::~TranscribeTokenizer() = default;

std::vector<int32_t> TranscribeTokenizer::encode(const std::string & text) const {
    const std::vector<int> ids = impl_->tokenizer.encode(text);
    return std::vector<int32_t>(ids.begin(), ids.end());
}

std::string TranscribeTokenizer::decode(const std::vector<int32_t> & ids) const {
    std::string text;
    for (int32_t id : ids) {
        if (id >= 0 && (size_t) id < impl_->pieces.size()) {
            text += impl_->pieces[(size_t) id];
        }
    }
    return text;
}

std::vector<int32_t> transcribe_audio_span(const TranscribeConfig & config, const TranscribeTokenizer & tokenizer,
                                           int audio_tokens) {
    std::vector<int32_t> span;
    const int32_t pad = config.tokens.audio_pad;
    const int every = config.time_marker_every_seconds;
    const int per_marker = (int) (config.audio_tokens_per_second * (float) every);
    if (!config.time_markers || audio_tokens <= 0 || every <= 0 || per_marker <= 0) {
        append_pads(span, pad, audio_tokens);
        return span;
    }
    const int duration = (int) ((double) audio_tokens / (double) config.audio_tokens_per_second);
    const int consumed = append_marked_pads(span, pad, every, per_marker, duration, digit_ids(tokenizer));
    append_pads(span, pad, audio_tokens - consumed);
    return span;
}

std::vector<int32_t> transcribe_prompt(const TranscribeConfig & config, const TranscribeTokenizer & tokenizer,
                                       int audio_tokens, const std::string & prompt) {
    const TranscribeTokens & tokens = config.tokens;
    std::vector<int32_t> ids{tokens.im_start};
    append(ids, tokenizer.encode("system\n" + config.system_prompt));
    ids.push_back(tokens.im_end);
    append(ids, tokenizer.encode("\n"));
    ids.push_back(tokens.im_start);
    append(ids, tokenizer.encode("user\n"));
    ids.push_back(tokens.audio_start);
    append(ids, transcribe_audio_span(config, tokenizer, audio_tokens));
    ids.push_back(tokens.audio_end);
    append(ids, prompt_body(config, tokenizer, prompt));
    ids.push_back(tokens.im_end);
    append(ids, tokenizer.encode("\n"));
    ids.push_back(tokens.im_start);
    append(ids, tokenizer.encode("assistant\n"));
    return ids;
}

std::string strip_whitespace(const std::string & text) {
    const size_t first = text.find_first_not_of(WHITESPACE);
    if (first == std::string::npos) {
        return {};
    }
    return text.substr(first, text.find_last_not_of(WHITESPACE) - first + 1);
}

void TranscriptParser::reset() {
    state_ = State::SeekStart;
    token_.clear();
    text_.clear();
    pending_after_end_.clear();
    end_token_.clear();
    speaker_.clear();
    has_start_ = false;
    has_end_ = false;
}

void TranscriptParser::restart_on_bracket(char ch) {
    reset();
    if (ch == '[') {
        state_ = State::ReadStart;
    }
}

void TranscriptParser::seek_start(char ch) {
    if (ch == '[') {
        token_.clear();
        state_ = State::ReadStart;
    }
}

void TranscriptParser::read_start(char ch) {
    if (ch == ']') {
        double start = 0;
        if (!parse_timestamp(token_, start)) {
            reset();
            return;
        }
        start_ = start;
        has_start_ = true;
        state_ = State::ExpectSpeakerOpen;
        token_.clear();
        return;
    }
    if (is_timestamp_char(ch)) {
        token_.push_back(ch);
        if (token_.size() <= MAX_TIMESTAMP_CHARS) {
            return;
        }
    }
    restart_on_bracket(ch);
}

void TranscriptParser::expect_speaker_open(char ch) {
    if (ch == '[') {
        token_.clear();
        state_ = State::ReadSpeaker;
    } else if (!is_space(ch)) {
        reset();
    }
}

void TranscriptParser::read_speaker(char ch) {
    if (ch == ']') {
        if (!is_speaker(token_)) {
            reset();
            return;
        }
        speaker_ = token_;
        text_.clear();
        state_ = State::ReadText;
        token_.clear();
        return;
    }
    if (is_speaker_char(ch)) {
        token_.push_back(ch);
        if (token_.size() <= MAX_SPEAKER_CHARS) {
            return;
        }
    }
    restart_on_bracket(ch);
}

void TranscriptParser::read_text(char ch) {
    if (ch == '[') {
        token_.clear();
        state_ = State::ReadEnd;
    } else {
        text_.push_back(ch);
    }
}

void TranscriptParser::accept_end() {
    end_token_ = token_;
    pending_after_end_.clear();
    has_end_ = true;
    state_ = State::AfterEnd;
}

void TranscriptParser::reject_end() {
    text_ += "[" + token_ + "]";
    state_ = State::ReadText;
}

void TranscriptParser::read_end(char ch) {
    if (ch == ']') {
        double end = 0;
        if (parse_timestamp(token_, end) && has_start_ && end >= start_) {
            end_ = end;
            accept_end();
        } else {
            reject_end();
        }
        token_.clear();
        return;
    }
    if (is_timestamp_char(ch)) {
        token_.push_back(ch);
        if (token_.size() <= MAX_TIMESTAMP_CHARS) {
            return;
        }
    }
    text_ += "[" + token_ + std::string(1, ch);
    token_.clear();
    state_ = State::ReadText;
}

void TranscriptParser::after_end(char ch, std::vector<TranscriptSegment> & segments) {
    if (ch == '[') {
        emit(segments);
        token_.clear();
        state_ = State::ReadStart;
        return;
    }
    if (is_space(ch)) {
        pending_after_end_.push_back(ch);
        return;
    }
    text_ += "[" + end_token_ + "]" + pending_after_end_ + std::string(1, ch);
    pending_after_end_.clear();
    has_end_ = false;
    end_token_.clear();
    state_ = State::ReadText;
}

void TranscriptParser::emit(std::vector<TranscriptSegment> & segments) {
    if (!has_start_ || !has_end_ || speaker_.empty()) {
        reset();
        return;
    }
    const std::string text = strip_whitespace(text_);
    if (!text.empty()) {
        segments.push_back({start_, end_, speaker_, text});
    }
    reset();
}

void TranscriptParser::consume(char ch, std::vector<TranscriptSegment> & segments) {
    switch (state_) {
        case State::SeekStart:         seek_start(ch); break;
        case State::ReadStart:         read_start(ch); break;
        case State::ExpectSpeakerOpen: expect_speaker_open(ch); break;
        case State::ReadSpeaker:       read_speaker(ch); break;
        case State::ReadText:          read_text(ch); break;
        case State::ReadEnd:           read_end(ch); break;
        case State::AfterEnd:          after_end(ch, segments); break;
    }
}

std::vector<TranscriptSegment> TranscriptParser::feed(const std::string & chunk) {
    std::vector<TranscriptSegment> segments;
    for (char ch : chunk) {
        consume(ch, segments);
    }
    return segments;
}

std::vector<TranscriptSegment> TranscriptParser::close() {
    std::vector<TranscriptSegment> segments;
    if (state_ == State::AfterEnd) {
        emit(segments);
    }
    reset();
    return segments;
}

std::vector<TranscriptSegment> parse_transcript(const std::string & text) {
    TranscriptParser parser;
    std::vector<TranscriptSegment> segments = parser.feed(text);
    const std::vector<TranscriptSegment> tail = parser.close();
    segments.insert(segments.end(), tail.begin(), tail.end());
    return segments;
}

} // namespace tts_cpp::moss::detail
