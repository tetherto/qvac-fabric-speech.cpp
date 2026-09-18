// BPE detokenize(): merges pieces, SentencePiece space markers, strips specials.

#include "sentencepiece_bpe.h"

#include <string>

namespace parakeet {

namespace {

bool is_special_or_invalid(const BpeVocab & vocab, int32_t id) {
    if (id < 0 || id >= static_cast<int32_t>(vocab.pieces.size())) return true;
    return id == vocab.blank_id || id == vocab.bos_id || id == vocab.eos_id || id == vocab.pad_id;
}

void append_piece(const std::string & piece, std::string & out) {
    for (size_t i = 0; i < piece.size(); ) {
        const unsigned char c0 = static_cast<unsigned char>(piece[i]);
        if (c0 == 0xE2 && i + 2 < piece.size() &&
            static_cast<unsigned char>(piece[i+1]) == 0x96 &&
            static_cast<unsigned char>(piece[i+2]) == 0x81) {
            out.push_back(' ');
            i += 3;
        } else {
            out.push_back(piece[i]);
            ++i;
        }
    }
}

}

void append_token_pieces(const BpeVocab & vocab,
                         const std::vector<int32_t> & token_ids,
                         std::string & pieces) {
    for (int32_t id : token_ids) {
        if (is_special_or_invalid(vocab, id)) continue;
        append_piece(vocab.pieces[id], pieces);
    }
}

std::string strip_leading_spaces(const std::string & text) {
    size_t start = 0;
    while (start < text.size() && text[start] == ' ') ++start;
    return text.substr(start);
}

std::string detokenize(const BpeVocab & vocab,
                       const std::vector<int32_t> & token_ids) {
    std::string out;
    out.reserve(token_ids.size() * 3);
    append_token_pieces(vocab, token_ids, out);
    return strip_leading_spaces(out);
}

bool token_is_word_start(const BpeVocab & vocab, int32_t token_id) {
    if (token_id < 0 || token_id >= static_cast<int32_t>(vocab.pieces.size())) return false;
    if (token_id == vocab.blank_id || token_id == vocab.bos_id ||
        token_id == vocab.eos_id   || token_id == vocab.pad_id) return false;
    const std::string & piece = vocab.pieces[token_id];
    if (piece.size() < 3) return false;
    const unsigned char c0 = static_cast<unsigned char>(piece[0]);
    const unsigned char c1 = static_cast<unsigned char>(piece[1]);
    const unsigned char c2 = static_cast<unsigned char>(piece[2]);
    return c0 == 0xE2 && c1 == 0x96 && c2 == 0x81;
}

}
