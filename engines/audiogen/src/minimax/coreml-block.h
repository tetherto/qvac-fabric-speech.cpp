#pragma once

#include "ggml.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tts_cpp::minimax::detail {

struct CoreMLBlock;

struct CoreMLBlockConfig {
    std::string stage;
    int64_t layer = 0;
    int64_t hidden = 0;
    int64_t heads = 0;
    int64_t kv_heads = 0;
    int64_t head_dim = 0;
};

struct CoreMLArray {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<float> values;
};

std::string coreml_source_fingerprint(const std::string & path);
std::shared_ptr<CoreMLBlock> load_coreml_block(const std::string & path, const std::string & fingerprint,
                                            const CoreMLBlockConfig & config);
bool coreml_block_supports(const std::shared_ptr<CoreMLBlock> & block, int64_t tokens, int64_t cache, int64_t batch);
bool coreml_block_predict(CoreMLBlock & block, const std::vector<CoreMLArray> & inputs,
                          std::vector<CoreMLArray> & outputs);
void coreml_block_set_error(CoreMLBlock & block, const std::string & error);
std::string coreml_block_error(const std::shared_ptr<CoreMLBlock> & block);
uint64_t coreml_block_predictions(const std::shared_ptr<CoreMLBlock> & block);

ggml_tensor * coreml_transformer_block(ggml_context * ctx, ggml_cgraph * graph,
                                      const std::shared_ptr<CoreMLBlock> & block,
                                      ggml_tensor * hidden, ggml_tensor * positions, ggml_tensor * mask,
                                      ggml_tensor * keys, ggml_tensor * values, ggml_tensor * rows,
                                      int64_t cache_length, int64_t heads, int64_t head_dim, bool lm_layout);

}
