#include "coreml-models.h"
#include "mm3-model.h"

#include <cstdio>
#include <filesystem>

namespace tts_cpp::minimax::detail {
namespace {

template<typename Layers>
size_t attach_layers(Layers & layers, const std::string & root, const std::string & fingerprint,
                     CoreMLBlockConfig config) {
    size_t loaded = 0;
    for (size_t i = 0; i < layers.size(); ++i) {
        config.layer = static_cast<int64_t>(i);
        const auto path = root + "/" + config.stage + "-block-" + std::to_string(i) + ".mlmodelc";
        layers[i].coreml = load_coreml_block(path, fingerprint, config);
        if (layers[i].coreml) ++loaded;
    }
    return loaded;
}

template<typename Layers>
uint64_t prediction_count(const Layers & layers) {
    uint64_t count = 0;
    for (const auto & layer : layers) count += coreml_block_predictions(layer.coreml);
    return count;
}

std::string source_fingerprint(const std::string & path) {
    std::error_code error;
    if (!std::filesystem::is_directory(path + "-coreml", error)) return {};
    return coreml_source_fingerprint(path);
}

}

void attach_coreml_blocks(MM3Model & model) {
    const auto lm_hash = source_fingerprint(model.lm_file.path);
    const auto synth_hash = source_fingerprint(model.synth_file.path);
    const auto & lm = model.lm_cfg;
    const auto & depth = model.synth_cfg.depth;
    const auto & dit = model.synth_cfg.dit;
    const size_t count = attach_layers(model.lm.blk, model.lm_file.path + "-coreml", lm_hash,
        {"lm", 0, lm.embedding_length, lm.head_count, lm.head_count_kv, lm.key_length}) +
        attach_layers(model.synth.depth.blk, model.synth_file.path + "-coreml", synth_hash,
        {"depth", 0, depth.embedding_length, depth.head_count, depth.head_count, depth.head_dim}) +
        attach_layers(model.synth.dit.blk, model.synth_file.path + "-coreml", synth_hash,
        {"dit", 0, dit.embedding_length, dit.head_count, dit.head_count, dit.head_dim});
    if (count) std::fprintf(stderr, "[MM3-CoreML] %zu resident transformer blocks; compute policy: CPU and ANE\n", count);
}

void report_coreml_predictions(const MM3Model & model) {
    const auto lm = prediction_count(model.lm.blk);
    const auto depth = prediction_count(model.synth.depth.blk);
    const auto dit = prediction_count(model.synth.dit.blk);
    if (lm || depth || dit) std::fprintf(stderr, "[MM3-CoreML] Resident-model prediction totals: LM=%llu depth=%llu DiT=%llu\n",
        static_cast<unsigned long long>(lm), static_cast<unsigned long long>(depth), static_cast<unsigned long long>(dit));
}

}
