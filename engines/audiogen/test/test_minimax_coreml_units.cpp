#include "minimax/coreml-block.h"

#include <cstdio>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace {

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

void missing_sidecars_use_native_blocks() {
    using namespace tts_cpp::minimax::detail;
    const CoreMLBlockConfig config{"lm", 0, 32, 4, 2, 8};
    auto block = load_coreml_block("/nonexistent/minimax-block.mlmodelc", "missing", config);
    require(!block, "missing sidecar must fall back");
    require(!coreml_block_supports(block, 1, 256, 2), "missing sidecar must not claim support");
    require(coreml_block_error(block).empty(), "native fallback must not report a Core ML failure");
    require(coreml_block_predictions(block) == 0, "native fallback must not report Core ML predictions");
    auto * ctx = ggml_init({ggml_tensor_overhead() * 4, nullptr, true});
    require(ctx != nullptr, "failed to initialize test context");
    auto * hidden = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 1, 2);
    auto * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    auto * result = coreml_transformer_block(ctx, nullptr, block, hidden, positions,
                                            nullptr, nullptr, nullptr, nullptr, 0, 4, 8, false);
    ggml_free(ctx);
    require(result == nullptr, "missing block must preserve native graph construction");
}

#ifdef AUDIOGEN_USE_COREML
void fingerprints_bind_sidecars_to_exact_source_bytes() {
    using tts_cpp::minimax::detail::coreml_source_fingerprint;
    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto directory = std::filesystem::temp_directory_path() / ("minimax-hash-" + std::to_string(unique));
    require(std::filesystem::create_directory(directory), "failed to create fingerprint fixture directory");
    const auto path = directory / "source.gguf";
    {
        std::ofstream file(path, std::ios::binary);
        file << "abc";
    }
    const auto original = coreml_source_fingerprint(path.string());
    {
        std::ofstream file(path, std::ios::binary);
        file << "abd";
    }
    const auto changed = coreml_source_fingerprint(path.string());
    std::filesystem::remove_all(directory);
    require(original == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "incorrect source SHA256");
    require(original != changed, "different weights must invalidate sidecars");
    require(coreml_source_fingerprint(path.string()).empty(), "unreadable source must fail closed");
}
#endif

}

int main() {
    try {
        missing_sidecars_use_native_blocks();
#ifdef AUDIOGEN_USE_COREML
        fingerprints_bind_sidecars_to_exact_source_bytes();
#endif
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL %s\n", error.what());
        return 1;
    }
}
