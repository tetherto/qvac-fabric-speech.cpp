// test-nemotron-diarization-coreml-source: a generated diarizer is prepared
// only from the file whose weights the model loaded. The model path names A
// before and after the load, but B (another file with the same contents and
// shapes) at the moment the GGUF is opened, so the model holds B's weights.
// Two stats of the path around the load both see A; the identity of the one
// descriptor the weights were read through is B's. The diarizers must not
// generate from A: the prepare job fails with source_changed and the cache
// holds no artifact. Loaded again with nothing swapped, the minimal capacity
// is generated and ready.
//
// Usage: test-nemotron-diarization-coreml-source <Nemotron q8_0 GGUF> <work dir>
// Never skips: every check needs Core ML generation (macOS 15, the generator
// built in).

#include "parakeet_ctc.h"

#include <cstdio>
#include <filesystem>
#include <string>
#include <system_error>

namespace fs = std::filesystem;

namespace {

constexpr int kThreads = 4;
constexpr int kGpuLayers = 99;

bool g_passed = true;
fs::path g_dir;

void check(bool condition, const char * label) {
    std::printf("[%s] %s\n", condition ? "ok  " : "FAIL", label);
    g_passed = g_passed && condition;
}

// "open": the path names B; "read": the path names A again.
void swap_around_open(const char * point) {
    std::error_code ec;
    if (std::string(point) == "open") {
        fs::rename(g_dir / "m.gguf", g_dir / "a.gguf", ec);
        fs::rename(g_dir / "b.gguf", g_dir / "m.gguf", ec);
    } else {
        fs::rename(g_dir / "m.gguf", g_dir / "b.gguf", ec);
        fs::rename(g_dir / "a.gguf", g_dir / "m.gguf", ec);
    }
    if (ec) std::printf("swap at %s: %s\n", point, ec.message().c_str());
}

std::string load_status(const fs::path & cache) {
    parakeet::CoremlOptions options;
    options.cache_dir = cache.string();
    options.wait = parakeet::CoremlOptions::Wait::construct;
    parakeet::ParakeetCtcModel model;
    const int rc = parakeet::load_from_gguf((g_dir / "m.gguf").string(), model, kThreads, kGpuLayers, false, "auto",
                                            options);
    check(rc == 0, "the model loads");
    return rc == 0 ? parakeet::model_coreml_status_json(model) : std::string();
}

bool has_artifacts(const fs::path & cache) {
    std::error_code ec;
    const fs::path artifacts = cache / "qvac-coreml-v1" / "artifacts";
    return fs::exists(artifacts, ec) && !fs::is_empty(artifacts, ec);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <work dir>\n", argv[0]);
        return 2;
    }
    g_dir = argv[2];
    std::error_code ec;
    fs::remove_all(g_dir, ec);
    fs::create_directories(g_dir);
    fs::copy_file(argv[1], g_dir / "m.gguf", ec);
    fs::copy_file(argv[1], g_dir / "b.gguf", ec);
    if (ec) {
        std::fprintf(stderr, "cannot stage the GGUF: %s\n", ec.message().c_str());
        return 1;
    }

    parakeet::set_gguf_load_observer(&swap_around_open);
    const std::string swapped = load_status(g_dir / "cache-swapped");
    parakeet::set_gguf_load_observer(nullptr);
    std::printf("swapped: %s\n", swapped.c_str());
    check(swapped.find("source_changed") != std::string::npos,
          "the path named another file when the weights were opened: source_changed");
    check(swapped.find("\"state\":\"ready\"") == std::string::npos, "no generated diarizer is ready");
    check(!has_artifacts(g_dir / "cache-swapped"), "nothing is generated from the file the path names now");

    const std::string plain = load_status(g_dir / "cache-plain");
    std::printf("plain: %s\n", plain.c_str());
    check(plain.find("source_changed") == std::string::npos && plain.find("\"state\":\"ready\"") != std::string::npos &&
              has_artifacts(g_dir / "cache-plain"),
          "nothing swapped: the minimal capacity is generated and ready");

    fs::remove_all(g_dir, ec);
    std::printf("%s\n", g_passed ? "PASS" : "FAIL");
    return g_passed ? 0 : 1;
}
