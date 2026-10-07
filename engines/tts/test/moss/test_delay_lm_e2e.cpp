// Full-checkpoint CPU/Vulkan logits with identical inputs. CI only: release
// CPU weights before loading Vulkan, and do not load the decoder at all.
#include "moss/delay_lm.h"
#include "moss/frontend.h"
#include "moss/generation.h"
#include "../test_env_portable.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>

using namespace tts_cpp::moss::detail;
namespace fs = std::filesystem;

namespace {
void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

bool compare(std::ofstream & csv, size_t step, int head,
             const std::vector<float> & cpu, const std::vector<float> & gpu) {
    require(!cpu.empty() && cpu.size() == gpu.size(), "logit shape mismatch");
    double error = 0, norm = 0, worst = 0;
    for (size_t i = 0; i < cpu.size(); ++i) {
        require(std::isfinite(cpu[i]) && std::isfinite(gpu[i]), "non-finite logits");
        const double delta = (double) cpu[i] - gpu[i];
        error += delta * delta;
        norm += (double) cpu[i] * cpu[i];
        worst = std::max(worst, std::fabs(delta));
    }
    const double relative = std::sqrt(error / std::max(norm, 1e-20));
    const auto cpu_top = std::max_element(cpu.begin(), cpu.end()) - cpu.begin();
    const auto gpu_top = std::max_element(gpu.begin(), gpu.end()) - gpu.begin();
    csv << step << ',' << head << ',' << worst << ',' << relative << ','
        << cpu_top << ',' << gpu_top << '\n';
    return worst < 0.002 || relative < 0.01;
}
} // namespace

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s LM.gguf TEXT.txt OUTPUT_DIR\n", argv[0]);
        return 77;
    }
    try {
        setenv("TTS_CPP_GPU_BACKEND", "vulkan", 1);
        const fs::path output(argv[3]);
        fs::create_directories(output);
        std::ifstream input(argv[2]);
        require(input.good(), "cannot read prompt");
        std::string text((std::istreambuf_iterator<char>(input)), {});
        while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) text.pop_back();
        require(!text.empty(), "empty prompt");

        std::vector<DelayRow> prompt, trajectory;
        std::vector<DelayLogits> expected;
        {
            DelayLM cpu(argv[1], false, 4, 4096);
            require(std::string(cpu.backend_name()) == "CPU", "CPU reference backend required");
            Frontend frontend(cpu);
            prompt = frontend.build_prompt(cpu.config(), text, "en", 0, {});
            std::ofstream ids(output / "prompt-ids.txt");
            require(ids.good(), "cannot write prompt IDs");
            for (const auto & row : prompt) ids << row.text << '\n';
            DelayState state(cpu.config(), prompt, frontend.tokens().pad, frontend.tokens().im_end);
            SamplingConfig sampling;
            std::mt19937 rng(1234);
            expected.push_back(cpu.prefill(prompt));
            for (int step = 0; step < 32; ++step) {
                trajectory.push_back(state.step(expected.back(), sampling, rng));
                expected.push_back(cpu.step(trajectory.back()));
            }
        }
        DelayLM gpu(argv[1], true, 4, 4096);
        require(std::string(gpu.backend_name()).find("Vulkan") == 0, "Vulkan backend required");
        std::fprintf(stderr, "[moss-delay-lm-e2e] backend: %s\n", gpu.backend_name());
        std::ofstream csv(output / "logits.csv");
        require(csv.good(), "cannot write logits report");
        csv << "step,head,max_abs,relative_l2,cpu_argmax,vulkan_argmax\n";
        size_t failures = 0;
        for (size_t step = 0; step < expected.size(); ++step) {
            const auto actual = step == 0 ? gpu.prefill(prompt) : gpu.step(trajectory[step - 1]);
            const auto & reference = expected[step];
            failures += !compare(csv, step, -1, reference.text, actual.text);
            require(reference.audio.size() == actual.audio.size(), "audio head count mismatch");
            for (size_t head = 0; head < reference.audio.size(); ++head) {
                failures += !compare(csv, step, (int) head, reference.audio[head], actual.audio[head]);
            }
        }
        csv.flush();
        require(csv.good(), "cannot flush logits report");
        std::fprintf(stderr, "MOSS delay LM: %zu comparisons outside tolerance\n", failures);
        require(failures == 0, "full-checkpoint CPU/Vulkan logits diverged; see logits.csv");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "moss delay lm e2e: %s\n", error.what());
        return 1;
    }
}
