#include "pocket/ggml_utils.h"
#include <iostream>

int main() {
    try {
        ggml_backend_load_all();
        auto backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!backend) throw std::runtime_error("CPU backend missing");
        {
            tts_cpp::pocket::detail::PocketGraph g(backend);
            auto * a = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F16, 64, 32);
            auto * b = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, 64, 16);
            auto * output = ggml_mul_mat(g.ctx, a, b);
            const auto bytes = g.prepare(output, true, 4);
            size_t arena = 0;
            ggml_gallocr_reserve_n_size(g.allocator, g.graph, nullptr, nullptr, &arena);
            if (bytes <= arena) throw std::runtime_error("CPU scratch omitted");
            if (a->data || b->data || output->data) throw std::runtime_error("preflight allocated tensors");
            const auto repeated = tts_cpp::pocket::detail::price_cpu_graph(backend, g.allocator, g.graph, 4);
            if (repeated != bytes) throw std::runtime_error("preflight changed on repeat");
        }
        ggml_backend_free(backend);
        std::cout << "Pocket graph memory: metadata-only CPU scratch included\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
