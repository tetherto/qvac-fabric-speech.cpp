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
        {
            using namespace tts_cpp::pocket::detail;
            OpenCLScratch pools;
            // Generic F32 GEMM may need two independently retained copies.
            PocketGraph g(backend);
            auto * a_storage = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, 64, 64);
            auto * a = ggml_view_2d(g.ctx, a_storage, 64, 32, a_storage->nb[1]*2, 0);
            auto * storage = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, 64, 32);
            auto * b = ggml_view_2d(g.ctx, storage, 64, 16, storage->nb[1]*2, 0);
            auto * output = ggml_mul_mat(g.ctx, a, b);
            g.prepare(output, true);
            measure_opencl_scratch(g.graph, pools);
            if (pools.src0 < ggml_nbytes(a) || pools.src1 < ggml_nbytes(b))
                throw std::runtime_error("OpenCL strided GEMM scratch omitted");
            // A later graph adds a distinct persistent transpose pool. It
            // cannot replace the earlier graph's src0/src1 pool maxima.
            PocketGraph image(backend);
            auto * w = ggml_new_tensor_2d(image.ctx, GGML_TYPE_F32, 64, 128);
            auto * x = ggml_new_tensor_2d(image.ctx, GGML_TYPE_F32, 64, 16);
            image.prepare(ggml_mul_mat(image.ctx, w, x), true);
            OpenCLScratch transposes;
            measure_opencl_scratch(image.graph, transposes);
            const auto bound = opencl_scratch_bound(pools, transposes);
            if (bound < opencl_init_bytes + pools.src0 + pools.src1 + ggml_nbytes(w))
                throw std::runtime_error("OpenCL independently retained pools omitted");
            if (a->data || b->data || output->data || w->data || x->data)
                throw std::runtime_error("OpenCL scratch pricing allocated model/graph tensors");
        }
        {
            using namespace tts_cpp::pocket::detail;
            HexagonRuntime runtime;
            runtime.tensor_guard_bytes = 4096; runtime.allocation_page_bytes = 16384;
            runtime.max_buffer_bytes = 8192;
            if (hexagon_buffer_overhead(runtime) < 4096 + 16383)
                throw std::runtime_error("Hexagon guard/page rounding omitted");
            PocketGraph g(backend);
            auto * exact = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, 2048);
            validate_hexagon_tensor(exact, runtime);
            auto * oversized = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, 2049);
            bool rejected = false;
            try { validate_hexagon_tensor(oversized, runtime); }
            catch (const std::runtime_error &) { rejected = true; }
            if (!rejected) throw std::runtime_error("Hexagon oversized RPC tensor accepted");
            if (exact->data || oversized->data) throw std::runtime_error("Hexagon limit check allocated tensors");
        }
        {
            using namespace tts_cpp::pocket::detail;
            constexpr uint64_t mib = 1024 * 1024;
            HexagonRuntime runtime;
            runtime.max_buffer_bytes = 2048 * mib - 4096;
            PocketGraph g(backend);
            // Two individually valid 1.125 GiB tensors can share a 2.25 GiB
            // allocation when MBUF=3072, above the 2 GiB minus guard cap.
            auto * a = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, 1152 * mib / sizeof(float));
            auto * b = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, 1152 * mib / sizeof(float));
            validate_hexagon_tensor(a, runtime);
            validate_hexagon_tensor(b, runtime);
            for (const auto limit : {2048 * mib, 3072 * mib}) {
                bool rejected = false;
                try { validate_hexagon_packing_limit(limit, runtime.max_buffer_bytes); }
                catch (const BackendUnavailable & error) {
                    rejected = std::string(error.what()).find("GGML_HEXAGON_MBUF") != std::string::npos;
                }
                if (!rejected) throw std::runtime_error("Hexagon unsafe packed buffer configuration accepted");
            }
            // The default, the largest whole-MiB safe setting, and the exact
            // effective payload boundary remain valid.
            for (const auto limit : {1024 * mib, 2047 * mib, runtime.max_buffer_bytes})
                validate_hexagon_packing_limit(limit, runtime.max_buffer_bytes);
            if (a->data || b->data) throw std::runtime_error("Hexagon packing check allocated payloads");
        }
        ggml_backend_free(backend);
        std::cout << "Pocket graph memory: metadata-only CPU scratch included\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
