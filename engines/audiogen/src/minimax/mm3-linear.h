#pragma once

#include "ggml.h"

#include <cstdint>

// Applies a weight matrix to every column of `x` in one matrix product. The
// CFG pair rides in ne[2] of the MiniMax activations; handing that layout to
// ggml_mul_mat as a broadcast batch makes the GPU matrix-vector kernels stream
// the weights once per batch entry, so the columns are folded into ne[1]
// first and the original shape is restored afterwards. `x` must be contiguous.
// GGML_PREC_F32 keeps the activations and the accumulation in f32, which the
// matrix-vector kernels do at no cost; large products leave it at the default
// so they keep the half-precision tensor-core GEMMs.
static ggml_tensor * mm3_linear(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * x,
                                ggml_prec precision = GGML_PREC_DEFAULT) {
    const int64_t columns = ggml_nelements(x) / x->ne[0];
    ggml_tensor * y       = ggml_mul_mat(ctx, weight, ggml_reshape_2d(ctx, x, x->ne[0], columns));
    ggml_mul_mat_set_prec(y, precision);
    return ggml_reshape_4d(ctx, y, weight->ne[1], x->ne[1], x->ne[2], x->ne[3]);
}
