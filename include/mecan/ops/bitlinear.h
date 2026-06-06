#pragma once

#include "tensor.h"

namespace mecan {
namespace ops {

    /**
     * Specialized Ternary (1.58-bit) Kernels
     * --------------------------------------
     * Optimized for 50B+ throughput on limited memory.
     * Uses simple addition/subtraction instead of multiplication
     * since weights are only {-1, 0, 1}.
     */
    
    void bitlinear_forward(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output);

} // namespace ops
} // namespace mecan
