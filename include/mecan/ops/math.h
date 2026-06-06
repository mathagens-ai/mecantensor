#pragma once

#include "tensor.h"

namespace mecan {
namespace ops {

    /**
     * TST optimized math operations.
     * These route through SIMD kernels based on the device.
     */

    void add(const Tensor& a, const Tensor& b, Tensor& out);
    void sub(const Tensor& a, const Tensor& b, Tensor& out);
    void mul(const Tensor& a, const Tensor& b, Tensor& out);
    
    // Matrix Multiplication (The high-performance core)
    void matmul(const Tensor& a, const Tensor& b, Tensor& out);

    // TSSR specific Ternary scaling
    void ternary_threshold(const Tensor& input, Tensor& output, float threshold = 0.5f);

} // namespace ops
} // namespace mecan
