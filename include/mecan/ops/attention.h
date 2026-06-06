#pragma once
#include "mecan/tensor.h"

namespace mecan {
namespace ops {

    /**
     * Fused Ternary Attention
     * -----------------------
     * Bypasses the PyTorch O(N^2) memory bottleneck.
     * Computes the Q * K^T * V attention blending entirely in CPU cache
     * block-by-block without ever allocating the full N x N matrix in RAM.
     * Optimized for SSD-backed KV caching.
     */
    void flash_ternary_attention(
        const Tensor& query, 
        const Tensor& key_cache, 
        const Tensor& value_cache, 
        Tensor& output
    );

} // namespace ops
} // namespace mecan
