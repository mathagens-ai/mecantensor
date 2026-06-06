#pragma once
#include "mecan/tensor.h"

namespace mecan {
namespace qsbits {

    /**
     * ------------------------------------------------------
     * Hardware-Agnostic XNOR + POPCOUNT logic.
     * Bypasses ALUs completely, calculating forward passes via raw CPU/GPU gates.
     * Supports 200T parameter streaming via packed binary routing.
     */
    // input:  [B, K_packed]
    // weight: [O, K_packed]
    // output: [B, O]
    void qsbits_forward(const Tensor& input, const Tensor& binary_weights, Tensor& output);

    // Block-scaled variant: applies per-group FP32 scales to recover dynamic range.
    // scales: [O, N_groups], input_scales: [1] or scalar
    void qsbits_forward_scaled(
        const Tensor& input,
        const Tensor& binary_weights,
        const Tensor& scales,
        float input_scale,
        int group_size,
        Tensor& output);

} // namespace qsbits
} // namespace mecan
