#pragma once
// MecanTensor: Pooling Operations (MaxPool, AvgPool, Adaptive)
#include <cstddef>
#include <cstdint>

namespace mecan {
namespace ops {

    // MaxPool2d: standard sliding-window maximum
    void max_pool2d(
        const float* input, float* output,
        size_t N, size_t C, size_t H, size_t W,
        int kH, int kW, int stride, int pad);

    // AvgPool2d: standard sliding-window average
    void avg_pool2d(
        const float* input, float* output,
        size_t N, size_t C, size_t H, size_t W,
        int kH, int kW, int stride, int pad);

    // AdaptiveAvgPool2d: automatically computes kernel/stride to reach target size
    void adaptive_avg_pool2d(
        const float* input, float* output,
        size_t N, size_t C, size_t H, size_t W,
        size_t out_H, size_t out_W);

    // AdaptiveMaxPool2d
    void adaptive_max_pool2d(
        const float* input, float* output,
        size_t N, size_t C, size_t H, size_t W,
        size_t out_H, size_t out_W);

} // namespace ops
} // namespace mecan
