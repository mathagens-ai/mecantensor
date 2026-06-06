#pragma once
// MecanTensor: Normalization Operations (BatchNorm, LayerNorm, InstanceNorm)
#include <cstddef>
#include <cstdint>

namespace mecan {
namespace ops {

    // BatchNorm2d: normalize across batch dimension per channel
    // input: NCHW, gamma/beta: C, running_mean/var: C
    void batch_norm2d(
        const float* input, float* output,
        size_t N, size_t C, size_t H, size_t W,
        const float* gamma, const float* beta,
        const float* running_mean, const float* running_var,
        float epsilon, bool training);

    // LayerNorm: normalize across last D dimensions
    void layer_norm(
        const float* input, float* output,
        size_t outer, size_t norm_size,
        const float* gamma, const float* beta,
        float epsilon);

    // InstanceNorm2d: normalize each sample+channel independently
    void instance_norm2d(
        const float* input, float* output,
        size_t N, size_t C, size_t H, size_t W,
        const float* gamma, const float* beta,
        float epsilon);

    // GroupNorm: normalize across channel groups
    void group_norm(
        const float* input, float* output,
        size_t N, size_t C, size_t H, size_t W,
        int num_groups,
        const float* gamma, const float* beta,
        float epsilon);

} // namespace ops
} // namespace mecan
