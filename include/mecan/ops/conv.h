#pragma once
#include "mecan/tensor.h"
#include <cstddef>

namespace mecan {
namespace ops {

    /**
     * Conv2D: 2D Convolution using im2col + OpenBLAS SGEMM
     * This is the exact same algorithm PyTorch/cuDNN uses:
     *   1. Unroll the input patches into a column matrix (im2col)
     *   2. Multiply the column matrix by the filter matrix (SGEMM)
     *   3. Reshape to output tensor
     * 
     * Input:  [N, C_in,  H,  W]  stored as contiguous float array
     * Filter: [C_out, C_in, kH, kW]
     * Output: [N, C_out, H_out, W_out]
     * 
     * H_out = (H + 2*pad - kH) / stride + 1
     * W_out = (W + 2*pad - kW) / stride + 1
     */
    void conv2d(
        const float* input,   size_t N, size_t C_in,  size_t H,  size_t W,
        const float* filter,  size_t C_out, size_t kH, size_t kW,
        float* output,
        int stride = 1, int pad = 0
    );

    /**
     * Conv3D: 3D Convolution using vol2col + OpenBLAS SGEMM
     * For volumetric data: MRI scans, 3D point clouds, video frames.
     * 
     * Input:  [N, C_in, D, H, W]
     * Filter: [C_out, C_in, kD, kH, kW]
     * Output: [N, C_out, D_out, H_out, W_out]
     */
    void conv3d(
        const float* input,   size_t N, size_t C_in,  size_t D, size_t H, size_t W,
        const float* filter,  size_t C_out, size_t kD, size_t kH, size_t kW,
        float* output,
        int stride = 1, int pad = 0
    );

    void conv1d(
        const float* input,   size_t N, size_t C_in,  size_t L,
        const float* filter,  size_t C_out, size_t kL,
        float* output,
        int stride = 1, int pad = 0
    );

    void conv_transpose2d(
        const float* input,   size_t N, size_t C_in,  size_t H,  size_t W,
        const float* filter,  size_t C_out, size_t kH, size_t kW,
        float* output,
        int stride = 1, int pad = 0, int output_padding = 0
    );

    void depthwise_conv2d(
        const float* input,   size_t N, size_t C_in,  size_t H,  size_t W,
        const float* filter,  size_t kH, size_t kW,
        float* output,
        int stride = 1, int pad = 0
    );

} // namespace ops
} // namespace mecan
