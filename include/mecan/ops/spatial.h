#pragma once
#include "mecan/tensor.h"

namespace mecan {
namespace ops {

    /**
     * Spatial Operations (Zero-Allocation)
     * ------------------------------------
     * Used for Generative upscaling (DCGAN ConvTranspose).
     * Replaces PyTorch im2col padding with math-based striding.
     */
    void zero_pad_strided_upscale(
        const Tensor& input, 
        Tensor& output, 
        int stride
    );

} // namespace ops
} // namespace mecan
