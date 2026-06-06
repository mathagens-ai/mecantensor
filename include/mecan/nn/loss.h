#pragma once
#include "mecan/tensor.h"
#include "mecan/autograd/variable.h"

namespace mecan {
namespace nn {

    /**
     * Loss & Fused Gradient Module
     * ----------------------------
     * Replaces PyTorch's heavy Error Tensor allocation.
     * Calculates Mean Absolute Error (MAE) and computes
     * dLoss directly to optimization, saving massive RAM.
     */
    class FusedLossMAE {
    public:
        // Returns the scalar loss value, but computes and applies 
        // the LGC gradients immediately without storing intermediate error tensors.
        static float compute_and_backward(
            autograd::Variable& predictions, 
            const Tensor& targets
        );
    };

} // namespace nn
} // namespace mecan
