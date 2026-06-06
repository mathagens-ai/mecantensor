#include "mecan/nn/loss.h"
#include <omp.h>
#include <cmath>

namespace mecan {
namespace nn {

    float FusedLossMAE::compute_and_backward(autograd::Variable& predictions, const Tensor& targets) {
        size_t n = predictions.data.numel();
        const float* pred_ptr = predictions.data.data_ptr<float>();
        const float* targ_ptr = targets.data_ptr<float>();
        
        // Ensure gradients exist
        if (!predictions.grad.defined()) {
            predictions.grad = Tensor(predictions.data.shape(), core::ScalarType::Float32, predictions.data.device());
        }
        float* grad_ptr = predictions.grad.data_ptr<float>();

        float total_loss = 0.0f;
        float inv_n = 1.0f / static_cast<float>(n);

        // Single Fused Pass
        #pragma omp parallel for reduction(+:total_loss)
        for (int64_t i = 0; i < (int64_t)n; ++i) {
            float diff = pred_ptr[i] - targ_ptr[i];
            
            // 1. Calculate MAE Loss accumulator
            total_loss += std::abs(diff);

            // 2. Fused Backward Derivative (dLoss / dPred)
            // L1 derivative is just sign(diff)
            grad_ptr[i] = (diff > 0) ? inv_n : ((diff < 0) ? -inv_n : 0.0f);
        }

        return total_loss * inv_n;
    }

} // namespace nn
} // namespace mecan
