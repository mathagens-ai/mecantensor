#pragma once

#include "tensor.h"
#include "autograd/variable.h"
#include <vector>

namespace mecan {
namespace optim {

    /**
     * LGC Optimizer: Linear Gradient Consolidation
     * -------------------------------------------
     * A high-performance, AdamW-free weight updater.
     * Uses second-order quadratic curvature to ensure stability
     * without the massive memory overhead of AdamW moments.
     */
    class LGCOptimizer {
    private:
        float lr_;
        float momentum_decay_;
        
        // These can be moved to SSD Infusion if they exceed RAM
        std::vector<float> linear_momentum_;
        std::vector<float> quadratic_curvature_;

    public:
        LGCOptimizer(size_t param_count, float lr = 1e-3, float decay = 0.5f);

        // Primary update function
        void step(Tensor& weights, const Tensor& gradients);
        void step(const std::vector<autograd::Variable*>& params);
        void zero_grad(const std::vector<autograd::Variable*>& params);
        
        inline void set_lr(float lr) { lr_ = lr; }
    };

} // namespace optim
} // namespace mecan
