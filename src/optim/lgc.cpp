#include "optim/lgc.h"
#include <omp.h>
#include <cmath>
#include <algorithm>

namespace mecan {
namespace optim {

    LGCOptimizer::LGCOptimizer(size_t param_count, float lr, float decay) 
        : lr_(lr), momentum_decay_(decay) {
        
        linear_momentum_.resize(param_count, 0.0f);
        quadratic_curvature_.resize(param_count, 1.0f); // Initialize at 1.0 to avoid zero-div
    }

    void LGCOptimizer::step(Tensor& weights, const Tensor& gradients) {
        float* w_ptr = weights.data_ptr<float>();
        const float* g_ptr = gradients.data_ptr<float>();
        size_t n = weights.numel();
        if (linear_momentum_.size() < n) linear_momentum_.resize(n, 0.0f);
        if (quadratic_curvature_.size() < n) quadratic_curvature_.resize(n, 1.0f);

        // Optimized LGC Loop
        #pragma omp parallel for
        for (int64_t i = 0; i < (int64_t)n; ++i) {
            float g = g_ptr[i];
            float g_sq = g * g;

            // 1. Consolidate Gradient Momentum
            linear_momentum_[i] = (linear_momentum_[i] * momentum_decay_) + g;

            // 2. Adjust Quadratic Curvature (Second-order approximation)
            quadratic_curvature_[i] = (quadratic_curvature_[i] * momentum_decay_) + g_sq;

            // 3. Compute Stability-Aware Step (LGC Formula)
            float safe_curvature = std::max(1e-8f, quadratic_curvature_[i]);
            float delta = linear_momentum_[i] / safe_curvature;

            // 4. Weight Update
            w_ptr[i] -= lr_ * delta;
        }
    }

    void LGCOptimizer::step(const std::vector<autograd::Variable*>& params) {
        for (autograd::Variable* p : params) {
            if (!p || !p->data.defined() || !p->grad.defined()) continue;
            if (p->data.dtype() != core::ScalarType::Float32 || p->grad.dtype() != core::ScalarType::Float32) continue;
            step(p->data, p->grad);
        }
    }

    void LGCOptimizer::zero_grad(const std::vector<autograd::Variable*>& params) {
        for (autograd::Variable* p : params) {
            if (p) p->zero_grad();
        }
    }

} // namespace optim
} // namespace mecan
