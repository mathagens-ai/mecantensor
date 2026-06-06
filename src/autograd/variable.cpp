#include "autograd/variable.h"
#include "autograd/engine.h"
namespace mecan {
namespace autograd {

    void Variable::zero_grad() {
        if (!requires_grad || grad.numel() == 0) {
            return;
        }

        if (grad.dtype() == core::ScalarType::Float32) {
            float* ptr = grad.data_ptr<float>();
            for (size_t i = 0; i < grad.numel(); ++i) {
                ptr[i] = 0.0f;
            }
            return;
        }

        if (grad.dtype() == core::ScalarType::Int8 || grad.dtype() == core::ScalarType::Ternary) {
            int8_t* ptr = grad.data_ptr<int8_t>();
            for (size_t i = 0; i < grad.numel(); ++i) {
                ptr[i] = 0;
            }
        }
    }

    void Variable::backward() {
        if (!requires_grad) return;
        Engine::backward(*this);
    }

} // namespace autograd
} // namespace mecan
