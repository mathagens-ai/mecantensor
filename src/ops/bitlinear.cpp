// BitLinear: Ternary/Int8/FP32-compatible forward kernel
#include "ops/bitlinear.h"

#include <omp.h>
#include <stdexcept>

namespace mecan {
namespace ops {

void bitlinear_forward(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output) {
    if (input.ndimension() != 2 || weight.ndimension() != 2 || output.ndimension() != 2 || bias.ndimension() != 1) {
        throw std::runtime_error("bitlinear_forward expects input/weight/output as 2D and bias as 1D.");
    }
    if (input.dtype() != core::ScalarType::Float32 || bias.dtype() != core::ScalarType::Float32 ||
        output.dtype() != core::ScalarType::Float32) {
        throw std::runtime_error("bitlinear_forward expects FP32 input/bias/output.");
    }

    const size_t batch = input.size(0);
    const size_t in_features = input.size(1);
    const size_t out_features = weight.size(0);

    if (weight.size(1) != in_features || output.size(0) != batch || output.size(1) != out_features ||
        bias.size(0) != out_features) {
        throw std::runtime_error("bitlinear_forward shape mismatch.");
    }

    const float* in_ptr = input.data_ptr<float>();
    const float* b_ptr = bias.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();

    const bool weight_is_fp32 = (weight.dtype() == core::ScalarType::Float32);
    const bool weight_is_lowbit = (weight.dtype() == core::ScalarType::Int8 || weight.dtype() == core::ScalarType::Ternary);
    if (!weight_is_fp32 && !weight_is_lowbit) {
        throw std::runtime_error("bitlinear_forward unsupported weight dtype.");
    }

    #pragma omp parallel for schedule(dynamic, 8)
    for (int64_t b = 0; b < static_cast<int64_t>(batch); ++b) {
        const float* row_in = in_ptr + (b * in_features);
        for (int64_t o = 0; o < static_cast<int64_t>(out_features); ++o) {
            float sum = b_ptr[o];

            if (weight_is_fp32) {
                const float* row_w = weight.data_ptr<float>() + (o * in_features);
                for (int64_t k = 0; k < static_cast<int64_t>(in_features); ++k) {
                    sum += row_in[k] * row_w[k];
                }
            } else {
                const int8_t* row_w = weight.data_ptr<int8_t>() + (o * in_features);
                for (int64_t k = 0; k < static_cast<int64_t>(in_features); ++k) {
                    const int8_t w = row_w[k];
                    if (w == 1) sum += row_in[k];
                    else if (w == -1) sum -= row_in[k];
                }
            }

            out_ptr[b * out_features + o] = sum;
        }
    }
}

} // namespace ops
} // namespace mecan
