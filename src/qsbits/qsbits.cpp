// QSBits: Sub-1-Bit Quantized XNOR Engine
#include "mecan/qsbits/qsbits.h"

#include <omp.h>
#include <stdexcept>

#if defined(_MSC_VER)
    #include <intrin.h>
#endif

#include "../hlas/hlas.h"

namespace mecan {
namespace qsbits {

void qsbits_forward(const Tensor& input, const Tensor& binary_weights, Tensor& output) {
    if (input.ndimension() != 2 || binary_weights.ndimension() != 2 || output.ndimension() != 2) {
        throw std::runtime_error("qsbits_forward expects 2D tensors.");
    }
    if ((input.dtype() != core::ScalarType::Int8 && input.dtype() != core::ScalarType::Ternary) ||
        (binary_weights.dtype() != core::ScalarType::Int8 && binary_weights.dtype() != core::ScalarType::Ternary) ||
        output.dtype() != core::ScalarType::Float32) {
        throw std::runtime_error("qsbits_forward expects Int8/Ternary inputs and Float32 output.");
    }

    const size_t batch = input.size(0);
    const size_t out_features = binary_weights.size(0);
    const size_t k_packed = input.size(1);
    
    if (binary_weights.size(1) != k_packed) {
        throw std::runtime_error("qsbits_forward shape mismatch: packed K differs between input and weights.");
    }
    if (output.size(0) != batch || output.size(1) != out_features) {
        throw std::runtime_error("qsbits_forward output shape mismatch.");
    }

    const uint8_t* in_ptr = input.data_ptr<uint8_t>();
    const uint8_t* w_ptr = binary_weights.data_ptr<uint8_t>();
    float* out_ptr = output.data_ptr<float>();

    #pragma omp parallel for schedule(dynamic, 4)
    for (int64_t b = 0; b < static_cast<int64_t>(batch); ++b) {
        const uint8_t* row = in_ptr + (b * k_packed);
        float* out_row = out_ptr + (b * out_features);
        
        mecan::hlas::get_engine()->qsbits_xnor(
            static_cast<int>(out_features), 
            static_cast<int>(k_packed), 
            row, 
            w_ptr, 
            out_row
        );
    }
}

void qsbits_forward_scaled(
    const Tensor& input,
    const Tensor& binary_weights,
    const Tensor& scales,
    float input_scale,
    int group_size,
    Tensor& output)
{
    if (input.ndimension() != 2 || binary_weights.ndimension() != 2 || output.ndimension() != 2) {
        throw std::runtime_error("qsbits_forward_scaled expects 2D tensors.");
    }
    if (scales.ndimension() != 2) {
        throw std::runtime_error("qsbits_forward_scaled expects 2D scales tensor [O, N_groups].");
    }

    const size_t batch = input.size(0);
    const size_t out_features = binary_weights.size(0);
    const size_t k_packed = input.size(1);

    if (binary_weights.size(1) != k_packed) {
        throw std::runtime_error("qsbits_forward_scaled shape mismatch: packed K differs.");
    }
    if (output.size(0) != batch || output.size(1) != out_features) {
        throw std::runtime_error("qsbits_forward_scaled output shape mismatch.");
    }

    int g_packed = group_size / 8;
    if (g_packed <= 0 || (k_packed % g_packed) != 0) {
        throw std::runtime_error("qsbits_forward_scaled: group_size must divide K evenly and be a multiple of 8.");
    }

    const uint8_t* in_ptr = input.data_ptr<uint8_t>();
    const uint8_t* w_ptr = binary_weights.data_ptr<uint8_t>();
    const float* s_ptr = scales.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();

    #pragma omp parallel for schedule(dynamic, 4)
    for (int64_t b = 0; b < static_cast<int64_t>(batch); ++b) {
        const uint8_t* row = in_ptr + (b * k_packed);
        float* out_row = out_ptr + (b * out_features);

        mecan::hlas::get_engine()->qsbits_xnor_scaled(
            static_cast<int>(out_features),
            static_cast<int>(k_packed),
            row,
            w_ptr,
            s_ptr,
            input_scale,
            group_size,
            out_row
        );
    }
}

} // namespace qsbits
} // namespace mecan
