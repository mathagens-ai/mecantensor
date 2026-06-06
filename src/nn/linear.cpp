#include "nn/linear.h"
#include "ops/bitlinear.h"
#include <sstream>

namespace mecan {
namespace nn {

    BitLinear::BitLinear(size_t in_f, size_t out_f, core::Device device)
        : in_features_(in_f), out_features_(out_f) {
        
        // Weights are Ternary (INT8 container)
        Tensor w_data({out_f, in_f}, core::ScalarType::Ternary, device);
        weight = autograd::Variable(w_data, true);

        // Biases are FP32
        Tensor b_data({out_f}, core::ScalarType::Float32, device);
        bias = autograd::Variable(b_data, true);
    }

    autograd::Variable BitLinear::forward(autograd::Variable input) {
        std::vector<size_t> out_shape = input.data.shape();
        out_shape.back() = out_features_;
        
        Tensor output_data(out_shape, core::ScalarType::Float32, input.data.device());
        
        // Execute Ternary TSSR Forward
        ops::bitlinear_forward(input.data, weight.data, bias.data, output_data);
        
        return autograd::Variable(output_data, input.requires_grad);
    }

    std::string BitLinear::summary() const {
        std::ostringstream os;
        os << "BitLinear(in_features=" << in_features_
           << ", out_features=" << out_features_
           << ", bias=True, weight_dtype=Ternary)";
        return os.str();
    }

} // namespace nn
} // namespace mecan
