#pragma once

#include "tensor.h"
#include <memory>
#include <vector>
#include <string>
#include <cstdint>

namespace mecan {
namespace autograd {

    class Function; // Forward declaration

    // A Variable is a Tensor that tracks its history for Autograd.
    class Variable {
    public:
        Tensor data;
        Tensor grad;
        bool requires_grad;
        std::string source_node;
        uint64_t node_id;
        
        // The function that produced this variable (the 'creator')
        std::shared_ptr<Function> grad_fn;
        std::shared_ptr<Function> grad_accumulator;

        Variable() : requires_grad(false), source_node("input"), node_id(0) {}
        Variable(Tensor data, bool requires_grad = false) 
            : data(data), requires_grad(requires_grad), source_node("input"), node_id(0) {
            if (requires_grad) {
                // Initialize empty gradient of the same shape
                grad = Tensor(data.shape(), data.dtype(), data.device());
                zero_grad();
            }
        }

        // Backward trigger
        inline bool is_leaf() const { return requires_grad && !grad_fn; }
        void backward();
        void zero_grad();
    };

} // namespace autograd
} // namespace mecan
