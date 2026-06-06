#pragma once

#include <vector>
#include "function.h"

namespace mecan {
namespace autograd {

    /**
     * The Engine traverses the graph (Image 1 & 2) in reverse topological order.
     * It ensures that a node is only processed after all its 
     * dependent gradients have been summed (Image 2 - "Grads added together").
     */
    class Engine {
    public:
        static void backward(Variable& v);

    private:
        static Tensor make_seed_grad(const Tensor& reference);
        static void add_inplace(Tensor& dst, const Tensor& src);
        static std::vector<std::shared_ptr<Function>> topological_sort(const std::shared_ptr<Function>& root);
    };

} // namespace autograd
} // namespace mecan
