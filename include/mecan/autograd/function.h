#pragma once

#include <vector>
#include <memory>
#include <string>
#include <utility>
#include "variable.h"

namespace mecan {
namespace autograd {

    /**
     * Base class for all Autograd operations.
     * Represents the "Green Boxes" in the user documentation.
     */
    class Function : public std::enable_shared_from_this<Function> {
    public:
        explicit Function(std::string op_name = "Function") : op_name(std::move(op_name)) {}
        virtual ~Function() = default;

        // Executes the backward math for this specific operation
        virtual std::vector<Variable> apply(const std::vector<Variable>& grads) = 0;

        // Pointers to the variables that were inputs to the forward pass
        // These are needed to calculate the gradients during backward.
        std::vector<std::shared_ptr<Function>> next_functions;

        void add_next_function(std::shared_ptr<Function> fn) {
            if (fn) next_functions.push_back(fn);
        }

        std::string op_name;
    };

} // namespace autograd
} // namespace mecan
