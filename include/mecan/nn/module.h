#pragma once

#include "autograd/variable.h"
#include <string>
#include <vector>
#include <map>

namespace mecan {
namespace nn {

    /**
     * Supports Infusion metadata for 50B+ scaling on 8GB RAM.
     */
    class Module {
    protected:
        std::string name_;
        bool training_ = true;

    public:
        virtual ~Module() = default;
        virtual autograd::Variable forward(autograd::Variable input) = 0;

        virtual void train() { training_ = true; }
        virtual void eval() { training_ = false; }

        // Parameter management
        virtual std::vector<autograd::Variable*> parameters() = 0;
    };

} // namespace nn
} // namespace mecan
