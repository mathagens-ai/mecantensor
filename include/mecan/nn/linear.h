#pragma once

#include "module.h"
#include <string>

namespace mecan {
namespace nn {

    /**
     * BitLinear: The 1.58-bit Ternary Layer
     * ------------------------------------
     * Essential for 50B+ parameters on 8GB RAM.
     * Weights are stored as {-1, 0, 1} and packed to save space.
     */
    class BitLinear : public Module {
    private:
        size_t in_features_;
        size_t out_features_;
        
    public:
        autograd::Variable weight; // Ternary weights
        autograd::Variable bias;   // High-precision bias

        BitLinear(size_t in_f, size_t out_f, core::Device device = core::Device(core::DeviceType::SSD_Infusion));

        autograd::Variable forward(autograd::Variable input) override;
        std::string summary() const;
        
        std::vector<autograd::Variable*> parameters() override {
            return {&weight, &bias};
        }
    };

} // namespace nn
} // namespace mecan
