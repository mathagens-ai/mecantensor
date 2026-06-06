#pragma once

#include "mecan/runtime/interfaces.h"

namespace mecan {
namespace runtime {

class DefaultQuantKernel : public IQuantKernel {
public:
    bool can_run(QuantScheme scheme, core::DeviceType device) const override;
    void run_matmul(
        const Tensor& input,
        const Tensor& weights,
        Tensor& output,
        QuantScheme scheme,
        AccumulationPolicy policy) override;
};

// Kernel fallback ladder:
// native low-bit -> mixed -> fp32
void run_quantized_matmul_with_fallback(
    IQuantKernel& kernel,
    const Tensor& input,
    const Tensor& weights,
    Tensor& output,
    QuantScheme scheme,
    core::DeviceType preferred_device,
    AccumulationPolicy preferred_policy = AccumulationPolicy::NativeLowBit);

} // namespace runtime
} // namespace mecan
