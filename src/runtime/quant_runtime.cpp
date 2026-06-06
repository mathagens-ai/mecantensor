#include "mecan/runtime/quant_runtime.h"

#include <stdexcept>
#include <vector>

#include "mecan/midbits/midbits.h"
#include "mecan/ops/math.h"
#include "mecan/qsbits/qsbits.h"

namespace mecan {
namespace runtime {

namespace {

void run_midbits_matmul(const Tensor& input, const Tensor& weights, Tensor& output) {
    if (input.ndimension() != 2 || weights.ndimension() != 2 || output.ndimension() != 2) {
        throw std::runtime_error("MidBits matmul expects 2D tensors.");
    }
    if (input.dtype() != core::ScalarType::Float32 || output.dtype() != core::ScalarType::Float32) {
        throw std::runtime_error("MidBits matmul currently expects FP32 input/output.");
    }
    if (weights.dtype() != core::ScalarType::Int8 && weights.dtype() != core::ScalarType::Ternary) {
        throw std::runtime_error("MidBits matmul currently expects Int8/Ternary packed weights.");
    }

    const int batch = static_cast<int>(input.size(0));
    const int k = static_cast<int>(input.size(1));
    const int out_features = static_cast<int>(weights.size(0));
    if (weights.size(1) * 16 != static_cast<size_t>(k)) {
        throw std::runtime_error("MidBits weight shape mismatch (expected K/16 packed columns).");
    }
    if (output.size(0) != static_cast<size_t>(batch) || output.size(1) != static_cast<size_t>(out_features)) {
        throw std::runtime_error("MidBits output shape mismatch.");
    }

    const uint8_t* w = reinterpret_cast<const uint8_t*>(weights.data_ptr<int8_t>());
    const float* x = input.data_ptr<float>();
    float* y = output.data_ptr<float>();

    const int num_chunks = k / 16;
    for (int i = 0; i < batch; ++i) {
        midbits::matvec_0_75b(w, x + (i * k), y + (i * out_features), out_features, num_chunks * 16);
    }
}

} // namespace

bool DefaultQuantKernel::can_run(QuantScheme scheme, core::DeviceType device) const {
    // CPU and OpenCL are enabled by default in this phase; GPU-specific backends
    // can be wired later via runtime backend plugins.
    switch (scheme) {
        case QuantScheme::QSBits1:
        case QuantScheme::MidBits075:
        case QuantScheme::Ternary158:
        case QuantScheme::FP32:
            return (device == core::DeviceType::CPU || device == core::DeviceType::GPU_OpenCL);
        case QuantScheme::FP16:
        case QuantScheme::BF16:
            return (device == core::DeviceType::GPU_OpenCL || device == core::DeviceType::GPU_CUDA ||
                    device == core::DeviceType::GPU_ROCM || device == core::DeviceType::GPU_SYCL ||
                    device == core::DeviceType::GPU_Metal || device == core::DeviceType::GPU_Vulkan);
        default:
            return false;
    }
}

void DefaultQuantKernel::run_matmul(
    const Tensor& input,
    const Tensor& weights,
    Tensor& output,
    QuantScheme scheme,
    AccumulationPolicy policy) {
    (void)policy;

    switch (scheme) {
        case QuantScheme::QSBits1:
            qsbits::qsbits_forward(input, weights, output);
            return;
        case QuantScheme::MidBits075:
            run_midbits_matmul(input, weights, output);
            return;
        case QuantScheme::FP32:
            ops::matmul(input, weights, output);
            return;
        default:
            throw std::runtime_error("DefaultQuantKernel: unsupported quant scheme for matmul.");
    }
}

void run_quantized_matmul_with_fallback(
    IQuantKernel& kernel,
    const Tensor& input,
    const Tensor& weights,
    Tensor& output,
    QuantScheme scheme,
    core::DeviceType preferred_device,
    AccumulationPolicy preferred_policy) {
    if (kernel.can_run(scheme, preferred_device)) {
        kernel.run_matmul(input, weights, output, scheme, preferred_policy);
        return;
    }

    if (kernel.can_run(scheme, core::DeviceType::CPU)) {
        kernel.run_matmul(input, weights, output, scheme, AccumulationPolicy::MixedFP32);
        return;
    }

    if (input.dtype() == core::ScalarType::Float32 && weights.dtype() == core::ScalarType::Float32) {
        ops::matmul(input, weights, output);
        return;
    }

    throw std::runtime_error("run_quantized_matmul_with_fallback: no viable kernel path found.");
}

} // namespace runtime
} // namespace mecan
