#pragma once

#include <cstdint>
#include <string>

namespace mecan {
namespace core {

    // Define data precision formats
    enum class ScalarType {
        Float64,
        Float32,
        Float16,
        BFloat16,
        Int32,
        Int8,
        Ternary  // Native support for 1.58-bit logic
    };

    // Hardware target devices
    enum class DeviceType {
        CPU,
        GPU_OpenCL,
        GPU_CUDA,
        GPU_ROCM,
        GPU_Vulkan,
        GPU_Metal,
        GPU_SYCL,
        FPGA_PYNQ,
        SSD_Infusion  // Virtual device for out-of-core memory
    };

    struct Device {
        DeviceType type;
        int index;

        Device(DeviceType type = DeviceType::CPU, int index = 0)
            : type(type), index(index) {}

        bool operator==(const Device& other) const {
            return type == other.type && index == other.index;
        }

        std::string to_string() const {
            switch (type) {
                case DeviceType::CPU: return "CPU:" + std::to_string(index);
                case DeviceType::GPU_OpenCL: return "GPU_OpenCL:" + std::to_string(index);
                case DeviceType::GPU_CUDA: return "GPU_CUDA:" + std::to_string(index);
                case DeviceType::GPU_ROCM: return "GPU_ROCM:" + std::to_string(index);
                case DeviceType::GPU_Vulkan: return "GPU_Vulkan:" + std::to_string(index);
                case DeviceType::GPU_Metal: return "GPU_Metal:" + std::to_string(index);
                case DeviceType::GPU_SYCL: return "GPU_SYCL:" + std::to_string(index);
                case DeviceType::FPGA_PYNQ: return "FPGA:" + std::to_string(index);
                case DeviceType::SSD_Infusion: return "SSD:" + std::to_string(index);
                default: return "Unknown";
            }
        }
    };

    // Utility to get byte size of a scalar type
    inline size_t element_size(ScalarType type) {
        switch (type) {
            case ScalarType::Float64: return 8;
            case ScalarType::Float32: return 4;
            case ScalarType::Int32:   return 4;
            case ScalarType::Float16: return 2;
            case ScalarType::BFloat16:return 2;
            case ScalarType::Int8:    return 1;
            case ScalarType::Ternary: return 1; // Handled as int8_t in byte arrays
            default: return 0;
        }
    }

} // namespace core
} // namespace mecan
