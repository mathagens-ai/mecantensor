#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "mecan/core/types.h"

namespace mecan {
namespace runtime {

enum class QuantScheme {
    FP32 = 0,
    FP16 = 1,
    BF16 = 2,
    Ternary158 = 3,
    QSBits1 = 4,
    MidBits075 = 5
};

enum class MemoryTier {
    VRAM = 0,
    RAM = 1,
    NVME = 2
};

enum class AccumulationPolicy {
    NativeLowBit = 0,
    MixedFP16 = 1,
    MixedFP32 = 2
};

struct PageLayout {
    uint64_t page_bytes = 1ULL << 20; // 1 MB default page
    uint32_t alignment_bytes = 64;
    bool compressed = false;
};

struct MemoryBudget {
    uint64_t vram_bytes = 0;
    uint64_t ram_bytes = 0;
    uint64_t nvme_bytes = 0;
    bool hard_limit = true;
};

struct DeviceCapability {
    std::string backend_name;
    int device_index = 0;
    uint64_t global_memory_bytes = 0;
    uint32_t compute_units = 0;
    uint32_t simd_width_bits = 0;
    bool supports_fp16 = false;
    bool supports_bf16 = false;
    bool supports_int8 = false;
    bool supports_qsbits = false;
    bool supports_midbits = false;
};

struct PagedTensorMetadata {
    std::string tensor_name;
    std::vector<size_t> shape;
    core::ScalarType storage_dtype = core::ScalarType::Float32;
    core::DeviceType preferred_device = core::DeviceType::CPU;
    QuantScheme quant_scheme = QuantScheme::FP32;
    PageLayout page_layout{};
    uint64_t total_nbytes = 0;
};

inline const char* quant_scheme_name(QuantScheme scheme) {
    switch (scheme) {
        case QuantScheme::FP32: return "FP32";
        case QuantScheme::FP16: return "FP16";
        case QuantScheme::BF16: return "BF16";
        case QuantScheme::Ternary158: return "TERNARY158";
        case QuantScheme::QSBits1: return "QSBITS1";
        case QuantScheme::MidBits075: return "MIDBITS075";
        default: return "UNKNOWN";
    }
}

} // namespace runtime
} // namespace mecan
