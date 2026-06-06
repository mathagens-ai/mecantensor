#pragma once

#include <cstddef>
#include <memory>
#include "types.h"

namespace mecan {
namespace core {

    // Abstract allocator interface
    class Allocator {
    public:
        virtual ~Allocator() = default;
        virtual void* allocate(size_t nbytes) = 0;
        virtual void deallocate(void* ptr) = 0;
        virtual Device device() const = 0;
    };

    // Standard high-speed CPU RAM Allocator (SIMD Aligned)
    class CPUAllocator : public Allocator {
    public:
        void* allocate(size_t nbytes) override;
        void deallocate(void* ptr) override;
        Device device() const override { return Device(DeviceType::CPU); }
    };

    // Advanced SSD Infusion allocator for 50B+ parameters that exceed RAM bounds
    class SSDInfusionAllocator : public Allocator {
    private:
        // Future MMAP configuration
    public:
        void* allocate(size_t nbytes) override;
        void deallocate(void* ptr) override;
        Device device() const override { return Device(DeviceType::SSD_Infusion); }
    };

    // Global allocator getter
    Allocator* get_allocator(DeviceType device_type);

} // namespace core
} // namespace mecan
