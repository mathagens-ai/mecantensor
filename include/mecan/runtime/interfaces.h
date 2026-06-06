#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "mecan/runtime/descriptors.h"
#include "mecan/tensor.h"

namespace mecan {
namespace runtime {

class IDeviceBackend {
public:
    virtual ~IDeviceBackend() = default;
    virtual std::string name() const = 0;
    virtual bool is_available() const = 0;
    virtual DeviceCapability capability(int device_index) const = 0;
    virtual bool supports(QuantScheme scheme, int device_index) const = 0;
    virtual void synchronize(int device_index) = 0;
};

class IMemoryPager {
public:
    virtual ~IMemoryPager() = default;
    virtual void register_tensor(const PagedTensorMetadata& metadata) = 0;
    virtual void ensure_resident(const std::string& tensor_name, MemoryTier target) = 0;
    virtual void prefetch(const std::string& tensor_name, size_t page_begin, size_t page_count, MemoryTier target) = 0;
    virtual void evict(const std::string& tensor_name, MemoryTier target) = 0;
    virtual uint64_t used_bytes(MemoryTier tier) const = 0;
    virtual MemoryBudget budget() const = 0;
};

class IQuantKernel {
public:
    virtual ~IQuantKernel() = default;
    virtual bool can_run(QuantScheme scheme, core::DeviceType device) const = 0;
    virtual void run_matmul(
        const Tensor& input,
        const Tensor& weights,
        Tensor& output,
        QuantScheme scheme,
        AccumulationPolicy policy) = 0;
};

class ICollectiveTransport {
public:
    virtual ~ICollectiveTransport() = default;
    virtual std::string name() const = 0;
    virtual bool initialize(int world_size, int rank) = 0;
    virtual void finalize() = 0;
    virtual int world_size() const = 0;
    virtual int rank() const = 0;
    virtual bool allreduce_sum_f32(float* data, size_t count) = 0;
    virtual bool broadcast_f32(float* data, size_t count, int root_rank) = 0;
    virtual bool barrier() = 0;
};

} // namespace runtime
} // namespace mecan
