#include "mecan/runtime/backend_registry.h"

#include <algorithm>

#include "mecan/hal/hal.h"

namespace mecan {
namespace runtime {

namespace {

class HALBackendAdapter final : public IDeviceBackend {
public:
    explicit HALBackendAdapter(hal::BackendAPI api)
        : api_(api) {}

    std::string name() const override {
        return std::string("HAL::") + hal::backend_api_name(api_);
    }

    bool is_available() const override {
        return backend() != nullptr;
    }

    DeviceCapability capability(int device_index) const override {
        DeviceCapability out{};
        out.backend_name = name();
        out.device_index = device_index;

        hal::Backend* b = backend();
        if (!b) {
            return out;
        }
        const auto devices = b->enumerate_devices();
        if (devices.empty()) {
            return out;
        }
        const auto& d = devices[std::min<size_t>(static_cast<size_t>(std::max(0, device_index)), devices.size() - 1)];

        out.global_memory_bytes = d.global_mem_bytes;
        out.compute_units = static_cast<uint32_t>(d.compute_units);
        out.simd_width_bits = d.simd_width;
        out.supports_fp16 = d.supports_fp16;
        out.supports_bf16 = d.supports_bf16;
        out.supports_int8 = d.supports_int8;
        out.supports_qsbits = d.supports_binary;
        out.supports_midbits = d.supports_midbits;
        return out;
    }

    bool supports(QuantScheme scheme, int device_index) const override {
        const DeviceCapability cap = capability(device_index);
        switch (scheme) {
            case QuantScheme::FP32: return true;
            case QuantScheme::FP16: return cap.supports_fp16;
            case QuantScheme::BF16: return cap.supports_bf16;
            case QuantScheme::Ternary158: return cap.supports_int8;
            case QuantScheme::QSBits1: return cap.supports_qsbits;
            case QuantScheme::MidBits075: return cap.supports_midbits;
            default: return false;
        }
    }

    void synchronize(int device_index) override {
        hal::Backend* b = backend();
        if (b) {
            b->synchronize(device_index);
        }
    }

private:
    hal::Backend* backend() const {
        auto& reg = hal::BackendRegistry::instance();
        static bool discovered = false;
        if (!discovered) {
            reg.discover_all();
            discovered = true;
        }
        return reg.get_backend(api_);
    }

    hal::BackendAPI api_;
};

} // namespace

DeviceBackendRegistry& DeviceBackendRegistry::instance() {
    static DeviceBackendRegistry reg;
    return reg;
}

void DeviceBackendRegistry::register_backend(std::shared_ptr<IDeviceBackend> backend) {
    if (!backend) {
        return;
    }
    backends_.push_back(std::move(backend));
}

void DeviceBackendRegistry::register_default_backends() {
    if (!backends_.empty()) {
        return;
    }

    register_backend(std::make_shared<HALBackendAdapter>(hal::BackendAPI::NATIVE_CPU));
    register_backend(std::make_shared<HALBackendAdapter>(hal::BackendAPI::OPENCL));
    register_backend(std::make_shared<HALBackendAdapter>(hal::BackendAPI::CUDA));
    register_backend(std::make_shared<HALBackendAdapter>(hal::BackendAPI::ROCM_HIP));
    register_backend(std::make_shared<HALBackendAdapter>(hal::BackendAPI::METAL));
    register_backend(std::make_shared<HALBackendAdapter>(hal::BackendAPI::ONEAPI_SYCL));
    register_backend(std::make_shared<HALBackendAdapter>(hal::BackendAPI::VULKAN_COMPUTE));
}

std::shared_ptr<IDeviceBackend> DeviceBackendRegistry::find_by_name(const std::string& backend_name) const {
    if (backends_.empty()) {
        const_cast<DeviceBackendRegistry*>(this)->register_default_backends();
    }
    auto it = std::find_if(
        backends_.begin(),
        backends_.end(),
        [&backend_name](const std::shared_ptr<IDeviceBackend>& b) {
            return b && b->name() == backend_name;
        });
    if (it == backends_.end()) {
        return nullptr;
    }
    return *it;
}

std::shared_ptr<IDeviceBackend> DeviceBackendRegistry::find_best_backend(QuantScheme scheme) const {
    if (backends_.empty()) {
        const_cast<DeviceBackendRegistry*>(this)->register_default_backends();
    }
    for (const auto& backend : backends_) {
        if (backend && backend->is_available() && backend->supports(scheme, 0)) {
            return backend;
        }
    }
    return nullptr;
}

std::vector<std::shared_ptr<IDeviceBackend>> DeviceBackendRegistry::list_available() const {
    if (backends_.empty()) {
        const_cast<DeviceBackendRegistry*>(this)->register_default_backends();
    }
    std::vector<std::shared_ptr<IDeviceBackend>> out;
    for (const auto& backend : backends_) {
        if (backend && backend->is_available()) {
            out.push_back(backend);
        }
    }
    return out;
}

} // namespace runtime
} // namespace mecan
